#!/usr/bin/env python3
"""Download raw HRRR forecasts (bbox-subset) for bias-correction training.

Fetches NOAA HRRR surface output (``wrfsfc``, 3 km CONUS) from the AWS Open Data
bucket ``noaa-hrrr-bdp-pds`` for the 00/06/12/18Z cycles, hourly leads f00..f48,
crops to the Kitchener-Waterloo bounding box on HRRR's **native Lambert-conformal
grid** (no reprojection/regridding -- that's a downstream modelling choice), and
writes one compact NetCDF per init-cycle:

    {data_root}/hrrr/{YYYY}/{YYYYMMDD}/hrrr_{YYYYMMDD}_t{HH}z.nc
        dims:  (lead, y, x)
        vars:  t2m (K), u10 (m/s), v10 (m/s), tp (kg/m^2, accumulated)
        coords: latitude/longitude (2-D), lead (hours)

Only the four needed GRIB messages per lead are downloaded via the ``.idx``
byte-range trick (HTTP Range headers), so we never pull whole HRRR files.

Notes / caveats:
  * 48 h leads only exist from HRRRv4 (~2020-12-02). Earlier HRRRv3 extended
    cycles cap at f36; the missing leads simply 404 and are skipped (we do NOT
    hard-code version dates -- availability is probed from the archive).
  * ``tp`` is HRRR's accumulated APCP as-is (bucket resets over the run); f00 is
    the 0-0 h accumulation (all zeros). If a lead ever lacks an APCP message it is
    filled with NaN. De-accumulation is left to downstream.

Mirrors the download/cache patterns in
``benchmarking-site/data/hrrr_interpolated/compute_benchmark.py``.
"""

from __future__ import annotations

import argparse
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr

from weather_download_common import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    PAD_DEG,
    REPO_ROOT,
    download_bytes,
    parse_idx_ranges,
    resolve_data_root,
)

AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
CACHE_DIR = REPO_ROOT / ".cache" / "hrrr_grib"
INIT_HOURS_UTC = (0, 6, 12, 18)
DEFAULT_START = "2018-07-13"  # first full day of HRRRv3 operational archive
MAX_LEAD_DEFAULT = 48

# GRIB inventory needles -> output variable name. Order defines NetCDF vars.
GRIB_NEEDLES = {
    "t2m": ":TMP:2 m above ground:",
    "u10": ":UGRD:10 m above ground:",
    "v10": ":VGRD:10 m above ground:",
    "tp": ":APCP:surface:",
}
VAR_UNITS = {"t2m": "K", "u10": "m s-1", "v10": "m s-1", "tp": "kg m-2"}


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def grib_url(day: date, cycle: int, fxx: int) -> str:
    return (
        f"{AWS_BASE}/hrrr.{day:%Y%m%d}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsfcf{fxx:02d}.grib2"
    )


def fetch_idx(url: str, cache_key: str, retries: int) -> str | None:
    """Return the ``.idx`` inventory text, or None if the lead does not exist."""
    idx_path = CACHE_DIR / f"{cache_key}.idx"
    if idx_path.exists():
        return idx_path.read_text()
    try:
        text = download_bytes(url + ".idx", retries=retries).decode(
            "utf-8", errors="replace"
        )
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None  # lead not published (e.g. f37+ pre-HRRRv4)
        raise
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(text)
    return text


def download_fields(url: str, cache_key: str, retries: int) -> dict[str, Path] | None:
    """Byte-range download the needed GRIB messages; return {var: grib blob path}.

    Returns None if the lead's ``.idx`` is unavailable (missing lead).
    """
    idx_text = fetch_idx(url, cache_key, retries)
    if idx_text is None:
        return None
    ranges = parse_idx_ranges(idx_text, GRIB_NEEDLES)

    paths: dict[str, Path] = {}
    for field in GRIB_NEEDLES:
        if field not in ranges:
            # e.g. APCP absent at f00; that field is simply skipped this lead.
            continue
        blob = CACHE_DIR / f"{cache_key}_{field}.grib2"
        if not blob.exists():
            start, end = ranges[field]
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(download_bytes(url, start, end, retries=retries))
        paths[field] = blob
    return paths


# --- static bbox index window on HRRR's curvilinear grid (computed once) ---
_WINDOW: tuple[slice, slice] | None = None


def bbox_window(sample_grib: Path) -> tuple[slice, slice]:
    """Row/col slice covering the padded bbox; HRRR's grid is static so cache it."""
    global _WINDOW
    if _WINDOW is not None:
        return _WINDOW
    with cfgrib.open_dataset(sample_grib) as ds:
        lat = np.asarray(ds.latitude.values, dtype=float)
        lon = np.asarray(ds.longitude.values, dtype=float)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    inside = (
        (lat >= LAT_MIN - PAD_DEG)
        & (lat <= LAT_MAX + PAD_DEG)
        & (lon >= LON_MIN - PAD_DEG)
        & (lon <= LON_MAX + PAD_DEG)
    )
    if not inside.any():
        raise RuntimeError("Bounding box does not intersect the HRRR domain")
    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    _WINDOW = (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1))
    return _WINDOW


def read_field(grib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (values, lat, lon) cropped to the bbox window for one GRIB message."""
    jsl, isl = bbox_window(grib_path)
    with cfgrib.open_dataset(grib_path) as ds:
        var = list(ds.data_vars)[0]
        vals = np.asarray(ds[var].values, dtype=np.float32)[jsl, isl]
        lat = np.asarray(ds.latitude.values, dtype=np.float32)[jsl, isl]
        lon = np.asarray(ds.longitude.values, dtype=np.float32)[jsl, isl]
    lon = np.where(lon > 180.0, lon - 360.0, lon).astype(np.float32)
    return vals, lat, lon


def purge_cycle_cache(day: date, cycle: int) -> None:
    """Delete this init-cycle's cached GRIB blobs (full-CONUS, ~5 MB/lead).

    Without this the cache grows to terabytes over a full backfill. Called after
    the cropped NetCDF is safely written. cfgrib also writes ``*.grib2.*.idx``
    sidecars, so glob the whole prefix.
    """
    for p in CACHE_DIR.glob(f"{day:%Y%m%d}_t{cycle:02d}z_f*"):
        try:
            p.unlink()
        except OSError:
            pass  # best-effort; a lingering handle just leaves one file behind


def build_cycle_dataset(
    day: date, cycle: int, max_lead: int, retries: int
) -> xr.Dataset | None:
    """Download + crop all leads of one init-cycle into a single Dataset."""
    leads: list[int] = []
    fields: dict[str, list[np.ndarray]] = {v: [] for v in GRIB_NEEDLES}
    lat2d = lon2d = None

    for fxx in range(0, max_lead + 1):
        cache_key = f"{day:%Y%m%d}_t{cycle:02d}z_f{fxx:02d}"
        paths = download_fields(grib_url(day, cycle, fxx), cache_key, retries)
        if paths is None:
            break  # no more leads published for this cycle
        if not paths:
            continue  # idx present but none of our fields matched; skip lead
        leads.append(fxx)
        shape = None
        for field in GRIB_NEEDLES:
            if field in paths:
                vals, lat2d_, lon2d_ = read_field(paths[field])
                if lat2d is None:
                    lat2d, lon2d = lat2d_, lon2d_
                shape = vals.shape
                fields[field].append(vals)
            else:
                fields[field].append(None)  # placeholder; filled with NaN below
        # backfill any None placeholders now that we know the crop shape
        for field in GRIB_NEEDLES:
            if fields[field][-1] is None and shape is not None:
                fields[field][-1] = np.full(shape, np.nan, dtype=np.float32)

    if not leads or lat2d is None:
        return None

    ny, nx = lat2d.shape
    data_vars = {}
    for field in GRIB_NEEDLES:
        stacked = np.stack(fields[field], axis=0)  # (lead, y, x)
        data_vars[field] = (
            ("lead", "y", "x"),
            stacked,
            {"units": VAR_UNITS[field]},
        )

    init = datetime(day.year, day.month, day.day, cycle, tzinfo=timezone.utc)
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "lead": ("lead", np.asarray(leads, dtype=np.int16), {"units": "hours"}),
            "latitude": (("y", "x"), lat2d),
            "longitude": (("y", "x"), lon2d),
        },
        attrs={
            "source": "NOAA HRRR (noaa-hrrr-bdp-pds), wrfsfc 3km CONUS",
            "initialization": init.strftime("%Y-%m-%dT%H:00:00Z"),
            "grid": "native Lambert-conformal (not regridded)",
            "bbox": f"lat[{LAT_MIN},{LAT_MAX}] lon[{LON_MIN},{LON_MAX}] pad={PAD_DEG}",
        },
    )


def process_cycle(
    day: date,
    cycle: int,
    data_root: Path,
    max_lead: int,
    retries: int,
    resume: bool,
    keep_grib: bool,
) -> str:
    out_path = (
        data_root / "hrrr" / f"{day:%Y}" / f"{day:%Y%m%d}" / f"hrrr_{day:%Y%m%d}_t{cycle:02d}z.nc"
    )
    if resume and out_path.exists():
        return f"skip {out_path.name}"

    ds = build_cycle_dataset(day, cycle, max_lead, retries)
    if ds is None:
        return f"MISSING {day:%Y%m%d} t{cycle:02d}z (no leads published)"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    # atomic-ish write: temp file in the same dir, then replace
    tmp = out_path.parent / (out_path.name + ".tmp")
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.replace(out_path)
    nleads = ds.sizes["lead"]
    ds.close()
    if not keep_grib:
        # Drop the ~5 MB/lead full-CONUS GRIB cache now that the crop is written,
        # so the cache never grows past a few in-flight cycles (~TBs otherwise).
        purge_cycle_cache(day, cycle)
    return f"wrote {out_path.name} ({nleads} leads)"


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Output root (else $WEATHERLOO_DATA_ROOT, else repo data/)")
    parser.add_argument("--start-date", default=DEFAULT_START, help="YYYY-MM-DD inclusive (default HRRRv3 start)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive (default: today UTC)")
    parser.add_argument("--cycles", default="0,6,12,18", help="Comma-separated init hours UTC")
    parser.add_argument("--max-lead", type=int, default=MAX_LEAD_DEFAULT, help="Max forecast hour to request")
    parser.add_argument("--workers", type=int, default=3, help="Parallel init-cycle workers (3 avoids AWS resets)")
    parser.add_argument("--download-retries", type=int, default=6, help="Retries per GRIB/idx download")
    parser.add_argument("--resume", action="store_true", help="Skip cycles whose NetCDF already exists")
    parser.add_argument("--keep-grib", action="store_true", help="Keep cached full-CONUS GRIB blobs (else purged per cycle; cache is ~5 MB/lead)")
    parser.add_argument("--dry-run", action="store_true", help="Process only the start date's first cycle")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    start = parse_utc(f"{args.start_date}T00:00:00Z").date()
    end = (
        parse_utc(f"{args.end_date}T00:00:00Z").date()
        if args.end_date
        else datetime.now(timezone.utc).date()
    )
    cycles = tuple(int(h.strip()) for h in args.cycles.split(","))

    jobs = [(d, c) for d in daterange(start, end) for c in cycles]
    if args.dry_run:
        jobs = [(start, cycles[0])]

    print(f"HRRR -> {data_root / 'hrrr'} | {len(jobs)} init-cycles | leads f00-f{args.max_lead}")

    def run(job):
        d, c = job
        return process_cycle(
            d, c, data_root, args.max_lead, args.download_retries, args.resume, args.keep_grib
        )

    if args.workers <= 1:
        for job in jobs:
            print(f"  {run(job)}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run, job): job for job in jobs}
            for fut in as_completed(futures):
                d, c = futures[fut]
                try:
                    print(f"  {fut.result()}")
                except Exception as exc:  # noqa: BLE001 - report and continue backfill
                    print(f"  FAILED {d:%Y%m%d} t{c:02d}z: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()

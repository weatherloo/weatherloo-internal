#!/usr/bin/env python3
"""Download raw HRRR forecasts (bbox-subset) for bias-correction training.

Fetches NOAA HRRR surface output (``wrfsfc``, 3 km CONUS) from the AWS Open Data
bucket ``noaa-hrrr-bdp-pds`` for the 00/06/12/18Z cycles, hourly leads f00..f48,
crops to the Kitchener-Waterloo bounding box on HRRR's **native Lambert-conformal
grid** (no reprojection/regridding -- that's a downstream modelling choice), and
writes one compact NetCDF per init-cycle:

    {data_root}/hrrr/{YYYY}/{YYYYMMDD}/hrrr_{YYYYMMDD}_t{HH}z.nc
        dims:  (lead, y, x)
        vars:  t2m (K), u10 (m/s), v10 (m/s), tp (kg/m^2, 1-hour precip),
               q2 (kg/kg), psfc (Pa), pblh (m), hgt (m), tsk (K), ust (m/s)
        coords: latitude/longitude (2-D), lead (hours)

Only the needed GRIB messages per lead are downloaded via the ``.idx``
byte-range trick (HTTP Range headers), so we never pull whole HRRR files.
Temporary GRIB blobs live under a *private* subdirectory of
``{data_root}/.cache/hrrr_grib/`` (``job-{SLURM_JOB_ID}/`` or ``pid-{pid}/``,
plus ``proc-{pid}/`` per process worker) so parallel Slurm/process workers never
share one NFS directory — that race segfaults eccodes. Blobs are deleted after
each lead is cropped into memory (with a ``--max-cache-gb`` safety prune).
Cycle-level parallelism uses ``ProcessPoolExecutor`` (eccodes is not thread-safe);
HTTP field fetches within a lead use a short-lived thread pool. Pass
``--shared-cache`` only for debugging.

Notes / caveats:
  * 48 h leads only exist from HRRRv4 (~2020-12-02). Earlier HRRRv3 extended
    cycles cap at f36; missing leads 404 and are skipped (including mid-cycle
    holes -- we do NOT hard-code version dates; availability is probed).
  * ``tp`` is **1-hour precipitation** (kg/m^2): GRIB APCP is accumulated from
    init, then de-accumulated lead-to-lead (``acc[f]-acc[f-1]``; on bucket reset
    use ``acc[f]``). f00 is the 0–0 h bucket (zeros). Missing APCP -> NaN.

Mirrors the download/cache patterns in
``benchmarking-site/data/hrrr_interpolated/compute_benchmark.py``.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    cleanup_cycle_cache,
    download_bytes,
    enforce_cache_budget,
    parse_idx_ranges,
    private_hrrr_cache_dir,
    resolve_data_root,
    resolve_hrrr_cache_dir,
    unlink_grib_and_sidecars,
    unlink_quiet,
)

AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
INIT_HOURS_UTC = (0, 6, 12, 18)
DEFAULT_START = "2018-07-13"  # first full day of HRRRv3 operational archive
MAX_LEAD_DEFAULT = 48
# Soft cap per private cache dir; home SSD crashes ~20 GB of tiny GRIBs.
DEFAULT_MAX_CACHE_GB = 8.0
# Mid-cycle archive holes of ~4 leads exist; stop after this many consecutive 404s
# once we have started receiving leads (avoids probing f37–f48 on HRRRv3).
MAX_CONSECUTIVE_MISSING_LEADS = 6
# zlib level for NetCDF writes (1 = fast; 4 was a noticeable post-download drag).
NETCDF_COMPLEVEL = 1

# GRIB inventory needles -> output variable name. Order defines NetCDF vars.
GRIB_NEEDLES = {
    "t2m": ":TMP:2 m above ground:",
    "u10": ":UGRD:10 m above ground:",
    "v10": ":VGRD:10 m above ground:",
    "tp": ":APCP:surface:",
    "q2": ":SPFH:2 m above ground:",  # specific humidity in GRIB -> mixing ratio on write
    "psfc": ":PRES:surface:",
    "pblh": ":HPBL:surface:",
    "hgt": ":HGT:surface:",
    "tsk": ":TMP:surface:",
    "ust": ":FRICV:surface:",
}
VAR_UNITS = {
    "t2m": "K",
    "u10": "m s-1",
    "v10": "m s-1",
    "tp": "kg m-2",
    "q2": "kg kg-1",
    "psfc": "Pa",
    "pblh": "m",
    "hgt": "m",
    "tsk": "K",
    "ust": "m s-1",
}
TP_ATTRS = {
    "units": "kg m-2",
    "long_name": "1-hour total precipitation",
    "accum": "1-hour",
    "cell_methods": "time: sum (interval: 1 hour)",
}
REQUIRED_VARS = tuple(GRIB_NEEDLES.keys())


def deaccumulate_apcp(acc: np.ndarray, leads: np.ndarray | list[int] | None = None) -> np.ndarray:
    """Convert forecast-accumulated APCP to 1-hour (or inter-lead) precip.

    For consecutive leads: ``hourly[i] = max(0, acc[i] - acc[i-1])``, except on
    a bucket reset (``acc[i] < acc[i-1]``) where the bucket value itself is used.
    Lead gaps > 1 h keep the multi-hour difference (not divided); normal archives
    are hourly so this is true 1-hour precip. ``leads`` is unused but kept for
    call-site clarity / future gap handling.
    """
    del leads  # reserved for gap-aware scaling
    acc = np.asarray(acc, dtype=np.float32)
    out = np.empty_like(acc)
    if acc.shape[0] == 0:
        return out
    out[0] = acc[0]
    for i in range(1, acc.shape[0]):
        diff = acc[i] - acc[i - 1]
        # Bucket reset: accumulation restarted; use current bucket as period precip.
        out[i] = np.where(np.isnan(diff), np.nan, np.where(diff >= 0, diff, acc[i])).astype(
            np.float32
        )
    return out


def tp_is_hourly(ds: xr.Dataset) -> bool:
    if "tp" not in ds.data_vars:
        return False
    return ds["tp"].attrs.get("accum") == "1-hour"


def ensure_hourly_tp(ds: xr.Dataset) -> tuple[xr.Dataset, bool]:
    """In-place convert accumulated ``tp`` to hourly if attrs say it is not yet."""
    if "tp" not in ds.data_vars or tp_is_hourly(ds):
        return ds, False
    hourly = deaccumulate_apcp(np.asarray(ds["tp"].values), ds.lead.values)
    out = ds.copy()
    out["tp"] = (("lead", "y", "x"), hourly, dict(TP_ATTRS))
    return out, True


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def grib_url(day: date, cycle: int, fxx: int) -> str:
    return (
        f"{AWS_BASE}/hrrr.{day:%Y%m%d}/conus/"
        f"hrrr.t{cycle:02d}z.wrfsfcf{fxx:02d}.grib2"
    )


def fetch_idx(cache_dir: Path, url: str, cache_key: str, retries: int) -> str | None:
    """Return the ``.idx`` inventory text, or None if the lead does not exist."""
    idx_path = cache_dir / f"{cache_key}.idx"
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


def download_fields(
    cache_dir: Path,
    url: str,
    cache_key: str,
    retries: int,
    fields: tuple[str, ...] | None = None,
) -> dict[str, Path] | None:
    """Byte-range download the needed GRIB messages; return {var: grib blob path}.

    Returns None if the lead's ``.idx`` is unavailable (missing lead).
    Field blobs for one lead are fetched concurrently (HTTP only — no eccodes).
    """
    fields_to_fetch = fields or REQUIRED_VARS
    needles = {k: GRIB_NEEDLES[k] for k in fields_to_fetch}
    idx_text = fetch_idx(cache_dir, url, cache_key, retries)
    if idx_text is None:
        return None
    ranges = parse_idx_ranges(idx_text, needles)

    cache_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, Path, int, int]] = []
    paths: dict[str, Path] = {}
    for field in fields_to_fetch:
        if field not in ranges:
            # e.g. APCP absent at f00; that field is simply skipped this lead.
            continue
        blob = cache_dir / f"{cache_key}_{field}.grib2"
        if blob.exists():
            paths[field] = blob
        else:
            start, end = ranges[field]
            pending.append((field, blob, start, end))

    def _fetch(item: tuple[str, Path, int, int]) -> tuple[str, Path]:
        field, blob, start, end = item
        blob.write_bytes(download_bytes(url, start, end, retries=retries))
        return field, blob

    if pending:
        workers = min(len(pending), len(GRIB_NEEDLES))
        if workers == 1:
            field, blob = _fetch(pending[0])
            paths[field] = blob
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for field, blob in pool.map(_fetch, pending):
                    paths[field] = blob
    return paths


# --- static bbox index window on HRRR's curvilinear grid (computed once) ---
_WINDOW: tuple[slice, slice] | None = None


def bbox_window(sample_grib: Path) -> tuple[slice, slice]:
    """Row/col slice covering the padded bbox; HRRR's grid is static so cache it."""
    global _WINDOW
    if _WINDOW is not None:
        return _WINDOW
    ds = cfgrib.open_dataset(sample_grib)
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


def spfh_to_mixing_ratio(q: np.ndarray) -> np.ndarray:
    """Convert GRIB specific humidity (kg/kg) to mass mixing ratio r = q/(1-q)."""
    q = np.asarray(q, dtype=np.float64)
    return (q / (1.0 - q)).astype(np.float32)


def read_field(grib_path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (values, lat, lon) cropped to the bbox window for one GRIB message."""
    jsl, isl = bbox_window(grib_path)
    ds = cfgrib.open_dataset(grib_path)
    var = list(ds.data_vars)[0]
    vals = np.asarray(ds[var].values, dtype=np.float32)[jsl, isl]
    if field == "q2":
        vals = spfh_to_mixing_ratio(vals)
    lat = np.asarray(ds.latitude.values, dtype=np.float32)[jsl, isl]
    lon = np.asarray(ds.longitude.values, dtype=np.float32)[jsl, isl]
    lon = np.where(lon > 180.0, lon - 360.0, lon).astype(np.float32)
    return vals, lat, lon


def open_cycle_nc(path: Path) -> xr.Dataset:
    """Open an HRRR cycle NetCDF without decoding lead hours as timedeltas."""
    return xr.open_dataset(path, decode_timedelta=False)


def cycle_file_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with open_cycle_nc(path) as ds:
            return all(v in ds.data_vars for v in REQUIRED_VARS) and tp_is_hourly(ds)
    except Exception:
        return False


def missing_cycle_vars(path: Path) -> tuple[str, ...]:
    if not path.is_file() or path.stat().st_size == 0:
        return REQUIRED_VARS
    try:
        with open_cycle_nc(path) as ds:
            return tuple(v for v in REQUIRED_VARS if v not in ds.data_vars)
    except Exception:
        return REQUIRED_VARS


def lead_hours_list(ds: xr.Dataset) -> list[int]:
    lead = ds["lead"]
    vals = lead.values
    if np.issubdtype(vals.dtype, np.timedelta64):
        return [int(v / np.timedelta64(1, "h")) for v in vals]
    return [int(v) for v in vals]


def write_cycle_netcdf(ds: xr.Dataset, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": NETCDF_COMPLEVEL} for v in ds.data_vars}
    tmp = out_path.parent / (out_path.name + ".tmp")
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.replace(out_path)


def build_cycle_dataset(
    cache_dir: Path,
    day: date,
    cycle: int,
    max_lead: int,
    retries: int,
    fields: tuple[str, ...] | None = None,
    lead_list: list[int] | None = None,
) -> xr.Dataset | None:
    """Download + crop leads of one init-cycle into a single Dataset.

    When ``fields`` is set, only those GRIB messages are fetched (for patching
    existing NetCDFs). When ``lead_list`` is set, only those forecast hours
    are read (used when upgrading a file that already has a lead axis).

    GRIB blobs are deleted as soon as each lead is read into memory so the
    on-disk cache stays small even across many parallel workers.
    """
    fields_to_fetch = fields or REQUIRED_VARS
    leads: list[int] = []
    field_arrays: dict[str, list[np.ndarray]] = {v: [] for v in fields_to_fetch}
    lat2d = lon2d = None
    consecutive_miss = 0
    fxx_iter: list[int] | range = lead_list if lead_list is not None else range(0, max_lead + 1)
    upgrade_mode = lead_list is not None

    for fxx in fxx_iter:
        cache_key = f"{day:%Y%m%d}_t{cycle:02d}z_f{fxx:02d}"
        paths = download_fields(
            cache_dir, grib_url(day, cycle, fxx), cache_key, retries, fields_to_fetch
        )
        if paths is None:
            if upgrade_mode:
                # Keep lead alignment with the existing file; fill missing fields with NaN.
                leads.append(fxx)
                for field in fields_to_fetch:
                    field_arrays[field].append(None)
            else:
                consecutive_miss += 1
                if leads and consecutive_miss >= MAX_CONSECUTIVE_MISSING_LEADS:
                    break
            continue
        consecutive_miss = 0
        if not paths:
            unlink_quiet(cache_dir / f"{cache_key}.idx")
            if upgrade_mode:
                leads.append(fxx)
                for field in fields_to_fetch:
                    field_arrays[field].append(None)
            continue  # idx present but none of our fields matched; skip lead in full build
        leads.append(fxx)
        shape = None
        for field in fields_to_fetch:
            if field in paths:
                vals, lat2d_, lon2d_ = read_field(paths[field], field)
                if lat2d is None:
                    lat2d, lon2d = lat2d_, lon2d_
                shape = vals.shape
                field_arrays[field].append(vals)
                unlink_grib_and_sidecars(paths[field])
            else:
                field_arrays[field].append(None)  # placeholder; filled with NaN below
        unlink_quiet(cache_dir / f"{cache_key}.idx")
        # backfill any None placeholders now that we know the crop shape
        for field in fields_to_fetch:
            if field_arrays[field][-1] is None and shape is not None:
                field_arrays[field][-1] = np.full(shape, np.nan, dtype=np.float32)

    if not leads or lat2d is None:
        return None

    leads_arr = np.asarray(leads, dtype=np.int16)
    data_vars = {}
    for field in fields_to_fetch:
        stacked = np.stack(field_arrays[field], axis=0)  # (lead, y, x)
        if field == "tp":
            stacked = deaccumulate_apcp(stacked, leads_arr)
            data_vars[field] = (("lead", "y", "x"), stacked, dict(TP_ATTRS))
        else:
            data_vars[field] = (
                ("lead", "y", "x"),
                stacked,
                {"units": VAR_UNITS[field]},
            )

    init = datetime(day.year, day.month, day.day, cycle, tzinfo=timezone.utc)
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "lead": ("lead", leads_arr, {"long_name": "forecast_lead_hour"}),
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
    cache_dir: Path,
    max_lead: int,
    retries: int,
    resume: bool,
    max_cache_bytes: int,
) -> str:
    out_path = (
        data_root / "hrrr" / f"{day:%Y}" / f"{day:%Y%m%d}" / f"hrrr_{day:%Y%m%d}_t{cycle:02d}z.nc"
    )
    if resume and cycle_file_complete(out_path):
        with open_cycle_nc(out_path) as existing:
            fixed, changed = ensure_hourly_tp(existing.load())
        if changed:
            write_cycle_netcdf(fixed, out_path)
            cleanup_cycle_cache(cache_dir, day, cycle)
            return f"deaccum-tp {out_path.name}"
        cleanup_cycle_cache(cache_dir, day, cycle)
        return f"skip {out_path.name}"

    missing = missing_cycle_vars(out_path) if resume and out_path.exists() else REQUIRED_VARS
    if resume and not missing:
        with open_cycle_nc(out_path) as existing:
            fixed, changed = ensure_hourly_tp(existing.load())
        if changed:
            write_cycle_netcdf(fixed, out_path)
            cleanup_cycle_cache(cache_dir, day, cycle)
            return f"deaccum-tp {out_path.name}"
        cleanup_cycle_cache(cache_dir, day, cycle)
        return f"skip {out_path.name}"

    freed = enforce_cache_budget(cache_dir, max_cache_bytes)
    if freed:
        print(f"  cache prune freed {freed / (1024**3):.2f} GiB (cap {max_cache_bytes / (1024**3):.1f} GiB)")

    if resume and out_path.exists() and missing != REQUIRED_VARS:
        with open_cycle_nc(out_path) as existing:
            lead_list = lead_hours_list(existing)
        patch = build_cycle_dataset(
            cache_dir,
            day,
            cycle,
            max_lead,
            retries,
            fields=missing,
            lead_list=lead_list,
        )
        if patch is None:
            cleanup_cycle_cache(cache_dir, day, cycle)
            return f"MISSING {day:%Y%m%d} t{cycle:02d}z (no leads to patch)"
        with open_cycle_nc(out_path) as existing:
            merged = existing.load()
        for var in patch.data_vars:
            merged[var] = patch[var]
        merged, _ = ensure_hourly_tp(merged)
        write_cycle_netcdf(merged, out_path)
        cleanup_cycle_cache(cache_dir, day, cycle)
        enforce_cache_budget(cache_dir, max_cache_bytes)
        return f"patched {out_path.name} (+{','.join(missing)}, {merged.sizes['lead']} leads)"

    ds = build_cycle_dataset(cache_dir, day, cycle, max_lead, retries)
    if ds is None:
        cleanup_cycle_cache(cache_dir, day, cycle)
        return f"MISSING {day:%Y%m%d} t{cycle:02d}z (no leads published)"

    write_cycle_netcdf(ds, out_path)
    # NetCDF is durable on data_root; drop this cycle's GRIB leftovers immediately.
    cleanup_cycle_cache(cache_dir, day, cycle)
    enforce_cache_budget(cache_dir, max_cache_bytes)
    return f"wrote {out_path.name} ({ds.sizes['lead']} leads)"


def _run_cycle_job(
    payload: tuple[
        date,
        int,
        str,
        str,
        int,
        int,
        bool,
        int,
    ],
) -> str:
    """ProcessPool worker: one init-cycle with a per-process GRIB cache.

    eccodes/cfgrib is not thread-safe; processes isolate native state. Each
    worker also gets its own cache subdirectory so prune/delete cannot race.
    """
    day, cycle, data_root_s, cache_parent_s, max_lead, retries, resume, max_cache_bytes = payload
    data_root = Path(data_root_s)
    proc_cache = Path(cache_parent_s) / f"proc-{os.getpid()}"
    proc_cache.mkdir(parents=True, exist_ok=True)
    try:
        return process_cycle(
            day,
            cycle,
            data_root,
            proc_cache,
            max_lead,
            retries,
            resume,
            max_cache_bytes,
        )
    except Exception as exc:  # stringify in worker; some urllib errors are not pickleable
        return f"FAILED {day:%Y%m%d} t{cycle:02d}z: {type(exc).__name__}: {exc}"


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Output root (else $WEATHERLOO_DATA_ROOT, else repo data/)")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="GRIB cache *base* (else $WEATHERLOO_HRRR_CACHE, else {data_root}/.cache/hrrr_grib); "
        "a private job-/pid- subdir is used unless --shared-cache",
    )
    parser.add_argument(
        "--shared-cache",
        action="store_true",
        help="Use the cache base dir directly (unsafe with parallel Slurm jobs)",
    )
    parser.add_argument(
        "--max-cache-gb",
        type=float,
        default=DEFAULT_MAX_CACHE_GB,
        help=f"Prune oldest cache files above this size (default {DEFAULT_MAX_CACHE_GB}; home crashes ~20)",
    )
    parser.add_argument("--start-date", default=DEFAULT_START, help="YYYY-MM-DD inclusive (default HRRRv3 start)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive (default: today UTC)")
    parser.add_argument("--cycles", default="0,6,12,18", help="Comma-separated init hours UTC")
    parser.add_argument("--max-lead", type=int, default=MAX_LEAD_DEFAULT, help="Max forecast hour to request")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel init-cycle *processes* (eccodes is not thread-safe; default 4)",
    )
    parser.add_argument("--download-retries", type=int, default=6, help="Retries per GRIB/idx download")
    parser.add_argument("--resume", action="store_true", help="Skip cycles whose NetCDF already exists")
    parser.add_argument("--dry-run", action="store_true", help="Process only the start date's first cycle")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    cache_base = resolve_hrrr_cache_dir(data_root, args.cache_dir)
    cache_dir = cache_base if args.shared_cache else private_hrrr_cache_dir(cache_base)
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_cache_bytes = int(max(0.0, args.max_cache_gb) * (1024**3))

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

    print(
        f"HRRR -> {data_root / 'hrrr'} | cache {cache_dir} "
        f"(max {args.max_cache_gb:g} GiB) | {len(jobs)} init-cycles | leads f00-f{args.max_lead} "
        f"| workers={args.workers} (processes)"
    )

    payloads = [
        (
            d,
            c,
            str(data_root),
            str(cache_dir),
            args.max_lead,
            args.download_retries,
            args.resume,
            max_cache_bytes,
        )
        for d, c in jobs
    ]

    try:
        if args.workers <= 1:
            for payload in payloads:
                print(f"  {_run_cycle_job(payload)}")
        else:
            # spawn: avoid forking after eccodes/cfgrib may have initialized.
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
                futures = {pool.submit(_run_cycle_job, payload): payload for payload in payloads}
                for fut in as_completed(futures):
                    d, c = futures[fut][0], futures[fut][1]
                    try:
                        print(f"  {fut.result()}")
                    except Exception as exc:  # noqa: BLE001 - report and continue backfill
                        print(f"  FAILED {d:%Y%m%d} t{c:02d}z: {exc}")
        print("Done.")
    finally:
        # Drop the private subdirectory; leave the shared base for other jobs.
        if not args.shared_cache and cache_dir.is_dir() and cache_dir != cache_base:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download raw station observations into {data_root}/observations/.

What this script downloads
- ECCC climate stations: discovers all stations within the Weatherloo bbox that
  have hourly data overlapping the requested date range, then downloads raw
  hourly observations via the MSC GeoMet OGC API (api.weather.gc.ca).
- UWaterloo "Eric D. Soulis" station: downloads the public bulk CSVs (15-min)
  for each year (HOBO subset for 2015+).

Outputs (default data_root resolved like other downloaders)
{data_root}/observations/
  eccc/
    stations.json                      # discovered station metadata (bbox-filtered)
    stn_<STN_ID>_<slug>/
      meta.json
      hourly/
        eccc_climate_hourly_<year>.json
  uwaterloo/
    eric_d_soulis/
      meta.json
      15min/
        uw_hobo_15min_<year>.csv
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from weather_download_common import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, PAD_DEG, resolve_data_root

ECCC_CLIMATE_STATIONS = "https://api.weather.gc.ca/collections/climate-stations/items"
ECCC_CLIMATE_HOURLY = "https://api.weather.gc.ca/collections/climate-hourly/items"

SOULIS_HOBO_URL = "https://www.civil.uwaterloo.ca/weather/download/Hobo_15minutedata_{year}.csv"


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "station"


def _download(url: str, timeout_s: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "weatherloo-internal/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _parse_years(spec: str | None, *, start: date, end: date) -> tuple[int, ...]:
    """Parse --years as '2018,2019,2020' (subset of [start.year..end.year])."""
    all_years = tuple(range(start.year, end.year + 1))
    if not spec:
        return all_years
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ya, yb = int(a), int(b)
            lo, hi = (ya, yb) if ya <= yb else (yb, ya)
            years.update(range(lo, hi + 1))
        else:
            years.add(int(part))
    years = {y for y in years if start.year <= y <= end.year}
    return tuple(sorted(years))


def _parse_station_latlon(props: dict) -> tuple[float | None, float | None]:
    # climate-stations LATITUDE/LONGITUDE are typically scaled ints (1e7).
    lat = props.get("LATITUDE")
    lon = props.get("LONGITUDE")
    try:
        if isinstance(lat, int):
            lat_f = lat / 1e7
        else:
            lat_f = float(lat)
        if isinstance(lon, int):
            lon_f = lon / 1e7
        else:
            lon_f = float(lon)
        return lat_f, lon_f
    except Exception:
        return None, None


def discover_eccc_hourly_stations(
    *,
    start: date,
    end: date,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> list[dict]:
    """Return station property dicts for bbox stations with hourly overlap."""
    # bbox: minX, minY, maxX, maxY (lon,lat)
    bbox = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    offset = 0
    limit = 1000
    stations: list[dict] = []
    # Filter stations with hourly data whose HLY date window intersects [start,end].
    # Many properties are queryable directly (see /queryables); filter syntax here
    # matches what the existing ECCC hourly downloader uses.
    filt = (
        "properties.HAS_HOURLY_DATA='Y' "
        f"AND properties.HLY_LAST_DATE>='{_ymd(start)}' "
        f"AND properties.HLY_FIRST_DATE<='{_ymd(end)}'"
    )

    while True:
        params = urllib.parse.urlencode(
            {
                "bbox": bbox,
                "filter": filt,
                "limit": limit,
                "offset": offset,
            }
        )
        url = f"{ECCC_CLIMATE_STATIONS}?{params}"
        data = json.loads(_download(url).decode("utf-8"))
        feats = data.get("features", [])
        if not feats:
            break
        for f in feats:
            stations.append(f.get("properties", {}))
        if len(feats) < limit:
            break
        offset += limit
    # De-dupe by STN_ID
    out: dict[int, dict] = {}
    for p in stations:
        stn_id = p.get("STN_ID")
        if isinstance(stn_id, int):
            out[stn_id] = p
    return [out[k] for k in sorted(out)]


def fetch_eccc_hourly_year(stn_id: int, year: int) -> list[dict]:
    """Paginate ECCC climate-hourly for one station and UTC calendar year."""
    rows: list[dict] = []
    offset = 0
    limit = 10000
    filt = f"properties.STN_ID={stn_id} AND properties.UTC_YEAR={year}"
    while True:
        params = urllib.parse.urlencode(
            {
                "filter": filt,
                "limit": limit,
                "offset": offset,
            }
        )
        url = f"{ECCC_CLIMATE_HOURLY}?{params}"
        data = json.loads(_download(url).decode("utf-8"))
        feats = data.get("features", [])
        if not feats:
            break
        for f in feats:
            rows.append(f.get("properties", {}))
        if len(feats) < limit:
            break
        offset += limit
    return rows


@dataclass(frozen=True)
class StationJob:
    stn_id: int
    station_name: str
    slug: str
    out_dir: Path
    years: tuple[int, ...]
    resume: bool


def _run_station_job(job: StationJob) -> tuple[int, int]:
    """Return (stn_id, n_year_files_written)."""
    n_written = 0
    for y in job.years:
        out = job.out_dir / "hourly" / f"eccc_climate_hourly_{y}.json"
        if job.resume and out.exists() and out.stat().st_size > 0:
            continue
        rows = fetch_eccc_hourly_year(job.stn_id, y)
        _write_json(out, {"station_id": job.stn_id, "station_name": job.station_name, "year": y, "rows": rows})
        n_written += 1
    return job.stn_id, n_written


def download_soulis_hobo(*, out_dir: Path, year: int, resume: bool) -> Path:
    url = SOULIS_HOBO_URL.format(year=year)
    dest = out_dir / "15min" / f"uw_hobo_15min_{year}.csv"
    if resume and dest.exists() and dest.stat().st_size > 0:
        return dest
    data = _download(url)
    _write_bytes(dest, data)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--start", type=str, default="2018-07-13")
    p.add_argument("--end", type=str, default="2026-07-15")
    p.add_argument(
        "--pad-deg",
        type=float,
        default=0.0,
        help="Optional bbox padding in degrees (default: 0; HRRR/ERA5 use 0.5).",
    )
    p.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma list or ranges, e.g. '2018-2020,2023' (defaults to full span).",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-eccc", action="store_true")
    p.add_argument("--skip-soulis", action="store_true")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    data_root = resolve_data_root(args.data_root)
    obs_root = data_root / "observations"
    eccc_root = obs_root / "eccc"
    uw_root = obs_root / "uwaterloo" / "eric_d_soulis"
    obs_root.mkdir(parents=True, exist_ok=True)

    years = _parse_years(args.years, start=start, end=end)

    if not args.skip_eccc:
        pad = float(args.pad_deg or 0.0)
        lat_min = LAT_MIN - pad
        lat_max = LAT_MAX + pad
        lon_min = LON_MIN - pad
        lon_max = LON_MAX + pad
        stations = discover_eccc_hourly_stations(
            start=start, end=end, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
        )
        _write_json(
            eccc_root / "stations.json",
            {
                "schema_version": "1",
                "source": "eccc_climate_stations",
                "bbox": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
                "period": {"start": args.start, "end": args.end},
                "generated_at_utc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "stations": stations,
            },
        )

        jobs: list[StationJob] = []
        for st in stations:
            stn_id = st.get("STN_ID")
            if not isinstance(stn_id, int):
                continue
            name = str(st.get("STATION_NAME") or f"STN_{stn_id}")
            slug = _slugify(name)
            lat, lon = _parse_station_latlon(st)
            st_dir = eccc_root / f"stn_{stn_id}_{slug}"
            _write_json(
                st_dir / "meta.json",
                {
                    "schema_version": "1",
                    "source": "eccc_climate_stations",
                    "stn_id": stn_id,
                    "station_name": name,
                    "coordinates": {"lat": lat, "lon": lon},
                    "hourly_window": {"first": st.get("HLY_FIRST_DATE"), "last": st.get("HLY_LAST_DATE")},
                    "download": {"collection": "climate-hourly", "by": "UTC_YEAR"},
                },
            )
            jobs.append(
                StationJob(
                    stn_id=stn_id,
                    station_name=name,
                    slug=slug,
                    out_dir=st_dir,
                    years=years,
                    resume=bool(args.resume),
                )
            )

        if jobs:
            n_workers = max(1, int(args.workers))
            if n_workers == 1:
                results = [_run_station_job(j) for j in jobs]
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=n_workers) as pool:
                    results = pool.map(_run_station_job, jobs)
            wrote = sum(n for _sid, n in results)
            print(f"[eccc] stations={len(jobs)} year_files_written={wrote}")

    if not args.skip_soulis:
        _write_json(
            uw_root / "meta.json",
            {
                "schema_version": "1",
                "station_id": "eric_d_soulis",
                "station_name": "Eric D. Soulis station",
                "source": "uwaterloo_civil_weather",
                "coordinates": {"lat": 43.4668, "lon": -80.5164},
                "period": {"start": args.start, "end": args.end},
                "notes": {
                    "public_bulk": "2015+ uses HOBO 15-minute subset bulk CSVs",
                    "time_basis": "Archive timestamps are local standard time (UTC-5, no DST)",
                },
            },
        )
        for y in years:
            if y < 2015:
                # Main-logger yearly CSV exists historically, but for this requested range (2018+)
                # the HOBO bulk is the public file.
                continue
            dest = download_soulis_hobo(out_dir=uw_root, year=y, resume=bool(args.resume))
            print(f"[soulis] {y} -> {dest}")

    print(f"Done. observations_root={obs_root}")


if __name__ == "__main__":
    main()


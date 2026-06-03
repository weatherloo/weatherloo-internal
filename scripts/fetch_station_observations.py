#!/usr/bin/env python3
"""Download station observations and write normalized 6-hourly JSON."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBS_ROOT = ROOT / "benchmarking-site" / "data" / "observations"

ECCC_API = "https://api.weather.gc.ca/collections/climate-hourly/items"
CYYZ_STN_ID = 51459

SOULIS_MAIN_URL = (
    "https://www.civil.uwaterloo.ca/weather/download/{year}_weather_station_data.csv"
)
SOULIS_HOBO_URL = (
    "https://www.civil.uwaterloo.ca/weather/download/Hobo_15minutedata_{year}.csv"
)

MISSING = -9999.9
TARGET_HOURS_UTC = (0, 6, 12, 18)
# UW archive timestamps are local standard time (no DST), same as Canada Eastern Standard.
UW_ARCHIVE_TZ = timezone(timedelta(hours=-5))


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "weatherloo-internal/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def parse_utc(s: str) -> datetime:
    s = s.strip()
    if "T" in s:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    else:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def six_hourly_targets(year: int) -> list[datetime]:
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, 23, tzinfo=timezone.utc)
    out: list[datetime] = []
    cur = start
    while cur <= end:
        if cur.hour in TARGET_HOURS_UTC:
            out.append(cur)
        cur += timedelta(hours=1)
    return out


def clean_metric(val: float | None, *, non_negative: bool = False) -> float | None:
    if val is None or is_missing(val):
        return None
    if non_negative and val < 0:
        return None
    return val


def sample_at_utc(
    samples: list[tuple[datetime, dict[str, float | None]]],
    target: datetime,
    window_minutes: int = 45,
) -> dict[str, float | None]:
    """Pick value at exact UTC target, else nearest within window (all times UTC)."""
    exact = [vals for ts, vals in samples if ts == target]
    if exact:
        return exact[0]

    in_hour = [
        (ts, vals)
        for ts, vals in samples
        if ts.replace(minute=0, second=0, microsecond=0)
        == target.replace(minute=0, second=0, microsecond=0)
    ]
    if in_hour:
        _, vals = min(in_hour, key=lambda pair: abs(pair[0] - target))
        return vals

    best: tuple[datetime, dict[str, float | None]] | None = None
    best_delta = timedelta(days=999)
    window = timedelta(minutes=window_minutes)
    for ts, vals in samples:
        delta = abs(ts - target)
        if delta <= window and delta < best_delta:
            best_delta = delta
            best = (ts, vals)
    if best is None:
        return {"t2m": None, "wind_speed": None}
    return best[1]


def build_observations_json(
    *,
    station_id: str,
    station_name: str,
    source: str,
    source_station_id: int | str | None,
    coordinates: dict[str, float],
    year: int,
    variables_meta: dict,
    samples: list[tuple[datetime, dict[str, float | None]]],
    extra_meta: dict | None = None,
) -> dict:
    targets = six_hourly_targets(year)
    observations = []
    for t in targets:
        vals = sample_at_utc(samples, t)
        observations.append(
            {
                "valid_time": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "t2m": vals["t2m"],
                "wind_speed": vals["wind_speed"],
                "flags": {
                    "t2m": None if vals["t2m"] is not None else "missing",
                    "wind_speed": None
                    if vals["wind_speed"] is not None
                    else "missing",
                },
            }
        )

    period_start = targets[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    period_end = targets[-1].strftime("%Y-%m-%dT%H:%M:%SZ")

    doc = {
        "schema_version": "1",
        "station_id": station_id,
        "station_name": station_name,
        "source": source,
        "source_station_id": source_station_id,
        "coordinates": coordinates,
        "period": {"start": period_start, "end": period_end},
        "cadence_hours": 6,
        "time_standard": "UTC",
        "resampling": {
            "grid_utc_hours": list(TARGET_HOURS_UTC),
            "method": "exact_utc_instant, else same_utc_hour, else nearest_within_45min",
        },
        "variables": variables_meta,
        "observations": observations,
    }
    if extra_meta:
        doc["processing_notes"] = extra_meta
    return doc


def fetch_eccc_hourly(stn_id: int, year: int) -> list[dict]:
    """Paginate ECCC climate-hourly for one station and calendar year."""
    features: list[dict] = []
    offset = 0
    limit = 10000
    # UTC_YEAR so boundary hours (e.g. 2025-01-01T00:00:00Z) are included even when
    # LOCAL_YEAR is still the previous calendar year.
    filt = f"properties.STN_ID={stn_id} AND properties.UTC_YEAR={year}"

    while True:
        params = urllib.parse.urlencode(
            {
                "filter": filt,
                "limit": limit,
                "offset": offset,
            }
        )
        url = f"{ECCC_API}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "weatherloo-internal/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        batch = data.get("features", [])
        if not batch:
            break
        for f in batch:
            features.append(f["properties"])
        if len(batch) < limit:
            break
        offset += limit

    return features


def eccc_to_samples(rows: list[dict]) -> list[tuple[datetime, dict[str, float | None]]]:
    samples: list[tuple[datetime, dict[str, float | None]]] = []
    for p in rows:
        utc = p.get("UTC_DATE")
        if not utc:
            continue
        ts = parse_utc(utc)
        temp = p.get("TEMP")
        wind = p.get("WIND_SPEED")
        t2m = clean_metric(float(temp)) if temp is not None and temp != "" else None
        ws = (
            clean_metric(float(wind), non_negative=True)
            if wind is not None and wind != ""
            else None
        )
        samples.append((ts, {"t2m": t2m, "wind_speed": ws}))
    return samples


def is_missing(val: float) -> bool:
    return val <= MISSING or val < -9000


def load_soulis_main_csv(path: Path, year: int) -> list[tuple[datetime, dict[str, float | None]]]:
    """15-min UW main logger archive (1998–2014). Temp in Fahrenheit."""
    samples: list[tuple[datetime, dict[str, float | None]]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Ambient Air Temperature col 9, Wind Speed col 12
        for row in reader:
            if len(row) < 13:
                continue
            try:
                y = int(row[1])
                jday = int(row[2])
                minute_of_day = int(float(row[3]))
            except (ValueError, IndexError):
                continue
            if y != year:
                continue
            dt_lst = datetime(year, 1, 1, tzinfo=UW_ARCHIVE_TZ) + timedelta(
                days=jday - 1, minutes=minute_of_day
            )
            dt = dt_lst.astimezone(timezone.utc)
            try:
                temp_f = float(row[9])
                wind = float(row[12])
            except ValueError:
                continue
            t2m = clean_metric(round(f_to_c(temp_f), 2)) if not is_missing(temp_f) else None
            ws = clean_metric(wind, non_negative=True)
            samples.append((dt, {"t2m": t2m, "wind_speed": ws}))
    return samples


def load_soulis_hobo_csv(path: Path) -> list[tuple[datetime, dict[str, float | None]]]:
    """15-min HOBO subset (2015+ public bulk). Temp °C, wind km/h at ~4.4 m."""
    samples: list[tuple[datetime, dict[str, float | None]]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 12:
                continue
            try:
                y, m, d, h, mi = (int(row[i].strip()) for i in range(5))
                temp = float(row[5].strip())
                wind = float(row[10].strip())
            except (ValueError, IndexError):
                continue
            # Archive notes: local standard time (EST, no DST). Approximate as UTC-5.
            dt = datetime(y, m, d, h, mi, tzinfo=UW_ARCHIVE_TZ).astimezone(timezone.utc)
            t2m = clean_metric(round(temp, 2))
            ws = clean_metric(round(wind, 2), non_negative=True)
            samples.append((dt, {"t2m": t2m, "wind_speed": ws}))
    return samples


def write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def process_cyyz(year: int) -> None:
    station_dir = OBS_ROOT / "cyyz"
    raw_path = station_dir / "raw" / f"eccc_hourly_{year}.json"

    print(f"[cyyz] Fetching ECCC hourly STN_ID={CYYZ_STN_ID} for {year}…")
    rows = fetch_eccc_hourly(CYYZ_STN_ID, year)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"features": rows}, indent=2) + "\n")
    print(f"[cyyz] {len(rows)} hourly records → {raw_path}")

    samples = eccc_to_samples(rows)
    doc = build_observations_json(
        station_id="cyyz",
        station_name="Toronto Pearson (CYYZ)",
        source="eccc_climate_hourly",
        source_station_id=CYYZ_STN_ID,
        coordinates={"lat": 43.6777, "lon": -79.6248},
        year=year,
        variables_meta={
            "t2m": {"description": "2 m air temperature", "units": "degC"},
            "wind_speed": {"description": "10 m wind speed", "units": "km/h"},
        },
        samples=samples,
    )
    out = station_dir / f"observations_6h_{year}.json"
    write_json(out, doc)
    n_ok = sum(1 for o in doc["observations"] if o["t2m"] is not None)
    print(f"[cyyz] Wrote {out} ({n_ok}/{len(doc['observations'])} times with t2m)")


def process_soulis(year: int) -> None:
    station_dir = OBS_ROOT / "eric_d_soulis"
    raw_dir = station_dir / "raw"

    use_main = year <= 2014
    if use_main:
        url = SOULIS_MAIN_URL.format(year=year)
        raw_path = raw_dir / f"uw_main_15min_{year}.csv"
        source = "uw_soulis_main_15min"
        notes = {
            "time_interpretation": (
                "Archive clock is local standard time (UTC-5, no DST); "
                "converted to UTC for valid_time."
            ),
            "temperature": "Converted from Fahrenheit (Ambient Air Temperature).",
            "wind_speed": "Main logger wind speed column; units as archive (typically km/h).",
        }
    else:
        url = SOULIS_HOBO_URL.format(year=year)
        raw_path = raw_dir / f"uw_hobo_15min_{year}.csv"
        source = "uw_soulis_hobo_15min"
        notes = {
            "time_interpretation": (
                "Archive clock is local standard time (UTC-5, no DST); "
                "converted to UTC for valid_time."
            ),
            "temperature": "HOBO Temperature column (°C).",
            "wind_speed": "Wind Speed - Average 4.4 m (km/h); not 10 m — height differs from benchmark label.",
            "data_limitation": "Public bulk CSV for 2015+ is HOBO subset only; full main-logger yearly files stop at 2014.",
        }

    print(f"[eric_d_soulis] Downloading {url}…")
    try:
        download(url, raw_path)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"[eric_d_soulis] Download failed ({e.code}): {url}") from e
    print(f"[eric_d_soulis] Saved raw → {raw_path}")

    if use_main:
        samples = load_soulis_main_csv(raw_path, year)
    else:
        samples = load_soulis_hobo_csv(raw_path)

    doc = build_observations_json(
        station_id="eric_d_soulis",
        station_name="Eric D. Soulis station",
        source=source,
        source_station_id=None,
        coordinates={"lat": 43.4668, "lon": -80.5164},
        year=year,
        variables_meta={
            "t2m": {"description": "2 m air temperature (near-surface)", "units": "degC"},
            "wind_speed": {
                "description": "Wind speed (see processing_notes for sensor height)",
                "units": "km/h",
            },
        },
        samples=samples,
        extra_meta=notes,
    )
    out = station_dir / f"observations_6h_{year}.json"
    write_json(out, doc)
    n_ok = sum(1 for o in doc["observations"] if o["t2m"] is not None)
    print(f"[eric_d_soulis] Wrote {out} ({n_ok}/{len(doc['observations'])} times with t2m)")


def write_stations_registry() -> None:
    registry = {
        "schema_version": "1",
        "stations": [
            {
                "id": "cyyz",
                "name": "Toronto Pearson (CYYZ)",
                "coordinates": {"lat": 43.6777, "lon": -79.6248},
                "eccc_stn_id": CYYZ_STN_ID,
                "observations_glob": "cyyz/observations_6h_*.json",
            },
            {
                "id": "eric_d_soulis",
                "name": "Eric D. Soulis station",
                "coordinates": {"lat": 43.4668, "lon": -80.5164},
                "observations_glob": "eric_d_soulis/observations_6h_*.json",
            },
        ],
    }
    write_json(OBS_ROOT / "stations.json", registry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    OBS_ROOT.mkdir(parents=True, exist_ok=True)
    process_cyyz(args.year)
    process_soulis(args.year)
    write_stations_registry()
    print("Done.")


if __name__ == "__main__":
    main()

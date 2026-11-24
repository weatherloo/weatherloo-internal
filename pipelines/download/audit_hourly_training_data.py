#!/usr/bin/env python3
"""Inventory local HRRR cycles and hourly station observations without downloading data."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import timedelta, timezone
from pathlib import Path

from fetch_station_observations import eccc_to_samples, load_soulis_hobo_csv


REPO_ROOT = Path(__file__).resolve().parents[2]
OBS_ROOT = REPO_ROOT / "benchmarking-site" / "data" / "observations"
DATA_ROOT = Path(os.environ.get("WEATHERLOO_DATA_ROOT", REPO_ROOT / "data"))
HRRR_NAME = re.compile(r"hrrr_(\d{8})_t(\d{2})z\.nc$")


def hourly_observations(samples, year: int) -> dict:
    """Choose the closest observation within 15 minutes of each UTC hour."""
    hours = {}
    for timestamp, values in samples:
        timestamp = timestamp.astimezone(timezone.utc)
        hour = timestamp.replace(minute=0, second=0, microsecond=0)
        if timestamp - hour > timedelta(minutes=30):
            hour += timedelta(hours=1)
        delta = abs((timestamp - hour).total_seconds())
        if hour.year != year or delta > 15 * 60:
            continue
        previous = hours.get(hour)
        if previous is None or delta < previous[0]:
            hours[hour] = (delta, values)
    return {
        "temperature_hours": sum(v[1]["t2m"] is not None for v in hours.values()),
        "wind_hours": sum(v[1]["wind_speed"] is not None for v in hours.values()),
        "both_hours": sum(
            v[1]["t2m"] is not None and v[1]["wind_speed"] is not None
            for v in hours.values()
        ),
    }


def audit_year(year: int, hrrr_root: Path, obs_root: Path) -> dict:
    cycle_counts = {f"{hour:02d}Z": 0 for hour in (0, 6, 12, 18)}
    for path in (hrrr_root / str(year)).glob("*/hrrr_*_t*z.nc"):
        match = HRRR_NAME.fullmatch(path.name)
        if match and int(match.group(1)[:4]) == year:
            key = f"{match.group(2)}Z"
            if key in cycle_counts:
                cycle_counts[key] += 1

    stations = {}
    sources = {
        "eric_d_soulis": obs_root / "eric_d_soulis" / "raw" / f"uw_hobo_15min_{year}.csv",
        "cyyz": obs_root / "cyyz" / "raw" / f"eccc_hourly_{year}.json",
    }
    for station, path in sources.items():
        if not path.is_file():
            stations[station] = {"source_present": False}
            continue
        if station == "eric_d_soulis":
            samples = load_soulis_hobo_csv(path)
        else:
            samples = eccc_to_samples(json.loads(path.read_text())["features"])
        stations[station] = {"source_present": True, **hourly_observations(samples, year)}
    return {"year": year, "hrrr_cycles": cycle_counts, "stations": stations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--hrrr-root", type=Path, default=DATA_ROOT / "raw" / "hrrr")
    parser.add_argument("--obs-root", type=Path, default=OBS_ROOT)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("start year must be at most end year")
    print(json.dumps({
        "hrrr_root": str(args.hrrr_root),
        "note": "HRRR counts are files, not verified lead/variable coverage or forecast availability.",
        "years": [
            audit_year(year, args.hrrr_root, args.obs_root)
            for year in range(args.start_year, args.end_year + 1)
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

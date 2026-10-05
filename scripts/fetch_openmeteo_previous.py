#!/usr/bin/env python3
"""Fetch archived forecasts from Open-Meteo's Previous Runs API.

This is the "forecast you would have seen" reference series for the
retrospective comparison site. Unlike the benchmark files under
``benchmarking-site/data/<method>/``, which store *error metrics* against the
verifying observation, this stores the forecast **values** directly, so the
site can show a reference forecast without reconstructing it from the truth it
is meant to be compared against.

Lead-time mapping
-----------------
The Previous Runs API exposes forecasts at fixed offsets from valid time:
``*_previous_day1`` is the value predicted 24 h before valid time,
``_previous_day2`` 48 h before, and so on. Only whole-day offsets exist, so of
the project's [6, 12, ..., 72] lead grid only 24 h, 48 h and 72 h can be
sourced here; other leads raise rather than silently snapping to a
neighbouring offset.

Units are requested explicitly as degC / km/h to match
``data/observations/<station>/observations_6h_*.json``.

Caveat carried from the station metadata: Open-Meteo reports wind at 10 m,
while the Eric D. Soulis HOBO archive reports ~4.4 m. Those are not the same
quantity; the height is recorded in the output metadata rather than corrected.

Usage:
    python scripts/fetch_openmeteo_previous.py --lead 48 \
        --start 2026-01-01 --end 2026-07-25
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "benchmarking-site" / "data" / "openmeteo_previous"

API = "https://api.open-meteo.com/v1/forecast"

STATIONS = {
    "eric_d_soulis": {"lat": 43.4668, "lon": -80.5164, "name": "Eric D. Soulis station"},
    "cyyz": {"lat": 43.6777, "lon": -79.6248, "name": "Toronto Pearson (CYYZ)"},
}

TARGET_HOURS_UTC = (0, 6, 12, 18)
SUPPORTED_LEADS = (24, 48, 72)


def previous_day_suffix(lead_hours: int) -> str:
    if lead_hours not in SUPPORTED_LEADS:
        raise SystemExit(
            f"lead {lead_hours}h unavailable: the Previous Runs API only exposes whole-day "
            f"offsets, so this source supports {SUPPORTED_LEADS} "
            f"(use the in-repo NWP benchmarks for sub-daily leads)"
        )
    return f"_previous_day{lead_hours // 24}"


def fetch(lat: float, lon: float, start: str, end: str, lead_hours: int) -> dict:
    suffix = previous_day_suffix(lead_hours)
    hourly = [f"temperature_2m{suffix}", f"wind_speed_10m{suffix}"]
    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(hourly),
            "start_date": start,
            "end_date": end,
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        }
    )
    url = f"{API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "weatherloo-internal/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def to_records(payload: dict, lead_hours: int) -> list[dict]:
    """Keep only the 6-hourly UTC grid the rest of the project verifies on."""
    suffix = previous_day_suffix(lead_hours)
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get(f"temperature_2m{suffix}") or []
    winds = hourly.get(f"wind_speed_10m{suffix}") or []
    if not times:
        raise SystemExit(
            "no 'hourly.time' in response -- check that the requested date range is "
            "inside the archive (most models start 2024-01-01)"
        )

    records = []
    for i, ts in enumerate(times):
        dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        if dt.hour not in TARGET_HOURS_UTC:
            continue
        t2m = temps[i] if i < len(temps) else None
        ws = winds[i] if i < len(winds) else None
        init = dt - timedelta(hours=lead_hours)
        records.append(
            {
                "valid_time": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "initialization": init.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lead_hours": lead_hours,
                "t2m": t2m,
                "wind_speed": ws,
            }
        )
    return records


def build_doc(station_id: str, lead_hours: int, records: list[dict], payload: dict) -> dict:
    meta = STATIONS[station_id]
    return {
        "schema_version": "1",
        "source": "open_meteo_previous_runs",
        "source_url": API,
        "station_id": station_id,
        "station_name": meta["name"],
        "coordinates": {"lat": meta["lat"], "lon": meta["lon"]},
        "lead_hours": lead_hours,
        "cadence_hours": 6,
        "time_standard": "UTC",
        "variables": {
            "t2m": {"description": "2 m air temperature", "units": "degC"},
            "wind_speed": {
                "description": "10 m wind speed",
                "units": "km/h",
                "note": (
                    "10 m, whereas the eric_d_soulis observation archive reports ~4.4 m; "
                    "not height-adjusted"
                ),
            },
        },
        "model": payload.get("model") or payload.get("models"),
        "processing_notes": {
            "lead_mapping": (
                f"_previous_day{lead_hours // 24}: value predicted {lead_hours} h before "
                "valid time. Anchored to lead offset, not to a named init cycle, so the "
                "issuing run is approximately -- not exactly -- the stated initialization."
            ),
            "grid": "hourly response filtered to 00/06/12/18 UTC",
        },
        "records": records,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lead", type=int, default=48, help=f"lead hours {SUPPORTED_LEADS}")
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--station", default=None, help="station id (default: all)")
    args = p.parse_args()

    stations = [args.station] if args.station else list(STATIONS)
    for sid in stations:
        if sid not in STATIONS:
            raise SystemExit(f"unknown station {sid!r}; known: {list(STATIONS)}")
        meta = STATIONS[sid]
        print(f"[{sid}] fetching lead={args.lead}h {args.start}..{args.end}")
        payload = fetch(meta["lat"], meta["lon"], args.start, args.end, args.lead)
        records = to_records(payload, args.lead)
        doc = build_doc(sid, args.lead, records, payload)

        out = OUT_ROOT / sid / f"previous_{args.lead}h_{args.start}_{args.end}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
        n_ok = sum(1 for r in records if r["t2m"] is not None)
        print(f"[{sid}] wrote {out} ({n_ok}/{len(records)} slots with t2m)")


if __name__ == "__main__":
    main()

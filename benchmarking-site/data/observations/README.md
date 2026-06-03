# Station observations (ground truth)

Canonical **observed** data at benchmark locations. Kept separate from `data/<method>/` benchmark outputs.

## Layout

```
observations/
  README.md                 # this file
  stations.json             # registry: ids, coords, upstream source ids
  <station_id>/
    meta.json               # copy of station metadata + processing notes
    raw/                    # untouched upstream files (reproducibility)
    observations_6h_<year>.json   # normalized 6-hourly series (what pipelines read)
```

## Canonical file: `observations_6h_<year>.json`

One file per station per year. **All `valid_time` values are UTC** (`…Z`), on the same 6-hourly grid as benchmark initializations (**00, 06, 12, 18 UTC**) and lead-time verification at **+6h, +12h, … +72h UTC**.

```json
{
  "schema_version": "1",
  "station_id": "cyyz",
  "station_name": "Toronto Pearson (CYYZ)",
  "source": "eccc_climate_hourly",
  "source_station_id": 51459,
  "coordinates": { "lat": 43.6777, "lon": -79.6248 },
  "period": { "start": "2025-01-01T00:00:00Z", "end": "2025-12-31T18:00:00Z" },
  "cadence_hours": 6,
  "time_standard": "UTC",
  "variables": {
    "t2m": { "description": "2 m air temperature", "units": "degC" },
    "wind_speed": { "description": "10 m wind speed", "units": "km/h" }
  },
  "observations": [
    {
      "valid_time": "2025-01-01T00:00:00Z",
      "t2m": -7.9,
      "wind_speed": 14.0,
      "flags": { "t2m": null, "wind_speed": null }
    }
  ]
}
```

### Time standard (UTC everywhere)

| Layer | UTC handling |
|-------|----------------|
| **CYYZ** | ECCC `UTC_DATE` for timing; downloads use **`UTC_YEAR`** (not `LOCAL_YEAR`) so 00Z rows on Jan 1 are included. |
| **Soulis** | Archive labels are **local standard time (UTC−5, no DST)** per UW; converted to UTC before resampling. |
| **Output grid** | `00:00`, `06:00`, `12:00`, `18:00` **UTC** every day. |
| **Benchmark epic** | Initializations at **00, 06, 12, 18 UTC** every day; lead times in hours UTC. |

A slot can still be `null` when no observation exists near that **UTC** instant (e.g. ECCC may not report exactly at `00:00 UTC` for every calendar day). That is a data gap in UTC, not local-time mixing.

### Resampling rules (6-hourly UTC)

1. Observation at **exact** UTC target instant, if present.
2. Else closest observation in the **same UTC hour**.
3. Else **nearest within ±45 minutes** (UTC).
4. Else `null` with flag `"missing"`.

### Station sources

| `station_id` | Source | Notes |
|--------------|--------|--------|
| `cyyz` | [MSC GeoMet `climate-hourly`](https://api.weather.gc.ca/collections/climate-hourly/items) | `STN_ID` 51459 (TORONTO INTL A). `TEMP` °C, `WIND_SPEED` km/h. |
| `eric_d_soulis` | [UW Soulis archive](https://www.civil.uwaterloo.ca/weatherstation/data-archives/) | University station (not ECCC). 1998–2014: full `*_weather_station_data.csv` (15 min, temp °F → °C). 2015+: public bulk is `Hobo_15minutedata_*.csv` only (see `meta.json`). |

## Regenerate

```bash
python3 scripts/fetch_station_observations.py --year 2025
```

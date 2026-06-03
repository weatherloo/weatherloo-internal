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

One file per station per year. Timestamps align with benchmark initializations (00Z) and lead-time verification (6h steps through 72h).

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

### Resampling rules (6-hourly)

- Target valid times: every **6 hours UTC** (`…T00:00:00Z`, `06:00`, `12:00`, `18:00`).
- Upstream is **hourly** (ECCC) or **15-minute** (UW Soulis archive).
- For each target time, use the **nearest** observation within ±45 minutes; if none, value is `null` and flag `"missing"`.

### Station sources

| `station_id` | Source | Notes |
|--------------|--------|--------|
| `cyyz` | [MSC GeoMet `climate-hourly`](https://api.weather.gc.ca/collections/climate-hourly/items) | `STN_ID` 51459 (TORONTO INTL A). `TEMP` °C, `WIND_SPEED` km/h. |
| `eric_d_soulis` | [UW Soulis archive](https://www.civil.uwaterloo.ca/weatherstation/data-archives/) | University station (not ECCC). 1998–2014: full `*_weather_station_data.csv` (15 min, temp °F → °C). 2015+: public bulk is `Hobo_15minutedata_*.csv` only (see `meta.json`). |

## Regenerate

```bash
python3 scripts/fetch_station_observations.py --year 2025
```

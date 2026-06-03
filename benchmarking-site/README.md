# Forecast benchmark site (local)

Barebones internal dashboard for [issue #1](https://github.com/weatherloo/weatherloo-internal/issues/1): compare deterministic forecast skill at **Eric D. Soulis** and **Toronto Pearson (CYYZ)** for **t2m** and **10 m wind speed** across lead times.

**Implementing a benchmark method?** Read [`AGENTS.md`](AGENTS.md) first.

## Run locally

From this directory:

```bash
python3 -m http.server 8080
```

Open http://localhost:8080 — click a station on the **Southern Ontario** map (OpenLayers + CARTO dark basemap), pick a method, view RMSE / MAE / bias / ACC vs lead time.

## Data layout

### Ground truth (station observations)

See [`data/observations/README.md`](data/observations/README.md). Normalized 6-hourly `t2m` and `wind_speed` per station under `data/observations/<station_id>/observations_6h_<year>.json`.

Regenerate with:

```bash
python3 scripts/fetch_station_observations.py --year 2025
```

### Benchmark method outputs

One JSON file per **method** + **initialization** (365 files per method in production). Shape matches the epic:

```json
{
  "method": "gfs_interpolated",
  "initialization": "2025-01-15T00:00:00Z",
  "locations": {
    "cyyz": {
      "lat": 43.6777,
      "lon": -79.6248,
      "variables": {
        "t2m": {
          "lead_times_hours": [6, 12, ...],
          "rmse": [...],
          "mae": [...],
          "bias": [...],
          "acc": [...]
        },
        "wind_speed": { ... }
      }
    },
    "eric_d_soulis": { ... }
  }
}
```

Place files under `data/<method_id>/`. Optional `data/<method_id>/index.json` lists filenames to load:

```json
{ "files": ["2025-01-01.json", "2025-01-02.json", "..."] }
```

Without `index.json`, the app tries `<method_id>_sample.json`. The UI **averages** metrics across all loaded runs for the selected location (placeholder: single file).

## Placeholders

- `data/gfs_interpolated/gfs_interpolated_sample.json`
- `data/climatology/climatology_sample.json`

Add other methods under `data/<method_id>/` when benchmark pipelines produce output.

# Forecast Showdown

A consumer-weather-app front end for one retrospective question:

> For the **2026-07-20 00Z** run at the **Eric D. Soulis** station, which 48-hour
> temperature forecast was closest to what actually happened?

Six forecasts are compared against the station's own observations: the three
Weatherloo ML methods, and the three raw NWP models they post-process.

## Result for this run

| # | Method | Group | RMSE | MAE | Source |
|---|--------|-------|------|-----|--------|
| 1 | **LSTM** | Weatherloo | **0.99 °C** | 0.86 | ECMWF AIFS |
| 2 | ECMWF AIFS | baseline | 1.79 | 1.36 | — |
| 3 | U-Net | Weatherloo | 1.85 | 1.41 | GFS 0.25° |
| 4 | HRRR | baseline | 2.01 | 1.51 | — |
| 5 | CNN-LSTM | Weatherloo | 2.24 | 1.77 | HRRR 3 km |
| 6 | GFS | baseline | 4.89 | 4.05 | — |

Scored on leads +6 h … +48 h — the leads every method produced a value for, so
the hourly CNN-LSTM is not graded on a different set of points than the
6-hourly models.

Two honest findings worth keeping in view:

- The **LSTM roughly halves** raw ECMWF AIFS error on this run, and is the only
  method that clearly beats its own input.
- The **CNN-LSTM makes HRRR slightly worse** (2.24 vs 2.01). Its predicted bias
  is near-constant at about −0.36 °C across all 49 leads, which matches its
  training history: `val_mae` sat at ~2.82 °C over 7 early-stopped epochs, i.e.
  it learned little beyond the mean bias. That is the model as trained, not a
  bug in the inference path.

## Running it

```bash
npm install
npm run dev      # or: npm run build && npm run preview
```

The app reads a single file, `public/data/showdown.json`, produced by the
pipeline below. Nothing else is fetched at runtime.

## Regenerating the data

From the repository root:

```bash
# ground truth (hourly, from the raw 15-min HOBO archive) + raw NWP baselines
python scripts/showdown/build_truth_and_baselines.py --init 2026-07-20T00Z

# the three Weatherloo methods
python scripts/showdown/run_lstm.py      --init 2026-07-20T00Z --method ecmwf_aifs
python scripts/showdown/run_unet.py      --init 2026-07-20T00Z
python scripts/showdown/run_cnn_lstm.py  --init 2026-07-20T00Z

# merge into the document the site consumes
python scripts/showdown/combine.py --init 2026-07-20T00Z
cp benchmarking-site/data/showdown/showdown_20260720T00Z.json \
   forecast-showdown/public/data/showdown.json
```

`run_unet.py` and `run_cnn_lstm.py` pull GRIB byte ranges from the NOAA AWS
Open Data buckets and cache them under `.cache/` (gitignored). Requires
`torch`, `tensorflow-cpu`, `cfgrib`/`eccodes`, `xarray`, `scipy`.

## Where each number comes from

| Series | Provenance |
|---|---|
| Ground truth | `data/observations/eric_d_soulis/raw/*_15min_2026.csv`, resampled to the hour (±30 min, no interpolation). Archive clock is local standard time (UTC−5, no DST) and is shifted to UTC. |
| LSTM | `lstm_training/infer_recent.py` with the per-lead checkpoints in `lstm_training/retrain_output/eric_d_soulis_t2m/ecmwf_aifs_<lead>h/`. |
| U-Net | `models/unet/checkpoints/best_model.pt` over a GFS 0.25° southern-Ontario grid; corrected = GFS − predicted residual, bilinear to the station. |
| CNN-LSTM | `artifacts/hrrr_bias_correction/final_model.keras` over a 30×30 HRRR crop in native units; corrected = HRRR − predicted bias. |
| Raw baselines | `benchmarking-site/data/<method>/2026-07-20T00Z.json`. Those files store *error*, not forecast value, so each forecast is reconstructed as `obs_6h + bias` — the identity `scripts/build_hindcast.py` already relies on. |

## Open-Meteo is missing, and why

The brief asked for Open-Meteo as the public-forecast comparison. It is **not**
in this build: `api.open-meteo.com`, `previous-runs-api.open-meteo.com` and
`archive-api.open-meteo.com` are all refused by this environment's egress
policy (403 on CONNECT), and no Open-Meteo response is cached in the repo.

The raw NWP baselines stand in for it — they are the same models Open-Meteo
serves for southern Ontario, and they are real data for this init. To add the
genuine series: allowlist `api.open-meteo.com`, run

```bash
python scripts/fetch_openmeteo_previous.py --lead 24 --start 2026-07-20 --end 2026-07-22
python scripts/fetch_openmeteo_previous.py --lead 48 --start 2026-07-20 --end 2026-07-22
```

then re-run `combine.py`. The site already renders an "Unavailable" entry for
it, driven by the `unavailable` block in the JSON, so it will pick the series
up without a code change.

## Design

The shell follows the reference weather-app video: fixed left rail, a pill
selector in the top bar, a centred ~620 px card column, oversized hero number,
soft tinted background, and a bottom-sheet detail drawer. The pill selects
which method the whole page describes; tapping any hour opens the drawer with
every method ranked for that hour.

Method colours are categorical slots 1–3 of the validated palette
(blue / orange / aqua), which clear the all-pairs CVD and normal-vision floors
in both light and dark. The raw baselines deliberately use recessive neutrals
so they read as the comparison set rather than as more Weatherloo models.
Ground truth is ink, not a fourth hue — it is the reference, not a category.

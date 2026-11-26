# HyperDS on issued NWP forecasts (HRRR, AIFS)

The ERA5-initialized temporal models forecast 48 hours from one coarse analysis
field with no dynamics. On 2024 Soulis validation they scored 4.7–4.9°C
temperature RMSE, against 1.86°C for raw HRRR (see MULTIYEAR.md). This
variant gives the encoder the issued forecast instead, so each lead becomes the
paper's same-valid-time downscaling task.

For initialization T0 and lead h:

```text
NWP fields at anchors 0,6,...,48 h ─┐
two pre-T0 GOES frames ─────────────┴─> author encoders + fusion -> anchor contexts
hour h (any of 0..48): context = time-linear blend of the two bracketing anchors
query (lat, lon) + lead h + valid hour-of-day/day-of-year + (obs − NWP) at T0 per station
    -> correction added to the NWP value at hour h (native grid at stations)
```

Time is treated like space. The paper encodes a field once and queries the
generated decoder at sub-grid points. Here the encoder runs at the 6-hourly
anchors, and the decoder is queried at any hour in between. The value it corrects at
hour h is the forecast at h: exact for hourly HRRR, and for 6-hourly AIFS linear
in time, with precipitation taken as the stored rate for the interval that contains h. Both ERA5 and the
station labels are hourly, so AIFS output is hourly and supervised at every
hour. Each training step scores 12 random hours from 1–48, the time version of
the paper's random sub-grid cells. Validation scores every hour against the
raw forecast with the same interpolation (`issued_lead_metrics` and
`interpolated_hour_metrics` are reported separately).

- `nwp`: anchor contexts are encoded independently (direct).
- `nwp_autoregressive`: each anchor's predicted correction field (8×16,
  upsampled) is added to the next anchor's encoder input. Bias carries over from
  one 6-hour step to the next; the weather state itself is never fed back.
- Labels: ERA5 at the valid time (grid MSE) + 1.0 × station MSE on the v3
  Soulis overlay and ECCC stations (the paper weights station loss 0.05; station
  truth is the objective here). Station errors are divided by the raw forecast's
  own RMSE for each variable, measured on the training split, so a typical
  forecast error weighs the same for wind, temperature, pressure and precipitation.
  The checkpoint is selected by the Soulis skill ratio (model MSE ÷ raw-forecast
  MSE on the same pairs), averaged over wind and temperature. Below 1 means it
  beats the forecast. `best_wind_speed.pt` and `best_t2m.pt` keep the best epoch
  for each variable.
- Warm start: the encoders and fusion come from the ERA5 `paper` checkpoint.
  Each decoder's output layer and the new time/observation projections start at
  zero, so epoch 0 reproduces the raw NWP forecast exactly.
- Run `r1` (stopped after 2–3 epochs) skipped the zero output layer and used the
  0.05 station weight. It inherited the ERA5 model's coarse-to-fine residual and
  made Soulis temperature worse than raw AIFS (1.31 vs 1.17°C RMSE, 2024-11/12).
  `r2` added both fixes and was superseded by `r3` before training (hourly
  queries; L4/A10 GPUs at about a third of the A100 cost).
- In `r3` (stopped at epochs 5–6), the station loss was scaled by climatological
  spread, as in the paper. That made a 1°C miss count about 25x less than a
  2 m/s wind miss. Soulis wind beat AIFS (1.67 vs 2.49 m/s MAE) while
  temperature got worse (1.03 vs 0.88°C), and selection followed wind. `r4`
  scaled the loss by forecast error. The pooled scale still let Soulis's large
  sensor-height wind error dominate selection (scaled wind 2.6 vs t2m 0.66), so
  `r5` selects by skill ratio.
- Issue-time inputs: station observations at T0 (lead 0) are used as the
  observation-minus-forecast feature. HRRR 48 h runs publish about 1–2 h after
  T0, and AIFS later than that, so T0 observations come before issue time.
  Lead 0 is excluded from all forecast metrics.

| Source | Catalog | Leads | Train | Val | Test |
| --- | --- | --- | --- | --- | --- |
| HRRR v4 | `noaa-hrrr-forecast-48-hour` | 0–48 h hourly | 2020-12-03 → 2023 | 2024 | 2025 |
| AIFS Single | `ecmwf-aifs-single-forecast` (from 2024-04-01) | 0–48 h, 6-hourly | 2024-04 → 2024-10 | 2024-11/12 | 2025 |

HRRRv3 (archive start 2018-07-13 to 2020-12-02) issued only 36 h forecasts at 00/06/12/18Z, so HRRR training starts with HRRRv4. That also keeps training on the same version evaluated in 2024/2025. AIFS has no archive before April 2024, so it trains on about seven months. Training on
2026 would use data from after the 2025 test year and is not done.

HRRR is averaged from native 3 km points into each 0.25° ERA5 cell (57–68
points per cell). AIFS is native 0.25° and uses ERA5's grid points directly.
Station values are bilinear on each model's native grid. The HRRR Soulis
values match the earlier 2024 validation extraction to within 1e-6.

Every evaluation reports raw-NWP MAE/RMSE on the same finite pairs as the model
(`station_metrics.nwp`, `by_lead.*.nwp_rmse`), so each epoch shows directly
whether the correction beats the forecast it starts from.

## Remote layout

```text
weatherloo-temporal-data:/processed/hyperds_nwp/{hrrr,aifs}_v1/shards/YYYY-MM/{fields,station,init_time,era5_row,split}.npy
weatherloo-temporal-checkpoints:/hyperds_nwp/{hrrr,aifs}_v1_r5/{nwp,nwp_autoregressive}/{latest,best,best_t2m,best_wind_speed}.pt, metrics.jsonl
```

```bash
modal deploy models/hyperds_full/modal_nwp.py
python -c 'import modal; print(modal.Function.from_name("weatherloo-hyperds-nwp","run").spawn("hrrr").object_id)'
```

`run(source)` ingests the source's months (resumable), audits the result, runs
`test_nwp.py`, then trains both modes on L4/A10 in five-epoch stages up to 30
epochs. The final stage scores the best checkpoint on 2025 and writes
`predictions.npz`, which follows the `site_scoring.py` schema and adds an
`nwp_forecast` array.

## Results (`r5`, 30 epochs, best checkpoint by 2024 validation; 2025 test)

Scores use the site's 2025 truth (`pipelines/benchmarks/site_scoring.py`), matched to each comparison
method, at leads 6–48 h. RMSE, temperature in °C and wind in km/h. Exports are
not yet published to `benchmarking-site/data/`.

| Method | Soulis t2m | Soulis wind | CYYZ t2m | CYYZ wind |
| --- | ---: | ---: | ---: | ---: |
| HyperDS-HRRR direct | 2.11 | **2.97** | 1.60 | 5.49 |
| HyperDS-HRRR autoregressive | 2.05 | 3.06 | 1.59 | 5.54 |
| HyperDS-AIFS direct | 1.96 | 3.58 | 1.48 | **5.46** |
| HyperDS-AIFS autoregressive | 1.93 | 4.69 | 1.35 | 6.20 |
| `ecmwf_aifs` | 1.95 | 10.05 | 1.43 | 7.05 |
| `lstm_ecmwf_aifs` | **1.73** | — | 1.20* | — |
| `hrrr_interpolated` | 2.28 | 10.82 | 1.98 | 7.28 |
| `cnn_lstm_bias_correction` | 2.23 | 5.63 | — | — |
| `linear_regression` | 2.71 | 3.25 | 1.76 | 6.11 |
| `graphcast` | 2.19 | 10.00 | 1.41 | 7.30 |

\* `lstm_ecmwf_aifs` covers only 2,858 matched CYYZ cases; on those cases,
HyperDS-AIFS autoregressive scores 1.26.

Against raw HRRR, both HRRR models reduce Soulis temperature MSE by 18–20% on
hourly 2025 test hours (RMSE 1.95–1.97 vs 2.18), and the gain grows with lead.
The AIFS models gain most at the hours between AIFS's 6-hourly outputs.
Temperature skill is limited by the base forecast. AIFS-based models beat
HRRR-based ones despite seven months of training data versus three years.
`lstm_ecmwf_aifs` remains the temperature bar.

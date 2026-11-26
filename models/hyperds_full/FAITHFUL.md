# Paper-faithful HyperDS on HRRR (plus time)

This replicates [arXiv:2401.11960](https://arxiv.org/abs/2401.11960) using the
pinned author modules. The one intended extension is time: the model forecasts
0–48 h from an HRRR initialization. Code: `faithful_train.py` (model),
`faithful_baselines.py` (UNet/EDSR), `stations.py` (labels and split) and
`modal_faithful.py` (remote jobs). Tests: `test_faithful.py`.

## Paper contract

| Paper (and author config) | Here |
| --- | --- |
| Coarse field average-pooled 4x (0.25° → 1°) | HRRR regridded to 0.25°, then average-pooled 4x to 1° |
| Five surface variables u10, v10, t2m, sp, tp1h | Same, from HRRR |
| Two satellite frames, 4 bands (Himawari) | Two GOES-East frames, the same four spectral roles |
| ResNet-18 dual encoders; satellite self-attention + field cross-attention; MULTI_VAR generated decoders | Same author modules at revision `1fe6d0b` |
| Decoder auxiliary value = input field interpolated at the query | Same: the 1° input field, interpolated (no native HRRR point values) |
| 10 random subgrid samples in every high-res cell, mean vs grid label; stations inserted into their cells | Same, over every 0.25° cell |
| Loss = grid MSE + 0.05 × station MSE, mean/variance-normalized | Same |
| Adam 1e-4, cosine, T_max 5, 50 epochs | Same |
| Batch 4 per GPU × 4 GPUs | Batch 4 × 4 gradient-accumulation steps (effective 16) |
| Random initialization | Same (no warm start, no zero-initialized outputs) |
| Station observations are labels only | Same: no observation inputs |
| Random station split 1266/200/400 plus a time split; test = unseen stations in the unseen period | Same proportions on 314 stations, after merging co-located IDs within 2 km (Soulis and Toronto Pearson forced into test). Time: train 2020-12-03→2023, val 2024, test 2025 |
| Checkpoint = lowest station loss on validation stations | Same, validation stations in 2024 |
| Metrics: station MSE/MAE | Same, per test-station group, plus Soulis and CYYZ |
| Baselines: interpolation of the 1° and 0.25° fields; UNet; EDSR | HRRR 1° and 0.25° interpolation; UNet; EDSR (satellite conv → pooled → concatenated) |

## Time extension

- **Anchor contexts:** the encoders run at 6-hourly anchors (0, 6, …, 48 h). Any
  hour h in 0–48 is decoded from a time-linear blend of the two bracketing
  anchor contexts.
- **Time inputs:** a lead-hour encoding, plus valid hour-of-day and day-of-year.
  These start at zero, as projections of the time features.
- **Sampled hours:** each step trains on 4 random hours.
- **`nwp_autoregressive`:** feeds each anchor's correction field forward to the
  next anchor's encoder input.
- **SR baselines:** UNet and EDSR treat each hour independently, with lead-hour
  and time-of-day/day-of-year channels.

## Differences that are not part of the time extension

- **Satellite timing:** the frames precede the initialization, because a
  forecast cannot use future images.
- **Region and data:** HRRR replaces ERA5 as input, and GOES replaces Himawari,
  for Ontario.
- **Validation grid score:** validation computes grid MSE on a fixed 128-cell
  subset as a diagnostic. Checkpoint selection uses the station loss, which
  covers every validation station and hour.
- **Small domain:** the 1° field is only 8×16 pixels, so the field encoder's
  deepest features are 1×1. Training drops incomplete batches because
  BatchNorm cannot train on one sample at that size.

## Station labels

- **Sources:** 73 ECCC climate-hourly stations and 273 IEM ASOS stations inside
  40–47.875°N, 88–72.125°W with records spanning 2020-12 to 2025-12, plus Soulis
  (v3 QC overlay). 327 have at least 50% hourly temperature and wind coverage. 13 duplicate ECCC IDs at the same site (within 2 km) are then merged, keeping the best-covered one, which leaves 314. Without this, one site (Wiarton A) appeared in both train and validation.
- **Variables:** hourly wind speed (m/s), t2m (°C), station pressure (hPa) and
  1-hour precipitation (mm).
- **ASOS processing:** routine METARs (:45–:59) are assigned to the next hour.
  Station pressure comes from the altimeter setting reduced to station
  elevation.
- **Location on Modal:** `weatherloo-temporal-data:/processed/hyperds_nwp/stations_v2/{labels.npz,stations.json,split.json}` (v1 is the undeduplicated source).

## Runs

- **Checkpoints and logs:** `weatherloo-temporal-checkpoints:/hyperds_faithful/hrrr_v2/{nwp,nwp_autoregressive,baseline_unet,baseline_edsr}/`.
- **Hardware and cost:** L4/A10 GPUs, peak 4.8 GiB. A probe measured about 13
  min per epoch for HyperDS.
- **Outputs:** each run writes `evaluation.json` and `predictions.npz` (all
  stations, 2025, hourly 0–48 h). CYYZ is aliased to `cyyz` for `benchmark.py`.

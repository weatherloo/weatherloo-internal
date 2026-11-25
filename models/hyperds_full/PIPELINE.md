# Region pipeline: HRRR station correction

This is the production path for site forecasts. It corrects the operational HRRRv4 48 h forecast at the configured stations with one model type: a CNN on the native HRRR patch around each station, plus that station's 0–48 h HRRR trajectory. The same model type forecasts temperature, wind and precipitation. HyperDS (see [FAITHFUL.md](FAITHFUL.md) and [NWP.md](NWP.md)) remains the research track for continuous fields. At Soulis, HyperDS is less accurate than this station model.

## Run

```bash
modal run models/hyperds_full/modal_region.py --region waterloo              # plan: validates config, lists pending stages
modal run models/hyperds_full/modal_region.py --region waterloo --execute    # runs only what is missing
modal run models/hyperds_full/modal_region.py --region waterloo --execute --force cnn
modal run models/hyperds_full/modal_region.py::rain                          # precipitation (targets with "precip": true)
```

The plan step runs the following checks before any compute is spent:

- Each target is in the station table.
- Each target lies in the anchor's native HRRR chunk.
- Which extraction months and model outputs already exist.

Each stage is idempotent. A month with a manifest is never re-read. The CNN stage reruns only with `--force cnn`. Changing extraction needs a new `version` in the config.

| Stage | Compute | Output (data Volume) |
| --- | --- | --- |
| `points` | CPU, one container per month | `/data/processed/points/hrrr_<anchor>_<version>/<YYYY-MM>/` |
| `patches` | CPU, one container per month and target | `/data/processed/patches/hrrr_<target>_context_<side>_b8_<version>/<YYYY-MM>/`: all 27 catalog fields on the patch, plus an 8×8 block-mean view of the whole native chunk |
| `cnn` | L4/A10 GPU, one per target, run concurrently | `/data/processed/regions/<name>/<target>/cnn.{json,npz}` |
| `final` | 2 CPU | `/data/processed/regions/<name>/<target>/{final.json,final.npz}` (2025 scores against raw HRRR, by lead) |
| `export` | 2 CPU | `/data/processed/regions/<name>/site_export/` (site npz, metadata, matched comparison against every site method) |
| `rain` | L4/A10 GPU | `/data/processed/regions/<name>/<target>/precip.{json,npz}` |

## Region config

`regions/<name>.json`, parsed by `regions.py`:

```json
{
  "name": "waterloo",
  "anchor": "eric_d_soulis",
  "method": "hrrr_cnn",
  "station_root": "/data/processed/hyperds_nwp/stations_v2",
  "k_neighbors": 30,
  "patch_half": 8,
  "version": "v1",
  "targets": [
    {"id": "eric_d_soulis", "site_id": "eric_d_soulis", "cnn_options": {"trajectory": true}, "precip": true},
    {"id": "eccc_51459", "site_id": "cyyz", "cnn_options": {"trajectory": true}}
  ]
}
```

- `anchor`: the station whose native HRRR chunk (265×300 points, about 800×900 km) is read once per initialization for the point features. Every target must lie inside that chunk.
- `cnn_options`: options passed to `cnn_correction.run_cnn`:
  - `trajectory`: the site's 0–48 h sequence
  - `context`: patch lead offsets
  - `wide`: the whole-chunk view
  - `variables`: patch fields; defaults to the 15 in `points.PATCH_VARIABLES`

  A `models` key other than `["cnn"]` is rejected.
- `precip`: also forecast precipitation. This needs a gauge record for the target; Soulis is wired up.
- `site_id`: the benchmarking-site station id. Targets without one are trained, but not exported to the site.
- `k_neighbors`: the stations nearest the target whose observations supply issue-time input features.

**New regions.** Add a `stations` spec to the config to build a region's table from IEM ASOS, with ECCC optional:

```json
"stations": {"bbox": [-124.6, 35.5, -118.5, 41.0], "asos_networks": ["CA", "NV"], "eccc": false}
```

- **What it needs:** a bbox of lon_min, lat_min, lon_max, lat_max. `asos_networks` are IEM state codes.
- **What `stations.build_region` does:** it keeps stations with at least 50% hourly temperature and wind coverage, and removes stations within 2 km of each other. Targets are always kept.
- **When it runs:** the `stations` stage builds the table on the first `--execute`, then plans the remaining stages.
- **Limitation:** the patch must lie inside the target's native HRRR chunk. `points.ContextReader` raises otherwise, and the plan catches it before compute.

## Method

- **Rows.** Each row is one initialization (00/06/12/18 UTC) and one lead from 1 to 48 h.
- **Target.** The model learns the observation minus the bilinearly interpolated HRRR point forecast, for 2 m temperature and 10 m wind speed.
- **Inputs.** Only information available at issue time is used:
  - a 17×17 native patch of 15 HRRR fields (about 50 km square), valid at the lead
  - the same 15 fields at the site for every lead 0–48 h (the trajectory), through a 1-D convolution with a marker on the predicted lead
  - about 60 scalars:
    - the run's 19 surface fields at the site
    - the three earlier runs at the same valid time, and their spread
    - observations and HRRR errors at or before initialization, at the target and its neighbours
    - clock descriptors

  `test_mos.py` checks the alignment of every lagged input.
- **Training.** Three seeds of the same network are trained and their outputs averaged, with early stopping on 2024. The split is initializations from 2020-12-03 through 2023 (targets before 2024) for training, 2024 for early stopping and the input ablations, and 2025 scored once.
- **Storage units.** Patches are stored as float16. Pressure is stored in hPa, and precipitation and snowfall in mm/h. In pascals and kg m⁻² s⁻¹ these fields overflow or underflow float16. The first patch extraction (`hrrr_<target>_17_v1`) has this bug; nothing in the pipeline reads it now.
- **Scoring.** The site export uses the site's canonical `observations_6h_2025.json` truth at 6-hourly leads. `comparison.json` matches pairs against every method's 2025 npz.

## Current results (2025, unseen)

Run on 2026-11-24 with the CNN at both sites; `final.json` holds the full by-lead scores. RMSE over hourly leads 1–48 h; temperature in °C, wind in km/h.

| Site | Variable | CNN | Raw HRRR | +1 h | +24 h | +48 h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Soulis | Temperature | **1.765** | 2.181 | 1.29 | 1.69 | 2.15 |
| Soulis | Wind | **2.729** | 10.862 | 2.47 | 2.63 | 3.00 |
| CYYZ | Temperature | **1.303** | 1.922 | 0.79 | 1.25 | 1.70 |
| CYYZ | Wind | **4.981** | 7.089 | 4.16 | 4.94 | 6.10 |

- **CYYZ before.** The gradient-boosting model used there earlier scored 1.369 °C and 5.209 km/h.

**Site-scored at 6-hourly leads, matched pairs (CNN vs method)**

| Method | Soulis temperature | Soulis wind | CYYZ temperature | CYYZ wind |
| --- | --- | --- | --- | --- |
| hrrr_interpolated | 1.77 vs 2.28 | 2.70 vs 10.82 | 1.34 vs 1.98 | 5.18 vs 7.28 |
| ecmwf_aifs | 1.77 vs 1.95 | 2.70 vs 10.05 | 1.34 vs 1.43 | 5.18 vs 7.05 |
| graphcast | 1.77 vs 2.19 | 2.70 vs 10.00 | 1.34 vs 1.41 | 5.18 vs 7.30 |
| lstm_ecmwf_aifs (in-sample, see below) | 1.77 vs 1.73 | n/a | 1.11 vs 1.21 | n/a |

**Day 2 against raw AIFS (temperature, CNN vs AIFS)**

| Site | +6 h | +24 h | +36 h | +42 h | +48 h |
| --- | --- | --- | --- | --- | --- |
| Soulis | 1.43 vs 1.78 | 1.68 vs 1.93 | 1.87 vs 2.00 | 1.98 vs 2.06 | 2.15 vs 2.11 |
| CYYZ | 1.05 vs 1.38 | 1.24 vs 1.41 | 1.41 vs 1.45 | 1.51 vs 1.47 | 1.69 vs 1.48 |

The CNN leads through about 36 h at both sites and trails AIFS at 42–48 h at CYYZ.

**`lstm_ecmwf_aifs`.** This file covers only its own late-2025 window, and it includes data it was trained on.

## Generalization test: San Francisco (2026-11-25)

The same code was run with one new config, `regions/san_francisco.json`, at San Francisco International (`asos_SFO`). Nothing in the model was tuned for this site.

- **Station table.** It was built by the `stations` stage from IEM ASOS in northern California and western Nevada: 69 stations, 63 of them in SFO's native chunk. Waterloo had 176.
- **SFO record.** Coverage is 99.9% in every year. Dry hours are reported as 0, so precipitation is scored as at Soulis.
- **Compute.** Station build, then points, patches, the CNN, scoring and precipitation. All stages ran unattended once the run resumed after a Modal spend-limit stop. Completed stages were skipped on resume.

**Temperature and wind, 2025, hourly leads 1–48 h** (RMSE)

| Variable | CNN | Raw HRRR | CNN +1 / +24 / +48 h | Raw HRRR +1 / +24 / +48 h |
| --- | ---: | ---: | --- | --- |
| Temperature (°C) | **1.148** | 1.797 | 0.85 / 1.19 / 1.36 | 1.44 / 1.86 / 2.06 |
| Wind (km/h) | **6.017** | 9.229 | 5.25 / 6.00 / 6.57 | 9.74 / 8.93 / 9.31 |

The errors are 36% and 35% lower than raw HRRR. At Soulis the reductions were 19% (temperature) and 75% (wind, mostly the anemometer-height bias).

**Precipitation, 2025**

| Product | CNN | Climatology | HRRR point | HRRR within 50 km |
| --- | ---: | ---: | ---: | ---: |
| `pop1` Brier | **0.0219** | 0.0322 | 0.0379 | 0.0269 |
| `pop6` Brier | **0.0329** | 0.0670 | 0.0594 | 0.0444 |
| `amt6` RMSE mm | 1.708 | 2.016 | 1.657 | **1.584** |
| `amt24` RMSE mm | **3.38** | 5.16 | 3.69 | 3.50 |

- **Brier skill against climatology.**

  | Product | CNN | HRRR point | HRRR within 50 km |
  | --- | ---: | ---: | ---: |
  | `pop1` | +0.32 | −0.18 | +0.16 |
  | `pop6` | +0.51 | +0.11 | +0.34 |
- **6-hour amounts.** The CNN does not beat HRRR on 6-hour amounts at SFO: RMSE 1.71 mm against 1.58–1.66, with mean absolute error level at 0.31–0.33. SFO's rain is almost all November–March, frontal and well forecast by HRRR. 24-hour amounts are better (3.38 mm against 3.50–3.69).
- **Calibration.** The chances run high in the middle range: hourly 0.45 verified 0.33 and 0.65 verified 0.52, and 6-hour 0.46 verified 0.34. 2025 at SFO was drier than average (396 mm).

**Conclusion.** The pipeline generalizes without code changes beyond the region config: temperature, wind and the chances of precipitation improve on raw HRRR by margins similar to Waterloo. The weak spot is short-window rain amounts in a climate where HRRR already handles them.

## History: earlier pipeline runs

The sections below record the runs that led here, which used two model types (gradient boosting and the CNN) and a blend or a per-variable selection. That code remains in `mos.py` and `precip.fit_gbm`, but the pipeline no longer runs it.

### Results (2025, unseen)

First full run, 2026-11-24 (`/data/processed/regions/waterloo/`). Two fits were re-run in one pipeline run: gradient boosting at both sites and the CNN at Soulis. Both reproduced the earlier experiment numbers.

**Soulis, hourly leads 1–48 h, matched pairs (RMSE; temperature in °C, wind in km/h)**

| Model | Temperature | Wind |
| --- | ---: | ---: |
| Raw HRRR | 2.181 | 10.862 |
| Gradient boosting | 1.815 | 2.746 |
| CNN | 1.819 | 2.781 |
| Fixed 50/50 average | **1.785** | **2.724** |
| Validation-weighted blend (pipeline output) | 1.789 | 2.763 |

- **Weights:** the 2024 weights put 0.35 (temperature) and 0.10 (wind) on gradient boosting. The CNN was clearly better on 2024 (wind 2.43 vs 2.53 km/h), but it did not keep that lead in 2025.
- **Validation curve:** the 2024 curve is almost flat between weights 0 and 0.5 (wind 2.433–2.454). The fitted weight therefore cost 0.04 km/h relative to the fixed 50/50 average.
- **Pre-specified alternative:** the 50/50 average was specified before the first CNN test. Any rule adopted now, such as shrinking towards equal weights, would be chosen after seeing 2025.

**Site-scored, 6-hourly leads, matched against each method's 2025 file (pipeline output vs method)**

| Method | Soulis temperature | Soulis wind | CYYZ temperature | CYYZ wind |
| --- | --- | --- | --- | --- |
| hrrr_interpolated | 1.78 vs 2.28 | 2.74 vs 10.82 | 1.42 vs 1.98 | 5.41 vs 7.28 |
| ecmwf_aifs | 1.78 vs 1.95 | 2.74 vs 10.05 | 1.42 vs 1.43 | 5.42 vs 7.05 |
| graphcast | 1.78 vs 2.19 | 2.74 vs 10.00 | 1.42 vs 1.41 | 5.41 vs 7.30 |
| cnn_lstm_bias_correction | 1.79 vs 2.23 | 2.74 vs 5.63 | n/a | n/a |
| lstm_ecmwf_aifs (in-sample, see note) | 1.78 vs 1.73 | n/a | 1.19 vs 1.21 | n/a |

- **CYYZ:** CYYZ uses gradient boosting only.
- **`lstm_ecmwf_aifs`:** this file covers only its own late-2025 window. Its 2025 predictions include data it was trained on. On its held-out window, gradient boosting scored 2.18 vs 2.10 at Soulis and 1.11 vs 1.16 at CYYZ.
- **Column label:** `comparison.json` labels our column `hyperds`, a name inherited from the HyperDS scorer.

### CNN input ablations (2026-11-24)

All runs use the corrected extraction. Every run is chosen on 2024 validation; 2025 is shown for each run so the picture is complete. RMSE is temperature °C / wind km/h at Soulis.

| Run | 2024 validation | 2025 CNN | 2025 50/50 with gradient boosting |
| --- | --- | --- | --- |
| Earlier CNN (float16 bug) | 1.450 / 2.433 | 1.819 / 2.781 | 1.785 / 2.724 |
| a0 control: same inputs, units fixed | 1.463 / 2.439 | 1.820 / 2.788 | 1.786 / 2.721 |
| a1 + patch at h−3 and h+3 | 1.463 / 2.388 | 1.793 / 2.774 | 1.769 / 2.705 |
| **a2 + site trajectory 0–48 h** | 1.455 / 2.387 | **1.765 / 2.729** | **1.757 / 2.686** |
| a3 all 27 fields | 1.466 / 2.420 | 1.789 / 2.883 | 1.771 / 2.750 |
| a4 + whole-chunk view (~800 km, 25 km boxes) | 1.470 / 2.386 | 1.813 / 2.799 | 1.772 / 2.728 |
| a5 trajectory + time context | 1.461 / 2.383 | 1.776 / 2.766 | 1.760 / 2.701 |
| a6 trajectory + time context + chunk view | 1.451 / 2.376 | 1.787 / 2.788 | 1.759 / 2.710 |

- **What helped.** The trajectory was the only input that helped both variables on 2024. Time context and the wide view helped 2024 wind only. The 12 extra fields (snow depth, precipitation type, sea-level pressure, ceiling, reflectivity) did not help.
- **Unit fix.** Fixing the units changed nothing measurable. The scalar features already carry pressure and precipitation.
- **Choice of a2.** a6 scored best on 2024, but by 0.004 °C and 0.011 km/h, which is within seed noise. The default is a2, the simplest of the tied variants, which also needs no chunk view. The 2025 results also favour a2, so this choice is not blind to the test year.

**Pipeline output** (single model per variable, chosen on 2024; `cnn.npz` is the a2 run, identical configuration):

| Target | Variable | Model chosen | 2024 validation RMSE (gradient boosting / CNN) | 2025 RMSE | Raw HRRR |
| --- | --- | --- | --- | ---: | ---: |
| Soulis | Temperature (°C) | CNN | 1.481 / 1.455 | **1.765** | 2.181 |
| Soulis | Wind (km/h) | CNN | 2.533 / 2.387 | **2.729** | 10.862 |
| CYYZ | Temperature (°C) | Gradient boosting (only model) | n/a | **1.369** | 1.922 |
| CYYZ | Wind (km/h) | Gradient boosting (only model) | n/a | **5.209** | 7.089 |

- **Gradient boosting at Soulis on 2025.** It scores 1.815 °C and 2.746 km/h, so the 2024 choice of the CNN held up in 2025.
- **By lead.** Soulis temperature is 1.29 / 1.69 / 2.15 °C at +1 / +24 / +48 h, against raw HRRR's 1.72 / 2.21 / 2.61.
- **Site-scored at 6-hourly leads, matched pairs.**
  - Soulis temperature: 1.77 vs AIFS 1.95, GraphCast 2.19 and HRRR interpolation 2.28.
  - Soulis wind: 2.70 vs 10.0–10.8 for the same three.
  - CYYZ temperature: 1.42, level with AIFS 1.43 and GraphCast 1.41.
- **Earlier averaged variants.** The 50/50 and validation-weighted averages in the sections above are kept for the record. They are not used.

## Precipitation (`precip.py`)

```bash
modal run models/hyperds_full/modal_region.py::rain --check   # gauge coverage and timing only
modal run models/hyperds_full/modal_region.py::rain           # train, select on 2024, score 2025
```

Targets with `"precip": true` get four products for every run out to 48 h:

- `pop1`: the chance of at least 0.2 mm (one gauge tip) in each hour.
- `pop6`: the same chance for each 6-hour block (hours 1–6, …, 43–48).
- `amt6`: the expected amount in each block.
- `amt24`: the sum of four blocks.

**Truth**

- **Soulis.** The HOBO tipping-bucket gauge, in local standard time (UTC−5). Its 15-minute amounts are summed to UTC hour-ending totals, and an hour counts only when all four samples exist. The weighing gauge is unused because it is missing in 2024–2025. Labels start in 2021, because the 2020 file has a different layout.
- **Neighbours.** ECCC/ASOS hourly precipitation from the station labels, used only as extra training rows.
- **HRRR.** `precipitation_surface` at lead h is the mean rate over the hour ending at h, so each lead verifies against the hour ending at that time.
- **Timing check.** The check used training years only. Wet/dry correlation with HRRR at the site peaks at lag 0 (0.40, against 0.38 at −1 h). The nearest ECCC gauges peak at −1 h or 0 h.
- **Coverage.**

  | Year | Coverage | Total |
  | --- | --- | --- |
  | 2021 | 99% | 1,009 mm |
  | 2022 | 23% | 134 mm |
  | 2023 | 99% | 961 mm |
  | 2024 | 99% | 847 mm |
  | 2025 | 85% | 601 mm |

  2025 is missing most of August and half of December.

**Snow caveat.** The tipping bucket undercatches snow, and it can record snow late, when it melts. Every score below is measured against the gauge. November–March numbers therefore measure agreement with what the gauge records, not true precipitation. HRRR's poor winter scores partly reflect snow the gauge never saw. April–October is the reliable comparison.

**Model**

- **CNN.** Rows are 6-hour blocks.
  - **Inputs:** the patch at the block's six leads, the site's 0–48 h trajectory, and block-mean scalars. The scalars include:
    - HRRR precipitation at the neighbouring leads, and 3 h and 6 h centred sums
    - the three earlier runs, with their mean, maximum and wet fraction
    - the observed precipitation in the 1, 3 and 6 hours ending at initialization, and the 0–6 h error of the previous run
    - the fraction of the 17×17 and 9×9 patch that is wet
  - **Outputs:** six hourly wet logits, a block wet logit, and six hourly and one block log-mean amounts, trained with Tweedie deviance.
  - **Training:** three seeds, early stopping on 2024.
- **Leakage test.** `test_precip.py` checks that changing any label after T0 leaves every issue-time input unchanged.

**Results, Soulis 2025, matched pairs**

| Product | Season | CNN | Climatology | HRRR point | HRRR within 50 km |
| --- | --- | ---: | ---: | ---: | ---: |
| `pop1` Brier | All months | **0.0481** | 0.0653 | 0.1031 | 0.0785 |
| `pop1` Brier | Apr–Oct | **0.0488** | 0.0692 | 0.0792 | 0.0599 |
| `pop6` Brier | All months | **0.0930** | 0.1481 | 0.2152 | 0.1814 |
| `pop6` Brier | Apr–Oct | **0.0903** | 0.1581 | 0.1462 | 0.1180 |
| `amt6` RMSE mm | All months | **1.909** | 2.280 | 2.576 | 2.428 |
| `amt6` RMSE mm | Apr–Oct | **2.383** | 2.834 | 2.941 | 2.797 |
| `amt24` RMSE mm | All months | **4.03** | 5.82 | 5.38 | 5.23 |
| `amt24` RMSE mm | Apr–Oct | **4.94** | 7.14 | 5.64 | 5.46 |

- **Brier skill against climatology, all months.**

  | Product | CNN | HRRR point | HRRR within 50 km |
  | --- | ---: | ---: | ---: |
  | `pop1` | +0.26 | −0.58 | −0.20 |
  | `pop6` | +0.37 | −0.45 | −0.23 |

  Raw HRRR, used as a yes/no or as a neighbourhood fraction, is worse than climatology at the gauge.
- **Mean absolute error (mm).**

  | Product | CNN | HRRR point |
  | --- | ---: | ---: |
  | `amt6` | 0.55 | 0.72 |
  | `amt24` | 1.66 | 2.33 |
- **Reliability.**
  - The hourly chance is close to calibrated across its range; for example, 0.55 verified 0.51 and 0.65 verified 0.64.
  - The 6-hour chance runs high in the middle range in 2025: 0.45 verified 0.29, and 0.55 verified 0.37. 2025 was a drier year: 601 mm with gaps, against 850–1,010 mm in training years.
- **Earlier study.** The study that also fitted gradient boosting (`precip.run(..., with_gbm=True)`) scored slightly better on the two chances: 0.0466 and 0.0913. It is not used, because the pipeline uses one model type.
- **Site.** Not exported: the benchmarking site scores only temperature and wind.

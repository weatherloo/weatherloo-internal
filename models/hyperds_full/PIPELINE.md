# Region pipeline: HRRR station correction

This is the production path for site forecasts. It corrects the operational HRRRv4 48 h forecast at the configured stations. Two models are fitted: gradient boosting on native-grid point predictors, and a CNN on a native-grid patch. Their blend is weighted on validation data. HyperDS (see [FAITHFUL.md](FAITHFUL.md) and [NWP.md](NWP.md)) remains the research track for continuous fields. At Soulis, HyperDS is less accurate than these station models.

## Run

```bash
modal run models/hyperds_full/modal_region.py --region waterloo              # plan: validates config, lists pending stages
modal run models/hyperds_full/modal_region.py --region waterloo --execute    # runs only what is missing
modal run models/hyperds_full/modal_region.py --region waterloo --execute --force gbm,cnn
```

The plan step runs the following checks before any compute is spent:

- Each target is in the station table.
- Each target lies in the anchor's native HRRR chunk.
- Which extraction months and model outputs already exist.

Each stage is idempotent. A month with a manifest is never re-read. A model stage reruns only when it is named in `--force`. Changing extraction needs a new `version` in the config.

| Stage | Compute | Output (data Volume) |
| --- | --- | --- |
| `points` | CPU, one container per month | `/data/processed/points/hrrr_<anchor>_<version>/<YYYY-MM>/` |
| `patches` | CPU, one container per month and CNN target | `/data/processed/patches/hrrr_<target>_context_<side>_b8_<version>/<YYYY-MM>/`: all 27 catalog fields on the patch, plus an 8×8 block-mean view of the whole native chunk |
| `gbm` | 16 CPU | `/data/processed/regions/<name>/<target>/gbm.{json,npz}` |
| `cnn` | L4/A10 GPU | `/data/processed/regions/<name>/<target>/cnn.{json,npz}` |
| `blend` | 2 CPU | `/data/processed/regions/<name>/<target>/{blend.json,final.npz}` |
| `export` | 2 CPU | `/data/processed/regions/<name>/site_export/` (site npz, metadata, matched comparison against every site method) |

## Region config

`regions/<name>.json`, parsed by `regions.py`:

```json
{
  "name": "waterloo",
  "anchor": "eric_d_soulis",
  "method": "hrrr_gbm_cnn",
  "station_root": "/data/processed/hyperds_nwp/stations_v2",
  "k_neighbors": 30,
  "patch_half": 8,
  "version": "v1",
  "targets": [
    {"id": "eric_d_soulis", "site_id": "eric_d_soulis", "models": ["gbm", "cnn"], "cnn_options": {"trajectory": true}},
    {"id": "eccc_51459", "site_id": "cyyz", "models": ["gbm"]}
  ]
}
```

- `anchor`: the station whose native HRRR chunk (265×300 points, about 800×900 km) is read once per initialization. Every target must lie inside that chunk.
- `models`: `gbm` is required. Add `cnn` to also extract a `(2·patch_half+1)²` patch and train the CNN.
- `cnn_options`: options passed to `cnn_correction.run_cnn`:
  - `trajectory`: the site's 0–48 h sequence
  - `context`: patch lead offsets
  - `wide`: the whole-chunk view
  - `variables`: patch fields; defaults to the 15 in `points.PATCH_VARIABLES`
- `site_id`: the benchmarking-site station id. Targets without one are trained and blended, but not exported to the site.
- `k_neighbors`: the stations nearest the target whose observations supply issue-time features.

**New regions.** Any of the 314 stations in `stations_v2` can be a target today. That table covers roughly 40–48°N, 72–88°W. A region outside it first needs a station table built with `stations.py`, whose bounding box and ASOS networks are currently constants. Making those part of the region config is the next step for full automation.

## Method

- **Rows.** Each row is one initialization (00/06/12/18 UTC) and one lead from 1 to 48 h.
- **Target.** The model learns the observation minus the bilinearly interpolated HRRR point forecast, for 2 m temperature and 10 m wind speed.
- **Features.** Only information available at issue time is used:
  - the run itself (19 surface fields)
  - the three earlier runs at the same valid time, and their spread
  - observations and HRRR errors at or before initialization, at the target and its neighbours
  - clock and station descriptors

  `test_mos.py` checks the alignment of every lagged feature.
- **Split.** Training uses initializations from 2020-12-03 through 2023, with targets before 2024. 2024 is used for every choice: gradient-boosting configuration, early stopping, CNN checkpoint and blend weight. 2025 is scored once.
- **Gradient boosting.** LightGBM, configuration selected on 2024 from a small grid plus pooled and target-only variants, then refitted with five seeds and averaged.
- **CNN.** Inputs:
  - a 17×17 native patch of 15 fields (about 50 km square) valid at the lead
  - the gradient-boosting scalar features
  - with `trajectory`, the 15 fields at the site for every lead 0–48 h, run through a 1-D convolution with a marker on the predicted lead

  Three seeds, early stopping on 2024.
- **Storage units.** Patches are stored as float16. Pressure is stored in hPa, and precipitation and snowfall in mm/h. In pascals and kg m⁻² s⁻¹ these fields overflow or underflow float16. The first patch extraction (`hrrr_<target>_17_v1`) has this bug; nothing in the pipeline reads it now.
- **Blend.** One weight per variable, chosen on 2024 RMSE from a 0.05 grid. A 2025 initialization without a CNN prediction falls back to gradient boosting. Both models' 2024 predictions were also used for early stopping, so the validation curve is slightly optimistic for both.
- **Scoring.** The site export uses the site's canonical `observations_6h_2025.json` truth at 6-hourly leads. `comparison.json` matches pairs against every method's 2025 npz.

## Results (2025, unseen)

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

**Pipeline with the trajectory CNN** (`cnn.npz` is the a2 run, identical configuration):

| Model | Temperature | Wind |
| --- | ---: | ---: |
| Raw HRRR | 2.181 | 10.862 |
| Gradient boosting | 1.815 | 2.746 |
| CNN (trajectory) | 1.765 | 2.729 |
| Fixed 50/50 | 1.757 | **2.686** |
| Validation-weighted blend (weights on gradient boosting: 0.40 temperature, 0.15 wind) | **1.753** | 2.706 |

- **By lead.** The blend's temperature RMSE is 1.24 / 1.67 / 2.13 °C at +1 / +24 / +48 h, against raw HRRR's 1.72 / 2.21 / 2.61.
- **Site-scored.** Soulis temperature is 1.75 vs AIFS 1.95, GraphCast 2.19 and HRRR interpolation 2.28. Soulis wind is 2.67 vs 9.996–10.82 for the same three.

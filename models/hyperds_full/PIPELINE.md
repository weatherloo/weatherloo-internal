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
| `patches` | CPU, one container per month and CNN target | `/data/processed/patches/hrrr_<target>_<side>_<version>/<YYYY-MM>/` |
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
    {"id": "eric_d_soulis", "site_id": "eric_d_soulis", "models": ["gbm", "cnn"]},
    {"id": "eccc_51459", "site_id": "cyyz", "models": ["gbm"]}
  ]
}
```

- `anchor`: the station whose native HRRR chunk (265×300 points, about 800×900 km) is read once per initialization. Every target must lie inside that chunk.
- `models`: `gbm` is required. Add `cnn` to also extract a `(2·patch_half+1)²` patch and train the CNN.
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
- **CNN.** A 17×17 native patch of 15 fields (about 50 km square) valid at the lead, plus the gradient-boosting scalar features. Three seeds, early stopping on 2024.
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

# Waterloo hourly, station-aware forecast: research and experiment plan

Status (2026-11-17): the primary implementation is [`models/hyperds_full/`](../../models/hyperds_full/README.md), using the released author encoders, fusion, generated decoders and losses with hourly forecast extensions. The experiment is being expanded to **2018–2023 training, 2024 validation, and 2025 final testing at Eric D. Soulis in Waterloo**, with all seasons and all four daily cycles. Both extensions will be compared against the existing benchmarking-site methods on matching 2025 forecast/observation pairs; Toronto is secondary. See [the multi-year experiment contract](../../models/hyperds_full/MULTIYEAR.md) for the remote workflow, data audits, chronology and export/comparison pipeline. The completed 50-epoch January runs are a pipeline pilot, not the final evaluation.

The earlier `models/temporal_station/` fixed-station networks and `models/temporal_hyperds/` field-only networks are comparison baselines, not the primary paper implementation. The research notes below preserve the original HRRR-oriented proposal; the full implementation's README describes the experiment actually running.

The local `audit_hourly_training_data.py` inventory found **no HRRR files** under the repo-local `data/raw/hrrr` for 2021–25. It found hourly Soulis and CYYZ raw archives, but Soulis has only **2,026 temperature/wind hours in 2022** (versus 8,668 in 2021 and 8,713 in 2023). The first runnable direct/autoregressive comparison is in `models/temporal_station/`; it uses the existing one-station Zarr schema and does not yet establish spatial generalization. These counts are local availability, not a claim about remote archive coverage.

## Research question

Can a model that predicts weather at arbitrary locations, guided by station observations during training, improve **hourly forecasts from +1 to +48 hours** around Waterloo? Does feeding each predicted hour back into the next step help or hurt compared with predicting all hours directly?

The target is an *issued forecast*: all inference inputs must have been available when the forecast was issued. The first target is 2 m temperature in °C at stations; 10 m wind speed in km/h follows after its measurement quality is checked. Add 1-hour precipitation later, with a separate occurrence/amount evaluation.

## What the paper actually establishes

[HyperDS](https://arxiv.org/html/2401.11960) takes a meteorological field at one time, a sequence of satellite observations, and location coordinates; its hypernetwork generates an implicit decoder for station-scale values. Station observations are **training labels**, rather than required real-time inputs at inference. Its experiments downsample ERA5 at the **same valid time** and reconstruct ERA5/station observations. That is downscaling, not a demonstrated +1–48 h forecast or autoregressive rollout. Its spatially held-out stations and station loss are valuable ideas to retain. The paper uses Himawari over East Asia; it is not the satellite source for Ontario. See [problem setting](https://arxiv.org/html/2401.11960#S3), [architecture and losses](https://arxiv.org/html/2401.11960#S4), and [experiment protocol](https://arxiv.org/html/2401.11960#S5).

## Repository starting point

| Existing component | Reuse | Gap for this experiment |
| --- | --- | --- |
| `pipelines/download/download_hrrr.py` and `download_era5.py` | Hourly HRRR f00–f48 and hourly ERA5 regional files | Inventory actual coverage before any backfill; ERA5 is a training target only. |
| `src/hrrr_bias_correction/make_data.py` | One 49-hour HRRR crop and Soulis labels per initialization; UTC matching; 2021–23/2024/2025 split | One station, fixed location, temperature/wind only; no observed history as input or autoregressive feedback. |
| `src/hrrr_bias_correction/model_arch.py` | CNN–LSTM baseline over the lead dimension | It predicts all lead biases from the known HRRR lead sequence; it is not an autoregressive weather rollout or arbitrary-coordinate decoder. |
| `models/unet-hrrr/` | HRRR/ERA5 U-Net prototype and station loss | Current single-lead grid approach has spatial/geometry assumptions to audit before reuse. |
| `benchmarking-site/data/observations/` | Existing CYYZ and Soulis raw observations and 6-hour benchmark series | Rebuild hourly series from raw sources; 6-hour JSON is inadequate for hourly training. |

`docs/lit-review/weather-forecasting.md` already sets a 48-hour regional target. Keep `WEATHERLOO_DATA_ROOT` as the portable root: immutable archives under `raw/`, reusable tensors under `processed/`, trials under `experiments/`, checkpoints under `checkpoints/`, logs under `logs/`, and final outputs under `published/`.

### Data audit before modeling

1. Verify HRRR cycle/lead completeness, variable units, and the subset of 2021–25 with matching hourly observations. Do not infer usable sample counts from filenames alone. Audit UTC conversion, daylight-saving handling, missing values, and duplicate initializations. Keep precipitation accumulation windows explicit.
2. Check station geometry. The repository's earlier builder recorded Soulis as **43.4668, −80.5164**, while the [station operator lists 43°28′25.6″ N, 80°33′27.5″ W](https://www.civil.uwaterloo.ca/weatherstation/station-information/) (about **43.47378, −80.55764**). New temporal training tensors use the operator's coordinates; the two-sample pilot used the earlier point. Reconcile historical benchmark coordinates before comparing against published benchmark numbers or making spatial claims.
3. Build an hourly station inventory from the [UW Soulis archive](https://www.civil.uwaterloo.ca/weatherstation/data-archives/) and [ECCC hourly climate collection](https://api.weather.gc.ca/collections/climate-hourly?f=html). Identify nearby ECCC stations with sufficiently overlapping temperature/wind data and stable station metadata. Use Soulis as an important test location, not the only training/validation station. The ECCC collection contains only a subset of Canadian stations, so check its coverage before committing to a spatial split.
4. Verify *availability time*, not only valid time, for HRRR runs, station reports, and satellite images. Set an explicit issue-time cutoff and latency policy. Future ERA5, future station observations, and future satellite images can supply labels but cannot enter inference features.

## Forecast contract and experiments

For each HRRR initialization `T0`, record `issue_time` (when that run is usable), forecast valid time `T0 + h`, lead `h`, and the exact inputs available by `issue_time`. The known HRRR forecast trajectory may be used as future forcing **if its run was available at issue time**. This makes the main task regional **forecast correction and station-scale downscaling**, rather than weather prediction from scratch.

Use a fixed regional HRRR crop large enough to include Waterloo and nearby training/held-out stations. The primary model must use HyperDS's field encoder and generated-weight coordinate decoder, queried at `(lat, lon, lead_hour)`; the first implementation is in `models/temporal_hyperds/`. Train station targets with multiple stations and spatial holdout. Add the paper's observational image encoder and fusion branch once issue-time-safe GOES inputs have been prepared. The fixed-station CNN models remain baselines, not substitutes for the paper-based model.

| Experiment | Inputs at issue time | Output | Purpose |
| --- | --- | --- | --- |
| A. Operational baselines | HRRR forecast; latest permitted station reports | Raw HRRR interpolation, persistence, seasonal/lead bias correction | Establish skill and a cost floor. |
| B. Existing temporal baseline | Full available HRRR lead sequence | CNN–LSTM station bias sequence | Measure what the repo's time-aware model already provides. |
| C. Direct temporal HyperDS | HRRR lead sequence, coordinate/lead queries, optional pre-issue observations | Generated decoder predicts `correction(lat, lon, h)` for all h | Test the paper's coordinate-query mechanism with explicit time. |
| D. Autoregressive temporal HyperDS | Recurrent encoded state, prior **predicted** correction, HRRR at next lead, coordinate/lead queries | One-hour correction, unrolled to +48 | Test whether feedback adds skill. No observed future states during evaluation. |
| E. Optional satellite ablation | Same as C/D plus images available by issue time | Same output | Measure the value of remote sensing after the station/HRRR system works. |

For D, feed the predicted regional latent/field state from hour `h` into hour `h+1`; query the decoder at stations or grid points. Keep HRRR's next lead as an explicit forcing signal. A separate no-HRRR rollout can test free-running prediction but is a harder, different problem. Train on short unrolls first, then 24–48 h unrolls; validate in fully free-running mode, without teacher-forced future labels. Use the same training records and issue-time information for C and D.

Training losses: station MAE/Huber on available station labels with masks; optional grid loss against ERA5 at each **valid time**; no ERA5 input at inference. Compare station-only and station+grid supervision. ERA5 is coarser than HRRR and is not station truth, so report station metrics separately. Evaluate wind as speed from predicted vector components or a speed-specific head; check sensor height/exposure and the previously observed large Soulis wind bias first.

## Data choices

| Source | Role now | Decision |
| --- | --- | --- |
| HRRR from the existing AWS downloader | Forecast input and baseline | Primary source; its 3 km grid and hourly leads already match the proposed horizon. |
| UW Soulis 15-minute archive | Hourly station truth; permitted recent-history feature after latency audit | Primary Waterloo target, with coordinate and wind-quality audit. |
| ECCC hourly station observations | More spatial labels and held-out stations | Required to support a claim of generalization to arbitrary locations. |
| ERA5 hourly regional subset | Optional dense grid loss and diagnostics | Use only at matching valid time as a target; it is a reanalysis, not the operational forecast input. |
| [NOAA GOES-East ABI archive](https://registry.opendata.aws/noaa-goes/) | Optional pre-issue satellite sequence | Use GOES-16 for earlier years and GOES-19 after the 2025 operational transition. Crop only the Ontario area, test usable sector/bands, night/cloud behavior, timestamps, and data cost before a backfill. |

## Evaluation and decision gates

- Preserve chronological train/validation/test isolation: use 2021–23 / 2024 / 2025 *if the coverage audit supports it*. Group by initialization so overlapping valid hours never straddle splits. Hold out one or more stations spatially; if too few qualify, restrict claims to known stations.
- Report station MAE and RMSE by lead (+1, +3, +6, +12, +24, +48 h), season, day/night, and station. Include sample counts, missingness, and paired uncertainty intervals by initialization day. For precipitation, add event detection and wet-hour amount scores later.
- First gate: evaluate the direct temporal HyperDS against raw HRRR and simple lead/season bias correction on spatially held-out stations and 2024 validation times. Second gate: compare autoregressive and direct temporal HyperDS on matched records, including +24–48 h drift. Satellite fusion is a separate ablation once issue-time-safe images are available.
- Keep every checkpoint's config, source revision, station list, issue-time rule, split dates, normalization stats, and data provenance. Publish only evaluated forecast artifacts; do not add a dashboard method on validation results alone.

## Modal execution and storage

The intended workflow is **remote first**: the laptop holds source code and small diagnostics, not bulk HRRR or ERA5 data. `models/temporal_station/modal_app.py` now has a bounded Modal CPU ingest pilot that downloads two HRRR cycles and UW station archives to remote scratch, builds and verifies one train and one validation tensor, and keeps only the processed Zarr store plus source checksums in a Volume. Its GPU function reads that Volume and writes checkpoints to another Volume. This establishes the storage path, not a complete training archive.

The fixed-station Zarr schema stores `49 × 30 × 30 × 3` float32 inputs per initialization, about **0.5 MiB per cycle** or **0.72 GiB per complete year** at four cycles per day before compression. This is a planning estimate from tensor shape, not a measured archive size. A multi-station dataset and metadata add storage; measure the first remote sample before projecting a full backfill.

1. Do the coverage/geometry audit and a one-week tensor build on a WATcloud CPU node or a Modal CPU function. Run a tiny smoke test; inspect exact shapes and forecast/target timelines before paying for GPU training. Modal's [large dataset ingestion guide](https://modal.com/docs/guide/dataset-ingestion) recommends temporary container disk for download/transform and persistent storage only for the output.
2. **If WATcloud storage is available:** keep any immutable raw archive under remote `WEATHERLOO_DATA_ROOT/raw/` there. Stage only processed regional training shards to a [Modal Volume](https://modal.com/docs/guide/volumes), transferring from WATcloud or a cloud bucket directly rather than through a laptop.
3. **If WATcloud storage is unavailable:** fetch public HRRR/source data from a Modal CPU job into temporary container disk, crop/transform there, and persist only processed shards and a source/time/checksum manifest to a Modal Volume. This requires adapting the current downloader/builder; do not retain an unnecessary multi-year raw copy. If reproducibility or repeated processing later warrants a raw archive, use an object bucket such as S3 or GCS and mount it via [CloudBucketMount](https://modal.com/docs/guide/cloud-bucket-mounts). Bucket mounts are suited to large sequential reads; Zarr's many small writes and append operations are better on a Modal Volume.
4. Use a separate Volume for checkpoints and logs, with explicit commits/reloads as required by its consistency model. Do not repeatedly copy the full HRRR/ERA5 archive into GPU containers.
5. Package the training entrypoint in a pinned Modal image, start with one [L4 or A10 GPU](https://modal.com/docs/guide/gpu), and record wall time, utilization, data loading time, and cost. Run a short checkpoint/resume test before a long job. Scale GPU size only if the memory/profile requires it.
6. As of this plan, [Modal lists](https://modal.com/pricing) the first **1 TiB of Volume storage** as included, then **$0.09/GiB-month**, with L4 at **$0.000222/s (~$0.80/GPU-hour)** and A10 at **$0.000306/s (~$1.10/GPU-hour)** before CPU, memory, and network charges. Verify current rates before a run, set a budget, and estimate full-job cost from the smoke-test throughput. Modal also lists academic compute-credit grants; eligibility is separate from the model plan.

## Implementation order

1. **Audit and forecast contract:** station coordinates/instruments, hourly coverage, HRRR completeness, publication latency, UTC/unit tests, and frozen 2021–25 split manifest. Deliver a small sample manifest with valid masks.
2. **Baselines and dataset:** reuse the existing HRRR/Soulis builder, expand to multiple stations and issue-time-safe history, score raw HRRR/persistence/bias correction/CNN–LSTM on the same hours.
3. **Direct temporal HyperDS:** generated-weight coordinate decoder + explicit lead coordinate, station loss, optional ERA5 grid loss; evaluate held-out times and stations.
4. **Autoregressive temporal HyperDS:** one-hour state update and predicted-correction feedback, full +48 rollout, matched comparison and drift plot.
5. **Satellite fusion:** add a cropped, pre-issue GOES sequence and the paper's second encoder/attention branch; compare against the field-only HyperDS ablation.

The two-cycle remote pilot is deliberately separate from a multi-year backfill and leaves existing Modal weather datasets untouched.

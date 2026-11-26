# Waterloo forecasting experiment: multi-year HyperDS

**Operational state, November 19 at 16:04 PST:** the Modal app
`ap-369RHF6NZkv5uCBNTOjaMw` was stopped from the CLI. Direct remains at
28/50 saved epochs and AR at 21/50; no optimizer is active. Preserve all
artifacts and do not restart automatically. A separate `weatherloo-hyperds-nwp`
app is active; its experiment is documented in [NWP.md](NWP.md). Monitoring
scope is awaiting user clarification.

## Objective and evaluation

**Active run:** `waterloo_2018_2025_stationqc_v3`, using the existing immutable
`waterloo_2018_2025_v2` weather shards and the checksummed `soulis_hourly_v3`
label overlay. Coordinator `fc-01M7QZJDQR997XQCVMBA178KSR` replaces the first
training attempt. The chronology, architecture, losses and epoch counts stay
fixed. The first attempt was stopped for observation QC repair, not resumed
with changed labels. Its files and manifests are retained.

The clean audit passed at 10:45 PST, with 319,615 Soulis temperature and
243,165 wind labels in training. Initialization-only pretraining began in
`fc-01M7QZMZ7QEXKAZ30WK94FHMFH` on an NVIDIA A10 at 10:46:59 PST and
completed all 50 epochs. Both full-data forecasting models launched at
12:20 PST; their optimizer activity is confirmed. All 23 checks passed
(six parser, eight data/evaluation, six architecture, three feasibility).

Train the released HyperDS architecture with direct hourly coordinates and autoregressive predicted-field feedback. Compare both forecasting methods with the repository's existing forecast methods at **Eric D. Soulis in Waterloo**. Toronto Pearson is a secondary benchmark location. Guelph and Elora supply additional regional labels; they do not replace Waterloo as the final evaluation target.

The primary question is whether either extension improves **2025 Waterloo temperature and wind forecasts**, overall and by forecast lead. The single-time paper model is a downscaling/pretraining reference, not a future-weather baseline.

The separate [feasibility experiment](FEASIBILITY.md) was stopped at the user's
request on November 18 at 11:21 PST because all multi-year data are ready. Its
saved artifacts are retained; its training/evaluation jobs must not be restarted
without a new user request. Only this full experiment continues, and the smaller
experiment is no longer part of the completion criteria.

## Frozen chronology

| Split | Years | Planned initializations | Use |
| --- | --- | ---: | --- |
| Train | 2018–2023 | 8,756 | Fit encoders, fusion and generated decoders on all available regional station labels and grid targets |
| Validation | 2024 | 1,456 | Select checkpoints using Soulis station loss only |
| Test | 2025 | 1,460 | Final Soulis results and secondary CYYZ results; never used by training or checkpoint selection |

Every day uses 00/06/12/18 UTC, with hourly leads 0–48. The last eight initializations before each train/validation boundary are excluded so their +48 h targets cannot enter the following split. The January pilot's Guelph spatial holdout is superseded by this chronological Waterloo forecast evaluation. Missing measurements remain masked, including the known Soulis 2022 gaps. Complete regional grid/satellite examples can still train when station measurements are missing.

The 96 monthly shards target **11,672 initializations**, about 313 times the pilot's training initialization count. Actual accepted counts, source failures, station coverage and monthly coverage are recorded. Every month must retain at least 80% of its planned cycles before the training gate passes.

## Inputs and storage

- Five ERA5 fields on the same regional grid and units as the paper implementation: u/v wind, temperature, surface pressure and hourly precipitation. Only the initialization field enters the forecast; all later fields are labels.
- Two GOES-East frames, each with four bands, per initialization. Early GOES-16 M3 and later M6 scans are supported. GOES-19 is used after NOAA's [operational transition on 2025-04-07 at 15:10 UTC](https://www.ospo.noaa.gov/data/messages/2025/04/MSG_20250407_1510.html).
- Older S3 objects were backfilled after observation time (for example, January 2018 products copied in July 2018). The experiment requires scan completion **and original filename production time** at least 15 minutes before initialization. Original dissemination latency is unverified; S3 LastModified is retained as copy provenance. This is a retrospective availability proxy.
- Tiny invalid-pixel holes in a satellite band are filled from the nearest valid pixel in that same scan. Each band records the affected fraction; more than 1% invalid pixels rejects that initialization. No future frame or weather target supplies a fill value.
- Soulis uses the benchmark registry coordinates **43.4668, −80.5164**. The January pilot used different coordinates; its results are not reused as final benchmark results.
- Normalization uses training shards only, with unique field hours assigned to their calendar month. Validation and test values do not enter normalization.
- Monthly stored fields and satellite tensors are read back and checked against SHA256, shape, finite-value, timing and unit checks before completion is recorded. Source identifiers, failure records, label coverage and manifests remain with each shard.

All ingestion and training run remotely:

```text
weatherloo-temporal-data:
  raw/observations/hyperds_2018_2026/
  processed/hyperds_full/waterloo_2018_2025_v2/shards/YYYY-MM/
  processed/hyperds_full/waterloo_2018_2025_v2/{manifest,audit}.json
  published/hyperds_full/waterloo_2018_2025_stationqc_v3/hyperds_{direct,autoregressive}/

weatherloo-temporal-checkpoints:
  hyperds_full/waterloo_2018_2025_stationqc_v3/{paper,direct,autoregressive}/
```

## Training and automatic comparison

The remote dataset audit gates training. The paper model trains first for 50 epochs. Both temporal extensions then start from its best 2024 checkpoint and each train for 50 epochs with identical splits and supervision. All three retain the paper's optimizer/loss settings and the implementation's ten-point subgrid means and spatial minibatches. Six-hour gradient windows bound memory without feeding future truth into the autoregressive state.

Training executes in five-epoch stages to stay within Modal's function timeout. Each epoch commits its checkpoint; each successful stage starts the next one. Checkpoints include optimizer, scheduler, CPU/CUDA RNG and NumPy RNG states, source hashes, normalization and the frozen split manifest. Full-year test predictions are saved in `predictions.npz` after each model's best validation checkpoint is selected.

After each temporal model completes, the remote benchmark function:

1. Reads the site's exact `observations_6h_2025.json` files, with `eric_d_soulis` primary and `cyyz` secondary.
2. Converts predicted wind from m/s to the site's km/h; retains temperature in °C.
3. Exports the site's NPZ shape, station/variable order, all 1,460 2025 cycles, and 12 lead slots. Leads 54–72 h and missing source/observation pairs remain NaN.
4. Compares with the existing saved 2025 GFS, HRRR, AIFS, GraphCast, GEFS, persistence, climatology and post-processing methods. Each pairwise comparison uses the same finite initialization/station/variable/lead pairs for both methods and reports counts.
5. Pools squared errors before taking RMSE. It also reports MAE and bias by lead and pooled across supported leads. ACC remains unavailable because per-record sign scores are not a valid pooled anomaly correlation.
6. Saves `hyperds_<mode>_2025.npz`, `metadata.json` and `comparison.json` remotely under `published/`. These can feed the existing site aggregate builder after review; this workflow does not publish pilot scores to the site.

The models currently use retrospective ERA5 initialization. Benchmark metadata records this; the comparison is not a claim of equal operational input availability against NWP-initialized methods.

At 12:20 PST on November 18, the initialization-only pretraining completed all
50 epochs and launched the two full-data forecasting models. Direct call
`fc-01M7R525Q8SVKQC3QR7044YDF9` and autoregressive call
`fc-01M7R525V9F8G9Z9KF6JWWMY27` both confirmed optimizer steps on NVIDIA A10
GPUs using all 8,688 training initializations. Each embeds hourly lead time,
trains against all 49 leads, and has its own 50-epoch budget. The AR model
additionally feeds predicted coarse weather fields forward; the direct model
queries each lead from the initialization inputs.

### Validation RMSE diagnostics

The frozen trainer's per-epoch logs contain physical-unit MAE and a combined
normalized station MSE. They do not contain separate temperature/wind RMSE;
those cannot be recovered from MAE or by taking a square root of the mixed
checkpoint score. `modal_diagnostics.py::validation_rmse` separately evaluates
the saved best/latest paper checkpoints on 2024 Soulis, with zero optimizer
steps and no changes to the frozen trainer. It saves small reports under
`checkpoints/hyperds_full/waterloo_2018_2025_stationqc_v3/diagnostics/`.

The epoch-45 evaluation on November 18 gave temperature RMSE 1.50455°C and
wind RMSE 4.97429 km/h, with 1,427 finite matched observations per variable.
These are initialization-time downscaling errors. No historical per-epoch
physical-unit RMSE curve was saved, and no future-forecast skill is implied.

For the requested **forecast RMSE by lead time**, once a temporal checkpoint
is saved, run the same read-only diagnostic with `--mode direct` or
`--mode autoregressive`. It evaluates the saved best 2024-selected checkpoint
over the complete 2024 validation set and saves per-hour 0–48 h Soulis
temperature/wind RMSE, MAE, bias and finite matched counts in the small JSON
report. Forecast aggregates exclude lead 0. Training, optimizer state and
2025 test labels are not changed or used by this diagnostic. Keep prediction
arrays remote; download only the JSON to plot the two RMSE-by-lead curves.

The first full-data direct checkpoint completed epoch 1 in 1,535.23 seconds
(25.6 minutes). Its 2024 Soulis validation MAE across +1–48 h is 3.81769°C
and 4.90532 km/h, with 68,417 finite matched pairs per variable. The separate
RMSE diagnostic completed at 12:58 PST on November 18: pooled forecast RMSE
4.90124°C and 6.13817 km/h. It saved
`diagnostics/validation_rmse_direct_best_epoch_1.json` with all hourly leads.
This early checkpoint is not the final 2025 forecast result. Counts per lead
range from 1,423 to 1,428; overlapping forecasts share observations. Its
diagnostic MAE agrees with the frozen epoch log, and its per-lead squared
errors and counts reproduce pooled RMSE. Do not rerun the same checkpoint
diagnostic at every heartbeat.

The first autoregressive epoch completed in 2,296.52 seconds (38.3 minutes).
Its 2024 Soulis forecast MAE is 4.27318°C and 5.98378 km/h. Its independent
epoch-1 RMSE diagnostic completed at 13:30 PST, giving pooled +1–48 h RMSE
5.49567°C and 7.32179 km/h over the same 68,417 matched pairs per variable.
The report is `diagnostics/validation_rmse_autoregressive_best_epoch_1.json`.
Both first-checkpoint diagnostics are now complete and verified against the
frozen epoch logs. Further RMSE evaluations are only needed at five-epoch
milestones when the selected best checkpoint changes, and at completion.

After AR completed five epochs, its selected epoch-4 checkpoint was evaluated
once at 15:59 PST on November 18. Pooled +1–48 h RMSE is 4.69755°C for
temperature and 8.31894 km/h for wind, over the same 68,417 pairs per
variable. This improves temperature from AR epoch 1 (5.49567°C) while wind
worsens (7.32179 km/h at epoch 1). The small report is
`diagnostics/validation_rmse_autoregressive_best_epoch_4.json`; its MAE and
pooled squared errors were checked against the frozen epoch log. The current
RMSE comparison uses direct epoch 1 and AR epoch 4. Both first-stage
milestone evaluations are covered; do not repeat these checkpoint scores
before a later five-epoch milestone or completion unless correcting an error.

At the 13:24 PST training snapshot, direct had saved two epochs and AR one.
Direct epoch 2 reduced temperature MAE to 3.53887°C but increased wind MAE
to 5.69139 km/h, leaving epoch 1 selected by combined station loss. The first
RMSE comparison therefore plots epoch 1 for both modes; it does not describe
the newer direct epoch-2 weights.

### Baselines and loss plots for every forecast chart

At the user's request on November 18, forecast chart updates must also show
HRRR, simple baselines and the temporal training-loss curves. The standalone
`modal_validation_baselines.py` leaves all five frozen training sources unchanged.
`prepare` fits per-hour mean corrections and univariate linear regressions of
**coarse ERA5 initialization persistence** on the 8,688 frozen 2018–2023 training
initializations, using the v3 Soulis labels. These are not HRRR corrections.
Wind corrections are floored at zero. No 2024/2025 labels fit coefficients.
It also creates raw ERA5 initialization persistence and observed persistence.
Observed persistence uses the **previous hour**'s station label (1 h lag),
because canonical nearest-time truth matching allows ±15 minutes and must not
expose a later measurement to the initialization baseline. The lag is recorded
in the fit/comparison JSON and plot legend. Existing fitted corrections are
reused when this small station-only cache is refreshed.

`hrrr` extracts 2024 00/06/12/18 UTC, hourly 0–48 h HRRR forecasts at Soulis
from NOAA HRRR processed by [dynamical.org](https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour/).
It bilinearly interpolates native Lambert conformal x/y u/v before deriving
wind speed. Only station arrays persist remotely; the laptop receives no weather
fields, checkpoint tensors or prediction arrays. Monthly station caches allow
recovery without repeating completed extraction. HRRR uses a separate Python
3.12/Zarr 3 image; training retains its existing Python 3.11/Zarr 2 image.

`modal_diagnostics.py::validation_rmse` additionally caches its Soulis prediction
arrays remotely. Existing selected direct epoch 1 and AR epoch 4 are evaluated
once more to create this cache for the newly requested matched comparisons;
later calls reuse completed epoch caches. `compare` checks initialization and
truth equality and applies a **common finite mask across all displayed methods**
per variable and lead. Its small report is
`diagnostics/validation_comparison_direct_<epoch>_autoregressive_<epoch>.json`.
Missing HRRR/initial observations remain missing, with the resulting counts
reported. Aggregate RMSE pools squared errors across +1–48 h. These matched
scores can differ from the earlier full-cohort reports, and must be labeled.

Baseline preparation started in `ap-QYUuSSSLPALag6LbMaejSE`; prediction-cache
diagnostics started in `ap-20ItV7WTYkuSXkGLq0ZxX7` (direct) and
`ap-j31IP0f0FUczBuhLkNoWYe` (AR). The first baseline image build
`ap-i2kIpZQkC1utG4n1PrMK07` failed because the catalog client requires Python
3.12; no data or training changed, and the separate image was corrected.
Preparation completed at 16:14 PST with all 8,688 training and 1,439 validation
initializations. HRRR extraction started in `ap-OdPyGWzVSqggzuBqjD46LK`;
earlier HRRR startup attempts `ap-XOk7ZYg2b6BNexjnb6FbwI` (missing module
mount) and `ap-a5LDkJ8BQRfiUG56OwB47L` (descending native y axis) are stopped
or failed; both issues are fixed without altering training or prepared data.
Inspect these calls before starting duplicates. After prepare completes, run
`modal run --detach models/hyperds_full/modal_validation_baselines.py::hrrr`,
then `::compare` once HRRR and both prediction caches are committed.

For every RMSE plot, also fetch the two small temporal `metrics.jsonl` files
and plot saved per-epoch training `grid_mse + 0.05*station_mse`, plus 2024
Soulis `validation_station_mse`. These are normalized losses, not physical
RMSE. Plot scripts `plot_temporal_loss.py` and `plot_validation_comparison.py`
are in the chat's existing local `training-loss` visualization directory.
Use a logarithmic training-loss axis to retain the large first AR epoch while
showing subsequent changes. Label both current saved epochs and the actual
selected checkpoints used for the forecast curves. The recurring 30-minute
monitor has been updated with these requirements.

### Completed initialization-time Waterloo comparison

The paper pretraining finished 50 epochs; its best checkpoint was epoch 46,
selected using 2024 Soulis station loss. A separate zero-optimizer-step CPU
diagnostic scored its saved **2025** lead-zero predictions and interpolated
ERA5 on identical finite station/time pairs:

| Method | Temperature RMSE (°C), n=1,248 | Wind RMSE (km/h), n=1,159 |
| --- | ---: | ---: |
| ERA5 1° coarse input | 1.80725 | 10.49507 |
| ERA5 0.25° original grid | 1.75531 | 10.37576 |
| HyperDS initialization-only | 1.72335 | 5.65330 |

HyperDS MAE is 1.16218°C and 4.61121 km/h. Against original 0.25° ERA5,
RMSE improves by about 1.8% for temperature and 45.5% for wind. Wind still has
a +4.35734 km/h mean bias. Wind is scored against Soulis's 4.4 m instrument;
ERA5 supplies 10 m wind, so the local wind comparison includes sensor-height
and exposure adaptation.

This establishes a local same-time downscaling gain over ERA5 interpolation.
The paper's full numerical replication is not established: its benchmark uses
Himawari/Weather2K and 400 spatially held-out Chinese stations, whereas Soulis
is included in earlier training years here and 2025 is a temporal holdout.
UNet/EDSR comparisons and the paper's full ablation suite have not been run
for this Waterloo initialization-only model. These scores do not establish
future-forecast skill.

Reproduce with `modal run models/hyperds_full/modal_diagnostics.py::paper_comparison`.
The small report is saved remotely as
`checkpoints:/hyperds_full/waterloo_2018_2025_stationqc_v3/diagnostics/paper_2025_downscaling_comparison.json`.
Prediction arrays and weather fields stay remote. The diagnostic checks saved
initialization order and frozen Soulis labels, bilinearly interpolates u/v
before deriving wind speed, and pools physical squared errors before RMSE.

## Operations

### Billing interruption, November 18 at approximately 16:16 PST

All active containers stopped after the workspace spending limit was reached.
The deployed `status` function rejects even a CPU call with
`ResourceExhaustedError: Function call failed: workspace billing cycle spend limit reached`.
CLI billing reports $59.99 metered usage and $25.42 billed after credits and
free-storage adjustments. This is an account billing block, not a model error.
The user was asked to raise the limit (suggested total $150, about $90 extra
headroom); no budget or billing setting has been changed by the agent.
The user subsequently reported adding more. At 16:23 PST, the open Modal
billing page showed a **Spend limit of $70** but the separate **Usage limit
still at $55**. Another status call was still rejected. The user has been
asked to raise the Usage limit as well; do not claim training has resumed until
a new stage actually starts. The billing page is
`https://modal.com/settings/cindehaa/usage?tab=overview`.

The last committed epochs are direct **9/50** and AR **6/50**, with selected
checkpoints still direct 1 and AR 4. Data, checkpoint and optimizer state remain
remote. The interrupted stage calls `fc-01M7RCCSA2KRWBMMXHVKCTCWZC`
(direct) and `fc-01M7RG3MXWPKFYH881EBS42RW8` (AR) were cancelled after
the account shutdown to prevent delayed duplicate execution on recovery.
After the user restores the spending allowance, first inspect recent calls and
metrics to ensure no later continuation is active, then spawn exactly one
deployed `train_stage` per mode with `stop_epoch=10`. The unchanged trainer
resumes committed optimizer/scheduler/RNG state automatically; successful epoch
10 stages continue the normal five-epoch chain. Do not rerun ingestion,
pretraining, station repair or `run_training`.

The direct epoch-1 validation station prediction cache committed successfully
at 16:15:39 PST. The AR cache evaluation was interrupted; its prior RMSE JSON
still exists, but its new station prediction NPZ does not. Baseline fitting and
validation arrays committed. HRRR reached at least 21 January initializations
but **no monthly station cache committed**. Auxiliary apps
`ap-OdPyGWzVSqggzuBqjD46LK` (HRRR) and
`ap-j31IP0f0FUczBuhLkNoWYe` (AR diagnostic) were explicitly stopped after
the billing interruption, so they must be relaunched once allowance is restored.
Then run `compare` and plot its matched JSON together with current training loss.
Until that report exists, do not imply the HRRR/baseline curves are available.
The existing 30-minute monitor remains active as requested and must report
the billing block honestly, without repeatedly submitting blocked compute.

### Allowance restored and training resumed, 16:24 PST

The user raised the Usage limit to **$70** (the Spend limit is also $70).
The deployed status function succeeded and reconfirmed all 96 completed months,
the audited 8,688/1,439/1,457 split, and the v3 overlay. Exactly one new
`train_stage(mode, 10)` per model was launched:

- Direct: `fc-01M7RK1EMZ4E0PKA168V07D0PB`, started 16:24:27 PST.
- Autoregressive: `fc-01M7RK1ERYP3GKDBVDR5BAY01J`, started 16:24:28 PST.

Both calls are running. The old interrupted calls are confirmed terminated.
Follow the new calls and their continuations; never start `run_training` again.
HRRR station extraction restarted in `ap-hrzUkjsNMPCt0Sx4J10e6d`, and the
missing AR station-prediction cache evaluation restarted in
`ap-nL5Fny9Uu8HDvgRjCip5cz`. Baseline preparation and the direct cache are
already committed and must not be recomputed. A $70 Usage limit gives only
about $15 further metered compute allowance above the current usage, whereas
remaining 50-epoch training is estimated at $65–80 using measured epoch durations
and current A10/CPU/memory rates. The user was informed; do not change limits.
Both resumed models confirmed actual optimizer steps at 16:25 PST. The AR
epoch-4 prediction cache committed at 16:28:25 PST; its diagnostic application
completed. The one-hour-lag observation baseline cache also committed in
`ap-GWXoPvblSUZzgeHMTHMeow`. Only HRRR extraction and the final CPU `compare`
remain for the requested updated baseline plot; no further GPU diagnostic is
needed for epochs 1/4.

### Matched 2024 baseline comparison completed, 16:37 PST

HRRR station extraction completed all 12 months and all 1,439 frozen validation
initializations at 16:36:34 PST. The CPU comparison in
`ap-vT3Dxxj98x0TATk8lYHKVq` completed with source/overlay hash checks,
identical initialization order, matching frozen truth and a common finite mask
across all seven methods. Each variable has **68,011 matched +1–48 h pairs**;
per-lead counts are recorded. These scores use direct **epoch 1** and AR
**epoch 4**, not the latest training weights. They differ slightly from the
earlier full-cohort scores because the lagged-observation baseline has missing
initial observations. All predictions stay remote; only small reports are local.

| Method | Temperature RMSE (°C) | Wind RMSE (km/h) |
| --- | ---: | ---: |
| Direct, selected epoch 1 | 4.90971 | 6.13146 |
| AR, selected epoch 4 | 4.70125 | 8.31353 |
| HRRR forecast | 1.85820 | 9.71222 |
| ERA5 initialization persistence | 5.74686 | 10.77530 |
| ERA5 persistence + mean correction | 5.74611 | 6.13159 |
| ERA5 persistence + linear regression | 5.60538 | 4.12211 |
| Observed persistence, 1 h lag | 6.32098 | 5.49023 |

The early neural checkpoints improve temperature over ERA5 persistence and its
simple corrections but are substantially worse than HRRR. Wind is worse than
the train-only linear-regression baseline for both models; direct wind RMSE is
essentially tied with a mean correction. Raw NWP 10 m wind versus Soulis 4.4 m
truth includes a height/exposure mismatch, so beating raw HRRR wind alone does
not establish forecast superiority. These are preliminary 2024 validation scores;
2025 remains reserved for unchanged completed models.

Report: `diagnostics/validation_comparison_direct_1_autoregressive_4.json`.
Provenance: `diagnostics/validation_hrrr_provenance.json`. HRRR station caches,
baseline fitting and both epoch prediction caches are complete. Future updates
must reuse them; only score a newly selected checkpoint at the established
milestones. The resumed full-data training calls remain active at 9/50 direct
and 6/50 AR saved epochs.

[Modal deployment and logs](https://modal.com/apps/cindehaa/main/deployed/weatherloo-hyperds-multiyear)

At the November 18 18:02 PST check, direct had saved **11/50** epochs and AR
**8/50**. Direct's epoch-10 stage completed successfully and continued at
17:01:45 PST in `fc-01M7RN5T7PP1Q2G8NZAPQ1ZHYT` through epoch 15.
AR remains in `fc-01M7RK1ERYP3GKDBVDR5BAY01J` through epoch 10. Both have
recent optimizer activity and no recent stderr errors. Selected checkpoints are
still direct epoch 1 and AR epoch 4, so the direct epoch-10 milestone requires
no repeated GPU evaluation. Latest full-cohort 2024 Soulis MAE is direct
3.32468°C / 6.33507 km/h and AR 3.56383°C / 8.29857 km/h (68,417 pairs
per variable). Latest measured epochs took 36.7 and 40.2 minutes respectively;
use these recent times rather than direct's earlier 26-minute duration.
The updated loss chart is `temporal_loss_direct_11_autoregressive_8.png`;
the matched RMSE/baseline comparison remains the selected epoch-1/epoch-4 plot.

Both first five-epoch temporal stages completed successfully on November 18.
At 18:35 PST, direct had saved **12/50** epochs and AR **9/50**; both stages
remained active without recent errors. Latest 2024 Soulis MAE was direct
3.31825°C / 6.28646 km/h and AR 3.94320°C / 6.72637 km/h (68,417 pairs
per variable), with recent epoch durations 36.9 / 40.4 minutes. Direct still
selects epoch 1. AR newly selects **epoch 9**: combined normalized Soulis MSE
0.221586726 versus previous epoch-4 best 0.224813103, a 1.44% improvement.
Physical RMSE for AR epoch 9 has not been evaluated; its next diagnostic is due
after the epoch-10 milestone, using the then-selected best checkpoint exactly
once and the already frozen baseline arrays. The available forecast comparison
still evaluates direct epoch 1 / AR epoch 4 and must be labeled as such, rather
than presented as the new selected AR checkpoint. Updated loss chart:
`temporal_loss_direct_12_autoregressive_9.png`.

At 19:08 PST, direct had saved **13/50** epochs and AR remained at **9/50**.
Both current calls were running with recent optimizer steps and no recent
stderr errors. Latest Soulis MAE was direct 3.13952°C / 7.39651 km/h and
AR 3.94320°C / 6.72637 km/h, each over 68,417 forecast pairs per variable.
Latest epoch durations were 36.8 / 40.4 minutes. Selected checkpoints remain
direct epoch 1 / AR epoch 9; direct's lower temperature MAE did not offset its
higher wind error in the combined selection criterion. AR has not yet reached
its epoch-10 diagnostic milestone, so no duplicate GPU evaluation was launched.
The available matched forecast comparison still scores direct 1 / AR 4.
Updated loss chart: `temporal_loss_direct_13_autoregressive_9.png`.

At 20:09 PST, direct had saved **15/50** epochs and AR **11/50**. Direct
continued through epoch 20 in `fc-01M7RZH1HCD6VY746AA3DRQP2Y` at 20:02:38;
AR continued through epoch 15 in `fc-01M7RWD52DNP8CJAW1HKMPY0KY` at
19:08:05. Both have optimizer activity and no recent stderr errors. Latest
full-cohort Soulis MAE is direct 2.94109°C / 7.02129 km/h and AR
3.67082°C / 7.22866 km/h (68,417 pairs per variable); epoch times are
35.0 / 40.6 minutes. Selected checkpoints remain direct 1 / AR 9, so the
direct epoch-15 milestone requires no repeated GPU evaluation.

The AR epoch-10 milestone diagnostic scored selected epoch 9 with zero
optimizer steps in `ap-hi1uGGUUjw2RPJFcgUk4ZT`, completing at 19:51 PST.
Full-cohort RMSE is 5.03755°C / 7.80296 km/h (68,417 pairs per variable).
Its cached station predictions were compared with all frozen baselines in
`ap-bdDaCgx69AZMuHMfvG61za`. No baseline refitting or HRRR downloading was
needed. New matched report:
`diagnostics/validation_comparison_direct_1_autoregressive_9.json`.
All seven methods use 68,011 identical +1–48 h pairs per variable, with
1,413–1,424 per forecast lead. Matched RMSE is direct 4.90971°C / 6.13146 km/h,
AR 5.04075°C / 7.80187 km/h, HRRR 1.85820°C / 9.71222 km/h and train-only
ERA5-persistence linear regression 5.60538°C / 4.12211 km/h. Compared with
AR epoch 4, temperature RMSE worsened and wind improved; the combined
normalized checkpoint-selection criterion does not imply every variable
improves. Common counts, pooled squared-error reconstruction and diagnostic
MAE agreement with the frozen epoch log passed. Source/overlay hashes and
truth equality passed the remote comparison checks. Plots are
`validation_comparison_direct_1_autoregressive_9.png` and
`temporal_loss_direct_15_autoregressive_11.png`. The next diagnostics are due
at later five-epoch milestones only if selected best weights change. Final
2025 forecasting evaluation and benchmark exports remain pending.

### Second allowance interruption and recovery, November 18

At the 22:44 PST check, both training calls had failed, the main app had zero
active tasks, and a status call was rejected with `ConflictError: workspace
... is disabled`. The last recorded optimizer activity was at 21:31 PST.
CLI billing reported $74.82544 metered usage; the last verified allowance was
$70, so this appeared to be another usage/billing block. No financial settings
were changed by the agent. Saved epochs were direct **17/50** and AR **13/50**.
Latest 2024 Soulis MAE was direct 2.97172°C / 7.20801 km/h and AR
3.69613°C / 7.02945 km/h, each with 68,417 pairs per variable. Selected best
epochs remained 1 / 9. The new loss chart is
`temporal_loss_direct_17_autoregressive_13.png`; the evaluated forecast
comparison remains direct 1 / AR 9 and requires no repeated diagnostic.

The user reported increasing allowance, and the status call succeeded again
at 22:46 PST, reconfirming all 96 audited months and the frozen splits/overlay.
Both interrupted calls were cancelled before replacement stages were spawned;
they are confirmed terminated. Direct resumed toward epoch 20 in
`fc-01M7S8XXWYPTK2TVEHKG9TFM9X` at 22:47:02 PST, confirmed an optimizer
step at 22:47:57, and retains the unchanged source hashes. The first AR resume
`fc-01M7S8XY20CV324MNAABH1X944` failed during PyTorch import in a worker
reused after cancellation, before training. That worker
`ta-01M7S8XY88014M809VRK1QC3FR` was stopped and only AR was retried in
`fc-01M7S9029HA3HS4CAFCVFMAZT0` toward epoch 15, started at 22:48:17 PST.
The replacement AR worker confirmed training on NVIDIA A10 at 22:49:09 and
an optimizer step at 22:49:12. Both models are now actually training from the
committed optimizer/scheduler/RNG state, rather than merely queued. Follow only
these two current calls and their normal five-epoch continuations.
The new allowance amount has not been verified. Do not change billing limits
or rerun ingestion, pretraining, baseline preparation or `run_training`.

### H100/A100 migration requested by the user, November 18 at 22:55 PST

The user explicitly authorized A100/H100 hardware for the remaining training.
`modal_multiyear.py::train_stage` now prefers one H100 per model, with A100
80 GB as a capacity fallback, four CPUs and 16 GiB host RAM. Only the worker GPU setting changed;
all five frozen model/training/data source hashes remain identical. The same
datasets, labels, splits, normalization, losses, batch sizes and optimizer state
are retained. GPU kernels can differ numerically across hardware.

Both A10 stages (`fc-01M7S8XXWYPTK2TVEHKG9TFM9X` and
`fc-01M7S9029HA3HS4CAFCVFMAZT0`) were cancelled with container termination
and confirmed terminated before new work was spawned. The unchanged saved
checkpoints are direct epoch 17 and AR epoch 13; only unfinished epochs restart.
The H100 worker configuration deployed successfully in 3.034 seconds. New calls:

- Initial H100-only direct request: `fc-01M7S9CS95DYHX23K6DXMAEVQ5`.
- Initial H100-only AR request: `fc-01M7S9CSE3HE32EXRHS11J3HSM`.

Both were queued at 22:55:05 PST and then cancelled with container termination
after waiting for capacity. They are confirmed terminated; neither completed
training. At 22:58 PST the same function was redeployed with the A100 fallback.
The replacement direct call is `fc-01M7S9KDTXJRMTH7941MBHFQPT` (through 20)
and AR is `fc-01M7S9KDZ0PCFGW8PV46PABMG1` (through 15). Both were running
at 22:59:21 PST. Follow these calls and their normal continuations; do not
start duplicate workers. Hardware-upgrade performance is not yet
measured: compare complete saved epoch durations after startup with the recent
A10 values of 35.2 minutes direct / 40.4 minutes AR. Keep all state remote and
continue the existing five-epoch diagnostic policy and final 2025 holdout.

Both replacement workers passed the 96-shard dataset audit at 23:00:38–40 PST.
Both workers logged `NVIDIA A100-SXM4-80GB` at 23:00:45–47, and both models
confirmed optimizer steps at 23:00:48 (direct) / 23:00:50 (AR). The upgraded
runs are actual training, not queued allocations. All frozen source hashes
match the previous checkpoints. Saved epochs remain 17 / 13 until the first
upgraded epochs finish; report measured GPU-specific timings when they commit.

At the 23:44–45 PST check, direct had committed **18/50** epochs and AR
remained at **13/50**, with epoch 14 still running. Both A100 calls have recent
optimizer activity and no recent stderr errors. Direct's first A100 epoch took
2,241.43 seconds (37.4 minutes), versus its last A10 epoch's 2,112.39 seconds
(35.2 minutes): about 6.1% slower in this single comparison, not a measured
speedup. AR has no completed A100 epoch yet; its last A10 epoch took 40.4
minutes. Latest Soulis MAE is direct 2.93789°C / 7.19247 km/h and AR
3.69613°C / 7.02945 km/h (68,417 pairs per variable). Selected checkpoints
remain direct 1 / AR 9, so no duplicate RMSE diagnostic was launched. Current
loss plot: `temporal_loss_direct_18_autoregressive_13.png`; the matched
forecast/baseline plot remains `validation_comparison_direct_1_autoregressive_9.png`.

A read-only `nvidia-smi` check sampled each active worker eight times at
approximately one-second intervals. Direct showed 38–44% GPU utilization and
1,429 MiB used; AR showed 45–55% and 2,283 MiB used, each on an 81,920 MiB
A100. These short snapshots show underutilization and ample memory capacity;
they do not identify the bottleneck or represent whole-epoch utilization.
No frozen source, training setting, checkpoint or worker was changed for this
check. Further epoch timing should be reported honestly before claiming that
the more expensive GPU improves training throughput.

### H100 continuation and third billing stop, November 19

Direct completed epoch 20 on A100 and continued in
`fc-01M7SFZ847XHMPC8BP9X3QPBRW`, logging `NVIDIA H100 80GB HBM3` at
00:52:09 PST. AR completed epoch 15 on A100 and continued in
`fc-01M7SG6YTNWH2YYWHCRD650BR3`, logging the same H100 device at 00:56:14.
Direct completed epoch 25 and continued toward 30 in
`fc-01M7SP8C4MX0EASZCCFPRSADM4` at 02:39:52, again on H100. All frozen
source hashes matched. H100 direct epochs 21–26 took 1,247.68–1,309.67 seconds;
AR epochs 16–19 took 1,884.07–1,925.22 seconds. Latest completed durations are
21.6 minutes direct and 31.8 minutes AR, versus last A10 values 35.2 / 40.4.
These are measured runtime improvements; H100 hourly pricing is higher.

Both workers subsequently stopped. Fresh status calls reject compute with
`ResourceExhaustedError: Function call failed: workspace billing cycle spend
limit reached`; both current stages show failure and the app has no active
training workers. The last recorded optimizer step is at 03:05 PST. At the
09:36 PST verification, CLI billing reported about $105.31 metered usage and
$70.40 billed after credits/adjustments. The user has been asked to increase
the Usage/Spend allowance; no financial setting was changed by the agent.
The remaining frozen training is estimated at roughly 25 H100 GPU-hours,
about $110 including host resources at the current $3.95 GPU-hour rate, before
evaluation/storage headroom. A $250 total Usage limit was suggested, not set.

Committed checkpoints remain direct **26/50** and AR **19/50**. Latest 2024
Soulis MAE is direct 2.87363°C / 7.53413 km/h and AR 3.54601°C /
6.99573 km/h, with 68,417 pairs per variable. Best checkpoints are direct
epoch 1 and newly selected AR **epoch 15** (combined normalized MSE
0.217018839; its MAE is 3.40890°C / 7.13872 km/h). The due epoch-15 AR
RMSE diagnostic has not run because compute is blocked. The existing matched
RMSE plot still evaluates direct 1 / AR 9 and must be labeled as historical
checkpoint scores, not the newly selected AR15. Current loss chart:
`temporal_loss_direct_26_autoregressive_19.png`.

The two interrupted calls above were cancelled with container termination to
prevent delayed duplicate execution. After allowance is restored, inspect
current calls/metrics first, then spawn exactly one deployed stage per mode:
direct through epoch 30 and AR through epoch 20. Resume committed optimizer,
scheduler and RNG state; do not rerun ingestion, pretraining, station repair or
`run_training`. Check existing diagnostic jobs/caches, evaluate the selected
AR checkpoint once, and reuse frozen baselines for the CPU comparison. Keep
all prediction arrays and checkpoints remote. Final 2025 forecasts/exports
remain reserved until unchanged models finish 50 epochs.

### A100-only recovery requested November 19 at 13:44 PST

The user increased the Usage limit and requested A100 GPUs. A fresh deployed
status call succeeded, confirming the billing block had cleared. The training
wrapper now requests `gpu="A100-80GB"` for both models and all subsequent
five-epoch stages. The five frozen model/train/data source hashes are unchanged.
The prior interrupted calls were cancelled with container termination again;
both are confirmed terminated before the replacement workers train.

Exactly one replacement stage per model was submitted at 13:45:54 PST:

- Direct: `fc-01M7TWBXFQG75GD1MAAJB55MMS`, resuming saved epoch 26 through 30.
- AR: `fc-01M7TWBXKBRG22VW89VP1YJ6RM`, resuming saved epoch 19 through 20.

Direct passed the 96-shard audit and source checks, logged NVIDIA
A100-SXM4-80GB at 13:46:55 PST and completed optimizer steps at 13:46:59 and
13:49:20. AR also passed the audit and source checks, logged the same A100
device at 13:51:06 and completed an optimizer step at 13:51:11. Direct is
computing epoch 27 and AR epoch 20. Committed optimizer, scheduler and RNG
state are retained. No new full epoch has saved yet, so progress remains
direct 26/50 and AR 19/50. Measure the newly completed A100 epoch durations;
the last saved 21.6 / 31.8 minute durations were from H100 workers.

The overdue read-only AR best-checkpoint diagnostic was launched once in
`ap-MUmS67kXmdOBkca1SQVPIf` at 13:46 PST. Its selected checkpoint is epoch 15;
no epoch-15 report or prediction cache existed before launch. Inspect this app
and the remote report/cache before any retry. When it completes, run the CPU
comparison using the existing HRRR and train-only baseline arrays, then update
the matched RMSE chart together with the saved-epoch loss chart. Until then,
the available scored forecast comparison remains direct 1 / AR 9.

At 14:24 PST both A100 training calls remain active, with optimizer steps at
14:23:28 direct and 14:23:38 AR, and no recent worker stderr errors. Saved
progress is still 26/50 and 19/50; the last complete 21.6 / 31.8 minute
durations belong to the previous H100 workers, not the current A100 epoch.

The AR15 diagnostic completed at 13:51:40 and committed its JSON and remote
prediction cache. Its full-cohort +1–48 h RMSE is 4.42452°C / 8.39376 km/h
over 68,417 pairs per variable, with zero optimizer steps. Do not repeat this
checkpoint evaluation. CPU comparison `ap-ofHer9hK4pOD0o8fIvEGDH` then completed,
reusing all existing HRRR and frozen baseline arrays. New matched report
`diagnostics/validation_comparison_direct_1_autoregressive_15.json` uses 68,011
identical pairs per variable, 1,413–1,424 at each forecast lead. Pooled
temperature/wind RMSE is direct1 4.90971°C / 6.13146 km/h, AR15 4.42611°C /
8.39550 km/h, HRRR 1.85820°C / 9.71222 km/h, and train-only ERA5-persistence
linear regression 5.60538°C / 4.12211 km/h. Relative to AR9, AR15 improves
temperature and worsens wind. HRRR still has lower temperature error and
linear regression lower wind error than either selected neural checkpoint.

The matched counts and pooled RMSE reconstruction were verified; AR15
diagnostic MAE agrees with its saved epoch log within 0.0002 physical units.
Source/overlay/truth checks passed in the remote comparison. The new forecast
plot is `validation_comparison_direct_1_autoregressive_15.png`, paired with
`temporal_loss_direct_26_autoregressive_19.png`. These are 2024 validation
results, not final 2025 test scores. ERA5 initialization is retrospective;
model wind is 10 m and Soulis wind 4.4 m.

At 14:52 PST, direct has saved 27/50 epochs and AR 20/50. The first resumed
A100 epochs took 3,210.90 s (53.5 min) direct and 3,097.30 s (51.6 min) AR,
slower than the preceding H100 epochs (21.6 / 31.8 min); these are individual
epoch measurements, not a controlled hardware benchmark. Latest 2024 Soulis
MAE is direct27 2.94996°C / 7.46670 km/h and AR20 3.72462°C / 10.09058 km/h,
with 68,417 pairs per variable. AR20 validation worsened; selected checkpoints
remain direct1 / AR15, whose existing RMSE diagnostics must not be repeated.
AR continued automatically at 14:42:50 in
`fc-01M7TZM5RNPAMW6M10G81MRB4X` through epoch25, logging A100-SXM4-80GB and
matching frozen source hashes. Direct's current call continues through 30.
Both have fresh optimizer steps and no recent stderr errors. Current loss plot
is `temporal_loss_direct_27_autoregressive_20.png`; the matched forecast
comparison remains direct1 / AR15. No new diagnostic was launched at the
AR20 milestone because the selected checkpoint was already evaluated.

At 15:52 PST, direct has saved 28/50 epochs and AR 21/50, with active
optimizer steps and no recent errors in the same A100 calls. Latest epoch
durations are 3,100.06 / 3,101.47 s (51.7 min each). Latest Soulis validation
MAE is direct28 2.98098°C / 7.28581 km/h and AR21 4.25887°C / 7.04400 km/h,
with 68,417 pairs per variable. AR selection loss recovered from the epoch-20
spike to 0.258288517 but did not beat selected epoch15; direct best remains
epoch1. No new RMSE diagnostic is due. Updated loss chart:
`temporal_loss_direct_28_autoregressive_21.png`. The scored forecast chart
continues to describe direct1 / AR15, not the newer saved weights.

Direct continued at 14:28 PST in `fc-01M7RCCSA2KRWBMMXHVKCTCWZC` and
autoregressive at 15:33 PST in `fc-01M7RG3MXWPKFYH881EBS42RW8`, each
resuming the saved run toward epoch 10. Follow later continuation calls rather
than launching duplicate training. At 15:54 PST, direct had saved 8 epochs
and AR 5; selected checkpoints were direct epoch 1 and AR epoch 4.

Launched 2026-11-18 UTC (2026-11-17 local time):

- Ingestion coordinator: `fc-01M7P90ED4543K62M9FR2WPDW9`.
- Current training coordinator: `fc-01M7QZJDQR997XQCVMBA178KSR`. The earlier `fc-01M7QVS7WFTK78T2VM2N7HN7Z6` launched the first trial; that trial's stage `fc-01M7QVVXMD8RWA4N2BB0TVY42E` was cancelled for the QC repair. The original waiting gate `fc-01M7P90EN6609CJHNXM33SWDGZ` is also cancelled.
- All seven data/evaluation tests and six author-architecture tests passed in remote smoke call `fc-01M7P8Y0JQSVQ4P76VMRA5BSV8`.
- Real GOES-16 January 2018 and GOES-19 July 2025 sample pairs passed shape/finite-value checks in call `fc-01M7P8Y0T3YPVX9ZEZXY2QWQVR`.
- The earlier ingestion attempt was cancelled after detecting historical S3 backfill timestamps; the active run uses the documented original-production cutoff and pixel-quality policy.

```bash
modal deploy models/hyperds_full/modal_multiyear.py
python -c 'import modal; print(modal.Function.from_name("weatherloo-hyperds-multiyear", "ingest").spawn().object_id)'
# Start run_training with that ingest call ID; it waits for the complete audit.
python -c 'import modal; print(modal.Function.from_name("weatherloo-hyperds-multiyear", "status").remote())'
modal app logs weatherloo-hyperds-multiyear --tail 30
```

The local/remote smoke suite checks calendar counts, leap years and split boundaries, backfilled GOES provenance with future-production rejection, monthly shard indexing, Waterloo checkpoint selection, exact benchmark truth, wind conversion, missing lead slots, matched sample masks and pooled RMSE. It also runs the six architecture/gradient tests against the pinned author implementation.

### Observation repair and completed ingestion

All 96 months completed on November 18: 8,688 training, 1,439 validation and
1,457 test initializations (11,584 accepted, 88 rejected). The Soulis 2020 archive
uses a Date/Time layout, which the original parser skipped. Both parsers now
retain each metric independently. The missing 4.4 m wind remains masked; 2 m
wind is never substituted. The canonical 2020 temperature series was rebuilt
from existing raw files, without another download.

`repair_observations` writes a checksummed
`processed/observations/soulis_hourly_v2.npz` label overlay, preserving immutable
raw files and weather shards. The root manifest records the overlay and updated
label counts. The replacement training gate repairs observations and reaudits
before launching the unchanged 50-epoch experiment. Labels after 2025 remain
masked in the final test.

The repaired audit passed at 09:39 PST. Main paper training call
`fc-01M7QVVXMD8RWA4N2BB0TVY42E` started on a Tesla T4 at 09:40 PST and
completed its first optimizer step. Corrected training-label counts are 319,975
Soulis temperatures and 248,826 Soulis wind measurements across lead hours.
The parser's three regression checks, eight data/evaluation checks (including
overlay immutability and the 2025 cutoff), six architecture checks and three
feasibility checks passed after the repair.

### Station QC correction, November 18 at 10:44 PST

Review of the 2021 feasibility comparisons exposed corrupted 2018/2019 station
rows. The raw 2018 archive contains average-wind values above 1,000 km/h, and
the 2019 archive contains temperatures up to 355°C. The v2 label repair had
accidentally omitted the original hourly-cache range filters. This was a
regression in the overlay builder, in addition to the original archive issues.

Both parsers now independently mask temperature outside [-80,60]°C and wind
outside [0,360] km/h, matching the existing cache bounds. Average wind above
a valid same-height gust by more than the 0.02 km/h rounding tolerance is also
masked. No replacement height, clipping or imputed observations are used.
The v3 overlay reapplies the original range filters and its audit rejects
violations. Six parser regression checks cover both layouts, sensor height,
range filtering and average/gust consistency. Original root manifests are
archived under `manifest_history/`; immutable shards and v2 overlays remain.

The corrected 2018 canonical observations contain 1,437 temperatures and
1,296 winds (maximum 22.31 km/h); corrected 2019 contains 444 temperatures and
448 winds. These were regenerated from cached raw files. 2025 final benchmark
truth is unchanged. The clean run starts from scratch in its own checkpoint
directory and requests L4/A10 GPUs, avoiding the slower T4 used by the first
attempt. The separate feasibility run is likewise repeated on its original
frozen cohort with v3 labels.

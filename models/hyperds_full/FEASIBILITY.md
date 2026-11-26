# Stopped feasibility experiment (historical)

**Stopped at the user's request on November 18, 2026, at 11:21 PST.** All 96
months are downloaded, so the smaller experiment is no longer needed. The
dedicated Modal app `weatherloo-hyperds-feasibility`
(`ap-jiu77vpCIEIXCKoH3fX1ch`) was stopped, including its training and evaluation
containers. Saved checkpoints, metrics, predictions and source manifests are
retained remotely. Do not restart it unless the user asks. Only the full
multi-year experiment continues; feasibility completion is no longer a gate.

**Stopped clean run:** `feasibility_2018_2020_stationqc_v2`. It repeats the exact
1,430/711/1,088 frozen 2018/2019/2020 rows and 2018 normalization from the
original `feasibility_2018_2020_v1` manifest, with the new checksummed Soulis
v3 labels. Later completed months are not added to training or selection.
The original run's weights, scores and exports remain remote for provenance.
All completion claims and scores below describe that earlier run until clean
results are explicitly marked as such.

Clean coordinator `fc-01M7QZPZ0SH9PT8QPHE3K7F9ED` completed preparation at
10:48:51 PST and launched paper training `fc-01M7QZV1C8NQQWARRWEZDSAYMJ`.
Preparation validated the unchanged cohort and v3 overlay checksum, rebuilt
the three baselines from corrected 2018 truth, and froze the 1,450-example
2021 holdout before fitting.

The first model scores led to discovery of corrupted station rows (including
2018 average wind above the recorded gust) and a v2 overlay regression that
skipped existing range QC. Both parsers and the overlay builder are repaired;
the same architecture and ten-epoch experiment are being rerun from scratch.
Treat earlier results as diagnostic measurements, not evidence of clean-model
forecasting skill. See MULTIYEAR.md for the exact QC and recovery.

This independent experiment starts training while the full 2018–2025 download continues. It uses the same released HyperDS architecture and shared training/evaluation functions. Its frozen snapshot references completed, verified shards in the remote Volume; bulk data is neither copied nor downloaded to the laptop.

## Frozen data

| Purpose | Year | Initializations | Coverage |
| --- | --- | ---: | --- |
| Training | 2018 | 1,430 | All 12 months |
| Waterloo checkpoint selection | 2019 | 711 | February, March, June, August, September, December |
| Waterloo feasibility test | 2020 | 1,088 | January–April, June, August, October–December |

The snapshot was frozen before model fitting. Later completed shards are not added during the run. Training and validation initializations whose +48 h targets cross into the next year are excluded. Field/satellite normalization uses 2018 source moments only. Soulis is the primary selection and test station; Toronto is secondary. All available regional station labels are used for fitting. The main experiment's 2024 validation and 2025 final test are not used here.

## Models and comparisons

- Ten epochs of single-time paper HyperDS from scratch on 2018. The earlier January model is not reused because it was trained on years later than this experiment's training split.
- Direct and autoregressive temporal models each train for ten epochs from that paper model's best 2019 checkpoint. Both predict hourly through +48 h, retain the full observation encoder/fusion and grid/station losses, and use the same data and optimizer settings. Their GPU jobs run in parallel.
- Per-epoch checkpoints, optimizer/scheduler/RNG states, source hashes, frozen source-manifest hashes and Waterloo validation metrics are saved remotely. Interrupted jobs resume their own latest checkpoint.
- Best checkpoints are tested on the frozen 2020 examples. Comparison uses the site's canonical 2020 observations, °C and km/h, at leads +6/+12/…/+48 h, on identical finite forecast/observation pairs.
- References are observed persistence, 2018-only DOY/hour seasonal climatology, and persistence of the exact coarse ERA5 initialization field supplied to the model. Persistence/climatology reuse the existing repository's observation and climatology helpers. RMSE is pooled from squared errors; MAE, bias, counts and per-lead results are retained.

The feasibility question is whether training stays stable through the full rollout, reduces held-out error, and improves on these simple references at Waterloo. Beating them would support further investigation; comparison with the site's NWP/AI forecast methods belongs to the full 2025 experiment. Ten epochs on one training year do not establish the full method's final forecasting skill. ERA5 initialization remains retrospective.

## Remote execution

[Feasibility app](https://modal.com/apps/cindehaa/main/deployed/weatherloo-hyperds-feasibility)

Coordinator: `fc-01M7PTN5HZRC7NXYDBWXC3V3YZ`. All 16 smoke checks passed: three feasibility checks, seven data/benchmark checks, and six architecture checks against the pinned author implementation.

```text
weatherloo-temporal-checkpoints:/hyperds_full/feasibility_2018_2020_v1/
  manifest.json, audit.json
  baselines/{persistence,climatology_2018,era5_initial_persistence}_2020.npz
  {paper,direct,autoregressive}/
    latest.pt, best.pt, metrics.jsonl
    evaluation.json, predictions.npz
    waterloo_summary.json                       # temporal modes
    benchmark/hyperds_<mode>_feasibility/
      hyperds_<mode>_feasibility_2020.npz
      metadata.json, comparison.json
```

```bash
modal deploy models/hyperds_full/modal_feasibility.py
python -c 'import modal; print(modal.Function.from_name("weatherloo-hyperds-feasibility", "status").remote())'
modal function calls weatherloo-hyperds-feasibility/train --tail 3 --json
modal app logs weatherloo-hyperds-feasibility --tail 20
```

The full multi-year coordinator, source shards and main checkpoint directories remain separate. These 2020 feasibility results are not published as the final 2025 benchmark.

## Completed training and corrected evaluation

All three models completed 10/10 epochs. During comparison, an existing Soulis
parser bug was found: its 2020 Date/Time CSV was skipped. Temperature was
recovered from the cached raw file; the actual 4.4 m wind sensor is missing for
2020. The 2 m sensor is not substituted. Training (2018) and checkpoint selection
(2019) use the legacy layout and do not require retraining.

Correction coordinator `fc-01M7QVSHCWWQZXNHPAD3T726T3` rescored the saved 2020
forecasts against corrected canonical temperature. It also freezes all 12
completed 2021 months as an additional held-out test, chosen for Soulis wind
coverage (1,444/1,460 canonical slots), before reading model errors. The 2018
normalization and 2019 best checkpoints stay fixed. This additional evaluation
performs zero optimizer steps and leaves the 2025 final test untouched.

Corrected reports live under `evaluation_v2/{2020,2021}/` in the same remote run
directory. The original incomplete reports remain for provenance. Use these
corrected reports for conclusions; the original 2020 hourly Soulis metrics had
almost no valid labels and are unsuitable for judging forecasting skill.

### Corrected 2020 Soulis temperature results

Read at 10:14 PST on November 18 from the remote `evaluation_v2/2020/`
summaries. These pool leads +6/+12/…/+48 h over the frozen nine-month test
snapshot; all errors are in °C. Each comparison matches finite pairs exactly.

| Method | MAE | RMSE | Forecast/observation pairs |
| --- | ---: | ---: | ---: |
| Direct temporal HyperDS | 4.0983 | 5.3189 | 8,668 |
| Autoregressive temporal HyperDS | 4.4791 | 5.7624 | 8,668 |
| 2018-only seasonal climatology | 3.9585 | 5.1488 | 8,668 |
| Coarse ERA5 initialization persistence | 4.5286 | 5.8405 | 8,668 |
| Observed persistence | 4.6479 | 6.0792 | 8,660 |

On the observed-persistence subset of 8,660 pairs, direct MAE/RMSE is
4.0992/5.3198 and autoregressive is 4.4803/5.7641. Both improve on persistence,
but neither improves on the seasonal climatology reference. Direct is ahead
of autoregressive on this test. Training is computationally feasible; these
results do not establish superior forecasting skill. Initialization is
retrospective ERA5. No valid 2020 Soulis 4.4 m wind score is available.

The unchanged selected checkpoints are now evaluating on 1,450 accepted 2021
initializations: direct call `fc-01M7QXFR26C3E0FRB9ZBS22Y6V` and autoregressive
call `fc-01M7QXFR5BHEZFZF28RKNDGZCK`. The 2025 final comparison remains separate.

### Original run: 2021 direct result and observation issue

The original direct model's matched 2021 Soulis temperature MAE/RMSE was
4.3639/5.5262°C over 11,446 pairs, versus seasonal climatology
4.2159/5.4126 and ERA5 initialization persistence 4.5382/5.8342. On the 11,325
observed-persistence pairs it scored 4.3617/5.5208 versus 4.7977/6.2318.
Wind MAE/RMSE was 4.3609/5.7102 km/h versus observed persistence 3.2662/4.5672
on 11,277 pairs. Temperature improved on persistence; wind did not. The extreme
2018-climatology wind errors triggered the raw-data audit and QC repair. They
are not meaningful wind-baseline evidence. The clean rerun regenerates the
2018 climatology from corrected canonical observations and preserves the same
2021 evaluation cohort, without using test errors to alter the model.

The clean run's prepare step freezes the 2021 manifest and baselines before
training. Each temporal training job automatically launches its own 2021
evaluation after testing on 2020. Main 2025 evaluation remains untouched.

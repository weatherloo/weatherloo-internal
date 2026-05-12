# January-February 2026 Evaluation

This report template documents the v1 Waterloo HRRR post-processing run.

## Scope

- variable: `2 m temperature`
- loss: station supervision only, plus coarse consistency and smoothness regularization
- lead times: shared model over `1-18 h`
- train: most of January 2026
- validation: last 5 days of January 2026
- test: February 2026

## Required outputs

The training script writes the following artifacts into the chosen output
directory:

- `loss_curves.png`
- `training_history.csv`
- `best_model.pt`
- `train_station_predictions.csv`
- `validation_station_predictions.csv`
- `test_station_predictions.csv`
- `*_metrics.json`
- `*_raw_nearest_by_lead.csv`
- `*_raw_bilinear_by_lead.csv`
- `*_bias_baseline_by_lead.csv`
- `*_model_by_lead.csv`

## Metrics to review

Primary station metrics:

- MAE
- RMSE
- mean bias
- the same metrics broken out by `lead_hour`

Compare:

1. raw HRRR nearest-grid value
2. raw HRRR bilinear interpolation
3. pooled bias baseline
4. dense-field model sampled at station coordinates

## Qualitative dense-field checks

The v1 dense field should be treated as qualitative. For a few representative
cases, inspect:

- raw `t2m`
- learned residual `delta_T`
- corrected `t2m`

Look specifically for:

- broad, smooth local corrections rather than pixel noise
- corrections that remain anchored to HRRR structure
- no obvious checkerboard artifacts

## Success bar

The run is successful if, on February 2026 station verification:

- the model improves over raw HRRR bilinear interpolation on MAE and/or RMSE
- mean bias is reduced materially
- residual maps are spatially plausible


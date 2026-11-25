# Full HyperDS with time and hourly rollout

This implementation imports the authors' actual ResNet encoders, observation/field Transformer fusion, and five generated-weight variable decoders from [ESDALab/HyperDS_tgrs](https://github.com/ESDALab/HyperDS_tgrs), pinned at `1fe6d0b08e759548655009e1747428f9136a7839`. The upstream code is cloned inside the remote image, not copied into this repository. The model from [arXiv:2401.11960](https://arxiv.org/abs/2401.11960) is the foundation; `models/temporal_hyperds/` holds the earlier, incomplete field-only ablation.

**Current objective:** compare direct and autoregressive **weather forecasts at Eric D. Soulis in Waterloo** against the existing benchmarking-site methods. The active multi-year workflow uses **2018–2023 for training, 2024 for Waterloo validation/checkpoint selection, and 2025 for final Waterloo testing**, at all four daily cycles. See [MULTIYEAR.md](MULTIYEAR.md) for the frozen experiment, remote jobs, and benchmark contract. The January results below are a completed pipeline pilot and do not establish forecasting skill.

**Site forecasts (current best at Waterloo):** [PIPELINE.md](PIPELINE.md) is the region-configurable HRRR station-correction pipeline. It uses one model type, a CNN on the native HRRR patch plus the station's 0–48 h trajectory, and runs with `modal_region.py`. `modal_points.py` keeps the experiments that led to it. The same CNN forecasts Soulis precipitation (hourly and 6-hour chance of precipitation, and 6 h and 24 h amounts) with `precip.py`.

**Paper-faithful HRRR replication:** [FAITHFUL.md](FAITHFUL.md) (paper protocol and baselines plus the time dimension). **HRRR/AIFS-input variant:** [NWP.md](NWP.md) replaces the ERA5 initialization input with the issued forecast trajectory; it supersedes the ERA5-init direct/AR runs, which were stopped on 2026-11-19 at 28/50 and 21/50 epochs (checkpoints kept).

An independent [feasibility run on completed data](FEASIBILITY.md) uses 2018 training, 2019 Waterloo validation and 2020 Waterloo testing while the full archive downloads. It trains both temporal extensions and compares them with observed persistence, seasonal climatology and initialization-field persistence.

## Paper components and implementation

| Component | Implementation |
| --- | --- |
| Five surface fields | `u10`, `v10`, `t2m`, surface pressure, hourly precipitation, in canonical order |
| Coarse field input | ERA5 0.25° field at initialization, average-pooled by 4 to 1° |
| Observation input | Two real pre-initialization GOES-16 L1b frames, four bands |
| Dual encoders | Author ResNet-18 field encoder and shared satellite encoder, block4 features |
| Implicit retrieval | Author satellite self-attention and field-query cross-attention, 64 channels and 8 heads |
| Weight generation and decoding | Author MULTI_VAR `VariableNet` for each of five variables; generated shallow weights and shared deeper weights |
| Decoder auxiliary input | Interpolated meteorological field value plus sinusoidal coordinate embedding |
| Subgrid sampling | Ten points per cell, with training station coordinates inserted in their cells; average the ten predictions before grid loss |
| Grid supervision | MSE against the fine 0.25° ERA5 labels |
| Station supervision | Masked MSE for temperature, wind speed, station pressure and precipitation where available; physical wind speed derived from u/v |
| Total loss | Grid MSE + 0.05 × station MSE |
| Optimization | Adam, initial learning rate 0.0001, cosine cycle of 5 epochs, 50 epochs, batch size 4 |
| Checkpoint selection | Multi-year run: lowest Soulis station MSE on 2024; January pilot: Elora |

The primary MULTI_VAR architecture is one of the paper's two alternative decoders. The released MULTI_BLOCK alternative and all benchmark/ablation runs are not part of this first full-model training job.

## Temporal extension

`paper` mode reproduces single-time downscaling: a coarse initialization field and two satellite frames predict station/grid values at that same time. `direct` mode adds hour to the coordinate representation and queries +0 through +48 h from that initialization. `autoregressive` mode decodes the next hour, averages predicted samples into a regional field, and uses that predicted field as the next input. Satellite observations remain the same pre-initialization pair throughout the rollout. No future ERA5 field or future station label is used as forecast input. At hour zero the direct coordinate embedding is exactly the paper embedding.

The two temporal models start from the trained `paper` checkpoint. Every training rollout covers 49 hourly leads, with gradients truncated every six hours. Validation uses the model's own predictions for the full rollout. Grid cells are sampled uniformly in minibatches of 128 per hour to bound memory, retaining ten interior samples per chosen cell. Feedback uses a separate ten-point mean in every coarse cell. These computational adaptations are recorded in checkpoints.

## January pilot data (completed)

This is a Waterloo adaptation of the full method, not a replication of the paper's numerical China benchmark. The original benchmark uses Himawari and Weather2K; Ontario uses GOES and local public stations.

- Training: 28 initialization cycles, January 2–15, 2021, at 00Z/12Z. Train stations: Soulis and Kitchener/Waterloo.
- Validation: 10 initialization cycles, January 2–6, 2024, at 00Z/12Z. Spatial validation station: Elora. Guelph is excluded from fitting and checkpoint selection and evaluated only at the end.
- ERA5 region: 40–47.75°N and 88–72.25°W, 32×64 fine cells. Targets include all 49 leads; normalization uses unique training hours only.
- GOES bands: C02 visible, C05 near infrared, C08 water vapor, C15 thermal infrared, mapped to the paper's four spectral roles. Visible/near-infrared radiances are converted to reflectance; infrared radiances to brightness temperature using file calibration coefficients. The frames are separated by at least nine minutes. Scan completion, creation and object publication must all precede initialization. The 00Z/12Z winter sample is predominantly dark, limiting visible-band information.
- Station labels: wind speed in m/s, temperature in °C, station pressure in hPa, hourly precipitation in mm. Missing station variables remain masked; they are never filled from ERA5. Soulis supplies temperature/wind, with pressure/precipitation learned from grid and available ECCC labels.
- This is a retrospective experiment initialized from ERA5. ERA5 assimilation and delayed publication prevent a claim of real-time operational availability.

`data.validate_store()` checks shapes, finite input fields, SHA256 values, temperature/pressure units, satellite publication cutoffs, chronological separation, and station label coverage before training.

The upstream `INF_Net.forward_single` concatenates pressure and temperature in reverse order. This wrapper calls the same variable decoders directly and preserves explicit `u,v,t,p,tp` order. Wind loss is computed after denormalizing u/v, avoiding a speed calculation on standardized components.

## Remote execution

Bulk source reads, transformation and training run on Modal. The laptop retains code and small diagnostics. Processed data and resumable checkpoints use the existing data and checkpoint Volumes:

```text
weatherloo-temporal-data:/processed/hyperds_full/waterloo_era5_goes_v1
weatherloo-temporal-checkpoints:/hyperds_full/waterloo_era5_goes_v1/{paper,direct,autoregressive}/
```

Each epoch saves `latest.pt` with optimizer/scheduler state, `best.pt`, and `metrics.jsonl`, then commits the checkpoint Volume. Completed modes also write `evaluation.json`. Source hashes, the pinned upstream revision, normalization, input provenance, splits and training settings are stored with checkpoints.

### January pilot results

All three modes completed 50 epochs on 2026-11-17. The single-time model ran on an NVIDIA A10. Both extensions started from its best checkpoint: direct hourly prediction on an A10 and autoregressive hourly prediction on an L4. All training calls and the durable coordinator reported success. Each mode has committed `best.pt`, `latest.pt`, `metrics.jsonl` and `evaluation.json`. Best checkpoints were selected at epochs 49 (paper), 41 (direct) and 49 (autoregressive), using only Elora station loss.

Final held-out Guelph MAE across leads +1 through +48 h:

| Model | Temperature (°C) | Wind speed (m/s) | Station pressure (hPa) | Hourly precipitation (mm) |
| --- | ---: | ---: | ---: | ---: |
| Direct | 1.878 | 1.356 | 9.225 | 0.0569 |
| Autoregressive | 1.879 | 1.203 | 6.555 | 0.0406 |

Each temporal score covers 480 initialization/lead pairs from ten validation initializations; overlapping forecasts are not independent observations. The single-time paper model's Guelph temperature MAE was 0.935°C at initialization only, which is a different task from the forecasting rows above. These are initial results on the small January dataset described above. Persistence comparisons, seasonal coverage and uncertainty estimates remain necessary to establish forecasting skill.

- [Modal app and logs](https://modal.com/apps/cindehaa/main/deployed/weatherloo-hyperds-full)
- Durable experiment call: `fc-01M7NXEFA47JXSQG1QVBXSPVM6`
- Completed paper call: `fc-01M7NXG0SB9D2XBF6ACQKGHV2H`
- Direct call: `fc-01M7NXPP1Y4E58BKQ4BE8DPVR9`
- Autoregressive call: `fc-01M7NXPP3VP6EVT6KTDMMXW6C2`

The remote CPU smoke checks and all six model tests passed before these runs. `data.validate_store()` passed all seven audit groups before each training mode. The coordinator ran independently of the laptop and waited for both extensions to complete.

```bash
modal deploy models/hyperds_full/modal_app.py
python -c 'import modal; print(modal.Function.from_name("weatherloo-hyperds-full", "run_experiment").spawn().object_id)'
modal app logs weatherloo-hyperds-full --tail 100
modal function calls weatherloo-hyperds-full/train --json
```

The CPU smoke function runs model gradient checks and tests for parity with the upstream fusion forward pass, satellite dependence, variable ordering/units, missing-label masks, subgrid means, and temporal/query invariants:

```bash
modal run models/hyperds_full/modal_app.py::smoke
```

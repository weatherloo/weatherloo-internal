# Hourly direct versus autoregressive baseline

This directory is a fixed-station comparison baseline. The paper-based implementation is in [`models/temporal_hyperds/`](../temporal_hyperds/README.md).

This first experiment reuses `src/hrrr_bias_correction/make_data.py`'s Zarr store. Each sample contains a 49-hour HRRR crop and Soulis station-bias labels. Both models receive the same HRRR forecast available at initialization. The direct model predicts lead biases in parallel; the autoregressive model feeds its **own predicted bias** into the next hourly step. Neither reads future observations as input.

This is a **fixed-station temporal baseline**, not yet a HyperDS-style arbitrary-location model. The existing store contains only Soulis labels. Resolve the station-coordinate discrepancy documented in `docs/lit-review/temporal-hyperds-plan.md` and build a multi-station dataset before claiming spatial generalization.

From the repository root:

```bash
python pipelines/download/audit_hourly_training_data.py
python -m pip install -r models/temporal_station/requirements.txt
python models/temporal_station/train.py --smoke
python models/temporal_station/train.py --model direct --max-samples 100
python models/temporal_station/train.py --model autoregressive --max-samples 100
```

The last two commands require the processed Zarr store under `WEATHERLOO_DATA_ROOT/processed/hrrr_bias_correction/hrrr`, including `train` and `val` groups. `--store` overrides it. Checkpoints go under `WEATHERLOO_DATA_ROOT/checkpoints/temporal_station/` by default. Validation output prints raw versus corrected MAE for temperature (°C) and wind speed (km/h). `--max-samples` is for a smoke run, not a final result. The audit counts local HRRR files but does not verify lead coverage or publication latency.

The Modal pilot downloads only the 2021-01-02 and 2024-01-02 00Z HRRR cycles to **remote temporary disk**. It fetches the matching UW station archives there, runs the existing tensor builder, verifies one train and one validation sample, and commits only the processed Zarr store and source checksums to `weatherloo-temporal-data:/processed/hrrr_bias_correction/hrrr`. No bulk HRRR data is saved on the laptop or in the Volume. The pilot leaves the existing `weatherloo-hrrr-data` Volume untouched.

From the repository root, after [setting up Modal](https://modal.com/docs/guide) and checking the active profile:

```bash
modal volume create --version=2 weatherloo-temporal-data
modal volume create --version=2 weatherloo-temporal-checkpoints
modal run models/temporal_station/modal_app.py::ingest
modal volume ls weatherloo-temporal-data /processed/hrrr_bias_correction/hrrr
```

The ingest refuses to overwrite an existing processed store. After inspecting the pilot manifest, the optional GPU smoke runs are:

```bash
modal run models/temporal_station/modal_app.py::main --model direct --max-samples 1 --epochs 2
modal run models/temporal_station/modal_app.py::main --model autoregressive --max-samples 1 --epochs 2
```

One sample per split verifies plumbing only; it cannot establish forecast skill. The runner prefers L4, then A10 or T4 when capacity is scarce, and writes checkpoints to `weatherloo-temporal-checkpoints:/temporal_station`.

The planned baseline run uses 00Z and 12Z cycles from January 2–15, 2021 (up to 28 training initializations) and January 2–6, 2024 (up to 10 validation initializations). It would build a separate remote store at `weatherloo-temporal-data:/processed/hrrr_bias_correction/hrrr_jan_2021_2024_00_12`, then train each baseline for 20 epochs. That run was canceled during ingest when the project objective was corrected toward HyperDS; it produced no new dataset or checkpoints. The existing two-sample pilot stays intact. The following commands are for a future baseline comparison only:

```bash
modal deploy models/temporal_station/modal_app.py
python -c 'import modal; f = modal.Function.from_name("weatherloo-temporal-station", "train_from_new_data"); print(f.spawn().object_id)'
```

The second command starts a durable background call and prints its Function Call ID. Inspect progress with `modal app logs weatherloo-temporal-station --tail 100` and `modal function calls weatherloo-temporal-station/train_from_new_data --json`. The earlier ephemeral `modal run --detach` launch was canceled during ingest when its local launcher exited; use the deployed App for this run.

This is an initial winter-only experiment, not a seasonal or spatial generalization result. New tensors use the [station operator's published coordinates](https://www.civil.uwaterloo.ca/weatherstation/station-information/); the earlier pilot used the repository's previous coordinates, so its errors are not directly comparable. See `docs/lit-review/temporal-hyperds-plan.md` for the remaining audit and model plan, and Modal's [Volume guide](https://modal.com/docs/guide/volumes) and [GPU guide](https://modal.com/docs/guide/gpu) for storage and compute behavior.

# HyperDS and the HRRR region pipeline: agent guide

This directory holds two tracks. Start with [README.md](README.md).

- **Region pipeline (production site forecasts).** One model type, a CNN on the native HRRR patch plus the station's 0–48 h trajectory. It forecasts temperature, wind and precipitation at configured stations. See [PIPELINE.md](PIPELINE.md).
- **HyperDS (research).** The paper-faithful replication and its HRRR/AIFS-input temporal variants. See [FAITHFUL.md](FAITHFUL.md) and [NWP.md](NWP.md).

## Where things run

- **Modal.** All bulk reads and training run there, using the `weatherloo-temporal-data` and `weatherloo-temporal-checkpoints` Volumes. Nothing bulk is written to the repository.
- **Commands.** `modal run` commands take a path from the repository root, for example `modal run models/hyperds_full/modal_region.py --region waterloo`. With no flags this only plans: it validates the config and lists pending stages without spending compute. Add `--execute` to run.
- **Before spending compute.** Ask before extraction over all months, any `--execute` that trains, or new ablations. They cost GPU or CPU time on the shared Modal account.

## Rules

- **Truth.** Use the site's station IDs, coordinates and units exactly as `benchmarking-site/AGENTS.md` lists them, in UTC.
- **Splits.** Training covers 2020-12-03 through 2023, with targets before 2024. 2024 is for early stopping and design choices. 2025 is scored once. Never choose anything on 2025.
- **Inputs.** Use only information available at issue time: the run, earlier runs, and observations at or before T0. Extend the leakage tests in `test_mos.py` and `test_precip.py` when adding inputs.
- **Patches.** Patches are stored as float16. Store new fields in units that fit float16; see `points.STORAGE_UNITS`.
- **Publishing.** The pipeline's site export stays on the data Volume. Publishing to `benchmarking-site/data/<method>/` belongs in `pipelines/benchmarks/methods/<method>/`, and needs the team's go-ahead.

## Validation

Run from this directory:

```bash
python -m unittest test_mos test_region test_cnn test_precip test_multiyear test_feasibility test_soulis
```

- `test_mos` needs LightGBM.
- `test_cnn` and `test_precip` need torch.
- `test_multiyear`, `test_feasibility` and `test_soulis` need xarray and zarr.
- `test_model`, `test_nwp` and `test_faithful` need the pinned upstream HyperDS clone. They run in the Modal image: `modal run models/hyperds_full/modal_faithful.py::smoke`.

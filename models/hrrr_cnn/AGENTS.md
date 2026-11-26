# HRRR station correction (CNN): agent guide

[README.md](README.md) has the pipeline, method and results. This model corrects HRRR at stations with one model type, a CNN on the native HRRR patch plus the station's 0–48 h trajectory. It forecasts temperature, wind and precipitation. It is not HyperDS; that lives in `models/hyperds_full/`.

## Where things run

- **Modal.** All bulk reads and training run there, using the `weatherloo-temporal-data` Volume under `/processed/`. Nothing bulk is written to the repository.
- **Commands.** `modal run` commands take a path from the repository root, for example `modal run models/hrrr_cnn/modal_region.py --region waterloo`. With no flags this only plans: it validates the config and lists pending stages without spending compute. Add `--execute` to run.
- **Before spending compute.** Ask before an `--execute` that extracts or trains, new ablations, or a new region. They cost GPU or CPU time on the shared Modal account.

## Rules

- **Truth.** Use the site's station IDs, coordinates and units exactly as `benchmarking-site/AGENTS.md` lists them, in UTC.
- **Splits.** Training covers 2020-12-03 through 2023, with targets before 2024. 2024 is for early stopping and design choices. 2025 is scored once. Never choose anything on 2025.
- **Inputs.** Use only information available at issue time: the run, earlier runs, and observations at or before T0. Extend the leakage tests in `test_mos.py` and `test_precip.py` when adding inputs.
- **Model type.** Region configs accept only the CNN. The gradient-boosting code is an experiment record.
- **Patches.** Patches are stored as float16. Store new fields in units that fit float16; see `points.STORAGE_UNITS`.
- **Publishing.** The site export stays on the data Volume. Publishing to `benchmarking-site/data/<method>/` belongs in `pipelines/benchmarks/methods/<method>/`, and needs the team's go-ahead.

## Validation

Run from this directory:

```bash
python -m unittest test_mos test_region test_cnn test_precip
```

`test_mos` needs LightGBM; `test_cnn` and `test_precip` need torch.

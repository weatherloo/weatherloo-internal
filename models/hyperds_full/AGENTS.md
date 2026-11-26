# HyperDS: agent guide

This directory is the HyperDS research track. It holds the paper-faithful replication ([FAITHFUL.md](FAITHFUL.md)) and its time-extended and HRRR/AIFS-input variants ([NWP.md](NWP.md), [MULTIYEAR.md](MULTIYEAR.md), [FEASIBILITY.md](FEASIBILITY.md)). Start with [README.md](README.md). The production site-forecast model (CNN) is in `models/hrrr_cnn/`.

## Where things run

- **Modal.** All bulk reads and training run there, using the `weatherloo-temporal-data` and `weatherloo-temporal-checkpoints` Volumes. `modal_app.py` defines the images and copies in the shared modules `pipelines/download/station_labels.py` and `pipelines/benchmarks/site_scoring.py`.
- **Dependencies.** `requirements.txt` is what the training image installs. `requirements-modal.txt` has the launcher and catalog packages.
- **Upstream modules.** The authors' modules are cloned at a pinned revision inside the image, not vendored.
- **Before spending compute.** Ask before training runs or large extractions.

## Rules

- **Truth.** Use the site's station IDs, coordinates and units exactly as `benchmarking-site/AGENTS.md` lists them, in UTC.
- **Splits.** Training covers 2020-12-03 through 2023, with 2024 for validation and checkpoint selection and 2025 for testing. Station splits are fixed in `station_labels.split_stations`.

## Validation

Run from this directory:

```bash
python -m unittest test_multiyear test_feasibility test_soulis
```

- Those tests need xarray and zarr.
- `test_model`, `test_nwp` and `test_faithful` need the pinned upstream clone. They run in the Modal image: `modal run models/hyperds_full/modal_faithful.py::smoke`.

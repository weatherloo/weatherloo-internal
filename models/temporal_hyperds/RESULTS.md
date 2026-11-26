# First temporal HyperDS training results

Both 20-epoch GPU runs completed successfully on Modal on November 17, 2026. The data build retained 28 training initializations from January 2–15, 2021 and 10 validation initializations from January 2–6, 2024, at 00Z and 12Z. Validation covers leads +1 to +48 h. Overlapping forecasts contribute separate forecast/lead pairs.

The model is the field-only temporal HyperDS prototype described in README.md. It generates coordinate-decoder weights from HRRR fields, includes forecast hour, and compares direct and autoregressive corrections. Satellite fusion and grid supervision are not implemented.

| Metric (MAE) | Raw HRRR | Direct | Autoregressive |
| --- | ---: | ---: | ---: |
| Seen stations temperature (°C; weighted by pairs) | 0.944 | 0.715 | 0.684 |
| Soulis temperature (°C) | 1.002 | 0.679 | 0.614 |
| Soulis wind speed (km/h) | 9.622 | 2.791 | 5.849 |
| Elora temperature (°C) | 0.933 | 0.790 | 0.709 |
| Elora wind speed (km/h) | 4.731 | 4.448 | 4.922 |
| Kitchener/Waterloo temperature (°C) | 0.899 | 0.676 | 0.725 |
| Kitchener/Waterloo wind speed (km/h) | 3.770 | 5.083 | 5.634 |
| Guelph (held out) temperature (°C) | 0.759 | 0.642 | 0.706 |
| Guelph (held out) wind speed (km/h) | 5.085 | 7.746 | 3.707 |

Checkpoints were selected by the lowest temperature MAE across Soulis, Elora and Kitchener/Waterloo on the chronological validation window: direct epoch 7, autoregressive epoch 3. Guelph labels were excluded from fitting and checkpoint selection. These are validation scores from a small winter pilot; there is no independent chronological test result yet. Wind skill is mixed, including worse direct-model errors at Guelph and worse errors for both models at Kitchener/Waterloo.

The checkpoint files were loaded and their model.py/train.py SHA256 values matched the current source. Bulk data and the authoritative checkpoints remain on Modal.

- App: [weatherloo-temporal-hyperds](https://modal.com/apps/cindehaa/main/deployed/weatherloo-temporal-hyperds)
- Workflow call: `fc-01M7MF3PG4YG3ZF3F9Y8AFHT2R`
- Data Volume: `weatherloo-temporal-data:/processed/temporal_hyperds/jan_2021_2024_multistation`
- Checkpoints: `weatherloo-temporal-checkpoints:/temporal_hyperds/jan_2021_2024_multistation/{direct,autoregressive}.pt`

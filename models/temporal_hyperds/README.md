# Temporal HyperDS prototype

This is the earlier field-only ablation. The complete author-module implementation, including satellite fusion and grid/station losses, is in [`models/hyperds_full/`](../hyperds_full/README.md).

Both initial 20-epoch GPU training runs completed successfully. See [the saved-checkpoint validation results](RESULTS.md) for metrics and remote artifact locations.

The [HyperDS paper](https://arxiv.org/abs/2401.11960) uses a hypernetwork to generate weights for a coordinate-query decoder, supervised by station values. This earlier `TemporalHyperDS` ablation tests that mechanism with forecast lead hour, using simplified encoders. Its results do not evaluate the full paper architecture.

```text
Issued HRRR forecast (f00..f48) -> field encoder -> temporal features
                                                 -> weight generator -> coordinate decoder (x, y, hour)
                                                                          -> temperature/wind correction
Autoregressive mode: previous predicted correction --------------------------^
```

`model.py` currently takes normalized HRRR temperature and wind fields and normalized spatial coordinates. It outputs normalized HRRR-minus-observation corrections at arbitrary queried points for each forecast hour. All 49 HRRR lead fields belong to one issued run; no future observations enter the model. The direct and autoregressive modes share the same field encoder and generated decoder structure. The autoregressive mode also carries a recurrent field state.

The HRRR product publication times still need auditing before interpreting +1 h and later scores as operationally available forecasts.

This is a **field-only HyperDS ablation**, not a reproduction of the full paper: it does not yet encode satellite images. `make_data.py` builds a common 64×64 HRRR crop with hourly labels at Soulis, Elora, Kitchener/Waterloo, and Guelph. Guelph has no January 2021 training labels and is held out during model fitting; January 2024 provides its spatial test labels. Four stations are enough for a pilot, not a broad arbitrary-location generalization claim. Satellite images must be restricted to those available at forecast issuance if added as a second encoder branch.

Run the local shape and rollout checks from the repository root:

```bash
python3 tests/test_temporal_hyperds.py
```

The previous `models/temporal_station/` network and its Modal training app are baselines. Its longer training call was canceled before a new dataset or checkpoint was produced.

The real pilot runs remotely on Modal. It downloads raw forecasts and observations to container scratch, writes only the processed multi-station Zarr store to `weatherloo-temporal-data:/processed/temporal_hyperds/jan_2021_2024_multistation`, then trains direct and autoregressive models on an L4/A10/T4 GPU. Checkpoints go to `weatherloo-temporal-checkpoints:/temporal_hyperds/jan_2021_2024_multistation`.

```bash
modal deploy models/temporal_hyperds/modal_app.py
python -c 'import modal; print(modal.Function.from_name("weatherloo-temporal-hyperds", "run_experiment").spawn().object_id)'
modal function calls weatherloo-temporal-hyperds/run_experiment --json --tail 1
modal app logs weatherloo-temporal-hyperds --tail 100
```

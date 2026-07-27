# U-Net forecast post-processing (residual learning)

A U-Net that **corrects GFS surface forecasts** over southern Ontario by learning
the GFS error against ERA5 reanalysis. Part of the Weatherloo benchmarking project
(see `benchmarking-site/AGENTS.md`).

## Approach

- **Input:** GFS forecast grid (0.25°) for the southern Ontario box — `t2m`, `u10`, `v10`.
- **Target:** ERA5 reanalysis on the same 0.25° grid at the forecast's valid time (ground truth).
- **Learning objective — residual:** the network predicts the **error** `(GFS − ERA5)`,
  not the absolute field. The corrected forecast is then

  ```
  corrected = GFS − predicted_error
  ```

  Residual learning is more stable than regressing absolute values: the network only
  has to model a small, roughly zero-mean correction rather than the full weather signal.

## Region of interest

Southern Ontario bounding box (covers both benchmark stations):

| | value |
|---|---|
| Latitude  | 41.0°N .. 46.0°N |
| Longitude | −84.0°E .. −74.0°E |
| Grid      | 0.25° → **21 × 41** (lat × lon) |

Stations (from `benchmarking-site/AGENTS.md`):

| Station | Lat | Lon |
|---|---|---|
| CYYZ (Toronto Pearson) | 43.6777 | −79.6248 |
| Eric D. Soulis (UW)    | 43.4668 | −80.5164 |

ERA5 and GFS share the same 0.25° / 1440×721 global grid, so the two region slices align cell-for-cell.

## Data sources

| | GFS (input) | ERA5 (target) |
|---|---|---|
| Host | AWS Open Data `s3://noaa-gfs-bdp-pds` (public) | GCP `gs://gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr` (`token="anon"`) |
| Access | byte-range GRIB via `.idx` + `cfgrib` | `xarray` + `gcsfs` (zarr v2) |
| Cycles / cadence | 00/06/12/18Z init | 6-hourly analysis |
| Leads used | f006 .. f048 (6-hourly) | valid time = init + lead |
| Variables | `TMP:2m`, `UGRD:10m`, `VGRD:10m` | `2m_temperature`, `10m_u/v_component_of_wind` |
| Cache | `.cache/gfs_grib/` (gitignored) | none (lazy zarr reads) |

### ⚠️ Usable date range

Although the task brief referenced 2020, the actual overlap of the two archives is **2021**:

- GFS **0.25° pgrb2** on `noaa-gfs-bdp-pds` begins **~2021-03-23** (earlier `gfs.YYYYMMDD/`
  prefixes exist but contain no 0.25° atmos files) — **no 2020 data**.
- This ERA5 store ends **2021-12-31**.

So the training window is roughly **2021-03-23 → 2021-12-31**. Winter/January 2021 GFS is
not available from this bucket; extending earlier would require a different GFS archive.

## Files

```
models/unet/
├── README.md            # this file
├── requirements.txt     # dependencies
├── config.yaml          # region bounds, data sources, sample dates, (placeholder) hyperparams
├── data/
│   ├── fetch_era5.py    # [DONE] load/verify ERA5 region slice + point extraction
│   ├── fetch_gfs.py     # [DONE] byte-range GFS GRIB + GFS-vs-ERA5 comparison
│   ├── dataset.py       # [DONE] PyTorch Dataset (GFS input, GFS−ERA5 residual target)
│   └── precache.py      # [DONE] parallel warm the grid cache before training
├── model/
│   └── unet.py          # [DONE] U-Net architecture (ResidualUNet)
├── train.py             # [DONE] training loop (+ denormalized val eval)
└── evaluate.py          # [DONE] station-level eval vs real 2021 obs (unet_postprocessing)
```

## Step 1 — verify data sources (implemented)

Both fetchers run standalone and print shapes, variable names, and sample values.
Run from the repo root using the project venv:

```bash
.venv/bin/python models/unet/data/fetch_era5.py   # ERA5 region + CYYZ samples
.venv/bin/python models/unet/data/fetch_gfs.py    # GFS region + GFS-vs-ERA5 error table
```

`fetch_gfs.py` prints the side-by-side **GFS − ERA5** `t2m` gap at CYYZ — the "error" the
U-Net learns. Example (2021 samples):

```
valid_time                GFS(degC)  ERA5(degC)   GFS-ERA5
2021-04-15T06:00:00+00:00      5.02        5.03      -0.02
2021-05-15T06:00:00+00:00     10.33        7.53      +2.81
2021-07-15T18:00:00+00:00     27.89       27.62      +0.27
2021-10-15T06:00:00+00:00     18.51       19.07      -0.56
2021-07-16T00:00:00+00:00     22.01       25.45      -3.44
mean |GFS - ERA5|: 1.42 degC   → non-trivial error signal to learn
```

Short leads (f006) sit close to analysis, so their error is small; longer leads (f024)
show the larger errors the model is meant to reduce. If the mean gap were `< 0.5 °C` the
script flags it (likely an analysis-vs-analysis mix-up or bad valid-time matching).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r models/unet/requirements.txt
```

## Next steps (not yet built)

1. ~~`data/dataset.py`~~ **[DONE]** — PyTorch `Dataset`: stacks GFS `(t2m,u10,v10)` region
   grids as input channels, target = `GFS − ERA5` residual; per-channel z-score normalization
   (stats cached to `data/stats.json`), un-normalized grids cached under `.cache/unet_training/`,
   chronological 80/20 train/val split. Self-test: `.venv/bin/python models/unet/data/dataset.py`.
2. ~~`model/unet.py`~~ **[DONE]** — small 2-level U-Net (`ResidualUNet`, 4→3 channels, 16→32→64
   features, ~117k params). Reflect-pads 21×41→24×48 internally so downsampling stays clean, crops
   back to 21×41. Self-test: `.venv/bin/python models/unet/model/unet.py`.
3. ~~`train.py`~~ **[DONE]** — residual regression (MSE on the normalized correction), Adam +
   ReduceLROnPlateau, early stopping (patience 10), best checkpoint → `checkpoints/best_model.pt`,
   per-epoch metrics → `training_log.csv`. After training, reports **denormalized** validation
   RMSE/MAE and corrected-vs-raw-GFS t2m skill. Sanity: `python train.py --sanity` (5 epochs, June
   week). Full: `python train.py`. Pre-warm the grid cache first with `data/precache.py`.
4. ~~`evaluate.py`~~ **[DONE]** — applies `corrected = GFS − pred_error`, bilinearly interpolates raw
   GFS and corrected grids to CYYZ / Eric D. Soulis, and scores both against **real 2021 station obs**.
   Summary → `eval_results/unet_postprocessing_val2021_summary.json`. Run: `python evaluate.py`.

### ⚠️ Step 5 finding — NOT dashboard-ready

`evaluate.py` scores four methods at the stations on the held-out 2021 val period (Nov 5–Dec 31, 454
samples) vs real obs: **raw GFS**, a zero-parameter **seasonal mean-bias** baseline, the **U-Net**, and an
**ERA5 oracle** (corrected = ERA5 exactly = the ceiling of GFS→ERA5 residual learning).

t2m RMSE (degC):

| station | raw GFS | seasonal bias | U-Net | ERA5 oracle |
|---|---|---|---|---|
| cyyz | 1.289 | 1.312 (−1.7%) | **1.404 (−8.9%)** | 0.877 (+32.0%) |
| eric_d_soulis | 1.475 | 1.451 (+1.6%) | **1.513 (−2.6%)** | 0.962 (+34.8%) |

**The U-Net is the worst correction for t2m — it loses to raw GFS *and* to a zero-parameter seasonal-mean
baseline.** Yet the ERA5 oracle shows ~32–35% is achievable, so the residual-learning *premise* is sound;
the *model* is the bottleneck. Root cause: the chronological split trains on Mar–Nov (no winter — GFS 0.25°
starts 2021-03-23) and validates on Nov–Dec; December (248/454 val samples) has no matching training season
and falls back to fall. The learned correction is miscalibrated for the cold-season val window.

Wind caveats: at **cyyz** the ERA5 oracle is *worse* than raw (−2.3%) — no headroom to chase. At
**eric_d_soulis** raw wind RMSE is ~16 km/h and almost all bias (obs ≈ 15 km/h below GFS); even the oracle
stays ~12.5 km/h, so the "+20.6% U-Net" gain is an artifact of shifting wind toward a mis-calibrated
low-reading station, not skill.

**Do not add `unet_postprocessing` to the dashboard** (chronological split). To realize the t2m headroom:
get winter into training (multi-year / different GFS archive), use a season-aware split, and first make the
model beat the seasonal-mean baseline — if a 117k-param U-Net can't, it's over-engineered for this signal.

### Interleaved-month split (`--split-mode interleaved_month`) — partial win

Splitting per month (valid-day ≤ 24 → train, else val) so **every month appears in both** sets fixes the
chronological split's "val is an unseen season" artifact: the val set now spans Mar–Dec (528 samples) and
the seasonal baseline gets a real DJF (December) field — no fallback. Retrained same arch/hyperparams
(`checkpoints/best_model_interleaved.pt`), t2m RMSE (degC) vs real obs:

| station | raw GFS | seasonal bias | U-Net | ERA5 oracle |
|---|---|---|---|---|
| cyyz | 1.709 | 1.693 (+1.0%) | 1.797 (**−5.2%**) | 1.215 (+28.9%) |
| eric_d_soulis | 2.201 | 2.003 (+9.0%) | **1.787 (+18.8%)** | 1.314 (+40.3%) |

**The interleaved U-Net beats the seasonal baseline on t2m at Eric D. Soulis (+18.8% vs +9.0%, ~half the
oracle) — a genuine win, flipped from −2.6% under the chronological split. But at CYYZ it still loses to
both raw and the baseline (−5.2%).** Post-processing helps where raw GFS is worst / local effects dominate
(Soulis) and hurts where GFS is already clean (Pearson airport).

⚠️ **Still no true winter.** Interleaving fixes only the *fallback* artifact, not the *missing-winter* gap:
the sole winter month present anywhere is December (DJF train=192, val=56); Jan/Feb do not exist in this
window (GFS 0.25° starts 2021-03-23, ERA5 ends 2021-12-31). Cross-split numbers aren't comparable (different
val populations). Verdict: encouraging at one station, not yet a clean dashboard add.

## Forecast lead as an input channel

The network input is **4 channels**: the three normalized GFS fields (`t2m`,
`u10`, `v10`) plus a spatially-constant plane carrying the forecast lead,
scaled by `LEAD_SCALE_HOURS` (48 h) so it lands in ~(0, 1]. The output stays
3 channels — the residual for the physical fields only.

Without it, one network trained across f006–f048 can only learn a single
blended correction: the residual it must predict grows with lead, but the grid
alone does not say whether it is a 6-hour or a 48-hour forecast. The result is
systematic over-correction at short leads and under-correction at long ones.

`dataset.build_model_input` is the single definition of that layout. Training,
`evaluate.py`, and the dashboard benchmark all build their tensors through it,
so the encoding cannot drift between them.

### Checkpoint compatibility

Checkpoints record `in_channels`, `out_channels`, `lead_channel`,
`lead_scale_hours`, and `sample_space` (the cycles/leads actually trained on).
`unet.model_from_checkpoint` rebuilds the matching architecture from those,
so pre-lead-channel checkpoints (3→3, no `lead_channel`) still load and run.

`sample_space` is what consumers should trust for coverage — **not
`config.yaml`**, which describes the *next* training run. After widening the
config, a checkpoint trained on the old narrower set would otherwise be scored
across cells it has never seen.

### Widening the cycle/lead range

1. Edit `init_hours_utc` / `forecast_hours` in `config.yaml` (keep
   `LEAD_SCALE_HOURS` equal to the longest lead).
2. `run_pipeline.py fetch` — enumeration is config-driven, so only the new
   cells download.
3. Retrain. `stats.json` records its own `sample_space` and **auto-recomputes**
   when it no longer matches: residual magnitude scales with lead, and the
   arrays stay `(3,)` either way, so a stale file would silently mis-scale
   rather than fail.

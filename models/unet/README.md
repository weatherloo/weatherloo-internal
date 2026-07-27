# UNet bias correction

Residual bias-correction model for gridded NWP output. `model.py` holds the
architecture; `checkpoints/` holds the trained weights and training log.

| File | What |
|------|------|
| `model.py` | `UNet`, `load_checkpoint`, `correct_fields` |
| `checkpoints/best_model.pt` | Weights + `stats` + training metadata |
| `checkpoints/stats.json` | Normalization constants (same as embedded `stats`) |
| `checkpoints/training_log.csv` | Per-epoch `train_loss,val_loss,lr` |

The architecture in `model.py` was recovered from the checkpoint's tensor
shapes. `load_checkpoint` loads with `strict=True`, so if the two ever drift
apart the run fails loudly instead of silently mis-wiring weights.

## Architecture

Two-level UNet, 117,395 parameters:

```
enc1        ConvBlock(3  -> 16)          ─┐ skip
enc2        ConvBlock(16 -> 32)          ─┼┐ skip
bottleneck  ConvBlock(32 -> 64)           ││
up2         ConvTranspose2d(64 -> 32, 2)  │┘
dec2        ConvBlock(64 -> 32)           │
up1         ConvTranspose2d(32 -> 16, 2)  │
dec1        ConvBlock(32 -> 16)          ─┘
head        Conv2d(16 -> 3, 1x1)
```

`ConvBlock` is `(Conv3x3(bias=False) -> BatchNorm -> ReLU) x2`; downsampling is
`max_pool2d(2)`. Two pooling levels means both spatial dims must be divisible by
4 — `correct_fields` reflect-pads up to a multiple of 4 and crops the residual
back, so callers can pass any patch size.

## Applying it

The network predicts a **residual**, not a forecast:

```python
from model import load_checkpoint, correct_fields

model, stats, meta = load_checkpoint("checkpoints/best_model.pt")
corrected = correct_fields(model, stats, patch)   # patch is (3, H, W)
```

```
x_norm    = (forecast − gfs.mean) / gfs.std
r_norm    = unet(x_norm)
corrected = forecast + (r_norm * residual.std + residual.mean)
```

Channels are ordered `(t2m, u10, v10)` with **`t2m` in °C** (not Kelvin) and
wind components in m/s — that is the space the normalization stats live in.
Correct `u10`/`v10` separately and derive wind speed afterwards; the model was
never trained on speed directly.

## Benchmark

`benchmarking-site/data/unet/compute_benchmark.py` scores this checkpoint on the
dashboard as a correction layer on GFS 0.25°, against station observations. It
shares its input, cycles, and lead times with `gfs_interpolated`, so the two are
a direct uncorrected-vs-corrected pair.

```bash
.venv/bin/python benchmarking-site/data/unet/compute_benchmark.py --dry-run
```

## Caveats for the current checkpoint

- **Small training set.** `train_days=24`, `n_samples=80`, chronological split.
- **Overfit past epoch ~11.** Best val loss is epoch 17 (0.5120) while train loss
  keeps falling to 0.378 by epoch 27; the halved LR at epoch 23 did not help.
  The saved checkpoint is the best-val one, not the last.
- **Residual sign is inferred, not recorded.** `correct_fields` *adds* the
  denormalized residual, i.e. it assumes the training target was
  `truth − forecast`. Nothing in the checkpoint records this. If the training
  script built the target as `forecast − truth`, the correction is applied
  backwards and every number on the dashboard flips sign — worth confirming
  against the line that builds the target tensor.

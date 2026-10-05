#!/usr/bin/env python3
"""Residual-correction U-Net for GFS post-processing.

Predicts the ``GFS - ERA5`` residual (3 channels: t2m, u10, v10) over the
southern Ontario region grid, so the corrected forecast is
``corrected = GFS - predicted_residual`` (see ``models/unet/README.md``).

Architecture
------------
Standard U-Net (encoder -> bottleneck -> decoder with skip connections), each
conv block being ``Conv2d -> BatchNorm2d -> ReLU`` (matches the Gronquist et al.
reference). Kept deliberately small — 16 -> 32 -> 64 feature channels — because
the region is tiny (21x41) and there is only ~9 months of training data; a
64 -> 128 -> 256 network would badly over-parameterize this.

Why only 2 downsampling levels + padding
----------------------------------------
The spatial grid is **21x41**, which is both small and odd in both dimensions.
A textbook 4-level U-Net (halving 4 times) would drive it to <2x3 — no useful
spatial structure left — and the repeated floor-division on odd sizes makes the
decoder upsampled shapes disagree with the encoder skip shapes (21 -> 10 -> 5,
but 5 -> 10 -> 20 != 21), forcing ad-hoc cropping at every level.

Instead we do **both** of the safeguards the brief allows, because together they
are the cleanest:

1. **Reflect-pad the input 21x41 -> 24x48 once** at the very start, and crop the
   output back to 21x41 at the very end. 24 and 48 are multiples of 4, so with
2. **only 2 downsampling levels** every up/down step halves/doubles exactly
   (24->12->6 and 48->24->12), and each decoder feature map lines up cell-for-cell
   with its encoder skip connection — no per-level cropping, no odd-size hazards.

Reflect padding (rather than zeros) avoids injecting artificial cold/zero-wind
borders that BatchNorm would otherwise treat as real signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Region grid size (lat x lon) and the padded size used internally.
GRID_H, GRID_W = 21, 41
PAD_H, PAD_W = 24, 48  # multiples of 4 -> two clean /2 downsampling levels


class ConvBlock(nn.Module):
    """Two ``Conv2d -> BatchNorm2d -> ReLU`` layers (standard U-Net double conv)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualUNet(nn.Module):
    """Small 2-level U-Net: (B,4,21,41) input -> (B,3,21,41) predicted residual.

    Inputs are the three normalized GFS channels (``t2m``, ``u10``, ``v10``)
    plus a constant plane carrying the forecast lead (see
    ``dataset.build_model_input``); outputs are the residual for the three
    physical channels only, so ``in_channels`` is one more than ``out_channels``.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 base_features: int = 16):
        super().__init__()
        f1, f2, f3 = base_features, base_features * 2, base_features * 4  # 16, 32, 64

        # Encoder
        self.enc1 = ConvBlock(in_channels, f1)   # 24x48
        self.enc2 = ConvBlock(f1, f2)            # 12x24
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f2, f3)      # 6x12

        # Decoder (transpose conv exactly doubles; dims are multiples of 4 so
        # every upsampled map matches its encoder skip cell-for-cell).
        self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)  # 6->12
        self.dec2 = ConvBlock(f3, f2)            # concat(f2 skip + f2 up) -> f2
        self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)  # 12->24
        self.dec1 = ConvBlock(f2, f1)            # concat(f1 skip + f1 up) -> f1

        self.head = nn.Conv2d(f1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad 21x41 -> 24x48 (reflect) so downsampling stays clean; crop back at end.
        pad_h, pad_w = PAD_H - GRID_H, PAD_W - GRID_W  # 3, 7
        # F.pad order is (left, right, top, bottom); split padding across both sides.
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2), mode="reflect")

        e1 = self.enc1(x)                 # (B, f1, 24, 48)
        e2 = self.enc2(self.pool(e1))     # (B, f2, 12, 24)
        b = self.bottleneck(self.pool(e2))  # (B, f3, 6, 12)

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))   # (B, f2, 12, 24)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, f1, 24, 48)
        out = self.head(d1)               # (B, out, 24, 48)

        # Crop the padded borders back to the true 21x41 grid.
        return out[:, :, pad_h // 2: pad_h // 2 + GRID_H,
                   pad_w // 2: pad_w // 2 + GRID_W]


# ---------------------------------------------------------------------------
# Checkpoint -> model
# ---------------------------------------------------------------------------
def uses_lead_channel(ckpt: dict) -> bool:
    """Whether a checkpoint's network expects the lead-time input channel."""
    return bool(ckpt.get("lead_channel", False))


def model_from_checkpoint(ckpt: dict, strict: bool = True) -> "ResidualUNet":
    """Build the network a checkpoint was trained as, and load its weights.

    Reads the channel layout from the checkpoint rather than assuming the
    current defaults, so pre-lead-channel checkpoints (3 in / 3 out, no
    ``lead_channel`` key) still load. ``strict=True`` means a checkpoint whose
    layout disagrees with the architecture fails loudly instead of silently
    mis-wiring weights.
    """
    out_ch = int(ckpt.get("out_channels", len(ckpt.get("channels", ("t2m", "u10", "v10")))))
    in_ch = int(ckpt.get("in_channels", out_ch + (1 if uses_lead_channel(ckpt) else 0)))
    model = ResidualUNet(in_channels=in_ch, out_channels=out_ch)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    return model


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:
    torch.manual_seed(0)
    model = ResidualUNet()
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print("=== ResidualUNet self-test ===")
    print(f"  total parameters: {n_params:,}")

    # --- Dummy batch ---
    print("\n=== Dummy batch (8, 4, 21, 41) -> (8, 3, 21, 41) ===")
    x = torch.randn(8, 4, GRID_H, GRID_W)
    with torch.no_grad():
        y = model(x)
    print(f"  input  shape: {tuple(x.shape)}")
    print(f"  output shape: {tuple(y.shape)}")
    assert tuple(y.shape) == (8, 3, GRID_H, GRID_W), "output spatial shape mismatch!"
    assert model.enc1.block[0].in_channels == 4, "expected the lead-time input channel!"
    n_nan, n_inf = int(torch.isnan(y).sum()), int(torch.isinf(y).sum())
    print(f"  NaN={n_nan}  Inf={n_inf}  min={y.min():+.3f} max={y.max():+.3f} mean={y.mean():+.3f}")
    assert n_nan == 0 and n_inf == 0, "dummy output has NaN/Inf!"

    # --- Real batch from GFSResidualDataset ---
    print("\n=== Real batch from GFSResidualDataset (2021-06-01 .. 2021-06-07) ===")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from dataset import GFSResidualDataset, CACHE_DIR  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402

    ds = GFSResidualDataset("train", "2021-06-01", "2021-06-07",
                            stats_path=CACHE_DIR / "stats_selftest.json")
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    xb, yb = next(iter(loader))
    with torch.no_grad():
        pred = model(xb)
    print(f"  real input  shape: {tuple(xb.shape)}")
    print(f"  target      shape: {tuple(yb.shape)}")
    print(f"  pred        shape: {tuple(pred.shape)}")
    assert pred.shape == yb.shape, "prediction shape != target shape!"
    n_nan, n_inf = int(torch.isnan(pred).sum()), int(torch.isinf(pred).sum())
    print(f"  pred NaN={n_nan}  Inf={n_inf}  "
          f"min={pred.min():+.3f} max={pred.max():+.3f} mean={pred.mean():+.3f}")
    assert n_nan == 0 and n_inf == 0, "real-data output has NaN/Inf!"

    print("\nOK: architecture preserves 21x41, small param count, no NaN/Inf on dummy or real data.")


if __name__ == "__main__":
    _selftest()

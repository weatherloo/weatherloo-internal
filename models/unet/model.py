"""UNet bias-correction model: gridded NWP forecast -> corrected forecast.

Architecture is recovered from ``checkpoints/best_model.pt`` and must stay in
lockstep with it — ``load_checkpoint`` loads the state dict with ``strict=True``
so any drift fails loudly rather than silently mis-wiring weights.

The network predicts a **residual** (truth − forecast) in normalized space for
three channels (``t2m``, ``u10``, ``v10``). Applying it is a three-step affine
sandwich, all of which lives in :func:`correct_fields`:

    x_norm    = (forecast − gfs.mean) / gfs.std
    r_norm    = unet(x_norm)
    corrected = forecast + (r_norm * residual.std + residual.mean)

Units follow the training stats: ``t2m`` in **°C** (not Kelvin), ``u10``/``v10``
in **m/s**.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = ("t2m", "u10", "v10")
#: Two 2x2 pooling levels, so both spatial dims must be divisible by 4.
SIZE_MULTIPLE = 4


class ConvBlock(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x2. Convs are bias-free; the BN provides the shift."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
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


class UNet(nn.Module):
    """Two-level UNet (16/32 encoder, 64 bottleneck) with a 1x1 residual head."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.bottleneck = ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[UNet, dict, dict]:
    """Return ``(model in eval mode, stats, checkpoint metadata)``.

    ``weights_only=True`` keeps loading safe; the checkpoint holds only tensors
    and plain dicts.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    channels = tuple(ckpt.get("channels", CHANNELS))
    model = UNet(in_ch=len(channels), out_ch=len(channels))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval().to(device)
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state", "stats")}
    return model, ckpt["stats"], meta


def load_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _pad_to_multiple(x: torch.Tensor, multiple: int = SIZE_MULTIPLE) -> tuple[torch.Tensor, int, int]:
    """Reflect-pad H/W up to ``multiple`` so pooling and upsampling round-trip.

    Reflection (rather than zeros) keeps the padded border physically plausible,
    which matters because the station sits only a few grid cells from the edge of
    the cropped patch.
    """
    h, w = x.shape[-2:]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


@torch.no_grad()
def correct_fields(
    model: UNet,
    stats: dict,
    fields: "np.ndarray",  # noqa: F821 — numpy imported by callers
    device: str = "cpu",
) -> "np.ndarray":  # noqa: F821
    """Apply the learned residual correction to a ``(3, H, W)`` forecast patch.

    ``fields`` is ordered ``(t2m °C, u10 m/s, v10 m/s)``. Returns the corrected
    patch with the same shape and units. NaNs are passed through: they are
    zero-filled before the forward pass (so they cannot poison neighbouring
    cells through the convolutions) and restored afterwards.
    """
    import numpy as np

    arr = np.asarray(fields, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != len(stats["gfs"]["mean"]):
        raise ValueError(
            f"expected ({len(stats['gfs']['mean'])}, H, W) fields, got {arr.shape}"
        )

    nan_mask = ~np.isfinite(arr)
    filled = np.where(nan_mask, 0.0, arr)

    gfs_mean = np.asarray(stats["gfs"]["mean"], dtype=np.float32)[:, None, None]
    gfs_std = np.asarray(stats["gfs"]["std"], dtype=np.float32)[:, None, None]
    res_mean = np.asarray(stats["residual"]["mean"], dtype=np.float32)[:, None, None]
    res_std = np.asarray(stats["residual"]["std"], dtype=np.float32)[:, None, None]

    x = torch.from_numpy((filled - gfs_mean) / gfs_std).unsqueeze(0).to(device)
    x, pad_h, pad_w = _pad_to_multiple(x)

    r_norm = model(x)

    if pad_h or pad_w:
        r_norm = r_norm[..., : r_norm.shape[-2] - pad_h, : r_norm.shape[-1] - pad_w]

    residual = r_norm.squeeze(0).cpu().numpy() * res_std + res_mean
    corrected = filled + residual
    return np.where(nan_mask, np.nan, corrected)

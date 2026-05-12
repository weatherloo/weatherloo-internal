from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DenseTemperatureResidualUNet(nn.Module):
    def __init__(
        self,
        *,
        dynamic_channels: int,
        static_channels: int = 0,
        lead_embedding_dim: int = 16,
        base_channels: int = 32,
        max_lead_hour: int = 18,
    ) -> None:
        super().__init__()
        self.total_input_channels = dynamic_channels + static_channels + lead_embedding_dim
        self.lead_embedding = nn.Embedding(max_lead_hour + 1, lead_embedding_dim)

        self.enc1 = ConvBlock(self.total_input_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = DownBlock(base_channels * 4, base_channels * 8)
        self.dec3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(
        self,
        dynamic: torch.Tensor,
        *,
        static: torch.Tensor | None = None,
        lead_hour: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = dynamic.shape
        if static is None:
            static = torch.zeros((batch_size, 0, height, width), device=dynamic.device, dtype=dynamic.dtype)

        lead_features = self.lead_embedding(lead_hour.clamp_min(0))
        lead_features = lead_features[..., None, None].expand(batch_size, -1, height, width)
        x = torch.cat([dynamic, static, lead_features], dim=1)

        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        bottleneck = self.bottleneck(skip3)
        x = self.dec3(bottleneck, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.head(x)


def sample_station_points(field: torch.Tensor, station_xy: torch.Tensor) -> torch.Tensor:
    """
    Bilinearly sample a dense field at normalized station coordinates.

    Parameters
    ----------
    field:
        Tensor of shape ``[batch, 1, height, width]``.
    station_xy:
        Tensor of shape ``[batch, num_stations, 2]`` in ``[-1, 1]`` grid coordinates.
    """

    grid = station_xy.unsqueeze(2)
    sampled = F.grid_sample(
        field,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(1).squeeze(-1)


def station_temperature_loss(
    corrected_field: torch.Tensor,
    *,
    station_xy: torch.Tensor,
    station_values_c: torch.Tensor,
    station_mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    sampled = sample_station_points(corrected_field, station_xy)
    mask = station_mask.bool()
    if not torch.any(mask):
        return corrected_field.new_tensor(0.0)
    return F.huber_loss(sampled[mask], station_values_c[mask], delta=delta)


def coarse_consistency_loss(
    corrected_field: torch.Tensor,
    baseline_field: torch.Tensor,
    *,
    downsample_factor: int = 4,
    delta: float = 0.5,
) -> torch.Tensor:
    if downsample_factor <= 1:
        return F.huber_loss(corrected_field, baseline_field, delta=delta)
    corrected_down = F.avg_pool2d(corrected_field, kernel_size=downsample_factor, stride=downsample_factor)
    baseline_down = F.avg_pool2d(baseline_field, kernel_size=downsample_factor, stride=downsample_factor)
    return F.huber_loss(corrected_down, baseline_down, delta=delta)


def smoothness_loss(residual_field: torch.Tensor) -> torch.Tensor:
    diff_y = residual_field[..., 1:, :] - residual_field[..., :-1, :]
    diff_x = residual_field[..., :, 1:] - residual_field[..., :, :-1]
    return diff_y.abs().mean() + diff_x.abs().mean()


@dataclass(frozen=True)
class V1LossConfig:
    station_weight: float = 1.0
    consistency_weight: float = 0.15
    regularization_weight: float = 0.01
    station_huber_delta: float = 1.0
    consistency_huber_delta: float = 0.5
    coarse_downsample_factor: int = 4


def compute_v1_loss(
    *,
    residual_field: torch.Tensor,
    baseline_field: torch.Tensor,
    station_xy: torch.Tensor,
    station_values_c: torch.Tensor,
    station_mask: torch.Tensor,
    config: V1LossConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or V1LossConfig()
    corrected_field = baseline_field + residual_field

    station_term = station_temperature_loss(
        corrected_field,
        station_xy=station_xy,
        station_values_c=station_values_c,
        station_mask=station_mask,
        delta=cfg.station_huber_delta,
    )
    consistency_term = coarse_consistency_loss(
        corrected_field,
        baseline_field,
        downsample_factor=cfg.coarse_downsample_factor,
        delta=cfg.consistency_huber_delta,
    )
    regularization_term = smoothness_loss(residual_field)

    total = (
        cfg.station_weight * station_term
        + cfg.consistency_weight * consistency_term
        + cfg.regularization_weight * regularization_term
    )
    return {
        "loss": total,
        "station_loss": station_term.detach(),
        "consistency_loss": consistency_term.detach(),
        "regularization_loss": regularization_term.detach(),
    }


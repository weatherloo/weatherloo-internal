"""Matched direct and autoregressive station-bias models for hourly HRRR leads."""

from __future__ import annotations

import torch
from torch import nn


class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, leads, height, width, channels = x.shape
        frames = x.reshape(batch * leads, height, width, channels).permute(0, 3, 1, 2)
        return self.net(frames).flatten(1).reshape(batch, leads, 32)


class DirectModel(nn.Module):
    """Predict each lead's station bias from the known HRRR forecast sequence."""

    def __init__(self):
        super().__init__()
        self.encoder = FrameEncoder()
        self.temporal = nn.Sequential(nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(33, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = self.temporal(features.transpose(1, 2)).transpose(1, 2)
        hour = torch.arange(x.shape[1], device=x.device, dtype=x.dtype) / 48.0
        hour = hour.expand(x.shape[0], -1).unsqueeze(-1)
        return self.head(torch.cat((features, hour), dim=-1))


class AutoregressiveModel(nn.Module):
    """Feed each predicted bias into the next hourly update; never use future labels."""

    def __init__(self):
        super().__init__()
        self.encoder = FrameEncoder()
        self.cell = nn.GRUCell(35, 64)  # HRRR features, hour, previous predicted bias
        self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        state = features.new_zeros((x.shape[0], 64))
        previous = features.new_zeros((x.shape[0], 2))
        predictions = []
        for hour in range(x.shape[1]):
            lead = features.new_full((x.shape[0], 1), hour / 48.0)
            state = self.cell(torch.cat((features[:, hour], lead, previous), dim=-1), state)
            previous = self.head(state)
            predictions.append(previous)
        return torch.stack(predictions, dim=1)

"""A time-extended, field-only HyperDS prototype.

The field encoder generates the weights of a coordinate decoder for each hour.
The decoder can be queried at any normalized (x, y) location. In autoregressive
mode its own previous correction is used in the next hourly step.
"""

import torch
from torch import nn
from torch.nn import functional as F


class TemporalHyperDS(nn.Module):
    def __init__(self, autoregressive: bool = False):
        super().__init__()
        self.autoregressive = autoregressive
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.temporal = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.recurrent = nn.GRUCell(65, 64)
        # Generated decoder: 8 inputs -> 32 hidden units -> 2 bias corrections.
        self.weight_generator = nn.Linear(64, 8 * 32 + 32 + 32 * 2 + 2)

    def _decode(self, context, coordinates, hour, field, previous):
        batch, points = coordinates.shape[:2]
        weights = self.weight_generator(context)
        w1, b1, w2, b2 = torch.split(weights, (8 * 32, 32, 32 * 2, 2), dim=-1)
        w1 = w1.reshape(batch, 8, 32)
        w2 = w2.reshape(batch, 32, 2)
        sampled = F.grid_sample(
            field, coordinates.reshape(batch, points, 1, 2),
            mode="bilinear", padding_mode="border", align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        lead = coordinates.new_full((batch, points, 1), hour)
        query = torch.cat((coordinates, lead, sampled, previous), dim=-1)
        hidden = F.relu(torch.bmm(query, w1) + b1[:, None])
        return torch.bmm(hidden, w2) + b2[:, None]

    def forward(self, forecast, coordinates):
        """forecast: [B, T, H, W, 3]; coordinates: [B, Q, 2] in [-1, 1].

        Returns normalized HRRR-minus-observation corrections [B, T, Q, 2].
        All forecast frames must come from the same issued HRRR run.
        """
        if forecast.ndim != 5 or forecast.shape[-1] != 3:
            raise ValueError("forecast must have shape [batch, hour, row, col, 3]")
        if coordinates.ndim != 3 or coordinates.shape[0] != forecast.shape[0] or coordinates.shape[2] != 2:
            raise ValueError("coordinates must have shape [batch, query, 2]")
        batch, hours, height, width, _ = forecast.shape
        fields = forecast.permute(0, 1, 4, 2, 3).reshape(batch * hours, 3, height, width)
        features = self.encoder(fields).reshape(batch, hours, 64)
        features = self.temporal(features.transpose(1, 2)).transpose(1, 2)
        fields = fields.reshape(batch, hours, 3, height, width)
        previous = forecast.new_zeros((batch, coordinates.shape[1], 2))
        state = forecast.new_zeros((batch, 64))
        predictions = []
        for step in range(hours):
            hour = step / max(hours - 1, 1)
            if self.autoregressive:
                state = self.recurrent(
                    torch.cat((features[:, step],
                               forecast.new_full((batch, 1), hour)), dim=-1), state
                )
                context = state
            else:
                context = features[:, step]
            change = self._decode(context, coordinates, hour, fields[:, step], previous)
            prediction = previous + change if self.autoregressive else change
            predictions.append(prediction)
            previous = prediction if self.autoregressive else previous
        return torch.stack(predictions, dim=1)

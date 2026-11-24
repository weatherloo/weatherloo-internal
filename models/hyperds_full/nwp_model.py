"""HyperDS driven by an issued NWP forecast trajectory (HRRR or AIFS), queried hourly.

Time is handled like space. The field encoder runs only at 6-hourly anchor
leads (0, 6, ..., 48). The generated decoder can be queried at any hour h in
between: its context is the time-linear blend of the two bracketing anchor
contexts, just as spatial queries fall between grid cells. The decoder's
auxiliary/residual-base value is the forecast at hour h: exact for hourly HRRR,
time-interpolated for 6-hourly AIFS. Training and evaluation use hourly ERA5
and station labels, so AIFS gets hourly output supervised at every hour.

Conditioning on time and issue-time information:
* lead-hour encoding (as in the ERA5 temporal extension),
* valid-time hour-of-day and day-of-year,
* station observation minus forecast at lead 0.

``nwp_autoregressive`` feeds each anchor's predicted correction field into the
next anchor's encoder input (6-hour steps for both sources).
"""

import torch
from torch import nn
from torch.nn import functional as F

from model import HyperDS, grid_average, subgrid_queries

MODES = ("nwp", "nwp_autoregressive")
FEEDBACK_SHAPE = (8, 16)
ANCHOR_STEP = 6
MAX_HOUR = 48
ANCHORS = tuple(range(0, MAX_HOUR + 1, ANCHOR_STEP))
PRECIPITATION = 4


def at_hour(values, leads, hour, variable_axis):
    """Forecast at ``hour`` from values [batch, lead, ...] issued at ``leads``.

    Exact when the hour is an issued lead. Otherwise instantaneous variables are
    linear in time and precipitation is the mean rate of the containing output
    interval, which is what the catalogs store at the interval's end lead.
    """
    leads = list(leads)
    if hour in leads:
        return values[:, leads.index(hour)]
    upper = next(i for i, lead in enumerate(leads) if lead > hour)
    lower = upper - 1
    weight = (hour - leads[lower]) / (leads[upper] - leads[lower])
    result = values[:, lower] * (1 - weight) + values[:, upper] * weight
    index = [slice(None)] * result.ndim
    index[variable_axis] = PRECIPITATION
    result[tuple(index)] = values[:, upper][tuple(index)]
    return result


def anchor_weights(hour):
    lower = min(hour // ANCHOR_STEP, len(ANCHORS) - 2)
    return lower, (hour - ANCHORS[lower]) / ANCHOR_STEP


class NWPHyperDS(HyperDS):
    def __init__(self, mode="nwp", satellite_shape=(128, 256), n_stations=5):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__("direct", satellite_shape)
        self.mode = mode
        # Zero-initialized so a paper warm start is reproduced before training.
        self.clock_projection = nn.Linear(4, 64)
        self.observation_projection = nn.Linear(4 * n_stations, 64)
        for layer in (self.clock_projection, self.observation_projection):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def conditioning(self, clock, observations):
        """clock [b,4]; observations [b,4*stations] -> [b,64] decoder embedding offset."""
        return self.clock_projection(clock) + self.observation_projection(observations)

    def decode(self, context, field, coordinates, hour=0, base=None, extra=None, return_residual=False):
        """field: raw normalized forecast at this hour; base: [b,k,5] native values for the first k queries."""
        values = F.grid_sample(field, coordinates[:, None], mode="bilinear",
                               padding_mode="border", align_corners=False).squeeze(2).transpose(1, 2)
        if base is not None:
            values = torch.cat((base, values[:, base.shape[1]:]), dim=1)
        embedding = self.spatial_encoding((coordinates + 1) / 2)
        h = coordinates.new_full((*coordinates.shape[:-1], 1), float(hour) / MAX_HOUR)
        embedding = embedding + self.hour_encoding(h) - self.hour_encoding(torch.zeros_like(h))
        if extra is not None:
            embedding = embedding + extra[:, None, :]
        residual = torch.cat([decoder(context, embedding, values[:, :, index])
                              for index, decoder in enumerate(self.decoders)], dim=-1)
        return (values + residual, residual) if return_residual else values + residual

    def encoder_input(self, field, correction):
        """Autoregressive feedback: forecast field plus the previous anchor's correction."""
        if self.mode != "nwp_autoregressive" or correction is None:
            return field
        return field + F.interpolate(correction, size=field.shape[-2:], mode="bilinear", align_corners=False)

    def encode_anchors(self, satellite_context, forecast, leads, clock, observations):
        """Context at each 6-hourly anchor; autoregressive mode chains anchor corrections."""
        contexts, correction = [], None
        for hour in ANCHORS:
            field = at_hour(forecast, leads, hour, 1)
            context = self.encode_field(self.encoder_input(field, correction), satellite_context)
            contexts.append(context)
            if self.mode == "nwp_autoregressive":
                query = subgrid_queries(field.shape[0], *FEEDBACK_SHAPE, field.device, random=False)
                _, residual = self.decode(context, field, query, hour, extra=self.conditioning(clock[:, hour], observations),
                                          return_residual=True)
                correction = grid_average(residual, *FEEDBACK_SHAPE)
        return contexts

    def decode_hour(self, contexts, forecast, forecast_station, leads, hour, queries, extra):
        """Prediction at any hour 0..48; the first stations-many queries use native station values."""
        lower, weight = anchor_weights(hour)
        context = contexts[lower] * (1 - weight) + contexts[lower + 1] * weight
        field = at_hour(forecast, leads, hour, 1)
        # forecast_station=None is the paper's design: every query starts from the input field interpolation.
        base = None if forecast_station is None else at_hour(forecast_station, leads, hour, 2)
        return self.decode(context, field, queries, hour, base=base, extra=extra)

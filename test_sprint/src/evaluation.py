from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def summarize_station_metrics(frame: pd.DataFrame) -> dict[str, float]:
    required = {"observed_c", "predicted_c"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"Missing required metric columns: {sorted(missing)}"
        raise ValueError(msg)

    y_true = frame["observed_c"].to_numpy(dtype=np.float32)
    y_pred = frame["predicted_c"].to_numpy(dtype=np.float32)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mean_bias": mean_bias(y_true, y_pred),
        "count": float(len(frame)),
    }


def summarize_metrics_by_lead(frame: pd.DataFrame) -> pd.DataFrame:
    if "lead_hour" not in frame.columns:
        msg = "Expected a 'lead_hour' column for lead-wise metrics."
        raise ValueError(msg)

    rows: list[dict[str, float]] = []
    for lead_hour, lead_frame in frame.groupby("lead_hour", sort=True):
        metrics = summarize_station_metrics(lead_frame)
        metrics["lead_hour"] = int(lead_hour)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("lead_hour").reset_index(drop=True)


@dataclass
class PooledBiasCorrector:
    """
    A simple fallback baseline for station supervision.

    The model learns mean bilinear HRRR error at three levels:
    1. station + lead
    2. station only
    3. global
    """

    global_bias_c: float = 0.0
    station_bias_c: dict[str, float] | None = None
    station_lead_bias_c: dict[tuple[str, int], float] | None = None

    def fit(self, frame: pd.DataFrame) -> "PooledBiasCorrector":
        required = {"station_id", "lead_hour", "observed_c", "hrrr_bilinear_c"}
        missing = required - set(frame.columns)
        if missing:
            msg = f"Missing required columns for bias baseline fit: {sorted(missing)}"
            raise ValueError(msg)

        residual = frame["hrrr_bilinear_c"] - frame["observed_c"]
        fit_frame = frame.copy()
        fit_frame["residual_c"] = residual

        self.global_bias_c = float(fit_frame["residual_c"].mean())
        self.station_bias_c = (
            fit_frame.groupby("station_id")["residual_c"].mean().astype(float).to_dict()
        )
        self.station_lead_bias_c = (
            fit_frame.groupby(["station_id", "lead_hour"])["residual_c"].mean().astype(float).to_dict()
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.station_bias_c is None or self.station_lead_bias_c is None:
            msg = "Bias corrector must be fit before predict()."
            raise RuntimeError(msg)

        preds: list[float] = []
        for _, row in frame.iterrows():
            key = (row["station_id"], int(row["lead_hour"]))
            if key in self.station_lead_bias_c:
                bias = self.station_lead_bias_c[key]
            elif row["station_id"] in self.station_bias_c:
                bias = self.station_bias_c[row["station_id"]]
            else:
                bias = self.global_bias_c
            preds.append(float(row["hrrr_bilinear_c"] - bias))
        return np.asarray(preds, dtype=np.float32)


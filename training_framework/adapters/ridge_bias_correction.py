"""Dummy second adapter: ridge regression on lagged bias. numpy only, no torch.

Exists to prove the adapter interface isn't LSTM-shaped, and to give CI a
sub-second model. Same NPZ contract and dataset dict as the LSTM adapter.
"""
import os

import numpy as np

from training_framework.registry import register


def _load_series(data_cfg):
    d = np.load(data_cfg["npz"], allow_pickle=True)
    si = list(d["station_ids"]).index(data_cfg["station"])
    vi = list(d["variables"]).index(data_cfg["variable"])
    li = list(d["lead_times_hours"]).index(data_cfg["lead_time"])
    bias = d["bias"][:, si, vi, li].astype(np.float32)
    inits = d["initializations"].copy()
    d.close()
    return bias, inits


@register("ridge_bias_correction")
class RidgeBiasCorrectionAdapter:
    checkpoint_name = "checkpoint.npy"
    model_label = "Ridge bias correction"

    @staticmethod
    def validate(cfg):
        path = cfg["data"]["npz"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"NPZ not found: {path}")
        bias, inits = _load_series(cfg["data"])  # raises ValueError if station/var/lead missing
        n_valid = int((~np.isnan(bias)).sum())
        if n_valid - cfg["model"].get("seq_len", 24) < 10:
            raise ValueError(f"Only {n_valid} valid samples; need seq_len + 10")
        return {"npz": path, "n_inits_total": int(len(inits)), "n_valid_samples": n_valid}

    @staticmethod
    def load_dataset(cfg):
        seq_len = cfg["model"].get("seq_len", 24)
        bias, inits = _load_series(cfg["data"])
        ok = ~np.isnan(bias)
        bias, inits = bias[ok], inits[ok]
        mean, std = float(bias.mean()), float(bias.std())
        if std == 0.0:
            raise ValueError("Bias series has zero variance, cannot normalize")
        z = ((bias - mean) / std).astype(np.float32)

        X = np.stack([z[i:i + seq_len] for i in range(len(z) - seq_len)])
        y = z[seq_len:]
        ts = inits[seq_len:]
        n_tr, n_va = int(len(X) * 0.70), int(len(X) * 0.15)  # same temporal split as LSTM
        return {
            "train": (X[:n_tr], y[:n_tr]),
            "val": (X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]),
            "test": (X[n_tr + n_va:], y[n_tr + n_va:], ts[n_tr + n_va:]),
            "normalization": {"mean": mean, "std": std},
            "n_samples": int(len(bias)),
            "n_sequences": int(len(X)),
        }

    @staticmethod
    def _fit_ridge(X, y, alpha):
        Xb = np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)])
        reg = alpha * np.eye(Xb.shape[1])
        reg[-1, -1] = 0.0  # don't penalize the intercept
        return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)

    @staticmethod
    def _predict(w, X):
        return np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)]) @ w

    @staticmethod
    def fit(cfg, dataset, run_dir):
        cls = RidgeBiasCorrectionAdapter
        alpha = cfg["model"].get("alpha", 1.0)
        X_tr, y_tr = dataset["train"]
        X_va, y_va = dataset["val"]
        X_te, y_te, ts_te = dataset["test"]
        mean, std = dataset["normalization"]["mean"], dataset["normalization"]["std"]

        w = cls._fit_ridge(X_tr, y_tr, alpha)
        np.save(os.path.join(run_dir, RidgeBiasCorrectionAdapter.checkpoint_name), w)

        preds_norm = cls._predict(w, X_te)
        preds, targets = preds_norm * std + mean, y_te * std + mean
        np.savez(os.path.join(run_dir, "predictions.npz"),
                 predictions=preds, targets=targets, timestamps=ts_te)
        return {
            "device": "cpu",
            "best_val_loss": float(np.mean((cls._predict(w, X_va) - y_va) ** 2)),
            "metrics_normalized": _metrics(preds_norm, y_te),
            "metrics_original": _metrics(preds, targets),
            "baseline_metrics": _metrics(np.zeros_like(targets), targets),
            "normalization": dataset["normalization"],
        }

    @staticmethod
    def load_checkpoint(cfg, checkpoint_path):
        return np.load(checkpoint_path), "cpu"

    @staticmethod
    def predict(cfg, model, device, dataset, normalization):
        X_te, _, ts_te = dataset["test"]
        return RidgeBiasCorrectionAdapter._predict(model, X_te) * normalization["std"] + normalization["mean"], ts_te


def _metrics(preds, targets):
    d = preds - targets
    return {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))), "bias": float(np.mean(d))}

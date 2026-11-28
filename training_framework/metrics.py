"""Framework-owned metrics (numpy only) so verbs don't import a model's training module."""
import numpy as np


def compute_metrics(preds, targets):
    diff = preds - targets
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
    }

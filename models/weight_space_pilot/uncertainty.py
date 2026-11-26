"""Weekly-block bootstrap for paired MAE differences in the pilot caches."""

import argparse
from datetime import date
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    data = np.load(args.cache)
    truth = data["truth"]
    dates = data["date"]
    weeks = np.array([date.fromisoformat(str(d)[:10]).isocalendar().week for d in dates])
    groups = [np.flatnonzero(weeks == week) for week in np.unique(weeks)]
    candidates = ("bias_edit", "gain_edit", "output_bias") if truth.ndim == 2 else (
        "bias_edit", "unit4_edit", "half_edit", "output_calibration")
    rng = np.random.default_rng(7)
    base_loss = np.abs(data["base"] - truth)
    if truth.ndim == 2:
        base_loss = base_loss.mean(axis=1)
    print(f"{args.cache.name}: {len(dates)} origins/forecasts, {len(groups)} weeks")
    for name in candidates:
        loss = np.abs(data[name] - truth)
        if truth.ndim == 2:
            loss = loss.mean(axis=1)
        improvement = base_loss - loss
        draws = np.empty(3000)
        for i in range(len(draws)):
            indices = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
            draws[i] = improvement[indices].mean()
        lower, upper = np.quantile(draws, (0.025, 0.975))
        print(f"{name}: MAE improvement={improvement.mean():+.4f} C; "
              f"95% interval [{lower:+.4f}, {upper:+.4f}]")


if __name__ == "__main__":
    main()

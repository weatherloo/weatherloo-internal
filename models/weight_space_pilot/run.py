"""Small, reproducible weight-edit pilot on IBM TinyTimeMixer and Soulis temperature.

Run with granite-tsfm on PYTHONPATH. Source data and pretrained weights are kept
outside this directory; generated predictions go to the ignored .cache directory.
"""

import argparse
import csv
from datetime import datetime, timedelta
import math
from pathlib import Path

import numpy as np
import torch
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "benchmarking-site/data/observations/eric_d_soulis/raw"
CACHE = ROOT / ".cache/weight_space_pilot"
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r1"
MODEL_REVISION = "f04bebdd4c13475b006ce72672e74c9dc28871dc"
LEADS = (6, 12, 24)
# Frozen after inspecting 2024. Each edit touches three output-head rows only.
BIAS_EDIT = (-0.015, -0.113, 0.084)
GAIN_EDIT = ((1.19, 0.061), (1.20, -0.025), (1.03, 0.056))


def load_hourly_temperature(station, max_year=2026):
    """Use exact hourly readings; UW HOBO's clock uses UTC-5 all year."""
    temperatures = {}
    if station == "cyyz":
        for year in (2023, 2024, 2025, 2026):
            if year > max_year:
                continue
            path = ROOT / f"benchmarking-site/data/observations/cyyz/raw/eccc_hourly_{year}.json"
            import json
            for row in json.loads(path.read_text())["features"]:
                value = row.get("TEMP")
                if value is not None and -60 < value < 50:
                    utc = datetime.fromisoformat(row["UTC_DATE"])
                    temperatures[utc] = float(value)
        return temperatures
    for year in (2023, 2024, 2025):
        path = RAW / f"uw_hobo_15min_{year}.csv"
        with path.open(newline="") as handle:
            rows = csv.reader(handle)
            next(rows)
            for row in rows:
                try:
                    y, month, day, hour, minute = map(int, row[:5])
                    value = float(row[5])
                    if minute != 0 or not (-60 < value < 50) or not math.isfinite(value):
                        continue
                    utc = datetime(y, month, day, hour) + timedelta(hours=5)
                    temperatures[utc] = value
                except (ValueError, IndexError):
                    continue
    return temperatures


def examples(year, temperatures):
    days = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
    starts = [datetime(year, 1, 1) + timedelta(days=d) for d in range(days)]
    contexts, truths, persistence, seasonal, dates, scales = [], [], [], [], [], []
    for origin in starts:
        times = [origin - timedelta(hours=511 - k) for k in range(512)]
        values = np.array([temperatures.get(t, np.nan) for t in times], dtype=np.float32)
        if np.isnan(values).sum() > 12 or np.isnan(values[-1]):
            continue
        target = [temperatures.get(origin + timedelta(hours=h), np.nan) for h in LEADS]
        prev_day = [temperatures.get(origin + timedelta(hours=h - 24), np.nan) for h in LEADS]
        if not (np.isfinite(target).all() and np.isfinite(prev_day).all()):
            continue
        # Interpolation uses past context only; neither target nor later observations.
        good = np.flatnonzero(np.isfinite(values))
        values = np.interp(np.arange(512), good, values[good]).astype(np.float32)
        contexts.append(values)
        truths.append(target)
        persistence.append([values[-1]] * len(LEADS))
        seasonal.append(prev_day)
        dates.append(origin.isoformat() + "Z")
        scales.append(values.std())
    return {"context": np.array(contexts), "truth": np.array(truths),
            "persistence": np.array(persistence), "seasonal": np.array(seasonal),
            "date": np.array(dates), "scale": np.array(scales)}


def predict(model, contexts):
    model.eval()
    forecasts = []
    with torch.inference_mode():
        for start in range(0, len(contexts), 32):
            batch = torch.from_numpy(contexts[start:start + 32, :, None])
            out = model(past_values=batch, return_loss=False).prediction_outputs
            forecasts.append(out[:, [h - 1 for h in LEADS], 0].cpu().numpy())
    return np.concatenate(forecasts)


def summarize(name, prediction, truth):
    err = prediction - truth
    print(f"{name}: n={len(err)}; MAE={np.abs(err).mean():.3f} C; RMSE={np.sqrt(np.mean(err**2)):.3f} C")
    print("  by lead: " + ", ".join(
        f"{h}h bias={err[:, i].mean():+.3f}, MAE={np.abs(err[:, i]).mean():.3f}"
        for i, h in enumerate(LEADS)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, choices=(2024, 2025, 2026))
    parser.add_argument("--station", choices=("eric_d_soulis", "cyyz"), default="eric_d_soulis")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    if args.year == 2026 and args.station != "cyyz":
        parser.error("2026 observations are available only for cyyz")
    torch.set_num_threads(4)
    data = examples(args.year, load_hourly_temperature(args.station, max_year=args.year))
    print(f"{args.station} {args.year}: {len(data['truth'])} daily forecast origins")
    model = TinyTimeMixerForPrediction.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION)
    data["base"] = predict(model, data["context"])
    if args.year == 2025:
        head = model.head.base_forecast_block
        with torch.no_grad():
            for lead, edit in zip(LEADS, BIAS_EDIT):
                head.bias[lead - 1] += edit
        data["bias_edit"] = predict(model, data["context"])
        with torch.no_grad():
            for lead, edit in zip(LEADS, BIAS_EDIT):
                head.bias[lead - 1] -= edit
            for lead, (gain, shift) in zip(LEADS, GAIN_EDIT):
                head.weight[lead - 1] *= gain
                head.bias[lead - 1] *= gain
                head.bias[lead - 1] += shift
        data["gain_edit"] = predict(model, data["context"])
        data["output_bias"] = data["base"] - np.array((0.140, 0.678, -0.402))
    for key in ("base", "persistence", "seasonal"):
        summarize(key, data[key], data["truth"])
    if args.year == 2025:
        for key in ("bias_edit", "gain_edit", "output_bias"):
            summarize(key, data[key], data["truth"])
    if args.save:
        CACHE.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.station == "eric_d_soulis" else f"_{args.station}"
        path = CACHE / f"baseline_{args.year}{suffix}.npz"
        np.savez_compressed(path, **data)
        print(f"saved {path}")


if __name__ == "__main__":
    main()

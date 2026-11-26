"""One-weight cooling repair with an untouched CYYZ 2026 holdout.

2024 chooses the correction size, 2025 chooses its shrinkage, and 2026 is
evaluated once after those choices are fixed. Run with granite-tsfm on PYTHONPATH.
"""

from datetime import datetime

import numpy as np
import torch

from run import (CACHE, MODEL_ID, MODEL_REVISION, TinyTimeMixerForPrediction,
                 examples, load_hourly_temperature, predict)


FEATURE = 916  # selected from 2024 head features that track 24-hour cooling
LEAD_INDEX = 11  # 12-hour forecast in the 96-output head
SHRINKAGE = (0.0, 0.25, 0.5, 0.75, 1.0)


def cooling(data):
    trend = data["context"][:, -1] - data["context"][:, -25]
    return trend < -5, np.maximum(0, -5 - trend)


def unit_weight_effect(model, context):
    base = predict(model, context)
    head = model.head.base_forecast_block
    with torch.no_grad():
        head.weight[LEAD_INDEX, FEATURE] += 1
    edited = predict(model, context)
    with torch.no_grad():
        head.weight[LEAD_INDEX, FEATURE] -= 1
    return base, edited[:, 1] - base[:, 1]


def fit_coefficient(error, effect, cool, method):
    grid = np.linspace(-4, 4, 8001)
    changed = np.abs(error[:, None] + effect[:, None] * grid)
    if method == "weight":
        # The edit acts outside the cooling condition too; penalize that damage.
        objective = changed[cool].mean(axis=0) + 0.25 * changed[~cool].mean(axis=0)
    else:
        objective = changed.mean(axis=0)
    return float(grid[np.argmin(objective)])


def effects(data, unit_effect):
    cool, excess_cooling = cooling(data)
    return cool, {
        "weight": unit_effect,
        "conditional_output": cool.astype(float),
        "trend_output": excess_cooling,
        "global_output": np.ones(len(cool)),
    }


def mae(error, mask):
    return float(np.abs(error[mask]).mean())


def weekly_interval(data, original_error, changed_error, mask):
    weeks = np.array([datetime.fromisoformat(str(t)[:-1]).isocalendar()[:2]
                      for t in data["date"]])
    keys = sorted(set(map(tuple, weeks)))
    blocks = [np.flatnonzero(np.all(weeks == key, axis=1)) for key in keys]
    rng = np.random.default_rng(2026)
    gains = []
    for _ in range(4000):
        rows = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        rows = rows[mask[rows]]
        if len(rows):
            gains.append(np.mean(np.abs(original_error[rows]) - np.abs(changed_error[rows])))
    return np.quantile(gains, [0.025, 0.975])


def main():
    torch.set_num_threads(4)
    model = TinyTimeMixerForPrediction.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    temperatures = load_hourly_temperature("cyyz", max_year=2025)
    prepared = {}
    for year in (2024, 2025):
        data = examples(year, temperatures)
        base, unit_effect = unit_weight_effect(model, data["context"])
        cool, effect = effects(data, unit_effect)
        prepared[year] = (data, base, cool, effect)
        print(f"{year}: n={len(cool)}, cooling={cool.sum()}, base 12h MAE={mae(base[:, 1]-data['truth'][:, 1], np.ones(len(cool), bool)):.3f}")

    train_data, train_base, train_cool, train_effect = prepared[2024]
    valid_data, valid_base, valid_cool, valid_effect = prepared[2025]
    train_error = train_base[:, 1] - train_data["truth"][:, 1]
    valid_error = valid_base[:, 1] - valid_data["truth"][:, 1]
    chosen = {}
    for method in train_effect:
        coefficient = fit_coefficient(train_error, train_effect[method], train_cool, method)
        alpha = min(SHRINKAGE, key=lambda a: mae(
            valid_error + a * coefficient * valid_effect[method],
            np.ones(len(valid_error), bool)))
        chosen[method] = alpha * coefficient
        val_changed = valid_error + chosen[method] * valid_effect[method]
        print(f"{method}: 2024 coefficient={coefficient:+.3f}; 2025 shrinkage={alpha:.2f}; "
              f"2025 cool MAE={mae(val_changed,valid_cool):.3f}; "
              f"all MAE={mae(val_changed,np.ones(len(val_changed),bool)):.3f}")

    # This is the first use of 2026 outcomes. No choice above depends on them.
    temperatures = load_hourly_temperature("cyyz", max_year=2026)
    data = examples(2026, temperatures)
    base, unit_effect = unit_weight_effect(model, data["context"])
    cool, effect = effects(data, unit_effect)
    head = model.head.base_forecast_block
    with torch.no_grad():
        head.weight[LEAD_INDEX, FEATURE] += chosen["weight"]
    actual_weight_forecast = predict(model, data["context"])
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CACHE / "ttm_conditional_weight_edit.pt")
    with torch.no_grad():
        head.weight[LEAD_INDEX, FEATURE] -= chosen["weight"]
    expected = base[:, 1] + chosen["weight"] * unit_effect
    assert np.max(np.abs(actual_weight_forecast[:, 1] - expected)) < 1e-4
    original_error = base[:, 1] - data["truth"][:, 1]
    print(f"2026 HOLDOUT: n={len(cool)}, cooling={cool.sum()}, "
          f"base cool MAE={mae(original_error,cool):.3f}, "
          f"other MAE={mae(original_error,~cool):.3f}, "
          f"all MAE={mae(original_error,np.ones(len(cool),bool)):.3f}")
    predictions = {"date": data["date"], "truth": data["truth"],
                   "base": base, "cooling": cool}
    for method, coefficient in chosen.items():
        changed_error = original_error + coefficient * effect[method]
        gain = np.abs(original_error) - np.abs(changed_error)
        all_rows = np.ones(len(cool), bool)
        ci_cool = weekly_interval(data, original_error, changed_error, cool)
        ci_all = weekly_interval(data, original_error, changed_error, all_rows)
        print(f"{method}: coefficient={coefficient:+.3f}; "
              f"cool MAE={mae(changed_error,cool):.3f} (gain={gain[cool].mean():+.3f}, "
              f"95% CI [{ci_cool[0]:+.3f},{ci_cool[1]:+.3f}]); "
              f"other MAE={mae(changed_error,~cool):.3f}; "
              f"all MAE={mae(changed_error,all_rows):.3f} "
              f"(gain={gain.mean():+.3f}, 95% CI [{ci_all[0]:+.3f},{ci_all[1]:+.3f}])")
        forecast = actual_weight_forecast if method == "weight" else base.copy()
        if method != "weight":
            forecast[:, 1] += coefficient * effect[method]
        predictions[method] = forecast
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "conditional_2026_cyyz.npz", **predictions)


if __name__ == "__main__":
    main()

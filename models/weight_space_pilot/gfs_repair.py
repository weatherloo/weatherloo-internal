"""Tiny neural GFS temperature bias corrector with chronological splits.

Uses published per-initialization GFS errors to reconstruct paired raw forecasts.
The test period is never used to train or select edits.
"""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GFS = ROOT / "benchmarking-site/data/gfs_interpolated"
CACHE = ROOT / ".cache/weight_space_pilot"
# Frozen from June-August diagnostics, before September-December evaluation.
BIAS_EDIT = -0.224
UNIT4_EDIT = (-0.712, -1.395)  # (output bias delta, output weight[0, 4] delta)
PLAIN_CALIBRATION = (-0.210, -0.115)  # error ~ intercept + slope*(forecast-20)/10


def load_pairs(station="eric_d_soulis"):
    obs_path = ROOT / f"benchmarking-site/data/observations/{station}/observations_6h_2025.json"
    observations = json.loads(obs_path.read_text())["observations"]
    truth = {row["valid_time"]: row["t2m"] for row in observations}
    rows = []
    for path in sorted(GFS.glob("2025-*.json")):
        payload = json.loads(path.read_text())
        init = datetime.fromisoformat(payload["initialization"].replace("Z", "+00:00"))
        variable = payload["locations"][station]["variables"]["t2m"]
        for lead, error in zip(variable["lead_times_hours"], variable["bias"]):
            valid = init + timedelta(hours=lead)
            observed = truth.get(valid.strftime("%Y-%m-%dT%H:%M:%SZ"))
            if error is None or observed is None:
                continue
            raw = observed + error  # GFS forecast; published bias = forecast - observation
            day = valid.timetuple().tm_yday
            angle = 2 * np.pi * day / 365
            hour_angle = 2 * np.pi * valid.hour / 24
            features = (raw / 20, lead / 72, np.sin(hour_angle),
                        np.cos(hour_angle), np.sin(angle), np.cos(angle))
            rows.append((init, raw, observed, error, features))
    return rows


def split(rows, start, end):
    selected = [r for r in rows if start <= r[0].date().isoformat() <= end]
    x = np.array([r[4] for r in selected], dtype=np.float32)
    raw = np.array([r[1] for r in selected], dtype=np.float32)
    truth = np.array([r[2] for r in selected], dtype=np.float32)
    bias = np.array([r[3] for r in selected], dtype=np.float32)
    return x, raw, truth, bias


def model():
    return torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def predict(net, data):
    x, raw, truth, bias = data
    with torch.inference_mode():
        estimated_bias = net(torch.from_numpy(x)).numpy().ravel()
        activation = net[1](net[0](torch.from_numpy(x))).numpy()
    return {"raw": raw, "truth": truth, "bias": bias,
            "corrected": raw - estimated_bias, "activation": activation}


def print_scores(name, predicted, truth):
    errors = predicted - truth
    print(f"{name}: n={len(errors)}, bias={errors.mean():+.3f} C, "
          f"MAE={np.abs(errors).mean():.3f} C, RMSE={np.sqrt(np.mean(errors**2)):.3f} C")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("diagnose", "test"))
    parser.add_argument("--station", choices=("eric_d_soulis", "cyyz"), default="eric_d_soulis")
    args = parser.parse_args()
    if args.phase == "diagnose" and args.station != "eric_d_soulis":
        parser.error("diagnostics and training are fixed to eric_d_soulis")
    torch.set_num_threads(4)
    rows = load_pairs(args.station)
    print(f"{args.station} paired forecast/observation rows: {len(rows)}")
    train = split(rows, "2025-01-01", "2025-05-31")
    dev = split(rows, "2025-06-04", "2025-08-31")
    test = split(rows, "2025-09-04", "2025-12-31")
    CACHE.mkdir(parents=True, exist_ok=True)
    checkpoint = CACHE / "gfs_tiny.pt"
    net = model()
    if args.phase == "diagnose":
        torch.manual_seed(7)
        net = model()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        x = torch.from_numpy(train[0])
        target = torch.from_numpy(train[3][:, None])
        for _ in range(200):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x), target)
            loss.backward()
            optimizer.step()
        torch.save(net.state_dict(), checkpoint)
        print(f"train n={len(x)}; final MSE={loss.item():.3f}; checkpoint={checkpoint}")
        pred = predict(net, dev)
        print_scores("raw GFS dev", pred["raw"], pred["truth"])
        print_scores("tiny model dev", pred["corrected"], pred["truth"])
        print("output bias",net[2].bias.detach().numpy().tolist())
        errors = pred["corrected"] - pred["truth"]
        print("dev error by forecast temperature tercile:")
        for lo, hi in zip([-np.inf, *np.quantile(pred["raw"], [1/3, 2/3])],
                          [*np.quantile(pred["raw"], [1/3, 2/3]), np.inf]):
            mask = (pred["raw"] >= lo) & (pred["raw"] < hi)
            print(f"  [{lo:.1f}, {hi:.1f}): n={mask.sum()}, bias={errors[mask].mean():+.3f}")
        scores = []
        for j in range(8):
            a = pred["activation"][:, j]
            delta = np.dot(errors, a) / np.dot(a, a)
            scores.append((j, delta, np.corrcoef(a, errors)[0, 1]))
        print("one-weight head candidates (unit, delta, corr):", scores)
    else:
        net.load_state_dict(torch.load(checkpoint, weights_only=True))
        pred = predict(net, test)
        print_scores("raw GFS test", pred["raw"], pred["truth"])
        print_scores("tiny model test", pred["corrected"], pred["truth"])
        with torch.no_grad():
            net[2].bias += BIAS_EDIT
        bias_edit = predict(net, test)["corrected"]
        print_scores("bias weight edit test", bias_edit, pred["truth"])
        with torch.no_grad():
            net[2].bias -= BIAS_EDIT
            net[2].bias += UNIT4_EDIT[0]
            net[2].weight[0, 4] += UNIT4_EDIT[1]
        unit4_edit = predict(net, test)["corrected"]
        print_scores("unit-4 weight edit test", unit4_edit, pred["truth"])
        with torch.no_grad():
            net[2].bias -= UNIT4_EDIT[0] / 2
            net[2].weight[0, 4] -= UNIT4_EDIT[1] / 2
        half_edit = predict(net, test)["corrected"]
        print_scores("half-strength unit-4 test", half_edit, pred["truth"])
        intercept, slope = PLAIN_CALIBRATION
        calibrated = pred["corrected"] - (intercept + slope * (pred["corrected"] - 20) / 10)
        print_scores("plain output calibration test", calibrated, pred["truth"])
        days = [r[0].date().isoformat() for r in rows
                if "2025-09-04" <= r[0].date().isoformat() <= "2025-12-31"]
        np.savez_compressed(CACHE / f"gfs_test_{args.station}.npz",
                            date=np.array(days), truth=pred["truth"], raw=pred["raw"],
                            base=pred["corrected"], bias_edit=bias_edit,
                            unit4_edit=unit4_edit, half_edit=half_edit,
                            output_calibration=calibrated)


if __name__ == "__main__":
    main()

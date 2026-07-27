"""
Retrain LSTM bias-correction models using the best_params found by past Optuna
sweeps. Two sweep_results.json sources are read and merged:

  1. lstm_training/*.zip -- one zip per NWP method, each containing
     <method>_<lead>h/sweep_results.json. Station and variable are NOT encoded
     anywhere in those zips/JSONs -- confirmed by inspection, not assumed --
     so they're tagged with --station/--variable (default cyyz/t2m).

  2. lstm_training/output/<station>_<variable>/<method>_<lead>h/sweep_results.json
     -- run_all.py's own output layout. Station and variable ARE encoded in
     the path here, so results for a new station (e.g. after running
     `python lstm_training/run_all.py --station eric_d_soulis`) are picked up
     automatically once you pass the matching --station/--variable.

If both sources have a result for the same (station, variable, method, lead),
the run_all.py-produced one wins (it's self-describing and more likely fresh).

Usage:
    # See what would be trained, no training:
    python lstm_training/retrain_best.py

    # Train exactly one combo as a smoke test, compare RMSE to the original sweep:
    python lstm_training/retrain_best.py --smoke_test

    # Train everything matching --leads (default 6 12) for cyyz/t2m:
    python lstm_training/retrain_best.py --all

    # Same, for the other station, all 12 lead times, once its sweep is done:
    python lstm_training/retrain_best.py --station eric_d_soulis --leads 6 12 18 24 30 36 42 48 54 60 66 72 --all
"""
import argparse
import json
import os
import re
import zipfile
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    BiasLSTM,
    build_features,
    compute_metrics,
    load_bias_series,
    make_sequences,
    parse_timestamps,
    run_eval,
    temporal_split,
)

LSTM_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(LSTM_DIR)

# Same mapping as run_all.py -- method name -> NPZ path relative to repo root.
METHODS = {
    "climatology":       "benchmarking-site/data/climatology/climatology_2025.npz",
    "ecmwf_aifs":        "benchmarking-site/data/ecmwf_aifs/ecmwf_aifs_2025.npz",
    "gefs_mean":         "benchmarking-site/data/gefs_mean/gefs_mean_2025.npz",
    "gfs_analysis":      "benchmarking-site/data/gfs_analysis/gfs_analysis_2025.npz",
    "gfs_interpolated":  "benchmarking-site/data/gfs_interpolated/gfs_interpolated_2025.npz",
    "graphcast":         "benchmarking-site/data/graphcast/graphcast_2025.npz",
    "hrrr_interpolated": "benchmarking-site/data/hrrr_interpolated/hrrr_interpolated_2025.npz",
    "linear_regression": "benchmarking-site/data/linear_regression/linear_regression_2025.npz",
    "persistence":       "benchmarking-site/data/persistence/persistence_2025.npz",
}

ENTRY_RE      = re.compile(r"^(?P<method>[a-z_]+)_(?P<lead>\d+)h/sweep_results\.json$")
DIR_COMBO_RE  = re.compile(r"^(?P<station>.+)_(?P<variable>t2m|wind_speed)$")
METHOD_LEAD_RE = re.compile(r"^(?P<method>[a-z_]+)_(?P<lead>\d+)h$")

# sweep.py's make_objective() call in main() uses the max_epochs=200 default --
# must match that to reproduce the recorded best_rmse.
SWEEP_MAX_EPOCHS = 200


def find_sweep_results_in_zips(lstm_dir, default_station, default_variable):
    """Yield one entry per sweep_results.json packed into lstm_training/*.zip.

    Reads straight from the zips: their internal path (<method>_<lead>h/...) is
    the only place method + lead_time are reliably encoded. Station/variable
    are NOT in the zips, so every combo here is tagged with the given
    defaults. The manually extracted 'climatology/' and 'ecmwifs/' folders
    next to the zips are ignored on purpose -- they duplicate (and in
    ecmwifs' case, mislabel: it holds gefs_mean data) what's already inside
    the zips.
    """
    combos = []
    for zip_name in sorted(os.listdir(lstm_dir)):
        if not zip_name.endswith(".zip"):
            continue
        zip_path = os.path.join(lstm_dir, zip_name)
        with zipfile.ZipFile(zip_path) as zf:
            for entry in zf.namelist():
                m = ENTRY_RE.match(entry.replace("\\", "/"))
                if not m:
                    continue
                method = m.group("method")
                lead = int(m.group("lead"))
                if method not in METHODS:
                    raise ValueError(f"{zip_name}:{entry} -- method '{method}' has no NPZ mapping in METHODS")
                with zf.open(entry) as f:
                    data = json.load(f)
                combos.append({
                    "station":     default_station,
                    "variable":    default_variable,
                    "method":      method,
                    "lead_time":   lead,
                    "best_params": data["best_params"],
                    "best_rmse":   data["best_rmse"],
                    "source":      f"{zip_name}:{entry}",
                })
    return combos


def find_sweep_results_in_output_dir(output_dir):
    """Yield one entry per sweep_results.json under
    output/<station>_<variable>/<method>_<lead>h/ -- run_all.py's own layout.
    Station and variable are read straight from the path, so this picks up
    any station (e.g. eric_d_soulis) once its sweep has been run.
    """
    combos = []
    if not os.path.isdir(output_dir):
        return combos
    for combo_folder in sorted(os.listdir(output_dir)):
        m1 = DIR_COMBO_RE.match(combo_folder)
        if not m1:
            continue
        station, variable = m1.group("station"), m1.group("variable")
        station_dir = os.path.join(output_dir, combo_folder)
        if not os.path.isdir(station_dir):
            continue
        for method_lead in sorted(os.listdir(station_dir)):
            m2 = METHOD_LEAD_RE.match(method_lead)
            if not m2:
                continue
            method, lead = m2.group("method"), int(m2.group("lead"))
            sr_path = os.path.join(station_dir, method_lead, "sweep_results.json")
            if not os.path.exists(sr_path):
                continue
            with open(sr_path) as f:
                data = json.load(f)
            combos.append({
                "station":     station,
                "variable":    variable,
                "method":      method,
                "lead_time":   lead,
                "best_params": data["best_params"],
                "best_rmse":   data["best_rmse"],
                "source":      os.path.relpath(sr_path, LSTM_DIR),
            })
    return combos


def find_sweep_results(lstm_dir, default_station, default_variable):
    """Merge both sources; on a (station, variable, method, lead) collision,
    the run_all.py output-dir source wins (self-describing, likely fresher).
    """
    zip_combos = find_sweep_results_in_zips(lstm_dir, default_station, default_variable)
    dir_combos = find_sweep_results_in_output_dir(os.path.join(lstm_dir, "output"))

    by_key = {}
    for c in zip_combos + dir_combos:  # dir_combos listed second so it overwrites on collision
        key = (c["station"], c["variable"], c["method"], c["lead_time"])
        by_key[key] = c
    return list(by_key.values())


def retrain_one(combo, data_dir, out_root, device, tag=""):
    station  = combo["station"]
    variable = combo["variable"]
    method   = combo["method"]
    lead     = combo["lead_time"]
    params   = combo["best_params"]

    npz_path = os.path.join(data_dir, os.path.relpath(METHODS[method], "benchmarking-site/data"))
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"NPZ not found for method '{method}': {npz_path}\n"
            f"Pass --data_dir pointing at the folder that contains "
            f"<method>/<method>_2025.npz for every method."
        )

    out_dir = os.path.join(out_root, f"{station}_{variable}", f"{method}_{lead}h")
    os.makedirs(out_dir, exist_ok=True)

    # --- Data loading + windowing: identical to train.py's main() ---
    bias, inits = load_bias_series(npz_path, station, variable, lead)
    valid = ~np.isnan(bias)
    bias  = bias[valid]
    inits = inits[valid]

    bias_mean = float(np.mean(bias))
    bias_std  = float(np.std(bias))
    if bias_std == 0.0:
        raise ValueError(f"Bias series has zero variance for {method}_{lead}h, cannot normalize")
    bias_norm = ((bias - bias_mean) / bias_std).astype(np.float32)

    hours, doys = parse_timestamps(inits)
    features = build_features(bias_norm, hours, doys)
    X, y, target_idx = make_sequences(features, params["seq_len"], inits, lead)

    target_timestamps = inits[target_idx]

    X_tr, y_tr, X_va, y_va, X_te, y_te, ts_tr, ts_va, ts_te = temporal_split(
        X, y, timestamps=target_timestamps
    )

    def make_loader(Xa, ya, shuffle=False):
        ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))
        return DataLoader(ds, batch_size=params["batch_size"], shuffle=shuffle)

    train_loader = make_loader(X_tr, y_tr, shuffle=True)
    val_loader   = make_loader(X_va, y_va)
    test_loader  = make_loader(X_te, y_te)

    print(f"{tag} samples={len(bias)}  sequences={len(X)}  "
          f"split train/val/test={len(X_tr)}/{len(X_va)}/{len(X_te)}", flush=True)

    # --- Model + optimizer: same as train.py, hyperparams from best_params ---
    model = BiasLSTM(
        input_size=5,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    # --- Training loop: identical structure to train.py's main() ---
    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses     = []
    val_losses       = []
    best_model_path  = os.path.join(out_dir, "best_model.pt")

    epochs_trained = 0
    for epoch in range(1, SWEEP_MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params["max_norm"])
            optimizer.step()
            running_loss += loss.item() * len(yb)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, _, _ = run_eval(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        epochs_trained = epoch

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{tag} epoch {epoch:4d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                print(f"{tag} early stop at epoch {epoch} (patience={params['patience']})", flush=True)
                break

    # --- Reload best weights, evaluate on test set: identical to train.py ---
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    _, preds_norm, targets_norm = run_eval(model, test_loader, criterion, device)
    metrics_norm = compute_metrics(preds_norm, targets_norm)

    preds_orig   = preds_norm   * bias_std + bias_mean
    targets_orig = targets_norm * bias_std + bias_mean
    metrics_orig = compute_metrics(preds_orig, targets_orig)

    np.savez(
        os.path.join(out_dir, "predictions.npz"),
        predictions=preds_orig,
        targets=targets_orig,
        timestamps=ts_te,
        train_losses=np.array(train_losses, dtype=np.float32),
        val_losses=np.array(val_losses, dtype=np.float32),
    )

    metrics = {
        "station":               station,
        "variable":              variable,
        "method":                method,
        "lead_time":             lead,
        "best_params":           params,
        "n_samples":             int(len(bias)),
        "n_sequences":           int(len(X)),
        "split":                 {"train": len(X_tr), "val": len(X_va), "test": len(X_te)},
        "epochs_trained":        epochs_trained,
        "best_val_loss":         float(best_val_loss),
        "metrics_normalized":    metrics_norm,
        "metrics_original":      metrics_orig,
        "original_sweep_best_rmse": combo["best_rmse"],
        "original_sweep_source":    combo["source"],
        "device":                str(device),
        "retrained_at":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    p = argparse.ArgumentParser(description="Retrain best-params models found in lstm_training/*.zip sweep results")
    p.add_argument("--station",   default="cyyz")
    p.add_argument("--variable",  default="t2m")
    p.add_argument("--leads",     type=int, nargs="+", default=[6, 12])
    p.add_argument("--data_dir",  default=os.path.join(REPO_ROOT, "benchmarking-site", "data"),
                   help="Folder containing <method>/<method>_2025.npz for every method")
    p.add_argument("--out_dir",   default=os.path.join(LSTM_DIR, "retrain_output"))
    p.add_argument("--smoke_test", action="store_true", help="Train only the first matching combo")
    p.add_argument("--all",        action="store_true", help="Train every matching combo (required beyond --smoke_test)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    all_combos = find_sweep_results(LSTM_DIR, args.station, args.variable)
    combos = [
        c for c in all_combos
        if c["station"] == args.station and c["variable"] == args.variable and c["lead_time"] in args.leads
    ]

    print(f"\nFound {len(all_combos)} sweep_results.json total (zips + lstm_training/output/)", flush=True)
    print(f"{len(combos)} combos match station={args.station}, variable={args.variable}, "
          f"lead_time in {args.leads}:", flush=True)
    for c in combos:
        print(f"  {c['station']}_{c['variable']}/{c['method']}_{c['lead_time']}h"
              f"  (sweep best_rmse={c['best_rmse']:.4f}, source={c['source']})", flush=True)

    if args.smoke_test:
        combos = combos[:1]
        if not combos:
            print("\nNo combos matched -- nothing to smoke test.", flush=True)
            return
        print(f"\nSMOKE TEST -- training only: {combos[0]['method']}_{combos[0]['lead_time']}h", flush=True)
    elif not args.all:
        print("\nNeither --smoke_test nor --all passed -- exiting without training anything.", flush=True)
        return
    else:
        print(f"\nTraining all {len(combos)} combos.", flush=True)

    total = len(combos)
    for i, c in enumerate(combos, start=1):
        tag = f"[{i}/{total}|{c['method']}_{c['lead_time']}h]"
        print(f"\n=== {tag} Retraining {c['station']}_{c['variable']}/{c['method']}_{c['lead_time']}h ===", flush=True)
        metrics = retrain_one(c, args.data_dir, args.out_dir, device, tag=tag)
        print(f"{tag} epochs trained:              {metrics['epochs_trained']}", flush=True)
        print(f"{tag} test RMSE (original scale):  {metrics['metrics_original']['rmse']:.4f}", flush=True)
        print(f"{tag} sweep best_rmse (original):  {c['best_rmse']:.4f}", flush=True)


if __name__ == "__main__":
    main()

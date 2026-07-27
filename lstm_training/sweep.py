import argparse
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import optuna
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


def make_objective(bias, bias_mean, bias_std, hours, doys, device, max_epochs=200):
    bias_norm = ((bias - bias_mean) / bias_std).astype("float32")
    features_full = build_features(bias_norm, hours, doys)

    def objective(trial):
        seq_len     = trial.suggest_categorical("seq_len",     [12, 24, 48, 72])
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        num_layers  = trial.suggest_categorical("num_layers",  [1, 2, 3])
        dropout     = trial.suggest_float("dropout", 0.0, 0.3)
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size",  [16, 32, 64])
        patience    = trial.suggest_categorical("patience",    [10, 15, 20, 25])
        max_norm    = trial.suggest_categorical("max_norm",    [0.5, 1.0, 2.0])

        X, y = make_sequences(features_full, seq_len)
        X_tr, y_tr, X_va, y_va, X_te, y_te = temporal_split(X, y)

        def loader(Xa, ya, shuffle=False):
            ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_loader = loader(X_tr, y_tr, shuffle=True)
        val_loader   = loader(X_va, y_va)
        test_loader  = loader(X_te, y_te)

        model = BiasLSTM(5, hidden_size, num_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val = float("inf")
        patience_counter = 0
        fd, tmp_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)

        try:
            for epoch in range(1, max_epochs + 1):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()

                val_loss, _, _ = run_eval(model, val_loader, criterion, device)
                scheduler.step(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), tmp_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            model.load_state_dict(torch.load(tmp_path, map_location=device, weights_only=True))
            _, preds_norm, targets_norm = run_eval(model, test_loader, criterion, device)
            preds_orig   = preds_norm   * bias_std + bias_mean
            targets_orig = targets_norm * bias_std + bias_mean
            return compute_metrics(preds_orig, targets_orig)["rmse"]
        finally:
            os.unlink(tmp_path)

    return objective


def main():
    p = argparse.ArgumentParser(description="Optuna sweep for LSTM bias correction")
    p.add_argument("--npz",       required=True)
    p.add_argument("--station",   default="cyyz")
    p.add_argument("--variable",  default="t2m")
    p.add_argument("--lead_time", type=int, default=6)
    p.add_argument("--n_trials",  type=int, default=100)
    p.add_argument("--out_dir",   default="lstm_training/output")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bias, inits = load_bias_series(args.npz, args.station, args.variable, args.lead_time)
    valid = ~np.isnan(bias)
    bias  = bias[valid]
    inits = inits[valid]
    print(f"Samples: {len(bias)} valid")

    bias_mean = float(np.mean(bias))
    bias_std  = float(np.std(bias))
    hours, doys = parse_timestamps(inits)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")

    def callback(study, trial):
        print(f"Trial {trial.number:3d}  RMSE={trial.value:.4f}  best={study.best_value:.4f}  {trial.params}")

    objective = make_objective(bias, bias_mean, bias_std, hours, doys, device)
    study.optimize(objective, n_trials=args.n_trials, callbacks=[callback])

    best = study.best_trial
    print(f"\nBest RMSE: {best.value:.4f}")
    print(f"Best params: {best.params}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = {
        "best_rmse":   best.value,
        "best_params": best.params,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ],
        "swept_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path = os.path.join(args.out_dir, "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

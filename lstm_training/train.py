import argparse
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bias_series(npz_path, station, variable, lead_time):
    data = np.load(npz_path, allow_pickle=True)
    si = list(data["station_ids"]).index(station)
    vi = list(data["variables"]).index(variable)
    li = list(data["lead_times_hours"]).index(lead_time)
    bias = data["bias"][:, si, vi, li].astype(np.float32)
    inits = data["initializations"].copy()
    data.close()
    return bias, inits


def parse_timestamps(inits):
    hours, doys = [], []
    for s in inits:
        dt = datetime.strptime(s.rstrip("Z"), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        hours.append(dt.hour)
        doys.append(dt.timetuple().tm_yday)
    return np.array(hours, dtype=np.float32), np.array(doys, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(bias_norm, hours, doys):
    """5 features per timestep: normalized bias + sin/cos hour + sin/cos doy."""
    sin_hour = np.sin(2 * math.pi * hours / 24).astype(np.float32)
    cos_hour = np.cos(2 * math.pi * hours / 24).astype(np.float32)
    sin_doy  = np.sin(2 * math.pi * doys / 365).astype(np.float32)
    cos_doy  = np.cos(2 * math.pi * doys / 365).astype(np.float32)
    return np.stack([bias_norm, sin_hour, cos_hour, sin_doy, cos_doy], axis=1)


def make_sequences(features, seq_len):
    """Sliding window. X: (N, seq_len, 5)  y: (N,) next-step bias (normalized)."""
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(features[i + seq_len, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def temporal_split(X, y, timestamps=None, train_frac=0.70, val_frac=0.15):
    n = len(X)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    parts = (
        X[:n_tr],          y[:n_tr],
        X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va],
        X[n_tr+n_va:],     y[n_tr+n_va:],
    )
    if timestamps is None:
        return parts
    return parts + (
        timestamps[:n_tr],
        timestamps[n_tr:n_tr+n_va],
        timestamps[n_tr+n_va:],
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiasLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_list, targets_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * len(yb)
            preds_list.append(pred.cpu().numpy())
            targets_list.append(yb.cpu().numpy())
    preds   = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)
    return total_loss / len(loader.dataset), preds, targets


def compute_metrics(preds, targets):
    diff = preds - targets
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae":  float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LSTM bias correction trainer")
    p.add_argument("--npz",         required=True,      help="Path to benchmarking NPZ file")
    p.add_argument("--station",     default="cyyz",     help="Station id (cyyz or eric_d_soulis)")
    p.add_argument("--variable",    default="t2m",      help="Variable (t2m or wind_speed)")
    p.add_argument("--lead_time",   type=int, default=6, help="Lead time in hours")
    p.add_argument("--seq_len",     type=int, default=24)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers",  type=int, default=2)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--max_epochs",  type=int, default=200)
    p.add_argument("--patience",    type=int, default=15)
    p.add_argument("--max_norm",    type=float, default=1.0, help="Gradient clipping max norm")
    p.add_argument("--out_dir",     default="lstm_training/output")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load + drop NaN
    bias, inits = load_bias_series(args.npz, args.station, args.variable, args.lead_time)
    valid = ~np.isnan(bias)
    n_dropped = int((~valid).sum())
    bias  = bias[valid]
    inits = inits[valid]
    print(f"Samples: {len(bias)} valid ({n_dropped} NaN dropped)")

    # Z-score normalize
    bias_mean = float(np.mean(bias))
    bias_std  = float(np.std(bias))
    if bias_std == 0.0:
        raise ValueError("Bias series has zero variance, cannot normalize")
    bias_norm = ((bias - bias_mean) / bias_std).astype(np.float32)

    # Features and sequences
    hours, doys = parse_timestamps(inits)
    features = build_features(bias_norm, hours, doys)
    X, y = make_sequences(features, args.seq_len)
    print(f"Sequences: {len(X)}  seq_len={args.seq_len}  features=5")

    # y[i] is the bias at inits[i + seq_len], so target timestamps are inits shifted by seq_len
    target_timestamps = inits[args.seq_len:]

    # Temporal split
    X_tr, y_tr, X_va, y_va, X_te, y_te, ts_tr, ts_va, ts_te = temporal_split(
        X, y, timestamps=target_timestamps
    )
    print(f"Split -- train: {len(X_tr)}  val: {len(X_va)}  test: {len(X_te)}")

    def make_loader(Xa, ya, shuffle=False):
        ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(X_tr, y_tr, shuffle=True)
    val_loader   = make_loader(X_va, y_va)
    test_loader  = make_loader(X_te, y_te)

    # Model + optimizer
    model = BiasLSTM(
        input_size=5,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses     = []
    val_losses       = []
    best_model_path  = os.path.join(args.out_dir, "best_model.pt")

    print(f"\nTraining (max {args.max_epochs} epochs, patience={args.patience})")
    print("-" * 60)

    epochs_trained = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            running_loss += loss.item() * len(yb)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, _, _ = run_eval(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        epochs_trained = epoch

        if epoch % 5 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}  "
                f"train={train_loss:.4f}  "
                f"val={val_loss:.4f}  "
                f"lr={lr_now:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stop at epoch {epoch} (patience={args.patience})")
                break

    # Reload best weights and evaluate on test set
    print("-" * 60)
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    _, preds_norm, targets_norm = run_eval(model, test_loader, criterion, device)

    metrics_norm = compute_metrics(preds_norm, targets_norm)

    preds_orig   = preds_norm   * bias_std + bias_mean
    targets_orig = targets_norm * bias_std + bias_mean
    metrics_orig = compute_metrics(preds_orig, targets_orig)

    baseline = compute_metrics(np.zeros_like(targets_orig), targets_orig)

    print("\nTest results -- normalized scale:")
    print(f"  RMSE={metrics_norm['rmse']:.4f}  MAE={metrics_norm['mae']:.4f}  bias={metrics_norm['bias']:.4f}")
    print("\nTest results -- original scale:")
    print(f"  RMSE={metrics_orig['rmse']:.4f}  MAE={metrics_orig['mae']:.4f}  bias={metrics_orig['bias']:.4f}")
    print("\nBaseline (predict zero):")
    print(f"  RMSE={baseline['rmse']:.4f}  MAE={baseline['mae']:.4f}  bias={baseline['bias']:.4f}")

    # Save config
    config = {
        "npz":         args.npz,
        "station":     args.station,
        "variable":    args.variable,
        "lead_time":   args.lead_time,
        "seq_len":     args.seq_len,
        "hidden_size": args.hidden_size,
        "num_layers":  args.num_layers,
        "dropout":     args.dropout,
        "lr":          args.lr,
        "batch_size":  args.batch_size,
        "max_epochs":  args.max_epochs,
        "patience":    args.patience,
        "max_norm":    args.max_norm,
        "normalization": {"mean": bias_mean, "std": bias_std},
        "n_samples":   int(len(bias)),
        "n_sequences": int(len(X)),
        "split":       {"train": len(X_tr), "val": len(X_va), "test": len(X_te)},
        "epochs_trained":  epochs_trained,
        "best_val_loss":   float(best_val_loss),
        "device":          str(device),
        "metrics_normalized": metrics_norm,
        "metrics_original":   metrics_orig,
        "baseline_metrics":   baseline,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save predictions + loss curves
    np.savez(
        os.path.join(args.out_dir, "predictions.npz"),
        predictions=preds_orig,
        targets=targets_orig,
        timestamps=ts_te,
        train_losses=np.array(train_losses, dtype=np.float32),
        val_losses=np.array(val_losses, dtype=np.float32),
    )

    print(f"\nArtifacts saved to {args.out_dir}/")
    print(f"  best_model.pt")
    print(f"  config.json")
    print(f"  predictions.npz")


if __name__ == "__main__":
    main()

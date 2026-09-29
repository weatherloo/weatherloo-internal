"""Adapter for the existing LSTM bias-correction model.

All model-specific logic -- the BiasLSTM architecture, the training loop,
early stopping -- is exactly what lstm_training/train.py already did. This
file only wraps it behind the five methods the framework calls; it does not
change how the model trains.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lstm_training.train import (
    BiasLSTM,
    build_features,
    compute_metrics,
    load_bias_series,
    make_sequences,
    parse_timestamps,
    run_eval,
    temporal_split,
)
from training_framework.registry import register


@register("lstm_bias_correction")
class LstmBiasCorrectionAdapter:

    # -- dry-run: cheap checks, no model, no full training arrays kept around --
    @staticmethod
    def validate(cfg):
        data_cfg = cfg["data"]
        npz_path = data_cfg["npz"]
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        npz = np.load(npz_path, allow_pickle=True)
        missing = [k for k in ("bias", "station_ids", "variables", "lead_times_hours", "initializations")
                   if k not in npz.files]
        if missing:
            raise ValueError(f"{npz_path}: missing array(s) {missing}")

        station_ids = list(npz["station_ids"])
        variables = list(npz["variables"])
        lead_times = list(npz["lead_times_hours"])
        problems = []
        if data_cfg["station"] not in station_ids:
            problems.append(f"station '{data_cfg['station']}' not in {station_ids}")
        if data_cfg["variable"] not in variables:
            problems.append(f"variable '{data_cfg['variable']}' not in {variables}")
        if data_cfg["lead_time"] not in lead_times:
            problems.append(f"lead_time {data_cfg['lead_time']} not in {lead_times}")
        if problems:
            raise ValueError(f"{npz_path}: " + "; ".join(problems))

        bias, inits = load_bias_series(npz_path, data_cfg["station"], data_cfg["variable"], data_cfg["lead_time"])
        n_valid = int((~np.isnan(bias)).sum())
        seq_len = cfg["model"].get("seq_len", 24)
        n_sequences = max(0, n_valid - seq_len)
        if n_sequences < 10:
            raise ValueError(
                f"Only {n_valid} valid (non-NaN) samples -> {n_sequences} sequences at "
                f"seq_len={seq_len}; need at least 10 to train."
            )
        return {
            "npz": npz_path,
            "n_inits_total": int(len(inits)),
            "n_valid_samples": n_valid,
            "n_sequences_est": n_sequences,
        }

    @staticmethod
    def load_dataset(cfg):
        data_cfg, model_cfg = cfg["data"], cfg["model"]
        bias, inits = load_bias_series(
            data_cfg["npz"], data_cfg["station"], data_cfg["variable"], data_cfg["lead_time"]
        )
        valid = ~np.isnan(bias)
        bias, inits = bias[valid], inits[valid]

        bias_mean = float(np.mean(bias))
        bias_std = float(np.std(bias))
        if bias_std == 0.0:
            raise ValueError("Bias series has zero variance, cannot normalize")
        bias_norm = ((bias - bias_mean) / bias_std).astype(np.float32)

        hours, doys = parse_timestamps(inits)
        features = build_features(bias_norm, hours, doys)
        seq_len = model_cfg.get("seq_len", 24)
        X, y = make_sequences(features, seq_len)
        target_ts = inits[seq_len:]

        X_tr, y_tr, X_va, y_va, X_te, y_te, ts_tr, ts_va, ts_te = temporal_split(
            X, y, timestamps=target_ts
        )
        return {
            "train": (X_tr, y_tr),
            "val": (X_va, y_va),
            "test": (X_te, y_te, ts_te),
            "normalization": {"mean": bias_mean, "std": bias_std},
            "n_samples": int(len(bias)),
            "n_sequences": int(len(X)),
        }

    @staticmethod
    def build_model(cfg):
        m = cfg["model"]
        return BiasLSTM(
            input_size=5,
            hidden_size=m.get("hidden_size", 64),
            num_layers=m.get("num_layers", 2),
            dropout=m.get("dropout", 0.1),
        )

    @staticmethod
    def fit(cfg, dataset, run_dir):
        m = cfg["model"]
        seed = cfg.get("run", {}).get("seed", 0)
        torch.manual_seed(seed)
        np.random.seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LstmBiasCorrectionAdapter.build_model(cfg).to(device)

        def make_loader(X, y, shuffle=False):
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            return DataLoader(ds, batch_size=m.get("batch_size", 32), shuffle=shuffle)

        X_tr, y_tr = dataset["train"]
        X_va, y_va = dataset["val"]
        X_te, y_te, ts_te = dataset["test"]
        train_loader = make_loader(X_tr, y_tr, shuffle=True)
        val_loader = make_loader(X_va, y_va)
        test_loader = make_loader(X_te, y_te)

        optimizer = torch.optim.Adam(model.parameters(), lr=m.get("lr", 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        max_epochs = m.get("max_epochs", 200)
        patience = m.get("patience", 15)
        max_norm = m.get("max_norm", 1.0)

        best_val = float("inf")
        patience_counter = 0
        train_losses, val_losses = [], []
        checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
        epochs_trained = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                running += loss.item() * len(yb)

            train_loss = running / len(train_loader.dataset)
            val_loss, _, _ = run_eval(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            epochs_trained = epoch

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        _, preds_norm, targets_norm = run_eval(model, test_loader, criterion, device)
        metrics_norm = compute_metrics(preds_norm, targets_norm)

        mean, std = dataset["normalization"]["mean"], dataset["normalization"]["std"]
        preds_orig = preds_norm * std + mean
        targets_orig = targets_norm * std + mean
        metrics_orig = compute_metrics(preds_orig, targets_orig)
        baseline = compute_metrics(np.zeros_like(targets_orig), targets_orig)

        np.savez(
            os.path.join(run_dir, "predictions.npz"),
            predictions=preds_orig,
            targets=targets_orig,
            timestamps=ts_te,
            train_losses=np.array(train_losses, dtype=np.float32),
            val_losses=np.array(val_losses, dtype=np.float32),
        )

        return {
            "epochs_trained": epochs_trained,
            "best_val_loss": float(best_val),
            "device": str(device),
            "metrics_normalized": metrics_norm,
            "metrics_original": metrics_orig,
            "baseline_metrics": baseline,
            "normalization": dataset["normalization"],
        }

    @staticmethod
    def load_checkpoint(cfg, checkpoint_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LstmBiasCorrectionAdapter.build_model(cfg).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        return model, device

    @staticmethod
    def predict(cfg, model, device, dataset, normalization):
        """Forward pass over the held-out (most recent) split -- the `infer`
        verb's job is to score a checkpoint on data it didn't train on."""
        X_te, _, ts_te = dataset["test"]
        loader = DataLoader(TensorDataset(torch.from_numpy(X_te)), batch_size=128)
        preds_norm = []
        with torch.no_grad():
            for (xb,) in loader:
                preds_norm.append(model(xb.to(device)).cpu().numpy())
        preds_norm = np.concatenate(preds_norm)
        preds = preds_norm * normalization["std"] + normalization["mean"]
        return preds, ts_te

#!/usr/bin/env python3
"""U-Net (GFS post-processing) forecast for a single initialization.

Pulls the GFS 0.25 deg region grid for each lead, runs the trained residual
U-Net, and interpolates the corrected field to the station.

The network predicts the GFS-minus-truth residual, so the corrected field is
``GFS - residual`` -- the same sign convention ``models/unet/evaluate.py``
uses. Normalization comes from the stats embedded in the checkpoint rather
than ``data/stats.json``, which has since drifted from what this checkpoint
was trained with.

Leads are restricted to the checkpoint's own ``sample_space``; the lead is an
input channel, so running off that grid would be extrapolation the trained
weights never saw.

    python scripts/showdown/run_unet.py --init 2026-07-20T00Z
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
UNET = ROOT / "models" / "unet"
sys.path.insert(0, str(UNET))
sys.path.insert(0, str(UNET / "data"))
sys.path.insert(0, str(UNET / "model"))


def parse_init(text: str) -> datetime:
    text = text.strip()
    for fmt in ("%Y-%m-%dT%H%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            cleaned = text.replace("Z", "+0000") if "Z" in text else text
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unrecognised init: {text!r}")


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def interp_point(lats: np.ndarray, lons: np.ndarray, field: np.ndarray,
                 lat: float, lon: float) -> float:
    """Bilinear sample of one (H,W) field, tolerating a descending lat axis."""
    if lats[0] > lats[-1]:
        lats, field = lats[::-1], field[::-1, :]
    lon_q = lon % 360.0
    return float(RegularGridInterpolator((lats, lons), field,
                                         bounds_error=False, fill_value=None)((lat, lon_q)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", default="2026-07-20T00Z")
    p.add_argument("--station", default="eric_d_soulis")
    p.add_argument("--checkpoint", default=str(UNET / "checkpoints" / "best_model.pt"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from dataset import build_model_input, denormalize_residual, load_config
    from fetch_gfs import gfs_region_grid
    from unet import model_from_checkpoint

    cfg = load_config()
    station = cfg["stations"][args.station]
    lat, lon = station["lat"], station["lon"]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    model = model_from_checkpoint(ckpt)
    model.eval()

    leads = list(ckpt["sample_space"]["leads"])
    init = parse_init(args.init)
    date = f"{init:%Y%m%d}"
    cycle = init.hour

    gfs_mean = np.asarray(stats["gfs"]["mean"], np.float32).reshape(-1, 1, 1)
    gfs_std = np.asarray(stats["gfs"]["std"], np.float32).reshape(-1, 1, 1)

    steps: list[dict] = []
    for lead in leads:
        valid = init + timedelta(hours=lead)
        try:
            grid = gfs_region_grid(cfg, date, cycle, lead)
        except Exception as exc:  # noqa: BLE001 - a missing GFS cycle must not sink the run
            print(f"  lead {lead:>3}h  GFS fetch failed: {exc}", file=sys.stderr)
            steps.append({"lead_hours": lead, "valid_time": iso(valid),
                          "value": None, "reason": "gfs_unavailable"})
            continue

        lats = grid["latitude"].values.astype(np.float64)
        lons = grid["longitude"].values.astype(np.float64)
        gfs = np.stack([grid[c].values for c in ("t2m", "u10", "v10")]).astype(np.float32)

        x = build_model_input((gfs - gfs_mean) / gfs_std, float(lead))
        with torch.no_grad():
            pred = model(torch.from_numpy(x[None].astype(np.float32))).numpy()[0]
        corrected = gfs - denormalize_residual(pred, stats)

        raw_t2m = interp_point(lats, lons, gfs[0], lat, lon)
        cor_t2m = interp_point(lats, lons, corrected[0], lat, lon)
        steps.append({
            "lead_hours": lead,
            "valid_time": iso(valid),
            "value": round(cor_t2m, 4),
            "raw_source_value": round(raw_t2m, 4),
        })
        print(f"  lead {lead:>3}h  {cor_t2m:.2f} degC (raw GFS {raw_t2m:.2f})", file=sys.stderr)

    doc = {
        "schema_version": "1",
        "model": "unet",
        "model_label": "U-Net",
        "base_source": "gfs",
        "station_id": args.station,
        "variable": "t2m",
        "units": "degC",
        "init": iso(init),
        "cadence_hours": 6,
        "provenance": {
            "code": "models/unet (evaluate.py inference path)",
            "checkpoint": str(Path(args.checkpoint).relative_to(ROOT)),
            "trained_epoch": ckpt.get("epoch"),
            "val_loss_normalized_mse": ckpt.get("val_loss"),
            "input": "GFS 0.25deg t2m/u10/v10 over the southern Ontario box, AWS noaa-gfs-bdp-pds byte-range GRIB",
            "correction": "corrected = GFS - predicted_residual, bilinear to station",
        },
        "steps": steps,
    }

    out = Path(args.out) if args.out else ROOT / "benchmarking-site" / "data" / "showdown" / f"unet_{init:%Y%m%dT%HZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    got = sum(1 for s in steps if s["value"] is not None)
    print(f"Wrote {out} ({got}/{len(leads)} leads)", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CNN-LSTM (HRRR bias correction) forecast for a single initialization.

Rebuilds the exact tensor ``src/hrrr_bias_correction/make_data.py`` produces --
``(49, 30, 30, 3)`` of HRRR ``t2m``/``u10``/``v10`` in **native units**
(Kelvin, m/s; the training pipeline does not normalize) on a 30x30 crop centred
on the grid point nearest the station -- then runs the trained Keras model to
get a per-lead bias and subtracts it from the interpolated HRRR value.

The network's target is ``forecast - observation``, so the corrected forecast
is ``HRRR - predicted_bias``.

HRRR fields come from AWS Open Data by byte-range GRIB request, the same
approach the GFS fetcher uses; only the three needed messages are pulled from
each of the 49 hourly files. Downloads are cached under ``.cache/hrrr_grib/``.

    python scripts/showdown/run_cnn_lstm.py --init 2026-07-20T00Z
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "hrrr_bias_correction"))

AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
CACHE = ROOT / ".cache" / "hrrr_grib"

N_LEADS = 49
CROP = 30
HALF = CROP // 2
K_TO_C = 273.15

NEEDLES = {
    "t2m": ":TMP:2 m above ground:",
    "u10": ":UGRD:10 m above ground:",
    "v10": ":VGRD:10 m above ground:",
}

STATION = {"eric_d_soulis": (43.4668, -80.5164)}


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


def _get(url: str, byte_range: tuple[int, int | None] | None = None,
         retries: int = 5) -> bytes:
    for attempt in range(retries):
        req = urllib.request.Request(url)
        if byte_range is not None:
            lo, hi = byte_range
            req.add_header("Range", f"bytes={lo}-{'' if hi is None else hi}")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == retries - 1:
                raise
            import time
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_hrrr_fields(init: datetime, fxx: int) -> Path:
    """Download the three needed GRIB messages for one lead into one file."""
    key = f"hrrr.{init:%Y%m%d}/conus/hrrr.t{init.hour:02d}z.wrfsfcf{fxx:02d}.grib2"
    out = CACHE / f"{init:%Y%m%dT%HZ}_f{fxx:02d}.grib2"
    if out.exists() and out.stat().st_size > 0:
        return out
    CACHE.mkdir(parents=True, exist_ok=True)

    idx_text = _get(f"{AWS_BASE}/{key}.idx").decode("utf-8", "replace")
    lines = [ln for ln in idx_text.splitlines() if ln.strip()]
    offsets: list[int] = []
    for ln in lines:
        parts = ln.split(":")
        if len(parts) > 1:
            offsets.append(int(parts[1]))

    blobs: list[bytes] = []
    for name, needle in NEEDLES.items():
        hit = next((k for k, ln in enumerate(lines) if needle in ln), None)
        if hit is None:
            raise RuntimeError(f"{needle} not in idx for f{fxx:02d}")
        lo = offsets[hit]
        hi = offsets[hit + 1] - 1 if hit + 1 < len(offsets) else None
        blobs.append(_get(f"{AWS_BASE}/{key}", (lo, hi)))

    tmp = out.with_suffix(".part")
    tmp.write_bytes(b"".join(blobs))
    tmp.replace(out)
    return out


def open_lead(path: Path) -> dict:
    """Read the three fields + grid from a byte-range GRIB file."""
    import cfgrib
    out: dict[str, np.ndarray] = {}
    for ds in cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}):
        for src, dst in (("t2m", "t2m"), ("u10", "u10"), ("v10", "v10")):
            if src in ds.data_vars and dst not in out:
                out[dst] = np.asarray(ds[src].values, dtype=np.float32)
                out.setdefault("latitude", np.asarray(ds["latitude"].values, np.float64))
                lon = np.asarray(ds["longitude"].values, np.float64)
                out.setdefault("longitude", np.where(lon > 180.0, lon - 360.0, lon))
    missing = [v for v in ("t2m", "u10", "v10") if v not in out]
    if missing:
        raise RuntimeError(f"missing {missing} in {path.name}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", default="2026-07-20T00Z")
    p.add_argument("--station", default="eric_d_soulis")
    p.add_argument("--model", default=str(ROOT / "artifacts" / "hrrr_bias_correction" / "final_model.keras"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from make_data import bilinear_curvilinear, crop_window, nearest_grid_index

    init = parse_init(args.init)
    lat0, lon0 = STATION[args.station]

    x = np.full((N_LEADS, CROP, CROP, 3), np.nan, dtype=np.float32)
    hrrr_point: list[float | None] = [None] * N_LEADS
    lat_c = lon_c = None

    for fxx in range(N_LEADS):
        try:
            path = fetch_hrrr_fields(init, fxx)
            fields = open_lead(path)
        except Exception as exc:  # noqa: BLE001 - one absent lead must not sink the run
            print(f"  f{fxx:02d} unavailable: {exc}", file=sys.stderr)
            continue

        if lat_c is None:
            lat2d, lon2d = fields["latitude"], fields["longitude"]
            j, i = nearest_grid_index(lat2d, lon2d, lat0, lon0)
            win = crop_window(j, i, lat2d.shape[0], lat2d.shape[1])
            if win is None:
                raise SystemExit("station too close to the HRRR grid edge for a 30x30 crop")
            jsl, isl = win
            lat_c, lon_c = lat2d[jsl, isl], lon2d[jsl, isl]

        t2m = fields["t2m"][jsl, isl]
        u10 = fields["u10"][jsl, isl]
        v10 = fields["v10"][jsl, isl]
        x[fxx, :, :, 0] = t2m
        x[fxx, :, :, 1] = u10
        x[fxx, :, :, 2] = v10
        hrrr_point[fxx] = bilinear_curvilinear(lat_c, lon_c, t2m, lat0, lon0) - K_TO_C

    got = int(np.isfinite(x[:, 0, 0, 0]).sum())
    print(f"  built X with {got}/{N_LEADS} leads", file=sys.stderr)
    if got == 0:
        raise SystemExit("no HRRR leads available for this init")

    # Training filled absent leads with 0.0 (config data.fill_x_nan).
    x_in = np.nan_to_num(x, nan=0.0)[None]

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model, compile=False)
    bias = np.asarray(model.predict(x_in, verbose=0))[0]  # (49, 2)

    steps: list[dict] = []
    for fxx in range(N_LEADS):
        valid = init + timedelta(hours=fxx)
        raw = hrrr_point[fxx]
        if raw is None or not np.isfinite(bias[fxx, 0]):
            steps.append({"lead_hours": fxx, "valid_time": iso(valid),
                          "value": None, "reason": "hrrr_lead_unavailable"})
            continue
        steps.append({
            "lead_hours": fxx,
            "valid_time": iso(valid),
            "value": round(float(raw - bias[fxx, 0]), 4),
            "raw_source_value": round(float(raw), 4),
            "predicted_bias": round(float(bias[fxx, 0]), 4),
        })

    doc = {
        "schema_version": "1",
        "model": "cnn_lstm",
        "model_label": "CNN-LSTM",
        "base_source": "hrrr",
        "station_id": args.station,
        "variable": "t2m",
        "units": "degC",
        "init": iso(init),
        "cadence_hours": 1,
        "provenance": {
            "code": "src/hrrr_bias_correction (make_data.py tensor layout)",
            "checkpoint": str(Path(args.model).relative_to(ROOT)),
            "input": "HRRR 3km t2m/u10/v10, 30x30 crop centred on the station, native units",
            "correction": "corrected = HRRR - predicted_bias (target is forecast - observation)",
        },
        "steps": steps,
    }

    out = Path(args.out) if args.out else ROOT / "benchmarking-site" / "data" / "showdown" / f"cnn_lstm_{init:%Y%m%dT%HZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    ok = sum(1 for s in steps if s["value"] is not None)
    print(f"Wrote {out} ({ok}/{N_LEADS} leads)", file=sys.stderr)


if __name__ == "__main__":
    main()

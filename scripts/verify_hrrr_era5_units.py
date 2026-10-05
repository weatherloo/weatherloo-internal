#!/usr/bin/env python3
"""Sanity-check patched HRRR/ERA5 NetCDFs: vars, units, and rough HRRR↔ERA5 diffs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

REQ = ("t2m", "u10", "v10", "tp", "q2", "psfc", "pblh", "hgt", "tsk", "ust")

# Expected physical ranges (bbox KW); loose so we catch unit mistakes, not weather.
RANGES = {
    "t2m": (230.0, 330.0),  # K
    "tsk": (230.0, 340.0),  # K
    "u10": (-50.0, 50.0),  # m/s
    "v10": (-50.0, 50.0),
    "q2": (0.0, 0.05),  # kg/kg mixing ratio
    "psfc": (85000.0, 105000.0),  # Pa
    "pblh": (0.0, 5000.0),  # m
    "hgt": (50.0, 800.0),  # m a.s.l. for KW region (includes low valleys)
    "ust": (0.0, 5.0),  # m/s
}

# HRRR tp: kg/m^2 (~mm) per hour; ERA5 tp: m per hour
TP_HRRR = (0.0, 100.0)
TP_ERA5 = (-1e-8, 0.1)  # allow tiny float noise around 0


def check_ranges(name: str, da: xr.DataArray, lo: float, hi: float) -> list[str]:
    errs: list[str] = []
    vals = np.asarray(da.values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        errs.append(f"{name}: all-NaN")
        return errs
    mn, mx, med = float(finite.min()), float(finite.max()), float(np.median(finite))
    if mn < lo or mx > hi:
        errs.append(f"{name}: range [{mn:.4g},{mx:.4g}] outside [{lo},{hi}] (median={med:.4g})")
    else:
        print(f"  OK {name}: median={med:.4g} min={mn:.4g} max={mx:.4g}")
    return errs


def verify_hrrr(path: Path) -> list[str]:
    errs: list[str] = []
    print(f"\n=== HRRR {path} ===")
    with xr.open_dataset(path) as ds:
        miss = [v for v in REQ if v not in ds.data_vars]
        if miss:
            return [f"missing vars: {miss}"]
        if ds["tp"].attrs.get("accum") != "1-hour":
            errs.append(f"tp.accum={ds['tp'].attrs.get('accum')!r} expected '1-hour'")
        else:
            print("  OK tp.accum=1-hour")
        for v, (lo, hi) in RANGES.items():
            errs.extend(check_ranges(v, ds[v], lo, hi))
        errs.extend(check_ranges("tp", ds["tp"], *TP_HRRR))
        # Hourly tp should not look like multi-day accumulated totals
        tp = np.asarray(ds["tp"].values)
        if np.nanmax(tp) > 200:
            errs.append(f"tp max={np.nanmax(tp):.4g} looks like still-accumulated APCP")
    return errs


def verify_era5(path: Path) -> list[str]:
    errs: list[str] = []
    print(f"\n=== ERA5 {path} ===")
    with xr.open_dataset(path) as ds:
        miss = [v for v in REQ if v not in ds.data_vars]
        if miss:
            return [f"missing vars: {miss}"]
        for v, (lo, hi) in RANGES.items():
            errs.extend(check_ranges(v, ds[v], lo, hi))
        errs.extend(check_ranges("tp", ds["tp"], *TP_ERA5))
    return errs


def compare_sources(hrrr_path: Path, era5_path: Path, init_utc: datetime) -> list[str]:
    """Compare HRRR f00 domain mean vs ERA5 nearest hour (same valid time)."""
    errs: list[str] = []
    print(f"\n=== HRRR f00 vs ERA5 @ {init_utc.isoformat()} ===")
    with xr.open_dataset(hrrr_path) as h, xr.open_dataset(era5_path) as e:
        # HRRR f00 (or first lead)
        lead0 = int(h.lead.values[0])
        h0 = h.isel(lead=0)
        # ERA5 nearest hour
        t = np.datetime64(init_utc.replace(tzinfo=None))
        e0 = e.sel(time=t, method="nearest")
        et = np.datetime64(e0.time.values)
        dt_h = abs((et - t) / np.timedelta64(1, "h"))
        if dt_h > 0.6:
            errs.append(f"ERA5 nearest time off by {dt_h:.2f} h")
        print(f"  HRRR lead={lead0}h  ERA5 time={et} (Δ={dt_h:.2f}h)")

        # Spatial means (different grids — only order-of-magnitude / unit checks)
        pairs = {
            "t2m": 5.0,  # K
            "tsk": 8.0,
            "u10": 5.0,
            "v10": 5.0,
            "q2": 0.005,
            "psfc": 1500.0,  # Pa
            "pblh": 800.0,
            "hgt": 80.0,
            "ust": 0.5,
        }
        for v, tol in pairs.items():
            hm = float(np.nanmean(h0[v].values))
            em = float(np.nanmean(e0[v].values))
            d = abs(hm - em)
            status = "OK" if d <= tol else "WARN"
            print(f"  {status} {v}: HRRR={hm:.4g} ERA5={em:.4g} |Δ|={d:.4g} (tol {tol})")
            if d > tol * 3:  # hard fail only on egregious unit mismatches
                errs.append(f"{v}: |HRRR-ERA5|={d:.4g} (HRRR={hm:.4g}, ERA5={em:.4g})")

        # tp: convert ERA5 m -> mm (= kg/m^2) for comparison
        htp = float(np.nanmean(h0["tp"].values))
        etp_mm = float(np.nanmean(e0["tp"].values)) * 1000.0
        print(f"  INFO tp: HRRR={htp:.4g} kg/m2  ERA5={etp_mm:.4g} mm (=m*1000)")
        if etp_mm > 50 and htp < 0.01:
            errs.append("tp units look inverted (ERA5 mm huge while HRRR near 0)")
        if htp > 50 and etp_mm < 0.01:
            errs.append("tp: HRRR still looks accumulated vs ERA5 hourly")
    return errs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--hrrr", default=None, help="HRRR cycle NetCDF path")
    p.add_argument("--era5", default=None, help="ERA5 month NetCDF path")
    p.add_argument("--init", default=None, help="Init UTC ISO e.g. 2018-07-13T12:00:00Z")
    args = p.parse_args()
    root = Path(args.data_root)
    errs: list[str] = []

    hrrr = Path(args.hrrr) if args.hrrr else None
    era5 = Path(args.era5) if args.era5 else None
    if hrrr is None:
        # pick first complete-looking file under data root
        for cand in sorted((root / "hrrr").rglob("hrrr_*_t*z.nc")):
            try:
                with xr.open_dataset(cand) as ds:
                    if all(v in ds.data_vars for v in REQ) and ds["tp"].attrs.get("accum") == "1-hour":
                        hrrr = cand
                        break
            except Exception:
                continue
    if era5 is None:
        for cand in sorted((root / "era5").rglob("era5_*.nc")):
            try:
                with xr.open_dataset(cand) as ds:
                    if all(v in ds.data_vars for v in REQ):
                        era5 = cand
                        break
            except Exception:
                continue

    if hrrr:
        errs.extend(verify_hrrr(hrrr))
    else:
        errs.append("no patched HRRR file found")
    if era5:
        errs.extend(verify_era5(era5))
    else:
        errs.append("no patched ERA5 file found")

    if hrrr and era5 and args.init:
        init = datetime.fromisoformat(args.init.replace("Z", "+00:00")).astimezone(timezone.utc)
        errs.extend(compare_sources(hrrr, era5, init))
    elif hrrr and era5:
        # infer init from filename hrrr_YYYYMMDD_tHHz.nc
        name = hrrr.name
        try:
            day = name.split("_")[1]
            hh = int(name.split("_t")[1][:2])
            init = datetime(int(day[:4]), int(day[4:6]), int(day[6:8]), hh, tzinfo=timezone.utc)
            errs.extend(compare_sources(hrrr, era5, init))
        except Exception as exc:
            errs.append(f"could not infer init for compare: {exc}")

    print("\n=== RESULT ===")
    if errs:
        for e in errs:
            print(f"FAIL: {e}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

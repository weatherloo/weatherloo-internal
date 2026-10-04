"""
Run LSTM sweeps for a given variable across all stations, methods, and lead times.
Sweeps run in parallel up to --workers processes.

Usage:
    python lstm_training/run_all.py --variable t2m --workers 8
    python lstm_training/run_all.py --variable wind_speed --workers 4 --n_trials 50
    python lstm_training/run_all.py --station eric_d_soulis --workers 8 --n_trials 50
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

STATIONS = ["cyyz", "eric_d_soulis"]
LEADS    = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS  = {
    method: str(REPO_ROOT / "data" / "benchmarks" / method / f"{method}_2025.npz")
    for method in (
        "climatology", "ecmwf_aifs", "gefs_mean", "gfs_analysis", "gfs_interpolated",
        "graphcast", "hrrr_interpolated", "linear_regression", "persistence",
    )
}

# Flat, deterministic 9x12 = 108 combos. Dict order is insertion order (py3.7+), so this is stable.
COMBINATIONS = list(product(METHODS.items(), LEADS))


def run_sweep(station, variable, method, npz, lead, n_trials, out_base):
    out_dir = f"{out_base}/{station}_{variable}/{method}_{lead}h"
    tag = f"[{station}|{method}|{lead}h]"

    if Path(f"{out_dir}/sweep_results.json").exists():
        print(f"{tag} SKIP — already done", flush=True)
        return station, method, lead, "skipped"

    print(f"{tag} START", flush=True)
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "sweep.py"),
        "--npz",       npz,
        "--station",   station,
        "--variable",  variable,
        "--lead_time", str(lead),
        "--n_trials",  str(n_trials),
        "--out_dir",   out_dir,
    ]

    stderr_lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in proc.stdout:
        print(f"{tag} {line}", end="", flush=True)

    proc.wait()
    stderr_out = proc.stderr.read()

    if proc.returncode != 0:
        print(f"{tag} FAILED\n{stderr_out[-800:]}", flush=True)
        return station, method, lead, "FAILED"

    print(f"{tag} DONE", flush=True)
    return station, method, lead, "done"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--station",   nargs="+", default=STATIONS, choices=STATIONS,
                   help="Limit to one or more stations (default: both)")
    p.add_argument("--variable",  default="t2m")
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--n_trials",  type=int, default=100)
    p.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "outputs" / "lstm_training" / "sweeps"),
    )
    args = p.parse_args()

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
    if not 1 <= task_id <= len(COMBINATIONS):
        sys.exit(f"SLURM_ARRAY_TASK_ID={task_id} out of range (must be 1-{len(COMBINATIONS)})")
    (method, npz), lead = COMBINATIONS[task_id - 1]

    jobs = [
        (station, args.variable, method, npz, lead, args.n_trials, args.out_dir)
        for station in args.station
    ]
    total = len(jobs)
    print(f"Queuing {total} sweeps  workers={args.workers}  n_trials={args.n_trials}\n")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_sweep, *j): j for j in jobs}
        for f in as_completed(futures):
            station, method, lead, status = f.result()
            done += 1
            print(f"[{done}/{total}] {station} {method} {lead}h — {status}")


if __name__ == "__main__":
    main()

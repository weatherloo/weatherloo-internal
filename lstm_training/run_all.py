"""
Run LSTM sweeps for a given variable across all stations, methods, and lead times.
Sweeps run in parallel up to --workers processes.

Usage:
    python lstm_training/run_all.py --variable t2m --workers 8
    python lstm_training/run_all.py --variable wind_speed --workers 4 --n_trials 50
"""
import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

STATIONS = ["cyyz", "eric_d_soulis"]
LEADS    = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
METHODS  = {
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


def run_sweep(station, variable, method, npz, lead, n_trials, out_base):
    out_dir = f"{out_base}/{station}_{variable}/{method}_{lead}h"
    tag = f"[{station}|{method}|{lead}h]"

    if Path(f"{out_dir}/sweep_results.json").exists():
        print(f"{tag} SKIP — already done", flush=True)
        return station, method, lead, "skipped"

    print(f"{tag} START", flush=True)
    cmd = [
        sys.executable, "lstm_training/sweep.py",
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
    p.add_argument("--variable",  default="t2m")
    p.add_argument("--workers",   type=int, default=4)
    p.add_argument("--n_trials",  type=int, default=100)
    p.add_argument("--out_dir",   default="lstm_training/output")
    args = p.parse_args()

    jobs = [
        (station, args.variable, method, npz, lead, args.n_trials, args.out_dir)
        for station in STATIONS
        for method, npz in METHODS.items()
        for lead in LEADS
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

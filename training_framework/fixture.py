"""Tiny synthetic dataset in the real benchmark NPZ layout, for CI and agent validation."""
import datetime as dt

import numpy as np

LEADS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]


def make_tiny_npz(path, n_inits=300, seed=0):
    rng = np.random.default_rng(seed)
    t0 = dt.datetime(2025, 1, 1)
    inits = [(t0 + dt.timedelta(hours=6 * i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n_inits)]
    bias = np.zeros((n_inits, 2, 1, len(LEADS)), dtype=np.float32)
    bias[:, 0, 0, 0] = np.sin(np.arange(n_inits) / 4) + 0.1 * rng.standard_normal(n_inits)  # cyyz t2m 6h
    np.savez(path, bias=bias, station_ids=np.array(["cyyz", "eric_d_soulis"]),
             variables=np.array(["t2m"]), lead_times_hours=np.array(LEADS),
             initializations=np.array(inits))

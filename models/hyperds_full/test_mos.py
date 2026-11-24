"""MOS feature leakage guards and a synthetic end-to-end study."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import mos
from points import VARIABLES

UTC = timezone.utc


def synthetic(temp, n_init=40, stations=("eric_d_soulis", "s1", "s2")):
    """Point files + labels where obs = HRRR + 1 degC + noise, so a learner must find +1."""
    rng = np.random.default_rng(0)
    root = Path(temp)
    start = int(datetime(2023, 12, 20, tzinfo=UTC).timestamp())
    inits = start + np.arange(n_init) * 6 * 3600
    values = rng.normal(size=(n_init, 49, len(stations), len(VARIABLES))).astype(np.float32)
    values[..., VARIABLES.index("temperature_2m")] = 10 + np.arange(49)[None, :, None] * .1 + rng.normal(size=(n_init, 49, len(stations)))
    values[..., VARIABLES.index("pressure_surface")] = 98000
    (root / "points" / "2024-01").mkdir(parents=True)
    np.savez(root / "points" / "2024-01" / "points.npz", values=values, init_time=inits)
    (root / "points" / "2024-01" / "manifest.json").write_text(json.dumps(
        {"stations": list(stations), "variables": list(VARIABLES)}))
    times = np.arange(start - 48 * 3600, start + (n_init + 10) * 6 * 3600, 3600)
    labels = np.full((len(times), len(stations), 4), np.nan, np.float32)
    position = {int(t): i for i, t in enumerate(times)}
    for i, t0 in enumerate(inits):
        for h in range(49):
            k = position[int(t0) + h * 3600]
            t2m = values[i, h, :, VARIABLES.index("temperature_2m")]
            labels[k, :, 1] = t2m + 1 + rng.normal(scale=.1, size=len(stations))
            ws = np.hypot(values[i, h, :, VARIABLES.index("wind_u_10m")], values[i, h, :, VARIABLES.index("wind_v_10m")])
            labels[k, :, 0] = ws * .5
    (root / "stations").mkdir()
    np.savez(root / "stations" / "labels.npz", times=times, labels=labels)
    meta = [{"id": s, "lat": 43.4 + j * .1, "lon": -80.5 + j * .1, "elevation": 300.} for j, s in enumerate(stations)]
    (root / "stations" / "stations.json").write_text(json.dumps({"stations": meta}))
    return root, values, inits, list(stations), times, labels, meta


class LeakageTest(unittest.TestCase):
    def test_issue_time_features_use_only_past_observations_and_earlier_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root, values, inits, ids, times, labels, meta = synthetic(temp)
            X, names, y, base, split = mos.build(values, inits, ids, list(VARIABLES), meta, times, labels, ids, ids)
            col = {n: i for i, n in enumerate(names)}
            position = {int(t): i for i, t in enumerate(times)}
            i, s = 10, 1
            t0 = int(inits[i])
            self.assertAlmostEqual(X[i, 0, s, col["obs_t_T0"]], labels[position[t0], s, 1], places=4)
            self.assertAlmostEqual(X[i, 0, s, col["obs_t_T0m3"]], labels[position[t0 - 3 * 3600], s, 1], places=4)
            self.assertTrue(np.all(X[i, :, s, col["obs_t_T0"]] == X[i, 0, s, col["obs_t_T0"]]))  # constant over leads
            t_index = VARIABLES.index("temperature_2m")
            # Previous run (T0-6h) at lead h+6 aligns with the current run at lead h (X starts at lead 1).
            h = 5
            expected = values[i - 1, h + 6, s, t_index] - values[i, h, s, t_index]
            self.assertAlmostEqual(X[i, h - 1, s, col["prev_run_t2m_minus_current"]], expected, places=4)
            self.assertTrue(np.isnan(X[i, 44, s, col["prev_run_t2m_minus_current"]]))  # lead 45 + 6 > 48
            # Lagged ensemble: run T0-12 h at lead h+12 for the same valid time.
            expected12 = values[i - 2, h + 12, s, t_index] - values[i, h, s, t_index]
            self.assertAlmostEqual(X[i, h - 1, s, col["prev12_t2m_minus_current"]], expected12, places=4)
            self.assertTrue(np.isnan(X[i, 37, s, col["prev12_t2m_minus_current"]]))  # lead 38 + 12 > 48
            # Recent error: obs at T0 minus the T0-6 run's 6-h forecast is part of the 24 h mean.
            errs = [labels[position[t0 - 6 * k * 3600], s, 1] - values[i - k - 1, 6, s, t_index] for k in range(4)]
            self.assertAlmostEqual(X[i, 0, s, col["recent_err_t_24h"]], np.mean(errs), places=4)
            # First init has no earlier runs: lagged features are missing, not invented.
            self.assertTrue(np.isnan(X[0, 0, s, col["recent_err_t_24h"]]))
            self.assertTrue(np.isnan(X[0, 0, s, col["prev_run_t2m_minus_current"]]))
            # Residual target is obs - HRRR at the valid time.
            self.assertAlmostEqual(y["t2m"][i, h - 1, s],
                                   labels[position[t0 + h * 3600], s, 1] - values[i, h, s, t_index], places=4)

    def test_split_boundaries_keep_targets_inside_their_split(self):
        t = lambda *a: int(datetime(*a, tzinfo=UTC).timestamp())
        self.assertEqual(mos.split_of(t(2023, 12, 29, 18)), "train")
        self.assertIsNone(mos.split_of(t(2023, 12, 30, 0)))
        self.assertEqual(mos.split_of(t(2024, 1, 1, 0)), "val")
        self.assertIsNone(mos.split_of(t(2024, 12, 31, 0)))
        self.assertEqual(mos.split_of(t(2025, 12, 31, 18)), "test")


class StudyExportTest(unittest.TestCase):
    def test_round2_export_has_validation_predictions_for_blending(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm not installed")
        with tempfile.TemporaryDirectory() as temp:
            root, *_ = synthetic(temp, n_init=80)
            out = Path(temp) / "gbm.json"
            threads, mos.GBM_BASE["num_threads"] = mos.GBM_BASE["num_threads"], 1
            try:
                mos.run_study_v2(root / "points", root / "stations", "eric_d_soulis", k=2, out=out, seeds=1)
            finally:
                mos.GBM_BASE["num_threads"] = threads
            with np.load(out.with_suffix(".npz")) as saved:
                self.assertEqual(saved["val_t2m"].shape, (len(saved["val_init"]), 48))
                self.assertTrue(len(saved["val_init"]) > 0)
                self.assertTrue(all(mos.split_of(int(t)) == "val" for t in saved["val_init"]))
                # Learner finds the +1 degC offset on validation rows.
                error = saved["val_t2m"] - saved["val_t2m_obs"]
                self.assertLess(abs(np.nanmean(error)), 0.3)


if __name__ == "__main__":
    unittest.main()

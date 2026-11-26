"""Precipitation: gauge aggregation, issue-time safety of features, block sums, scoring, model smoke tests."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import mos
import precip
from points import ALL_VARIABLES, VARIABLES, WIDE_VARIABLES

UTC = timezone.utc
T = lambda *a: int(datetime(*a, tzinfo=UTC).timestamp())


def synthetic(n_init=120, stations=("eric_d_soulis", "s1", "s2"), seed=0):
    """Points and labels where a station is wet in an hour exactly when HRRR says >= 0.5 mm there."""
    rng = np.random.default_rng(seed)
    start = T(2023, 12, 20)
    inits = start + np.arange(n_init) * 6 * 3600
    values = rng.normal(size=(n_init, 49, len(stations), len(VARIABLES))).astype(np.float32)
    times = np.arange(start - 72 * 3600, start + (n_init + 12) * 6 * 3600, 3600)
    position = {int(t): i for i, t in enumerate(times)}
    # One true rate per valid hour, shared by every run that covers it.
    truth = np.where(rng.random((len(times), len(stations))) < 0.2, rng.exponential(1.5, (len(times), len(stations))), 0)
    labels = np.full((len(times), len(stations), 4), np.nan, np.float32)
    labels[:, :, 3] = np.where(truth >= 0.5, truth, 0)
    labels[:, :, 1] = 10
    for i, t0 in enumerate(inits):
        rows = [position[int(t0) + h * 3600] for h in range(49)]
        values[i, :, :, VARIABLES.index("precipitation_surface")] = truth[rows] / 3600
    values[..., VARIABLES.index("pressure_surface")] = 98000
    meta = [{"id": s, "lat": 43.4 + j * .1, "lon": -80.5 + j * .1, "elevation": 300.} for j, s in enumerate(stations)]
    return values, inits, list(stations), times, labels, meta


class GaugeTest(unittest.TestCase):
    def test_hour_ending_totals_need_all_four_samples(self):
        base = T(2024, 7, 1, 10)
        stamps = base + np.array([15, 30, 45, 60, 75, 90, 105]) * 60  # 10:15 .. 11:45
        values = np.array([0.2, 0.0, 0.4, 0.2, 0.2, np.nan, 0.2])
        hours, totals = precip.hourly_totals(stamps, values)
        self.assertEqual(list(hours), [base + 3600, base + 7200])
        self.assertAlmostEqual(totals[0], 0.8)  # 10:15, 10:30, 10:45, 11:00 -> hour ending 11:00
        self.assertTrue(np.isnan(totals[1]))      # a missing sample voids the hour

    def test_hobo_reader_converts_local_standard_time_and_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "uw_hobo_15min_2024.csv"
            path.write_text("year,month,day,hour,minute,Temperature,Precipitation - Tipping Bucket,Other\n"
                            "2024,7,1,19,0,20.0,0.4,1\n2024,7,1,19,15,20.0,-9999.99,1\n")
            stamps, values = precip.read_hobo_tipping(path)
        self.assertEqual(int(stamps[0]), T(2024, 7, 2, 0))  # 19:00 EST = 00:00 UTC
        self.assertEqual(values[0], 0.4)
        self.assertTrue(np.isnan(values[1]))

    def test_target_precip_replaces_only_the_target_column(self):
        times = np.arange(5) * 3600
        labels = np.zeros((5, 2, 4), np.float32)
        out = precip.with_target_precip(labels, times, ["a", "b"], "a", np.array([3600, 7200]), np.array([1.0, 2.0]))
        np.testing.assert_array_equal(out[:, 0, 3], [np.nan, 1, 2, np.nan, np.nan])
        np.testing.assert_array_equal(out[:, 1, 3], 0)
        self.assertEqual(labels[1, 0, 3], 0)  # input untouched


class FeatureTest(unittest.TestCase):
    def setUp(self):
        self.values, self.inits, self.ids, self.times, self.labels, self.meta = synthetic()
        self.patch = np.random.default_rng(1).exponential(0.3, (len(self.inits), 49, 9, 9)).astype(np.float32)
        self.patch[5] = np.nan  # a missing patch read
        self.X, self.names, self.y, self.hrrr, self.split, self.block_nbhd = precip.build(
            self.values, self.inits, self.ids, list(VARIABLES), self.meta, self.times, self.labels, self.ids, self.ids,
            self.patch)
        self.col = {n: i for i, n in enumerate(self.names)}

    def test_targets_and_hrrr_align_with_the_valid_hour(self):
        i, h, s = 20, 7, 1
        position = {int(t): k for k, t in enumerate(self.times)}
        self.assertAlmostEqual(self.y[i, h - 1, s], self.labels[position[int(self.inits[i]) + h * 3600], s, 3], places=5)
        self.assertAlmostEqual(self.hrrr[i, h - 1, s],
                               self.values[i, h, s, VARIABLES.index("precipitation_surface")] * 3600, places=4)

    def test_issue_time_observations_end_at_initialization(self):
        i, s = 30, 2
        position = {int(t): k for k, t in enumerate(self.times)}
        t0 = int(self.inits[i])
        past = [self.labels[position[t0 - k * 3600], s, 3] for k in range(6)]
        self.assertAlmostEqual(self.X[i, 10, s, self.col["obs_pr_1h_T0"]], past[0], places=5)
        self.assertAlmostEqual(self.X[i, 10, s, self.col["obs_pr_6h_T0"]], sum(past), places=4)
        # Changing any label after T0 leaves every issue-time feature unchanged.
        later = self.labels.copy()
        later[position[t0 + 3600]:, :, 3] += 5
        X2, *_ = precip.build(self.values, self.inits, self.ids, list(VARIABLES), self.meta, self.times, later, self.ids,
                              self.ids, self.patch)
        for name in ("obs_pr_1h_T0", "obs_pr_3h_T0", "obs_pr_6h_T0", "err_pr_6h_T0"):
            np.testing.assert_array_equal(X2[i, :, :, self.col[name]], self.X[i, :, :, self.col[name]])

    def test_lagged_runs_verify_at_the_same_valid_hour(self):
        i, h, s = 40, 5, 0
        pr = self.values[..., VARIABLES.index("precipitation_surface")] * 3600
        self.assertAlmostEqual(self.X[i, h - 1, s, self.col["pr_prev6"]], pr[i - 1, h + 6, s], places=4)
        self.assertAlmostEqual(self.X[i, h - 1, s, self.col["pr_prev12"]], pr[i - 2, h + 12, s], places=4)
        self.assertTrue(np.isnan(self.X[i, 44, s, self.col["pr_prev6"]]))  # lead 45 + 6 > 48

    def test_neighbourhood_only_for_target_and_missing_is_not_dry(self):
        self.assertTrue(np.isfinite(self.X[10, :, 0, self.col["nb17_frac"]]).all())
        self.assertTrue(np.isnan(self.X[10, :, 1, self.col["nb17_frac"]]).all())
        self.assertTrue(np.isnan(self.X[5, :, 0, self.col["nb17_frac"]]).all())
        self.assertTrue(np.isnan(self.block_nbhd["frac"][5]).all())

    def test_blocks_sum_six_observed_hours(self):
        y = self.y.copy()
        y[3, 7, 0] = np.nan  # hour 8, block 2 (hours 7-12)
        Xb, names_b, yb, hb = precip.blocks(self.X, self.names, y, self.hrrr)
        self.assertEqual(Xb.shape[:3], (len(self.inits), 8, 3))
        self.assertAlmostEqual(yb[3, 0, 0], self.y[3, :6, 0].sum(), places=4)
        self.assertTrue(np.isnan(yb[3, 1, 0]))
        self.assertAlmostEqual(hb[3, 7, 1], self.hrrr[3, 42:48, 1].sum(), places=4)
        self.assertIn("sum_precip_mm_h", names_b)


class ScoringTest(unittest.TestCase):
    def test_brier_reliability_and_skill(self):
        event = np.array([1, 0, 1, 0], float)
        self.assertEqual(precip.brier(event, event), 0)
        self.assertAlmostEqual(precip.brier(np.full(4, .5), event), .25)
        valid = np.full(4, T(2025, 7, 1))
        scores = precip.score_probability({"perfect": event, "climatology": np.full(4, .5)}, event * 1.0, valid)
        self.assertEqual(scores["all"]["perfect"]["bss_vs_climatology"], 1.0)
        self.assertEqual(scores["nov_mar"]["count"], 0)

    def test_climatology_uses_training_frequency_with_fallback(self):
        obs = np.r_[np.ones(40), np.zeros(40)]
        month, hour = np.r_[np.full(40, 7), np.full(40, 1)], np.zeros(80, int)
        out = precip.climatology(obs, month, hour, np.array([7, 1, 7]), np.array([0, 0, 5]), np.mean)
        np.testing.assert_allclose(out, [1, 0, 1])  # (7, 5) unseen -> July frequency


class ModelSmokeTest(unittest.TestCase):
    def test_gbm_learns_hrrr_wet_signal(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm not installed")
        values, inits, ids, times, labels, meta = synthetic(n_init=160)
        bounds = {"TRAIN_END": T(2023, 12, 29), "VAL_END": T(2024, 1, 6), "TEST_END": T(2024, 1, 12)}
        with mock.patch.multiple(mos, **bounds):
            X, names, y, hrrr, split, _ = precip.build(values, inits, ids, list(VARIABLES), meta, times, labels, ids, ids)
            with mock.patch.dict(precip.GBM, {"num_threads": 1, "min_data_in_leaf": 20}):
                pred, report = precip.fit_gbm(X, y, split, "pop", seeds=1)
        val = split == "val"
        event = (y[val, :, 0] >= precip.WET).astype(float)
        self.assertLess(precip.brier(pred[val], event), 0.5 * event.mean() * (1 - event.mean()))
        self.assertIn(report["chosen"], ("target", "pooled"))

    def test_cnn_runs_end_to_end(self):
        values, inits, ids, times, labels, meta = synthetic(n_init=100)
        bounds = {"TRAIN_END": T(2023, 12, 29), "VAL_END": T(2024, 1, 4), "TEST_END": T(2024, 1, 12)}
        with tempfile.TemporaryDirectory() as temp, mock.patch.multiple(mos, **bounds):
            month = Path(temp) / "2024-01"
            month.mkdir()
            rng = np.random.default_rng(0)
            np.save(month / "patches.npy", rng.normal(size=(len(inits), 49, len(ALL_VARIABLES), 7, 7)).astype(np.float16))
            np.save(month / "wide.npy", np.zeros((len(inits), 49, len(WIDE_VARIABLES), 2, 2), np.float16))
            np.save(month / "init_time.npy", inits)
            (month / "manifest.json").write_text(json.dumps({"samples": len(inits), "variables": list(ALL_VARIABLES),
                                                             "wide_variables": list(WIDE_VARIABLES)}))
            X, names, y, hrrr, split, _ = precip.build(values, inits, ids, list(VARIABLES), meta, times, labels, ids, ids)
            data = {"X": X, "names": names, "y": y, "hrrr": hrrr, "split": split, "inits": inits}
            out, report = precip.train_cnn(data, temp, seeds=1, max_epochs=2, batch_size=64)
        test = split == "test"
        self.assertEqual(out["test"]["pop1"].shape, (len(inits), 48))
        self.assertTrue(np.isfinite(out["test"]["pop6"][test]).all())
        self.assertTrue(((out["test"]["pop1"][test] >= 0) & (out["test"]["pop1"][test] <= 1)).all())
        self.assertTrue((out["test"]["amt6"][test] >= 0).all())
        self.assertTrue(np.isnan(out["test"]["pop6"][split == "train"]).all())


if __name__ == "__main__":
    unittest.main()

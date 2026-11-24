"""Region config parsing, validation-weighted blending and the region site export."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import blend
import regions


def gbm_export(rng, n_test=20, n_val=30, cnn_better=False):
    """GBM/CNN exports where the truth is known, so the chosen weight is predictable."""
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    val_start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    init, val_init = start + np.arange(n_test) * 21600, val_start + np.arange(n_val) * 21600
    g, c = {"init": init, "val_init": val_init, "lead": np.arange(1, 49)}, {"init": init, "val_init": val_init}
    cnn, obs, val_cnn, val_obs = (np.zeros((n, 48, 2)) for n in (n_test, n_test, n_val, n_val))
    for j, v in enumerate(blend.VARIABLES):
        for prefix, n, cnn_out, obs_out in (("", n_test, cnn, obs), ("val_", n_val, val_cnn, val_obs)):
            truth = rng.normal(size=(n, 48))
            a, b = rng.normal(scale=1.0, size=(n, 48)), rng.normal(scale=1.0, size=(n, 48))
            # Independent equal-variance errors: the best weight is 0.5.
            g[f"{prefix}{v}"], g[f"{prefix}{v}_obs"] = truth + a, truth
            cnn_out[..., j], obs_out[..., j] = truth + (b * 0.2 if cnn_better else b), truth
        g[f"{v}_raw"] = g[v] + 3
    c.update({"cnn": cnn, "obs": obs, "val_cnn": val_cnn, "val_obs": val_obs})
    return g, c


class RegionConfigTest(unittest.TestCase):
    def test_waterloo_config_loads_with_expected_paths(self):
        region = regions.load("waterloo")
        self.assertEqual(region.anchor, "eric_d_soulis")
        self.assertEqual([t.id for t in region.targets], ["eric_d_soulis", "eccc_51459"])
        # Paths match the caches already extracted for Soulis.
        self.assertEqual(region.point_root, "/data/processed/points/hrrr_eric_d_soulis_v1")
        self.assertEqual(region.patch_root("eric_d_soulis"), "/data/processed/patches/hrrr_eric_d_soulis_17_v1")
        self.assertEqual(region.target("eccc_51459").site_id, "cyyz")

    def test_invalid_configs_are_rejected(self):
        base = {"name": "x", "anchor": "a", "targets": [{"id": "a", "models": ["gbm"]}]}
        regions.parse(base)
        with self.assertRaises(ValueError):
            regions.parse({**base, "targets": [{"id": "a", "models": ["cnn"]}]})  # gbm is required
        with self.assertRaises(ValueError):
            regions.parse({**base, "targets": [{"id": "a"}, {"id": "a"}]})
        with self.assertRaises(ValueError):
            regions.parse({**base, "anchor": "b"})

    def test_check_reports_missing_and_out_of_chunk_targets(self):
        region = regions.load("waterloo")
        problems = regions.check(region, [{"id": "eric_d_soulis"}], chunk_station_ids=["eric_d_soulis"])
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("eccc_51459" in p for p in problems))


class BlendTest(unittest.TestCase):
    def test_weight_is_chosen_on_validation_and_beats_components(self):
        g, c = gbm_export(np.random.default_rng(0), n_test=200, n_val=200)
        with tempfile.TemporaryDirectory() as temp:
            np.savez(Path(temp) / "cnn.npz", **c)
            cnn = blend.load_cnn(Path(temp) / "cnn.npz")
        report, final = blend.fit_blend(g, cnn)
        for v in blend.VARIABLES:
            self.assertAlmostEqual(report["weights_on_gbm"][v], 0.5, delta=0.1)
            t = report["variables"][v]["test_matched"]
            self.assertLess(t["blend"], min(t["gbm"], t["cnn"]))
        self.assertEqual(final["t2m"].shape, (200, 48))

    def test_better_cnn_gets_more_weight_and_missing_cnn_falls_back(self):
        g, c = gbm_export(np.random.default_rng(1), n_test=50, n_val=200, cnn_better=True)
        c["cnn"][3] = np.nan  # a failed patch read in 2025
        with tempfile.TemporaryDirectory() as temp:
            np.savez(Path(temp) / "cnn.npz", **c)
            cnn = blend.load_cnn(Path(temp) / "cnn.npz")
        report, final = blend.fit_blend(g, cnn)
        self.assertLessEqual(report["weights_on_gbm"]["t2m"], 0.1)
        np.testing.assert_allclose(final["t2m"][3], g["t2m"][3])
        self.assertEqual(report["variables"]["t2m"]["cnn_fallback_inits"], 1)

    def test_gbm_only_target_and_missing_validation_export(self):
        g, _ = gbm_export(np.random.default_rng(2))
        report, final = blend.fit_blend(g)
        self.assertEqual(report["weights_on_gbm"], {"t2m": 1.0, "wind": 1.0})
        np.testing.assert_array_equal(final["wind"], g["wind"])
        with tempfile.TemporaryDirectory() as temp:
            np.savez(Path(temp) / "old.npz", **{k: v for k, v in g.items() if not k.startswith("val_")})
            with self.assertRaises(ValueError):
                blend.load_gbm(Path(temp) / "old.npz")


class SiteExportTest(unittest.TestCase):
    def test_export_scores_against_site_truth_with_correct_units(self):
        region = regions.load("waterloo")
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        init = np.array([int(start.timestamp()), int((start + timedelta(hours=6)).timestamp())])
        finals = {}
        for t, (temp_c, wind_ms) in zip(region.targets, ((5.0, 2.0), (1.0, 4.0))):
            finals[t.id] = {"init": init, "t2m": np.full((2, 48), temp_c), "wind": np.full((2, 48), wind_ms)}
        with tempfile.TemporaryDirectory() as temp:
            truth = Path(temp) / "observations"
            for site, (temp_c, wind_kmh) in (("eric_d_soulis", (4.0, 9.0)), ("cyyz", (1.5, 14.4))):
                rows = [{"valid_time": (start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "t2m": temp_c, "wind_speed": wind_kmh} for h in range(0, 80)]
                (truth / site).mkdir(parents=True)
                (truth / site / "observations_6h_2025.json").write_text(json.dumps({"observations": rows}))
            path = blend.site_export(region, finals, truth, Path(temp) / "out", "test_method")
            with np.load(path) as saved:
                stations = list(saved["station_ids"])
                leads = list(saved["lead_times_hours"])
                bias = saved["bias"]
                self.assertEqual(stations, ["eric_d_soulis", "cyyz"])
                s, w = stations.index("eric_d_soulis"), list(saved["variables"]).index("wind_speed")
                self.assertAlmostEqual(bias[0, s, list(saved["variables"]).index("t2m"), leads.index(6)], 1.0)
                self.assertAlmostEqual(bias[1, s, w, leads.index(48)], 2.0 * 3.6 - 9.0)
                self.assertTrue(np.isnan(bias[0, s, w, leads.index(54)]))  # beyond 48 h
                self.assertTrue(np.isnan(bias[2]).all())  # initialization without a forecast
            meta = json.loads((Path(temp) / "out" / "metadata.json").read_text())
            self.assertIn("HRRRv4", meta["initialization_regime"])
            self.assertEqual(meta["secondary_station"], "cyyz")


if __name__ == "__main__":
    unittest.main()

"""Region config parsing, the CNN-only final step and the region site export."""

import repo_paths  # noqa: F401  (shared pipelines modules on sys.path)
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import final
import regions
import station_labels as stations


def cnn_export(rng, n_test=20):
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    obs = rng.normal(size=(n_test, 48, 2))
    return {"init": start + np.arange(n_test) * 21600, "obs": obs, "cnn": obs + rng.normal(scale=.5, size=obs.shape),
            "hrrr": obs + rng.normal(scale=2., size=obs.shape)}


class RegionConfigTest(unittest.TestCase):
    def test_waterloo_config_loads_with_expected_paths(self):
        region = regions.load("waterloo")
        self.assertEqual(region.anchor, "eric_d_soulis")
        self.assertEqual([t.id for t in region.targets], ["eric_d_soulis", "eccc_51459"])
        # Paths match the caches already extracted for Soulis.
        self.assertEqual(region.point_root, "/data/processed/points/hrrr_eric_d_soulis_v1")
        self.assertEqual(region.patch_root("eric_d_soulis"),
                         "/data/processed/patches/hrrr_eric_d_soulis_context_17_b8_v1")
        self.assertEqual(region.target("eric_d_soulis").cnn_options, {"trajectory": True})
        self.assertEqual(region.target("eccc_51459").cnn_options, {"trajectory": True})
        self.assertTrue(region.target("eric_d_soulis").precip)
        self.assertEqual(region.target("eccc_51459").site_id, "cyyz")

    def test_invalid_configs_are_rejected(self):
        base = {"name": "x", "anchor": "a", "targets": [{"id": "a"}]}
        regions.parse(base)
        regions.parse({**base, "targets": [{"id": "a", "models": ["cnn"]}]})  # explicit CNN is fine
        with self.assertRaises(ValueError):  # one model type only
            regions.parse({**base, "targets": [{"id": "a", "models": ["gbm"]}]})
        with self.assertRaises(ValueError):
            regions.parse({**base, "targets": [{"id": "a"}, {"id": "a"}]})
        with self.assertRaises(ValueError):
            regions.parse({**base, "anchor": "b"})
        with self.assertRaises(ValueError):
            regions.parse({**base, "targets": [{"id": "a", "cnn_options": {"lr": 1}}]})

    def test_check_reports_missing_and_out_of_chunk_targets(self):
        region = regions.load("waterloo")
        problems = regions.check(region, [{"id": "eric_d_soulis"}], chunk_station_ids=["eric_d_soulis"])
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("eccc_51459" in p for p in problems))


class FinalTest(unittest.TestCase):
    def test_final_forecast_is_the_cnn_and_is_scored_against_raw_hrrr(self):
        data = cnn_export(np.random.default_rng(0))
        with tempfile.TemporaryDirectory() as temp:
            np.savez(Path(temp) / "cnn.npz", **data)
            cnn = final.load_cnn(Path(temp) / "cnn.npz")
            np.savez(Path(temp) / "old.npz", **{k: v for k, v in data.items() if k != "hrrr"})
            with self.assertRaises(ValueError):
                final.load_cnn(Path(temp) / "old.npz")
        report, forecast = final.score(cnn)
        np.testing.assert_array_equal(forecast["t2m"], data["cnn"][..., 0])
        np.testing.assert_array_equal(forecast["wind"], data["cnn"][..., 1])
        for v, unit in (("t2m", 1.0), ("wind", 3.6)):
            entry = report["variables"][v]
            self.assertLess(entry["rmse"], entry["raw_rmse"])
            self.assertEqual(len(entry["rmse_by_lead"]), 48)
        self.assertAlmostEqual(report["variables"]["wind"]["rmse"] / 3.6,
                               float(np.sqrt(np.mean((data["cnn"][..., 1] - data["obs"][..., 1]) ** 2))), places=6)


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
            path = final.site_export(region, finals, truth, Path(temp) / "out", "test_method")
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


class StationBuildTest(unittest.TestCase):
    def test_san_francisco_config_needs_no_existing_table(self):
        region = regions.load("san_francisco")
        self.assertEqual(region.anchor, "asos_SFO")
        self.assertEqual(region.station_build["asos_networks"], ["CA", "NV"])
        self.assertIsNone(region.target("asos_SFO").site_id)
        with self.assertRaises(ValueError):
            regions.parse({"name": "x", "anchor": "a", "targets": [{"id": "a"}], "stations": {"bbox": [1, 2, 0, 3]}})

    def test_region_table_keeps_targets_filters_coverage_and_deduplicates(self):
        inventory = [{"id": "asos_SFO", "source": "asos", "asos": "SFO", "lat": 37.62, "lon": -122.37},
                     {"id": "asos_SQL", "source": "asos", "asos": "SQL", "lat": 37.51, "lon": -122.25},
                     {"id": "asos_TWIN", "source": "asos", "asos": "TWIN", "lat": 37.511, "lon": -122.251},
                     {"id": "asos_GAP", "source": "asos", "asos": "GAP", "lat": 38.0, "lon": -122.0}]
        coverage = {"SFO": 0.3, "SQL": 0.9, "TWIN": 0.8, "GAP": 0.1}

        def fake_fetch(station, n_hours):
            labels = np.full((n_hours, 4), np.nan, np.float32)
            labels[:int(coverage[station["asos"]] * n_hours)] = (3.0, 15.0, 1000.0, 0.0)
            return labels, 1

        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(stations, "inventory", return_value=[dict(s) for s in inventory]), \
                mock.patch.object(stations, "fetch_asos", side_effect=fake_fetch):
            manifest = stations.build_region(temp, (-124, 35, -118, 41), ("CA",), keep=["asos_SFO"])
            with np.load(Path(temp) / "labels.npz") as saved:
                labels = saved["labels"]
        ids = [s["id"] for s in manifest["stations"]]
        self.assertEqual(ids, ["asos_SFO", "asos_SQL"])  # target kept despite low coverage; GAP too sparse
        self.assertEqual(manifest["duplicates_removed"], {"asos_TWIN": "asos_SQL"})
        self.assertEqual(labels.shape[1], 2)
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(stations, "inventory", return_value=[dict(s) for s in inventory[1:]]):
            with self.assertRaises(ValueError):
                stations.build_region(temp, (-124, 35, -118, 41), ("CA",), keep=["asos_SFO"])


if __name__ == "__main__":
    unittest.main()

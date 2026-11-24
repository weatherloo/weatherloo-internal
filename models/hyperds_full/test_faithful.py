"""Paper-faithful HRRR HyperDS: pooling, decoder base, subgrid sampling, station split, train/eval smoke."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

import faithful_baselines as fb
import faithful_train as ft
import nwp_data
from nwp_model import NWPHyperDS
from stations import deduplicate, eccc_degrees, split_stations, station_pressure
from test_nwp import synthetic_stores


class PaperContractTest(unittest.TestCase):
    def test_input_is_four_times_average_pooled(self):
        forecast = torch.arange(2 * 3 * 5 * 32 * 64, dtype=torch.float32).reshape(2, 3, 5, 32, 64)
        pooled = ft.coarse(forecast)
        self.assertEqual(tuple(pooled.shape), (2, 3, 5, 8, 16))
        torch.testing.assert_close(pooled[1, 2], F.avg_pool2d(forecast[1, 2], 4))

    def test_decoder_auxiliary_value_is_the_interpolated_input_field(self):
        torch.manual_seed(0)
        model = NWPHyperDS("nwp", satellite_shape=(32, 64), n_stations=0).eval()
        for decoder in model.decoders:  # isolate the base: zero generated residual
            torch.nn.init.zeros_(decoder.out_fc.weight)
            torch.nn.init.zeros_(decoder.out_fc.bias)
        leads = list(range(49))
        fields = torch.randn(1, 49, 5, 8, 16)
        sat = model.encode_satellite(torch.randn(1, 2, 4, 32, 64))
        with torch.no_grad():
            contexts = model.encode_anchors(sat, fields, leads, torch.randn(1, 49, 4), torch.zeros(1, 0))
            query = torch.rand(1, 7, 2) * 2 - 1
            prediction = ft.predict_hour(model, contexts, fields, leads, 13, query, torch.randn(1, 49, 4), torch.zeros(1, 0))
        expected = F.grid_sample(fields[:, 13], query[:, None], align_corners=False, padding_mode="border").squeeze(2).transpose(1, 2)
        torch.testing.assert_close(prediction, expected)

    def test_full_resolution_input_option(self):
        try:
            ft.set_input_pool(1)
            forecast = torch.randn(2, 3, 5, 32, 64)
            torch.testing.assert_close(ft.coarse(forecast), forecast)
            for kind in ("unet", "edsr"):
                model = fb.SRBaseline(kind)
                out = model(forecast[:, 0], torch.randn(2, 2, 4, 128, 256), 6, torch.randn(2, 4))
                self.assertEqual(tuple(out.shape), (2, 5, 32, 64))
        finally:
            ft.set_input_pool(4)
        with self.assertRaises(ValueError):
            ft.set_input_pool(3)

    def test_every_cell_gets_ten_samples_and_stations_are_inserted(self):
        stations = torch.tensor([[[-.99, -.99], [.4, .8], [.41, .81]]])
        query = ft.grid_queries(1, "cpu", stations)
        self.assertEqual(tuple(query.shape), (1, 32 * 64 * 10, 2))
        self.assertTrue(((query >= -1) & (query <= 1)).all())
        for station in stations[0]:
            self.assertTrue((query[0] == station).all(dim=-1).any())
        cells = ((query[0] + 1) / 2 * torch.tensor([64, 32])).long()
        flat = cells[:, 1].clamp(0, 31) * 64 + cells[:, 0].clamp(0, 63)
        self.assertTrue((torch.bincount(flat, minlength=2048) == 10).all())

    def test_station_split_uses_paper_proportions_and_forces_waterloo_toronto_to_test(self):
        stations = [{"id": f"s{i}"} for i in range(340)] + [{"id": "eric_d_soulis"}, {"id": "eccc_51459"}]
        split = split_stations(stations)
        ids = {k: {stations[i]["id"] for i in v} for k, v in split.items()}
        self.assertTrue({"eric_d_soulis", "eccc_51459"} <= ids["test"])
        self.assertFalse(ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
        self.assertEqual(sum(len(v) for v in split.values()), len(stations))
        self.assertAlmostEqual(len(split["test"]) / len(stations), 400 / 1866, delta=.01)
        self.assertAlmostEqual(len(split["val"]) / len(stations), 200 / 1866, delta=.01)

    def test_co_located_stations_are_merged_before_splitting(self):
        stations = [{"id": "eric_d_soulis", "lat": 43.4668, "lon": -80.5164, "coverage_t2m_wind": .4},
                    {"id": "eccc_a", "lat": 44.744, "lon": -81.109, "coverage_t2m_wind": .6},
                    {"id": "eccc_b", "lat": 44.7445, "lon": -81.1092, "coverage_t2m_wind": .9},
                    {"id": "eccc_c", "lat": 43.47, "lon": -80.52, "coverage_t2m_wind": .99},
                    {"id": "asos_d", "lat": 46.0, "lon": -80.0, "coverage_t2m_wind": .5}]
        keep, duplicates = deduplicate(stations)
        self.assertEqual([stations[i]["id"] for i in keep], ["eric_d_soulis", "eccc_b", "asos_d"])
        self.assertEqual(duplicates, {"eccc_a": "eccc_b", "eccc_c": "eric_d_soulis"})

    def test_station_metadata_conversions(self):
        self.assertAlmostEqual(eccc_degrees(455700000), 45.95, places=6)
        self.assertAlmostEqual(eccc_degrees(-771900000), -77.316667, places=5)
        self.assertAlmostEqual(station_pressure(29.92, 0), 1013.2, places=1)
        self.assertLess(station_pressure(29.92, 300), 980)


def faithful_stores(temp):
    """Synthetic ERA5/NWP shards plus an 11-station hourly label store and split."""
    leads = list(range(49))
    era5, nwp, _ = synthetic_stores(temp, leads)
    station_root = Path(temp) / "stations"
    station_root.mkdir()
    stations = [{"id": s["id"], "lat": s["lat"], "lon": s["lon"]} for s in nwp_data.STATIONS]
    stations[4]["id"] = "eccc_51459"
    stations += [{"id": f"x{i}", "lat": 41 + i * .5, "lon": -86 + i} for i in range(6)]
    start = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    times = start + np.arange(24 * 10) * 3600
    rng = np.random.default_rng(0)
    labels = rng.normal(size=(len(times), len(stations), 4)).astype(np.float32)
    labels[..., 0] = np.abs(labels[..., 0])
    labels[::7, 3] = np.nan
    np.savez(station_root / "labels.npz", times=times, labels=labels)
    (station_root / "stations.json").write_text(json.dumps({"stations": stations}))
    split = {"train": [s["id"] for s in stations[1:4] + stations[5:9]],
             "val": [stations[9]["id"]], "test": [stations[0]["id"], "eccc_51459", stations[10]["id"]]}
    (station_root / "split.json").write_text(json.dumps(split))
    era5_manifest = json.loads((era5 / "manifest.json").read_text())
    nwp_manifest = json.loads((nwp / "manifest.json").read_text())
    dataset = ft.FaithfulDataset(era5, nwp, station_root, "val", era5_manifest, nwp_manifest)
    ids = [s["id"] for s in stations]
    groups = {k: [ids.index(s) for s in v] for k, v in split.items()} | {"eric_d_soulis": [0]}
    return dataset, ids, groups, leads


class FaithfulEndToEndTest(unittest.TestCase):
    def test_synthetic_train_and_evaluate(self):
        torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as temp:
            dataset, ids, groups, leads = faithful_stores(temp)
            target, sat, station, forecast, clock = dataset[0]
            self.assertEqual(tuple(station.shape), (49, len(ids), 4))
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, drop_last=True)
            mean, std = torch.tensor([0, 0, 0, 980., 0]), torch.ones(5)
            for mode in ("nwp", "nwp_autoregressive"):
                model = NWPHyperDS(mode, satellite_shape=(32, 64), n_stations=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                before = [p.detach().clone() for p in model.parameters()]
                metrics = ft.train_epoch(model, train_loader, optimizer, "cpu", mean, std, leads, groups["train"],
                                         dataset.coordinates, hours_per_step=2, accumulation=2)
                self.assertEqual(metrics["steps"], 1 * 2)
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
                output = Path(temp) / f"{mode}.npz"
                report = ft.evaluate(model, loader, "cpu", mean, std, leads, dataset.coordinates, groups, "val",
                                     prediction_output=output, station_ids=ids)
                self.assertTrue(np.isfinite(report["validation_station_loss"]))
                for group in ("train", "val", "test", "eric_d_soulis"):
                    self.assertTrue(set(report["groups"][group]) >= {"model", "interp_1deg", "interp_025deg"})
                with np.load(output) as saved:
                    self.assertEqual(saved["forecast"].shape, (3, 49, len(ids), 4))
                    self.assertIn("cyyz", list(saved["station_ids"]))
                    self.assertIn("eric_d_soulis", list(saved["station_ids"]))

    def test_super_resolution_baselines(self):
        torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as temp:
            dataset, ids, groups, leads = faithful_stores(temp)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, drop_last=True)
            mean, std = torch.tensor([0, 0, 0, 980., 0]), torch.ones(5)
            for kind in ("unet", "edsr"):
                model = fb.SRBaseline(kind)
                grid = model(torch.randn(2, 5, 8, 16), torch.randn(2, 2, 4, 128, 256), 6, torch.randn(2, 4))
                self.assertEqual(tuple(grid.shape), (2, 5, 32, 64))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                metrics = fb.train_epoch(model, train_loader, optimizer, "cpu", mean, std, groups["train"],
                                         dataset.coordinates, hours_per_step=2)
                self.assertEqual(metrics["steps"], 2)
                output = Path(temp) / f"{kind}.npz"
                report = fb.evaluate(model, loader, "cpu", mean, std, dataset.coordinates, groups, "val", output, ids)
                self.assertTrue(np.isfinite(report["validation_station_loss"]))
                with np.load(output) as saved:
                    self.assertEqual(saved["forecast"].shape, (3, 49, len(ids), 4))


if __name__ == "__main__":
    unittest.main()

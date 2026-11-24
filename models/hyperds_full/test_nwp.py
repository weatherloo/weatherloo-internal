"""NWP-input HyperDS: builder geometry, chronology, warm start, feedback, and a synthetic train/eval step."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from model import HyperDS
import nwp_data
from nwp_model import PRECIPITATION, NWPHyperDS, anchor_weights, at_hour
from nwp_train import (NWPDataset, clock_features, evaluate, forecast_error_scales, observation_features,
                       scaled_station_loss, train_epoch, zero_residual)

UTC = timezone.utc


class GeometryTest(unittest.TestCase):
    def test_bilinear_station_weights_are_exact_for_linear_fields_on_either_axis_order(self):
        xs, ys = np.linspace(0, 10, 11), np.linspace(20, 0, 9)  # descending y like north-up grids
        field = 3 * xs[None, :] - 2 * ys[:, None] + 1
        weights = nwp_data.station_weights(xs, ys, [(2.3, 7.1), (9.9, 0.4)])
        values = nwp_data.interpolate_stations(field[None], weights)[0]
        np.testing.assert_allclose(values, [3 * 2.3 - 2 * 7.1 + 1, 3 * 9.9 - 2 * 0.4 + 1], rtol=1e-12)
        with self.assertRaises(ValueError):
            nwp_data.station_weights(xs, ys, [(11, 5)])

    def test_splits_keep_targets_before_the_next_split(self):
        self.assertIsNone(nwp_data.split_for("hrrr", datetime(2020, 12, 2, 18, tzinfo=UTC)))
        self.assertEqual(nwp_data.split_for("hrrr", datetime(2020, 12, 3, tzinfo=UTC)), "train")
        self.assertIsNone(nwp_data.split_for("hrrr", datetime(2023, 12, 30, 6, tzinfo=UTC)))
        self.assertEqual(nwp_data.split_for("hrrr", datetime(2023, 12, 29, 18, tzinfo=UTC)), "train")
        self.assertEqual(nwp_data.split_for("aifs", datetime(2024, 10, 29, 18, tzinfo=UTC)), "train")
        self.assertIsNone(nwp_data.split_for("aifs", datetime(2024, 10, 30, 0, tzinfo=UTC)))
        self.assertEqual(nwp_data.split_for("aifs", datetime(2025, 12, 31, 18, tzinfo=UTC)), "test")
        self.assertIsNone(nwp_data.split_for("aifs", datetime(2024, 3, 31, 18, tzinfo=UTC)))
        self.assertEqual(nwp_data.months("aifs")[0], "2024-04")
        self.assertEqual(nwp_data.months("hrrr")[0], "2020-12")
        self.assertEqual(len(nwp_data.months("hrrr")), 61)

    def test_scaled_station_loss_weights_variables_by_forecast_error(self):
        prediction = torch.tensor([[[3., 4., 11., 900., 0.]]])  # wind 5 m/s, t2m 11
        target = torch.tensor([[[7., 10., float("nan"), float("nan")]]])  # wind miss 2, t2m miss 1
        loss = scaled_station_loss(prediction, target, torch.zeros(5), torch.ones(5), [0], torch.tensor([2., 1., 1., 1.]))
        self.assertAlmostEqual(loss.item(), 1.0, places=6)  # (2/2)^2 and (1/1)^2, averaged

    def test_clock_and_observation_features(self):
        clock = clock_features([0, 6 * 3600, 86400])
        np.testing.assert_allclose(clock[0, :2], [0, 1], atol=1e-7)
        np.testing.assert_allclose(clock[1, :2], [1, 0], atol=1e-7)
        np.testing.assert_allclose(clock[2, :2], clock[0, :2], atol=1e-6)
        observed = np.array([[5, 10, 900, 0], [np.nan, 12, 900, 0]], dtype=np.float32)
        forecast = np.array([[3, 4, 8, 900, 0], [0, 1, 10, 900, 0]], dtype=np.float32)
        std = np.array([1, 1, 2, 1, 1], dtype=np.float32)
        features = observation_features(observed, forecast, std).reshape(2, 4)
        np.testing.assert_allclose(features[0], [0, 1, 1, 1], atol=1e-6)
        np.testing.assert_allclose(features[1], [0, 1, 0, 1], atol=1e-6)


class ModelTest(unittest.TestCase):
    def test_zero_initialized_conditioning_reproduces_temporal_warm_start(self):
        torch.manual_seed(1)
        temporal = HyperDS("direct", satellite_shape=(32, 64)).eval()
        nwp = NWPHyperDS("nwp", satellite_shape=(32, 64)).eval()
        missing, unexpected = nwp.load_state_dict(temporal.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(k.startswith(("clock_projection", "observation_projection")) for k in missing))
        field, sat, query = torch.randn(2, 5, 32, 64), torch.randn(2, 2, 4, 32, 64), torch.rand(2, 3, 2) * 2 - 1
        with torch.no_grad():
            context = temporal.encode_field(field, temporal.encode_satellite(sat))
            expected = temporal.decode(context, field, query, 7)
            extra = nwp.conditioning(torch.randn(2, 4), torch.randn(2, 20))
            actual = nwp.decode(context, field, query, 7, extra=extra)
        torch.testing.assert_close(actual, expected)

    def test_time_interpolation_is_exact_at_leads_linear_between_and_keeps_interval_precipitation(self):
        leads = list(range(0, 49, 6))
        values = torch.arange(9, dtype=torch.float32)[None, :, None].repeat(1, 1, 5)  # [b, lead, var]
        values[0, :, PRECIPITATION] = torch.arange(9) * 10.
        torch.testing.assert_close(at_hour(values, leads, 12, 1), values[:, 2])
        middle = at_hour(values, leads, 15, 1)
        torch.testing.assert_close(middle[0, :4], torch.full((4,), 2.5))
        self.assertEqual(middle[0, PRECIPITATION].item(), 30.)  # rate over (12, 18]
        hourly = torch.randn(1, 49, 3, 5)
        torch.testing.assert_close(at_hour(hourly, list(range(49)), 17, 2), hourly[:, 17])
        self.assertEqual(anchor_weights(48), (7, 1.0))
        self.assertEqual(anchor_weights(15), (2, 0.5))

    def test_zero_residual_starts_exactly_at_the_forecast_at_every_hour(self):
        torch.manual_seed(2)
        leads = list(range(0, 49, 6))
        model = NWPHyperDS("nwp_autoregressive", satellite_shape=(32, 64)).eval()
        zero_residual(model)
        sat = model.encode_satellite(torch.randn(1, 2, 4, 32, 64))
        forecast, station = torch.randn(1, 9, 5, 32, 64), torch.randn(1, 9, 5, 5)
        clock, obs = torch.randn(1, 49, 4), torch.randn(1, 20)
        with torch.no_grad():
            contexts = model.encode_anchors(sat, forecast, leads, clock, obs)
            self.assertEqual(len(contexts), 9)
            for hour in (6, 7, 47):
                queries = torch.cat((torch.rand(1, 5, 2) * 2 - 1, torch.rand(1, 20, 2) * 2 - 1), dim=1)
                prediction = model.decode_hour(contexts, forecast, station, leads, hour, queries,
                                               model.conditioning(clock[:, hour], obs))
                torch.testing.assert_close(prediction[:, :5], at_hour(station, leads, hour, 2))

    def test_autoregressive_anchor_feedback_reaches_later_contexts_only_in_ar_mode(self):
        leads = list(range(0, 49, 6))
        for mode, expected in (("nwp_autoregressive", True), ("nwp", False)):
            torch.manual_seed(3)
            model = NWPHyperDS(mode, satellite_shape=(32, 64))
            sat = model.encode_satellite(torch.randn(2, 2, 4, 32, 64))
            forecast = torch.randn(2, 9, 5, 32, 64, requires_grad=True)
            contexts = model.encode_anchors(sat, forecast, leads, torch.randn(2, 49, 4), torch.randn(2, 20))
            contexts[1].square().sum().backward()
            reached = forecast.grad[:, 0].abs().sum().item() > 0
            self.assertEqual(reached, expected, mode)


def synthetic_stores(root, leads, n=3):
    """Tiny ERA5 shard + NWP shard with the production layouts."""
    import xarray as xr

    root = Path(root)
    era5, nwp = root / "era5", root / "nwp"
    key, split = "2024-01", "val"
    inits = [datetime(2024, 1, 2, tzinfo=UTC) + timedelta(hours=6 * i) for i in range(n)]
    rng = np.random.default_rng(0)
    grid = rng.normal(size=(n, 49, 5, 32, 64)).astype(np.float32)
    grid[:, :, 3] += 980
    station = rng.normal(size=(n, 49, 5, 4)).astype(np.float32)
    station[:, :, :, 0] = np.abs(station[:, :, :, 0])
    station[0, 5, 1, 1] = np.nan
    stations = [{"id": s["id"], "lat": s["lat"], "lon": s["lon"]} for s in nwp_data.STATIONS]
    query = np.array([[2 * (s["lon"] + 88.125) / 16 - 1, 2 * (s["lat"] - 39.875) / 8 - 1] for s in stations], np.float32)
    xr.Dataset({"grid": (("sample", "lead", "variable", "latitude", "longitude"), grid),
                "satellite": (("sample", "frame", "band", "y", "x"), rng.normal(size=(n, 2, 4, 32, 64)).astype(np.float32)),
                "station": (("sample", "lead", "station_id", "target"), station),
                "query_xy": (("station_id", "xy"), query),
                "init_time": ("sample", np.array([int(t.timestamp()) for t in inits], np.int64))}).to_zarr(
        era5 / "shards" / key, group=split, mode="w", consolidated=False)
    (era5 / "shards" / key / "manifest.json").write_text(json.dumps(
        {"split": split, "records": [{"initialization": t.isoformat()} for t in inits]}))
    times = np.arange(int(inits[0].timestamp()), int(inits[-1].timestamp()) + 49 * 3600, 3600, dtype=np.int64)
    overlay = np.abs(rng.normal(size=(len(times), 2))).astype(np.float32)
    np.savez(era5 / "overlay.npz", times=times, values=overlay)
    (era5 / "manifest.json").write_text(json.dumps({
        "stations": stations, "station_overlay": {"path": "overlay.npz"},
        "training_station_indices": [0, 1, 2, 3, 4], "selection_station_indices": [0],
        "normalization": {"field_mean": [0, 0, 0, 980, 0], "field_std": [1, 1, 1, 1, 1],
                          "satellite_mean": [0] * 4, "satellite_std": [1] * 4}}))

    class FakeReader:
        provenance = {"catalog": "fake"}

        def read(self, init):
            g = rng.normal(size=(len(leads), 5, 32, 64)).astype(np.float32)
            g[:, 3] += 980
            return g, rng.normal(size=(len(leads), 5, 5)).astype(np.float32)

    original = nwp_data.SOURCES["aifs"]["leads"]
    nwp_data.SOURCES["aifs"]["leads"] = leads
    nwp_data.SOURCES["aifs"]["splits"]["val"] = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC))
    try:
        detail = nwp_data.build_month("aifs", key, era5 / "shards" / key, nwp / "shards" / key, FakeReader())
    finally:
        nwp_data.SOURCES["aifs"]["leads"] = original
        nwp_data.SOURCES["aifs"]["splits"]["val"] = (datetime(2024, 11, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC))
    manifest = {"source": "aifs", "leads": leads, "stations": [s["id"] for s in stations],
                "months": [{"month": key, "path": f"shards/{key}", "era5_shard": f"shards/{key}",
                            "era5_split": split, "samples": detail["samples"]}]}
    (nwp / "manifest.json").write_text(json.dumps(manifest))
    return era5, nwp, detail


class EndToEndTest(unittest.TestCase):
    def check(self, leads):
        torch.manual_seed(4)
        with tempfile.TemporaryDirectory() as temp:
            era5, nwp, detail = synthetic_stores(temp, leads)
            self.assertEqual(detail["samples"], 3)
            self.assertEqual(detail["split_counts"]["val"], 3)
            era5_manifest = json.loads((era5 / "manifest.json").read_text())
            nwp_manifest = json.loads((nwp / "manifest.json").read_text())
            dataset = NWPDataset(era5, nwp, "val", era5_manifest, nwp_manifest)
            self.assertEqual(len(dataset), 3)
            target, sat, query, station, forecast, forecast_station, obs, clock = dataset[0]
            self.assertEqual(tuple(target.shape), (49, 5, 32, 64))  # hourly labels for any source
            self.assertEqual(tuple(forecast.shape), (len(leads), 5, 32, 64))
            self.assertEqual(tuple(station.shape), (49, 5, 4))
            self.assertEqual(tuple(obs.shape), (20,))
            self.assertEqual(tuple(clock.shape), (49, 4))
            scales = forecast_error_scales(dataset)
            self.assertEqual(len(scales), 4)
            self.assertTrue(all(np.isfinite(scales)) and min(scales) > 0)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            mean, std = torch.tensor([0, 0, 0, 980., 0]), torch.ones(5)
            for mode in ("nwp", "nwp_autoregressive"):
                model = NWPHyperDS(mode, satellite_shape=(32, 64))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                metrics = train_epoch(model, loader, optimizer, "cpu", mean, std, leads, [0, 1, 2, 3, 4],
                                      torch.tensor(scales), hours_per_step=3, grid_cells=8)
                self.assertEqual(metrics["steps"], 2 * 3)
                self.assertGreater(model.clock_projection.weight.abs().sum().item(), 0)
                output = Path(temp) / f"{mode}.npz"
                result = evaluate(model, loader, "cpu", mean, std, leads, [0], scales, output,
                                  [s["id"] for s in nwp_data.STATIONS])
                self.assertEqual(set(result["selection_by_variable"]), {"wind_speed", "t2m"})
                metrics_ = result["station_metrics"]
                expected = np.mean([(metrics_["model"]["rmse"][0][v] / metrics_["nwp"]["rmse"][0][v]) ** 2 for v in (0, 1)])
                self.assertAlmostEqual(result["selection_score"], expected, places=5)
                self.assertTrue(np.isfinite(result["validation_station_mse"]))
                self.assertIn("nwp", result["station_metrics"])
                self.assertEqual(len(result["by_lead"]), 49)
                self.assertEqual(result["interpolated_hour_metrics"] is None, len(leads) == 49)
                with np.load(output) as saved:
                    self.assertEqual(saved["forecast"].shape, (3, 49, 5, 4))
                    np.testing.assert_array_equal(saved["lead_times_hours"], np.arange(49))
                    # Raw-NWP baseline at issued leads is the stored forecast itself.
                    issued = list(leads).index(6)
                    raw = np.load(nwp / "shards/2024-01/station.npy")[:, issued, :, 2]
                    np.testing.assert_allclose(saved["nwp_forecast"][:, 6, :, 1], raw, rtol=1e-5)

    def test_six_hourly_source_is_queried_and_scored_hourly(self):
        self.check(list(range(0, 49, 6)))

    def test_hourly_source(self):
        self.check(list(range(49)))


if __name__ == "__main__":
    unittest.main()

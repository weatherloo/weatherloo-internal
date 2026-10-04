import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "climatology",
    "cnn_lstm_bias_correction",
    "ecmwf_aifs",
    "gefs_mean",
    "gfs_analysis",
    "gfs_interpolated",
    "graphcast",
    "hrdps_analysis",
    "hrrr_interpolated",
    "linear_regression",
    "persistence",
    "unet",
)


class RepositoryLayoutTests(unittest.TestCase):
    def test_benchmark_source_and_data_have_separate_homes(self):
        for method in METHODS:
            with self.subTest(method=method):
                pipeline = ROOT / "pipelines" / "benchmarking" / method
                benchmark_data = ROOT / "data" / "benchmarks" / method
                legacy_entry = (
                    ROOT / "benchmarking-site" / "data" / method / "compute_benchmark.py"
                )

                self.assertTrue((pipeline / "compute_benchmark.py").is_file())
                self.assertTrue(benchmark_data.is_dir())
                self.assertTrue(legacy_entry.is_file())
                self.assertIn("pipelines", legacy_entry.read_text())
                self.assertFalse((benchmark_data / "compute_benchmark.py").exists())

    def test_generated_outputs_are_outside_source_directories(self):
        for path in (
            "outputs/artifacts/hrrr_bias_correction",
            "outputs/lstm_training/sweeps",
            "outputs/lstm_training/retrained",
            "outputs/models/unet/checkpoints",
            "outputs/models/unet-hrrr/checkpoints",
            "outputs/logs",
            "outputs/viz",
        ):
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_dir())

        for path in (
            "artifacts",
            "logs",
            "viz",
            "lstm_training/output",
            "lstm_training/retrain_output",
            "models/unet/checkpoints",
            "models/unet-hrrr/checkpoints",
        ):
            with self.subTest(path=path):
                self.assertFalse((ROOT / path).exists())


if __name__ == "__main__":
    unittest.main()

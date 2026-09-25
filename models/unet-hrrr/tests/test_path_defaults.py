import importlib.util
import sys
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "path_defaults.py"
spec = importlib.util.spec_from_file_location("path_defaults", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PathDefaultsTests(unittest.TestCase):
    def test_prefers_unet_data_root(self):
        env = {
            "UNET_DATA_ROOT": "/tmp/unet-data",
            "WEATHERLOO_DATA_ROOT": "/tmp/weatherloo",
        }
        self.assertEqual(module.resolve_default_data_dir(env), "/tmp/unet-data")

    def test_uses_weatherloo_data_root_raw(self):
        env = {"WEATHERLOO_DATA_ROOT": "/tmp/weatherloo"}
        self.assertEqual(module.resolve_default_data_dir(env), "/tmp/weatherloo/raw")

    def test_falls_back_to_repo_data_raw(self):
        env = {}
        expected = str((Path(__file__).resolve().parents[3] / "data" / "raw").resolve())
        self.assertEqual(module.resolve_default_data_dir(env), expected)


if __name__ == "__main__":
    unittest.main()

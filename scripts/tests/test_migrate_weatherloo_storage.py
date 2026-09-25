import importlib.util
import sys
import tempfile
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "migrate_weatherloo_storage.py"
spec = importlib.util.spec_from_file_location("migrate_weatherloo_storage", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class MigrateWeatherlooStorageTests(unittest.TestCase):
    def test_build_mappings_has_expected_targets(self):
        with (
            tempfile.TemporaryDirectory() as ts,
            tempfile.TemporaryDirectory() as td,
            tempfile.TemporaryDirectory() as tr,
        ):
            source_root = Path(ts)
            data_root = Path(td)
            repo_root = Path(tr)
            mappings = module.build_mappings(source_root, data_root, repo_root)
            by_label = {m.label: m for m in mappings}
            self.assertEqual(by_label["raw-hrrr"].dst, data_root / "raw" / "hrrr")
            self.assertEqual(by_label["raw-hrrr"].src, source_root / "hrrr")
            self.assertEqual(by_label["cache-root"].dst, data_root / "cache")
            self.assertEqual(by_label["repo-data"].dst, data_root / "processed" / "repo_data")
            self.assertEqual(by_label["repo-artifacts"].src, repo_root / "artifacts")

    def test_same_file_uses_size_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.txt"
            b = root / "b.txt"
            c = root / "c.txt"
            a.write_text("same")
            b.write_text("same")
            c.write_text("different")
            self.assertTrue(module.same_file(a, b))
            self.assertFalse(module.same_file(a, c))

    def test_copy_or_link_uses_resolved_symlink_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src.txt"
            dst = root / "dst.txt"
            src.write_text("abc")
            module.copy_or_link(src, dst, symlink=True, dry_run=False)
            self.assertTrue(dst.is_symlink())
            self.assertEqual(dst.resolve(), src.resolve())


if __name__ == "__main__":
    unittest.main()

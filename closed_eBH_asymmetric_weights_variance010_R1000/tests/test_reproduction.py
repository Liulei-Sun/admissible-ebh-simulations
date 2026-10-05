"""Guard scientific comparisons against changed selections and stale input."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from compare_reproduction import compare_learned_stage, validate_generated_files
from study_common import sha256_file


class ReproductionTests(unittest.TestCase):
    def test_changed_membership_is_not_accepted_as_a_reproduction(self):
        source = ROOT / "data" / "batch1_seed20260803" / "n20" / "clean"
        with tempfile.TemporaryDirectory() as temporary:
            fresh = Path(temporary) / "fresh"
            archived = Path(temporary) / "archived"
            for directory in (fresh, archived):
                directory.mkdir()
                for name in ("procedure1_results.csv", "procedure1_membership.csv"):
                    frame = pd.read_csv(source / name, keep_default_na=False)
                    frame.loc[frame.replication == 1].to_csv(directory / name, index=False)
            self.assertEqual(compare_learned_stage(fresh, archived, "procedure1"), 1)
            path = fresh / "procedure1_membership.csv"
            changed = pd.read_csv(path)
            changed.loc[0, "method1_member"] = 1 - changed.loc[0, "method1_member"]
            changed.to_csv(path, index=False)
            with self.assertRaises(AssertionError):
                compare_learned_stage(fresh, archived, "procedure1")

    def test_resume_detects_changed_generated_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {}
            source = ROOT / "data"
            for original in source.rglob("*.csv"):
                if original.name.startswith("procedure"):
                    continue
                relative = original.relative_to(source).as_posix()
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("e\n1\n", encoding="utf-8")
                files[relative] = sha256_file(path)
            manifest_path = root / "generation_manifest.json"
            manifest = {"schema": "generated-inputs-v1", "files": files}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            validate_generated_files(root)
            path.write_text("e\n2\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_generated_files(root)

    def test_empty_or_absolute_manifest_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for manifest in ({"files": {}},
                             {"schema": "generated-inputs-v1", "files": {}},
                             {"schema": "generated-inputs-v1", "files": {str(root / "input.csv"): "x"}}):
                (root / "generation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    validate_generated_files(root)


if __name__ == "__main__":
    unittest.main()

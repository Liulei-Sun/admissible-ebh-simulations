"""A portable audit preserves historical metadata and verifies the science."""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from audit_results import compare_historical_audit


class HistoricalAuditTests(unittest.TestCase):
    def setUp(self):
        self.historical = json.loads((ROOT / "audit" / "audit.json").read_text(encoding="utf-8"))

    def test_different_runtime_and_roundoff_are_accepted(self):
        current = copy.deepcopy(self.historical)
        current["runtime"]["python"] = "3.12.14"
        current["audit_script_sha256"] = "revised-source"
        current["solver_source_sha256"] = {"solver.py": "revised-source"}
        current["results"][0]["FDR"] += 1e-17
        for key in current["maximum_scaled_regeneration_errors"]:
            current["maximum_scaled_regeneration_errors"][key] = 1e-14
        compare_historical_audit(self.historical, current)

    def test_changed_scientific_values_counts_and_excess_errors_fail(self):
        for field in ("result", "count", "error"):
            current = copy.deepcopy(self.historical)
            if field == "result":
                current["results"][0]["TPR"] += .01
            elif field == "count":
                current["counts"]["learned_result_rows"] -= 1
            else:
                key = next(iter(current["maximum_scaled_regeneration_errors"]))
                current["maximum_scaled_regeneration_errors"][key] = 1e-8
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                compare_historical_audit(self.historical, current)


if __name__ == "__main__":
    unittest.main()

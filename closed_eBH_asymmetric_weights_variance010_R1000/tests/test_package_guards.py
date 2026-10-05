"""Fail-closed path checks used by the portable package verifier."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from verify_package import safe_relative  # noqa: E402


class PackageGuardTests(unittest.TestCase):
    def test_accepts_normal_relative_paths(self):
        self.assertEqual(
            safe_relative("data/batch1_seed20260803/n20/clean/learned_instances.csv").as_posix(),
            "data/batch1_seed20260803/n20/clean/learned_instances.csv",
        )

    def test_rejects_traversal_absolute_and_backslash_paths(self):
        for value in ("../escape", "/absolute", "data/../escape", "data\\file.csv", ""):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                safe_relative(value)


if __name__ == "__main__":
    unittest.main()

"""Tests for the publication-only study contract and audit certificate."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "solver"))

from audit_results import compare_frame, exact_adjusted_certified  # noqa: E402
from original_formula_oracle import original_formula_merger  # noqa: E402
from study_common import (  # noqa: E402
    PERSISTENT_NOISE_SALT,
    format4,
    load_config,
    ordinary_ebh,
    splitmix64,
)


class StudyContractTests(unittest.TestCase):
    def test_only_reported_conditions_are_configured(self):
        config = load_config(ROOT)
        self.assertEqual(
            list(config["conditions"]),
            ["clean", "signed_square", "persistent_noise"],
        )
        persistent = config["conditions"]["persistent_noise"]
        self.assertEqual(persistent["error_variance"], 0.1)
        self.assertEqual(persistent["error_standard_deviation"], math.sqrt(0.1))

    def test_seed_stream_identity(self):
        primary = splitmix64(20260803 ^ splitmix64(1))
        persistent = splitmix64(primary ^ PERSISTENT_NOISE_SALT)
        self.assertEqual(primary, 6841645300402837787)
        self.assertEqual(persistent, 1187528303394557131)

    def test_adjusted_audit_matches_all_intersections(self):
        rng = np.random.default_rng(20260814)
        for k in range(3, 10):
            for _ in range(12):
                e = np.exp(rng.normal(0.4, 1.5, size=k))
                score = np.exp(rng.normal(0.0, 1.2, size=k))
                alpha = float(rng.uniform(0.05, 0.4))
                selected = rng.random(k) < 0.55
                if not selected.any():
                    selected[0] = True
                r = int(selected.sum())
                brute_certified = True
                for size in range(1, k + 1):
                    for subset in itertools.combinations(range(k), size):
                        index = np.asarray(subset, dtype=int)
                        count = int(selected[index].sum())
                        if count == 0:
                            continue
                        merger = original_formula_merger(e, score, subset)
                        if count - alpha * r * merger > 1e-10:
                            brute_certified = False
                            break
                    if not brute_certified:
                        break
                self.assertEqual(
                    exact_adjusted_certified(e, score, alpha, selected),
                    brute_certified,
                )

    def test_testing_ebh_is_deterministic_under_ties(self):
        e = np.asarray([40.0, 40.0, 0.0, 0.0])
        selected = ordinary_ebh(e, 0.05)
        self.assertTrue(np.array_equal(selected, np.asarray([True, True, False, False])))

    def test_stored_empty_index_lists_round_trip_as_empty_strings(self):
        expected = pd.DataFrame(
            {"replication": [1, 2], "selected_indices": ["", "3;7"]}
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "round_trip.csv"
            expected.to_csv(path, index=False)
            compare_frame(path, expected)

    def test_publication_rounding_is_explicitly_half_up(self):
        self.assertEqual(format4(0.15835), "0.1584")
        self.assertEqual(format4(0.31510), "0.3151")


if __name__ == "__main__":
    unittest.main()

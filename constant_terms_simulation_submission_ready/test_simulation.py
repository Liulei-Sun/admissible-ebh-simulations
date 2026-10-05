"""Independent rational-arithmetic oracles and public-CLI regression tests.

Run with ``python -m unittest -v``. These tests require only the pinned NumPy
dependency and the Python standard library.
"""
from __future__ import annotations

import csv
from fractions import Fraction
from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import run_simulation as sim
from verify_results import compare_csv


def exact_certificate(e_values, rejection_mask, alpha, beta):
    """Implement the manuscript's all-intersection definition independently."""
    k = len(e_values)
    r = rejection_mask.bit_count()
    if r == 0:
        return True
    for intersection in range(1, 1 << k):
        n = intersection.bit_count()
        ell = (intersection & rejection_mask).bit_count()
        constant = (beta * k - n) / (beta * k - alpha * n) if n < beta * k else Fraction(0)
        average = sum(
            (Fraction(e_values[i]) for i in range(k) if intersection & (1 << i)),
            Fraction(0),
        ) / n
        if constant + (1 - constant) * average < Fraction(ell, r) / alpha:
            return False
    return True


class AlgorithmTests(unittest.TestCase):
    def test_exact_oracle_including_large_alpha_and_ties(self):
        rng = np.random.default_rng(21812)
        for k, alpha, beta in product(
            (2, 3, 5),
            (Fraction(1, 20), Fraction(1, 2), Fraction(3, 5), Fraction(9, 10)),
            (Fraction(0), Fraction(1, 5), Fraction(1, 2), Fraction(1)),
        ):
            vectors = [
                [0] * k,
                [20] * k,
                sorted(rng.integers(0, 41, k).tolist(), reverse=True),
            ]
            for values in vectors:
                e = np.asarray(values, dtype=float)
                base = max(
                    (r for r in range(1, k + 1) if Fraction(values[r - 1]) >= k / (alpha * r)),
                    default=0,
                )
                self.assertEqual(sim.ebh_size(e, float(alpha)), base)
                maximum = 0
                constrained_maximum = 0
                for mask in range(1 << k):
                    if exact_certificate(values, mask, alpha, beta):
                        maximum = max(maximum, mask.bit_count())
                        if mask & ((1 << base) - 1) == (1 << base) - 1:
                            constrained_maximum = max(constrained_maximum, mask.bit_count())
                self.assertEqual(sim.largest_prefix(e, float(alpha), float(beta)), maximum)
                self.assertEqual(
                    sim.largest_prefix(e, float(alpha), float(beta), minimum_size=base),
                    constrained_maximum,
                )
                for r in range(1, k + 1):
                    expected = exact_certificate(values, (1 << r) - 1, alpha, beta)
                    self.assertEqual(sim.candidate_fast(e, r, float(alpha), float(beta)), expected)
                    self.assertEqual(sim.candidate_convex(e, r, float(alpha), float(beta)), expected)

    def test_sharp_boundaries_and_beta_display(self):
        self.assertEqual(sim.admissibility_threshold(5000, 0.05), 0.004)
        self.assertEqual(sim.admissibility_threshold(5000, 0.1), 0.002)
        self.assertEqual(sim.admissibility_threshold(2, 0.05), 0.5)
        for beta in (0.0, 0.004, 0.1234, 0.1235, 1e-14, 0.8, 1.0):
            self.assertEqual(float(sim.format_beta(beta)), beta)
        self.assertEqual(sim.format_beta(0.8), "0.80")

    def test_no_cancellation_of_weak_tail(self):
        e = np.array([1e200, 90.0, 50.0, 10.0, 0.0])
        for beta in (0.0, 0.4, 0.8, 1.0):
            for r in range(1, e.size + 1):
                expected = sim.candidate_bruteforce(e, r, 0.05, beta)
                self.assertEqual(sim.candidate_convex(e, r, 0.05, beta), expected)

    def test_overflow_is_reported(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            sim.simulate_scenario(
                np.zeros((2, 10)), k=10, nonnulls=2, signal=100,
                alpha=0.05, betas=(0.0,),
            )


class CommandLineTests(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run(
            [sys.executable, str(Path(sim.__file__)), *arguments],
            text=True, capture_output=True, check=False,
        )

    def test_invalid_signal_seed_and_baseline(self):
        for arguments in (
            ("--signal", "nan"), ("--signal", "inf"),
            ("--seed", "-1"), ("--betas", "1e-14,0.8"),
            ("--self-test-only", "--skip-self-test"),
        ):
            result = self.invoke(*arguments)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_small_run_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = [Path(directory) / name for name in ("first", "second")]
            for output in outputs:
                result = self.invoke(
                    "--hypotheses", "20", "--nonnulls", "2,5",
                    "--replications", "8", "--betas", "0,0.1234,0.1235,1",
                    "--skip-self-test", "--output-dir", str(output),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((output / "run_metadata.json").is_file())
                table = (output / "constant_terms_tpr_table.tex").read_text(encoding="utf-8")
                self.assertIn("0.1234", table)
                self.assertIn("0.1235", table)
                with (output / "constant_terms_results.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 10)
                self.assertTrue(all(float(row["ebh_containment_rate"]) == 1.0 for row in rows))
            compare_csv(outputs[0] / "constant_terms_results.csv", outputs[1] / "constant_terms_results.csv")

    def test_reference_verifier_rejects_changed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = [Path(directory) / name for name in ("left.csv", "right.csv")]
            left.write_text("tpr,python_version\n0.2,3.12.13\n", encoding="utf-8")
            right.write_text("tpr,python_version\n0.2,3.12.14\n", encoding="utf-8")
            self.assertEqual(compare_csv(left, right), 1)
            right.write_text("tpr,python_version\n0.3,3.12.14\n", encoding="utf-8")
            with self.assertRaises(AssertionError):
                compare_csv(left, right)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Exhaustive checks for Method 1 with gamma=1."""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "solver"))

from choice2_oracle import ABS_TOL, ordinary_ebh  # noqa: E402
from method1_oracle import Method1Oracle, method1_merger  # noqa: E402
import optimize_maximal_learned as outer  # noqa: E402


def brute_violations(e, score, alpha, selected):
    k = e.size
    report_size = int(np.sum(selected))
    found = []
    if report_size == 0:
        return found
    for size in range(1, k + 1):
        for subset in itertools.combinations(range(k), size):
            count = int(np.sum(selected[np.asarray(subset, dtype=int)]))
            if count == 0:
                continue
            merger = method1_merger(e, score, subset)
            violation = count - alpha * report_size * merger
            if violation > ABS_TOL:
                found.append((violation, subset, merger))
    return found


class Method1OracleTests(unittest.TestCase):
    def test_empty_is_certified_but_ebh_need_not_be(self):
        e = np.asarray([20.0, 0.0])
        score = np.asarray([0.001, 1.0])
        oracle = Method1Oracle(e, score, 0.1)
        empty = np.zeros(2, dtype=bool)
        ebh = ordinary_ebh(e, 0.1)
        self.assertTrue(oracle.is_certified(empty))
        self.assertTrue(np.array_equal(ebh, np.asarray([True, False])))
        self.assertFalse(oracle.is_certified(ebh))

    def test_formula_scale_invariance_and_equal_scores(self):
        rng = np.random.default_rng(572091)
        for k in range(2, 10):
            e = np.exp(rng.normal(size=k))
            score = np.exp(rng.normal(size=k))
            for size in range(1, k + 1):
                for subset in itertools.combinations(range(k), size):
                    idx = np.asarray(subset, dtype=int)
                    expected = float(np.dot(
                        score[idx] / np.sum(score[idx]), e[idx]
                    ))
                    observed = method1_merger(e, score, subset)
                    self.assertAlmostEqual(observed, expected, places=12)
                    self.assertAlmostEqual(
                        method1_merger(e, 19.7 * score, subset),
                        observed,
                        places=12,
                    )
                    self.assertAlmostEqual(
                        method1_merger(e, np.ones(k), subset),
                        float(np.mean(e[idx], dtype=np.longdouble)),
                        places=12,
                    )

    def test_separator_matches_every_intersection(self):
        rng = np.random.default_rng(337810)
        for k in range(2, 10):
            for _ in range(20):
                e = np.exp(rng.normal(1.0, 1.5, size=k))
                score = np.exp(rng.normal(0.0, 1.2, size=k))
                alpha = float(rng.uniform(0.08, 0.45))
                selected = rng.random(k) < 0.5
                oracle = Method1Oracle(e, score, alpha)
                brute = brute_violations(e, score, alpha, selected)
                cuts = oracle.separate(selected, max_cuts=max(1, k))
                self.assertEqual(bool(cuts), bool(brute))
                for violation, subset, merger in cuts:
                    self.assertGreater(violation, ABS_TOL)
                    self.assertAlmostEqual(
                        merger, method1_merger(e, score, subset), places=12
                    )

    def test_outer_optimizer_matches_lexicographic_global_maximum(self):
        rng = np.random.default_rng(775312)

        def no_greedy(oracle, initial, witness_limit):
            del oracle, witness_limit
            return initial.copy(), 0, {}

        old_oracle = outer.LearnedOracle
        old_greedy = outer.deterministic_greedy
        try:
            outer.LearnedOracle = Method1Oracle
            outer.deterministic_greedy = no_greedy
            for k in range(3, 9):
                for replication in range(5):
                    e = np.exp(rng.normal(3.0, 2.0, size=k))
                    score = np.exp(rng.normal(0.0, 1.2, size=k))
                    nonnull = rng.random(k) < 0.4
                    nonnull[0] = True
                    alpha = 0.2
                    seed = np.zeros(k, dtype=bool)
                    instance = outer.Instance(
                        replication=100 * k + replication,
                        e=e,
                        score=score,
                        nonnull=nonnull,
                        prefix=seed,
                    )
                    best_size = -1
                    best_e_sum = np.longdouble(-np.inf)
                    best_memberships = set()
                    for mask in range(1 << k):
                        candidate = np.asarray(
                            [bool(mask & (1 << i)) for i in range(k)],
                            dtype=bool,
                        )
                        if not brute_violations(e, score, alpha, candidate):
                            size = int(np.sum(candidate))
                            e_sum = np.sum(e[candidate], dtype=np.longdouble)
                            membership = tuple(int(i) for i in np.flatnonzero(candidate))
                            if size > best_size or (
                                    size == best_size and e_sum > best_e_sum):
                                best_size = size
                                best_e_sum = e_sum
                                best_memberships = {membership}
                            elif size == best_size and e_sum == best_e_sum:
                                best_memberships.add(membership)

                    result = outer.optimize_instance(
                        instance, alpha, 1, 30, 30, 4, 30.0
                    )
                    membership = tuple(
                        int(value)
                        for value in result.maximal_indices.split(";")
                        if value
                    )
                    selected = np.zeros(k, dtype=bool)
                    selected[np.asarray(membership, dtype=int)] = True
                    selected_e_sum = np.sum(e[selected], dtype=np.longdouble)
                    self.assertEqual(result.status, "proven_maximal")
                    self.assertEqual(result.secondary_status, "proven_optimal")
                    self.assertEqual(
                        result.secondary_rule,
                        "maximize_sum_testing_e_values_at_maximum_cardinality",
                    )
                    self.assertEqual(result.maximal_size, best_size)
                    self.assertEqual(result.proof_upper_bound, best_size)
                    self.assertEqual(membership, tuple(np.flatnonzero(selected)))
                    self.assertIn(membership, best_memberships)
                    self.assertLessEqual(
                        abs(selected_e_sum - best_e_sum),
                        np.longdouble("1e-12")
                        * max(np.longdouble(1.0), abs(best_e_sum)),
                    )
                    self.assertLessEqual(
                        abs(np.longdouble(result.secondary_value) - best_e_sum),
                        np.longdouble("1e-12")
                        * max(np.longdouble(1.0), abs(best_e_sum)),
                    )
                    self.assertFalse(brute_violations(e, score, alpha, selected))
        finally:
            outer.LearnedOracle = old_oracle
            outer.deterministic_greedy = old_greedy


if __name__ == "__main__":
    unittest.main()

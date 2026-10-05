#!/usr/bin/env python3
"""Exhaustive checks for the user's original prior-score formula."""

from __future__ import annotations

import itertools
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "solver"))

from choice2_oracle import (  # noqa: E402
    ABS_TOL,
    SeparationUnknown,
    ordinary_ebh,
)
from original_formula_oracle import (  # noqa: E402
    OriginalFormulaOracle,
    PREFIX_FALLBACK,
    original_formula_merger,
)
import optimize_maximal_learned as outer  # noqa: E402


def brute_violations(e, score, alpha, selected):
    k = e.size
    r = int(np.sum(selected))
    found = []
    if r == 0:
        return found
    for size in range(1, k + 1):
        for subset in itertools.combinations(range(k), size):
            count = int(np.sum(selected[np.asarray(subset, dtype=int)]))
            if count == 0:
                continue
            merger = original_formula_merger(e, score, subset)
            violation = count - alpha * r * merger
            if violation > ABS_TOL:
                found.append((violation, subset, merger))
    return found


class OriginalFormulaOracleTests(unittest.TestCase):
    def test_largest_published_set_uses_exact_gray_separator(self):
        oracle = OriginalFormulaOracle(np.ones(24), np.ones(24), .05)
        selected = np.arange(24) < 23
        with patch.object(oracle, "_exact_violation_prefix_gray", return_value=None) as exact:
            self.assertIsNone(oracle._exact_violation_prefix(selected))
            exact.assert_called_once()

    def test_formula_floor_scale_and_equal_scores(self):
        rng = np.random.default_rng(490071)
        for k in range(2, 10):
            e = np.exp(rng.normal(size=k))
            score = np.exp(rng.normal(size=k))
            for size in range(1, k + 1):
                for subset in itertools.combinations(range(k), size):
                    idx = np.asarray(subset, dtype=int)
                    local = score[idx].astype(np.longdouble)
                    weights = (
                        np.longdouble(1.0) / k
                        + np.longdouble(k - size) / k
                        * local / np.sum(local, dtype=np.longdouble)
                    )
                    expected = float(np.dot(weights, e[idx]))
                    observed = original_formula_merger(e, score, subset)
                    self.assertAlmostEqual(observed, expected, places=12)
                    self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=13)
                    self.assertGreaterEqual(
                        float(np.min(weights)) + 1e-14, 1.0 / k
                    )
                    self.assertAlmostEqual(
                        original_formula_merger(e, 11.9 * score, subset),
                        observed,
                        places=12,
                    )
                    self.assertAlmostEqual(
                        original_formula_merger(e, np.ones(k), subset),
                        float(np.mean(e[idx], dtype=np.longdouble)),
                        places=12,
                    )

    def test_qubo_expansion_matches_raw_separation_expression(self):
        rng = np.random.default_rng(19373)
        for k in range(2, 9):
            for _ in range(12):
                e = np.exp(rng.normal(0.5, 1.8, size=k))
                score = np.exp(rng.normal(0.0, 1.5, size=k))
                alpha = float(rng.uniform(0.05, 0.45))
                selected = rng.random(k) < 0.5
                oracle = OriginalFormulaOracle(e, score, alpha)
                costs = oracle._qubo_costs(selected)
                r = int(np.sum(selected))

                for mask in range(1 << k):
                    candidate = np.asarray(
                        [bool(mask & (1 << i)) for i in range(k)],
                        dtype=bool,
                    )
                    chosen_pairs = (
                        candidate[oracle._qubo_pair_i]
                        & candidate[oracle._qubo_pair_j]
                    )
                    expanded = (
                        np.dot(costs[:k], candidate.astype(float))
                        + np.dot(costs[k:], chosen_pairs.astype(float))
                    )

                    idx = np.flatnonzero(candidate)
                    if idx.size == 0:
                        raw = np.longdouble(0.0)
                    else:
                        local_score = oracle.score[idx].astype(np.longdouble)
                        local_e = oracle.e[idx].astype(np.longdouble)
                        score_sum = np.sum(local_score, dtype=np.longdouble)
                        e_sum = np.sum(local_e, dtype=np.longdouble)
                        score_e_sum = np.dot(local_score, local_e)
                        size = np.longdouble(idx.size)
                        count = np.longdouble(np.sum(selected[idx]))
                        raw = (
                            np.longdouble(alpha)
                            * np.longdouble(r)
                            * (
                                score_sum * e_sum
                                + (np.longdouble(k) - size) * score_e_sum
                            )
                            - np.longdouble(k)
                            * (count - np.longdouble(ABS_TOL))
                            * score_sum
                        )
                    self.assertAlmostEqual(
                        expanded / max(1.0, abs(float(raw))),
                        float(raw) / max(1.0, abs(float(raw))),
                        places=10,
                    )

    def test_separator_matches_exhaustive_enumeration(self):
        rng = np.random.default_rng(781920)
        for k in range(3, 9):
            for _ in range(8):
                e = np.exp(rng.normal(1.0, 1.5, size=k))
                score = np.exp(rng.normal(0.0, 1.2, size=k))
                alpha = float(rng.uniform(0.08, 0.45))
                selected = rng.random(k) < 0.5
                if not np.any(selected):
                    selected[int(rng.integers(k))] = True
                oracle = OriginalFormulaOracle(
                    e, score, alpha, separator_time_limit=30.0
                )
                brute = brute_violations(e, score, alpha, selected)
                cuts = oracle.separate(selected, max_cuts=1)
                self.assertEqual(bool(cuts), bool(brute))
                if cuts:
                    violation, subset, merger = cuts[0]
                    self.assertGreater(violation, ABS_TOL)
                    self.assertAlmostEqual(
                        merger,
                        original_formula_merger(e, score, subset),
                        places=12,
                    )

    def test_exact_prefix_separator_matches_all_intersections(self):
        rng = np.random.default_rng(20260801)
        for k in range(2, 11):
            for case in range(30):
                e = np.exp(rng.normal(size=k))
                if case % 7 == 0:
                    e = np.round(e, 1)
                score = np.exp(rng.normal(size=k))
                alpha = float(rng.uniform(0.01, 0.8))
                selected = rng.random(k) < rng.uniform(0.05, 0.95)
                if not np.any(selected):
                    selected[int(rng.integers(k))] = True
                oracle = OriginalFormulaOracle(e, score, alpha)
                exact = oracle._exact_violation_prefix(selected)
                self.assertIsNot(exact, PREFIX_FALLBACK)
                brute = brute_violations(e, score, alpha, selected)
                self.assertEqual(exact is not None, bool(brute))
                if exact is not None:
                    violation, subset, merger = exact
                    self.assertGreater(violation, ABS_TOL)
                    self.assertAlmostEqual(
                        merger,
                        original_formula_merger(e, score, subset),
                        places=12,
                    )

    def test_gray_prefix_separator_matches_established_exact_enumeration(self):
        rng = np.random.default_rng(20260814)
        for r in range(1, 13):
            for _ in range(2):
                k = 24
                e = np.exp(rng.normal(0.5, 1.7, size=k))
                score = np.exp(rng.normal(0.0, 1.4, size=k))
                alpha = float(rng.uniform(0.03, 0.4))
                selected = np.zeros(k, dtype=bool)
                selected[rng.choice(k, size=r, replace=False)] = True
                oracle = OriginalFormulaOracle(e, score, alpha)
                established = oracle._exact_violation_prefix(selected)
                gray = oracle._exact_violation_prefix_gray(selected)
                self.assertIsNot(established, PREFIX_FALLBACK)
                self.assertIsNot(gray, PREFIX_FALLBACK)
                self.assertEqual(established is None, gray is None)
                if established is not None and gray is not None:
                    self.assertEqual(established[1], gray[1])
                    self.assertAlmostEqual(established[0], gray[0], places=12)
                    self.assertAlmostEqual(established[2], gray[2], places=12)

        # The production dispatcher must use exact Gray enumeration above the
        # small-table cap instead of the numerically fragile QUBO formulation.
        k = 20
        selected = np.zeros(k, dtype=bool)
        selected[:17] = True
        oracle = OriginalFormulaOracle(
            np.exp(rng.normal(size=k)),
            np.exp(rng.normal(size=k)),
            0.05,
        )
        dispatched = oracle._exact_violation_prefix(selected)
        direct = oracle._exact_violation_prefix_gray(selected)
        self.assertIsNot(dispatched, PREFIX_FALLBACK)
        self.assertEqual(dispatched is None, direct is None)

    def test_exact_prefix_separator_handles_ties_and_boundary_fallback(self):
        e = np.asarray([0.2, 0.2, 0.2, 1.0, 1.0, 3.0])
        score = np.asarray([0.01, 100.0, 0.3, 17.0, 0.02, 9.0])
        selected = np.asarray([True, False, True, False, False, False])
        oracle = OriginalFormulaOracle(e, score, 0.2)
        exact = oracle._exact_violation_prefix(selected)
        brute = brute_violations(e, score, 0.2, selected)
        self.assertEqual(exact is not None, bool(brute))

        witness_oracle = OriginalFormulaOracle(
            np.asarray([0.01, 0.02, 0.03]),
            np.asarray([1.0, 2.0, 3.0]),
            0.1,
        )
        witness_selected = np.asarray([True, False, False])
        with patch.object(witness_oracle, "merger", return_value=1e20):
            ambiguous = witness_oracle._exact_violation_prefix(witness_selected)
        self.assertIs(ambiguous, PREFIX_FALLBACK)

    def test_ordinary_ebh_is_certified_exhaustively(self):
        rng = np.random.default_rng(671221)
        for k in range(2, 9):
            for _ in range(20):
                e = np.exp(rng.normal(2.5, 2.0, size=k))
                score = np.exp(rng.normal(size=k))
                selected = ordinary_ebh(e, 0.2)
                self.assertFalse(brute_violations(e, score, 0.2, selected))

    def test_highs_status_and_direct_witness_rules(self):
        oracle = OriginalFormulaOracle(
            np.asarray([0.1, 0.2, 0.3]),
            np.asarray([1.0, 2.0, 3.0]),
            0.1,
        )
        selected = np.asarray([True, False, False])

        timeout = SimpleNamespace(
            status=1,
            message="time limit",
            fun=None,
            x=None,
            mip_dual_bound=None,
            mip_node_count=0,
        )
        with patch("original_formula_oracle.milp", return_value=timeout):
            with self.assertRaises(SeparationUnknown):
                oracle._exact_violation_highs(selected)

        ambiguous = SimpleNamespace(
            status=0,
            message="optimal",
            fun=-1.0,
            x=np.zeros(oracle.k + oracle._qubo_pair_i.size),
            mip_dual_bound=-1.0,
            mip_node_count=1,
        )
        with patch("original_formula_oracle.milp", return_value=ambiguous):
            with self.assertRaises(SeparationUnknown):
                oracle._exact_violation_highs(selected)

        # A directly rechecked witness is safe even if the solver stopped
        # before proving its global optimum.
        witness_x = np.zeros(oracle.k + oracle._qubo_pair_i.size)
        witness_x[0] = 1.0
        unfinished_with_witness = SimpleNamespace(
            status=1,
            message="time limit",
            fun=-0.5,
            x=witness_x,
            mip_dual_bound=-2.0,
            mip_node_count=2,
        )
        with patch(
                "original_formula_oracle.milp",
                return_value=unfinished_with_witness):
            violation = oracle._exact_violation_highs(selected)
        self.assertIsNotNone(violation)
        self.assertGreater(violation[0], ABS_TOL)
        self.assertEqual(violation[1], (0,))

    def test_outer_optimizer_matches_lexicographic_maximum_over_ebh_supersets(self):
        rng = np.random.default_rng(817731)

        def no_greedy(oracle, initial, witness_limit):
            del oracle, witness_limit
            return initial.copy(), 0, {}

        old_oracle = outer.LearnedOracle
        old_greedy = outer.deterministic_greedy
        try:
            outer.LearnedOracle = lambda e, score, alpha: OriginalFormulaOracle(
                e, score, alpha, separator_time_limit=30.0
            )
            outer.deterministic_greedy = no_greedy
            for k in range(3, 9):
                for replication in range(4):
                    e = np.exp(rng.normal(3.0, 2.0, size=k))
                    score = np.exp(rng.normal(0.0, 1.2, size=k))
                    nonnull = rng.random(k) < 0.4
                    nonnull[0] = True
                    alpha = 0.2
                    seed = ordinary_ebh(e, alpha)
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
                        if not np.all(candidate[seed]):
                            continue
                        if not brute_violations(
                                e, score, alpha, candidate):
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
                        instance, alpha, 1, 20, 20, 4, 30.0
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
                    self.assertTrue(np.all(selected[seed]))
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

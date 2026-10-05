#!/usr/bin/env python3
"""Compute certified maximal extensions for exported learned-weight instances.

The C++ simulation exports the testing e-values, normalized training scores,
nonnull labels, and its certified top-e-value prefix.  This script first makes
a deterministic sequence of certified one-coordinate additions, ordered by
the combined evidence s_i e_i.  It then uses exact constraint generation and
SciPy/HiGHS MILP solves to find a maximum-cardinality certified superset of
that warm start.  After cardinality is proved, a second exact cut-generation
solve selects, among all such maximum-cardinality supersets, one maximizing
the sum of testing e-values.  The secondary rule uses no nonnull labels.
Separation is combinatorially exact; floating-point comparisons use the
tolerances below.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

# Prevent each worker's numerical libraries from starting their own thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ABS_TOL = 1e-10
CAPACITY_TOL = 1e-9


@dataclass
class Instance:
    replication: int
    e: np.ndarray
    score: np.ndarray
    nonnull: np.ndarray
    prefix: np.ndarray


@dataclass
class OptimizationResult:
    replication: int
    problem_sha256: str
    status: str
    secondary_rule: str
    secondary_status: str
    secondary_value: float
    secondary_normalized_value: float
    secondary_normalization: float
    prefix_size: int
    greedy_size: int
    maximal_size: int
    prefix_indices: str
    greedy_indices: str
    maximal_indices: str
    prefix_true: int
    greedy_true: int
    maximal_true: int
    prefix_false: int
    greedy_false: int
    maximal_false: int
    prefix_tpr: float
    greedy_tpr: float
    maximal_tpr: float
    prefix_fdp: float
    greedy_fdp: float
    maximal_fdp: float
    greedy_added: int
    optimizer_added: int
    elapsed_seconds: float
    mip_solves: int
    separation_calls: int
    final_cut_count: int
    initial_singleton_upper_bound: int
    proof_upper_bound: int
    message: str


def make_kappa(score: np.ndarray) -> np.ndarray:
    k = score.size
    decreasing = np.sort(score)[::-1]
    prefix = np.concatenate(([0.0], np.cumsum(decreasing, dtype=np.longdouble)))
    minimum = np.longdouble(decreasing[-1])
    kappa = np.zeros(k + 1)
    for a in range(2, k):
        numerator = prefix[a - 1] - np.longdouble(k - 1) * minimum
        kappa[a] = float(max(np.longdouble(0.0), numerator) / np.longdouble(k - a))
    return kappa


class LearnedOracle:
    def __init__(self, e: np.ndarray, score: np.ndarray, alpha: float):
        self.e = np.asarray(e, dtype=float)
        self.score = np.asarray(score, dtype=float)
        self.alpha = float(alpha)
        if self.e.ndim != 1 or self.score.ndim != 1 or self.e.shape != self.score.shape:
            raise ValueError("e and score must be one-dimensional arrays of equal length")
        if self.e.size < 2:
            raise ValueError("the optimizer requires K >= 2")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0,1)")
        if not np.all(np.isfinite(self.e)) or np.any(self.e < 0.0):
            raise ValueError("e-values must be finite and nonnegative")
        if not np.all(np.isfinite(self.score)) or np.any(self.score <= 0.0):
            raise ValueError("scores must be finite and strictly positive")
        self.k = self.e.size
        self.kappa = make_kappa(self.score)
        self.raw_weights = [None] * (self.k + 1)
        for a in range(2, self.k):
            self.raw_weights[a] = self.score + self.kappa[a]
        if self.k == 2:
            self.pair_merger = np.full((2, 2), float(np.mean(self.e)))
        else:
            pair_weight = self.raw_weights[2]
            numerator = np.outer(pair_weight * self.e, np.ones(self.k))
            numerator += numerator.T
            denominator = np.add.outer(pair_weight, pair_weight)
            self.pair_merger = numerator / denominator

    def merger(self, subset: tuple[int, ...]) -> float:
        a = len(subset)
        idx = np.fromiter(subset, dtype=int, count=a)
        if a == 1:
            return float(self.e[idx[0]])
        if a == self.k:
            return float(np.mean(self.e, dtype=np.longdouble))
        weights = self.raw_weights[a][idx].astype(np.longdouble)
        values = self.e[idx].astype(np.longdouble)
        return float(np.dot(weights, values) / np.sum(weights))

    @staticmethod
    def _smallest_sum(values: np.ndarray, count: int) -> float:
        if count <= 0:
            return 0.0
        if count >= values.size:
            return float(np.sum(values, dtype=np.longdouble))
        selected = np.argpartition(values, count - 1)[:count]
        return float(np.sum(values[selected], dtype=np.longdouble))

    @staticmethod
    def _smallest_indices(indices: np.ndarray,
                          values: np.ndarray,
                          count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=int)
        if count >= indices.size:
            return indices
        selected = np.argpartition(values, count - 1)[:count]
        return indices[selected]

    def _rowwise_order_statistics(self,
                                  inside: np.ndarray,
                                  outside: np.ndarray,
                                  a: int,
                                  r: int,
                                  need_orders: bool = False):
        """Evaluate every feasible intersection count for one subset size.

        The earlier implementation called ``np.argpartition`` separately for
        every value of q=|A intersection R|.  At K=400 those Python calls are
        the main runtime bottleneck.  Here the same transformed values are
        sorted row-wise in one NumPy call.  Each row still attains the exact
        minimum over subsets with the specified (a,q), so this changes only
        the implementation, not the separating inequalities.
        """
        q_lo = max(1, a - outside.size)
        q_hi = min(a, inside.size)
        q_values = np.arange(q_lo, q_hi + 1, dtype=int)
        weights = self.raw_weights[a]
        def smallest_rows(indices: np.ndarray, counts: np.ndarray):
            sums = np.zeros(q_values.size, dtype=np.longdouble)
            if indices.size == 0:
                orders = np.empty((q_values.size, 0), dtype=int)
                return sums, orders
            transformed = (
                weights[indices][None, :]
                * (self.alpha * r * self.e[indices][None, :] - q_values[:, None])
            )
            if need_orders:
                orders = np.argsort(transformed, axis=1, kind="stable")
                ordered = np.take_along_axis(transformed, orders, axis=1)
            else:
                orders = None
                ordered = np.sort(transformed, axis=1, kind="stable")
            cumulative = np.cumsum(ordered, axis=1, dtype=np.longdouble)
            positive = counts > 0
            rows = np.flatnonzero(positive)
            sums[positive] = cumulative[rows, counts[positive] - 1]
            return sums, orders

        inside_counts = q_values
        outside_counts = a - q_values
        inside_sums, inside_orders = smallest_rows(inside, inside_counts)
        outside_sums, outside_orders = smallest_rows(outside, outside_counts)
        return (
            q_values,
            inside_counts,
            outside_counts,
            inside_sums + outside_sums,
            inside_orders,
            outside_orders,
        )

    @staticmethod
    def _subset_from_orders(inside: np.ndarray,
                            outside: np.ndarray,
                            inside_order: np.ndarray,
                            outside_order: np.ndarray,
                            inside_count: int,
                            outside_count: int) -> tuple[int, ...]:
        chosen_in = inside[inside_order[:inside_count]]
        chosen_out = outside[outside_order[:outside_count]]
        return tuple(sorted(int(i) for i in np.concatenate((chosen_in, chosen_out))))

    def is_certified(self, selected: np.ndarray) -> bool:
        """Exact order-statistic check, stopping at the first violation."""
        inside = np.flatnonzero(selected)
        outside = np.flatnonzero(~selected)
        r = inside.size
        if r == 0:
            return True
        if float(np.mean(self.e, dtype=np.longdouble)) + ABS_TOL < 1.0 / self.alpha:
            return False
        if float(np.min(self.e[inside])) + ABS_TOL < 1.0 / (self.alpha * r):
            return False

        for a in range(2, self.k):
            _, _, _, totals, _, _ = self._rowwise_order_statistics(
                inside, outside, a, r
            )
            if np.any(totals < -ABS_TOL):
                return False
        return True

    def first_violation(self, selected: np.ndarray) -> tuple[tuple[int, ...], float] | None:
        """Return one violated subset, or None when the set is certified."""
        inside = np.flatnonzero(selected)
        outside = np.flatnonzero(~selected)
        r = inside.size
        if r == 0:
            return None
        global_mean = float(np.mean(self.e, dtype=np.longdouble))
        if global_mean + ABS_TOL < 1.0 / self.alpha:
            return tuple(range(self.k)), global_mean
        for i in inside:
            if self.e[i] + ABS_TOL < 1.0 / (self.alpha * r):
                return (int(i),), float(self.e[i])
        for a in range(2, self.k):
            q_values, in_counts, out_counts, totals, in_orders, out_orders = (
                self._rowwise_order_statistics(inside, outside, a, r, need_orders=True)
            )
            violated = np.flatnonzero(totals < -ABS_TOL)
            if violated.size:
                row = int(violated[0])
                subset = self._subset_from_orders(
                    inside, outside, in_orders[row], out_orders[row],
                    int(in_counts[row]), int(out_counts[row]),
                )
                return subset, self.merger(subset)
        return None

    def early_violations(self,
                         selected: np.ndarray,
                         limit: int) -> list[tuple[tuple[int, ...], float]]:
        """Collect the first few violations without completing a full scan."""
        inside = np.flatnonzero(selected)
        outside = np.flatnonzero(~selected)
        r = inside.size
        if r == 0:
            return []
        found: list[tuple[tuple[int, ...], float]] = []
        global_mean = float(np.mean(self.e, dtype=np.longdouble))
        if global_mean + ABS_TOL < 1.0 / self.alpha:
            return [(tuple(range(self.k)), global_mean)]
        for i in inside:
            if self.e[i] + ABS_TOL < 1.0 / (self.alpha * r):
                found.append(((int(i),), float(self.e[i])))
                if len(found) >= limit:
                    return found
        for a in range(2, self.k):
            _, in_counts, out_counts, totals, in_orders, out_orders = (
                self._rowwise_order_statistics(inside, outside, a, r, need_orders=True)
            )
            for row in np.flatnonzero(totals < -ABS_TOL):
                row = int(row)
                subset = self._subset_from_orders(
                    inside, outside, in_orders[row], out_orders[row],
                    int(in_counts[row]), int(out_counts[row]),
                )
                found.append((subset, self.merger(subset)))
                if len(found) >= limit:
                    return found
        return found

    def separate(self,
                 selected: np.ndarray,
                 max_cuts: int | None = None) -> list[tuple[float, tuple[int, ...], float]]:
        """Return violated cuts as (violation, A, E_A), strongest first."""
        if max_cuts is not None and max_cuts <= 0:
            raise ValueError("max_cuts must be positive when supplied")
        inside = np.flatnonzero(selected)
        outside = np.flatnonzero(~selected)
        r = inside.size
        if r == 0:
            return []
        violations: list[tuple[float, tuple[int, ...], float]] = []

        # Singleton violations are normally present from the initial master,
        # but retaining this check makes the oracle self-contained.
        for i in inside:
            violation = 1.0 - self.alpha * r * self.e[i]
            if violation > ABS_TOL:
                violations.append((float(violation), (int(i),), float(self.e[i])))

        # The full-set merger is the ordinary mean.
        global_violation = r - self.alpha * r * float(
            np.mean(self.e, dtype=np.longdouble)
        )
        if global_violation > ABS_TOL:
            full = tuple(range(self.k))
            violations.append((global_violation, full, float(np.mean(self.e))))

        for a in range(2, self.k):
            q_values, in_counts, out_counts, totals, in_orders, out_orders = (
                self._rowwise_order_statistics(inside, outside, a, r, need_orders=True)
            )
            for row in np.flatnonzero(totals < -ABS_TOL):
                row = int(row)
                subset = self._subset_from_orders(
                    inside, outside, in_orders[row], out_orders[row],
                    int(in_counts[row]), int(out_counts[row]),
                )
                merger = self.merger(subset)
                violation = int(q_values[row]) - self.alpha * r * merger
                if violation > ABS_TOL:
                    violations.append((float(violation), subset, merger))

        violations.sort(key=lambda item: item[0], reverse=True)
        if max_cuts is not None:
            return violations[:max_cuts]
        return violations


def deterministic_greedy(oracle: LearnedOracle,
                         initial: np.ndarray,
                         witness_limit: int) -> tuple[np.ndarray, int, dict[tuple[int, ...], float]]:
    """Make certified singleton additions, ordered by full-data evidence."""
    if witness_limit <= 0:
        raise ValueError("witness_limit must be positive")
    selected = initial.copy()
    log_utility = np.log(np.maximum(oracle.score, np.finfo(float).tiny))
    log_utility += np.log(np.maximum(oracle.e, np.finfo(float).tiny))
    order = np.lexsort((np.arange(oracle.k), -log_utility))
    added_total = 0
    witness_cuts: dict[tuple[int, ...], float] = {}

    while True:
        added_this_pass = 0
        for i in order:
            if selected[i]:
                continue
            # A necessary singleton condition avoids most oracle calls.
            next_r = int(np.sum(selected)) + 1
            if oracle.e[i] + ABS_TOL < 1.0 / (oracle.alpha * next_r):
                continue
            candidate = selected.copy()
            candidate[i] = True
            witnesses = oracle.early_violations(candidate, witness_limit)
            if not witnesses:
                selected = candidate
                added_total += 1
                added_this_pass += 1
            else:
                for subset, merger in witnesses:
                    witness_cuts[subset] = merger
        if added_this_pass == 0:
            return selected, added_total, witness_cuts


def singleton_upper_bound(e: np.ndarray, alpha: float, lower: int) -> int:
    upper = lower
    for r in range(max(1, lower), e.size + 1):
        if int(np.sum(e + ABS_TOL >= 1.0 / (alpha * r))) >= r:
            upper = r
    return upper


def conservative_capacity(a: int, alpha: float, r: np.ndarray | int, merger: float):
    value = alpha * np.asarray(r, dtype=float) * merger
    # The upward tolerance can only weaken a cut near an integer boundary; it
    # cannot exclude a mathematically feasible rejection set.
    capacity = np.floor(value + CAPACITY_TOL * np.maximum(1.0, np.abs(value)))
    return np.minimum(a, capacity).astype(int)


def layered_master(k: int,
                   alpha: float,
                   e: np.ndarray,
                   seed: np.ndarray,
                   r_values: np.ndarray,
                   cuts: dict[tuple[int, ...], float],
                   time_limit: float):
    nr = r_values.size
    nvar = k + nr
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    for j in range(nr):
        rows.append(row)
        cols.append(k + j)
        vals.append(1.0)
    lower.append(1.0)
    upper.append(1.0)
    row += 1

    for i in range(k):
        rows.append(row)
        cols.append(i)
        vals.append(1.0)
    for j, r in enumerate(r_values):
        rows.append(row)
        cols.append(k + j)
        vals.append(-float(r))
    lower.append(0.0)
    upper.append(0.0)
    row += 1

    for subset, merger in cuts.items():
        a = len(subset)
        capacity = conservative_capacity(a, alpha, r_values, merger)
        for i in subset:
            rows.append(row)
            cols.append(i)
            vals.append(1.0)
        for j in np.flatnonzero(capacity):
            rows.append(row)
            cols.append(k + int(j))
            vals.append(-float(capacity[j]))
        lower.append(-np.inf)
        upper.append(0.0)
        row += 1

    matrix = coo_matrix((vals, (rows, cols)), shape=(row, nvar)).tocsr()
    constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    lb = np.concatenate((seed.astype(float), np.zeros(nr)))
    ub = np.ones(nvar)
    largest_r = int(r_values[-1])
    ub[:k][e + ABS_TOL < 1.0 / (alpha * largest_r)] = 0.0
    objective = np.concatenate((np.zeros(k), -r_values.astype(float)))
    # SciPy forwards the absolute-gap, serial-search, and seed options to
    # HiGHS but emits a generic notice because they are not in its own schema.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        return milp(
            objective,
            integrality=np.ones(nvar),
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options={
                "time_limit": time_limit,
                "mip_rel_gap": 0.0,
                "mip_abs_gap": 0.0,
                "threads": 1,
                "random_seed": 0,
            },
        )


def fixed_cardinality_master(k: int,
                             alpha: float,
                             e: np.ndarray,
                             seed: np.ndarray,
                             target_r: int,
                             cuts: dict[tuple[int, ...], float],
                             pair_merger: np.ndarray,
                             time_limit: float):
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    for i in range(k):
        rows.append(row)
        cols.append(i)
        vals.append(1.0)
    lower.append(float(target_r))
    upper.append(float(target_r))
    row += 1

    lb = seed.astype(float)
    ub_variable = np.ones(k)
    ub_variable[e + ABS_TOL < 1.0 / (alpha * target_r)] = 0.0

    # Add the complete family of pair constraints.  They are sparse at fixed
    # cardinality and often turn most non-seed coordinates into a conflict
    # graph, substantially strengthening the master at negligible memory cost.
    for i in range(k):
        for j in range(i + 1, k):
            capacity = int(conservative_capacity(2, alpha, target_r, pair_merger[i, j]))
            if capacity >= 2:
                continue
            if capacity == 0:
                ub_variable[i] = 0.0
                ub_variable[j] = 0.0
            elif seed[i]:
                ub_variable[j] = 0.0
            elif seed[j]:
                ub_variable[i] = 0.0
            else:
                rows.extend((row, row))
                cols.extend((i, j))
                vals.extend((1.0, 1.0))
                lower.append(-np.inf)
                upper.append(1.0)
                row += 1

    for subset, merger in cuts.items():
        if len(subset) <= 2:
            continue
        a = len(subset)
        capacity = int(conservative_capacity(a, alpha, target_r, merger))
        if capacity >= a:
            continue
        for i in subset:
            rows.append(row)
            cols.append(i)
            vals.append(1.0)
        lower.append(-np.inf)
        upper.append(float(capacity))
        row += 1

    matrix = coo_matrix((vals, (rows, cols)), shape=(row, k)).tocsr()
    constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        return milp(
            np.zeros(k),
            integrality=np.ones(k),
            bounds=Bounds(lb, ub_variable),
            constraints=constraints,
            options={
                "time_limit": time_limit,
                "mip_rel_gap": 0.0,
                "mip_abs_gap": 0.0,
                "threads": 1,
                "random_seed": 0,
            },
        )


def load_highspy():
    """Import highspy, using the local wheel installation when present."""
    try:
        import highspy  # type: ignore
        return highspy
    except ImportError:
        vendor = Path(__file__).resolve().parents[1] / "vendor" / "highspy"
        if vendor.exists():
            sys.path.append(str(vendor))
            import highspy  # type: ignore
            return highspy
        return None


class PersistentFixedMaster:
    """A fixed-cardinality HiGHS model that accepts new cuts without restart."""
    def __init__(self,
                 k: int,
                 alpha: float,
                 e: np.ndarray,
                 seed: np.ndarray,
                 target_r: int,
                 cuts: dict[tuple[int, ...], float],
                 pair_merger: np.ndarray,
                 time_limit: float,
                 objective: np.ndarray | None = None):
        highspy = load_highspy()
        if highspy is None:
            raise RuntimeError("highspy>=1.15 is required for persistent fixed-r solves")
        self.highspy = highspy
        self.k = k
        self.alpha = alpha
        self.target_r = target_r
        self.time_limit = time_limit
        self.seed = seed.copy()
        if not (0 <= target_r <= k):
            raise ValueError("target_r must lie between zero and K")
        if int(np.sum(seed)) > target_r:
            raise ValueError("the fixed-cardinality target cannot be below the seed size")
        if objective is None:
            self.objective = None
        else:
            objective = np.asarray(objective, dtype=float)
            if objective.shape != (k,) or not np.all(np.isfinite(objective)):
                raise ValueError("objective must be a finite vector of length K")
            self.objective = objective.copy()
        self.model = highspy.Highs()
        self.model.setOptionValue("output_flag", False)
        self.model.setOptionValue("time_limit", time_limit)
        self.model.setOptionValue("mip_rel_gap", 0.0)
        self.model.setOptionValue("mip_abs_gap", 0.0)
        self.model.setOptionValue("threads", 1)
        self.model.setOptionValue("random_seed", 0)

        lower = seed.astype(float)
        upper = np.ones(k)
        if target_r > 0:
            upper[e + ABS_TOL < 1.0 / (alpha * target_r)] = 0.0
        pair_rows: list[tuple[tuple[int, ...], float]] = []

        for i in range(k):
            for j in range(i + 1, k):
                capacity = int(
                    conservative_capacity(2, alpha, target_r, pair_merger[i, j])
                )
                if capacity >= 2:
                    continue
                if capacity == 0:
                    upper[i] = 0.0
                    upper[j] = 0.0
                elif seed[i]:
                    upper[j] = 0.0
                elif seed[j]:
                    upper[i] = 0.0
                else:
                    pair_rows.append(((i, j), 1.0))

        self.model.addVars(k, lower, upper)
        indices = np.arange(k, dtype=np.int32)
        integrality = np.full(k, highspy.HighsVarType.kInteger, dtype=np.uint8)
        self.model.changeColsIntegrality(k, indices, integrality)
        if self.objective is not None:
            # HiGHS minimizes by default.  Negating the normalized testing
            # e-values is exactly equivalent to maximizing their sum.
            self.model.changeColsCost(k, indices, -self.objective)

        initial_rows: list[tuple[tuple[int, ...], float]] = [
            (tuple(range(k)), float(target_r)), *pair_rows
        ]
        for subset, merger in cuts.items():
            if len(subset) <= 2:
                continue
            capacity = int(
                conservative_capacity(len(subset), alpha, target_r, merger)
            )
            if capacity < len(subset):
                initial_rows.append((subset, float(capacity)))
        self._add_rows(initial_rows, equality_first=True)

    def _add_rows(self,
                  rows: list[tuple[tuple[int, ...], float]],
                  equality_first: bool = False) -> None:
        if not rows:
            return
        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        lower = np.full(len(rows), -self.highspy.kHighsInf)
        upper = np.empty(len(rows))
        for row_number, (subset, bound) in enumerate(rows):
            indices.extend(subset)
            values.extend([1.0] * len(subset))
            starts.append(len(indices))
            upper[row_number] = bound
        if equality_first:
            lower[0] = self.target_r
        self.model.addRows(
            len(rows),
            lower,
            upper,
            len(indices),
            np.asarray(starts, dtype=np.int32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(values, dtype=float),
        )

    def add_violations(self,
                       violations: list[tuple[float, tuple[int, ...], float]]) -> None:
        rows = []
        for _, subset, merger in violations:
            capacity = int(
                conservative_capacity(len(subset), self.alpha, self.target_r, merger)
            )
            if capacity < len(subset):
                rows.append((subset, float(capacity)))
        self._add_rows(rows)

    def solve(self) -> tuple[str, np.ndarray | None, str]:
        self.model.setOptionValue("time_limit", self.time_limit)
        self.model.run()
        status = self.model.getModelStatus()
        if status == self.highspy.HighsModelStatus.kInfeasible:
            return "infeasible", None, str(status)
        solution = self.model.getSolution()
        info = self.model.getInfo()
        tolerance = 1e-7
        if solution.value_valid:
            values = np.asarray(solution.col_value, dtype=float)
            candidate = values > 0.5
            near_binary = np.all(np.minimum(np.abs(values), np.abs(values - 1.0)) <= tolerance)
            feasible_status = (
                info.primal_solution_status == self.highspy.kSolutionStatusFeasible
            )
            numerical_feasible = (
                np.isfinite(info.max_primal_infeasibility)
                and np.isfinite(info.max_integrality_violation)
                and info.max_primal_infeasibility <= tolerance
                and info.max_integrality_violation <= tolerance
            )
            structurally_feasible = (
                int(np.sum(candidate)) == self.target_r
                and bool(np.all(candidate[self.seed]))
            )
            if near_binary and feasible_status and numerical_feasible and structurally_feasible:
                if self.objective is None:
                    # A feasible witness suffices for the cardinality search,
                    # even if the feasibility solve used its whole time limit.
                    return "candidate", candidate, str(status)
                if status == self.highspy.HighsModelStatus.kOptimal:
                    return "optimal", candidate, str(status)
                # A feasible incumbent is not enough for the secondary rule:
                # its objective must be globally proved optimal.
                return "unproven", None, str(status)
        return "timeout", None, str(status)


def summarize_set(selected: np.ndarray, nonnull: np.ndarray) -> tuple[int, int, float, float]:
    size = int(np.sum(selected))
    true = int(np.sum(nonnull[selected]))
    false = size - true
    alternatives = int(np.sum(nonnull))
    tpr = true / alternatives
    fdp = false / size if size else 0.0
    return true, false, tpr, fdp


def selected_indices(selected: np.ndarray) -> str:
    """Serialize a membership vector compactly and deterministically."""
    return ";".join(str(int(i)) for i in np.flatnonzero(selected))


def problem_sha256(instance: Instance, alpha: float) -> str:
    """Bind an optimizer result to one canonical numerical problem instance."""
    digest = hashlib.sha256()
    digest.update(b"closed-ebh-maximal-instance-v1\0")
    digest.update(np.asarray([instance.replication], dtype="<i8").tobytes())
    digest.update(
        np.asarray(
            [alpha, ABS_TOL, CAPACITY_TOL, float(instance.e.size)], dtype="<f8"
        ).tobytes()
    )
    digest.update(np.asarray(instance.e, dtype="<f8").tobytes())
    digest.update(np.asarray(instance.score, dtype="<f8").tobytes())
    digest.update(np.asarray(instance.nonnull, dtype=np.uint8).tobytes())
    digest.update(np.asarray(instance.prefix, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def optimize_instance(instance: Instance,
                      alpha: float,
                      greedy_witness_limit: int,
                      layer_batch: int,
                      fixed_batch: int,
                      layer_window: int,
                      master_time_limit: float) -> OptimizationResult:
    started = time.perf_counter()
    trace_enabled = os.environ.get("CHOICE2_TRACE", "0") == "1"
    trace_file = os.environ.get("CHOICE2_TRACE_FILE")
    def trace(message: str) -> None:
        if trace_enabled:
            rendered = f"[rep {instance.replication}] {message}"
            print(rendered, flush=True)
            if trace_file:
                with Path(trace_file).open("a", encoding="utf-8") as handle:
                    handle.write(rendered + "\n")
    oracle = LearnedOracle(instance.e, instance.score, alpha)
    prefix = instance.prefix.copy()
    if not oracle.is_certified(prefix):
        raise RuntimeError(f"replication {instance.replication}: exported prefix is not certified")

    greedy, greedy_added, greedy_cuts = deterministic_greedy(
        oracle, prefix, greedy_witness_limit
    )
    if not oracle.is_certified(greedy):
        raise RuntimeError(f"replication {instance.replication}: greedy set lost certification")

    prefix_size = int(np.sum(prefix))
    greedy_size = int(np.sum(greedy))
    upper_bound = singleton_upper_bound(instance.e, alpha, greedy_size)
    trace(f"seed={prefix_size}, greedy={greedy_size}, singleton_upper={upper_bound}")
    initial_singleton_upper_bound = upper_bound
    cuts: dict[tuple[int, ...], float] = {
        (i,): float(instance.e[i]) for i in range(instance.e.size)
    }
    cuts.update(greedy_cuts)
    mip_solves = 0
    separation_calls = 0
    message = ""
    secondary_rule = "maximize_sum_testing_e_values_at_maximum_cardinality"
    secondary_status = "not_run_cardinality_unproven"
    secondary_value = math.nan
    secondary_normalized_value = math.nan
    secondary_normalization = float(np.max(instance.e))
    if secondary_normalization <= 0.0:
        secondary_normalization = 1.0

    maximal = greedy.copy()
    status = "proven_maximal"

    if upper_bound > greedy_size:
        r_values = np.arange(greedy_size, upper_bound + 1, dtype=int)
        current_upper = upper_bound

        # Quickly collapse the weak singleton upper bound using cardinality-
        # layered lazy cuts before solving individual cardinality levels.
        for layer_iteration in range(12):
            trace(f"layer {layer_iteration + 1}: r<= {int(r_values[-1])}, cuts={len(cuts)}")
            solution = layered_master(
                instance.e.size,
                alpha,
                instance.e,
                greedy,
                r_values,
                cuts,
                master_time_limit,
            )
            mip_solves += 1
            if solution.status != 0 or solution.x is None:
                status = "timeout"
                message = f"layered master: {solution.message}"
                break
            candidate = solution.x[: instance.e.size] > 0.5
            current_upper = int(np.sum(candidate))
            trace(f"layer {layer_iteration + 1}: candidate r={current_upper}")
            violations = oracle.separate(candidate, max_cuts=layer_batch)
            separation_calls += 1
            trace(f"layer {layer_iteration + 1}: violations={len(violations)}")
            if not violations:
                maximal = candidate
                upper_bound = current_upper
                break
            for _, subset, merger in violations:
                cuts[subset] = merger
            if current_upper <= greedy_size + layer_window:
                upper_bound = current_upper
                break
        else:
            upper_bound = current_upper

        if status != "timeout" and int(np.sum(maximal)) == greedy_size:
            found = False
            # A relaxation-certified upper bound is searched downward.  At a
            # fixed cardinality the cuts become sparse integer capacities.
            for target_r in range(upper_bound, greedy_size, -1):
                trace(f"fixed r={target_r}: build with cuts={len(cuts)}")
                master = PersistentFixedMaster(
                    instance.e.size,
                    alpha,
                    instance.e,
                    greedy,
                    target_r,
                    cuts,
                    oracle.pair_merger,
                    master_time_limit,
                )
                while True:
                    solve_status, candidate, solve_message = master.solve()
                    mip_solves += 1
                    trace(f"fixed r={target_r}: master={solve_status}")
                    if solve_status == "infeasible":
                        break
                    if solve_status == "timeout" or candidate is None:
                        status = "timeout"
                        message = f"fixed-r={target_r} master: {solve_message}"
                        break
                    violations = oracle.separate(candidate, max_cuts=fixed_batch)
                    separation_calls += 1
                    trace(f"fixed r={target_r}: violations={len(violations)}")
                    if not violations:
                        maximal = candidate
                        found = True
                        break
                    for _, subset, merger in violations:
                        cuts[subset] = merger
                    master.add_violations(violations)
                if status == "timeout" or found:
                    break

    cardinality_proven = status == "proven_maximal"
    if cardinality_proven and not oracle.is_certified(maximal):
        raise RuntimeError(f"replication {instance.replication}: final set is not certified")

    if cardinality_proven:
        # The first-stage seed remains fixed.  Thus Method 1 retains its empty
        # seed and Method 2 retains ordinary eBH when their runners replace the
        # legacy greedy step with the identity operation.
        target_r = int(np.sum(maximal))
        normalized_e = instance.e / secondary_normalization
        trace(
            f"secondary r={target_r}: maximize testing e-value sum "
            f"with cuts={len(cuts)}"
        )
        secondary_master = PersistentFixedMaster(
            instance.e.size,
            alpha,
            instance.e,
            greedy,
            target_r,
            cuts,
            oracle.pair_merger,
            master_time_limit,
            objective=normalized_e,
        )
        while True:
            solve_status, candidate, solve_message = secondary_master.solve()
            mip_solves += 1
            trace(f"secondary r={target_r}: master={solve_status}")
            if solve_status == "infeasible":
                # ``maximal`` is a certified feasible point for this same
                # fixed-cardinality model.  Infeasibility therefore signals a
                # model inconsistency and must never produce a reported set.
                raise RuntimeError(
                    f"replication {instance.replication}: secondary master "
                    "is infeasible despite a certified cardinality witness"
                )
            if solve_status != "optimal" or candidate is None:
                status = "timeout"
                secondary_status = "objective_unproven"
                message = (
                    f"secondary fixed-r={target_r} master did not prove the "
                    f"objective: {solve_message}"
                )
                break
            # Add every violation returned by the exact oracle.  A bounded
            # batch affects runtime only: acceptance still requires a complete
            # separation call returning no violation.
            violations = oracle.separate(candidate, max_cuts=fixed_batch)
            separation_calls += 1
            trace(f"secondary r={target_r}: violations={len(violations)}")
            if not violations:
                maximal = candidate
                secondary_status = "proven_optimal"
                secondary_value = float(
                    np.sum(instance.e[maximal], dtype=np.longdouble)
                )
                secondary_normalized_value = (
                    secondary_value / secondary_normalization
                )
                break
            for _, subset, merger in violations:
                cuts[subset] = merger
            secondary_master.add_violations(violations)

    if status == "proven_maximal" and not oracle.is_certified(maximal):
        raise RuntimeError(f"replication {instance.replication}: final set is not certified")

    prefix_true, prefix_false, prefix_tpr, prefix_fdp = summarize_set(
        prefix, instance.nonnull
    )
    greedy_true, greedy_false, greedy_tpr, greedy_fdp = summarize_set(
        greedy, instance.nonnull
    )
    maximal_true, maximal_false, maximal_tpr, maximal_fdp = summarize_set(
        maximal, instance.nonnull
    )
    return OptimizationResult(
        replication=instance.replication,
        problem_sha256=problem_sha256(instance, alpha),
        status=status,
        secondary_rule=secondary_rule,
        secondary_status=secondary_status,
        secondary_value=secondary_value,
        secondary_normalized_value=secondary_normalized_value,
        secondary_normalization=secondary_normalization,
        prefix_size=prefix_size,
        greedy_size=greedy_size,
        maximal_size=int(np.sum(maximal)),
        prefix_indices=selected_indices(prefix),
        greedy_indices=selected_indices(greedy),
        maximal_indices=selected_indices(maximal),
        prefix_true=prefix_true,
        greedy_true=greedy_true,
        maximal_true=maximal_true,
        prefix_false=prefix_false,
        greedy_false=greedy_false,
        maximal_false=maximal_false,
        prefix_tpr=prefix_tpr,
        greedy_tpr=greedy_tpr,
        maximal_tpr=maximal_tpr,
        prefix_fdp=prefix_fdp,
        greedy_fdp=greedy_fdp,
        maximal_fdp=maximal_fdp,
        greedy_added=greedy_added,
        optimizer_added=int(np.sum(maximal)) - greedy_size,
        elapsed_seconds=time.perf_counter() - started,
        mip_solves=mip_solves,
        separation_calls=separation_calls,
        final_cut_count=len(cuts),
        initial_singleton_upper_bound=initial_singleton_upper_bound,
        proof_upper_bound=(int(np.sum(maximal)) if cardinality_proven else upper_bound),
        message=message,
    )


def read_instances(path: Path,
                   limit: int | None = None,
                   replication_filter: int | None = None,
                   replication_start: int | None = None,
                   replication_end: int | None = None) -> list[Instance]:
    frame = pd.read_csv(path)
    required = {
        "replication", "index", "e", "score", "nonnull", "prefix_member"
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"missing columns in {path}: {sorted(missing)}")
    instances = []
    for replication, group in frame.groupby("replication", sort=True):
        replication = int(replication)
        if replication_filter is not None and replication != replication_filter:
            continue
        if replication_start is not None and replication < replication_start:
            continue
        if replication_end is not None and replication > replication_end:
            continue
        group = group.sort_values("index")
        indices = group["index"].to_numpy(dtype=int)
        if not np.array_equal(indices, np.arange(indices.size)):
            raise RuntimeError(
                f"replication {replication}: indices must be exactly 0,...,K-1"
            )
        instances.append(
            Instance(
                replication=replication,
                e=group["e"].to_numpy(dtype=float),
                score=group["score"].to_numpy(dtype=float),
                nonnull=group["nonnull"].to_numpy(dtype=bool),
                prefix=group["prefix_member"].to_numpy(dtype=bool),
            )
        )
        if limit is not None and len(instances) >= limit:
            break
    return instances


def write_results(path: Path, results: list[OptimizationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(results[0]).keys()) if results else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.replication):
            writer.writerow(asdict(result))
    temporary.replace(path)


def write_memberships(path: Path,
                      instances: list[Instance],
                      results: list[OptimizationResult]) -> None:
    """Write long-form membership flags so every reported set can be replayed."""
    result_by_replication = {result.replication: result for result in results}
    with path.open("w", newline="") as handle:
        fields = [
            "replication", "index", "prefix_member", "greedy_member", "maximal_member"
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for instance in sorted(instances, key=lambda item: item.replication):
            result = result_by_replication[instance.replication]
            prefix = {int(value) for value in result.prefix_indices.split(";") if value}
            greedy = {int(value) for value in result.greedy_indices.split(";") if value}
            maximal = {int(value) for value in result.maximal_indices.split(";") if value}
            for index in range(instance.e.size):
                writer.writerow({
                    "replication": instance.replication,
                    "index": index,
                    "prefix_member": int(index in prefix),
                    "greedy_member": int(index in greedy),
                    "maximal_member": int(index in maximal),
                })


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def highspy_version() -> str:
    module = load_highspy()
    if module is None:
        return "not installed"
    major = getattr(module, "HIGHS_VERSION_MAJOR", None)
    minor = getattr(module, "HIGHS_VERSION_MINOR", None)
    patch = getattr(module, "HIGHS_VERSION_PATCH", None)
    if None not in (major, minor, patch):
        return f"{major}.{minor}.{patch}"
    return str(module.Highs().version())


def write_optimizer_config(path: Path,
                           args: argparse.Namespace,
                           input_path: Path,
                           results: list[OptimizationResult],
                           wall_seconds: float) -> None:
    config = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "replications_written": len(results),
        "command": sys.argv,
        "parameters": {
            "alpha": args.alpha,
            "limit": args.limit,
            "replication": args.replication,
            "replication_start": args.replication_start,
            "replication_end": args.replication_end,
            "workers": args.workers,
            "greedy_witness_limit": args.greedy_witness_limit,
            "layer_batch": args.layer_batch,
            "fixed_batch": args.fixed_batch,
            "layer_window": args.layer_window,
            "master_time_limit": args.master_time_limit,
            "checkpoint_every": args.checkpoint_every,
            "progress_every": args.progress_every,
        },
        "floating_tolerances": {
            "absolute_separation": ABS_TOL,
            "capacity_rounding": CAPACITY_TOL,
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": installed_version("scipy"),
            "highspy": highspy_version(),
        },
        "wall_seconds": wall_seconds,
    }
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def mean_mcse(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    mcse = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, mcse


def write_summary(path: Path, results: list[OptimizationResult]) -> None:
    complete = [result for result in results if result.status == "proven_maximal"]
    summary: dict[str, object] = {
        "replications_requested": len(results),
        "replications_proven_maximal": len(complete),
        "replications_incomplete": len(results) - len(complete),
    }
    if complete:
        for prefix in ("prefix", "greedy", "maximal"):
            for metric in ("tpr", "fdp", "size"):
                values = np.asarray(
                    [getattr(result, f"{prefix}_{metric}") for result in complete],
                    dtype=float,
                )
                mean, mcse = mean_mcse(values)
                summary[f"{prefix}_{metric}_mean"] = mean
                summary[f"{prefix}_{metric}_mcse"] = mcse
        gain = np.asarray(
            [result.maximal_tpr - result.prefix_tpr for result in complete], dtype=float
        )
        size_gain = np.asarray(
            [result.maximal_size - result.prefix_size for result in complete], dtype=float
        )
        summary["paired_tpr_gain_mean"], summary["paired_tpr_gain_mcse"] = mean_mcse(gain)
        summary["paired_size_gain_mean"], summary["paired_size_gain_mcse"] = mean_mcse(size_gain)
        summary["fraction_strict_tpr_improvement"] = float(np.mean(gain > 0))
        summary["fraction_strict_size_improvement"] = float(np.mean(size_gain > 0))
        elapsed = np.asarray([result.elapsed_seconds for result in complete])
        summary["optimization_seconds_mean"] = float(np.mean(elapsed))
        summary["optimization_seconds_median"] = float(np.median(elapsed))
        summary["optimization_seconds_max"] = float(np.max(elapsed))
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--replication", type=int, default=None)
    parser.add_argument("--replication-start", type=int, default=None)
    parser.add_argument("--replication-end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--greedy-witness-limit", type=int, default=20)
    parser.add_argument("--layer-batch", type=int, default=1000)
    parser.add_argument("--fixed-batch", type=int, default=10000)
    parser.add_argument("--layer-window", type=int, default=12)
    parser.add_argument("--master-time-limit", type=float, default=60.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < args.alpha < 1.0):
        raise RuntimeError("alpha must lie in (0,1)")
    for name in (
        "workers", "greedy_witness_limit", "layer_batch", "fixed_batch",
        "checkpoint_every",
    ):
        if getattr(args, name) <= 0:
            raise RuntimeError(f"{name.replace('_', '-')} must be positive")
    if args.layer_window < 0:
        raise RuntimeError("layer-window must be nonnegative")
    if args.master_time_limit <= 0.0:
        raise RuntimeError("master-time-limit must be positive")
    if args.progress_every < 0:
        raise RuntimeError("progress-every must be nonnegative")
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("limit must be positive when supplied")
    if args.replication is not None and args.replication <= 0:
        raise RuntimeError("replication must be positive when supplied")
    if args.replication_start is not None and args.replication_start <= 0:
        raise RuntimeError("replication-start must be positive when supplied")
    if args.replication_end is not None and args.replication_end <= 0:
        raise RuntimeError("replication-end must be positive when supplied")
    if (args.replication_start is not None and args.replication_end is not None
            and args.replication_start > args.replication_end):
        raise RuntimeError("replication-start cannot exceed replication-end")
    if args.replication is not None and (
        args.replication_start is not None or args.replication_end is not None
    ):
        raise RuntimeError("replication cannot be combined with a replication range")
    instances = read_instances(
        args.input,
        args.limit,
        args.replication,
        args.replication_start,
        args.replication_end,
    )
    if not instances:
        raise RuntimeError("no instances found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "maximal_learned_results.csv"
    summary_path = args.output_dir / "maximal_learned_summary.json"

    kwargs = {
        "alpha": args.alpha,
        "greedy_witness_limit": args.greedy_witness_limit,
        "layer_batch": args.layer_batch,
        "fixed_batch": args.fixed_batch,
        "layer_window": args.layer_window,
        "master_time_limit": args.master_time_limit,
    }
    results: list[OptimizationResult] = []
    started = time.perf_counter()

    if args.workers == 1:
        for number, instance in enumerate(instances, start=1):
            result = optimize_instance(instance, **kwargs)
            results.append(result)
            if number % args.checkpoint_every == 0:
                write_results(results_path, results)
            if (
                args.progress_every > 0
                and (
                    number % args.progress_every == 0
                    or number == len(instances)
                    or result.status != "proven_maximal"
                )
            ):
                print(
                    f"completed {number}/{len(instances)}: rep={result.replication}, "
                    f"{result.prefix_size}->{result.maximal_size}, "
                    f"status={result.status}, seconds={result.elapsed_seconds:.2f}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(optimize_instance, instance, **kwargs): instance.replication
                for instance in instances
            }
            for number, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if number % args.checkpoint_every == 0:
                    write_results(results_path, results)
                if (
                    args.progress_every > 0
                    and (
                        number % args.progress_every == 0
                        or number == len(instances)
                        or result.status != "proven_maximal"
                    )
                ):
                    print(
                        f"completed {number}/{len(instances)}: rep={result.replication}, "
                        f"{result.prefix_size}->{result.maximal_size}, "
                        f"status={result.status}, seconds={result.elapsed_seconds:.2f}",
                        flush=True,
                    )

    write_results(results_path, results)
    write_summary(summary_path, results)
    wall_seconds = time.perf_counter() - started
    memberships_path = args.output_dir / "maximal_learned_memberships.csv"
    config_path = args.output_dir / "optimizer_run_config.json"
    write_memberships(memberships_path, instances, results)
    write_optimizer_config(config_path, args, args.input, results, wall_seconds)
    print(
        f"optimization complete in {wall_seconds:.2f}s\n"
        f"results: {results_path}\nsummary: {summary_path}\n"
        f"memberships: {memberships_path}\nconfig: {config_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # pragma: no cover - command-line boundary
        print(f"error: {error}", file=sys.stderr)
        raise

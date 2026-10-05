#!/usr/bin/env python3
"""Exact hybrid separator for the fixed-floor residual score formula.

For an intersection A, every member receives the baseline weight 1/K.  The
remaining mass, (K-|A|)/K, is allocated in proportion to its learned score.
For small reported sets, an outside-prefix theorem reduces separation to
exact deterministic integer enumeration. Larger sets use the established
McCormick-linearized HiGHS MILP. A candidate is certified only by complete
prefix enumeration or a globally optimal MILP; time limits remain unknown.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from choice2_oracle import (
    ABS_TOL,
    RAW_OPT_TOL,
    SeparationUnknown,
    SeparatorDiagnostics,
    ordinary_ebh,
)


PREFIX_ENUMERATION_CAP = 16
# Cover the largest published adjusted-normalized rejection set (size 23).
GRAY_PREFIX_ENUMERATION_CAP = 23
PREFIX_FALLBACK = object()


def original_formula_merger(
        e: np.ndarray,
        score: np.ndarray,
        subset: tuple[int, ...] | np.ndarray) -> float:
    """Evaluate 1/K + proportional-prior residual weights on one subset."""
    values = np.asarray(e, dtype=float)
    scores = np.asarray(score, dtype=float)
    idx = np.asarray(subset, dtype=int)
    if values.ndim != 1 or scores.shape != values.shape:
        raise ValueError("e and score must be equal-length vectors")
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("subset must be nonempty and one-dimensional")
    local_score = scores[idx].astype(np.longdouble)
    local_e = values[idx].astype(np.longdouble)
    total_score = np.sum(local_score, dtype=np.longdouble)
    if not np.isfinite(total_score) or total_score <= 0.0:
        raise ValueError("scores must be finite and strictly positive")
    k = values.size
    size = idx.size
    uniform_part = np.sum(local_e, dtype=np.longdouble) / np.longdouble(k)
    prior_mean = np.dot(local_score, local_e) / total_score
    prior_part = np.longdouble(k - size) * prior_mean / np.longdouble(k)
    return float(uniform_part + prior_part)


class OriginalFormulaOracle:
    """Exact closed-eBH oracle for the user's original score weights."""

    def __init__(self,
                 e: np.ndarray,
                 score: np.ndarray,
                 alpha: float,
                 separator_time_limit: float = 300.0,
                 separator_threads: int = 1):
        self.e = np.asarray(e, dtype=float)
        original_score = np.asarray(score, dtype=float)
        self.alpha = float(alpha)
        self.separator_time_limit = float(separator_time_limit)
        self.separator_threads = int(separator_threads)
        if self.e.ndim != 1 or original_score.ndim != 1:
            raise ValueError("e and score must be one-dimensional")
        if self.e.shape != original_score.shape or self.e.size < 2:
            raise ValueError("e and score must have equal length at least two")
        if not np.all(np.isfinite(self.e)) or np.any(self.e < 0.0):
            raise ValueError("e-values must be finite and nonnegative")
        if (not np.all(np.isfinite(original_score))
                or np.any(original_score <= 0.0)):
            raise ValueError("scores must be finite and strictly positive")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0,1)")
        if (not math.isfinite(self.separator_time_limit)
                or self.separator_time_limit <= 0.0):
            raise ValueError("separator_time_limit must be positive and finite")
        if self.separator_threads <= 0:
            raise ValueError("separator_threads must be positive")

        self.k = self.e.size
        # The formula is scale invariant.  Mean-one normalization keeps the
        # quadratic coefficients at a useful numerical scale.
        mean_score = np.mean(original_score, dtype=np.longdouble)
        self.score = np.asarray(original_score / float(mean_score), dtype=float)
        self.ebh_seed = ordinary_ebh(self.e, self.alpha)
        self._certified_cache: set[bytes] = {self.ebh_seed.tobytes()}
        self.surrogate_oracles: list[object] = []
        self.last_diagnostics: SeparatorDiagnostics | None = None
        self.pair_merger = self._make_pair_mergers()
        self._build_qubo_linearization()

    def _build_qubo_linearization(self) -> None:
        """Build the R-independent McCormick matrix used by every separation.

        The first K variables are the binary intersection indicators.  One
        continuous variable z_ij represents x_i*x_j for every unordered pair.
        Since x is binary, the three McCormick inequalities enforce the
        product exactly while leaving every z variable continuous.
        """
        k = self.k
        pair_i, pair_j = np.triu_indices(k, k=1)
        self._qubo_pair_i = pair_i.astype(np.int32, copy=False)
        self._qubo_pair_j = pair_j.astype(np.int32, copy=False)
        pair_count = int(pair_i.size)
        variable_count = k + pair_count

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        upper = np.empty(3 * pair_count, dtype=float)
        row = 0
        for pair_number, (i, j) in enumerate(zip(pair_i, pair_j)):
            product = k + pair_number

            # z_ij <= x_i
            rows.extend((row, row))
            columns.extend((product, int(i)))
            values.extend((1.0, -1.0))
            upper[row] = 0.0
            row += 1

            # z_ij <= x_j
            rows.extend((row, row))
            columns.extend((product, int(j)))
            values.extend((1.0, -1.0))
            upper[row] = 0.0
            row += 1

            # z_ij >= x_i + x_j - 1
            rows.extend((row, row, row))
            columns.extend((product, int(i), int(j)))
            values.extend((-1.0, 1.0, 1.0))
            upper[row] = 1.0
            row += 1

        matrix = coo_matrix(
            (values, (rows, columns)),
            shape=(row, variable_count),
        ).tocsr()
        self._qubo_constraints = LinearConstraint(
            matrix,
            np.full(row, -np.inf, dtype=float),
            upper,
        )
        self._qubo_bounds = Bounds(
            np.zeros(variable_count, dtype=float),
            np.ones(variable_count, dtype=float),
        )
        self._qubo_integrality = np.concatenate((
            np.ones(k, dtype=np.uint8),
            np.zeros(pair_count, dtype=np.uint8),
        ))

    def _qubo_costs(self, selected: np.ndarray) -> np.ndarray:
        """Return the exact linearized costs for one proposed rejection set.

        Let x encode an intersection and d encode ``selected``.  Writing
        S=sum(s_i*x_i), E=sum(e_i*x_i), T=sum(s_i*e_i*x_i), a=sum(x_i),
        q=sum(d_i*x_i), and r=sum(d_i), a violation larger than ABS_TOL is
        equivalent to

          alpha*r*(S*E + (K-a)*T) - K*(q-ABS_TOL)*S < 0.

        The returned unary and pair costs are the exact expansion of the
        left-hand side.  The empty intersection has cost zero, so a proved
        optimum of zero certifies the proposed rejection set.
        """
        chosen = np.asarray(selected, dtype=bool)
        if chosen.shape != (self.k,):
            raise ValueError("selected has the wrong shape")
        r = int(np.sum(chosen))
        membership = chosen.astype(float)
        beta = self.alpha * r
        unary = self.k * self.score * (
            beta * self.e - membership + ABS_TOL
        )
        i = self._qubo_pair_i
        j = self._qubo_pair_j
        pair = (
            beta * (self.score[i] - self.score[j]) * (self.e[j] - self.e[i])
            - self.k * (
                membership[i] * self.score[j]
                + membership[j] * self.score[i]
            )
        )
        return np.concatenate((unary, pair))

    def _make_pair_mergers(self) -> np.ndarray:
        answer = np.empty((self.k, self.k), dtype=float)
        np.fill_diagonal(answer, self.e)
        for i in range(self.k):
            for j in range(i + 1, self.k):
                value = original_formula_merger(
                    self.e, self.score, (i, j)
                )
                answer[i, j] = value
                answer[j, i] = value
        return answer

    def merger(self, subset: tuple[int, ...]) -> float:
        return original_formula_merger(self.e, self.score, subset)

    def _quick_violations(
            self,
            selected: np.ndarray,
            limit: int) -> list[tuple[float, tuple[int, ...], float]]:
        inside = np.flatnonzero(selected)
        r = int(inside.size)
        if r == 0:
            return []
        found: list[tuple[float, tuple[int, ...], float]] = []

        singleton_violation = 1.0 - self.alpha * r * self.e[inside]
        for row in np.flatnonzero(singleton_violation > ABS_TOL):
            i = int(inside[int(row)])
            found.append(
                (float(singleton_violation[row]), (i,), float(self.e[i]))
            )

        global_merger = float(np.mean(self.e, dtype=np.longdouble))
        global_violation = r - self.alpha * r * global_merger
        if global_violation > ABS_TOL:
            found.append(
                (float(global_violation), tuple(range(self.k)), global_merger)
            )

        if self.k > 2:
            membership = selected.astype(np.int8)
            count = membership[:, None] + membership[None, :]
            violation = count - self.alpha * r * self.pair_merger
            violation[count == 0] = -np.inf
            violation[np.tril_indices(self.k)] = -np.inf
            candidates = np.flatnonzero(violation.ravel() > ABS_TOL)
            if candidates.size > limit:
                local = violation.ravel()[candidates]
                keep = np.argpartition(local, -limit)[-limit:]
                candidates = candidates[keep]
            for flat in candidates:
                i, j = np.unravel_index(int(flat), violation.shape)
                found.append((
                    float(violation[i, j]),
                    (int(i), int(j)),
                    float(self.pair_merger[i, j]),
                ))
        found.sort(key=lambda item: item[0], reverse=True)
        return found[:limit]

    def _surrogate_violations(
            self,
            selected: np.ndarray,
            limit: int) -> list[tuple[float, tuple[int, ...], float]]:
        if not self.surrogate_oracles:
            return []
        r = int(np.sum(selected))
        found: dict[tuple[int, ...], tuple[float, tuple[int, ...], float]] = {}
        candidate_limit = max(500, 10 * limit)
        for oracle in self.surrogate_oracles:
            for _, subset, _ in oracle.separate(
                    selected, max_cuts=candidate_limit):
                merger = self.merger(subset)
                count = int(np.sum(selected[np.asarray(subset, dtype=int)]))
                violation = count - self.alpha * r * merger
                if violation > ABS_TOL:
                    found[subset] = (float(violation), subset, merger)
        answer = sorted(found.values(), key=lambda item: item[0], reverse=True)
        return answer[:limit]

    def _exact_violation_highs(
            self,
            selected: np.ndarray
    ) -> tuple[float, tuple[int, ...], float] | None:
        """Find a violating intersection or prove that none exists.

        HiGHS may return an incumbent before proving optimality.  Any
        numerically integral incumbent is rechecked directly in long-double
        arithmetic and may safely be returned when it is a genuine witness.
        In contrast, absence of a witness is accepted only with a globally
        optimal solver status and a nonnegative QUBO optimum.
        """
        chosen = np.asarray(selected, dtype=bool)
        inside = np.flatnonzero(chosen)
        r = int(inside.size)
        if r == 0:
            return None
        raw_cost = self._qubo_costs(chosen)
        objective_scale = max(1.0, float(np.max(np.abs(raw_cost))))
        # SciPy exposes the relative MIP gap directly and forwards other
        # HiGHS options verbatim.  The forwarded absolute-gap, serial-search,
        # and seed options are intentional; suppress only SciPy's generic
        # notice about that forwarding.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected:.*",
                category=RuntimeWarning,
            )
            result = milp(
                raw_cost / objective_scale,
                integrality=self._qubo_integrality,
                bounds=self._qubo_bounds,
                constraints=self._qubo_constraints,
                options={
                    "time_limit": self.separator_time_limit,
                    "mip_rel_gap": 0.0,
                    "mip_abs_gap": 0.0,
                    "presolve": True,
                    # The SciPy wrapper cannot safely reset HiGHS' global
                    # thread scheduler between solves.  Disabling parallel
                    # search gives deterministic one-stream behavior.
                    "parallel": False,
                    "random_seed": 0,
                },
            )

        objective_value = (
            float(result.fun) * objective_scale
            if result.fun is not None and np.isfinite(result.fun)
            else math.nan
        )
        dual_bound_scaled = getattr(result, "mip_dual_bound", math.nan)
        dual_bound = (
            float(dual_bound_scaled) * objective_scale
            if dual_bound_scaled is not None and np.isfinite(dual_bound_scaled)
            else math.nan
        )
        nodes = getattr(result, "mip_node_count", 0)
        self.last_diagnostics = SeparatorDiagnostics(
            status=int(result.status),
            message=f"HiGHS {result.message}",
            objective=objective_value,
            dual_bound=dual_bound,
            elapsed_nodes=int(nodes) if nodes is not None else 0,
        )

        if result.x is not None:
            indicator_values = np.asarray(result.x[:self.k], dtype=float)
            near_binary = np.all(
                np.minimum(
                    np.abs(indicator_values),
                    np.abs(indicator_values - 1.0),
                ) <= 1e-7
            )
            if near_binary:
                candidate = indicator_values > 0.5
                idx = np.flatnonzero(candidate)
                if idx.size:
                    local_score = self.score[idx].astype(np.longdouble)
                    local_e = self.e[idx].astype(np.longdouble)
                    score_sum = np.sum(local_score, dtype=np.longdouble)
                    weighted_mean = (
                        np.dot(local_score, local_e) / score_sum
                    )
                    size = np.longdouble(idx.size)
                    merger_ld = (
                        np.sum(local_e, dtype=np.longdouble)
                        / np.longdouble(self.k)
                        + (np.longdouble(self.k) - size)
                        / np.longdouble(self.k)
                        * weighted_mean
                    )
                    actual_count = int(np.sum(chosen[idx]))
                    violation_ld = (
                        np.longdouble(actual_count)
                        - np.longdouble(self.alpha)
                        * np.longdouble(r)
                        * merger_ld
                    )
                    if violation_ld > np.longdouble(ABS_TOL):
                        subset = tuple(int(i) for i in idx)
                        return (
                            float(violation_ld),
                            subset,
                            float(merger_ld),
                        )

        if int(result.status) != 0:
            raise SeparationUnknown(
                "HiGHS original-formula QUBO separator did not prove "
                f"optimality: {result.message}"
            )
        if (not math.isfinite(objective_value)
                or objective_value < -RAW_OPT_TOL):
            raise SeparationUnknown(
                "original-formula QUBO separator found a negative or "
                "nonfinite optimum without a directly validated witness"
            )
        return None

    def _exact_violation_prefix(
            self,
            selected: np.ndarray
    ) -> object | tuple[float, tuple[int, ...], float] | None:
        """Separate small reported sets by an exact outside-prefix theorem.

        Conditional on the nonempty intersection B with the reported set R,
        a fixed-floor merger is minimized by a prefix of R-complement sorted
        by increasing e-value.  Every calculation below uses exact rational
        arithmetic on the input binary64 values.  Consequently a returned
        ``None`` is a genuine certificate, not a floating-point heuristic.
        Larger reported sets fall back to the unchanged exact HiGHS MILP.
        """
        chosen = np.asarray(selected, dtype=bool)
        inside = np.flatnonzero(chosen)
        r = int(inside.size)
        if r == 0:
            return None
        if PREFIX_ENUMERATION_CAP < r <= GRAY_PREFIX_ENUMERATION_CAP:
            return self._exact_violation_prefix_gray(chosen)
        if r > PREFIX_ENUMERATION_CAP:
            return PREFIX_FALLBACK

        outside = np.flatnonzero(~chosen)
        if outside.size:
            outside = outside[np.lexsort((outside, self.e[outside]))]

        def common_dyadic(values: np.ndarray) -> tuple[list[int], int]:
            ratios = [float(value).as_integer_ratio() for value in values]
            powers = [
                denominator.bit_length() - 1
                for _, denominator in ratios
            ]
            common_power = max(powers, default=0)
            integers = [
                numerator << (common_power - power)
                for (numerator, _), power in zip(ratios, powers)
            ]
            return integers, common_power

        # Binary64 inputs are dyadic rationals. Common-denominator integer
        # sums make every merger and threshold comparison below exact while
        # avoiding the overhead of general Fraction arithmetic.
        e_integer, e_power = common_dyadic(self.e)
        score_integer, _ = common_dyadic(self.score)
        prefix_e = [0]
        prefix_score = [0]
        prefix_score_e = [0]
        for raw_index in outside:
            index = int(raw_index)
            local_e = e_integer[index]
            local_score = score_integer[index]
            prefix_e.append(prefix_e[-1] + local_e)
            prefix_score.append(prefix_score[-1] + local_score)
            prefix_score_e.append(
                prefix_score_e[-1] + local_score * local_e
            )

        mask_count = 1 << r
        subset_e = [0] * mask_count
        subset_score = [0] * mask_count
        subset_score_e = [0] * mask_count
        subset_size = [0] * mask_count
        for mask in range(1, mask_count):
            bit = mask & -mask
            position = bit.bit_length() - 1
            previous = mask ^ bit
            index = int(inside[position])
            subset_e[mask] = subset_e[previous] + e_integer[index]
            subset_score[mask] = (
                subset_score[previous] + score_integer[index]
            )
            subset_score_e[mask] = (
                subset_score_e[previous]
                + score_integer[index] * e_integer[index]
            )
            subset_size[mask] = subset_size[previous] + 1

        alpha_numerator, alpha_denominator = self.alpha.as_integer_ratio()
        tolerance_numerator, tolerance_denominator = ABS_TOL.as_integer_ratio()
        e_denominator = 1 << e_power
        best_numerator: int | None = None
        best_denominator = 1
        best_mask = 0
        best_prefix = 0
        candidate_count = 0
        for mask in range(1, mask_count):
            count = subset_size[mask]
            for prefix_length in range(outside.size + 1):
                size = count + prefix_length
                e_sum = subset_e[mask] + prefix_e[prefix_length]
                score_sum = subset_score[mask] + prefix_score[prefix_length]
                score_e_sum = (
                    subset_score_e[mask] + prefix_score_e[prefix_length]
                )
                merger_numerator = (
                    e_sum * score_sum + (self.k - size) * score_e_sum
                )
                merger_denominator = self.k * e_denominator * score_sum
                violation_numerator = (
                    count * alpha_denominator * merger_denominator
                    - alpha_numerator * r * merger_numerator
                )
                violation_denominator = alpha_denominator * merger_denominator
                candidate_count += 1
                if (
                    best_numerator is None
                    or violation_numerator * best_denominator
                    > best_numerator * violation_denominator
                ):
                    best_numerator = violation_numerator
                    best_denominator = violation_denominator
                    best_mask = mask
                    best_prefix = prefix_length

        if best_numerator is None:
            raise AssertionError("nonempty reported set produced no prefix candidates")
        best_violation = best_numerator / best_denominator
        self.last_diagnostics = SeparatorDiagnostics(
            status=0,
            message=(
                "exact dyadic-integer outside-prefix enumeration "
                f"({candidate_count} candidates)"
            ),
            objective=-best_violation,
            dual_bound=-best_violation,
            elapsed_nodes=candidate_count,
        )
        if (
            best_numerator * tolerance_denominator
            <= tolerance_numerator * best_denominator
        ):
            return None

        mandatory = [
            int(inside[position])
            for position in range(r)
            if best_mask & (1 << position)
        ]
        subset = tuple(sorted(mandatory + [
            int(index) for index in outside[:best_prefix]
        ]))
        merger = self.merger(subset)
        count = int(np.sum(chosen[np.asarray(subset, dtype=int)]))
        violation = count - self.alpha * r * merger
        if violation > ABS_TOL:
            return float(violation), subset, merger
        # Exact rational arithmetic found a violation but the existing direct
        # binary64/long-double check regards it as numerically ambiguous.
        # Delegate only this boundary case to the established MILP.
        return PREFIX_FALLBACK

    def _exact_violation_prefix_gray(
            self,
            selected: np.ndarray
    ) -> object | tuple[float, tuple[int, ...], float] | None:
        """Exact low-memory separator for moderately sized reported sets.

        For a fixed nonempty B contained in the report, the outside-prefix
        merger is unimodal.  If the current score-weighted mean is m, adding
        the next outside coordinate changes the merger with the sign of
        e_next-m.  Since outside e-values are sorted, the minimizing prefix is
        therefore the first prefix whose next e-value is at least its current
        score-weighted mean.  The crossing condition is monotone, so it is
        found by binary search using exact dyadic-integer arithmetic.  Gray
        code enumeration updates the four sufficient statistics of B with one
        exact addition/subtraction per subset and avoids the large subset-sum
        tables used by the small-r implementation.
        """
        chosen = np.asarray(selected, dtype=bool)
        inside = np.flatnonzero(chosen)
        r = int(inside.size)
        if r == 0:
            return None
        if r > GRAY_PREFIX_ENUMERATION_CAP:
            return PREFIX_FALLBACK
        outside = np.flatnonzero(~chosen)
        if outside.size:
            outside = outside[np.lexsort((outside, self.e[outside]))]

        def common_dyadic(values: np.ndarray) -> tuple[list[int], int]:
            ratios = [float(value).as_integer_ratio() for value in values]
            powers = [den.bit_length() - 1 for _, den in ratios]
            common_power = max(powers, default=0)
            integers = [
                num << (common_power - power)
                for (num, _), power in zip(ratios, powers)
            ]
            return integers, common_power

        e_integer, e_power = common_dyadic(self.e)
        score_integer, _ = common_dyadic(self.score)
        prefix_e = [0]
        prefix_score = [0]
        prefix_score_e = [0]
        outside_e: list[int] = []
        for raw_index in outside:
            index = int(raw_index)
            local_e = e_integer[index]
            local_score = score_integer[index]
            outside_e.append(local_e)
            prefix_e.append(prefix_e[-1] + local_e)
            prefix_score.append(prefix_score[-1] + local_score)
            prefix_score_e.append(prefix_score_e[-1] + local_score * local_e)

        alpha_numerator, alpha_denominator = self.alpha.as_integer_ratio()
        tolerance_numerator, tolerance_denominator = ABS_TOL.as_integer_ratio()
        e_denominator = 1 << e_power
        best_numerator: int | None = None
        best_denominator = 1
        best_mask = 0
        best_prefix = 0
        subset_e = 0
        subset_score = 0
        subset_score_e = 0
        subset_size = 0
        previous_gray = 0
        mask_count = 1 << r
        outside_count = len(outside_e)

        for step in range(1, mask_count):
            gray = step ^ (step >> 1)
            changed = gray ^ previous_gray
            position = (changed & -changed).bit_length() - 1
            index = int(inside[position])
            if gray & changed:
                subset_e += e_integer[index]
                subset_score += score_integer[index]
                subset_score_e += score_integer[index] * e_integer[index]
                subset_size += 1
            else:
                subset_e -= e_integer[index]
                subset_score -= score_integer[index]
                subset_score_e -= score_integer[index] * e_integer[index]
                subset_size -= 1
            previous_gray = gray

            # First outside item whose e-value is no smaller than the current
            # weighted mean after the preceding prefix.  The comparison is
            # exact after cancelling the common dyadic denominators.
            low = 0
            high = outside_count
            while low < high:
                middle = (low + high) // 2
                if (
                    outside_e[middle]
                    * (subset_score + prefix_score[middle])
                    >= subset_score_e + prefix_score_e[middle]
                ):
                    high = middle
                else:
                    low = middle + 1
            prefix_length = low
            size = subset_size + prefix_length
            e_sum = subset_e + prefix_e[prefix_length]
            score_sum = subset_score + prefix_score[prefix_length]
            score_e_sum = subset_score_e + prefix_score_e[prefix_length]
            merger_numerator = (
                e_sum * score_sum + (self.k - size) * score_e_sum
            )
            merger_denominator = self.k * e_denominator * score_sum
            violation_numerator = (
                subset_size * alpha_denominator * merger_denominator
                - alpha_numerator * r * merger_numerator
            )
            violation_denominator = alpha_denominator * merger_denominator
            if (
                best_numerator is None
                or violation_numerator * best_denominator
                > best_numerator * violation_denominator
            ):
                best_numerator = violation_numerator
                best_denominator = violation_denominator
                best_mask = gray
                best_prefix = prefix_length

        if best_numerator is None:
            raise AssertionError("nonempty reported set produced no prefix candidates")
        best_violation = best_numerator / best_denominator
        self.last_diagnostics = SeparatorDiagnostics(
            status=0,
            message=(
                "exact dyadic Gray-code/outside-prefix enumeration "
                f"({mask_count - 1} report subsets)"
            ),
            objective=-best_violation,
            dual_bound=-best_violation,
            elapsed_nodes=mask_count - 1,
        )
        if (
            best_numerator * tolerance_denominator
            <= tolerance_numerator * best_denominator
        ):
            return None

        mandatory = [
            int(inside[position])
            for position in range(r)
            if best_mask & (1 << position)
        ]
        subset = tuple(sorted(mandatory + [
            int(index) for index in outside[:best_prefix]
        ]))
        merger = self.merger(subset)
        count = int(np.sum(chosen[np.asarray(subset, dtype=int)]))
        violation = count - self.alpha * r * merger
        if violation > ABS_TOL:
            return float(violation), subset, merger
        return PREFIX_FALLBACK

    def separate(
            self,
            selected: np.ndarray,
            max_cuts: int | None = None
    ) -> list[tuple[float, tuple[int, ...], float]]:
        chosen = np.asarray(selected, dtype=bool)
        if chosen.shape != (self.k,):
            raise ValueError("selected has the wrong shape")
        if max_cuts is not None and max_cuts <= 0:
            raise ValueError("max_cuts must be positive when supplied")
        key = chosen.tobytes()
        if key in self._certified_cache:
            return []
        limit = max_cuts if max_cuts is not None else 1
        quick = self._quick_violations(chosen, limit)
        if quick:
            return quick
        surrogate = self._surrogate_violations(chosen, limit)
        if surrogate:
            return surrogate
        exact = self._exact_violation_prefix(chosen)
        if exact is PREFIX_FALLBACK:
            exact = self._exact_violation_highs(chosen)
        if exact is None:
            self._certified_cache.add(key)
            return []
        return [exact]

    def is_certified(self, selected: np.ndarray) -> bool:
        chosen = np.asarray(selected, dtype=bool)
        if chosen.shape == (self.k,) and np.array_equal(chosen, self.ebh_seed):
            return True
        return not self.separate(chosen, max_cuts=1)

    def first_violation(
            self,
            selected: np.ndarray) -> tuple[tuple[int, ...], float] | None:
        violations = self.separate(selected, max_cuts=1)
        if not violations:
            return None
        _, subset, merger = violations[0]
        return subset, merger

    def early_violations(
            self,
            selected: np.ndarray,
            limit: int) -> list[tuple[tuple[int, ...], float]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        violations = self.separate(selected, max_cuts=1)
        return [(subset, merger) for _, subset, merger in violations]

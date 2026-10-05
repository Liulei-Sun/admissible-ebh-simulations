#!/usr/bin/env python3
"""Exact certificate oracle for the locally minimally-shrunk Choice-2 weights.

For every proper, nonempty intersection A, Choice 2 adds the smallest common
score offset needed to make every normalized weight at least 1/K.  Unlike the
size-calibrated rule in the original simulation, that offset depends on the
actual members of A.  Consequently the original order-statistic separator is
not valid.  This module uses an exact mixed-integer formulation instead.

The separator minimizes, over all intersections A meeting a proposed
rejection set R,

    alpha * |R| * numerator(A) - |A intersect R| * denominator(A).

The Choice-2 offset, the minimum selected score, and all binary-continuous
products are represented explicitly.  A negative optimum is exactly a closed
e-BH violation (up to the documented floating-point tolerances).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ABS_TOL = 1e-10
RAW_OPT_TOL = 1e-9


class SeparationUnknown(RuntimeError):
    """Raised when the exact separator did not obtain a numerical proof."""


def ordinary_ebh(e: np.ndarray, alpha: float) -> np.ndarray:
    """Return ordinary e-BH with deterministic index tie-breaking."""
    values = np.asarray(e, dtype=float)
    if values.ndim != 1:
        raise ValueError("e must be one-dimensional")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("e-values must be finite and nonnegative")

    k = values.size
    order = np.lexsort((np.arange(k), -values))
    ordered = values[order]
    eligible = ordered + ABS_TOL >= k / (alpha * np.arange(1, k + 1))
    selected = np.zeros(k, dtype=bool)
    if np.any(eligible):
        count = int(np.flatnonzero(eligible)[-1]) + 1
        selected[order[:count]] = True
    return selected


def _choice2_raw_offset(score: np.ndarray, k: int) -> np.longdouble:
    """Return the common raw-score offset for one proper intersection."""
    a = score.size
    if not (1 <= a < k):
        raise ValueError("the raw offset is defined only for 1 <= |A| < K")
    total = np.sum(score, dtype=np.longdouble)
    minimum = np.min(score).astype(np.longdouble)
    numerator = max(np.longdouble(0.0), total - np.longdouble(k) * minimum)
    return numerator / np.longdouble(k - a)


def choice2_merger(e: np.ndarray,
                   score: np.ndarray,
                   subset: tuple[int, ...] | np.ndarray) -> float:
    """Evaluate the Choice-2 weighted mean for a nonempty intersection."""
    values = np.asarray(e, dtype=float)
    scores = np.asarray(score, dtype=float)
    idx = np.asarray(subset, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("subset must be a nonempty one-dimensional index set")
    if values.shape != scores.shape or values.ndim != 1:
        raise ValueError("e and score must be equal-length vectors")
    k = values.size
    if idx.size == 1:
        return float(values[idx[0]])
    if idx.size == k:
        return float(np.mean(values, dtype=np.longdouble))

    local_score = scores[idx].astype(np.longdouble)
    offset = _choice2_raw_offset(local_score, k)
    raw = local_score + offset
    local_e = values[idx].astype(np.longdouble)
    return float(np.dot(raw, local_e) / np.sum(raw, dtype=np.longdouble))


@dataclass(frozen=True)
class SeparatorDiagnostics:
    status: int
    message: str
    objective: float
    dual_bound: float
    elapsed_nodes: int


class Choice2Oracle:
    """Exact Choice-2 merger and closed-eBH separation oracle."""

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
        if self.e.shape != original_score.shape:
            raise ValueError("e and score must have equal length")
        if self.e.size < 2:
            raise ValueError("Choice2Oracle requires K >= 2")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0,1)")
        if not np.all(np.isfinite(self.e)) or np.any(self.e < 0.0):
            raise ValueError("e-values must be finite and nonnegative")
        if not np.all(np.isfinite(original_score)) or np.any(original_score <= 0.0):
            raise ValueError("scores must be finite and strictly positive")
        if not math.isfinite(self.separator_time_limit) or self.separator_time_limit <= 0.0:
            raise ValueError("separator_time_limit must be positive and finite")
        if self.separator_threads <= 0:
            raise ValueError("separator_threads must be positive")

        self.k = self.e.size
        # Choice 2 is scale-invariant.  Sum-normalization gives tight and
        # instance-independent bounds (rho <= 1 and denominator <= K) in the
        # exact separator, materially improving MIP conditioning.
        score_sum = np.sum(original_score, dtype=np.longdouble)
        self.score = np.asarray(original_score / float(score_sum), dtype=float)
        self.ebh_seed = ordinary_ebh(self.e, self.alpha)
        self._certified_cache: set[bytes] = {self.ebh_seed.tobytes()}
        # Optional fast candidate generators may be attached by the runner.
        # They never certify a set: every proposed subset is re-evaluated with
        # the exact Choice-2 merger, and SCIP is still called if none violates.
        self.surrogate_oracles: list[object] = []
        self.last_diagnostics: SeparatorDiagnostics | None = None
        self.pair_merger = self._make_pair_mergers()

    def _make_pair_mergers(self) -> np.ndarray:
        if self.k == 2:
            return np.full((2, 2), float(np.mean(self.e, dtype=np.longdouble)))
        answer = np.empty((self.k, self.k), dtype=float)
        np.fill_diagonal(answer, self.e)
        for i in range(self.k):
            si = np.longdouble(self.score[i])
            ei = np.longdouble(self.e[i])
            for j in range(i + 1, self.k):
                sj = np.longdouble(self.score[j])
                total = si + sj
                minimum = min(si, sj)
                offset = max(
                    np.longdouble(0.0),
                    total - np.longdouble(self.k) * minimum,
                ) / np.longdouble(self.k - 2)
                wi = si + offset
                wj = sj + offset
                merger = float((wi * ei + wj * np.longdouble(self.e[j])) / (wi + wj))
                answer[i, j] = merger
                answer[j, i] = merger
        return answer

    def merger(self, subset: tuple[int, ...]) -> float:
        return choice2_merger(self.e, self.score, subset)

    def _quick_violations(
            self,
            selected: np.ndarray,
            limit: int) -> list[tuple[float, tuple[int, ...], float]]:
        """Return the strongest singleton, full-set, and pair violations."""
        inside = np.flatnonzero(selected)
        r = inside.size
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
            q = membership[:, None] + membership[None, :]
            violation = q - self.alpha * r * self.pair_merger
            violation[q == 0] = -np.inf
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

    def _quick_violation(self,
                         selected: np.ndarray) -> tuple[float, tuple[int, ...], float] | None:
        found = self._quick_violations(selected, 1)
        return found[0] if found else None

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
                q = int(np.sum(selected[np.asarray(subset, dtype=int)]))
                violation = q - self.alpha * r * merger
                if violation > ABS_TOL:
                    found[subset] = (float(violation), subset, merger)
        answer = sorted(found.values(), key=lambda item: item[0], reverse=True)
        return answer[:limit]

    @staticmethod
    def _append_row(rows: list[int],
                    cols: list[int],
                    vals: list[float],
                    lower: list[float],
                    upper: list[float],
                    coefficients: list[tuple[int, float]],
                    lo: float,
                    hi: float) -> None:
        row = len(lower)
        for col, value in coefficients:
            if value != 0.0:
                rows.append(row)
                cols.append(col)
                vals.append(float(value))
        lower.append(float(lo))
        upper.append(float(hi))

    def _exact_proper_violation_fixed_q(
            self,
            selected: np.ndarray) -> tuple[float, tuple[int, ...], float] | None:
        """Exact separator using fixed intersection counts.

        Fixing q=|A intersect R| removes the only product between the
        rejection membership and the raw-weight denominator.  The resulting
        models are much tighter than the single all-q formulation.  We solve
        the no-shrink and active-floor regimes separately for every q.
        """
        inside = np.flatnonzero(selected)
        r = int(inside.size)
        if r == 0 or self.k <= 2:
            return None
        k = self.k
        score_max = float(np.max(self.score))
        all_messages: list[str] = []
        all_objectives: list[float] = []
        all_duals: list[float] = []
        all_nodes: list[int] = []

        def add_common_minimum_constraints(
                x0: int,
                y0: int,
                min_col: int,
                nvar: int,
                q: int):
            rows: list[int] = []
            cols: list[int] = []
            vals: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            add = self._append_row
            add(rows, cols, vals, lower, upper,
                [(x0 + i, 1.0) for i in range(k)], 2.0, float(k - 1))
            add(rows, cols, vals, lower, upper,
                [(x0 + int(i), 1.0) for i in inside], float(q), float(q))
            add(rows, cols, vals, lower, upper,
                [(y0 + i, 1.0) for i in range(k)], 1.0, 1.0)
            for i in range(k):
                add(rows, cols, vals, lower, upper,
                    [(y0 + i, 1.0), (x0 + i, -1.0)], -np.inf, 0.0)
            add(rows, cols, vals, lower, upper,
                [(min_col, 1.0)]
                + [(y0 + i, -self.score[i]) for i in range(k)],
                0.0, 0.0)
            for i in range(k):
                big_m = score_max - float(self.score[i])
                add(rows, cols, vals, lower, upper,
                    [(min_col, 1.0), (x0 + i, big_m)],
                    -np.inf, float(self.score[i]) + big_m)
            return rows, cols, vals, lower, upper

        def solve_one(q: int, active: bool):
            x0 = 0
            y0 = k
            min_col = 2 * k
            if active:
                rho_col = 2 * k + 1
                z0 = 2 * k + 2
                nvar = 3 * k + 2
            else:
                rho_col = -1
                z0 = -1
                nvar = 2 * k + 1

            rows, cols, vals, lower, upper = add_common_minimum_constraints(
                x0, y0, min_col, nvar, q
            )
            add = self._append_row
            u_terms = (
                [(x0 + i, self.score[i]) for i in range(k)]
                + [(min_col, -float(k))]
            )
            if active:
                add(rows, cols, vals, lower, upper,
                    u_terms, 0.0, np.inf)
                equality = (
                    [(rho_col, float(k))]
                    + [(z0 + i, -1.0) for i in range(k)]
                    + [(x0 + i, -self.score[i]) for i in range(k)]
                    + [(min_col, float(k))]
                )
                add(rows, cols, vals, lower, upper, equality, 0.0, 0.0)
                for i in range(k):
                    z = z0 + i
                    x = x0 + i
                    add(rows, cols, vals, lower, upper,
                        [(z, 1.0), (rho_col, -1.0)], -np.inf, 0.0)
                    add(rows, cols, vals, lower, upper,
                        [(z, 1.0), (x, -1.0)], -np.inf, 0.0)
                    add(rows, cols, vals, lower, upper,
                        [(z, 1.0), (rho_col, -1.0), (x, -1.0)],
                        -1.0, np.inf)
            else:
                add(rows, cols, vals, lower, upper,
                    u_terms, -np.inf, 0.0)

            matrix = coo_matrix(
                (vals, (rows, cols)), shape=(len(lower), nvar)
            ).tocsr()
            constraints = LinearConstraint(
                matrix,
                np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float),
            )
            variable_lower = np.zeros(nvar, dtype=float)
            variable_upper = np.ones(nvar, dtype=float)
            variable_lower[min_col] = float(np.min(self.score))
            variable_upper[min_col] = score_max
            integrality = np.zeros(nvar, dtype=np.uint8)
            integrality[x0:x0 + k] = 1

            transformed = self.alpha * r * self.e - q
            objective = np.zeros(nvar, dtype=float)
            objective[x0:x0 + k] = self.score * transformed
            if active:
                objective[z0:z0 + k] = transformed

            return milp(
                objective,
                integrality=integrality,
                bounds=Bounds(variable_lower, variable_upper),
                constraints=constraints,
                options={
                    "time_limit": self.separator_time_limit,
                    "mip_rel_gap": 0.0,
                    "presolve": True,
                },
            ), x0

        # Large q is typically the most restrictive and therefore finds a
        # witness fastest.  Certification still checks every q and both
        # regimes.
        for q in range(r, 0, -1):
            for active in (False, True):
                solution, x0 = solve_one(q, active)
                raw_dual = getattr(solution, "mip_dual_bound", None)
                dual = float(raw_dual) if raw_dual is not None else math.inf
                raw_nodes = getattr(solution, "mip_node_count", None)
                nodes = int(raw_nodes) if raw_nodes is not None else 0
                objective_value = (
                    float(solution.fun) if solution.fun is not None else math.inf
                )
                all_messages.append(
                    f"q={q},active={int(active)}: {solution.message}"
                )
                all_objectives.append(objective_value)
                all_duals.append(dual)
                all_nodes.append(nodes)

                if solution.x is not None:
                    candidate = solution.x[x0:x0 + k] > 0.5
                    subset = tuple(int(i) for i in np.flatnonzero(candidate))
                    if 2 <= len(subset) <= k - 1:
                        merger = self.merger(subset)
                        actual_q = int(np.sum(candidate[inside]))
                        violation = actual_q - self.alpha * r * merger
                        if actual_q == q and violation > ABS_TOL:
                            self.last_diagnostics = SeparatorDiagnostics(
                                status=int(solution.status),
                                message=all_messages[-1],
                                objective=objective_value,
                                dual_bound=dual,
                                elapsed_nodes=sum(all_nodes),
                            )
                            return float(violation), subset, merger

                if solution.status == 2:
                    continue
                if solution.status != 0:
                    self.last_diagnostics = SeparatorDiagnostics(
                        status=int(solution.status),
                        message=all_messages[-1],
                        objective=objective_value,
                        dual_bound=dual,
                        elapsed_nodes=sum(all_nodes),
                    )
                    raise SeparationUnknown(
                        "Choice-2 fixed-q separator did not prove optimality: "
                        f"q={q}, active={active}, {solution.message}"
                    )
                if objective_value < -RAW_OPT_TOL:
                    raise SeparationUnknown(
                        "Choice-2 fixed-q separator found a negative raw optimum "
                        "without a directly validated witness"
                    )

        self.last_diagnostics = SeparatorDiagnostics(
            status=0,
            message=f"{len(all_messages)} fixed-q regime proofs completed",
            objective=min(all_objectives),
            dual_bound=min(all_duals),
            elapsed_nodes=sum(all_nodes),
        )
        return None

    def _exact_proper_violation_scip(
            self,
            selected: np.ndarray) -> tuple[float, tuple[int, ...], float] | None:
        """Exact all-q separator using SCIP indicator constraints."""
        from pyscipopt import Model, quicksum

        inside = np.flatnonzero(selected)
        r = int(inside.size)
        if r == 0 or self.k <= 2:
            return None
        k = self.k
        model = Model("choice2_separator")
        model.hideOutput()
        model.setRealParam("limits/time", self.separator_time_limit)
        model.setRealParam("limits/gap", 0.0)
        model.setRealParam("limits/absgap", 0.0)
        model.setIntParam("parallel/maxnthreads", self.separator_threads)
        model.setIntParam("randomization/randomseedshift", 0)
        model.setRealParam("numerics/feastol", 1e-9)

        x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(k)]
        y = [model.addVar(lb=0.0, ub=1.0, name=f"y_{i}") for i in range(k)]
        minimum = model.addVar(
            lb=float(np.min(self.score)),
            ub=float(np.max(self.score)),
            name="minimum",
        )
        rho = model.addVar(lb=0.0, ub=1.0, name="rho")
        z = [model.addVar(lb=0.0, ub=1.0, name=f"z_{i}") for i in range(k)]
        active = model.addVar(vtype="B", name="active")
        denominator = model.addVar(lb=0.0, ub=float(k), name="denominator")
        v = [model.addVar(lb=0.0, ub=float(k), name=f"v_{j}") for j in range(r)]

        size = quicksum(x)
        model.addCons(size >= 2)
        model.addCons(size <= k - 1)
        model.addCons(quicksum(x[int(i)] for i in inside) >= 1)

        model.addCons(quicksum(y) == 1)
        model.addCons(minimum == quicksum(self.score[i] * y[i] for i in range(k)))
        score_max = float(np.max(self.score))
        for i in range(k):
            model.addCons(y[i] <= x[i])
            model.addCons(
                minimum <= self.score[i] + (score_max - self.score[i]) * (1 - x[i])
            )

        for i in range(k):
            model.addCons(z[i] <= rho)
            model.addCons(z[i] <= x[i])
            model.addCons(z[i] >= rho - (1 - x[i]))

        score_total = quicksum(self.score[i] * x[i] for i in range(k))
        u = score_total - k * minimum
        active_equality = k * rho - quicksum(z) - u
        model.addConsIndicator(u <= 0, binvar=active, activeone=False)
        model.addConsIndicator(rho <= 0, binvar=active, activeone=False)
        model.addConsIndicator(u >= 0, binvar=active, activeone=True)
        model.addConsIndicator(active_equality <= 0, binvar=active, activeone=True)
        model.addConsIndicator(active_equality >= 0, binvar=active, activeone=True)

        model.addCons(denominator == score_total + quicksum(z))
        for row, coordinate in enumerate(inside):
            i = int(coordinate)
            model.addCons(v[row] <= denominator)
            model.addCons(v[row] <= k * x[i])
            model.addCons(v[row] >= denominator - k * (1 - x[i]))

        numerator = quicksum(
            self.score[i] * self.e[i] * x[i] + self.e[i] * z[i]
            for i in range(k)
        )
        separating_expression = self.alpha * r * numerator - quicksum(v)
        model.setObjective(separating_expression, sense="minimize")
        model.optimize()

        status = str(model.getStatus())
        objective_value = math.nan
        if model.getNSols() > 0:
            solution = model.getBestSol()
            objective_value = float(model.getSolVal(solution, separating_expression))
            candidate = np.asarray(
                [model.getSolVal(solution, variable) > 0.5 for variable in x],
                dtype=bool,
            )
            subset = tuple(int(i) for i in np.flatnonzero(candidate))
            if 2 <= len(subset) <= k - 1 and np.any(candidate[inside]):
                merger = self.merger(subset)
                q = int(np.sum(candidate[inside]))
                violation = q - self.alpha * r * merger
                if violation > ABS_TOL:
                    self.last_diagnostics = SeparatorDiagnostics(
                        status=0,
                        message=f"SCIP {status}",
                        objective=objective_value,
                        dual_bound=float(model.getDualbound()),
                        elapsed_nodes=int(model.getNNodes()),
                    )
                    return float(violation), subset, merger

        self.last_diagnostics = SeparatorDiagnostics(
            status=0 if status == "optimal" else 1,
            message=f"SCIP {status}",
            objective=objective_value,
            dual_bound=float(model.getDualbound()),
            elapsed_nodes=int(model.getNNodes()),
        )
        if status != "optimal":
            raise SeparationUnknown(
                f"SCIP Choice-2 separator did not prove optimality: {status}"
            )
        if objective_value < -RAW_OPT_TOL:
            raise SeparationUnknown(
                "SCIP separator found a negative optimum without a directly "
                "validated witness"
            )
        return None

    def _exact_proper_violation(self,
                                selected: np.ndarray) -> tuple[float, tuple[int, ...], float] | None:
        """Globally separate all intersections with 2 <= |A| <= K-1."""
        inside = np.flatnonzero(selected)
        r = int(inside.size)
        if r == 0 or self.k <= 2:
            return None

        k = self.k
        x0 = 0
        y0 = k
        rho_col = 2 * k
        min_col = 2 * k + 1
        denominator_col = 2 * k + 2
        z0 = 2 * k + 3
        active_col = 3 * k + 3
        v0 = 3 * k + 4
        nvar = v0 + r

        integrality = np.zeros(nvar, dtype=np.uint8)
        integrality[x0:x0 + k] = 1
        # y is a continuous simplex supported on the selected coordinates.
        # Its weighted-average score is constrained to be no larger than
        # every selected score, so it must put all mass on selected minima.
        # Binary y variables are unnecessary and make K=400 much slower.
        integrality[active_col] = 1

        lower_bound = np.zeros(nvar, dtype=float)
        upper_bound = np.full(nvar, np.inf, dtype=float)
        upper_bound[x0:x0 + k] = 1.0
        upper_bound[y0:y0 + k] = 1.0
        upper_bound[rho_col] = 1.0
        upper_bound[min_col] = float(np.max(self.score))
        upper_bound[denominator_col] = float(k)
        upper_bound[z0:z0 + k] = 1.0
        upper_bound[active_col] = 1.0
        upper_bound[v0:v0 + r] = float(k)
        lower_bound[min_col] = float(np.min(self.score))

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        add = self._append_row

        # A is a proper intersection of size at least two and meets R.
        add(rows, cols, vals, lower, upper,
            [(x0 + i, 1.0) for i in range(k)], 2.0, float(k - 1))
        add(rows, cols, vals, lower, upper,
            [(x0 + int(i), 1.0) for i in inside], 1.0, float(r))

        # One selected coordinate identifies the minimum score in A.
        add(rows, cols, vals, lower, upper,
            [(y0 + i, 1.0) for i in range(k)], 1.0, 1.0)
        for i in range(k):
            add(rows, cols, vals, lower, upper,
                [(y0 + i, 1.0), (x0 + i, -1.0)], -np.inf, 0.0)
        add(rows, cols, vals, lower, upper,
            [(min_col, 1.0)] + [(y0 + i, -self.score[i]) for i in range(k)],
            0.0, 0.0)
        score_max = float(np.max(self.score))
        for i in range(k):
            big_m = score_max - float(self.score[i])
            add(rows, cols, vals, lower, upper,
                [(min_col, 1.0), (x0 + i, big_m)],
                -np.inf, float(self.score[i]) + big_m)

        # z_i = rho * x_i, with 0 <= rho <= 1 after score normalization.
        for i in range(k):
            z = z0 + i
            x = x0 + i
            add(rows, cols, vals, lower, upper,
                [(z, 1.0), (rho_col, -1.0)], -np.inf, 0.0)
            add(rows, cols, vals, lower, upper,
                [(z, 1.0), (x, -1.0)], -np.inf, 0.0)
            add(rows, cols, vals, lower, upper,
                [(z, 1.0), (rho_col, -1.0), (x, -1.0)], -1.0, np.inf)

        # u = S-Km chooses the no-shrink (u<=0,rho=0) or active
        # (u>=0,(K-|A|)rho=u) branch.  Since sum(score)=1, u<=1.
        score_terms = [(x0 + i, self.score[i]) for i in range(k)]
        u_terms = score_terms + [(min_col, -float(k))]
        negative_u_bound = float(k) * score_max
        add(rows, cols, vals, lower, upper,
            u_terms + [(active_col, -1.0)], -np.inf, 0.0)
        add(rows, cols, vals, lower, upper,
            u_terms + [(active_col, -negative_u_bound)],
            -negative_u_bound, np.inf)
        add(rows, cols, vals, lower, upper,
            [(rho_col, 1.0), (active_col, -1.0)], -np.inf, 0.0)

        # In the active branch: K*rho-sum(z)=S-Km.  In the inactive
        # branch the equality is relaxed and rho is already forced to zero.
        equality_terms = (
            [(rho_col, float(k))]
            + [(z0 + i, -1.0) for i in range(k)]
            + [(x0 + i, -self.score[i]) for i in range(k)]
            + [(min_col, float(k))]
        )
        equality_m = max(1.0, negative_u_bound)
        add(rows, cols, vals, lower, upper,
            equality_terms + [(active_col, equality_m)],
            -np.inf, equality_m)
        add(rows, cols, vals, lower, upper,
            equality_terms + [(active_col, -equality_m)],
            -equality_m, np.inf)

        # D = sum_A(s_i+rho) is the positive raw-weight denominator.
        add(rows, cols, vals, lower, upper,
            [(denominator_col, 1.0)]
            + [(x0 + i, -self.score[i]) for i in range(k)]
            + [(z0 + i, -1.0) for i in range(k)],
            0.0, 0.0)

        # v_j = x_j*D for j in R, so sum_j v_j=|A intersect R|*D.
        denominator_max = float(k)
        for row_number, coordinate in enumerate(inside):
            v = v0 + row_number
            x = x0 + int(coordinate)
            add(rows, cols, vals, lower, upper,
                [(v, 1.0), (denominator_col, -1.0)], -np.inf, 0.0)
            add(rows, cols, vals, lower, upper,
                [(v, 1.0), (x, -denominator_max)], -np.inf, 0.0)
            add(rows, cols, vals, lower, upper,
                [(v, 1.0), (denominator_col, -1.0),
                 (x, -denominator_max)],
                -denominator_max, np.inf)

        matrix = coo_matrix(
            (vals, (rows, cols)), shape=(len(lower), nvar)
        ).tocsr()
        constraints = LinearConstraint(
            matrix, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        )

        objective = np.zeros(nvar, dtype=float)
        scale = self.alpha * r
        objective[x0:x0 + k] = scale * self.score * self.e
        objective[z0:z0 + k] = scale * self.e
        objective[v0:v0 + r] = -1.0

        solution = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(lower_bound, upper_bound),
            constraints=constraints,
            options={
                "time_limit": self.separator_time_limit,
                "mip_rel_gap": 0.0,
                "presolve": True,
            },
        )
        raw_dual = getattr(solution, "mip_dual_bound", None)
        dual_bound = float(raw_dual) if raw_dual is not None else math.inf
        raw_nodes = getattr(solution, "mip_node_count", None)
        nodes = int(raw_nodes) if raw_nodes is not None else 0
        objective_value = float(solution.fun) if solution.fun is not None else math.nan
        self.last_diagnostics = SeparatorDiagnostics(
            status=int(solution.status),
            message=str(solution.message),
            objective=objective_value,
            dual_bound=dual_bound,
            elapsed_nodes=nodes,
        )

        if solution.x is not None:
            candidate = solution.x[x0:x0 + k] > 0.5
            subset = tuple(int(i) for i in np.flatnonzero(candidate))
            if 2 <= len(subset) <= k - 1 and np.any(candidate[inside]):
                merger = self.merger(subset)
                q = int(np.sum(candidate[inside]))
                violation = q - self.alpha * r * merger
                if violation > ABS_TOL:
                    return float(violation), subset, merger

        if solution.status != 0:
            raise SeparationUnknown(
                f"Choice-2 separator did not prove optimality: {solution.message}"
            )
        if not math.isfinite(objective_value):
            raise SeparationUnknown("Choice-2 separator returned a nonfinite optimum")
        if objective_value < -RAW_OPT_TOL:
            raise SeparationUnknown(
                "Choice-2 separator found a negative raw optimum but the returned "
                "intersection did not pass direct numerical validation"
            )
        return None

    def separate(self,
                 selected: np.ndarray,
                 max_cuts: int | None = None) -> list[tuple[float, tuple[int, ...], float]]:
        """Return an exact violated intersection, or an empty proved list."""
        chosen = np.asarray(selected, dtype=bool)
        if chosen.shape != (self.k,):
            raise ValueError("selected has the wrong shape")
        if max_cuts is not None and max_cuts <= 0:
            raise ValueError("max_cuts must be positive when supplied")
        cache_key = chosen.tobytes()
        if cache_key in self._certified_cache:
            return []
        limit = max_cuts if max_cuts is not None else 1
        quick = self._quick_violations(chosen, limit)
        if quick:
            return quick
        surrogate = self._surrogate_violations(chosen, limit)
        if surrogate:
            return surrogate
        exact = self._exact_proper_violation_scip(chosen)
        if exact is None:
            self._certified_cache.add(cache_key)
            return []
        return [exact]

    def is_certified(self, selected: np.ndarray) -> bool:
        chosen = np.asarray(selected, dtype=bool)
        # The 1/K weight floor gives a direct mathematical certificate for
        # ordinary e-BH; avoid spending a MIP solve reproving this theorem on
        # every replication.
        if chosen.shape == (self.k,) and np.array_equal(chosen, self.ebh_seed):
            return True
        return not self.separate(chosen, max_cuts=1)

    def first_violation(self,
                        selected: np.ndarray) -> tuple[tuple[int, ...], float] | None:
        violations = self.separate(selected, max_cuts=1)
        if not violations:
            return None
        _, subset, merger = violations[0]
        return subset, merger

    def early_violations(self,
                         selected: np.ndarray,
                         limit: int) -> list[tuple[tuple[int, ...], float]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        violations = self.separate(selected, max_cuts=1)
        return [(subset, merger) for _, subset, merger in violations]

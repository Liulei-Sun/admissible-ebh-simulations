#!/usr/bin/env python3
"""Reproduce the revised admissible-constant simulation.

The optimized certification calculation is checked against literal enumeration
of all intersection subsets on random small problems before the simulation.
Admissibility labels refer to the underlying simultaneous closed-eBH
procedure, not to the reported point selector obtained by fixed tie-breaking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from itertools import product
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np


TOLERANCE = 2e-12
DEFAULT_BETAS = (
    0.0,
    0.004,
    0.80,
    0.90,
    1.0,
)


def format_beta(beta: float) -> str:
    """Preserve custom beta values while retaining the paper's display style."""
    if beta == 0.0:
        return "0"
    if beta >= 0.01 and float(f"{beta:.2f}") == beta:
        return f"{beta:.2f}"
    return repr(float(beta))


def constants_for_beta(k: int, alpha: float, beta: float) -> np.ndarray:
    n = np.arange(k + 1, dtype=float)
    constants = np.zeros(k + 1)
    active = (n > 0) & (n < beta * k)
    constants[active] = (beta * k - n[active]) / (beta * k - alpha * n[active])
    return constants


def admissibility_threshold(k: int, alpha: float) -> float:
    # Convert the displayed decimal exactly, avoiding floor errors near the
    # reciprocal-lattice discontinuities of the sharp threshold.
    alpha_exact = Fraction(str(alpha))
    q = alpha_exact.denominator // alpha_exact.numerator
    first = alpha_exact * q / (alpha_exact * (q + 1) - 1)
    second = Fraction(max(1, k - q + 1), 1)
    return float(min(first, second) / k)


def ebh_size(sorted_e: np.ndarray, alpha: float) -> int:
    """Return the number of rejections made by the base eBH procedure.

    The qualifying ranks need not form an initial segment, so the definition
    requires the largest qualifying rank rather than the number of qualifying
    ranks.  The comparison is exact for the stored floating-point e-values;
    downstream certification checks use ``TOLERANCE`` for accumulated sums.
    """
    k = sorted_e.size
    ranks = np.arange(1, k + 1, dtype=np.intp)
    qualifying = sorted_e >= k / (alpha * ranks)
    indices = np.flatnonzero(qualifying)
    return int(ranks[indices[-1]]) if indices.size else 0


def constants_dominate_self_consistent(
    k: int, alpha: float, beta: float
) -> bool:
    """Check the sharp condition ensuring certification of every eSC set."""
    n = np.arange(1, k + 1, dtype=float)
    constants = constants_for_beta(k, alpha, beta)[1:]
    lhs = alpha * constants + k * (1.0 - constants) / n
    scale = np.maximum(1.0, np.abs(lhs))
    return bool(np.all(lhs + TOLERANCE * scale >= 1.0))


@dataclass(frozen=True)
class ConstraintCache:
    inside: np.ndarray
    outside: np.ndarray
    size: np.ndarray
    constant: np.ndarray
    required: np.ndarray


# Large K creates large certification grids. Keep only a small working set
# instead of retaining every r.
@lru_cache(maxsize=8)
def constraint_cache(k: int, alpha: float, beta: float, r: int) -> ConstraintCache:
    ell_grid, outside_grid = np.meshgrid(
        np.arange(1, r + 1, dtype=np.int16),
        np.arange(0, k - r + 1, dtype=np.int16),
        indexing="ij",
    )
    # int16 is sufficient for K <= 32767 and substantially reduces the
    # memory footprint of the K=5000 experiment. NumPy accepts it for indexing.
    ell = ell_grid.ravel()
    outside = outside_grid.ravel()
    size = ell + outside
    constants = constants_for_beta(k, alpha, beta)
    return ConstraintCache(
        inside=ell,
        outside=outside,
        size=size,
        constant=constants[size],
        required=ell / (alpha * r),
    )


def candidate_fast(sorted_e: np.ndarray, r: int, alpha: float, beta: float) -> bool:
    k = sorted_e.size
    cache = constraint_cache(k, alpha, beta, r)
    # Accumulate the weakest entries directly.  This avoids subtracting large
    # neighboring prefix sums when the e-values are highly dispersed.
    weakest_inside = np.cumsum(sorted_e[:r][::-1])
    weakest_outside = np.empty(k - r + 1)
    weakest_outside[0] = 0.0
    np.cumsum(sorted_e[r:][::-1], out=weakest_outside[1:])
    weakest_sums = (
        weakest_inside[cache.inside - 1]
        + weakest_outside[cache.outside]
    )
    merged = (
        cache.constant
        + (1.0 - cache.constant) * weakest_sums / cache.size
    )
    scale = np.maximum.reduce((np.ones_like(merged), np.abs(merged), cache.required))
    return bool(np.all(merged + TOLERANCE * scale >= cache.required))


def candidate_convex(sorted_e: np.ndarray, r: int, alpha: float, beta: float) -> bool:
    """Certify a candidate using the convexity of weakest-outside sums.

    For each possible number ``ell`` of rejected hypotheses in an intersection,
    the original exhaustive check varies only over the number ``j`` outside the
    rejection set.  The weakest-outside prefix sum is convex in ``j`` because
    its increments are sorted increasingly.  In each of the two constant-term
    regimes, the constraint is therefore minimized where the next outside
    e-value crosses a single slope.  This reduces memory from O(r(K-r)) to O(K)
    and work from O(r(K-r)) to O(r log K), without changing the constraints.
    """
    k = sorted_e.size
    outside_count = k - r

    weakest_inside = np.empty(r + 1)
    weakest_inside[0] = 0.0
    np.cumsum(sorted_e[:r][::-1], out=weakest_inside[1:])

    outside_ascending = sorted_e[r:][::-1]
    weakest_outside = np.empty(outside_count + 1)
    weakest_outside[0] = 0.0
    np.cumsum(outside_ascending, out=weakest_outside[1:])

    ell = np.arange(1, r + 1, dtype=np.intp)
    inside_sum = weakest_inside[1:]
    constants = constants_for_beta(k, alpha, beta)
    active_sizes = np.flatnonzero(constants > 0.0)
    largest_active_size = int(active_sizes[-1]) if active_sizes.size else 0

    def constraints_hold(j: np.ndarray, valid: np.ndarray) -> bool:
        if not np.any(valid):
            return True
        ell_v = ell[valid]
        j_v = j[valid]
        n_v = ell_v + j_v
        merged = (
            constants[n_v]
            + (1.0 - constants[n_v])
            * (inside_sum[valid] + weakest_outside[j_v])
            / n_v
        )
        required = ell_v / (alpha * r)
        scale = np.maximum.reduce((np.ones_like(merged), np.abs(merged), required))
        return bool(np.all(merged + TOLERANCE * scale >= required))

    # Positive-constant regime: n=ell+j <= largest_active_size.  After
    # clearing positive factors, the j-dependent term is proportional to
    # B_j - j (r-ell)/(r(1-alpha)).
    active_hi = np.minimum(outside_count, largest_active_size - ell)
    active_valid = active_hi >= 0
    active_slope = (r - ell) / (r * (1.0 - alpha))
    active_minimizer = np.searchsorted(
        outside_ascending, active_slope, side="left"
    ).astype(np.intp, copy=False)
    active_j = np.minimum(active_minimizer, np.maximum(active_hi, 0))
    if not constraints_hold(active_j, active_valid):
        return False

    # Zero-constant regime: n >= largest_active_size+1.  The transformed
    # j-dependent term is B_j - j*ell/(alpha*r).
    inactive_lo = np.maximum(0, largest_active_size + 1 - ell)
    inactive_valid = inactive_lo <= outside_count
    inactive_slope = ell / (alpha * r)
    inactive_minimizer = np.searchsorted(
        outside_ascending, inactive_slope, side="left"
    ).astype(np.intp, copy=False)
    inactive_j = np.minimum(
        outside_count, np.maximum(inactive_minimizer, inactive_lo)
    )
    return constraints_hold(inactive_j, inactive_valid)


def candidate_set_bruteforce(
    e_values: np.ndarray,
    rejected: np.ndarray,
    alpha: float,
    beta: float,
) -> bool:
    """Check every intersection constraint for an arbitrary rejection set."""
    k = e_values.size
    if k > 20:
        raise ValueError("brute-force validation is restricted to K <= 20")
    if rejected.shape != (k,) or rejected.dtype != np.bool_:
        raise ValueError("rejected must be a Boolean vector matching e_values")
    r = int(rejected.sum())
    if r == 0:
        return True
    constants = constants_for_beta(k, alpha, beta)
    for mask in range(1, 1 << k):
        chosen = np.array([(mask >> i) & 1 for i in range(k)], dtype=bool)
        n = int(chosen.sum())
        ell = int(np.count_nonzero(chosen & rejected))
        merged = constants[n] + (1.0 - constants[n]) * float(e_values[chosen].mean())
        required = ell / (alpha * r)
        scale = max(1.0, abs(merged), required)
        if merged + TOLERANCE * scale < required:
            return False
    return True


def candidate_bruteforce(
    sorted_e: np.ndarray, r: int, alpha: float, beta: float
) -> bool:
    rejected = np.zeros(sorted_e.size, dtype=bool)
    rejected[:r] = True
    return candidate_set_bruteforce(sorted_e, rejected, alpha, beta)


def maximum_size_containing_ebh_bruteforce(
    sorted_e: np.ndarray, alpha: float, beta: float
) -> int:
    """Enumerate all eBH-containing sets; used only in small-K self-tests."""
    k = sorted_e.size
    if k > 12:
        raise ValueError("global brute-force validation is restricted to K <= 12")
    base_size = ebh_size(sorted_e, alpha)
    base_mask = (1 << base_size) - 1
    maximum = -1
    for mask in range(1 << k):
        if mask & base_mask != base_mask:
            continue
        size = mask.bit_count()
        if size <= maximum:
            continue
        rejected = np.array(
            [(mask >> i) & 1 for i in range(k)], dtype=bool
        )
        if candidate_set_bruteforce(sorted_e, rejected, alpha, beta):
            maximum = size
    if maximum < base_size:
        raise AssertionError("the eBH set was not certified")
    return maximum


def self_test() -> None:
    rng = np.random.default_rng(9_918_273)
    if format_beta(0.004) != "0.004":
        raise AssertionError("small beta display formatting failed")
    if ebh_size(np.array([3.0, 2.5]), 0.5) != 2:
        raise AssertionError("eBH must use the largest qualifying rank")
    if ebh_size(np.zeros(3), 0.05) != 0:
        raise AssertionError("empty eBH case failed")
    for k, alpha, beta in product(
        range(2, 10),
        (0.05, 0.10),
        DEFAULT_BETAS,
    ):
        if not constants_dominate_self_consistent(k, alpha, beta):
            raise AssertionError((k, alpha, beta, "eSC domination failed"))
        for _ in range(30):
            e = np.sort(rng.lognormal(0.2, 1.1, k))[::-1]
            for r in range(1, k + 1):
                fast = candidate_fast(e, r, alpha, beta)
                convex = candidate_convex(e, r, alpha, beta)
                brute = candidate_bruteforce(e, r, alpha, beta)
                if fast != brute or convex != brute:
                    raise AssertionError(
                        (k, alpha, beta, r, fast, convex, brute, e)
                    )

    # Independently enumerate all rejection sets on small problems.  This
    # verifies that the optimized top-prefix search attains the global maximum
    # cardinality among certified sets constrained to contain eBH.
    for k, alpha, beta in product(
        range(2, 8),
        (0.05, 0.10),
        DEFAULT_BETAS,
    ):
        for _ in range(2):
            target_size = int(rng.integers(1, k + 1))
            target_threshold = k / (alpha * target_size)
            high = target_threshold * rng.uniform(
                1.05, 2.0, size=target_size
            )
            low = rng.uniform(0.0, 0.5 / alpha, size=k - target_size)
            e = np.sort(np.concatenate((high, low)))[::-1]
            base_size = ebh_size(e, alpha)
            if base_size != target_size:
                raise AssertionError((k, alpha, target_size, base_size, e))
            optimized = largest_prefix(
                e, alpha, beta, minimum_size=base_size
            )
            brute = maximum_size_containing_ebh_bruteforce(e, alpha, beta)
            if optimized != brute:
                raise AssertionError(
                    (k, alpha, beta, base_size, optimized, brute, e)
                )

    if abs(admissibility_threshold(5000, 0.05) - 0.004) > 1e-12:
        raise AssertionError("beta-star calculation failed")


def largest_prefix(
    sorted_e: np.ndarray,
    alpha: float,
    beta: float,
    *,
    minimum_size: int = 0,
) -> int:
    """Return the global maximum certified size subject to a prefix floor.

    For the symmetric mergers used here, exchanging a selected smaller
    e-value with an unselected larger one preserves certification.  Therefore
    a certified set of any fixed size exists if and only if the top-e-value
    prefix of that size is certified.  When ``minimum_size`` is the eBH size,
    this returns the maximum cardinality among certified supersets of eBH.
    """
    k = sorted_e.size
    if not 0 <= minimum_size <= k:
        raise ValueError("minimum_size must lie between 0 and K")
    constants = constants_for_beta(k, alpha, beta)
    prefix = np.empty(k + 1)
    prefix[0] = 0.0
    np.cumsum(sorted_e, out=prefix[1:])

    r_grid = np.arange(1, k + 1, dtype=np.intp)
    required_r = 1.0 / alpha
    merged_r = constants[r_grid] + (1.0 - constants[r_grid]) * prefix[1:] / r_grid
    scale_r = np.maximum.reduce(
        (np.ones(k), np.abs(merged_r), np.full(k, required_r))
    )
    necessary_r = merged_r + TOLERANCE * scale_r >= required_r

    merged_one = constants[1] + (1.0 - constants[1]) * sorted_e
    required_one = 1.0 / (alpha * r_grid)
    scale_one = np.maximum.reduce(
        (np.ones(k), np.abs(merged_one), required_one)
    )
    necessary_one = (
        merged_one + TOLERANCE * scale_one >= required_one
    )

    # Strong necessary constraints from intersections obtained by deleting the
    # m strongest e-values. For a candidate of size r>m, such an intersection
    # contains r-m rejected hypotheses. Each m yields an explicit upper bound
    # on r; the cumulative minimum applies all bounds with m<r at once.
    m_grid = np.arange(k, dtype=np.intp)
    n_grid = k - m_grid
    # Accumulate the weakest entries directly. Subtracting neighboring large
    # prefix sums can erase the entire tail when e-values are highly dispersed.
    weakest_sum = np.cumsum(sorted_e[::-1])[::-1]
    weakest_merged = (
        constants[n_grid]
        + (1.0 - constants[n_grid]) * weakest_sum / n_grid
    )
    # Incorporate the same relative tolerance as the exact constraint.  Since
    # every required value here is at most 1/alpha, this inflation preserves a
    # genuinely necessary (never exclusionary) upper bound.
    tail_scale = np.maximum.reduce(
        (
            np.ones(k),
            np.abs(weakest_merged),
            np.full(k, 1.0 / alpha),
        )
    )
    weakest_merged_safe = weakest_merged + TOLERANCE * tail_scale
    denominator = 1.0 - alpha * weakest_merged_safe
    tail_bounds = np.full(k, np.inf)
    finite = denominator > 0.0
    tail_bounds[finite] = m_grid[finite] / denominator[finite]
    cumulative_tail_bound = np.minimum.accumulate(tail_bounds)
    necessary_tail = r_grid <= cumulative_tail_bound + 1e-10

    necessary = necessary_r & necessary_one & necessary_tail
    necessary &= r_grid >= minimum_size
    # The minimum is theoretically certified when it is the eBH size.  Keep it
    # in the exact checks even if floating-point pruning lands on a boundary.
    if minimum_size > 0:
        necessary[minimum_size - 1] = True
    candidates = r_grid[necessary]
    for r in candidates[::-1]:
        if candidate_convex(sorted_e, int(r), alpha, beta):
            return int(r)
    if minimum_size > 0:
        raise AssertionError("the required eBH prefix was not certified")
    return 0


def mean_and_se(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def simulate_scenario(
    normals: np.ndarray,
    *,
    k: int,
    nonnulls: int,
    signal: float,
    alpha: float,
    betas: tuple[float, ...],
) -> list[dict[str, float | int | str]]:
    replications = normals.shape[0]
    means = np.zeros(k)
    means[:nonnulls] = signal
    observations = normals + means
    try:
        with np.errstate(over="raise", invalid="raise"):
            e_values = np.exp(signal * observations - 0.5 * signal * signal)
    except FloatingPointError as exc:
        raise ValueError(
            "signal produced non-finite likelihood-ratio e-values; "
            "choose a smaller finite signal"
        ) from exc
    order = np.argsort(-e_values, axis=1, kind="stable")
    sorted_e = np.take_along_axis(e_values, order, axis=1)
    sorted_alt = order < nonnulls
    alternative_prefix = np.concatenate(
        (
            np.zeros((replications, 1), dtype=np.int16),
            np.cumsum(sorted_alt, axis=1),
        ),
        axis=1,
    )

    ebh_selected_size = np.fromiter(
        (ebh_size(sorted_e[j], alpha) for j in range(replications)),
        dtype=np.int16,
        count=replications,
    )
    ebh_true_positives = alternative_prefix[
        np.arange(replications), ebh_selected_size
    ]
    ebh_tpr = ebh_true_positives / nonnulls
    ebh_fdp = np.divide(
        ebh_selected_size - ebh_true_positives,
        ebh_selected_size,
        out=np.zeros(replications, dtype=float),
        where=ebh_selected_size > 0,
    )

    tpr_by_beta: list[np.ndarray] = []
    fdp_by_beta: list[np.ndarray] = []
    size_by_beta: list[np.ndarray] = []
    original_tpr_by_beta: list[np.ndarray] = []
    original_size_by_beta: list[np.ndarray] = []
    for beta in betas:
        if not constants_dominate_self_consistent(k, alpha, beta):
            raise AssertionError((k, alpha, beta, "eSC domination failed"))
        for j, base_size in enumerate(ebh_selected_size):
            if base_size > 0 and not candidate_convex(
                sorted_e[j], int(base_size), alpha, beta
            ):
                raise AssertionError(
                    (j, beta, int(base_size), "eBH prefix was not certified")
                )
        original_selected_size = np.fromiter(
            (
                largest_prefix(sorted_e[j], alpha, beta, minimum_size=0)
                for j in range(replications)
            ),
            dtype=np.int16,
            count=replications,
        )
        selected_size = np.fromiter(
            (
                largest_prefix(
                    sorted_e[j],
                    alpha,
                    beta,
                    minimum_size=int(ebh_selected_size[j]),
                )
                for j in range(replications)
            ),
            dtype=np.int16,
            count=replications,
        )
        if not np.array_equal(selected_size, original_selected_size):
            raise AssertionError((beta, "constrained selector changed the top prefix"))
        if np.any(selected_size < ebh_selected_size):
            raise AssertionError((beta, "reported set does not contain eBH"))
        true_positives = alternative_prefix[np.arange(replications), selected_size]
        original_true_positives = alternative_prefix[
            np.arange(replications), original_selected_size
        ]
        if np.any(true_positives < ebh_true_positives):
            raise AssertionError((beta, "true positives fell below eBH"))
        tpr = true_positives / nonnulls
        original_tpr = original_true_positives / nonnulls
        fdp = np.divide(
            selected_size - true_positives,
            selected_size,
            out=np.zeros(replications, dtype=float),
            where=selected_size > 0,
        )
        tpr_by_beta.append(tpr)
        fdp_by_beta.append(fdp)
        size_by_beta.append(selected_size)
        original_tpr_by_beta.append(original_tpr)
        original_size_by_beta.append(original_selected_size)
        # Cache entries cannot be reused across beta values and can otherwise
        # retain hundreds of MB in large-K experiments.
        constraint_cache.cache_clear()

    baseline_tpr = tpr_by_beta[0]
    baseline_size = size_by_beta[0]
    rows: list[dict[str, float | int | str]] = []
    beta_star = admissibility_threshold(k, alpha)

    def make_row(
        *,
        procedure: str,
        beta: float | None,
        status: str,
        tpr: np.ndarray,
        fdp: np.ndarray,
        selected_size: np.ndarray,
        original_tpr: np.ndarray,
        original_selected_size: np.ndarray,
    ) -> dict[str, float | int | str]:
        tpr_mean, tpr_se = mean_and_se(tpr)
        fdr, fdr_se = mean_and_se(fdp)
        difference_beta0_mean, difference_beta0_se = mean_and_se(
            tpr - baseline_tpr
        )
        difference_ebh_mean, difference_ebh_se = mean_and_se(tpr - ebh_tpr)
        rejection_gain = selected_size.astype(np.int32) - ebh_selected_size
        selector_size_change = (
            selected_size.astype(np.int32)
            - original_selected_size.astype(np.int32)
        )
        selector_tpr_change = tpr - original_tpr
        return {
            "procedure": procedure,
            "beta": "" if beta is None else beta,
            "beta_star": beta_star,
            "status": status,
            "status_scope": (
                "not applicable"
                if beta is None
                else "underlying simultaneous procedure"
            ),
            "tpr": tpr_mean,
            "tpr_se": tpr_se,
            "paired_tpr_difference_vs_beta0": difference_beta0_mean,
            "paired_difference_se": difference_beta0_se,
            "paired_tpr_difference_vs_ebh": difference_ebh_mean,
            "paired_difference_vs_ebh_se": difference_ebh_se,
            "empirical_fdr": fdr,
            "fdr_se": fdr_se,
            "mean_rejections": float(selected_size.mean()),
            "mean_rejection_gain_vs_ebh": float(rejection_gain.mean()),
            "minimum_rejection_gain_vs_ebh": int(rejection_gain.min()),
            "ebh_containment_rate": float(
                np.mean(selected_size >= ebh_selected_size)
            ),
            "selection_change_rate_vs_beta0": float(
                np.mean(selected_size != baseline_size)
            ),
            "selection_change_rate_vs_ebh": float(
                np.mean(selected_size != ebh_selected_size)
            ),
            "selector_set_change_count_vs_original": int(
                np.count_nonzero(selected_size != original_selected_size)
            ),
            "selector_set_change_rate_vs_original": float(
                np.mean(selected_size != original_selected_size)
            ),
            "maximum_absolute_rejection_count_change_vs_original": int(
                np.max(np.abs(selector_size_change))
            ),
            "mean_rejection_count_change_vs_original": float(
                selector_size_change.mean()
            ),
            "changed_true_positive_count_vs_original": int(
                np.count_nonzero(selector_tpr_change)
            ),
            "mean_tpr_change_vs_original": float(selector_tpr_change.mean()),
        }

    rows.append(
        make_row(
            procedure="eBH",
            beta=None,
            status="benchmark",
            tpr=ebh_tpr,
            fdp=ebh_fdp,
            selected_size=ebh_selected_size,
            original_tpr=ebh_tpr,
            original_selected_size=ebh_selected_size,
        )
    )
    for beta, tpr, fdp, selected_size, original_tpr, original_selected_size in zip(
        betas,
        tpr_by_beta,
        fdp_by_beta,
        size_by_beta,
        original_tpr_by_beta,
        original_size_by_beta,
        strict=True,
    ):
        rows.append(
            make_row(
                procedure="weighted-mean closed eBH",
                beta=beta,
                status=(
                    "admissible"
                    if beta <= beta_star
                    else "inadmissible"
                ),
                tpr=tpr,
                fdp=fdp,
                selected_size=selected_size,
                original_tpr=original_tpr,
                original_selected_size=original_selected_size,
            )
        )
    return rows


def write_csv(
    rows: list[dict[str, float | int | str]],
    output: Path,
    *,
    replications: int,
    k: int,
    alpha: float,
    signal: float,
    seed: int,
) -> None:
    enriched = [
        {
            **row,
            "replications": replications,
            "K": k,
            "alpha": alpha,
            "signal": signal,
            "seed": seed,
            "rng": "numpy.random.default_rng (PCG64)",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        }
        for row in rows
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)


def write_latex_table(
    rows: list[dict[str, float | int | str]],
    nonnull_counts: tuple[int, ...],
    output: Path,
    *,
    alpha: float,
    beta_star: float,
    replications: int,
) -> None:
    ebh_lookup = {
        int(row["nonnulls"]): row
        for row in rows
        if row["procedure"] == "eBH"
    }
    beta_rows = [row for row in rows if row["procedure"] != "eBH"]
    lookup = {
        (int(row["nonnulls"]), float(row["beta"])): row
        for row in beta_rows
    }
    betas = sorted({float(row["beta"]) for row in beta_rows})
    replications_tex = f"{replications:,}".replace(",", "{,}")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Empirical TPR of the base $\mathrm{eBH}_\alpha$ procedure and "
        r"the reported maximum-size rejection sets certified by "
        r"$\overline{\mathrm{eBH}}_{\alpha}^{\bm{\lambda}(\beta)}$ at "
        rf"$\alpha={alpha:g}$ based on "
        rf"${replications_tex}$ independent replications. Monte Carlo standard errors "
        r"are shown in parentheses. Admissibility labels concern the "
        r"underlying simultaneous procedures, not the reported point "
        r"selectors.}",
        r"\label{tab:constant-terms-tpr}",
        r"\footnotesize",
        r"\begingroup",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabularx}{\textwidth}{@{}X" + "c" * len(nonnull_counts) + "@{}}",
        r"\toprule",
        "& \\multicolumn{" + str(len(nonnull_counts))
        + r"}{c}{Number of non-nulls $|N^c|$} \\",
        r"\cmidrule(lr){2-" + str(len(nonnull_counts) + 1) + "}",
        r"& "
        + " & ".join(f"${count}$" for count in nonnull_counts)
        + r" \\",
        r"\midrule",
        r"\addlinespace[2pt]",
    ]
    ebh_cells = [r"$\mathrm{eBH}_\alpha$"]
    for count in nonnull_counts:
        row = ebh_lookup[count]
        ebh_cells.append(
            rf"{float(row['tpr']):.4f} ({float(row['tpr_se']):.6f})"
        )
    lines.append(" & ".join(ebh_cells) + r" \\")
    lines.append(r"\midrule")
    for status in ("admissible", "inadmissible"):
        relation = r"\leq" if status == "admissible" else ">"
        lines.append(
            r"\multicolumn{" + str(len(nonnull_counts) + 1)
            + r"}{l}{\emph{Choices with " + status
            + rf" underlying simultaneous procedures: $\beta {relation} "
            + rf"\beta^*={beta_star:g}$"
            + r"}}\\"
        )
        lines.append(r"\addlinespace[2pt]")
        for beta in betas:
            if (beta <= beta_star) != (status == "admissible"):
                continue
            label = (
                r"$\beta=0$ ($\overline{\mathrm{eBH}}_\alpha^{\mathsf{m}}$)"
                if beta == 0.0
                else (
                    rf"$\beta={format_beta(beta)}$"
                )
            )
            cells = [label]
            for count in nonnull_counts:
                row = lookup[(count, beta)]
                cells.append(
                    rf"{float(row['tpr']):.4f} "
                    rf"({float(row['tpr_se']):.6f})"
                )
            lines.append(" & ".join(cells) + r" \\")
        if status == "admissible":
            lines.append(r"\midrule")
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabularx}",
            r"\endgroup",
            r"\end{table}",
            "",
        )
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def write_selector_comparison_csv(
    rows: list[dict[str, float | int | str]],
    output: Path,
    *,
    replications: int,
) -> None:
    """Write a conspicuous audit of the old and revised selectors."""
    audit_rows: list[dict[str, float | int | str]] = []
    for row in rows:
        if row["procedure"] == "eBH":
            continue
        containment_rate = float(row["ebh_containment_rate"])
        audit_rows.append(
            {
                "nonnulls": int(row["nonnulls"]),
                "beta": float(row["beta"]),
                "status": str(row["status"]),
                "status_scope": str(row["status_scope"]),
                "replications": replications,
                "ebh_containment_rate": containment_rate,
                "ebh_containment_violation_count": int(
                    round(replications * (1.0 - containment_rate))
                ),
                "selector_set_change_count_vs_original": int(
                    row["selector_set_change_count_vs_original"]
                ),
                "selector_set_change_rate_vs_original": float(
                    row["selector_set_change_rate_vs_original"]
                ),
                "maximum_absolute_rejection_count_change_vs_original": int(
                    row[
                        "maximum_absolute_rejection_count_change_vs_original"
                    ]
                ),
                "mean_rejection_count_change_vs_original": float(
                    row["mean_rejection_count_change_vs_original"]
                ),
                "changed_true_positive_count_vs_original": int(
                    row["changed_true_positive_count_vs_original"]
                ),
                "mean_tpr_change_vs_original": float(
                    row["mean_tpr_change_vs_original"]
                ),
            }
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)


def write_complete_latex_table(
    rows: list[dict[str, float | int | str]],
    output: Path,
    *,
    alpha: float,
    replications: int,
) -> None:
    """Write all primary and diagnostic simulation summaries in one table."""
    replications_tex = f"{replications:,}".replace(",", "{,}")
    beta_replication_cases = replications * sum(
        row["procedure"] != "eBH" for row in rows
    )
    beta_replication_cases_tex = f"{beta_replication_cases:,}".replace(
        ",", "{,}"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete results for the admissible-constant-term simulation at "
        rf"$\alpha={alpha:g}$ based on ${replications_tex}$ paired replications. "
        r"Monte Carlo standard errors are in parentheses.  The TPR differences "
        r"and their standard errors are paired against the base "
        r"$\mathrm{eBH}_\alpha$ procedure.  Every reported certified set contained "
        r"the eBH set. The original and constrained searches had equal maximum "
        r"prefix sizes and, with their common fixed tie-break, selected the same "
        rf"sets in all ${beta_replication_cases_tex}$ beta-replication cases. "
        r"Status refers to the underlying simultaneous procedure, not the "
        r"reported point selector.}",
        r"\label{tab:constant-terms-complete}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{@{}rllrrrrrr@{}}",
        r"\toprule",
        r"$|N^c|$ & Procedure & Simultaneous status & TPR (MCSE) & FDR (MCSE) & "
        r"Mean $|R|$ & $\Delta$TPR (paired SE) & Mean $\Delta|R|$ & "
        r"Min. $\Delta|R|$ \\",
        r"\midrule",
    ]
    previous_nonnulls: int | None = None
    for row in rows:
        nonnulls = int(row["nonnulls"])
        if previous_nonnulls is not None and nonnulls != previous_nonnulls:
            lines.append(r"\addlinespace[3pt]")
        previous_nonnulls = nonnulls
        if row["procedure"] == "eBH":
            label = r"$\mathrm{eBH}_\alpha$"
        else:
            beta = float(row["beta"])
            label = (
                r"$\beta=0$ ($\overline{\mathrm{eBH}}_\alpha^{\mathsf m}$)"
                if beta == 0.0
                else rf"$\beta={format_beta(beta)}$"
            )
        lines.append(
            f"${nonnulls}$ & {label} & {row['status']} & "
            rf"${float(row['tpr']):.4f}\;({float(row['tpr_se']):.4f})$ & "
            rf"${float(row['empirical_fdr']):.4f}\;({float(row['fdr_se']):.4f})$ & "
            rf"${float(row['mean_rejections']):.3f}$ & "
            rf"${float(row['paired_tpr_difference_vs_ebh']):+.4f}\;"
            rf"({float(row['paired_difference_vs_ebh_se']):.4f})$ & "
            rf"${float(row['mean_rejection_gain_vs_ebh']):+.3f}$ & "
            rf"${int(row['minimum_rejection_gain_vs_ebh']):+d}$ \\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
            "",
        )
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def write_complete_markdown(
    rows: list[dict[str, float | int | str]],
    output: Path,
    *,
    alpha: float,
    replications: int,
) -> None:
    """Write a readable, complete companion to the machine-readable CSV."""
    beta_rows = [row for row in rows if row["procedure"] != "eBH"]
    total_beta_cases = replications * len(beta_rows)
    total_changes = sum(
        int(row["selector_set_change_count_vs_original"])
        for row in beta_rows
    )
    lines = [
        "# Complete simulation results",
        "",
        f"Nominal FDR level: `{alpha:g}`. Paired replications per scenario: "
        f"`{replications:,}`.",
        "",
        "Admissibility status refers to the underlying simultaneous "
        "weighted-mean closed-eBH procedure. It does not classify the "
        "reported point selector.",
        "",
        "The revised selector explicitly requires the reported maximum-size "
        "certified set to contain the eBH rejection set. The original and "
        "constrained searches had the same maximum prefix size. Because both "
        "use the same descending-e-value order and fixed original-index "
        "tie-break, they selected the same set. Every beta selection contained "
        f"eBH, and there were **{total_changes} changed sets among "
        f"{total_beta_cases:,} beta-replication cases**.",
        "",
        "TPR and FDR entries show Monte Carlo standard errors in parentheses. "
        "The TPR gain and its SE are paired against eBH.",
        "",
    ]
    for nonnulls in sorted({int(row["nonnulls"]) for row in rows}):
        lines.extend(
            (
                f"## {nonnulls} non-nulls",
                "",
                "| Procedure | Underlying simultaneous status | TPR (MCSE) | FDR (MCSE) | Mean R | "
                "TPR gain vs eBH (paired SE) | Mean / min R gain | "
                "eBH containment | Old/new changed sets |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in rows:
            if int(row["nonnulls"]) != nonnulls:
                continue
            if row["procedure"] == "eBH":
                label = "eBH"
                containment = "--"
                selector_changes = "--"
            else:
                label = f"beta={format_beta(float(row['beta']))}"
                containment = f"{100.0 * float(row['ebh_containment_rate']):.1f}%"
                selector_changes = (
                    f"{int(row['selector_set_change_count_vs_original'])}/"
                    f"{replications}"
                )
            lines.append(
                f"| {label} | {row['status']} | "
                f"{float(row['tpr']):.6f} ({float(row['tpr_se']):.6f}) | "
                f"{float(row['empirical_fdr']):.6f} "
                f"({float(row['fdr_se']):.6f}) | "
                f"{float(row['mean_rejections']):.3f} | "
                f"{float(row['paired_tpr_difference_vs_ebh']):+.6f} "
                f"({float(row['paired_difference_vs_ebh_se']):.6f}) | "
                f"{float(row['mean_rejection_gain_vs_ebh']):+.3f} / "
                f"{int(row['minimum_rejection_gain_vs_ebh']):+d} | "
                f"{containment} | {selector_changes} |"
            )
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=1_000)
    parser.add_argument("--hypotheses", type=int, default=5_000)
    parser.add_argument(
        "--nonnulls", type=parse_int_tuple, default=(500, 1_000, 1_500)
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--signal", type=float, default=3.0)
    parser.add_argument("--betas", type=parse_float_tuple, default=DEFAULT_BETAS)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument(
        "--self-test-only", action="store_true",
        help="run exhaustive algorithm checks and exit without simulating",
    )
    args = parser.parse_args()

    if args.replications <= 1:
        parser.error("replications must exceed one")
    if args.hypotheses < 2:
        parser.error("hypotheses must be at least two")
    if args.hypotheses > np.iinfo(np.int16).max:
        parser.error("hypotheses must not exceed 32767")
    if len(set(args.nonnulls)) != len(args.nonnulls):
        parser.error("nonnull counts must be distinct")
    if any(count <= 0 or count >= 0.4 * args.hypotheses for count in args.nonnulls):
        parser.error("every non-null count must be positive and strictly below 40% of K")
    if not 0.0 < args.alpha < 1.0:
        parser.error("alpha must lie in (0,1)")
    if not math.isfinite(args.signal) or args.signal <= 0.0:
        parser.error("signal must be positive and finite")
    if args.seed < 0:
        parser.error("seed must be a nonnegative integer")
    if len(set(args.betas)) != len(args.betas) or any(not 0.0 <= b <= 1.0 for b in args.betas):
        parser.error("betas must be distinct values in [0,1]")
    if not args.betas or args.betas[0] != 0.0:
        parser.error("the first beta must be 0 for paired baseline comparisons")
    if args.self_test_only and args.skip_self_test:
        parser.error("--self-test-only cannot be combined with --skip-self-test")

    if not args.skip_self_test:
        self_test()
        print(
            "Self-test passed: optimized certification and constrained "
            "selection equal exhaustive checks."
        )
    if args.self_test_only:
        return

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    common_normals = rng.standard_normal((args.replications, args.hypotheses))
    combined_rows: list[dict[str, float | int | str]] = []
    for nonnulls in args.nonnulls:
        print(f"Simulating {nonnulls} non-nulls ({args.replications} replications)...", flush=True)
        scenario_rows = simulate_scenario(
            common_normals,
            k=args.hypotheses,
            nonnulls=nonnulls,
            signal=args.signal,
            alpha=args.alpha,
            betas=args.betas,
        )
        for row in scenario_rows:
            row["nonnulls"] = nonnulls
            row["nonnull_proportion"] = nonnulls / args.hypotheses
        combined_rows.extend(scenario_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "constant_terms_results.csv"
    table_path = args.output_dir / "constant_terms_tpr_table.tex"
    complete_table_path = args.output_dir / "constant_terms_complete_table.tex"
    complete_markdown_path = (
        args.output_dir / "constant_terms_complete_results.md"
    )
    selector_audit_path = args.output_dir / "selector_comparison.csv"
    write_csv(
        combined_rows,
        csv_path,
        replications=args.replications,
        k=args.hypotheses,
        alpha=args.alpha,
        signal=args.signal,
        seed=args.seed,
    )
    write_latex_table(
        combined_rows,
        args.nonnulls,
        table_path,
        alpha=args.alpha,
        beta_star=admissibility_threshold(args.hypotheses, args.alpha),
        replications=args.replications,
    )
    write_complete_latex_table(
        combined_rows,
        complete_table_path,
        alpha=args.alpha,
        replications=args.replications,
    )
    write_complete_markdown(
        combined_rows,
        complete_markdown_path,
        alpha=args.alpha,
        replications=args.replications,
    )
    write_selector_comparison_csv(
        combined_rows,
        selector_audit_path,
        replications=args.replications,
    )
    metadata = {
        "section": "7.2",
        "replications": args.replications,
        "hypotheses": args.hypotheses,
        "nonnulls": args.nonnulls,
        "alpha": args.alpha,
        "signal": args.signal,
        "betas": args.betas,
        "seed": args.seed,
        "rng": "numpy.random.default_rng (PCG64)",
        "common_random_numbers_across_scenarios": True,
        "certification_relative_tolerance": TOLERANCE,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": sys.argv[1:],
        "self_test_run": not args.skip_self_test,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "nonnulls procedure  simultaneous status  TPR (MCSE)       "
        "paired difference vs eBH (SE)"
    )
    for row in combined_rows:
        procedure = (
            "eBH      "
            if row["procedure"] == "eBH"
            else f"beta={format_beta(float(row['beta']))}"
        )
        print(
            f"{int(row['nonnulls']):8d} {procedure:9s} "
            f"{str(row['status']):12s} "
            f"{float(row['tpr']):.4f} ({float(row['tpr_se']):.4f})   "
            f"{float(row['paired_tpr_difference_vs_ebh']):+.4f} "
            f"({float(row['paired_difference_vs_ebh_se']):.4f})"
        )
    print(f"CSV:   {csv_path}")
    print(f"Table: {table_path}")
    print(f"Complete table: {complete_table_path}")
    print(f"Complete Markdown: {complete_markdown_path}")
    print(f"Selector audit: {selector_audit_path}")


if __name__ == "__main__":
    main()

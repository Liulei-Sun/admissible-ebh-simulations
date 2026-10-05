#!/usr/bin/env python3
"""Learned asymmetric closed eBH 2 for the requested fixed-floor formula.

This runner deliberately reuses the audited outer cardinality optimizer and
checkpointing code from ``optimize_choice2_learned.py``.  Only the merger
oracle, problem hash, and method labels are replaced.  As in the Choice-2
runner, ordinary eBH is fixed as a required subset and no greedy coordinates
are fixed, so the result is a genuine maximum-cardinality certified superset
    of eBH rather than merely a maximal extension of a greedy set.  The shared
    optimizer then maximizes the testing-e-value sum at that cardinality.
"""

from __future__ import annotations

import hashlib
import shutil

import numpy as np

import optimize_choice2_learned as runner
from original_formula_oracle import OriginalFormulaOracle


def make_original_oracle(e,
                         score,
                         alpha,
                         separator_time_limit,
                         separator_threads):
    oracle = OriginalFormulaOracle(
        e,
        score,
        alpha,
        separator_time_limit=separator_time_limit,
        separator_threads=separator_threads,
    )
    # Fast fixed-size rules generate candidate witnesses only.  Every such
    # intersection is re-evaluated under the exact original formula, and only
    # exact dyadic prefix enumeration or the globally solved HiGHS model may
    # certify a candidate.
    calibrated = runner.SIZE_CALIBRATED_ORACLE(e, score, alpha)
    natural = runner.SIZE_CALIBRATED_ORACLE(e, score, alpha)
    for size in range(2, natural.k):
        natural.raw_weights[size] = natural.score
    if natural.k == 2:
        natural.pair_merger = np.full((2, 2), float(np.mean(natural.e)))
    else:
        pair_weight = natural.score
        numerator = np.outer(pair_weight * natural.e, np.ones(natural.k))
        numerator += numerator.T
        natural.pair_merger = numerator / np.add.outer(pair_weight, pair_weight)
    oracle.surrogate_oracles = [calibrated, natural]
    return oracle


def original_problem_hash(instance, alpha: float) -> str:
    digest = hashlib.sha256()
    digest.update(b"closed-ebh-original-prior-floor-maxcard-maxesum-v2\0")
    digest.update(np.asarray([alpha, float(instance.e.size)], dtype="<f8").tobytes())
    digest.update(np.asarray(instance.e, dtype="<f8").tobytes())
    digest.update(np.asarray(instance.score, dtype="<f8").tobytes())
    digest.update(np.asarray(instance.nonnull, dtype=np.uint8).tobytes())
    digest.update(np.asarray(instance.prefix, dtype=np.uint8).tobytes())
    return digest.hexdigest()


_base_optimize_one = runner.optimize_one
_base_write_products = runner.write_products
runner.make_choice2_oracle = make_original_oracle
runner.choice2_problem_hash = original_problem_hash


def optimize_one(payload):
    row = _base_optimize_one(payload)
    row["method"] = "original_uniform_floor_plus_prior_scores"
    row["selection_rule"] = (
        "maximum_cardinality_certified_superset_of_eBH_then_maximum_testing_evalue_sum"
    )
    row["separator_backend"] = "exact_dyadic_prefix_or_HiGHS_QUBO_MILP"
    return row


runner.optimize_one = optimize_one


def write_products(results, output_dir, reference_results):
    _base_write_products(results, output_dir, reference_results)
    for source_name, target_name in (
            ("choice2_results.csv", "original_formula_results.csv"),
            ("choice2_summary.json", "original_formula_summary.json"),
            ("choice2_vs_current_paired.csv",
             "original_formula_vs_reference_paired.csv")):
        source = output_dir / source_name
        if source.exists():
            shutil.copyfile(source, output_dir / target_name)
    memberships_path = output_dir / "choice2_memberships.csv"
    if memberships_path.exists():
        memberships = runner.pd.read_csv(memberships_path).rename(columns={
            "choice2_member": "original_formula_member"
        })
        runner.atomic_write_csv(
            memberships, output_dir / "original_formula_memberships.csv"
        )


runner.write_products = write_products


if __name__ == "__main__":
    raise SystemExit(runner.main())

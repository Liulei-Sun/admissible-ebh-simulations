#!/usr/bin/env python3
"""Method 1: maximize size, then testing-e-value sum, without an eBH seed."""

from __future__ import annotations

import hashlib
import shutil

import numpy as np

import optimize_choice2_learned as runner
from method1_oracle import Method1Oracle


def make_method1_oracle(e,
                        score,
                        alpha,
                        separator_time_limit,
                        separator_threads):
    del separator_time_limit, separator_threads
    return Method1Oracle(e, score, alpha)


def empty_seed(e: np.ndarray, alpha: float) -> np.ndarray:
    del alpha
    return np.zeros(np.asarray(e).size, dtype=bool)


def method1_problem_hash(instance, alpha: float) -> str:
    digest = hashlib.sha256()
    digest.update(b"closed-ebh-method1-proportional-maxcard-maxesum-v2\0")
    digest.update(np.asarray([alpha, float(instance.e.size)], dtype="<f8").tobytes())
    digest.update(np.asarray(instance.e, dtype="<f8").tobytes())
    digest.update(np.asarray(instance.score, dtype="<f8").tobytes())
    digest.update(np.asarray(instance.nonnull, dtype=np.uint8).tobytes())
    digest.update(np.asarray(instance.prefix, dtype=np.uint8).tobytes())
    return digest.hexdigest()


_base_optimize_one = runner.optimize_one
_base_write_products = runner.write_products
runner.make_choice2_oracle = make_method1_oracle
runner.choice2_problem_hash = method1_problem_hash
runner.ordinary_ebh = empty_seed


def optimize_one(payload):
    row = _base_optimize_one(payload)
    row["method"] = "method1_proportional_scores_gamma1"
    row["selection_rule"] = (
        "unrestricted_maximum_cardinality_then_maximum_testing_evalue_sum"
    )
    row["separator_backend"] = "exact_fixed_count_order_statistics"
    row["ordinary_eBH_required"] = False
    return row


runner.optimize_one = optimize_one


def write_products(results, output_dir, reference_results):
    _base_write_products(results, output_dir, reference_results)
    for source_name, target_name in (
            ("choice2_results.csv", "method1_results.csv"),
            ("choice2_summary.json", "method1_summary.json"),
            ("choice2_vs_current_paired.csv", "method1_vs_reference_paired.csv")):
        source = output_dir / source_name
        if source.exists():
            shutil.copyfile(source, output_dir / target_name)
    memberships_path = output_dir / "choice2_memberships.csv"
    if memberships_path.exists():
        memberships = runner.pd.read_csv(memberships_path).rename(columns={
            "choice2_member": "method1_member"
        })
        runner.atomic_write_csv(
            memberships, output_dir / "method1_memberships.csv"
        )


runner.write_products = write_products


if __name__ == "__main__":
    raise SystemExit(runner.main())

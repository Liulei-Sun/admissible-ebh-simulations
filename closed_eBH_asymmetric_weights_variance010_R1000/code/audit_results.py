#!/usr/bin/env python3
"""Independently audit all reported simulation inputs and results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from study_common import (
    CONDITION_ORDER,
    METHOD_LABELS,
    METHOD_ORDER,
    PERSISTENT_NOISE_SALT,
    SIGNED_SQUARE_RESIDUAL_SALT,
    atomic_csv,
    atomic_json,
    encode_indices,
    format4,
    largest_mean_set,
    load_config,
    mean_mcse,
    ordinary_ebh,
    parse_indices,
    problem_hash,
    reconstruct_training_observations,
    require,
    selection_metrics,
    sha256_file,
    splitmix64,
    stable_scores,
)


Z_975 = 1.959963984540054
METRIC_COLUMNS = [
    "condition", "nonnulls", "method", "method_label", "batch_id",
    "batch_seed", "local_replication", "global_replication", "rejections",
    "true_discoveries", "false_discoveries", "TPR", "FDP",
    "selected_indices", "source_result_file", "source_result_sha256",
]
CONTAINMENT_COLUMNS = [
    "condition", "nonnulls", "batch_id", "batch_seed", "local_replication",
    "global_replication", "ebh_size", "ebh_indices", "procedure1_size",
    "procedure1_indices", "procedure1_contains_ebh", "procedure2_size",
    "procedure2_indices", "procedure2_contains_ebh",
]
_HASH_CACHE: dict[Path, str] = {}


def cached_sha256(path: Path) -> str:
    resolved = path.resolve()
    if resolved not in _HASH_CACHE:
        _HASH_CACHE[resolved] = sha256_file(resolved)
    return _HASH_CACHE[resolved]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Independently recertify every learned reported set.",
    )
    return parser.parse_args()


def scaled_error(observed: np.ndarray, expected: np.ndarray) -> float:
    a = np.asarray(observed, dtype=float)
    b = np.asarray(expected, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)), initial=0.0))


def close_array(
    observed: np.ndarray,
    expected: np.ndarray,
    label: str,
    errors: dict[str, float],
    tolerance: float = 5e-13,
) -> None:
    error = scaled_error(observed, expected)
    errors[label] = max(errors.get(label, 0.0), error)
    require(error <= tolerance, f"{label}: scaled error {error:.3g} exceeds tolerance")


def check_metrics(row: pd.Series, metrics: dict[str, Any], prefix: str = "") -> None:
    mapping = {
        "rejections": f"{prefix}size" if prefix else "rejections",
        "true_discoveries": f"{prefix}true" if prefix else "true_discoveries",
        "false_discoveries": f"{prefix}false" if prefix else "false_discoveries",
    }
    rates = {
        "TPR": f"{prefix}tpr" if prefix else "tpr",
        "FDP": f"{prefix}fdp" if prefix else "fdp",
    }
    for key, column in mapping.items():
        require(int(row[column]) == int(metrics[key]), f"stored {column} mismatch")
    for key, column in rates.items():
        require(
            math.isclose(float(row[column]), float(metrics[key]), rel_tol=0.0, abs_tol=2e-14),
            f"stored {column} mismatch",
        )


def validate_seed_table(
    path: Path, batch: dict[str, Any], errors: dict[str, float]
) -> dict[int, int]:
    frame = pd.read_csv(path)
    require(
        list(frame.columns)
        == [
            "local_replication", "global_replication",
            "primary_replication_seed", "persistent_noise_seed",
        ],
        f"seed schema mismatch: {path}",
    )
    require(frame.shape[0] == 500, f"seed row count mismatch: {path}")
    persistent: dict[int, int] = {}
    for row in frame.itertuples(index=False):
        replication = int(row.local_replication)
        expected_primary = splitmix64(int(batch["seed"]) ^ splitmix64(replication))
        expected_persistent = splitmix64(expected_primary ^ PERSISTENT_NOISE_SALT)
        require(
            int(row.global_replication)
            == int(batch["global_replication_start"]) + replication - 1,
            "global replication map mismatch",
        )
        require(int(row.primary_replication_seed) == expected_primary, "primary seed mismatch")
        require(
            int(row.persistent_noise_seed) == expected_persistent,
            "persistent seed mismatch",
        )
        persistent[replication] = expected_persistent
    errors["seed_identity"] = 0.0
    return persistent


def validate_panel_inputs(
    panel_root: Path,
    batch: dict[str, Any],
    nonnulls: int,
    config: dict[str, Any],
    persistent_seeds: dict[int, int],
    errors: dict[str, float],
) -> dict[str, pd.DataFrame]:
    inputs = {
        condition: pd.read_csv(panel_root / condition / "learned_instances.csv")
        for condition in CONDITION_ORDER
    }
    expected_columns = {
        "clean": {
            "replication", "index", "e", "score", "nonnull", "prefix_size",
            "prefix_member", "train_sum", "test_sum", "e_full", "scenario",
        },
        "signed_square": {
            "replication", "index", "e", "score", "nonnull", "prefix_size",
            "prefix_member", "train_sum", "test_sum", "contaminated_train_sum",
            "e_naive_full", "score_tilt", "scenario", "parameter", "severity",
        },
        "persistent_noise": {
            "replication", "index", "e", "score", "nonnull", "prefix_size",
            "prefix_member", "train_sum", "test_sum", "persistent_standard_normal",
            "per_observation_shift", "aggregate_training_shift",
            "contaminated_train_sum", "e_naive_full", "score_tilt", "scenario",
            "parameter", "severity",
        },
    }
    for condition, frame in inputs.items():
        require(set(frame.columns) == expected_columns[condition], f"{condition} input schema mismatch")
        require(frame.shape[0] == 100000, f"{condition} input row count mismatch")
        require(not frame.duplicated(["replication", "index"]).any(), "duplicate input key")
        require(set(frame["replication"].astype(int)) == set(range(1, 501)), "input replication IDs mismatch")
        require(set(frame["index"].astype(int)) == set(range(200)), "input coordinate IDs mismatch")
        require((frame["scenario"].astype(str) == condition).all(), "condition label mismatch")
        inputs[condition] = frame.sort_values(["replication", "index"]).reset_index(drop=True)

    shared_integer = ["replication", "index", "nonnull", "prefix_size", "prefix_member"]
    shared_float = ["e", "train_sum", "test_sum"]
    clean = inputs["clean"]
    for condition in ("signed_square", "persistent_noise"):
        other = inputs[condition]
        require(
            np.array_equal(
                clean[shared_integer].to_numpy(dtype=np.int64),
                other[shared_integer].to_numpy(dtype=np.int64),
            ),
            f"{condition}: frozen integer fields changed",
        )
        for column in shared_float:
            close_array(
                other[column].to_numpy(dtype=float),
                clean[column].to_numpy(dtype=float),
                f"{condition}.shared.{column}",
                errors,
            )

    alpha = float(config["alpha"])
    eta = float(config["e_value_tilt"])
    mu = float(config["signal_mean"])
    n_train = int(config["training_observations"])
    n_test = int(config["testing_observations"])
    test_offset = 0.5 * n_test * eta * eta
    full_offset = 0.5 * (n_train + n_test) * eta * eta
    error_sd = math.sqrt(0.1)

    grouped = {
        condition: {
            int(rep): group.sort_values("index")
            for rep, group in frame.groupby("replication", sort=True)
        }
        for condition, frame in inputs.items()
    }
    for replication in range(1, 501):
        primary = splitmix64(int(batch["seed"]) ^ splitmix64(replication))
        rng = np.random.Generator(np.random.PCG64(primary))
        labels = rng.permutation(200)
        truth = np.zeros(200, dtype=bool)
        truth[labels[:nonnulls]] = True
        gaussian = rng.standard_normal((200, 2))
        train = n_train * mu * truth + math.sqrt(n_train) * gaussian[:, 0]
        test = n_test * mu * truth + math.sqrt(n_test) * gaussian[:, 1]
        e = np.exp(eta * test - test_offset)
        ebh = ordinary_ebh(e, alpha)
        clean_group = grouped["clean"][replication]
        require(np.array_equal(clean_group["index"].to_numpy(dtype=int), np.arange(200)), "input order mismatch")
        require(np.array_equal(clean_group["nonnull"].to_numpy(dtype=bool), truth), "truth regeneration mismatch")
        require(np.array_equal(clean_group["prefix_member"].to_numpy(dtype=bool), ebh), "eBH prefix mismatch")
        require((clean_group["prefix_size"].astype(int) == int(np.sum(ebh))).all(), "eBH prefix size mismatch")
        close_array(clean_group["train_sum"], train, "clean.train_sum", errors)
        close_array(clean_group["test_sum"], test, "clean.test_sum", errors)
        close_array(clean_group["e"], e, "clean.testing_e", errors)
        close_array(clean_group["score"], stable_scores(train, eta), "clean.score", errors)
        close_array(
            clean_group["e_full"],
            np.exp(eta * (train + test) - full_offset),
            "clean.full_e",
            errors,
        )

        residual_seed = splitmix64(primary ^ SIGNED_SQUARE_RESIDUAL_SALT)
        observations = reconstruct_training_observations(train, n_train, residual_seed)
        signed_sum = np.sum(np.sign(observations) * np.square(np.abs(observations)), axis=1)
        signed_group = grouped["signed_square"][replication]
        close_array(signed_group["contaminated_train_sum"], signed_sum, "signed_square.sum", errors)
        close_array(signed_group["score"], stable_scores(signed_sum, eta), "signed_square.score", errors)
        close_array(
            signed_group["e_naive_full"],
            np.exp(eta * (signed_sum + test) - full_offset),
            "signed_square.naive_e",
            errors,
        )
        require(
            (signed_group["parameter"].astype(str) == "gamma").all()
            and np.allclose(signed_group["severity"], 1.0)
            and np.allclose(signed_group["score_tilt"], eta),
            "signed-square metadata mismatch",
        )

        persistent_seed = persistent_seeds[replication]
        z = np.random.Generator(np.random.PCG64(persistent_seed)).standard_normal(200)
        error = error_sd * z
        persistent_sum = train + n_train * error
        persistent_group = grouped["persistent_noise"][replication]
        close_array(persistent_group["persistent_standard_normal"], z, "persistent_noise.z", errors)
        close_array(persistent_group["per_observation_shift"], error, "persistent_noise.error", errors)
        close_array(
            persistent_group["aggregate_training_shift"],
            n_train * error,
            "persistent_noise.aggregate_shift",
            errors,
        )
        close_array(
            persistent_group["contaminated_train_sum"],
            persistent_sum,
            "persistent_noise.sum",
            errors,
        )
        close_array(
            persistent_group["score"],
            stable_scores(persistent_sum, eta),
            "persistent_noise.score",
            errors,
        )
        close_array(
            persistent_group["e_naive_full"],
            np.exp(eta * (persistent_sum + test) - full_offset),
            "persistent_noise.naive_e",
            errors,
        )
        require(
            (persistent_group["parameter"].astype(str) == "variance").all()
            and np.allclose(persistent_group["severity"], 0.1)
            and np.allclose(persistent_group["score_tilt"], eta),
            "persistent-error metadata mismatch",
        )
    return inputs


def audit_direct(
    path: Path,
    expected: dict[int, np.ndarray],
    truth: dict[int, np.ndarray],
) -> dict[int, dict[str, Any]]:
    frame = pd.read_csv(path)
    require(frame.shape[0] == 500, f"direct result row count mismatch: {path}")
    require(set(frame["replication"].astype(int)) == set(range(1, 501)), "direct result IDs mismatch")
    output = {}
    for _, row in frame.iterrows():
        replication = int(row["replication"])
        selected = parse_indices(row.get("selected_indices", ""))
        expected_set = set(int(i) for i in np.flatnonzero(expected[replication]))
        require(selected == expected_set, f"direct selected set mismatch: {path}/rep{replication}")
        require(str(row["status"]) == "proven_maximum_cardinality", "direct status mismatch")
        metrics = selection_metrics(selected, truth[replication])
        check_metrics(row, metrics)
        output[replication] = {"selected": selected, "metrics": metrics}
    return output


def exact_adjusted_certified(
    e: np.ndarray,
    score: np.ndarray,
    alpha: float,
    selected: np.ndarray,
    tolerance: float = 1e-10,
) -> bool:
    """Exact dyadic outside-prefix certificate for every reported size here."""
    values = np.asarray(e, dtype=float)
    scores = np.asarray(score, dtype=float)
    chosen = np.asarray(selected, dtype=bool)
    inside = np.flatnonzero(chosen)
    r = int(inside.size)
    require(r <= 23, "publication adjusted-set size exceeds exact audit cap")
    if r == 0:
        return True
    outside = np.flatnonzero(~chosen)
    if outside.size:
        outside = outside[np.lexsort((outside, values[outside]))]

    def common_dyadic(array: np.ndarray) -> tuple[list[int], int]:
        ratios = [float(value).as_integer_ratio() for value in array]
        powers = [denominator.bit_length() - 1 for _, denominator in ratios]
        common_power = max(powers, default=0)
        return [
            numerator << (common_power - power)
            for (numerator, _), power in zip(ratios, powers)
        ], common_power

    e_integer, e_power = common_dyadic(values)
    score_integer, _ = common_dyadic(scores)
    prefix_e = [0]
    prefix_score = [0]
    prefix_score_e = [0]
    outside_e = []
    for raw_index in outside:
        index = int(raw_index)
        local_e = e_integer[index]
        local_score = score_integer[index]
        outside_e.append(local_e)
        prefix_e.append(prefix_e[-1] + local_e)
        prefix_score.append(prefix_score[-1] + local_score)
        prefix_score_e.append(prefix_score_e[-1] + local_score * local_e)
    alpha_num, alpha_den = float(alpha).as_integer_ratio()
    tol_num, tol_den = float(tolerance).as_integer_ratio()
    e_den = 1 << e_power
    subset_e = subset_score = subset_score_e = subset_size = previous_gray = 0
    for step in range(1, 1 << r):
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
        low, high = 0, len(outside_e)
        while low < high:
            middle = (low + high) // 2
            if (
                outside_e[middle] * (subset_score + prefix_score[middle])
                >= subset_score_e + prefix_score_e[middle]
            ):
                high = middle
            else:
                low = middle + 1
        length = low
        size = subset_size + length
        e_sum = subset_e + prefix_e[length]
        score_sum = subset_score + prefix_score[length]
        score_e_sum = subset_score_e + prefix_score_e[length]
        merger_num = e_sum * score_sum + (values.size - size) * score_e_sum
        merger_den = values.size * e_den * score_sum
        violation_num = (
            subset_size * alpha_den * merger_den - alpha_num * r * merger_num
        )
        violation_den = alpha_den * merger_den
        if violation_num * tol_den > tol_num * violation_den:
            return False
    return True


def audit_learned(
    condition_root: Path,
    instances: pd.DataFrame,
    method: str,
    alpha: float,
    full: bool,
    method1_class: Any,
) -> tuple[dict[int, dict[str, Any]], int]:
    result_path = condition_root / f"{method}_results.csv"
    membership_path = condition_root / f"{method}_membership.csv"
    results = pd.read_csv(result_path)
    memberships = pd.read_csv(membership_path)
    require(results.shape[0] == 500, f"learned result count mismatch: {result_path}")
    require(set(results["replication"].astype(int)) == set(range(1, 501)), "learned result IDs mismatch")
    member_column = "method1_member" if method == "procedure1" else "original_formula_member"
    require(
        set(memberships.columns) == {"replication", "index", "ebh_member", member_column},
        f"membership schema mismatch: {membership_path}",
    )
    require(memberships.shape[0] == 100000, "membership row count mismatch")
    require(not memberships.duplicated(["replication", "index"]).any(), "duplicate membership key")
    grouped_input = {
        int(rep): group.sort_values("index")
        for rep, group in instances.groupby("replication", sort=True)
    }
    grouped_membership = {
        int(rep): group.sort_values("index")
        for rep, group in memberships.groupby("replication", sort=True)
    }
    expected_method = (
        "method1_proportional_scores_gamma1"
        if method == "procedure1"
        else "original_uniform_floor_plus_prior_scores"
    )
    expected_rule = (
        "unrestricted_maximum_cardinality_then_maximum_testing_evalue_sum"
        if method == "procedure1"
        else "maximum_cardinality_certified_superset_of_eBH_then_maximum_testing_evalue_sum"
    )
    output = {}
    recertified = 0
    for _, row in results.sort_values("replication").iterrows():
        replication = int(row["replication"])
        group = grouped_input[replication]
        membership = grouped_membership[replication]
        require(np.array_equal(group["index"].to_numpy(dtype=int), np.arange(200)), "learned input order mismatch")
        require(np.array_equal(membership["index"].to_numpy(dtype=int), np.arange(200)), "membership order mismatch")
        e = group["e"].to_numpy(dtype=float)
        score = group["score"].to_numpy(dtype=float)
        truth = group["nonnull"].to_numpy(dtype=bool)
        ebh = group["prefix_member"].to_numpy(dtype=bool)
        prefix = np.zeros(200, dtype=bool) if method == "procedure1" else ebh
        require(str(row["status"]) == "proven_maximal", "cardinality proof status mismatch")
        require(str(row["secondary_status"]) == "proven_optimal", "secondary proof status mismatch")
        require(
            str(row["secondary_rule"]) == "maximize_sum_testing_e_values_at_maximum_cardinality",
            "secondary rule mismatch",
        )
        require(int(row["proof_upper_bound"]) == int(row["maximal_size"]), "open proof bound")
        require(int(row["K"]) == 200, "learned K mismatch")
        require(str(row["method"]) == expected_method, "learned method identity mismatch")
        require(str(row["selection_rule"]) == expected_rule, "learned selection rule mismatch")
        require(
            str(row["problem_sha256"])
            == problem_hash(e, score, truth, prefix, alpha, method),
            "learned problem hash mismatch",
        )
        selected = parse_indices(row.get("maximal_indices", ""))
        require(len(selected) == int(row["maximal_size"]), "learned size mismatch")
        selected_mask = np.zeros(200, dtype=bool)
        if selected:
            selected_mask[np.asarray(sorted(selected), dtype=int)] = True
        require(
            np.array_equal(membership[member_column].to_numpy(dtype=bool), selected_mask),
            "learned membership/result mismatch",
        )
        if method == "procedure1":
            require(not membership["ebh_member"].to_numpy(dtype=bool).any(), "Procedure 1 has a seed")
        else:
            require(np.array_equal(membership["ebh_member"].to_numpy(dtype=bool), ebh), "Procedure 2 seed mismatch")
            require(set(np.flatnonzero(ebh)).issubset(selected), "Procedure 2 omits testing eBH")
        prefix_set = set(int(i) for i in np.flatnonzero(prefix))
        require(parse_indices(row.get("prefix_indices", "")) == prefix_set, "stored prefix mismatch")
        require(int(row["prefix_size"]) == len(prefix_set), "stored prefix size mismatch")
        secondary = float(np.sum(e[selected_mask], dtype=np.longdouble))
        require(
            math.isclose(float(row["secondary_value"]), secondary, rel_tol=2e-12, abs_tol=2e-10),
            "secondary objective value mismatch",
        )
        metrics = selection_metrics(selected, truth)
        check_metrics(row, metrics, prefix="maximal_")
        if full:
            certified = (
                method1_class(e, score, alpha).is_certified(selected_mask)
                if method == "procedure1"
                else exact_adjusted_certified(e, score, alpha, selected_mask)
            )
            require(certified, f"independent certification failed: {result_path}/rep{replication}")
            recertified += 1
        output[replication] = {
            "selected": selected,
            "metrics": metrics,
            "ebh": set(int(i) for i in np.flatnonzero(ebh)),
        }
    return output, recertified


def metric_record(
    condition: str,
    nonnulls: int,
    method: str,
    batch: dict[str, Any],
    replication: int,
    selected: set[int],
    metrics: dict[str, Any],
    source: Path,
    root: Path,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "nonnulls": nonnulls,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "batch_id": batch["id"],
        "batch_seed": batch["seed"],
        "local_replication": replication,
        "global_replication": int(batch["global_replication_start"]) + replication - 1,
        **metrics,
        "selected_indices": encode_indices(selected),
        "source_result_file": source.relative_to(root).as_posix(),
        "source_result_sha256": cached_sha256(source),
    }


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, nonnulls, method, label), group in metrics.groupby(
        ["condition", "nonnulls", "method", "method_label"], sort=False
    ):
        tpr, tpr_mcse = mean_mcse(group["TPR"].to_numpy(dtype=float))
        fdr, fdr_mcse = mean_mcse(group["FDP"].to_numpy(dtype=float))
        rows.append(
            {
                "condition": condition,
                "nonnulls": int(nonnulls),
                "method": method,
                "method_label": label,
                "TPR": tpr,
                "TPR_MCSE": tpr_mcse,
                "FDR": fdr,
                "FDR_MCSE": fdr_mcse,
                "FDR_MC95_low": fdr - Z_975 * fdr_mcse,
                "FDR_MC95_high": fdr + Z_975 * fdr_mcse,
                "replications": int(group.shape[0]),
                "total_rejections": int(group["rejections"].sum()),
                "total_true_discoveries": int(group["true_discoveries"].sum()),
                "total_false_discoveries": int(group["false_discoveries"].sum()),
                "nonempty_replications": int((group["rejections"] > 0).sum()),
                "positive_fdp_replications": int((group["FDP"] > 0).sum()),
            }
        )
    frame = pd.DataFrame(rows)
    frame["_condition"] = frame["condition"].map(CONDITION_ORDER)
    frame["_method"] = frame["method"].map(METHOD_ORDER)
    frame = frame.sort_values(["_condition", "nonnulls", "_method"]).drop(
        columns=["_condition", "_method"]
    )
    require(frame.shape[0] == 30, "summary must contain 30 rows")
    return frame.reset_index(drop=True)


def build_containment_summary(containment: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, nonnulls), group in containment.groupby(
        ["condition", "nonnulls"], sort=False
    ):
        nonempty = group[group["ebh_size"] > 0]
        for method, column in (
            ("procedure1", "procedure1_contains_ebh"),
            ("procedure2", "procedure2_contains_ebh"),
        ):
            contained = int(group[column].astype(bool).sum())
            contained_nonempty = int(nonempty[column].astype(bool).sum())
            rows.append(
                {
                    "condition": condition,
                    "nonnulls": int(nonnulls),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "replications": int(group.shape[0]),
                    "contained": contained,
                    "containment_failures": int(group.shape[0]) - contained,
                    "containment_rate": contained / int(group.shape[0]),
                    "ebh_nonempty": int(nonempty.shape[0]),
                    "contained_given_ebh_nonempty": contained_nonempty,
                    "failures_given_ebh_nonempty": int(nonempty.shape[0]) - contained_nonempty,
                    "containment_rate_given_ebh_nonempty": contained_nonempty / int(nonempty.shape[0]),
                }
            )
    frame = pd.DataFrame(rows)
    frame["_condition"] = frame["condition"].map(CONDITION_ORDER)
    frame["_method"] = frame["method"].map(METHOD_ORDER)
    return frame.sort_values(["_condition", "nonnulls", "_method"]).drop(
        columns=["_condition", "_method"]
    ).reset_index(drop=True)


def render_report(summary: pd.DataFrame, containment: pd.DataFrame) -> str:
    lines = [
        "# Verified R=1000 asymmetric-weight simulation", "",
        "All values below are means over 1,000 replications; Monte Carlo standard errors are in parentheses.", "",
    ]
    names = {
        "clean": "Clean training data",
        "signed_square": "Signed-square training distortion",
        "persistent_noise": "Persistent error, epsilon_i ~ N(0,0.1)",
    }
    for condition in CONDITION_ORDER:
        lines.extend([f"## {names[condition]}", ""])
        for nonnulls in (20, 30):
            block = summary[(summary["condition"] == condition) & (summary["nonnulls"] == nonnulls)]
            lines.extend(
                [
                    f"### {nonnulls} nonnull hypotheses", "",
                    "| Procedure | TPR (MCSE) | FDR (MCSE) |",
                    "|---|---:|---:|",
                ]
            )
            for _, row in block.iterrows():
                lines.append(
                    f"| {row['method_label']} | {format4(row['TPR'])} "
                    f"({format4(row['TPR_MCSE'])}) | {format4(row['FDR'])} "
                    f"({format4(row['FDR_MCSE'])}) |"
                )
            lines.append("")
    lines.extend(["## Containment of testing-only eBH", ""])
    for _, row in containment.iterrows():
        lines.append(
            f"- {names[row['condition']]}, {int(row['nonnulls'])} nonnulls, "
            f"{row['method_label']}: {int(row['contained'])}/1000."
        )
    lines.extend(
        [
            "", "The normal Monte Carlo intervals in `summary.csv` quantify simulation error; they are not theoretical FDR guarantees.", "",
        ]
    )
    return "\n".join(lines)


def compare_frame(path: Path, expected: pd.DataFrame) -> None:
    observed = pd.read_csv(path)
    require(observed.shape == expected.shape, f"stored table shape mismatch: {path}")
    require(list(observed.columns) == list(expected.columns), f"stored table schema mismatch: {path}")
    for column in expected.columns:
        left = observed[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a = left.to_numpy(dtype=float)
            b = right.to_numpy(dtype=float)
            require(
                np.allclose(a, b, rtol=0.0, atol=2e-14, equal_nan=True),
                f"stored numeric column mismatch: {path}/{column}",
            )
        else:
            a = left.fillna("").astype(str).to_numpy()
            b = right.fillna("").astype(str).to_numpy()
            require(
                np.array_equal(a, b),
                f"stored text column mismatch: {path}/{column}",
            )


def validate_correction_record(root: Path) -> tuple[Path, int]:
    """Bind the transparent legacy-row correction record to current files."""
    path = root / "audit" / "correction_record.json"
    require(path.is_file(), "missing Procedure-2 correction record")
    record = json.loads(path.read_text(encoding="utf-8"))
    require(
        record.get("schema")
        == "closed-ebh-procedure2-exact-recertification-correction-v1",
        "correction-record schema mismatch",
    )
    rows = record.get("rows")
    stages = record.get("stages")
    require(isinstance(rows, list) and isinstance(stages, list), "malformed correction record")
    require(
        int(record.get("corrected_row_count", -1)) == 21 == len(rows),
        "correction-record row count mismatch",
    )
    require(
        int(record.get("affected_stage_count", -1)) == 3 == len(stages),
        "correction-record stage count mismatch",
    )
    keyed_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["stage"]), int(row["replication"]))
        require(key not in keyed_rows, "duplicate correction-record row")
        require(
            row.get("old_exactly_certified") is False
            and row.get("new_exactly_certified") is True,
            "correction certification flag mismatch",
        )
        require(
            str(row.get("new_status")) == "proven_maximal"
            and str(row.get("new_secondary_status")) == "proven_optimal"
            and int(row.get("new_proof_upper_bound")) == int(row.get("new_size")),
            "corrected proof metadata mismatch",
        )
        keyed_rows[key] = row
    recorded_keys = set()
    for stage in stages:
        relative = str(stage["stage"])
        condition_root = root / "data" / relative
        result_path = condition_root / "procedure2_results.csv"
        membership_path = condition_root / "procedure2_membership.csv"
        require(
            cached_sha256(result_path) == str(stage["result_sha256_after"])
            and cached_sha256(membership_path) == str(stage["membership_sha256_after"]),
            "corrected-stage hash mismatch",
        )
        results = pd.read_csv(result_path).set_index("replication")
        replications = [int(value) for value in stage["corrected_replications"]]
        require(len(replications) == len(set(replications)), "duplicate corrected replication")
        for replication in replications:
            key = (relative, replication)
            require(key in keyed_rows, "corrected stage/row map mismatch")
            row = keyed_rows[key]
            current = results.loc[replication]
            require(
                int(current["maximal_size"]) == int(row["new_size"])
                and str(current["maximal_indices"]) == str(row["new_indices"])
                and str(current["status"]) == str(row["new_status"])
                and int(current["proof_upper_bound"])
                == int(row["new_proof_upper_bound"]),
                "corrected row no longer matches the canonical result",
            )
            recorded_keys.add(key)
    require(recorded_keys == set(keyed_rows), "orphan correction-record row")
    for item in record.get("schema_only_cleanups", []):
        cleaned = root / str(item["path"])
        require(
            cached_sha256(cleaned) == str(item["sha256_after"]),
            "schema-only cleanup hash mismatch",
        )
    return path, len(rows)


def compare_historical_audit(observed: dict, current: dict) -> None:
    """Compare scientific evidence without requiring the historical machine.

    The strict package manifest binds the source files in this release. The
    archived audit's runtime and source hashes describe its original run and
    must not be overwritten merely because a later audit uses patched code or
    another Python patch version. Floating regeneration errors have already
    passed per-field tolerances and can differ across platforms.
    """
    require(set(observed) == set(current), "audit record schema mismatch")
    descriptive = {"runtime", "audit_script_sha256", "solver_source_sha256",
                   "maximum_scaled_regeneration_errors", "results"}
    for key in set(current) - descriptive:
        require(observed[key] == current[key], f"audit record mismatch: {key}")
    old_errors = observed["maximum_scaled_regeneration_errors"]
    new_errors = current["maximum_scaled_regeneration_errors"]
    require(set(old_errors) == set(new_errors), "audit regeneration-error fields changed")
    require(all(math.isfinite(float(error)) and 0 <= float(error) <= 5e-13
                for error in new_errors.values()), "audit regeneration tolerance exceeded")
    require(len(observed["results"]) == len(current["results"]), "audit result count mismatch")
    for old, new in zip(observed["results"], current["results"]):
        require(set(old) == set(new), "audit result schema mismatch")
        for key in new:
            if key in {"TPR", "TPR_MCSE", "FDR", "FDR_MCSE"}:
                require(math.isclose(float(old[key]), float(new[key]),
                                     rel_tol=2e-12, abs_tol=2e-14),
                        f"audit numerical result mismatch: {key}")
            else:
                require(old[key] == new[key], f"audit result identity mismatch: {key}")


def run_audit(root: Path, write: bool, full: bool) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    solver_dir = root / "code" / "solver"
    sys.path.insert(0, str(solver_dir))
    from method1_oracle import Method1Oracle  # pylint: disable=import-error,import-outside-toplevel

    correction_path, corrected_rows = validate_correction_record(root)
    metrics_rows: list[dict[str, Any]] = []
    containment_rows: list[dict[str, Any]] = []
    errors: dict[str, float] = {}
    recertified = 0
    stage_records: list[dict[str, Any]] = []
    for batch in config["batches"]:
        batch_root = root / "data" / str(batch["id"])
        persistent_seeds = validate_seed_table(batch_root / "seeds.csv", batch, errors)
        for nonnulls in config["nonnull_counts"]:
            panel_root = batch_root / f"n{nonnulls}"
            inputs = validate_panel_inputs(
                panel_root, batch, int(nonnulls), config, persistent_seeds, errors
            )
            truth = {
                int(rep): group.sort_values("index")["nonnull"].to_numpy(dtype=bool)
                for rep, group in inputs["clean"].groupby("replication", sort=True)
            }
            e_by_rep = {
                int(rep): group.sort_values("index")["e"].to_numpy(dtype=float)
                for rep, group in inputs["clean"].groupby("replication", sort=True)
            }
            testing_expected = {
                rep: largest_mean_set(e, float(config["alpha"]))
                for rep, e in e_by_rep.items()
            }
            testing_path = panel_root / "testing_mean_results.csv"
            testing = audit_direct(testing_path, testing_expected, truth)
            condition_results: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
            for condition in CONDITION_ORDER:
                condition_root = panel_root / condition
                procedure1, count1 = audit_learned(
                    condition_root,
                    inputs[condition],
                    "procedure1",
                    float(config["alpha"]),
                    full,
                    Method1Oracle,
                )
                print(
                    f"audited {batch['id']} n{nonnulls} {condition} procedure1",
                    flush=True,
                )
                procedure2, count2 = audit_learned(
                    condition_root,
                    inputs[condition],
                    "procedure2",
                    float(config["alpha"]),
                    full,
                    Method1Oracle,
                )
                print(
                    f"audited {batch['id']} n{nonnulls} {condition} procedure2",
                    flush=True,
                )
                recertified += count1 + count2
                for method in ("procedure1", "procedure2"):
                    input_path = condition_root / "learned_instances.csv"
                    result_path = condition_root / f"{method}_results.csv"
                    membership_path = condition_root / f"{method}_membership.csv"
                    stage_records.append(
                        {
                            "batch_id": batch["id"],
                            "batch_seed": int(batch["seed"]),
                            "nonnulls": int(nonnulls),
                            "condition": condition,
                            "procedure": method,
                            "input": input_path.relative_to(root).as_posix(),
                            "input_sha256": sha256_file(input_path),
                            "result": result_path.relative_to(root).as_posix(),
                            "result_sha256": sha256_file(result_path),
                            "membership": membership_path.relative_to(root).as_posix(),
                            "membership_sha256": sha256_file(membership_path),
                            "rows": 500,
                            "status": "all cardinalities proven and all secondary objectives proven",
                            "replay_runner": (
                                "code/solver/optimize_method1_learned.py"
                                if method == "procedure1"
                                else "code/solver/run_learned_asymmetric_closed_ebh_2.py"
                            ),
                        }
                    )
                condition_results[condition] = {
                    "procedure1": procedure1,
                    "procedure2": procedure2,
                }

                direct: dict[str, dict[int, dict[str, Any]]] = {}
                if condition == "clean":
                    values = {
                        int(rep): group.sort_values("index")["e_full"].to_numpy(dtype=float)
                        for rep, group in inputs[condition].groupby("replication", sort=True)
                    }
                    direct_path = condition_root / "full_mean_results.csv"
                    direct["full_mean"] = audit_direct(
                        direct_path,
                        {rep: largest_mean_set(value, float(config["alpha"])) for rep, value in values.items()},
                        truth,
                    )
                else:
                    values = {
                        int(rep): group.sort_values("index")["e_naive_full"].to_numpy(dtype=float)
                        for rep, group in inputs[condition].groupby("replication", sort=True)
                    }
                    direct_path = condition_root / "naive_mean_results.csv"
                    direct["naive_mean"] = audit_direct(
                        direct_path,
                        {rep: largest_mean_set(value, float(config["alpha"])) for rep, value in values.items()},
                        truth,
                    )

                for replication in range(1, 501):
                    ebh_set = set(int(i) for i in np.flatnonzero(ordinary_ebh(e_by_rep[replication], float(config["alpha"]))))
                    test_source = testing_path
                    metrics_rows.append(
                        metric_record(
                            condition, int(nonnulls), "testing_mean", batch, replication,
                            testing[replication]["selected"], testing[replication]["metrics"], test_source, root,
                        )
                    )
                    metrics_rows.append(
                        metric_record(
                            condition, int(nonnulls), "testing_ebh", batch, replication,
                            ebh_set, selection_metrics(ebh_set, truth[replication]),
                            panel_root / "clean" / "learned_instances.csv", root,
                        )
                    )
                    for method in ("procedure1", "procedure2"):
                        source = condition_root / f"{method}_results.csv"
                        item = condition_results[condition][method][replication]
                        metrics_rows.append(
                            metric_record(
                                condition, int(nonnulls), method, batch, replication,
                                item["selected"], item["metrics"], source, root,
                            )
                        )
                    direct_method = "full_mean" if condition == "clean" else "naive_mean"
                    item = direct[direct_method][replication]
                    metrics_rows.append(
                        metric_record(
                            condition, int(nonnulls), direct_method, batch, replication,
                            item["selected"], item["metrics"], direct_path, root,
                        )
                    )
                    p1 = condition_results[condition]["procedure1"][replication]["selected"]
                    p2 = condition_results[condition]["procedure2"][replication]["selected"]
                    containment_rows.append(
                        {
                            "condition": condition,
                            "nonnulls": int(nonnulls),
                            "batch_id": batch["id"],
                            "batch_seed": int(batch["seed"]),
                            "local_replication": replication,
                            "global_replication": int(batch["global_replication_start"]) + replication - 1,
                            "ebh_size": len(ebh_set),
                            "ebh_indices": encode_indices(ebh_set),
                            "procedure1_size": len(p1),
                            "procedure1_indices": encode_indices(p1),
                            "procedure1_contains_ebh": ebh_set.issubset(p1),
                            "procedure2_size": len(p2),
                            "procedure2_indices": encode_indices(p2),
                            "procedure2_contains_ebh": ebh_set.issubset(p2),
                        }
                    )

    metrics = pd.DataFrame(metrics_rows)[METRIC_COLUMNS]
    metrics["_condition"] = metrics["condition"].map(CONDITION_ORDER)
    metrics["_method"] = metrics["method"].map(METHOD_ORDER)
    metrics = metrics.sort_values(
        ["_condition", "nonnulls", "_method", "global_replication"]
    ).drop(columns=["_condition", "_method"]).reset_index(drop=True)
    require(metrics.shape[0] == 30000, "pooled metrics must contain 30,000 rows")
    containment = pd.DataFrame(containment_rows)[CONTAINMENT_COLUMNS]
    containment["_condition"] = containment["condition"].map(CONDITION_ORDER)
    containment = containment.sort_values(
        ["_condition", "nonnulls", "global_replication"]
    ).drop(columns=["_condition"]).reset_index(drop=True)
    require(containment.shape[0] == 6000, "containment output must contain 6,000 rows")
    require(containment["procedure2_contains_ebh"].all(), "Procedure 2 containment failure")
    summary = build_summary(metrics)
    containment_summary = build_containment_summary(containment)

    result_paths = {
        "replication_metrics": root / "results" / "replication_metrics.csv",
        "summary": root / "results" / "summary.csv",
        "containment_by_replication": root / "results" / "containment_by_replication.csv",
        "containment_summary": root / "results" / "containment_summary.csv",
        "report": root / "results" / "report.md",
    }
    stage_provenance_path = root / "audit" / "stage_provenance.json"
    stage_provenance = {
        "schema": "closed-ebh-publication-stage-provenance-v1",
        "stage_count": len(stage_records),
        "optimizer": config["optimizer"],
        "paths_are_package_relative": True,
        "stages": stage_records,
    }
    require(len(stage_records) == 24, "stage provenance must contain 24 stages")
    report = render_report(summary, containment_summary)
    if write:
        atomic_csv(result_paths["replication_metrics"], metrics)
        atomic_csv(result_paths["summary"], summary)
        atomic_csv(result_paths["containment_by_replication"], containment)
        atomic_csv(result_paths["containment_summary"], containment_summary)
        temporary = result_paths["report"].with_suffix(".md.tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(result_paths["report"])
        atomic_json(stage_provenance_path, stage_provenance)
    else:
        compare_frame(result_paths["replication_metrics"], metrics)
        compare_frame(result_paths["summary"], summary)
        compare_frame(result_paths["containment_by_replication"], containment)
        compare_frame(result_paths["containment_summary"], containment_summary)
        require(result_paths["report"].read_text(encoding="utf-8") == report, "report mismatch")
        require(
            json.loads(stage_provenance_path.read_text(encoding="utf-8"))
            == stage_provenance,
            "stage provenance mismatch",
        )

    outputs = {
        key: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for key, path in result_paths.items()
    }
    outputs["stage_provenance"] = {
        "path": stage_provenance_path.relative_to(root).as_posix(),
        "sha256": sha256_file(stage_provenance_path),
    }
    outputs["correction_record"] = {
        "path": correction_path.relative_to(root).as_posix(),
        "sha256": sha256_file(correction_path),
    }
    solver_hashes = {
        path.name: sha256_file(path)
        for path in sorted((root / "code" / "solver").glob("*.py"))
    }
    cells = []
    for _, row in summary.iterrows():
        cells.append(
            {
                "condition": row["condition"],
                "nonnulls": int(row["nonnulls"]),
                "method": row["method"],
                "TPR": float(row["TPR"]),
                "TPR_MCSE": float(row["TPR_MCSE"]),
                "FDR": float(row["FDR"]),
                "FDR_MCSE": float(row["FDR_MCSE"]),
            }
        )
    audit = {
        "schema": "closed-ebh-asymmetric-weights-publication-audit-v1",
        "study_config_sha256": sha256_file(root / "config" / "study.json"),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "full_independent_feasibility_recertification": bool(full),
        "counts": {
            "input_rows": 1200000,
            "learned_stages": 24,
            "learned_result_rows": 12000,
            "learned_membership_rows": 2400000,
            "learned_sets_independently_feasibility_recertified": recertified,
            "global_cardinality_proof_records_checked": 12000,
            "secondary_optimality_records_checked": 12000,
            "legacy_procedure2_rows_recomputed": corrected_rows,
            "replication_metric_rows": int(metrics.shape[0]),
            "containment_rows": int(containment.shape[0]),
            "summary_cells": int(summary.shape[0]),
            "procedure2_containment_failures": int((~containment["procedure2_contains_ebh"]).sum()),
        },
        "maximum_scaled_regeneration_errors": dict(sorted(errors.items())),
        "solver_source_sha256": solver_hashes,
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "highspy": importlib.metadata.version("highspy"),
        },
        "results": cells,
        "outputs": outputs,
        "status": "verified",
    }
    audit_path = root / "audit" / "audit.json"
    if write:
        atomic_json(audit_path, audit)
    else:
        observed = json.loads(audit_path.read_text(encoding="utf-8"))
        if not full:
            require(
                observed.get("full_independent_feasibility_recertification") is True
                and int(
                    observed.get("counts", {}).get(
                        "learned_sets_independently_feasibility_recertified", -1
                    )
                )
                == 12000,
                "stored audit is not a full feasibility recertification",
            )
            audit["full_independent_feasibility_recertification"] = True
            audit["counts"]["learned_sets_independently_feasibility_recertified"] = 12000
        compare_historical_audit(observed, audit)
    return audit


def main() -> int:
    args = parse_args()
    audit = run_audit(args.root, args.write, args.full)
    print(json.dumps({"status": audit["status"], "counts": audit["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

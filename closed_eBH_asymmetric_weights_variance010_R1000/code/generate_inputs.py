#!/usr/bin/env python3
"""Regenerate the three reported simulation inputs from the fixed seeds."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from study_common import (
    PERSISTENT_NOISE_SALT,
    SIGNED_SQUARE_RESIDUAL_SALT,
    atomic_csv,
    atomic_json,
    encode_indices,
    largest_mean_set,
    load_config,
    ordinary_ebh,
    reconstruct_training_observations,
    selection_metrics,
    sha256_file,
    splitmix64,
    stable_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-to-published",
        action="store_true",
        help="Numerically compare regenerated inputs with root/data.",
    )
    return parser.parse_args()


def direct_row(
    replication: int,
    method: str,
    selected: np.ndarray,
    truth: np.ndarray,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "replication": replication,
        "method": method,
        **extra,
        "status": "proven_maximum_cardinality",
        "elapsed_seconds": 0.0,
        "selected_indices": encode_indices(selected),
    }
    metrics = selection_metrics(selected, truth)
    row.update(
        {
            "rejections": metrics["rejections"],
            "true_discoveries": metrics["true_discoveries"],
            "false_discoveries": metrics["false_discoveries"],
            "tpr": metrics["TPR"],
            "fdp": metrics["FDP"],
        }
    )
    return row


def generate_panel(
    config: dict[str, Any], batch: dict[str, Any], nonnulls: int, output: Path
) -> dict[str, str]:
    k = int(config["K"])
    reps = int(batch["local_replications"])
    alpha = float(config["alpha"])
    mu = float(config["signal_mean"])
    eta = float(config["e_value_tilt"])
    n_train = int(config["training_observations"])
    n_test = int(config["testing_observations"])
    error_sd = float(
        config["conditions"]["persistent_noise"]["error_standard_deviation"]
    )
    test_offset = 0.5 * n_test * eta * eta
    full_offset = 0.5 * (n_train + n_test) * eta * eta

    truth = np.zeros((reps, k), dtype=np.uint8)
    train_sum = np.empty((reps, k), dtype=float)
    test_sum = np.empty((reps, k), dtype=float)
    e_test = np.empty((reps, k), dtype=float)
    prefix = np.zeros((reps, k), dtype=np.uint8)
    clean_score = np.empty((reps, k), dtype=float)
    e_full = np.empty((reps, k), dtype=float)
    signed_sum = np.empty((reps, k), dtype=float)
    signed_score = np.empty((reps, k), dtype=float)
    signed_naive = np.empty((reps, k), dtype=float)
    persistent_z = np.empty((reps, k), dtype=float)
    persistent_error = np.empty((reps, k), dtype=float)
    persistent_sum = np.empty((reps, k), dtype=float)
    persistent_score = np.empty((reps, k), dtype=float)
    persistent_naive = np.empty((reps, k), dtype=float)
    primary_seeds: list[int] = []
    persistent_seeds: list[int] = []
    testing_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    signed_naive_rows: list[dict[str, Any]] = []
    persistent_naive_rows: list[dict[str, Any]] = []

    for offset in range(reps):
        replication = offset + 1
        primary_seed = splitmix64(int(batch["seed"]) ^ splitmix64(replication))
        rng = np.random.Generator(np.random.PCG64(primary_seed))
        labels = rng.permutation(k)
        local_truth = np.zeros(k, dtype=bool)
        local_truth[labels[:nonnulls]] = True
        gaussian = rng.standard_normal((k, 2))
        local_train = n_train * mu * local_truth + math.sqrt(n_train) * gaussian[:, 0]
        local_test = n_test * mu * local_truth + math.sqrt(n_test) * gaussian[:, 1]
        local_e = np.exp(eta * local_test - test_offset)
        local_prefix = ordinary_ebh(local_e, alpha)
        local_clean_score = stable_scores(local_train, eta)
        local_full = np.exp(eta * (local_train + local_test) - full_offset)

        residual_seed = splitmix64(primary_seed ^ SIGNED_SQUARE_RESIDUAL_SALT)
        observations = reconstruct_training_observations(
            local_train, n_train, residual_seed
        )
        local_signed = np.sum(
            np.sign(observations) * np.square(np.abs(observations)), axis=1
        )
        local_signed_score = stable_scores(local_signed, eta)
        local_signed_naive = np.exp(
            eta * (local_signed + local_test) - full_offset
        )

        persistent_seed = splitmix64(primary_seed ^ PERSISTENT_NOISE_SALT)
        local_z = np.random.Generator(
            np.random.PCG64(persistent_seed)
        ).standard_normal(k)
        local_error = error_sd * local_z
        local_persistent = local_train + n_train * local_error
        local_persistent_score = stable_scores(local_persistent, eta)
        local_persistent_naive = np.exp(
            eta * (local_persistent + local_test) - full_offset
        )

        truth[offset] = local_truth
        train_sum[offset] = local_train
        test_sum[offset] = local_test
        e_test[offset] = local_e
        prefix[offset] = local_prefix
        clean_score[offset] = local_clean_score
        e_full[offset] = local_full
        signed_sum[offset] = local_signed
        signed_score[offset] = local_signed_score
        signed_naive[offset] = local_signed_naive
        persistent_z[offset] = local_z
        persistent_error[offset] = local_error
        persistent_sum[offset] = local_persistent
        persistent_score[offset] = local_persistent_score
        persistent_naive[offset] = local_persistent_naive
        primary_seeds.append(primary_seed)
        persistent_seeds.append(persistent_seed)

        testing_rows.append(
            direct_row(
                replication,
                "mean_testing_only",
                largest_mean_set(local_e, alpha),
                local_truth,
            )
        )
        full_rows.append(
            direct_row(
                replication,
                "mean_full_data",
                largest_mean_set(local_full, alpha),
                local_truth,
            )
        )
        signed_naive_rows.append(
            direct_row(
                replication,
                "mean_naive_full_data",
                largest_mean_set(local_signed_naive, alpha),
                local_truth,
                scenario="signed_square",
                parameter="gamma",
                severity=1.0,
            )
        )
        persistent_naive_rows.append(
            direct_row(
                replication,
                "mean_naive_persistent_noise_full_data",
                largest_mean_set(local_persistent_naive, alpha),
                local_truth,
                scenario="persistent_noise",
                parameter="variance",
                severity=0.1,
            )
        )

    replication_column = np.repeat(np.arange(1, reps + 1), k)
    index_column = np.tile(np.arange(k), reps)
    prefix_size = np.repeat(np.sum(prefix, axis=1), k)
    shared = {
        "replication": replication_column,
        "index": index_column,
        "e": e_test.ravel(),
        "nonnull": truth.ravel(),
        "prefix_size": prefix_size,
        "prefix_member": prefix.ravel(),
        "train_sum": train_sum.ravel(),
        "test_sum": test_sum.ravel(),
    }

    clean = pd.DataFrame(
        {
            **shared,
            "score": clean_score.ravel(),
            "e_full": e_full.ravel(),
            "scenario": "clean",
        }
    )[
        [
            "replication", "index", "e", "score", "nonnull", "prefix_size",
            "prefix_member", "train_sum", "test_sum", "e_full", "scenario",
        ]
    ]
    signed = pd.DataFrame(
        {
            **shared,
            "score": signed_score.ravel(),
            "contaminated_train_sum": signed_sum.ravel(),
            "e_naive_full": signed_naive.ravel(),
            "score_tilt": eta,
            "scenario": "signed_square",
            "parameter": "gamma",
            "severity": 1.0,
        }
    )[
        [
            "replication", "index", "e", "score", "nonnull", "prefix_size",
            "prefix_member", "train_sum", "test_sum", "contaminated_train_sum",
            "e_naive_full", "score_tilt", "scenario", "parameter", "severity",
        ]
    ]
    persistent = pd.DataFrame(
        {
            **shared,
            "score": persistent_score.ravel(),
            "persistent_standard_normal": persistent_z.ravel(),
            "per_observation_shift": persistent_error.ravel(),
            "aggregate_training_shift": (n_train * persistent_error).ravel(),
            "contaminated_train_sum": persistent_sum.ravel(),
            "e_naive_full": persistent_naive.ravel(),
            "score_tilt": eta,
            "scenario": "persistent_noise",
            "parameter": "variance",
            "severity": 0.1,
        }
    )[
        [
            "replication", "index", "e", "score", "nonnull", "prefix_size",
            "prefix_member", "train_sum", "test_sum", "persistent_standard_normal",
            "per_observation_shift", "aggregate_training_shift",
            "contaminated_train_sum", "e_naive_full", "score_tilt", "scenario",
            "parameter", "severity",
        ]
    ]

    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "testing_mean_results.csv", pd.DataFrame(testing_rows))
    atomic_csv(output / "clean" / "learned_instances.csv", clean)
    atomic_csv(output / "clean" / "full_mean_results.csv", pd.DataFrame(full_rows))
    atomic_csv(output / "signed_square" / "learned_instances.csv", signed)
    atomic_csv(
        output / "signed_square" / "naive_mean_results.csv",
        pd.DataFrame(signed_naive_rows),
    )
    persistent_direct = pd.DataFrame(persistent_naive_rows).drop(
        columns=["elapsed_seconds"]
    )
    atomic_csv(output / "persistent_noise" / "learned_instances.csv", persistent)
    atomic_csv(
        output / "persistent_noise" / "naive_mean_results.csv", persistent_direct
    )
    return {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*.csv"))
    }


def compare_csv(generated: Path, published: Path) -> dict[str, float]:
    left = pd.read_csv(generated)
    right = pd.read_csv(published)
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        raise RuntimeError(f"schema mismatch against published file: {published}")
    ignored = {"elapsed_seconds"}
    maximum = 0.0
    for column in left.columns:
        if column in ignored:
            continue
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            a = left[column].to_numpy(dtype=float)
            b = right[column].to_numpy(dtype=float)
            error = np.abs(a - b) / np.maximum(1.0, np.abs(b))
            maximum = max(maximum, float(np.max(error, initial=0.0)))
            if maximum > 5e-13:
                raise RuntimeError(f"numeric mismatch against published file: {published}")
        else:
            a = left[column].fillna("").astype(str).to_numpy()
            b = right[column].fillna("").astype(str).to_numpy()
            if not np.array_equal(a, b):
                raise RuntimeError(f"text mismatch against published file: {published}")
    return {"maximum_scaled_numeric_error": maximum}


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("output directory already exists")
    config = load_config(root)
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {"schema": "generated-inputs-v1", "panels": {}}
    for batch in config["batches"]:
        batch_root = output / str(batch["id"])
        seed_rows = []
        for replication in range(1, int(batch["local_replications"]) + 1):
            primary = splitmix64(int(batch["seed"]) ^ splitmix64(replication))
            persistent = splitmix64(primary ^ PERSISTENT_NOISE_SALT)
            seed_rows.append(
                {
                    "local_replication": replication,
                    "global_replication": int(batch["global_replication_start"])
                    + replication
                    - 1,
                    "primary_replication_seed": primary,
                    "persistent_noise_seed": persistent,
                }
            )
        atomic_csv(batch_root / "seeds.csv", pd.DataFrame(seed_rows))
        for nonnulls in config["nonnull_counts"]:
            key = f"{batch['id']}/n{nonnulls}"
            panel_root = batch_root / f"n{nonnulls}"
            manifest["panels"][key] = generate_panel(
                config, batch, int(nonnulls), panel_root
            )
    manifest["files"] = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*.csv"))
    }
    if args.compare_to_published:
        comparisons = {}
        for relative in sorted(manifest["files"]):
            generated = output / relative
            published = root / "data" / relative
            comparisons[relative] = compare_csv(generated, published)
        manifest["published_comparison"] = comparisons
    atomic_json(output / "generation_manifest.json", manifest)
    print(
        f"Generated {len(manifest['files'])} input/direct-result files under {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

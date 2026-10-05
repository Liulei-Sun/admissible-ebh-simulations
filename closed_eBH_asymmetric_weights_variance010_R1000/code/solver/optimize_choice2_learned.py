#!/usr/bin/env python3
"""Run the Choice-2 learned closed-eBH maximal-selection experiment.

The saved simulation instances are reused exactly.  For each replication this
script recomputes ordinary eBH from the testing e-values, then finds a
maximum-cardinality Choice-2-certified superset of that eBH seed.  Maximum
cardinality is stronger than the inclusion-maximality required by the paper.

The exact Choice-2 separator is supplied by ``choice2_oracle.py``.  Rows are
checkpointed after every completed replication by default.  A timeout or
separator failure is never labelled maximal.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import optimize_maximal_learned as outer
from choice2_oracle import Choice2Oracle, ordinary_ebh


SIZE_CALIBRATED_ORACLE = outer.LearnedOracle


@lru_cache(maxsize=1)
def solver_contract_sha256() -> str:
    """Bind checkpoints to the exact solver implementation being executed."""
    digest = hashlib.sha256(b"closed-ebh-checkpoint-contract-v1\0")
    for path in sorted(SCRIPT_DIR.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_choice2_oracle(e,
                        score,
                        alpha,
                        separator_time_limit,
                        separator_threads):
    oracle = Choice2Oracle(
        e,
        score,
        alpha,
        separator_time_limit=separator_time_limit,
        separator_threads=separator_threads,
    )
    # The released size-calibrated oracle is a fast source of likely violated
    # intersections.  A second surrogate uses the unshrunk natural scores.
    # Neither may certify anything; Choice2Oracle recomputes every proposed
    # cut with the exact local Choice-2 weights before accepting it.
    calibrated = SIZE_CALIBRATED_ORACLE(e, score, alpha)
    natural = SIZE_CALIBRATED_ORACLE(e, score, alpha)
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


def seed_only_greedy(oracle, initial, witness_limit):
    """Keep eBH as the constraint seed; the outer MIP performs the extension."""
    del oracle, witness_limit
    return initial.copy(), 0, {}


def choice2_problem_hash(instance, alpha: float) -> str:
    digest = hashlib.sha256()
    digest.update(b"closed-ebh-choice2-minimal-shrink-v1\0")
    digest.update(np.asarray([alpha, float(instance.e.size)], dtype="<f8").tobytes())
    digest.update(np.asarray(instance.e, dtype="<f8").tobytes())
    digest.update(np.asarray(instance.score, dtype="<f8").tobytes())
    digest.update(np.asarray(instance.nonnull, dtype=np.uint8).tobytes())
    digest.update(np.asarray(instance.prefix, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def optimize_one(payload):
    (instance, alpha, separator_time_limit, separator_threads,
     layer_batch, fixed_batch, layer_window, master_time_limit) = payload
    started = time.perf_counter()
    old_prefix_size = int(np.sum(instance.prefix))
    ebh = ordinary_ebh(instance.e, alpha)
    choice2_instance = replace(instance, prefix=ebh)

    # The legacy module's outer master and cut-capacity logic are merger-family
    # agnostic.  Only its oracle and initial greedy step are replaced.
    outer.LearnedOracle = lambda e, score, level: make_choice2_oracle(
        e, score, level, separator_time_limit, separator_threads
    )
    outer.deterministic_greedy = seed_only_greedy

    try:
        result = outer.optimize_instance(
            choice2_instance,
            alpha,
            1,
            layer_batch,
            fixed_batch,
            layer_window,
            master_time_limit,
        )
        row = asdict(result)
        row["solver_code_sha256"] = solver_contract_sha256()
        row["problem_sha256"] = choice2_problem_hash(choice2_instance, alpha)
        row["method"] = "choice2_minimal_local_shrink"
        row["selection_rule"] = "maximum_cardinality_certified_superset_of_eBH"
        row["K"] = int(instance.e.size)
        row["ebh_seed_size"] = int(np.sum(ebh))
        row["old_exported_prefix_size"] = old_prefix_size
        row["separator_backend"] = "SCIP_indicator_MILP"
        row["separator_time_limit"] = separator_time_limit
        row["separator_threads"] = separator_threads

        maximal = {
            int(value) for value in row["maximal_indices"].split(";") if value
        }
        seed = set(int(i) for i in np.flatnonzero(ebh))
        if not seed.issubset(maximal):
            raise RuntimeError("final set does not contain ordinary eBH")
        if row["status"] == "proven_maximal":
            if int(row["proof_upper_bound"]) != int(row["maximal_size"]):
                raise RuntimeError("maximality bound does not equal final cardinality")
        return row
    except Exception as exc:  # retained in checkpoint; never called maximal
        return {
            "replication": int(instance.replication),
            "solver_code_sha256": solver_contract_sha256(),
            "problem_sha256": choice2_problem_hash(choice2_instance, alpha),
            "status": "error",
            "method": "choice2_minimal_local_shrink",
            "selection_rule": "maximum_cardinality_certified_superset_of_eBH",
            "K": int(instance.e.size),
            "ebh_seed_size": int(np.sum(ebh)),
            "old_exported_prefix_size": old_prefix_size,
            "elapsed_seconds": time.perf_counter() - started,
            "separator_backend": "SCIP_indicator_MILP",
            "separator_time_limit": separator_time_limit,
            "separator_threads": separator_threads,
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def mean_mcse(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(values))
    mcse = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, mcse


def write_products(results: list[dict],
                   output_dir: Path,
                   reference_results: Path | None) -> None:
    frame = pd.DataFrame(results).sort_values("replication").reset_index(drop=True)
    results_path = output_dir / "choice2_results.csv"
    atomic_write_csv(frame, results_path)

    complete = frame[frame["status"] == "proven_maximal"].copy()
    summary: dict[str, object] = {
        "rows_total": int(frame.shape[0]),
        "rows_proven_maximal": int(complete.shape[0]),
        "rows_incomplete": int(frame.shape[0] - complete.shape[0]),
        "replications_proven": [int(value) for value in complete["replication"]],
    }
    for column in ("ebh_seed_size", "maximal_size", "maximal_tpr", "maximal_fdp"):
        if column in complete:
            mean, mcse = mean_mcse(complete[column].to_numpy(dtype=float))
            summary[f"{column}_mean"] = mean
            summary[f"{column}_mcse"] = mcse

    if reference_results is not None and reference_results.exists() and not complete.empty:
        reference = pd.read_csv(reference_results)
        paired = complete.merge(
            reference[["replication", "maximal_tpr", "maximal_size"]],
            on="replication",
            suffixes=("_choice2", "_current"),
            validate="one_to_one",
        )
        paired["tpr_difference"] = (
            paired["maximal_tpr_choice2"] - paired["maximal_tpr_current"]
        )
        paired["size_difference"] = (
            paired["maximal_size_choice2"] - paired["maximal_size_current"]
        )
        atomic_write_csv(
            paired[[
                "replication",
                "maximal_tpr_current",
                "maximal_tpr_choice2",
                "tpr_difference",
                "maximal_size_current",
                "maximal_size_choice2",
                "size_difference",
            ]],
            output_dir / "choice2_vs_current_paired.csv",
        )
        difference = paired["tpr_difference"].to_numpy(dtype=float)
        mean, mcse = mean_mcse(difference)
        summary.update({
            "paired_current_tpr_mean": float(np.mean(paired["maximal_tpr_current"])),
            "paired_choice2_tpr_mean": float(np.mean(paired["maximal_tpr_choice2"])),
            "paired_tpr_difference_mean": mean,
            "paired_tpr_difference_mcse": mcse,
            "paired_tpr_improved": int(np.sum(difference > 0.0)),
            "paired_tpr_tied": int(np.sum(difference == 0.0)),
            "paired_tpr_worse": int(np.sum(difference < 0.0)),
        })

    (output_dir / "choice2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    memberships = []
    for row in complete.to_dict(orient="records"):
        replication = int(row["replication"])
        seed = {int(value) for value in str(row["prefix_indices"]).split(";") if value}
        final = {int(value) for value in str(row["maximal_indices"]).split(";") if value}
        inferred_k = max(seed | final) + 1 if seed or final else 0
        # New outputs record K explicitly.  The fallback preserves the legacy
        # K=400 behavior when regenerating products from older checkpoints.
        k = int(row["K"]) if "K" in row and pd.notna(row["K"]) else max(
            inferred_k, 400
        )
        if k < inferred_k:
            raise RuntimeError("recorded K is smaller than a selected index")
        memberships.extend({
            "replication": replication,
            "index": index,
            "ebh_member": int(index in seed),
            "choice2_member": int(index in final),
        } for index in range(k))
    if memberships:
        atomic_write_csv(
            pd.DataFrame(memberships), output_dir / "choice2_memberships.csv"
        )


def write_checkpoint(results: list[dict], output_dir: Path) -> None:
    """Atomically persist resumable rows without rebuilding large products."""
    frame = pd.DataFrame(results).sort_values("replication").reset_index(drop=True)
    atomic_write_csv(frame, output_dir / "choice2_results.csv")


def validated_resume_rows(existing: pd.DataFrame,
                          instances: list,
                          alpha: float) -> list[dict]:
    """Retain only complete proofs for the exact requested problems.

    The active runner supplies both the seed rule and the method-specific
    hash, so switching inputs, alpha, or procedure cannot reuse stale rows.
    Incomplete rows are recomputed; mismatched or duplicate rows fail closed.
    """
    if existing.empty:
        return []
    if "replication" not in existing or existing["replication"].duplicated().any():
        raise RuntimeError("resume checkpoint has missing or duplicate replication IDs")
    by_replication = {int(instance.replication): instance for instance in instances}
    retained = []
    for row in existing.to_dict(orient="records"):
        replication = int(row["replication"])
        if replication not in by_replication:
            raise RuntimeError("resume checkpoint contains an unrequested replication")
        instance = by_replication[replication]
        seeded = replace(instance, prefix=ordinary_ebh(instance.e, alpha))
        if row.get("problem_sha256") != choice2_problem_hash(seeded, alpha):
            raise RuntimeError(
                f"resume checkpoint problem mismatch for replication {replication}; "
                "use a new output directory for changed inputs, alpha, or procedure"
            )
        if not (row.get("status") == "proven_maximal"
                and row.get("secondary_status") == "proven_optimal"
                and row.get("proof_upper_bound") == row.get("maximal_size")):
            continue
        if row.get("solver_code_sha256") != solver_contract_sha256():
            raise RuntimeError("resume checkpoint solver code changed or lacks a code signature")

        def indices(value):
            if pd.isna(value) or value == "":
                return set()
            parts = str(value).split(";")
            values = [int(part) for part in parts]
            if (len(values) != len(set(values))
                    or any(str(index) != part or index < 0 or index >= instance.e.size
                           for index, part in zip(values, parts))):
                raise RuntimeError("resume checkpoint has invalid selected indices")
            return set(values)

        selected = indices(row.get("maximal_indices"))
        seed = set(np.flatnonzero(seeded.prefix))
        if (row.get("K") != instance.e.size
                or row.get("maximal_size") != len(selected)
                or row.get("prefix_size") != len(seed)
                or indices(row.get("prefix_indices")) != seed
                or not seed.issubset(selected)):
            raise RuntimeError("resume checkpoint membership or size mismatch")
        mask = np.zeros(instance.e.size, dtype=bool)
        if selected:
            mask[np.array(sorted(selected), dtype=int)] = True
        true, false, tpr, fdp = outer.summarize_set(mask, instance.nonnull)
        objective = float(np.sum(instance.e[mask], dtype=np.longdouble))
        for column, value in (("maximal_true", true), ("maximal_false", false),
                              ("maximal_tpr", tpr), ("maximal_fdp", fdp),
                              ("secondary_value", objective)):
            stored = row.get(column)
            if stored is None or not math.isclose(float(stored), float(value),
                                                  rel_tol=2e-12, abs_tol=2e-10):
                raise RuntimeError(f"resume checkpoint {column} mismatch")
        # pandas reads blank index cells as NaN; product writers require the
        # canonical empty string, especially for Method 1's empty seed.
        row["maximal_indices"] = ";".join(str(index) for index in sorted(selected))
        row["prefix_indices"] = ";".join(str(index) for index in sorted(seed))
        retained.append(row)
    return retained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--replication-start", type=int)
    parser.add_argument("--replication-end", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--separator-time-limit", type=float, default=600.0)
    parser.add_argument("--separator-threads", type=int, default=1)
    parser.add_argument("--master-time-limit", type=float, default=300.0)
    parser.add_argument("--layer-batch", type=int, default=100)
    parser.add_argument("--fixed-batch", type=int, default=100)
    parser.add_argument("--layer-window", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0 or args.separator_threads <= 0:
        raise ValueError("workers and separator-threads must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances = outer.read_instances(
        args.input,
        replication_start=args.replication_start,
        replication_end=args.replication_end,
    )

    results_path = args.output_dir / "choice2_results.csv"
    retained: list[dict] = []
    completed: set[int] = set()
    if args.resume and results_path.exists():
        existing = pd.read_csv(results_path)
        retained = validated_resume_rows(existing, instances, args.alpha)
        completed = {int(row["replication"]) for row in retained}

    pending = [instance for instance in instances if instance.replication not in completed]
    payloads = [(
        instance,
        args.alpha,
        args.separator_time_limit,
        args.separator_threads,
        args.layer_batch,
        args.fixed_batch,
        args.layer_window,
        args.master_time_limit,
    ) for instance in pending]

    newly_completed = 0
    if args.workers == 1:
        iterator = (optimize_one(payload) for payload in payloads)
        for row in iterator:
            retained.append(row)
            newly_completed += 1
            print(
                f"replication {row['replication']}: {row['status']}, "
                f"size={row.get('maximal_size', 'NA')}, "
                f"tpr={row.get('maximal_tpr', 'NA')}",
                flush=True,
            )
            if newly_completed % args.checkpoint_every == 0:
                write_checkpoint(retained, args.output_dir)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(optimize_one, payload): payload[0].replication
                       for payload in payloads}
            for future in as_completed(futures):
                row = future.result()
                retained.append(row)
                newly_completed += 1
                print(
                    f"replication {row['replication']}: {row['status']}, "
                    f"size={row.get('maximal_size', 'NA')}, "
                    f"tpr={row.get('maximal_tpr', 'NA')}",
                    flush=True,
                )
                if newly_completed % args.checkpoint_every == 0:
                    write_checkpoint(retained, args.output_dir)

    write_products(retained, args.output_dir, args.reference_results)
    incomplete = [row for row in retained if row.get("status") != "proven_maximal"]
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())

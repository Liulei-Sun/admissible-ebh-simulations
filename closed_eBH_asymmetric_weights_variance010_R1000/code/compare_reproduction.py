#!/usr/bin/env python3
"""Compare fresh scientific results with the archive and rebuild their table.

Solver timings, search-node counts, and iteration histories are deliberately
excluded: they depend on the machine. Memberships, cardinality, objectives,
proof statuses, and scientific metrics must reproduce the archived results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from audit_results import build_summary, compare_frame
from generate_inputs import compare_csv
from study_common import (METHOD_LABELS, atomic_csv, atomic_json, load_config,
                          ordinary_ebh, parse_indices, require,
                          selection_metrics, sha256_file)


SCIENTIFIC_COLUMNS = [
    "replication", "K", "status", "secondary_status", "secondary_rule",
    "secondary_value", "method", "selection_rule", "proof_upper_bound",
    "maximal_size", "maximal_indices", "maximal_true", "maximal_false",
    "maximal_tpr", "maximal_fdp", "prefix_size", "prefix_indices",
]


def compare_learned_stage(generated: Path, published: Path, procedure: str) -> int:
    frames = []
    for directory in (generated, published):
        frame = pd.read_csv(directory / f"{procedure}_results.csv", keep_default_na=False)
        require(set(SCIENTIFIC_COLUMNS).issubset(frame), "missing scientific result columns")
        require(not frame["replication"].duplicated().any(), "duplicate result replication")
        frame = frame.sort_values("replication").reset_index(drop=True)
        for column in ("maximal_indices", "prefix_indices"):
            frame[column] = frame[column].map(lambda value: ";".join(
                str(index) for index in sorted(parse_indices(value))))
        frames.append(frame[SCIENTIFIC_COLUMNS])
    pd.testing.assert_frame_equal(frames[0], frames[1], check_dtype=False,
                                  check_exact=False, rtol=2e-12, atol=2e-10)
    members = [pd.read_csv(directory / f"{procedure}_membership.csv").sort_values(
        ["replication", "index"]).reset_index(drop=True)
        for directory in (generated, published)]
    pd.testing.assert_frame_equal(members[0], members[1], check_dtype=False, check_exact=True)
    return len(frames[0])


def validate_generated_files(data_root: Path) -> dict:
    manifest = json.loads((data_root / "generation_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == "generated-inputs-v1", "generation manifest schema mismatch")
    expected = set()
    for batch in ("batch1_seed20260803", "batch2_seed20260804"):
        expected.add(f"{batch}/seeds.csv")
        for nonnulls in (20, 30):
            panel = f"{batch}/n{nonnulls}"
            expected.add(f"{panel}/testing_mean_results.csv")
            for condition in ("clean", "signed_square", "persistent_noise"):
                expected.add(f"{panel}/{condition}/learned_instances.csv")
                direct = "full_mean" if condition == "clean" else "naive_mean"
                expected.add(f"{panel}/{condition}/{direct}_results.csv")
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == expected,
            "generation manifest must list exactly the 30 expected input/direct files")
    for relative, digest in files.items():
        pure = PurePosixPath(relative)
        require(relative == pure.as_posix() and not pure.is_absolute()
                and ".." not in pure.parts and "\\" not in relative
                and ":" not in relative, "unsafe generation manifest path")
        path = (data_root / relative).resolve()
        require(path.is_relative_to(data_root.resolve()), "unsafe generation manifest path")
        require(path.is_file() and sha256_file(path) == digest,
                f"generated input changed since generation: {relative}")
    return manifest


def compare_generated_files(root: Path, data_root: Path) -> dict:
    manifest = validate_generated_files(data_root)
    for relative in manifest["files"]:
        compare_csv(data_root / relative, root / "data" / relative)
    return manifest


def compare_reproduction(root: Path, output: Path) -> dict:
    config = load_config(root)
    data_root = output / "data"
    compare_generated_files(root, data_root)
    rows = []
    stage_rows = 0
    for batch in config["batches"]:
        for nonnulls in config["nonnull_counts"]:
            panel = data_root / batch["id"] / f"n{nonnulls}"
            for condition in config["conditions"]:
                generated = panel / condition
                published = root / "data" / batch["id"] / f"n{nonnulls}" / condition
                for procedure in ("procedure1", "procedure2"):
                    stage_rows += compare_learned_stage(generated, published, procedure)
                inputs = pd.read_csv(generated / "learned_instances.csv")
                direct_method = "full_mean" if condition == "clean" else "naive_mean"
                selections = {
                    "testing_mean": pd.read_csv(panel / "testing_mean_results.csv").set_index("replication"),
                    direct_method: pd.read_csv(generated / f"{direct_method}_results.csv").set_index("replication"),
                    **{method: pd.read_csv(generated / f"{method}_results.csv").set_index("replication")
                       for method in ("procedure1", "procedure2")},
                }
                for replication, group in inputs.groupby("replication", sort=True):
                    group = group.sort_values("index")
                    truth = group["nonnull"].to_numpy(dtype=bool)
                    selected_by_method = {"testing_ebh": ordinary_ebh(
                        group["e"].to_numpy(dtype=float), config["alpha"])}
                    for method, frame in selections.items():
                        column = "maximal_indices" if method.startswith("procedure") else "selected_indices"
                        selected_by_method[method] = parse_indices(frame.loc[replication, column])
                    for method, selected in selected_by_method.items():
                        rows.append(dict(condition=condition, nonnulls=nonnulls,
                            method=method, method_label=METHOD_LABELS[method],
                            global_replication=batch["global_replication_start"] + int(replication) - 1,
                            **selection_metrics(selected, truth)))
    require(stage_rows == 12000, "fresh learned result count must be 12,000")
    metrics = pd.DataFrame(rows)
    summary = build_summary(metrics)
    compare_frame(root / "results" / "summary.csv", summary)
    atomic_csv(output / "results" / "replication_metrics.csv", metrics)
    atomic_csv(output / "results" / "summary.csv", summary)
    report = dict(status="scientific_results_reproduced", learned_rows=stage_rows,
                  pooled_metric_rows=len(metrics), summary_rows=len(summary),
                  comparison="exact membership and proof status; numeric tolerance 2e-12 relative, 2e-10 absolute",
                  excluded_machine_dependent_fields="timings, search nodes, cuts, iteration histories")
    atomic_json(output / "reproduction_comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_reproduction(args.root.resolve(), args.output_dir.resolve()), indent=2))

#!/usr/bin/env python3
"""Regenerate inputs and rerun all 24 learned-procedure stages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

from study_common import load_config
from compare_reproduction import compare_reproduction, compare_generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    )
    with log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"stage failed; inspect {log}")


def verify_completed(path: Path) -> None:
    frame = pd.read_csv(path)
    if (
        frame.shape[0] != 500
        or set(frame["replication"].astype(int)) != set(range(1, 501))
        or not (frame["status"] == "proven_maximal").all()
        or not (frame["secondary_status"] == "proven_optimal").all()
        or not (
            frame["proof_upper_bound"].astype(int)
            == frame["maximal_size"].astype(int)
        ).all()
    ):
        raise RuntimeError(f"learned stage is incomplete: {path}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    config = load_config(root)
    if output == root or output.is_relative_to(root):
        raise RuntimeError("output directory must be outside the publication package")
    data_root = output / "data"
    generator = root / "code" / "generate_inputs.py"
    if not args.resume:
        if output.exists():
            raise RuntimeError("output directory already exists")
        output.mkdir(parents=True)
        run(
            [
                sys.executable,
                "-B",
                str(generator),
                "--root",
                str(root),
                "--output-dir",
                str(data_root),
                "--compare-to-published",
            ],
            root,
            output / "generation.log",
        )
    elif not (data_root / "generation_manifest.json").is_file():
        raise RuntimeError("--resume requires an existing generated data directory")
    compare_generated_files(root, data_root)
    if args.generate_only:
        print(f"Generated inputs verified against the publication data: {data_root}")
        return 0

    solver = root / "code" / "solver"
    optimizer = config["optimizer"]
    common_args = [
        "--alpha", str(config["alpha"]),
        "--workers", str(optimizer["workers"]),
        "--separator-time-limit", str(optimizer["separator_time_limit_seconds"]),
        "--separator-threads", str(optimizer["separator_threads"]),
        "--master-time-limit", str(optimizer["master_time_limit_seconds"]),
        "--layer-batch", str(optimizer["layer_batch"]),
        "--fixed-batch", str(optimizer["fixed_batch"]),
        "--layer-window", str(optimizer["layer_window"]),
        "--checkpoint-every", str(optimizer["checkpoint_every"]),
    ]
    stage_records = []
    for batch in config["batches"]:
        for nonnulls in config["nonnull_counts"]:
            for condition in ("clean", "signed_square", "persistent_noise"):
                condition_root = data_root / str(batch["id"]) / f"n{nonnulls}" / condition
                input_path = condition_root / "learned_instances.csv"
                for procedure, runner, source_names in (
                    (
                        "procedure1",
                        solver / "optimize_method1_learned.py",
                        ("method1_results.csv", "method1_memberships.csv"),
                    ),
                    (
                        "procedure2",
                        solver / "run_learned_asymmetric_closed_ebh_2.py",
                        ("original_formula_results.csv", "original_formula_memberships.csv"),
                    ),
                ):
                    stage = condition_root / f"_{procedure}_stage"
                    command = [
                        sys.executable,
                        "-B",
                        str(runner),
                        "--input",
                        str(input_path),
                        "--output-dir",
                        str(stage),
                        *common_args,
                    ]
                    if args.resume:
                        command.append("--resume")
                    run(command, root, output / "logs" / f"{batch['id']}_n{nonnulls}_{condition}_{procedure}.log")
                    result_source = stage / source_names[0]
                    membership_source = stage / source_names[1]
                    verify_completed(result_source)
                    shutil.copy2(result_source, condition_root / f"{procedure}_results.csv")
                    shutil.copy2(
                        membership_source,
                        condition_root / f"{procedure}_membership.csv",
                    )
                    stage_records.append(
                        {
                            "batch": batch["id"],
                            "nonnulls": nonnulls,
                            "condition": condition,
                            "procedure": procedure,
                            "status": "completed",
                        }
                    )
    (output / "stage_status.json").write_text(
        json.dumps({"stages": stage_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compare_reproduction(root, output)
    print(f"Completed {len(stage_records)} learned stages under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

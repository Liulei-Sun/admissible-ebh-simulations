#!/usr/bin/env python3
"""Portable fail-closed verifier for the publication simulation package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys

from audit_results import run_audit
from create_manifest import MANIFEST, SIDECAR
from study_common import format4, load_config, require, sha256_file


BANNED_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", "recovery", "checkpoints",
    "retained_original", "original_r1000", "provenance",
}
BANNED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".zip", ".tar", ".gz"}
BANNED_BASENAMES = {
    "choice2_results.csv", "choice2_results_primary.csv", "method1_results_primary.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Skip the expensive independent certification pass.",
    )
    return parser.parse_args()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    require(
        value == path.as_posix()
        and not path.is_absolute()
        and value not in {"", "."}
        and ".." not in path.parts
        and "\\" not in value,
        f"unsafe manifest path: {value!r}",
    )
    return path


def verify_manifest(root: Path) -> dict:
    manifest_path = root / MANIFEST
    sidecar_path = root / SIDECAR
    require(manifest_path.is_file() and sidecar_path.is_file(), "missing manifest or sidecar")
    sidecar = sidecar_path.read_text(encoding="utf-8").split()
    require(
        len(sidecar) >= 2 and sidecar[0] == sha256_file(manifest_path) and sidecar[-1] == MANIFEST,
        "manifest sidecar mismatch",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "closed-ebh-publication-package-manifest-v1", "manifest schema mismatch")
    require(manifest.get("package_root") == root.name, "package root identity mismatch")
    entries = manifest.get("files")
    require(isinstance(entries, list), "manifest file inventory is malformed")
    declared = {}
    folded = set()
    for item in entries:
        require(isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"}, "manifest entry schema mismatch")
        relative = safe_relative(str(item["path"]))
        name = relative.as_posix()
        require(name not in declared, f"duplicate manifest entry: {name}")
        require(name.casefold() not in folded, f"case-colliding manifest entry: {name}")
        folded.add(name.casefold())
        declared[name] = item
    observed = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"symbolic links are forbidden: {path}")
        if not path.is_file() or path.name in {MANIFEST, SIDECAR}:
            continue
        relative = path.relative_to(root).as_posix()
        parts = PurePosixPath(relative).parts
        require(not any(part in BANNED_PARTS for part in parts), f"banned path: {relative}")
        require(path.suffix.lower() not in BANNED_SUFFIXES, f"banned file type: {relative}")
        require(path.name not in BANNED_BASENAMES, f"banned generic alias: {relative}")
        observed[relative] = path
    require(set(observed) == set(declared), "manifest inventory differs from extracted files")
    for relative, path in observed.items():
        item = declared[relative]
        require(path.stat().st_size == int(item["bytes"]), f"size mismatch: {relative}")
        require(sha256_file(path) == str(item["sha256"]), f"hash mismatch: {relative}")
    require(int(manifest.get("file_count", -1)) == len(declared), "manifest file count mismatch")
    require(int(manifest.get("total_bytes", -1)) == sum(path.stat().st_size for path in observed.values()), "manifest byte total mismatch")
    return manifest


def verify_layout(root: Path) -> None:
    config = load_config(root)
    expected_batches = {str(item["id"]) for item in config["batches"]}
    observed_batches = {path.name for path in (root / "data").iterdir() if path.is_dir()}
    require(observed_batches == expected_batches, "unexpected batch directory")
    for batch in config["batches"]:
        batch_root = root / "data" / str(batch["id"])
        require((batch_root / "seeds.csv").is_file(), "missing seed table")
        observed_panels = {path.name for path in batch_root.iterdir() if path.is_dir()}
        require(observed_panels == {"n20", "n30"}, "unexpected nonnull panel")
        for nonnulls in config["nonnull_counts"]:
            panel = batch_root / f"n{nonnulls}"
            conditions = {path.name for path in panel.iterdir() if path.is_dir()}
            require(conditions == {"clean", "signed_square", "persistent_noise"}, "unexpected condition directory")
            require((panel / "testing_mean_results.csv").is_file(), "missing testing result")
            for condition in conditions:
                condition_root = panel / condition
                required = {
                    "learned_instances.csv",
                    "procedure1_results.csv", "procedure1_membership.csv",
                    "procedure2_results.csv", "procedure2_membership.csv",
                    "full_mean_results.csv" if condition == "clean" else "naive_mean_results.csv",
                }
                observed = {path.name for path in condition_root.iterdir() if path.is_file()}
                require(observed == required, f"condition file inventory mismatch: {condition_root}")


def verify_no_stale_designs(root: Path) -> None:
    targets = [
        root / "README.md", root / "RELEASE_NOTES.md", root / "config" / "study.json",
        *[
            path
            for path in sorted((root / "code").glob("*.py"))
            if path.name != "verify_package.py"
        ],
    ]
    banned = (
        "sigma035", "adaptive_pilot", "white_noise", "retained_original",
        "original_r1000", "scale_delta", "noise_sigma",
    )
    for path in targets:
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            require(token not in text, f"stale design token {token!r} in {path.relative_to(root)}")


def verify_publication(root: Path, audit: dict) -> None:
    path = root / "publication" / "learning_asymmetric_weights.tex"
    require(path.is_file(), "missing publication subsection")
    text = path.read_text(encoding="utf-8")
    require("N(0,0.1)" in text and "20\\varepsilon_i" in text, "publication error model mismatch")
    require("white-noise" not in text.lower() and "0.35\\varepsilon" not in text, "publication contains stale condition")
    markers = {
        "clean": r"\textit{Panel A: clean training data}",
        "signed_square": r"\textit{Panel B1: signed-square power distortion}",
        "persistent_noise": r"\textit{Panel B2: persistent medicine-specific error",
    }
    positions = {condition: text.index(marker) for condition, marker in markers.items()}
    require(
        positions["clean"] < positions["signed_square"] < positions["persistent_noise"],
        "publication panel order mismatch",
    )
    blocks = {
        "clean": text[positions["clean"] : positions["signed_square"]],
        "signed_square": text[
            positions["signed_square"] : positions["persistent_noise"]
        ],
        "persistent_noise": text[positions["persistent_noise"] :],
    }
    prefixes = {
        "full_mean": r"Mean $\overline{\mathrm{eBH}}$, full data",
        "testing_mean": r"Mean $\overline{\mathrm{eBH}}$, testing sample only",
        "testing_ebh": r"$\mathrm{eBH}$, testing sample only",
        "procedure1": "Learned asymmetric closed eBH 1",
        "procedure2": "Learned asymmetric closed eBH 2",
        "naive_mean": r"Naive mean $\overline{\mathrm{eBH}}$, pooled full data",
    }
    for cell in audit["results"]:
        row = (
            f"{prefixes[cell['method']]} & {cell['nonnulls']} & "
            f"{format4(cell['TPR'])} ({format4(cell['TPR_MCSE'])})"
            f" & {format4(cell['FDR'])} ({format4(cell['FDR_MCSE'])})"
        )
        require(
            row in blocks[cell["condition"]],
            "publication table row missing: "
            f"{cell['condition']}/{cell['nonnulls']}/{cell['method']}",
        )
    require(
        sum(" & 20 & " in line or " & 30 & " in line for line in text.splitlines())
        == 30,
        "publication table must contain exactly 30 result rows",
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    require(root.is_dir(), "package root does not exist")
    manifest = verify_manifest(root)
    verify_layout(root)
    verify_no_stale_designs(root)
    audit = run_audit(root, write=False, full=not args.structural_only)
    verify_publication(root, audit)
    print(
        json.dumps(
            {
                "status": "verified",
                "verification_mode": (
                    "structural_only" if args.structural_only else "full"
                ),
                "payload_files": manifest["file_count"],
                "learned_rows": audit["counts"]["learned_result_rows"],
                "independently_feasibility_recertified_this_run": (
                    0
                    if args.structural_only
                    else audit["counts"][
                        "learned_sets_independently_feasibility_recertified"
                    ]
                ),
                "published_full_recertification_record": audit["counts"][
                    "learned_sets_independently_feasibility_recertified"
                ],
                "conditions": list(load_config(root)["conditions"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

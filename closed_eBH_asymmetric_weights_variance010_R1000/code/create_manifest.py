#!/usr/bin/env python3
"""Create the strict SHA-256 inventory for a completed publication package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from study_common import atomic_json, load_config, sha256_file


MANIFEST = "MANIFEST_SHA256.json"
SIDECAR = "MANIFEST_SHA256.json.sha256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    config = load_config(root)
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symbolic links are forbidden: {path}")
        if not path.is_file() or path.name in {MANIFEST, SIDECAR}:
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "schema": "closed-ebh-publication-package-manifest-v1",
        "package_root": root.name,
        "study": {
            "K": config["K"],
            "alpha": config["alpha"],
            "replications": config["replications"],
            "nonnull_counts": config["nonnull_counts"],
            "conditions": list(config["conditions"]),
            "persistent_error_variance": config["conditions"]["persistent_noise"]["error_variance"],
        },
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "files": entries,
    }
    manifest_path = root / MANIFEST
    atomic_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    sidecar = root / SIDECAR
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(f"{digest}  {MANIFEST}\n", encoding="utf-8")
    temporary.replace(sidecar)
    print(json.dumps({"files": len(entries), "manifest_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

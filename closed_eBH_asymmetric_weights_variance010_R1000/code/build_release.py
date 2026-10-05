#!/usr/bin/env python3
"""Verify and build a single-root publication ZIP, then verify extraction."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run_verifier(root: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(root / "code" / "verify_package.py"), str(root)],
        cwd=root,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"verification failed: {root}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError("output directory already exists")
    run_verifier(root)
    output_dir.mkdir(parents=True)
    zip_path = output_dir / f"{root.name}_publication_ready.zip"
    sidecar_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    with zipfile.ZipFile(
        zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True
    ) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive.write(path, arcname=f"{root.name}/{relative}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len({name.casefold() for name in names}) != len(names):
            raise RuntimeError("ZIP contains duplicate or case-colliding names")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or member.parts[0] != root.name:
                raise RuntimeError(f"unsafe ZIP member: {name}")
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
    digest = file_sha256(zip_path)
    sidecar_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="closed_ebh_publication_verify_") as temporary:
        extraction = Path(temporary)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extraction)
        run_verifier(extraction / root.name)
    print(f"Created {zip_path}\nSHA-256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

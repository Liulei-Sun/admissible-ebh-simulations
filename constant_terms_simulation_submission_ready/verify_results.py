#!/usr/bin/env python3
"""Compare a fresh paper run with the committed reference outputs."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def compare_csv(reference: Path, reproduced: Path) -> int:
    with reference.open(newline="", encoding="utf-8") as handle:
        expected_reader = csv.DictReader(handle)
        expected = list(expected_reader)
        fields = expected_reader.fieldnames
    with reproduced.open(newline="", encoding="utf-8") as handle:
        actual_reader = csv.DictReader(handle)
        actual = list(actual_reader)
        if fields != actual_reader.fieldnames:
            raise AssertionError(f"{reference.name}: CSV columns differ")
    if len(expected) != len(actual):
        raise AssertionError(f"{reference.name}: CSV row counts differ")
    compared = 0
    for row_number, (left, right) in enumerate(zip(expected, actual, strict=True), 2):
        for key in fields or ():
            # Python patch releases may differ; retain both versions in the files.
            if key == "python_version":
                continue
            a, b = left[key], right[key]
            try:
                x, y = float(a), float(b)
            except ValueError:
                equal = a == b
            else:
                equal = math.isfinite(x) and math.isfinite(y) and math.isclose(
                    x, y, rel_tol=1e-12, abs_tol=1e-14
                )
            if not equal:
                raise AssertionError(
                    f"{reference.name}:{row_number}:{key}: {a!r} != {b!r}"
                )
            compared += 1
    return compared


def verify(reference: Path, reproduced: Path) -> None:
    compared = 0
    for filename in ("constant_terms_results.csv", "selector_comparison.csv"):
        compared += compare_csv(reference / filename, reproduced / filename)
    for filename in (
        "constant_terms_tpr_table.tex",
        "constant_terms_complete_table.tex",
        "constant_terms_complete_results.md",
    ):
        if (reference / filename).read_text(encoding="utf-8") != (
            reproduced / filename
        ).read_text(encoding="utf-8"):
            raise AssertionError(f"{filename}: rendered summary differs")
    print(
        f"PASS: {compared} CSV fields and all three formatted summaries match "
        "the reference (numeric rtol=1e-12, atol=1e-14; Python version excluded)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reproduced", type=Path, help="directory from a fresh default run")
    parser.add_argument("--reference", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    verify(args.reference, args.reproduced)


if __name__ == "__main__":
    main()

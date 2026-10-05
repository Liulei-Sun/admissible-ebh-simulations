#!/usr/bin/env python3
"""Run repository checks without modifying either published result bundle."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASYM = ROOT / "closed_eBH_asymmetric_weights_variance010_R1000"
CONST = ROOT / "constant_terms_simulation_submission_ready"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-only", action="store_true",
                        help="Skip recertifying all 12,000 stored learned sets.")
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    steps = [
        (CONST, ["-m", "unittest", "discover", "-s", ".", "-p", "test*.py", "-v"]),
        (CONST, ["run_simulation.py", "--self-test-only"]),
        (ASYM, ["-m", "unittest", "discover", "-s", "tests", "-v"]),
        (ASYM, ["code/verify_package.py", "."] +
         (["--structural-only"] if args.structural_only else [])),
        (ROOT, ["scripts/verify_manuscript_tables.py"]),
    ]
    for cwd, command in steps:
        print(f"\n[{cwd.name}] {' '.join(command)}", flush=True)
        subprocess.run([sys.executable, "-B", *command], cwd=cwd,
                       env=env, check=True)
    scope = "structural" if args.structural_only else "full stored-set feasibility"
    print(f"\nPASS: tests, manuscript tables, and {scope} verification.")
    print("This command does not rerun the 12,000 asymmetric optimizations.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Learned asymmetric closed eBH 2 with the requested fixed-floor weights."""

from __future__ import annotations

# Importing the implementation installs its exact oracle, problem hash,
# metadata, checkpoint aliases, and worker entry point on the shared runner.
import optimize_original_formula_learned as implementation


if __name__ == "__main__":
    raise SystemExit(implementation.runner.main())

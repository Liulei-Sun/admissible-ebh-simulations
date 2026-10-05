# Learning asymmetric weights: publication simulation package

This archive is the reproducibility package for manuscript **Section 7.1,
Learning asymmetric weights (Table 1)**. It contains exactly the three reported
training-data conditions:

1. clean training observations;
2. the signed-square transformation
   `X* = sign(X) |X|^2`; and
3. persistent hypothesis-specific error
   `X*_{t,i} = X_{t,i} + epsilon_i`, where one
   `epsilon_i ~ N(0,0.1)` is shared by all 20 training observations for
   hypothesis `i`.

There are no exploratory noise scales, abandoned white-noise scenarios,
checkpoint files, recovery directories, generic result aliases, or historical
package imports in this release.

## Design

- `K = 200`, `alpha = 0.05`, and 20 or 30 nonnull hypotheses;
- signal mean and e-value tilt both equal `0.45`;
- 20 observations learn the weights and 40 independent observations test;
- two batches of 500 replications use seeds `20260803` and `20260804`;
- the persistent-error standard deviation is the derived value `sqrt(0.1)`;
- the testing sample and testing-only procedures are identical across all
  training-data conditions.

Procedure 1 reports a globally maximum-cardinality certified set. Procedure 2
reports a maximum-cardinality certified set among those containing ordinary
testing-only eBH. At the optimal cardinality, both maximize the sum of testing
e-values. The exact design is recorded in `config/study.json`.

The canonical result files retain the solver's historical status token
`proven_maximal`. In this package it means maximum cardinality over each
method's stated feasible family (unrestricted for Procedure 1, restricted to
eBH supersets for Procedure 2): every
row also has `proof_upper_bound = maximal_size`. The separate
`secondary_status = proven_optimal` field records the optimality of the
testing-e-value secondary objective.

## Layout

- `config/`: the single authoritative study configuration.
- `data/`: the exact solver inputs and canonical outputs for both batches,
  both nonnull counts, and all three reported conditions.
- `code/`: deterministic generation, simulation, audit, verification, and
  packaging programs, plus the exact learned-set solvers.
- `tests/`: exhaustive small-problem oracle tests and package guard tests.
- `results/`: pooled replication-level results, summary tables, and report.
- `audit/`: the independent scientific audit record.
- `publication/`: the paper-ready LaTeX subsection.
- `environment/`: the exact Python package lock used for the final run.

Each condition directory has one solver input plus the canonical results and
full membership matrices for Procedures 1 and 2. Clean training additionally
has the full-data mean result; the two contaminated conditions have their
corresponding naive pooled-mean result. Testing-only mean results are stored
once per batch and nonnull count because they are shared by all conditions.

## Verify the published archive

Create the locked environment and run the portable verifier from the extracted
package root:

```bash
python -m venv ../.venv
../.venv/bin/python -m pip install -r environment/requirements-lock.txt
../.venv/bin/python -B code/verify_package.py .
```

On Windows, replace `../.venv/bin/python` with `..\.venv\Scripts\python.exe`.
The lock was created with Python 3.12.13; the audit also used Python 3.12.14
with the same four pinned packages. Keep the virtual environment outside this
package: its strict inventory deliberately rejects additional files. Keep this
directory under a repository root (with `.git` outside it) and retain its
directory name. Configure the repository with `* -text` in `.gitattributes`
before adding files, so Git preserves the checksummed CSV and source bytes.
Verification checks the strict SHA-256 inventory, all 1.2 million input rows,
12,000 learned result rows, 2.4 million membership rows, deterministic random
streams, all formulas and stored metrics, proof/status metadata, Procedure-2
containment, the 30,000 pooled metric rows, and every published table cell. It
also independently recertifies the feasibility of every reported learned set.
The transparent record of 21 legacy Procedure-2 rows recomputed after this
full recertification is `audit/correction_record.json`; none belongs to the
persistent-error condition.

Run the solver unit tests with:

```bash
../.venv/bin/python -B -m unittest discover -s tests -v
```

## Reproduce the simulation

The checked-in `data/` files are the authoritative run. To regenerate inputs
from the two fixed seeds and run all 24 learned stages in a separate directory:

```bash
../.venv/bin/python -B code/run_study.py \
  --root . \
  --output-dir ../independent_reproduction
```

This is computationally intensive. Use `--generate-only` to regenerate and
validate the Gaussian, signed-square, and persistent-error inputs without
running the optimizers. Existing published files are never overwritten. The
full run compares every learned membership, cardinality, objective, proof
status, and scientific metric against the publication data and writes a fresh
`results/summary.csv`, `results/replication_metrics.csv`, and
`reproduction_comparison.json` under the output directory. Timing, node counts,
and search histories are machine dependent and are excluded from comparisons.
Use the same command with `--resume` after interruption; checkpoints are bound
to the exact inputs, procedure, alpha, and solver source code. Membership,
metrics, and objective values are rechecked, and incomplete proofs are rerun.
After editing solver code, start a new output directory. Historical checkpoints
without a solver-source signature cannot be resumed by this revised runner.

The full verifier independently checks feasibility of all 12,000 stored
learned sets and recomputes direct methods and tables. It checks stored
optimality metadata but does not independently re-optimize all 12,000 sets.
Exhaustive small-problem tests and representative optimizer replays provide
additional optimization checks; a complete fresh optimizer run is substantially
more expensive than verification.

`audit/audit.json` preserves the original publication audit. A later verifier
compares its scientific counts, output hashes, and numerical results with the
current audit; historical runtime versions and source hashes are descriptive.
Current source integrity is checked by `MANIFEST_SHA256.json`. Numeric result
comparisons allow only documented floating-point roundoff tolerances.

## Rebuild the ZIP

After verification, a pristine extraction can rebuild a single-root release
archive without modifying the package:

```bash
../.venv/bin/python -B code/build_release.py \
  --root . \
  --output-dir ../rebuilt_release
```

The builder verifies before and after archiving and refuses to overwrite an
existing output.

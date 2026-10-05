# Publication-release notes

This release keeps every scientific input and canonical output used by the
reported clean, signed-square, and variance-0.1 persistent-error panels.

The package was simplified by removing exploratory parameter runs, alternative
persistent-error scales, historical import maps and duplicate archives,
recovery/checkpoint artifacts, stale launchers for other study designs, generic
aliases, and machine-specific absolute paths. The exact large-set adjusted
oracle used for final verification is included together with its exhaustive
tests.

No reported testing-only result changed. The final variance-0.1 results are the
values in `results/summary.csv` and `publication/learning_asymmetric_weights.tex`.

A new exact feasibility pass over all learned sets exposed 21 stale
Procedure-2 rows in three legacy clean/signed-square stages. Their earlier
large-set numerical separator had falsely certified infeasible sets. Those
rows were recomputed with the exact dyadic outside-prefix separator, their
cardinalities over eBH supersets and secondary objectives were reproved, and the complete
stage result and membership files were regenerated. The affected rows and
before/after hashes are recorded in `audit/correction_record.json`. All
persistent-error rows, all Procedure-1 rows, and every testing-only result were
unchanged.

## Repository audit (October 2026)

The GitHub preparation audit preserves all scientific input/output CSVs. It
clarifies that this is manuscript Section 7.1 and that Procedure 2 optimizes
over certified eBH supersets. Installation uses an external virtual environment
and tests use `-B`, preserving the strict package inventory. Checkpoint resume
now rejects changed problems and recomputes rows lacking secondary optimality.
The full reproduction command now compares scientific results and rebuilds the
summary from regenerated data; regression tests cover stale checkpoints,
modified generated inputs, and changed membership matrices.

The revised verifier preserves the original `audit/audit.json` and compares
scientific evidence separately from historical runtime/source metadata, so
another Python patch version can verify the archive without rewriting its
history. Numeric comparisons have explicit tolerances. The exact adjusted
Gray-code separator now covers size 23, the largest published adjusted set.
Resume requires the complete generated-file inventory and checks solver-source
signatures as well as membership, metrics, and objective values.

Full archive verification checks all learned-set feasibility and table cells;
it does not replace a fresh global optimization of every stored row. The
repository audit uses representative fresh optimizer replays spanning both
batches, non-null counts, training conditions, and methods. No scientific
result has been changed by this audit.

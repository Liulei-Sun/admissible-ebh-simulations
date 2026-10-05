# Reproducible simulations for admissible closed eBH procedures

Simulation code and reference results for **Admissibility and Complete Classes for False Discovery Rate Control with E-values**, by Liulei Sun and Ruodu Wang.

## Contents

| Manuscript section | Experiment | Package | Result |
|---|---|---|---|
| 7.1 | Learning asymmetric weights | [`closed_eBH_asymmetric_weights_variance010_R1000`](closed_eBH_asymmetric_weights_variance010_R1000/) | Table 1 |
| 7.2 | Effect of admissible constant terms | [`constant_terms_simulation_submission_ready`](constant_terms_simulation_submission_ready/) | Table 2 |

Both experiments use simulated data. The repository contains the data-generating code, fixed random seeds, reference results, tests, and instructions needed to reproduce the reported tables. No external empirical dataset is required. Each package has a detailed README describing its design and algorithms.

The asymmetric package includes its input data, result memberships, and scientific provenance records. Keep its directory name and internal file structure unchanged: its verifier checks a strict SHA-256 inventory. Create virtual environments and new simulation outputs at the repository's top level.

## Environment

The validated environment is CPython 3.12.14 on Windows x86-64 with NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.0, and highspy 1.15.1. `requirements-lock.txt` also pins transitive dependencies. The original reference-result metadata records Python 3.12.13.

From the repository root, install the environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

The commands below use the Windows interpreter path. On Linux or macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Use `-B` to avoid adding bytecode caches to the strictly inventoried package.

## Check the reference results

Run the test suites, structural/data checks, and comparison with manuscript Tables 1 and 2:

```powershell
.\.venv\Scripts\python.exe -B scripts/verify_all.py --structural-only
```

To additionally recertify the feasibility of every stored learned rejection set:

```powershell
.\.venv\Scripts\python.exe -B scripts/verify_all.py
```

The second command checks all 12,000 learned sets. Neither command reruns all asymmetric optimizations. Feasibility checks and checks of stored optimality metadata are distinct from independently resolving every optimization problem.

## Reproduce Section 7.2 and Table 2

```powershell
.\.venv\Scripts\python.exe -B constant_terms_simulation_submission_ready/run_simulation.py --output-dir reproductions/constant_terms
.\.venv\Scripts\python.exe -B constant_terms_simulation_submission_ready/verify_results.py reproductions/constant_terms
```

This runs 1,000 replications at each of the three nonnull counts, using all five reported beta choices and ordinary eBH. The verifier compares both result CSVs and all three formatted summaries with the reference files. The full experiment took several minutes in the validation environment.

The output includes `constant_terms_tpr_table.tex`, `constant_terms_results.csv`, complete diagnostic summaries, selector-comparison results, and runtime metadata. The LaTeX outputs are table fragments for inclusion in a manuscript.

## Reproduce Section 7.1 and Table 1

First, an input-only run can regenerate all inputs and direct-method results from the fixed seeds, comparing them with the reference files:

```powershell
.\.venv\Scripts\python.exe -B closed_eBH_asymmetric_weights_variance010_R1000/code/run_study.py --root closed_eBH_asymmetric_weights_variance010_R1000 --output-dir reproductions/asymmetric_inputs --generate-only
```

For a complete fresh run of all 24 optimization stages:

```powershell
.\.venv\Scripts\python.exe -B closed_eBH_asymmetric_weights_variance010_R1000/code/run_study.py --root closed_eBH_asymmetric_weights_variance010_R1000 --output-dir reproductions/asymmetric_full
```

On successful completion, the runner compares the scientific outputs with the reference results and writes freshly pooled metrics and `results/summary.csv` under the reproduction directory. The archive also contains the publication table fragment at `closed_eBH_asymmetric_weights_variance010_R1000/publication/learning_asymmetric_weights.tex`.

This is a multi-hour computation. The original optimization rows total approximately 17.1 aggregate solver-hours; four workers are configured, and some individual cases are much slower than the median. A timeout is an incomplete run. Consult the package README for validated resume behaviour.

A smaller end-to-end optimizer check is also available:

```powershell
.\.venv\Scripts\python.exe -B scripts/replay_representative.py
```

It solves local replications 1 and 2 in each of all 24 stages, covering both batches, both nonnull counts, all three training conditions, and both learned procedures. It compares 48 fresh solutions with the reference results and writes to a new `reproductions/asymmetric_representative/` directory. It is not a replacement for the complete experiment.

## Additional checks and validation scope

```powershell
.\.venv\Scripts\python.exe -B scripts/independent_metrics_check.py
.\.venv\Scripts\python.exe -B scripts/check_manuscript_footnote.py
```

These independently reconstruct the asymmetric summaries from selected hypotheses and truth labels, and check the two numerical examples in manuscript footnote 3 using exact rational arithmetic.

Local validation completed the following:

- All 40 unit tests and the constant-term exhaustive self-tests passed.
- The complete constant-term experiment reproduced the reference numerical outputs.
- All 30 asymmetric input/direct-result files were regenerated and checked.
- All 12,000 stored learned sets were independently checked for feasibility.
- All 48 representative fresh optimizations reproduced the scientific results.
- Every displayed estimate and Monte Carlo standard error in Tables 1 and 2 matched the reference output.

The complete set of 12,000 asymmetric optimizations was not rerun during this validation. The original data and numerical reference results are preserved. Numerical certification uses the tolerances documented in each package. The continuous-integration workflow checks Windows and Linux after publication; local validation was performed on Windows.

## Versions and citation

`CITATION.cff` identifies the accompanying manuscript and its authors. A publication release should be tagged and archived with a persistent DOI. Cite the specific archived version used for the manuscript so that later repository changes do not change the cited computational record.

The repository's `.gitattributes` preserves the bytes checked by the package manifest across operating systems. `.gitignore` excludes virtual environments, caches, and newly generated reproduction outputs.

## Licence

No licence is assigned by this preparation step. The authors should add their chosen licence before distributing the repository under open-source terms.

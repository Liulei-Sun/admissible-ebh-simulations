# Simulation: effect of admissible constant terms

This package reproduces the second simulation in Section 7.2 and Table 2 of
`JRSSB_main.pdf`. The asymmetric-weight package corresponds to Section 7.1.

## Design

- `K = 5000` hypotheses;
- `alpha = 0.05`;
- non-null counts `500`, `1000`, and `1500`, corresponding to 10%, 20%,
  and 30%;
- independent `N(0,1)` null observations and `N(3,1)` non-null observations;
- matched Gaussian likelihood-ratio e-values `E_i = exp(3 X_i - 9/2)`;
- the base eBH procedure as a benchmark;
- beta values `{0, .004, .80, .90, 1}`;
- `1,000` independent replications, paired across eBH and all beta values; and
- seed `20260719`.

The same `1,000 x 5,000` matrix of standard-normal draws is reused across the
three non-null-count scenarios, with nested non-null sets. Thus replications
within each scenario are independent, while estimates from different scenarios
are correlated through common random numbers.

At `K=5000` and `alpha=.05`, Proposition 6.7 gives

```text
q = floor(1/alpha) = 20,
beta* = min{(.05)(20)/((.05)(21)-1), 5000-20+1}/5000
      = min{20, 4981}/5000
      = .004.
```

Thus the underlying simultaneous weighted-mean closed-eBH procedures at
`beta=0` and `.004` are admissible, including equality at the boundary, while
those at `.80`, `.90`, and `1` are inadmissible. Throughout this package,
`status` and the words *admissible* and *inadmissible* classify the underlying
simultaneous procedure. They do not classify the reported point selector.

## Why the eBH benchmark is contained

The paper proves
`eSC_alpha <=_s closed-eBH_alpha^{lambda(beta)}` for every `beta` in `[0,1]`.
Since the eBH rejection set is self-consistent, it is certified by every member
of this one-parameter family. Equivalently, for an intersection of size `n`,
the sharp quantity checked by the code is at least one. If `n < beta K`,
direct substitution gives

```text
alpha lambda_0^n(beta) + K(1-lambda_0^n(beta))/n - 1
  = K(1-alpha)(1-beta)/(beta K-alpha n) >= 0;
```

if `n >= beta K`, the constant is zero and the quantity is `K/n >= 1`.
This covers every `beta` in `[0,1]`, including the endpoints.

For each replication and beta, the reported point set has maximum cardinality
among certified rejection sets containing the base eBH rejection set. If
multiple maximizers exist, the implementation favors larger e-values and then
uses stable original-index order for exact ties. Therefore containment is
explicit in the reporting rule as well as implied by the theoretical argument.
This conclusion concerns the fixed top-prefix selector used here: certification
alone does not imply that every arbitrary point selector induced by the same
simultaneous procedure must contain eBH.

For these symmetric mergers, an exchange argument shows that if any certified
set of size `r` exists, the top-`r` e-value prefix is certified. Let `k` be the
eBH rejection count, `r_old` the unconstrained maximum certified-prefix size,
and `r_new` the maximum size subject to eBH containment. Because the eBH prefix
is certified, `r_old >= k`; hence the top-`r_old` prefix contains eBH and is
feasible for the revised rule, giving `r_new >= r_old`. Conversely, any revised
feasible set of size `r` yields a certified top-`r` prefix, so `r <= r_old` and
therefore `r_new <= r_old`. Both implementations use the same order and
tie-break, so equality of sizes gives equality of selected sets.

The program recomputes both searches. Across all
`3 x 5 x 1,000 = 15,000` beta-by-replication cases, their maximum sizes and
selected sets agreed. Every reported beta set contained eBH: `15,000/15,000`.

## Run

Requirements:

- Python 3.11 or newer; and
- NumPy 2.3.5, pinned in `requirements.txt`.

The packaged files were generated with Python 3.12.13 and NumPy 2.3.5. The
CSV records both runtime versions. From this package directory, first create
an isolated environment and install the pinned dependency:

```bash
python -m venv .venv
# macOS / Linux:
.venv/bin/python -m pip install -r requirements.txt
# Windows PowerShell:
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands below use `python`; use the interpreter inside `.venv`, or
activate that environment first (`source .venv/bin/activate` on macOS/Linux;
`.venv\Scripts\Activate.ps1` in PowerShell).

```bash
python -m unittest -v
python run_simulation.py --self-test-only
python run_simulation.py --output-dir results_reproduced
python verify_results.py results_reproduced
```

The full run uses all 1,000 replications. `results_reproduced` keeps the
committed `results/` reference intact. Calling `python run_simulation.py`
without an output option overwrites the reference directory. The verifier
compares both CSVs (relative tolerance `1e-12`, absolute tolerance `1e-14`) and
all three formatted summaries. It permits a different Python patch version;
the pinned NumPy version and all design metadata must still match.

For a quick end-to-end run, use:

```bash
python run_simulation.py --hypotheses 40 --nonnulls 4,8,12 --replications 10 --output-dir results_smoke
```

A smoke run is a software check and does not reproduce Table 2. Runtime for
the full run depends on the computer; allow several minutes. The simulation
prints progress at the start of each non-null-count scenario.

The default run first checks both optimized certification calculations against
literal enumeration of every nonempty intersection subset on random small
problems. It also enumerates every eBH-containing rejection set on additional
small problems and verifies that the optimized selector attains the global
constrained maximum. It then regenerates:

- `results/constant_terms_results.csv`, containing the eBH benchmark and beta
  rows, complete design metadata, TPR, marginal MCSE, paired TPR differences,
  FDR diagnostics, containment rates, and selector audits;
- `results/constant_terms_tpr_table.tex`, the paper-ready TPR table;
- `results/constant_terms_complete_table.tex`, a complete diagnostic table;
- `results/selector_comparison.csv`, the direct unconstrained-versus-
  containment-constrained selector audit; and
- `results/constant_terms_complete_results.md`, all numerical results in a
  human-readable form.

Each new run also writes `run_metadata.json` with the exact settings, RNG,
runtime versions, certification tolerance, source SHA-256, and elapsed time.
The original reference directory is preserved with its original provenance.

The separate `unittest` suite uses exact rational arithmetic to check the
all-intersection definition and all rejection subsets on small problems,
including ties, zero e-values, boundary cases, and levels above one half. It
also checks numeric tail stability, invalid CLI arguments, overflow handling,
custom beta labels, seed repeatability, and that the result verifier detects
changed numerical results. These checks are independent of the float-based
enumeration included in the simulation's built-in self-test.

The package includes `RESULTS_SUMMARY.md` and generated LaTeX tables.
The compact table requires `booktabs`, `tabularx`, and `bm`; the complete
diagnostic table additionally uses `adjustbox`. The table files are LaTeX
fragments intended for inclusion in a manuscript, not standalone documents.

The implementation uses an algebraically exact reduction of the intersection
constraints, evaluated in floating point with relative tolerance `2e-12`, and
bounded caching. It does not enumerate all subsets at `K=5000`.

## Main numerical pattern

The empirical TPRs are:

| Procedure | 500 non-nulls | 1,000 non-nulls | 1,500 non-nulls |
|---|---:|---:|---:|
| eBH | .222682 | .350884 | .427490 |
| beta=0 | .275218 | .431957 | .531749 |
| beta=.004 | .275218 | .431957 | .531749 |
| beta=.80 | .275218 | .430644 | .523345 |
| beta=.90 | .272386 | .421131 | .511214 |
| beta=1 | .262304 | .409030 | .498039 |

The displayed inadmissible choices never exceed the two admissible choices:
`beta=.80` ties them in the 10% scenario and is lower in the other two, while
`beta=.90` and `1` are lower in all three scenarios. All beta procedures have
TPR at least that of eBH in every replication because their reported rejection
sets contain eBH.

This finite, model-specific pattern does not prove domination, an expected-TPR
ordering, monotonicity for unreported beta values, or inferiority of every
inadmissible procedure. Procedures at different beta values are generally
incomparable. Inadmissibility means that a suitably adjusted simultaneous
procedure strongly dominates the given simultaneous procedure; the dominator
need not be a smaller-beta member of this displayed family. Admissibility is a
no-uniform-domination property, not an ordering under every distribution.

## Reported beta grid

This package reproduces the grid reported in Table 2:
`{0, .004, .80, .90, 1}`. Additional exploratory beta values are outside the
scope of this release. The identical selections at `0` and `.004` are an
observed property of the specified experiment.

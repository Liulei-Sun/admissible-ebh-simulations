# Verified complete results

## Design and status labels

The simulation uses `K=5000`, `alpha=0.05`, non-null counts `500`, `1000`,
and `1500`, independent `N(0,1)` nulls and `N(3,1)` non-nulls, e-values
`E_i=exp(3X_i-9/2)`, beta values `{0,.004,.80,.90,1}`, `1,000` paired
replications, and seed `20260719`. A common `1,000 x 5,000` standard-normal
matrix is reused across the three non-null-count scenarios. Scenario-level
estimates are therefore correlated, while replications within each scenario
remain independent.

For `K=5000` and `alpha=0.05`, the exact boundary is

```text
q = floor(1/alpha) = 20,
beta* = min{0.05(20)/(0.05(21)-1), 5000-20+1}/5000
      = min{20, 4981}/5000
      = 0.004.
```

Accordingly, the underlying simultaneous procedures at beta `0` and `.004`
are admissible, including equality at `.004`, while those at `.80`, `.90`,
and `1` are inadmissible. These labels do not classify the reported point
selectors.

The paper's eSC-domination condition holds for every beta in `[0,1]`, so each
member of the family certifies the base eBH rejection set.
The reported point selector also imposes eBH containment explicitly.

## Main TPR estimates

Parentheses contain marginal Monte Carlo standard errors.

| Procedure | 500 non-nulls | 1,000 non-nulls | 1,500 non-nulls |
|---|---:|---:|---:|
| eBH | 0.222682 (0.001041476) | 0.350884 (0.000767410) | 0.427490 (0.000600482) |
| beta=0 | 0.275218 (0.001094875) | 0.431957 (0.000747640) | 0.531749 (0.000572546) |
| beta=.004 | 0.275218 (0.001094875) | 0.431957 (0.000747640) | 0.531749 (0.000572546) |
| beta=.80 | 0.275218 (0.001094875) | 0.430644 (0.000752204) | 0.523345 (0.000588361) |
| beta=.90 | 0.272386 (0.001100401) | 0.421131 (0.000769069) | 0.511214 (0.000599027) |
| beta=1 | 0.262304 (0.001105934) | 0.409030 (0.000782845) | 0.498039 (0.000608917) |

The two admissible choices select the same reported sets in every replication.
No displayed inadmissible choice has higher TPR than those choices: beta `.80`
ties them in the 10% scenario and is lower in the other two; beta `.90` and
`1` are lower in all three scenarios.

## Paired comparisons with beta zero

Because all methods use the same e-value vector within a replication, paired
differences are the appropriate finite-run comparisons. The following table
gives `TPR(beta) - TPR(beta=0)`; parentheses contain paired Monte Carlo
standard errors. These are also the differences relative to beta `.004`.

| beta | 500 non-nulls | 1,000 non-nulls | 1,500 non-nulls |
|---:|---:|---:|---:|
| .004 | 0 (0) | 0 (0) | 0 (0) |
| .80 | 0 (0) | -0.001313 (0.000042436) | -0.008403 (0.000103776) |
| .90 | -0.002832 (0.000122877) | -0.010826 (0.000149017) | -0.020535 (0.000167746) |
| 1 | -0.012914 (0.000275268) | -0.022927 (0.000228600) | -0.033709 (0.000213600) |

## FDR diagnostics

The empirical FDR estimates are:

| Procedure | 500 non-nulls | 1,000 non-nulls | 1,500 non-nulls |
|---|---:|---:|---:|
| eBH | .003522 | .004212 | .004089 |
| beta=0 | .005425 | .007206 | .007798 |
| beta=.004 | .005425 | .007206 | .007798 |
| beta=.80 | .005425 | .007153 | .007395 |
| beta=.90 | .005337 | .006753 | .006870 |
| beta=1 | .004874 | .006225 | .006356 |

The range is `0.003522` to `0.007798`, well below `alpha=.05`. These are
empirical diagnostics; the theoretical FDR guarantee does not rely on them.

## Selector-equivalence and containment audit

The revised rule reports a maximum-size certified rejection set containing
the eBH rejection set. Among multiple maximizers, it favors larger e-values
and then uses stable original-index order for exact ties.
This containment statement is specific to that fixed top-prefix selector;
it is not a claim about every arbitrary point selector induced by the same
simultaneous procedure.

For these symmetric mergers, an exchange argument shows that if any certified
set of size `r` exists, the top-`r` e-value prefix is certified. Because the
eBH prefix is certified for every beta, the unconstrained maximum top prefix
already contains eBH. The unconstrained and containment-constrained searches
therefore have the same maximum size; their common ordering and tie-break imply
that they select the same set.

This was checked in all `3 x 5 x 1,000 = 15,000` beta-by-replication cases:

- eBH containment: `15,000/15,000` (`100%`);
- containment violations: `0`;
- changed selected sets between the two searches: `0/15,000`; and
- maximum absolute rejection-count change: `0`.

The base eBH benchmark contributes another `3 x 1,000 = 3,000` evaluations,
but it is not part of the `15,000` beta-selector audit count.

## Interpretation boundary

These findings are finite Monte Carlo results for the displayed Gaussian
settings. They do not prove strong domination, an expected-TPR ordering,
monotonicity in beta, or lower power for every inadmissible simultaneous
procedure. Procedures at different beta values are generally incomparable.
The theoretical inadmissibility statement guarantees domination by a suitably
adjusted simultaneous procedure, not necessarily by beta `0`, beta `.004`, or
another member of this one-parameter family.

Full-precision numerical results and all diagnostics are in
`results/constant_terms_results.csv` and
`results/constant_terms_complete_results.md`.

# Complete simulation results

Nominal FDR level: `0.05`. Paired replications per scenario: `1,000`.

Admissibility status refers to the underlying simultaneous weighted-mean closed-eBH procedure. It does not classify the reported point selector.

The revised selector explicitly requires the reported maximum-size certified set to contain the eBH rejection set. The original and constrained searches had the same maximum prefix size. Because both use the same descending-e-value order and fixed original-index tie-break, they selected the same set. Every beta selection contained eBH, and there were **0 changed sets among 15,000 beta-replication cases**.

TPR and FDR entries show Monte Carlo standard errors in parentheses. The TPR gain and its SE are paired against eBH.

## 500 non-nulls

| Procedure | Underlying simultaneous status | TPR (MCSE) | FDR (MCSE) | Mean R | TPR gain vs eBH (paired SE) | Mean / min R gain | eBH containment | Old/new changed sets |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eBH | benchmark | 0.222682 (0.001041) | 0.003522 (0.000177) | 111.743 | +0.000000 (0.000000) | +0.000 / +0 | -- | -- |
| beta=0 | admissible | 0.275218 (0.001095) | 0.005425 (0.000196) | 138.370 | +0.052536 (0.000500) | +26.627 / +9 | 100.0% | 0/1000 |
| beta=0.004 | admissible | 0.275218 (0.001095) | 0.005425 (0.000196) | 138.370 | +0.052536 (0.000500) | +26.627 / +9 | 100.0% | 0/1000 |
| beta=0.80 | inadmissible | 0.275218 (0.001095) | 0.005425 (0.000196) | 138.370 | +0.052536 (0.000500) | +26.627 / +9 | 100.0% | 0/1000 |
| beta=0.90 | inadmissible | 0.272386 (0.001100) | 0.005337 (0.000196) | 136.934 | +0.049704 (0.000470) | +25.191 / +9 | 100.0% | 0/1000 |
| beta=1.00 | inadmissible | 0.262304 (0.001106) | 0.004874 (0.000191) | 131.807 | +0.039622 (0.000382) | +20.064 / +8 | 100.0% | 0/1000 |

## 1000 non-nulls

| Procedure | Underlying simultaneous status | TPR (MCSE) | FDR (MCSE) | Mean R | TPR gain vs eBH (paired SE) | Mean / min R gain | eBH containment | Old/new changed sets |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eBH | benchmark | 0.350884 (0.000767) | 0.004212 (0.000107) | 352.376 | +0.000000 (0.000000) | +0.000 / +0 | -- | -- |
| beta=0 | admissible | 0.431957 (0.000748) | 0.007206 (0.000126) | 435.102 | +0.081073 (0.000376) | +82.726 / +54 | 100.0% | 0/1000 |
| beta=0.004 | admissible | 0.431957 (0.000748) | 0.007206 (0.000126) | 435.102 | +0.081073 (0.000376) | +82.726 / +54 | 100.0% | 0/1000 |
| beta=0.80 | inadmissible | 0.430644 (0.000752) | 0.007153 (0.000126) | 433.757 | +0.079760 (0.000363) | +81.381 / +55 | 100.0% | 0/1000 |
| beta=0.90 | inadmissible | 0.421131 (0.000769) | 0.006753 (0.000124) | 424.003 | +0.070247 (0.000318) | +71.627 / +48 | 100.0% | 0/1000 |
| beta=1.00 | inadmissible | 0.409030 (0.000783) | 0.006225 (0.000121) | 411.600 | +0.058146 (0.000264) | +59.224 / +40 | 100.0% | 0/1000 |

## 1500 non-nulls

| Procedure | Underlying simultaneous status | TPR (MCSE) | FDR (MCSE) | Mean R | TPR gain vs eBH (paired SE) | Mean / min R gain | eBH containment | Old/new changed sets |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eBH | benchmark | 0.427490 (0.000600) | 0.004089 (0.000078) | 643.872 | +0.000000 (0.000000) | +0.000 / +0 | -- | -- |
| beta=0 | admissible | 0.531749 (0.000573) | 0.007798 (0.000096) | 803.899 | +0.104259 (0.000315) | +160.027 / +120 | 100.0% | 0/1000 |
| beta=0.004 | admissible | 0.531749 (0.000573) | 0.007798 (0.000096) | 803.899 | +0.104259 (0.000315) | +160.027 / +120 | 100.0% | 0/1000 |
| beta=0.80 | inadmissible | 0.523345 (0.000588) | 0.007395 (0.000094) | 790.872 | +0.095855 (0.000282) | +147.000 / +111 | 100.0% | 0/1000 |
| beta=0.90 | inadmissible | 0.511214 (0.000599) | 0.006870 (0.000091) | 772.132 | +0.083724 (0.000248) | +128.260 / +94 | 100.0% | 0/1000 |
| beta=1.00 | inadmissible | 0.498039 (0.000609) | 0.006356 (0.000088) | 751.844 | +0.070549 (0.000210) | +107.972 / +79 | 100.0% | 0/1000 |

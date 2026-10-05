# Verified R=1000 asymmetric-weight simulation

All values below are means over 1,000 replications; Monte Carlo standard errors are in parentheses.

## Clean training data

### 20 nonnull hypotheses

| Procedure | TPR (MCSE) | FDR (MCSE) |
|---|---:|---:|
| Mean eBH, full data | 0.5370 (0.0043) | 0.0071 (0.0008) |
| Mean eBH, testing sample only | 0.1790 (0.0042) | 0.0036 (0.0012) |
| eBH, testing sample only | 0.1584 (0.0039) | 0.0026 (0.0012) |
| Learned asymmetric closed eBH 1 | 0.4708 (0.0055) | 0.0044 (0.0006) |
| Learned asymmetric closed eBH 2 | 0.2478 (0.0051) | 0.0036 (0.0012) |

### 30 nonnull hypotheses

| Procedure | TPR (MCSE) | FDR (MCSE) |
|---|---:|---:|
| Mean eBH, full data | 0.6027 (0.0034) | 0.0056 (0.0005) |
| Mean eBH, testing sample only | 0.2506 (0.0040) | 0.0040 (0.0007) |
| eBH, testing sample only | 0.2096 (0.0038) | 0.0029 (0.0007) |
| Learned asymmetric closed eBH 1 | 0.5174 (0.0049) | 0.0041 (0.0005) |
| Learned asymmetric closed eBH 2 | 0.3500 (0.0046) | 0.0043 (0.0006) |

## Signed-square training distortion

### 20 nonnull hypotheses

| Procedure | TPR (MCSE) | FDR (MCSE) |
|---|---:|---:|
| Mean eBH, testing sample only | 0.1790 (0.0042) | 0.0036 (0.0012) |
| eBH, testing sample only | 0.1584 (0.0039) | 0.0026 (0.0012) |
| Learned asymmetric closed eBH 1 | 0.2742 (0.0060) | 0.0020 (0.0006) |
| Learned asymmetric closed eBH 2 | 0.2304 (0.0050) | 0.0033 (0.0012) |
| Naive mean eBH, pooled full data | 0.7652 (0.0032) | 0.0710 (0.0020) |

### 30 nonnull hypotheses

| Procedure | TPR (MCSE) | FDR (MCSE) |
|---|---:|---:|
| Mean eBH, testing sample only | 0.2506 (0.0040) | 0.0040 (0.0007) |
| eBH, testing sample only | 0.2096 (0.0038) | 0.0029 (0.0007) |
| Learned asymmetric closed eBH 1 | 0.2747 (0.0059) | 0.0016 (0.0004) |
| Learned asymmetric closed eBH 2 | 0.3151 (0.0046) | 0.0038 (0.0006) |
| Naive mean eBH, pooled full data | 0.7945 (0.0025) | 0.0580 (0.0015) |

## Persistent error, epsilon_i ~ N(0,0.1)

### 20 nonnull hypotheses

| Procedure | TPR (MCSE) | FDR (MCSE) |
|---|---:|---:|
| Mean eBH, testing sample only | 0.1790 (0.0042) | 0.0036 (0.0012) |
| eBH, testing sample only | 0.1584 (0.0039) | 0.0026 (0.0012) |
| Learned asymmetric closed eBH 1 | 0.1768 (0.0044) | 0.0045 (0.0010) |
| Learned asymmetric closed eBH 2 | 0.2016 (0.0044) | 0.0032 (0.0012) |
| Naive mean eBH, pooled full data | 0.5388 (0.0042) | 0.0708 (0.0023) |

### 30 nonnull hypotheses

| Procedure | TPR (MCSE) | FDR (MCSE) |
|---|---:|---:|
| Mean eBH, testing sample only | 0.2506 (0.0040) | 0.0040 (0.0007) |
| eBH, testing sample only | 0.2096 (0.0038) | 0.0029 (0.0007) |
| Learned asymmetric closed eBH 1 | 0.2038 (0.0041) | 0.0040 (0.0007) |
| Learned asymmetric closed eBH 2 | 0.2857 (0.0043) | 0.0038 (0.0006) |
| Naive mean eBH, pooled full data | 0.5853 (0.0033) | 0.0581 (0.0017) |

## Containment of testing-only eBH

- Clean training data, 20 nonnulls, Learned asymmetric closed eBH 1: 872/1000.
- Clean training data, 20 nonnulls, Learned asymmetric closed eBH 2: 1000/1000.
- Clean training data, 30 nonnulls, Learned asymmetric closed eBH 1: 796/1000.
- Clean training data, 30 nonnulls, Learned asymmetric closed eBH 2: 1000/1000.
- Signed-square training distortion, 20 nonnulls, Learned asymmetric closed eBH 1: 444/1000.
- Signed-square training distortion, 20 nonnulls, Learned asymmetric closed eBH 2: 1000/1000.
- Signed-square training distortion, 30 nonnulls, Learned asymmetric closed eBH 1: 234/1000.
- Signed-square training distortion, 30 nonnulls, Learned asymmetric closed eBH 2: 1000/1000.
- Persistent error, epsilon_i ~ N(0,0.1), 20 nonnulls, Learned asymmetric closed eBH 1: 338/1000.
- Persistent error, epsilon_i ~ N(0,0.1), 20 nonnulls, Learned asymmetric closed eBH 2: 1000/1000.
- Persistent error, epsilon_i ~ N(0,0.1), 30 nonnulls, Learned asymmetric closed eBH 1: 152/1000.
- Persistent error, epsilon_i ~ N(0,0.1), 30 nonnulls, Learned asymmetric closed eBH 2: 1000/1000.

The normal Monte Carlo intervals in `summary.csv` quantify simulation error; they are not theoretical FDR guarantees.

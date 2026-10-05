#!/usr/bin/env python3
"""Compare stored results with Tables 1 and 2 in the supplied manuscript.

Values below are independently transcribed from PDF pages 22 and 23.
This checks published rounding, not the validity of the underlying simulations.
"""
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASYM = ROOT / "closed_eBH_asymmetric_weights_variance010_R1000"
CONST = ROOT / "constant_terms_simulation_submission_ready"

# Each row: condition, method, then TPR, TPR MCSE, FDR, FDR MCSE
# for n=20 followed by n=30.
TABLE1 = [
    ("clean", "testing_ebh", ".1584 .0039 .0026 .0012 .2096 .0038 .0029 .0007"),
    ("clean", "testing_mean", ".1790 .0042 .0036 .0012 .2506 .0040 .0040 .0007"),
    ("clean", "procedure1", ".4708 .0055 .0044 .0006 .5174 .0049 .0041 .0005"),
    ("clean", "procedure2", ".2478 .0051 .0036 .0012 .3500 .0046 .0043 .0006"),
    ("clean", "full_mean", ".5370 .0043 .0071 .0008 .6027 .0034 .0056 .0005"),
    ("signed_square", "procedure1", ".2742 .0060 .0020 .0006 .2747 .0059 .0016 .0004"),
    ("signed_square", "procedure2", ".2304 .0050 .0033 .0012 .3151 .0046 .0038 .0006"),
    ("signed_square", "naive_mean", ".7652 .0032 .0710 .0020 .7945 .0025 .0580 .0015"),
    ("persistent_noise", "procedure1", ".1768 .0044 .0045 .0010 .2038 .0041 .0040 .0007"),
    ("persistent_noise", "procedure2", ".2016 .0044 .0032 .0012 .2857 .0043 .0038 .0006"),
    ("persistent_noise", "naive_mean", ".5388 .0042 .0708 .0023 .5853 .0033 .0581 .0017"),
]
# Separate beta=0 and .004 checks despite the combined manuscript row.
TABLE2 = [
    (None, ".2227 .001041 .3509 .000767 .4275 .000600"),
    (Decimal("0"), ".2752 .001095 .4320 .000748 .5317 .000573"),
    (Decimal(".004"), ".2752 .001095 .4320 .000748 .5317 .000573"),
    (Decimal(".8"), ".2752 .001095 .4306 .000752 .5233 .000588"),
    (Decimal(".9"), ".2724 .001100 .4211 .000769 .5112 .000599"),
    (Decimal("1"), ".2623 .001106 .4090 .000783 .4980 .000609"),
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check(value, expected, label):
    target = Decimal(expected)
    observed = Decimal(value).quantize(target, rounding=ROUND_HALF_UP)
    if observed != target:
        raise RuntimeError(f"{label}: expected {target}, found {observed} (raw {value})")


def main():
    rows1 = read_csv(ASYM / "results" / "summary.csv")
    rows2 = read_csv(CONST / "results" / "constant_terms_results.csv")
    count1 = count2 = 0
    for condition, method, values in TABLE1:
        expected = values.split()
        for panel, nonnulls in enumerate((20, 30)):
            matches = [r for r in rows1 if r["condition"] == condition and
                       r["method"] == method and int(r["nonnulls"]) == nonnulls]
            if len(matches) != 1:
                raise RuntimeError(f"Missing/duplicate Table 1 row {condition}/{method}/{nonnulls}")
            for i, metric in enumerate(("TPR", "TPR_MCSE", "FDR", "FDR_MCSE")):
                check(matches[0][metric], expected[panel * 4 + i],
                      f"Table 1 {condition}/{method}/{nonnulls}/{metric}")
                count1 += 1
    for beta, values in TABLE2:
        expected = values.split()
        for panel, nonnulls in enumerate((500, 1000, 1500)):
            matches = [r for r in rows2 if int(r["nonnulls"]) == nonnulls and
                       ((beta is None and r["procedure"] == "eBH") or
                        (beta is not None and r["beta"] and Decimal(r["beta"]) == beta))]
            if len(matches) != 1:
                raise RuntimeError(f"Missing/duplicate Table 2 row beta={beta}/{nonnulls}")
            for i, metric in enumerate(("tpr", "tpr_se")):
                check(matches[0][metric], expected[panel * 2 + i],
                      f"Table 2 beta={beta}/{nonnulls}/{metric}")
                count2 += 1
    print(f"PASS: Table 1 {count1} numerical entries; Table 2 {count2} entries "
          "including both beta values in the combined row.")


if __name__ == "__main__":
    main()

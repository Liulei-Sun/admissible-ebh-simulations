"""Exact-rational checks of manuscript p24 footnote 3, using its two value groups."""
from fractions import Fraction as F
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'reproductions'/'independent_checks'
OUT.mkdir(parents=True, exist_ok=True)
K=5000
alpha=F(1,20)

def constant(n,beta):
    B=beta*K
    return (B-n)/(B-alpha*n) if n<B else F(0)

def worst_group_gap(high_count, high_value, low_value, r, beta):
    # For each intersection composition, maximize its overlap with the top-r
    # report; this exactly minimizes the closed-eBH constraint slack.
    selected_high=min(r,high_count)
    selected_low=max(0,r-high_count)
    best=None
    witness=None
    # Footnote groups include either one high or ten low entries, so iterate
    # over whichever group is smaller.
    for h in range(high_count+1):
        for l in range(K-high_count+1):
            n=h+l
            if n==0:
                continue
            overlap=min(h,selected_high)+min(l,selected_low)
            lam=constant(n,beta)
            merged=lam+(1-lam)*F(h*high_value+l*low_value,n)
            gap=merged-F(overlap,1)/(alpha*r)
            if best is None or gap<best:
                best,witness=gap,{'high_in_intersection':h,'low_in_intersection':l,'overlap':overlap}
    return {'r':r,'beta':str(beta),'minimum_slack':str(best),'certified':best>=0,'witness':witness}

rows=[]
for r,beta in [(20,F(0)),(21,F(0)),(1,F(1,250)),(2,F(1,250)),(20,F(1,250)),(21,F(1,250))]:
    rows.append({'vector':'(10000, 19 x4999)',**worst_group_gap(1,10000,19,r,beta)})
for r,beta in [(4990,F(0)),(5000,F(0)),(4990,F(1,250)),(5000,F(1,250))]:
    rows.append({'vector':'(21 x4990, 0 x10)',**worst_group_gap(4990,21,0,r,beta)})
(OUT/'footnote_check.json').write_text(json.dumps(rows,indent=2))
expected=[True,False,True,False,False,False,True,False,True,True]
assert [r['certified'] for r in rows] == expected
print(json.dumps(rows,indent=2))
print('PASS: exact rational footnote checks.')

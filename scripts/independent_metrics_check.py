"""Independent recomputation from original input truth and selected-index lists."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'closed_eBH_asymmetric_weights_variance010_R1000'
OUT=REPO/'reproductions'/'independent_checks'
OUT.mkdir(parents=True, exist_ok=True)
records=[]
contain=[]
input_rows=0
formula_max_errors={}

def indices(value):
    if pd.isna(value) or str(value).strip()=='':
        return np.array([],dtype=int)
    return np.array([int(x) for x in str(value).split(';')],dtype=int)

def scaled_error(a,b):
    return float(np.max(np.abs(a-b)/np.maximum(1,np.abs(b))))

for batch in sorted((ROOT/'data').iterdir()):
    for nonnulls in (20,30):
        for cond in ('clean','signed_square','persistent_noise'):
            panel=batch/f'n{nonnulls}'/cond
            inp=pd.read_csv(panel/'learned_instances.csv').sort_values(['replication','index'])
            input_rows+=len(inp)
            e=inp.e.to_numpy().reshape(-1,200)
            truths=inp.nonnull.to_numpy().reshape(-1,200).astype(bool)
            score=inp.score.to_numpy().reshape(-1,200)
            logscore=.45*(inp.train_sum if cond=='clean' else inp.contaminated_train_sum).to_numpy().reshape(-1,200)
            expected_score=np.exp(logscore-logscore.max(axis=1,keepdims=True))
            formula_max_errors[f'{batch.name}/{nonnulls}/{cond}/score']=scaled_error(score,expected_score)
            expected_e=np.exp(.45*inp.test_sum.to_numpy().reshape(-1,200)-40*.45**2/2)
            assert scaled_error(e,expected_e)<1e-12
            assert scaled_error(score,expected_score)<1e-12
            assert np.all(truths.sum(axis=1)==nonnulls)
            if cond=='persistent_noise':
                assert scaled_error(inp.contaminated_train_sum.to_numpy(),
                    inp.train_sum.to_numpy()+20*np.sqrt(.1)*inp.persistent_standard_normal.to_numpy())<1e-12
            paths={
              'procedure1':panel/'procedure1_results.csv',
              'procedure2':panel/'procedure2_results.csv',
              'testing_mean':panel.parent/'testing_mean_results.csv',
              ('full_mean' if cond=='clean' else 'naive_mean'):panel/('full_mean_results.csv' if cond=='clean' else 'naive_mean_results.csv')
            }
            all_selected={}
            for method,path in paths.items():
                result=pd.read_csv(path).sort_values('replication')
                assert len(result)==500
                selected_column='maximal_indices' if method.startswith('procedure') else 'selected_indices'
                all_selected[method]=[indices(x) for x in result[selected_column]]
            all_selected['testing_ebh']=[]
            for e_row in e:
                order=np.argsort(-e_row,kind='stable')
                good=np.flatnonzero(e_row[order]>=200/(.05*np.arange(1,201)))
                all_selected['testing_ebh'].append(order[:good[-1]+1] if len(good) else np.array([],dtype=int))
            for method,selections in all_selected.items():
                for rep,sel in enumerate(selections):
                    r=len(sel)
                    tp=int(truths[rep,sel].sum())
                    records.append({'condition':cond,'nonnulls':nonnulls,'method':method,
                        'TPR':tp/nonnulls,'FDP':(r-tp)/r if r else 0,'R':r,'TP':tp})
                    if method.startswith('procedure'):
                        contain.append({'condition':cond,'nonnulls':nonnulls,'method':method,
                            'contained':set(all_selected['testing_ebh'][rep]).issubset(sel)})

raw=pd.DataFrame(records)
expected=pd.read_csv(ROOT/'results/summary.csv').set_index(['condition','nonnulls','method'])
errors=[]
for key,rows in raw.groupby(['condition','nonnulls','method']):
    r=expected.loc[key]
    vals={'TPR':rows.TPR.mean(),'TPR_MCSE':rows.TPR.std(ddof=1)/np.sqrt(1000),
          'FDR':rows.FDP.mean(),'FDR_MCSE':rows.FDP.std(ddof=1)/np.sqrt(1000)}
    errors.extend(abs(vals[k]-r[k]) for k in vals)
    assert rows.R.sum()==r.total_rejections and rows.TP.sum()==r.total_true_discoveries
assert max(errors)<1e-14
contain_results=pd.DataFrame(contain).groupby(['condition','nonnulls','method']).contained.sum()
assert contain_results.loc['clean',20,'procedure1']==872
assert all(contain_results.xs('procedure2',level='method')==1000)
out={'input_rows':input_rows,'replication_metrics_recomputed':len(raw),'summary_groups':len(expected),
     'maximum_summary_abs_error':float(max(errors)),
     'maximum_score_scaled_error':max(formula_max_errors.values()),
     'clean20_procedure1_contains_ebh':int(contain_results.loc['clean',20,'procedure1']),
     'all_procedure2_containment_counts':list(map(int,contain_results.xs('procedure2',level='method')))}
(OUT/'metrics_check.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

"""Summarize available matched-seed training only; never selects policies."""
from week_common import *
def render_training():
 rows=[];contrasts={}
 for seed in [101,102,103]:
  for arm in ['original','random','entropy','geometry']:
   p=ART/'train'/f'{arm}_s{seed}'
   if (p/'complete.json').exists() and (p/'test_latest.json').exists():
    v=read(p/'test_latest.json');secondary=read(p/'test_best.json') if (p/'test_best.json').exists() else None;rows.append(dict(seed=seed,arm=arm,primary_checkpoint='latest125',validation_best_secondary=secondary,**v))
 # Primary: fixed latest125 model, macro OOD regret, geometry minus comparator.
 # Three predeclared contrasts use Bonferroni98.333% paired intervals.
 for base in ['original','random','entropy']:
  seeds=[seed for seed in [101,102,103] if all((ART/'train'/f'{arm}_s{seed}'/'complete.json').exists() for arm in ['geometry',base])]
  if not seeds:continue
  differences={}
  for group in ['mix_gauss','edge_bias','grid_cluster']:
   values=[]
   for seed in seeds:
    left=ART/'train'/f'geometry_s{seed}'/'test_latest'/f'{group}.npz'
    right=ART/'train'/f'{base}_s{seed}'/'test_latest'/f'{group}.npz'
    with np.load(left) as a,np.load(right) as b:
     if not np.array_equal(a['net_ids'],b['net_ids']):raise RuntimeError('paired report IDs mismatch')
     values.append(a['regret']-b['regret'])
   differences[group]=np.stack(values).mean(0)
  rng=np.random.default_rng(202610071);boot=[]
  for _ in range(1000):
   boot.append(float(np.mean([v[rng.integers(0,len(v),len(v))].mean() for v in differences.values()])))
  contrasts[base]=dict(seeds=seeds,mean_ood_delta=float(np.mean([v.mean() for v in differences.values()])),
                      bonferroni_ci=np.quantile(boot,[.05/(2*3),1-.05/(2*3)]).tolist(),
                      interval_level=1-.05/3,inference_scope='test-instance uncertainty conditional on listed fitted seeds; not a population CI over training seeds')
 atomic(ART/'training_summary.json',dict(completed_runs=len(rows),expected_runs=12,rows=rows,primary_contrasts=contrasts,
                                       incomplete_runs=[f'{arm}_s{seed}' for seed in [101,102,103] for arm in ['original','random','entropy','geometry'] if not (ART/'train'/f'{arm}_s{seed}'/'complete.json').exists()]))
 return rows,contrasts
if __name__=='__main__':render_training()

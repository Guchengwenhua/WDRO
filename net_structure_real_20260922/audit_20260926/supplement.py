from common import *
from collections import defaultdict
from structure import features
import torch

def group(rows,keys):
 out=defaultdict(list)
 for r in rows:out[tuple(r[k] for k in keys)].append(r)
 return out.items()
def csvin(p):
 with open(p) as f:return list(csv.DictReader(f))
def avg(rows,k):return float(np.mean([float(r[k]) for r in rows]))
def main():
 pressure=sum([csvin(OUT/f'fixed_parent_s{s}.csv') for s in [101,102,103]],[]);summary=[]
 for (pool,lam),rows in group(pressure,['pool','lambda_value']):summary.append(dict(pool=pool,lambda_value=float(lam),centers=int(rows[0]['centers']),seeds=len(rows),nominal_pct=avg(rows,'nominal_regret')*100,max_regret_pct=avg(rows,'max_regret')*100,selected_regret_pct=avg(rows,'selected_regret')*100,selected_cost=avg(rows,'selected_cost'),nominal_fraction=avg(rows,'nominal_fraction')))
 csvout('fixed_parent_summary.csv',summary)
 dynamics=[]
 for folder in sorted((ART/'runs').iterdir()):
  cp=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False);h=cp['history'];dynamics.append(dict(method=cp['method'],generator=cp['generator'],seed=cp['seed'],lambda_zero_fraction=float(np.mean([r['lambda_before']==0 for r in h])),positive_budget_gap_fraction=float(np.mean([r['budget_gap']>0 for r in h])),max_lambda=max(r['lambda_after'] for r in h),mean_cost=np.mean([r['selected_cost'] for r in h]),mean_budget_gap=np.mean([r['budget_gap'] for r in h])))
 csvout('dual_dynamics.csv',dynamics)
 dsum=[]
 for (m,g),rows in group(dynamics,['method','generator']):dsum.append(dict(method=m,generator=g,lambda_zero_fraction=avg(rows,'lambda_zero_fraction'),mean_budget_gap=avg(rows,'mean_budget_gap'),mean_cost=avg(rows,'mean_cost')))
 csvout('dual_summary.csv',dsum)
 data=npz(ART/'data.npz');gate=read(ART/'gate.json');rows=[]
 def normalized(x):
  a=x-x.min(0);return np.asarray(.1+.8*a/a.max(),np.float32)
 def score(x):return float(np.max(abs((features(x)-gate['center'])/gate['scale'])))
 for i in np.flatnonzero(data['split']=='B_test')[::20]:
  x=data['pins'][i].copy();collapsed=x.mean(0)+.05*(x-x.mean(0));single=x.copy();single[0]=[.9,.9];axis=x.copy();axis[:,1]=x[:,1].mean()+.005*(x[:,1]-x[:,1].mean())
  for name,z in [('real',x),('global_scale_collapse_then_normalize',normalized(collapsed)),('single_outlier_then_normalize',normalized(single)),('axis_collapse_then_normalize',normalized(axis))]:
   sc=score(z);rows.append(dict(variant=name,net_id=data['net_ids'][i],score=sc,rejected=sc>gate['threshold'],max_change_from_original=float(abs(z-x).max())))
 csvout('normalized_stress.csv',rows);csvout('normalized_stress_summary.csv',[dict(variant=name[0],nets=len(g),rejected_fraction=float(np.mean([r['rejected'] for r in g])),max_coordinate_change=max(r['max_change_from_original'] for r in g)) for name,g in group(rows,['variant'])])
 # Pin down comparison conditions directly from the immutable observed pools.
 raw=npz(ART/'pools/raw_wdro.npz')['pins'];fg=npz(ART/'pools/feature_gate.npz')['pins'];changed=np.any(raw[:,1:]!=fg[:,1:],axis=(2,3))
 write(OUT/'mechanism_limits.json',dict(time=now(),feature_gate_changed_non_nominal_fraction=float(changed.mean()),feature_gate_changed_centers_fraction=float(changed.any(1).mean()),graph_loss_zero_weight_control_present=False,raw_vs_learned_common_proposals=False,tile_raw_vs_graph_tile_common_noise=True,all_ispd_pin_layers=[1],fixed_parent_evaluation_scope='acceptance supplement on predeclared seed 20260926, 32 A_train centers per design, all 3 frozen parent seeds, K8 and lambda 0/1; no retraining or test-driven selection',normalization_collapse_caveat='Global uniform collapse is removed by the input normalization; original collapse stress was outside the normalization contract'))
 print('SUPPLEMENT COMPLETE',flush=True)
if __name__=='__main__':main()

"""Acceptance statistics from saved lengths/references; no training or GPU replay."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,csv,collections
from pathlib import Path
import numpy as np
from scipy.stats import t,spearmanr
O=Path(__file__).resolve().parent;P=O.parent;R=P.parent
GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster']
def read(p):return json.loads(p.read_text())
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def table(n,rows):
 with (O/n).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sources=read(P/'sources.json');state=read(P/'artifacts/state.json');oldstate=read(R/'accelerated_research_20261006/artifacts/state.json');jobs=read(P/'jobs.json')['jobs']
walls={j.get('result_dir'):state['jobs'][j['id']]['wall_seconds_total'] for j in jobs if j['kind']=='train'}
def wall(folder):
 path=Path(folder)
 return walls[str(path)] if str(path) in walls else oldstate['jobs'][path.name]['wall_seconds_total']
arr={};runs=[];budgets=[];count=0;diff=0.;summarydiff=0.
for source in sources:
 p=Path(source['result_dir']);c=read(p/'complete.json');v=c['result'];cfg=read(p/'run_config.json')
 if v['completed_updates']!=source['updates'] or v['batch_size']!=4096:raise ValueError('actual budget mismatch')
 budgets.append(dict(job=p.name,family=source['family'],arm=source['arm'],seed=source['seed'],updates=v['completed_updates'],batch=v['batch_size'],samples=v['policy_update_nets'],final_lambda=v['method_state']['lambda'],actor_forward_nets=v['call_ledger']['actor_forward_nets'],geosteiner_nets=v['call_ledger']['geosteiner_nets'],exact_shortlist=v['method_state']['budget']['exact_rescore_nets'],nan_detected=v['nan_detected']))
 seconds=wall(p)
 if source['updates']==625:seconds+=wall(source['parent'])
 for outcome in ['latest','best']:
  values=[];ids=[];refs=[];saved=read(p/f'test_{outcome}.json')
  for g in GROUPS:
   with np.load(p/f'test_{outcome}'/f'{g}.npz') as z:
    e=z['length']/z['reference']-1;count+=len(e);diff=max(diff,float(np.max(abs(e-z['regret']))))
    if not np.isfinite(e).all() or e.min() < -1e-5:raise ValueError('invalid relative error')
    values.append(e);ids.append(z['net_ids']);refs.append(z['reference']);summarydiff=max(summarydiff,abs(float(e.mean())-saved['groups'][g]['mean']))
  key=(source['family'],source['updates'],source['arm'],source['seed'],outcome);arr[key]=(np.array(values),ids,refs)
  runs.append(dict(family=source['family'],degree=source['degree'],updates=source['updates'],arm=source['arm'],seed=source['seed'],outcome=outcome,mean_pct=float(np.mean(values)*100),ood_pct=float(np.mean(values[1:])*100),worst_ood_pct=max(float(v.mean()*100) for v in values[1:]),macro_p95_pct=float(np.mean([np.quantile(v,.95) for v in values])*100),train_hours=seconds/3600,checkpoint_update=saved['checkpoint_update']))
table('runs.csv',runs);table('budgets.csv',budgets)
aggs=[]
for key in sorted(set((r['family'],r['updates'],r['arm'],r['outcome']) for r in runs)):
 rr=[r for r in runs if (r['family'],r['updates'],r['arm'],r['outcome'])==key]
 aggs.append(dict(family=key[0],updates=key[1],arm=key[2],outcome=key[3],seeds=len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['mean_pct','ood_pct','worst_ood_pct','macro_p95_pct','train_hours']},sd_mean_pct=float(np.std([r['mean_pct'] for r in rr],ddof=1)),sd_ood_pct=float(np.std([r['ood_pct'] for r in rr],ddof=1))))
table('aggregate.csv',aggs);print('aggregate',json.dumps(aggs),flush=True)
contrasts=[]
for fam,u,arms,base in [('geometry',125,['original','random','entropy'],'geometry'),('bank',125,['k3','lagged3'],'geo'),('bank',625,['k3','lagged3'],'geo')]:
 for outcome in ['latest','best']:
  for arm in arms:
   ds=[]
   for seed in [101,102,103]:
    left,right=(base,arm) if fam=='geometry' else (arm,base)
    a,ai,ar=arr[fam,u,left,seed,outcome];b,bi,br=arr[fam,u,right,seed,outcome]
    if not all(np.array_equal(x,y) for x,y in zip(ai,bi)) or not all(np.array_equal(x,y) for x,y in zip(ar,br)):raise ValueError('pair IDs/reference mismatch')
    ds.append((a-b)[1:] if fam=='geometry' else a-b)
   effects=np.array([d.mean()*100 for d in ds]);d=np.mean(ds,axis=0)*100
   B=20000 if fam=='geometry' and outcome=='latest' else 5000;alpha=.05/3 if fam=='geometry' and outcome=='latest' else .05
   rng=np.random.default_rng(20261009);boot=np.zeros(B)
   for g in d:
    for start in range(0,B,50):
     n=min(50,B-start);boot[start:start+n]+=g[rng.integers(len(g),size=(n,len(g)))].mean(1)/len(d)
   contrasts.append(dict(family=fam,updates=u,outcome=outcome,comparison=left+' minus '+right,mean_delta_pp=float(effects.mean()),conditional_interval_pp=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),interval_level=1-alpha,replicates=B,seed_effects_pp=effects.tolist(),seed_t95_pp=(effects.mean()+t.ppf([.025,.975],2)*effects.std(ddof=1)/np.sqrt(3)).tolist()))
save('contrasts.json',contrasts);print('contrasts',json.dumps(contrasts),flush=True)
save('receipt.json',dict(status='completed_with_bounded_scientific_conclusions',queue_counts=dict(collections.Counter(r['status'] for r in state['jobs'].values())),sources=len(sources),saved_error_values=count,max_reconstruction_difference=diff,max_group_mean_difference=summarydiff,new_queue_elapsed_hours=(state['updated']-state['created'])/3600,new_gpu_process_hours=sum(r.get('wall_seconds_total',0) for r in state['jobs'].values() if r.get('gpu',-1)>=0)/3600,method='saved length/reference recomputation, no new inference or oracle execution',monitor=read(P/'artifacts/geometry_monitor/heartbeat.json')))
# Only final completed geometry checkpoints for the new monitoring summary.
mr=[];snaps=0;seconds=0.
for folder in (P/'artifacts/geometry_monitor').iterdir():
 if not folder.is_dir():continue
 ff=sorted(folder.glob('u*.json'));snaps+=len(ff)
 for f in ff:seconds+=read(f)['seconds']
 if not ff or 'geometry_training' not in folder.name:continue
 last=read(ff[-1])
 with np.load(ff[-1].with_suffix('.npz')) as z:sc=z['scores'];e=z['length']/z['reference']-1;families=z['family'];names=z['score_names'].tolist()
 for g in GROUPS:
  ix=families==g;gr=float(spearmanr(sc[ix,names.index('G')],e[ix]).statistic);hr=float(spearmanr(sc[ix,names.index('Hnorm')],e[ix]).statistic)
  mr.append(dict(job=last['job'],update=last['update'],group=g,G=gr,H=hr))
table('geometry_monitor_final.csv',mr)
save('monitor_summary.json',dict(snapshots=snaps,evaluation_hours=seconds/3600,geometry_final_cells=len(mr),G_range=[min(r['G'] for r in mr),max(r['G'] for r in mr)],G_above_H=sum(r['G']>r['H'] for r in mr)))
print('done',flush=True)

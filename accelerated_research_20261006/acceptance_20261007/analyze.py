import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,csv,collections,math
from pathlib import Path
import numpy as np
from scipy.stats import t,rankdata
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent.parent;P=OUT.parent
GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster']
def read(p):return json.loads(p.read_text())
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def csvout(n,rows):
 with (OUT/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(x):
 x=np.array(x);m=x.mean();n=len(x)
 return (m+t.ppf([.025,.975],n-1)*x.std(ddof=1)/np.sqrt(n)).tolist() if n>1 else None
def boot_groups(groups,seed=71026,B=2000):
 rng=np.random.default_rng(seed);b=np.zeros(B)
 for a in groups:
  for start in range(0,B,50):
   ix=rng.integers(len(a),size=(min(50,B-start),len(a)));b[start:start+len(ix)]+=a[ix].mean(1)/len(groups)
 return np.quantile(b,[.025,.975]).tolist()
def rho(a,b):return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
state=read(P/'artifacts/state.json');jobs=read(P/'jobs.json')['jobs'];inventory=[]
for j in jobs:
 s=state['jobs'][j['id']];inventory.append(dict(job=j['id'],kind=j['kind'],status=s['status'],attempts=s.get('attempts',0),marker_exists=Path(j['complete']).exists(),gpu=s.get('gpu'),seconds=s.get('wall_seconds_total',0),reason=s.get('reason','')))
csvout('inventory.csv',inventory)
save('lifecycle.json',dict(counts=dict(collections.Counter(r['status'] for r in inventory)),elapsed_hours=(state['updated']-state['created'])/3600,created=state['created'],ended=state['updated'],gpu_process_hours=sum(r['seconds'] for r in inventory if r['gpu'] is not None and r['gpu']>=0)/3600,cpu_process_hours=sum(r['seconds'] for r in inventory if r['gpu']==-1)/3600))
rows=[];arrays={};raw_discrepancy=0.;summary_discrepancy=0.;nvalues=0;logrows=[]
for folder in sorted((ROOT/'accelerated_bank_20261006/artifacts').iterdir()):
 if not folder.is_dir():continue
 event=folder/'logs/events.jsonl';train=[]
 if event.exists():
  for line in event.open():
   e=json.loads(line)
   if e.get('event')=='train':train.append(e)
 if train:
  logrows.append(dict(job=folder.name,logged_updates=len(train),unique_updates=len(set(e['update'] for e in train)),first_update=min(e['update'] for e in train),last_update=max(e['update'] for e in train),lambda_zero=sum(e['method_detail']['lambda_after']==0 for e in train),lambda_min=min(e['method_detail']['lambda_after'] for e in train),lambda_max=max(e['method_detail']['lambda_after'] for e in train),negative_target_fraction_mean=float(np.mean([e['negative_target_fraction'] for e in train]))))
 if not (folder/'complete.json').exists():continue
 complete=read(folder/'complete.json');cfg=read(folder/'run_config.json');dev=read(folder/'dev.json')
 for outcome in ['latest','best']:
  gs=[];ids=[];refs=[];groupmeans=[];p95=[]
  for g in GROUPS:
   with np.load(folder/f'test_{outcome}'/f'{g}.npz') as z:
    error=z['length']/z['reference']-1
    raw_discrepancy=max(raw_discrepancy,float(np.max(abs(error-z['regret']))));nvalues+=len(error)
    gs.append(error);ids.append(z['net_ids']);refs.append(z['reference']);groupmeans.append(float(error.mean()));p95.append(float(np.quantile(error,.95)))
  saved=read(folder/f'test_{outcome}.json');summary_discrepancy=max(summary_discrepancy,abs(np.mean(groupmeans)-saved['overall_mean']))
  arrays[folder.name,outcome]=(gs,ids,refs)
  rows.append(dict(job=folder.name,degree=complete['degree'],arm=complete['arm'],seed=complete['seed'],updates=complete['updates'],outcome=outcome,batch=complete['result']['batch_size'],samples=complete['result']['policy_update_nets'],checkpoint_update=saved['checkpoint_update'],mean_pct=float(np.mean(groupmeans)*100),worst_ood_pct=max(groupmeans[1:])*100,macro_p95_pct=float(np.mean(p95)*100),n=40000))
csvout('bank_runs.csv',rows);csvout('training_diagnostics.csv',logrows)
aggregate=[];contrasts=[]
for degree,updates in [(10,125),(10,625),(30,125)]:
 for outcome in ['latest','best']:
  for arm in ['geo','k3','lagged3','qualifiedbest3']:
   subset=[r for r in rows if (r['degree'],r['updates'],r['outcome'],r['arm'])==(degree,updates,outcome,arm)]
   if not subset:continue
   walls=[]
   for r in subset:
    wall=state['jobs'][r['job']]['wall_seconds_total']
    if updates==625:wall+=state['jobs'][r['job'].replace('625_','125_')]['wall_seconds_total']
    walls.append(wall)
   aggregate.append(dict(degree=degree,updates=updates,outcome=outcome,arm=arm,n_seeds=len(subset),mean_pct=float(np.mean([r['mean_pct'] for r in subset])),sd_pct=float(np.std([r['mean_pct'] for r in subset],ddof=1)),worst_ood_pct=float(np.mean([r['worst_ood_pct'] for r in subset])),macro_p95_pct=float(np.mean([r['macro_p95_pct'] for r in subset])),mean_train_seconds=float(np.mean(walls))))
   if arm=='geo':continue
   diffs=[];effects=[]
   for r in subset:
    geo=next(x for x in rows if (x['degree'],x['updates'],x['outcome'],x['arm'],x['seed'])==(degree,updates,outcome,'geo',r['seed']))
    a,ai,ar=arrays[r['job'],outcome];b,bi,br=arrays[geo['job'],outcome]
    if not all(np.array_equal(x,y) for x,y in zip(ai,bi)) or not all(np.array_equal(x,y) for x,y in zip(ar,br)):raise ValueError('pair mismatch')
    d=np.stack(a)-np.stack(b);diffs.append(d);effects.append(float(d.mean()*100))
   pooled=np.mean(diffs,axis=0)*100
   contrasts.append(dict(degree=degree,updates=updates,outcome=outcome,arm=arm,n_seeds=len(effects),seed_effects_pp=effects,delta_pp=float(np.mean(effects)),ci95_conditional_instances_pp=boot_groups(pooled),ci95_seed_t_pp=ci(effects)))
csvout('bank_aggregate.csv',aggregate);save('bank_contrasts.json',contrasts)
save('numeric_reconciliation.json',dict(bank_values=nvalues,max_error_reconstruction_difference=raw_discrepancy,max_mean_summary_difference=summary_discrepancy))
print('BANK',json.dumps([r for r in aggregate if r['outcome']=='latest']),flush=True)
# Independently reconstruct finite-candidate objectives from saved lengths, references and selected shortlists.
geometry=[];geom_mismatch=0.
for folder in sorted((ROOT/'accelerated_geometry_20261006/artifacts').glob('screen_d*')):
 summary=read(folder/'summary.json')
 with np.load(folder/'inputs.npz') as z:cost=z['cost'];family=z['family']
 ref=np.stack([np.load(folder/f'reference_{i:04d}.npz')['reference'] for i in range(len(cost))])
 all_delta=collections.defaultdict(list)
 for model in sorted(p for p in folder.iterdir() if p.is_dir() and (p/'predictions.npz').exists()):
  with np.load(model/'predictions.npz') as z:regret=z['length']/ref-1
  for scope,methods in read(model/'selection.json')['selected'].items():
   selected={}
   for method,index in methods.items():
    idx=np.array(index);obj=regret[np.arange(len(cost))[:,None],idx]
    if scope=='soft_dual':obj=obj-cost[np.arange(len(cost))[:,None],idx]
    selected[method]=obj.max(1)
   with np.load(model/f'{scope}_center_outcomes.npz') as z:
    for method,v in selected.items():geom_mismatch=max(geom_mismatch,float(np.max(abs(v-z[method]))))
   for base in ['entropy','random','original']:all_delta[scope,base].append(selected['geometry']-selected[base])
 for (scope,base),d in all_delta.items():
  a=np.mean(d,axis=0);record=summary['statistics'][scope+'/'+base]
  geometry.append(dict(degree=summary['degree'],split=summary['split'],scope=scope,baseline=base,n_centers=len(a),models=len(d),delta_pp=float(a.mean()*100),ci95_stratified_pp=boot_groups([a[family==g]*100 for g in GROUPS]),original_ci95_pp=[100*x for x in record['ci95']],original_holm_p=record['holm_p_six_contrasts']))
csvout('geometry_screen.csv',geometry);save('geometry_reconciliation.json',dict(max_center_objective_difference=geom_mismatch))
print('GEOMETRY',json.dumps([x for x in geometry if x['scope']=='soft_dual']),flush=True)
# Recompute final diagnostic correlations, pairing G and H on the same instances.
monitor=P/'artifacts/geometry_monitor';monitorrows=[];coverage=[]
for folder in sorted(monitor.iterdir()):
 if not folder.is_dir():continue
 files=sorted(folder.glob('u*.json'))
 if not files:continue
 first=read(files[0]);last=read(files[-1]);coverage.append(dict(job=last['job'],n_snapshots=len(files),first_update=first['update'],last_update=last['update'],cpu_evaluation_seconds=sum(read(f)['seconds'] for f in files),completed_training=(ROOT/'accelerated_bank_20261006/artifacts'/last['job']/'complete.json').exists()))
 with np.load(files[-1].with_suffix('.npz')) as z:s=z['scores'];err=z['length']/z['reference']-1;fam=z['family'];names=z['score_names'].tolist()
 for gi,g in enumerate(GROUPS):
  ix=fam==g;e=err[ix];G=s[ix,names.index('G')];H=s[ix,names.index('Hnorm')];rng=np.random.default_rng(20261007+gi);ds=[]
  for _ in range(1000):
   ii=rng.integers(len(e),size=len(e));ds.append(rho(G[ii],e[ii])-rho(H[ii],e[ii]))
  monitorrows.append(dict(job=last['job'],degree=last['degree'],seed=last['training_seed'],update=last['update'],group=g,G_rho=rho(G,e),H_rho=rho(H,e),delta_rho=rho(G,e)-rho(H,e),delta_ci95=np.quantile(ds,[.025,.975]).tolist(),G_enrichment=last['groups'][g]['G']['top10_enrichment'],H_enrichment=last['groups'][g]['Hnorm']['top10_enrichment']))
csvout('monitor_final.csv',monitorrows);csvout('monitor_coverage.csv',coverage)
print('MONITOR',len(coverage),sum(r['n_snapshots'] for r in coverage),sum(r['cpu_evaluation_seconds'] for r in coverage),flush=True)

from common import *
import itertools,time
from collections import defaultdict
def group(rows,keys):
 out=defaultdict(list)
 for r in rows:out[tuple(r[k] for k in keys)].append(r)
 return out.items()
def avg(rows,k):return float(np.mean([float(r[k]) for r in rows]))
def csvin(p):
 with open(p) as f:return list(csv.DictReader(f))

def main():
 data=npz(ART/'data.npz');test=np.isin(data['split'],['B_test','C_external']);meta={k:data[k][test] for k in ['net_ids','design_ids','family_ids','dataset_ids','split','lstar']};rows=[];perdesign=[];errors=[];cache={};maxerr=0.
 for path in sorted((ART/'test').glob('*.npz')):
  f=npz(path)
  for k,v in meta.items():
   if not np.array_equal(v,f[k]):errors.append(f'{path.name}:metadata:{k}')
  r=f['length']/meta['lstar']-1;err=float(np.max(abs(r-f['regret'])));maxerr=max(maxerr,err)
  if err>1e-12 or not np.isfinite(r).all() or r.min() < -1e-6:errors.append(path.name+':regret')
  keys=['method','seed','generator','budget','role','update'];scalars={k:f[k][0].item() for k in keys}
  if not all(np.all(f[k]==f[k][0]) for k in keys):errors.append(path.name+':identity')
  key=tuple(scalars[k] for k in keys[:-1]);cache[key]=r
  for split in ['B_test','C_external']:
   take=meta['split']==split;ds=meta['design_ids'];m=[float(r[take&(ds==d)].mean()) for d in np.unique(ds[take])]
   rows.append(dict(**scalars,split=split,nets=int(take.sum()),mean=float(r[take].mean()),p95=float(np.quantile(r[take],.95)),design_macro=float(np.mean(m)),worst_design=float(np.max(m))))
   for d in np.unique(ds[take]):
    v=r[take&(ds==d)];perdesign.append(dict(**scalars,split=split,design=d,nets=len(v),mean=float(v.mean()),p95=float(np.quantile(v,.95))))
 csvout('recomputed_main.csv',rows);csvout('recomputed_per_design.csv',perdesign)
 idx=['method','seed','generator','budget','role','split']
 def compare(new,old,keys,values):
  aa={tuple(str(r[k]) for k in keys):r for r in new};bb={tuple(str(r[k]) for k in keys):r for r in old};assert aa.keys()==bb.keys()
  return {k:max(abs(float(aa[i][k])-float(bb[i][k])) for i in aa) for k in values}
 differences=compare(rows,csvin(ROOT/'tables/main.csv'),idx,['mean','p95','design_macro','worst_design','update','nets'])
 if max(differences.values())>1e-12:errors.append('published main.csv mismatch')
 designdiff=compare(perdesign,csvin(ROOT/'tables/per_design.csv'),idx+['design'],['mean','p95','nets'])
 if max(designdiff.values())>1e-12:errors.append('published per_design.csv mismatch')
 aggregate=[]
 for key,g in group([r for r in rows if r['generator']==301 or r['method']=='parent'],['method','budget','role','split']):
  aggregate.append(dict(zip(['method','budget','role','split'],key),seeds=len(g),mean_pct=avg(g,'mean')*100,p95_pct=avg(g,'p95')*100,design_macro_pct=avg(g,'design_macro')*100,worst_design_pct=avg(g,'worst_design')*100,seed_sd_mean_pp=float(np.std([r['mean'] for r in g],ddof=1)*100)))
 csvout('aggregate.csv',aggregate)
 contrasts=[];pairs=[(m,'raw_wdro') for m in CFG['methods'] if m!='raw_wdro']+[('graph_dae','set_dae'),('graph_tile','tile_raw'),('graph_dae','vae_shell_wdro'),('set_dae','vae_shell_wdro'),('graph_tile','vae_shell_wdro'),('raw_wdro','erm_a')]
 for m,base in pairs:
  for budget in [256,1024,2048]:
   for role in ['last','val_best']:
    for split in ['B_test','C_external']:
     take=meta['split']==split;ds=meta['design_ids'][take];names=np.unique(ds);delta=np.stack([cache[m,s,301,budget,role][take]-cache[base,s,301,budget,role][take] for s in CFG['seeds']]);dm=np.array([[d[ds==design].mean() for design in names] for d in delta]);counts=np.array([(ds==d).sum() for d in names]);rng=np.random.default_rng(260926);si=rng.integers(3,size=(2000,3));di=rng.integers(len(names),size=(2000,len(names)))
     boot=dm[si[:,:,None],di[:,None,:]].mean((1,2));ci=np.quantile(boot,[.025,.975]);seedmean=delta.mean(1)
     contrasts.append(dict(method=m,baseline=base,budget=budget,role=role,split=split,net_mean_delta_pp=float(delta.mean()*100),design_macro_delta_pp=float(dm.mean()*100),joint_seed_design_ci_low_pp=float(ci[0]*100),joint_seed_design_ci_high_pp=float(ci[1]*100),seeds_improved=int((seedmean<0).sum()),designs_improved=int((dm.mean(0)<0).sum()),designs=len(names),ci_scope='exploratory paired resampling of 3 seeds and design clusters; no multiple-comparison correction'))
 csvout('paired_contrasts.csv',contrasts)
 attackrows=[]
 for path in sorted((ART/'attack').glob('*.npz')):
  f=npz(path);done=read(ART/'done'/f'{path.stem}.json');assert np.isfinite(f['worst_regret']).all();assert (f['worst_regret']>=f['nominal_regret']-1e-12).all()
  assert abs(f['worst_regret'].mean()-done['mean'])<1e-12
  for d in ['ALL',*np.unique(f['design_ids'])]:
   take=np.ones(len(f['worst_regret']),bool) if d=='ALL' else f['design_ids']==d
   attackrows.append(dict(method=done['method'],seed=done['seed'],generator=done['generator'],design=d,nets=int(take.sum()),mean=float(f['worst_regret'][take].mean()),p95=float(np.quantile(f['worst_regret'][take],.95)),nominal_mean=float(f['nominal_regret'][take].mean()),selected_cost_mean=float(f['selected_cost'][take].mean())))
 csvout('attack_per_design.csv',attackrows)
 attackagg=[]
 for (m,g),part in group([r for r in attackrows if r['design']=='ALL'],['method','generator']):attackagg.append(dict(method=m,generator=g,seeds=len(part),mean_pct=avg(part,'mean')*100,p95_pct=avg(part,'p95')*100,nominal_pct=avg(part,'nominal_mean')*100))
 csvout('attack_aggregate.csv',attackagg)
 gen=csvin(ROOT/'tables/generator_test.csv');genagg=[]
 for (kind,split),part in group(gen,['kind','split']):genagg.append(dict(kind=kind,split=split,design_sigma_seed_macro_mse=avg(part,'normalized_mse'),identity_mse=avg(part,'identity_mse'),relative_mse_reduction_pct=(1-avg(part,'normalized_mse')/avg(part,'identity_mse'))*100))
 csvout('generator_test_summary.csv',genagg)
 stress=csvin(ROOT/'tables/geometry_stress.csv');csvout('original_stress_summary.csv',[dict(variant=v[0],nets=len(p),rejected_fraction=np.mean([r['rejected']=='True' for r in p])) for v,p in group(stress,['variant'])])
 crossed=[]
 for (g,split,role),p in group([r for r in rows if r['method']=='graph_tile' and r['budget']==2048],['generator','split','role']):crossed.append(dict(generator=g,split=split,role=role,rest_seeds=len(p),mean_pct=avg(p,'mean')*100,p95_pct=avg(p,'p95')*100))
 csvout('crossed_generators.csv',crossed)
 write(OUT/'metrics.json',dict(status='PASS' if not errors else 'FAIL',errors=errors,natural_test_files=len(cache),prediction_rows=len(cache)*len(meta['net_ids']),metric_rows=len(rows),design_rows=len(perdesign),max_regret_formula_error=maxerr,main_csv_difference=differences,per_design_csv_difference=designdiff,attack_files=len(attackrows)//4,time=now()))
 print('METRICS',len(cache),'files',len(errors),'errors',flush=True)
 print(json.dumps([r for r in aggregate if r['budget']==2048 and r['role']=='val_best'],indent=2),flush=True)
 print('ATTACK',json.dumps([r for r in attackagg if r['generator']==301],indent=2),flush=True)
if __name__=='__main__':main()

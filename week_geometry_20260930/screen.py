"""New synthetic instances and post-normalization Hungarian-cost screening."""
import argparse,sys
from week_common import *
def main(args):
 runtime();import torch
 sys.path.insert(0,str(BASE/'wdro_rest_min'))
 from src.datasets import generate_terminals,normalize_fixed_degree_batch
 from src.costs import hungarian_l1_mean
 from src.geosteiner import GeoSteinerEvaluator
 from src.candidates import fixed_surrogate
 sys.path.insert(0,str(PROJECT/'geometry_uncertainty_20260930'))
 from research import Runner,SCORES
 f=metadata();degree=args.degree;split=args.split;metric=f['choices'][str(degree)]['metric']
 out=ART/f'screen_d{degree}_{split}';out.mkdir(exist_ok=True)
 if (out/'complete.json').exists():return
 base_seed=f[split+'_seed']+degree*100;count=32 if split=='dev' else 64
 protocol=read(BASE/'wdro_rest_min/configs/protocol_v1.json')
 rho=read(BASE/f'experiments/cache/stage6/calibration_d{degree}_s101.json')['hungarian_l1_mean']['q75']
 if not (out/'inputs.npz').exists():
  allpins=[];allids=[];families=[];allcost=[]
  for gi,group in enumerate(GROUPS):
   centers=[]
   for attempt in range(100):
    raw=generate_terminals(count,degree,base_seed+gi*1000+attempt,group,protocol['data']['distributions'][group])
    normalized,_,_,valid,_=normalize_fixed_degree_batch(raw)
    centers.extend(normalized[valid].astype(np.float32))
    if len(centers)>=count:break
   if len(centers)<count:raise RuntimeError('not enough legal centers')
   for ci,center in enumerate(centers[:count]):
    rng=np.random.default_rng(base_seed+gi*1000+ci+900000)
    candidates=[center]
    for j in range(32):
     amplitude=[.01,.02,.04,.08][(j//4)%4]
     for attempt in range(100):
      motion=rng.uniform(-1,1,(degree,2))
      if j%4==1:motion*= (np.abs(motion[:,0])>=np.quantile(np.abs(motion[:,0]),.75))[:,None]
      elif j%4==2:
       anchor=center[rng.integers(0,degree)];envelope=np.exp(-((center-anchor)**2).sum(-1)/(2*.2**2));motion=envelope[:,None]*motion[0]
      elif j%4==3:motion[:,j%2]*=.1
      z,_,_,valid,_=normalize_fixed_degree_batch(np.clip(center+amplitude*motion,.1,.9)[None])
      if valid[0]:
       candidate=z[0].astype(np.float32)
       d=np.abs(candidate[:,None].astype(float)-candidate[None].astype(float)).sum(-1);np.fill_diagonal(d,np.inf)
       if d.min()>=1e-6:break
     else:raise RuntimeError('candidate generation exhausted')
     candidates.append(candidate)
    pins=np.stack(candidates)
    # Costs computed on actual float32 final normalized model inputs, never raw motion.
    costs=np.array([hungarian_l1_mean(center,z) for z in pins])
    allpins.append(pins);allcost.append(costs);families.append(group)
    allids.append(f'week:{split}:{group}:d{degree}:seed{base_seed}:center{ci}')
  npz(out/'inputs.npz',pins=np.array(allpins),cost=np.array(allcost),net_ids=np.array(allids),family=np.array(families),rho=np.array(rho))
 with np.load(out/'inputs.npz',allow_pickle=False) as z:x=z['pins'];cost=z['cost'];family=z['family']
 flat=x.reshape(-1,degree,2);random=np.random.default_rng(base_seed+50000).random(x.shape[:2]);surrogate=fixed_surrogate(flat).reshape(x.shape[:2])
 for mid in MODELS[degree]:
  folder=out/mid;folder.mkdir(exist_ok=True)
  if (folder/'selection.json').exists():continue
  runner=Runner(f['models'][mid],256);paths=[]
  for a in range(0,len(flat),256):
   b=min(a+256,len(flat));p=folder/f'prediction_{a:06d}_{b:06d}.npz';paths.append(p)
   if p.exists():continue
   length,steps=runner.predict(flat[a:b]);npz(p,length=length,scores=steps.mean(1))
   beat(out,'screening',degree=degree,split=split,model=mid,committed=b,total=len(flat))
  chunks=[np.load(p) for p in paths];length=np.concatenate([z['length'] for z in chunks]).reshape(x.shape[:2]);scores=np.concatenate([z['scores'] for z in chunks]).reshape(x.shape[:2]+(len(SCORES),))
  for z in chunks:z.close()
  values={'original':surrogate,'random':random,'entropy':scores[:,:,SCORES.index('Hnorm')],'geometry':scores[:,:,SCORES.index(metric)]}
  selected={}
  for scope in ['soft_dual','hard_radius']:
   selected[scope]={}
   for method,v in values.items():
    choices=[]
    for i in range(len(x)):
     legal=np.arange(1,33) if scope=='soft_dual' else np.flatnonzero(cost[i,1:]<=rho)+1
     ranked=legal[np.argsort(-v[i,legal],kind='stable')[:3]]
     ids=np.r_[0,ranked,np.zeros(3-len(ranked),dtype=int)].astype(int);choices.append(ids.tolist())
    selected[scope][method]=choices
  npz(folder/'predictions.npz',length=length,scores=scores)
  atomic(folder/'selection.json',dict(selected=selected,metric=metric,oracle_used=False,soft_price=1.,hard_radius=rho))
  for hook in runner.hooks:hook.remove()
  del runner;torch.cuda.empty_cache()
 # The whole model matrix has frozen shortlist selections before reference labels.
 oracle=GeoSteinerEvaluator(PROJECT/'shared/legacy_dro/WDRO-REST/algorithms/libeval.so')
 references=[]
 for i,pins in enumerate(x):
  p=out/f'reference_{i:04d}.npz'
  if p.exists():
   with np.load(p) as z:r=z['reference']
  else:
   r=np.array([oracle.solve(net,require_integral=True) for net in pins]);npz(p,reference=r)
  references.append(r)
  if i%8==0:beat(out,'offline_reference',committed_centers=i+1,total_centers=len(x))
 ref=np.stack(references);aggregate={};cells={}
 for mid in MODELS[degree]:
  with np.load(out/mid/'predictions.npz') as z:error=z['length']/ref-1
  if error.min() < -1e-5:raise RuntimeError('negative relative error')
  selection=read(out/mid/'selection.json')['selected'];cells[mid]={}
  for scope,methods in selection.items():
   selected_values={};selected_regret={};selected_cost={}
   for name,index in methods.items():
    index=np.array(index);e=error[np.arange(len(x))[:,None],index];c=cost[np.arange(len(x))[:,None],index]
    objective=e-c if scope=='soft_dual' else e
    choose=objective.argmax(-1);rows=np.arange(len(x))
    selected_values[name]=objective[rows,choose];selected_regret[name]=e[rows,choose];selected_cost[name]=c[rows,choose]
   cells[mid][scope]={name:dict(mean_objective=float(v.mean()),mean_true_regret=float(selected_regret[name].mean()),mean_transport=float(selected_cost[name].mean())) for name,v in selected_values.items()}
   for baseline in ['entropy','random','original']:
    delta=selected_values['geometry']-selected_values[baseline]
    aggregate.setdefault((scope,baseline),[]).append(delta)
   npz(out/mid/(scope+'_center_outcomes.npz'),family=family,**selected_values)
 statistics={}
 for (scope,baseline),arrays in aggregate.items():
  # Same centers shared across models: average model results before resampling centers.
  delta=np.stack(arrays).mean(0)
  byfamily={g:paired(delta[family==g],base_seed+gi) for gi,g in enumerate(GROUPS)}
  statistics[scope+'/'+baseline]=dict(**paired(delta,base_seed),families=byfamily,model_count=len(arrays))
 keys=list(statistics);adjusted=holm([statistics[k]['p_two_sided'] for k in keys])
 for key,p in zip(keys,adjusted):statistics[key]['holm_p_six_contrasts']=p
 summary=dict(degree=degree,split=split,metric=metric,centers=len(x),model_count=len(MODELS[degree]),candidate_budget=32,exact_shortlist_budget=4,rho=rho,soft_lambda=1.,
              statistics=statistics,cells=cells,reference_use='offline all33 labels after shortlist freeze; online method charges four exact candidates; geometric/entropy actor inference has extra cost',
              training_scope_note='hard radius is an auxiliary diagnostic; faithful WDRO training uses soft dual and never a hard radius filter')
 atomic(out/'summary.json',summary);atomic(out/'complete.json',dict(status='complete',centers=len(x)));beat(out,'complete')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--degree',type=int,choices=[10,30],required=True);p.add_argument('--split',choices=['dev','test'],required=True);main(p.parse_args())

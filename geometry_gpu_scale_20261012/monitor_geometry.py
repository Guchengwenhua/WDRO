"""Read-only CPU diagnostics of committed checkpoints; never feeds training."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,math,signal,fcntl,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
P=Path(__file__).resolve().parent;ROOT=P.parent;OUT=P/'artifacts/geometry_monitor'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'wdro_precision_v2/wdro_rest_min'))
import src.rest as rest
rest.LEGACY_ROOT=Path('/remote-home/Guilin/DRO/DRO/WDRO-REST')
sys.path.insert(0,str(ROOT/'geometry_uncertainty_20260930'))
from research import summarize_steps,SCORES
GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster'];STOP=False
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def stop(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial')
 with q.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 q.replace(p)
def bundle(p,**v):
 q=p.with_suffix('.npz.partial')
 with q.open('wb') as f:np.savez_compressed(f,**v)
 q.replace(p)
def corr(a,b):
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
 return float(np.corrcoef(a,b)[0,1])
def statistics(scores,error,seed):
 result={};rng=np.random.default_rng(seed)
 for name in ['G','Hnorm']:
  score=scores[:,SCORES.index(name)];rho=corr(rankdata(score),rankdata(error));pearson=corr(score,error)
  high=error>=np.quantile(error,.9);chosen=np.argsort(-score,kind='stable')[:math.ceil(.1*len(error))]
  boots=[]
  for _ in range(500):
   ix=rng.integers(0,len(error),len(error));r=corr(rankdata(score[ix]),rankdata(error[ix]))
   if r is not None:boots.append(r)
  result[name]=dict(spearman=rho,spearman_ci95=np.quantile(boots,[.025,.975]).tolist() if boots else None,
      pearson=pearson,high_error_threshold=float(np.quantile(error,.9)),
      high_error_prevalence=float(high.mean()),top10_precision=float(high[chosen].mean()),
      top10_enrichment=float(high[chosen].mean()/high.mean()),
      selected_mean_regret=float(error[chosen].mean()),all_mean_regret=float(error.mean()),
      undefined_reason='constant score or regret' if rho is None else None)
 return result
DATA={}
def data(degree):
 if degree in DATA:return DATA[degree]
 rows=[];manifest=[]
 for gi,g in enumerate(GROUPS):
  p=ROOT/f'wdro_precision_v2/experiments/cache/datasets/protocol_v1/validation/{g}_d{degree}_s202_n2000.npz'
  meta=read(p.with_suffix('.meta.json'));ref=ROOT/'wdro_precision_v2/experiments/cache/lstar/protocol_v1'/f"{meta['array_bundle_sha256']}.npz"
  with np.load(p) as a,np.load(ref) as b:
   if not np.array_equal(a['net_ids'],b['net_ids']):raise ValueError('diagnostic reference IDs mismatch')
   ix=np.random.default_rng(202610061+degree*10+gi).permutation(len(a['pins']))[:256]
   x=a['pins'][ix];ids=a['net_ids'][ix];reference=b['lstar'][ix]
  rows.append((g,x,ids,reference));manifest.append(dict(group=g,dataset=str(p),reference=str(ref),indices=ix.tolist(),net_ids=ids.astype(str).tolist()))
 atomic(OUT/f'd{degree}_fixed_sample.json',dict(n_per_group=256,source_split='historical validation',selection='fixed independent RNG, no label selection',groups=manifest))
 DATA[degree]=rows;return rows
def evaluate(folder):
 cp=folder/'checkpoints/latest.pt';state=torch.load(cp,map_location='cpu',weights_only=False)
 update=int(state['update']);identity=state['run_identity'];degree=int(identity['degree']);seed=int(identity['training_seed'])
 name=folder.parent.parent.name+'__'+folder.name if folder.parent.name=='artifacts' else folder.parent.parent.parent.name+'__'+folder.name
 dest=OUT/name;dest.mkdir(exist_ok=True);path=dest/f'u{update:06d}.json'
 if path.exists():return None
 started=time.monotonic();actor=rest.build_actor(degree,torch.device('cpu'));actor.load_state_dict(state['actor_state_dict']);actor.eval();del state
 first=[];second=[]
 hooks=[actor.ptr1.register_forward_hook(lambda m,a,o:first.append(o.detach().numpy().astype(np.float64))),
        actor.ptr2.register_forward_hook(lambda m,a,o:second.append(o.detach().numpy().astype(np.float64)))]
 evaluator=rest.RestTreeEvaluator(rest.LEGACY_ROOT/'algorithms/libeval.so');groups={};ids_all=[];scores_all=[];errors_all=[];length_all=[];ref_all=[];families=[]
 try:
  for gi,(g,x,ids,ref) in enumerate(data(degree)):
   lengths=[];scores=[]
   for start in range(0,len(x),64):
    pins=x[start:start+64];first.clear();second.clear()
    with torch.inference_mode():top,_=actor(pins,True)
    lengths.append(evaluator.lengths(pins,top.numpy()));scores.append(summarize_steps(pins,first,second).mean(1))
   length=np.concatenate(lengths);score=np.concatenate(scores);error=length/ref-1
   if not np.isfinite(score).all() or not np.isfinite(error).all():raise ValueError('nonfinite diagnostic')
   if error.min() < -1e-5:raise ValueError('reference convention inconsistent')
   groups[g]=statistics(score,error,20261006+degree*10+gi)
   ids_all.append(ids);scores_all.append(score);errors_all.append(error);length_all.append(length);ref_all.append(ref);families.extend([g]*len(error))
 finally:
  for h in hooks:h.remove()
 bundle(dest/f'u{update:06d}.npz',net_ids=np.concatenate(ids_all),scores=np.concatenate(scores_all),regret=np.concatenate(errors_all),length=np.concatenate(length_all),reference=np.concatenate(ref_all),family=np.array(families),score_names=np.array(SCORES))
 result=dict(job=folder.name,source_dir=str(folder),checkpoint=str(cp),update=update,training_batch=int(identity['batch_size']),
   sample_updates=update*int(identity['batch_size']),degree=degree,training_seed=seed,n=1024,groups=groups,
   device='cpu',dtype='FP32',inference_batch=64,seconds=time.monotonic()-started,finished=time.time(),
   interpretation='Exploratory within-family checkpoint correlations; cached independent solver reference; validation reused, not fresh holdout; repeated unadjusted intervals; no training/selection feedback')
 atomic(path,result);print(json.dumps(dict(job=folder.name,update=update,seconds=result['seconds'],spearman_G={g:v['G']['spearman'] for g,v in groups.items()})),flush=True)
 return result
def render():
 rows=[]
 for p in sorted(OUT.glob('*/u*.json')):
  d=read(p)
  for g,v in d['groups'].items():
   for metric,r in v.items():rows.append(dict(job=d['job'],source_dir=d['source_dir'],degree=d['degree'],seed=d['training_seed'],update=d['update'],sample_updates=d['sample_updates'],group=g,metric=metric,**r))
 import csv
 if rows:
  q=OUT/'correlations.csv.partial'
  with q.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  q.replace(OUT/'correlations.csv')
 latest={}
 for r in rows:
  key=(r['source_dir'],r['group'],r['metric'])
  if key not in latest or latest[key]['update']<r['update']:latest[key]=r
 lines=['训练期间几何不确定性诊断','',
 '只读CPU诊断；每族256个固定历史validation实例；误差=L/独立缓存参考−1。G与Hnorm相关性是探索性结果，区间未作跨检查点多重比较校正，不反馈训练或模型选择。CPU batch64与正式GPU测试口径不同，不能替代正式性能指标。',
 '', '| 任务 | 更新 | 分布 | 指标 | Spearman | 95%区间 | 高分前10%困难富集倍数 |','|---|---:|---|---|---:|---|---:|']
 for r in latest.values():lines.append(f"| {r['job']} | {r['update']} | {r['group']} | {r['metric']} | {r['spearman']} | {r['spearman_ci95']} | {r['top10_enrichment']:.3f} |")
 lines+=['','完整时间序列见同目录 correlations.csv；每个JSON/NPZ记录实际观察到的检查点，未观察的中间版本没有补写。']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
def main():
 lock=(OUT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 seen={};failures={}
 while not STOP:
  changed=False;targets={}
  for j in read(P/'jobs.json')['jobs']:
   if j['kind']=='train':
    folder=Path(j['checkpoint_dir']).parent;targets[str(folder)]=folder
  for folder in targets.values():
   if STOP:break
   cp=folder/'checkpoints/latest.pt'
   if not cp.exists():continue
   st=cp.stat();stamp=(st.st_size,st.st_mtime_ns)
   if seen.get(str(cp))==stamp:continue
   atomic(OUT/'heartbeat.json',dict(pid=os.getpid(),status='evaluating',job=folder.name,time=time.time(),device='cpu',threads=1))
   try:
    result=evaluate(folder);seen[str(cp)]=stamp;changed |= result is not None
    failures.pop(str(cp),None)
   except Exception as e:
    failures[str(cp)]=dict(error=repr(e),time=time.time());atomic(OUT/'errors.json',failures);traceback.print_exc()
  if changed:render()
  controller=read(P/'artifacts/state.json');closed=controller['status']=='closed'
  atomic(OUT/'heartbeat.json',dict(pid=os.getpid(),status='complete' if closed else 'waiting_checkpoints',time=time.time(),device='cpu',threads=1,observed_files=len(seen),errors=failures))
  if closed:break
  for _ in range(60):
   if STOP:break
   time.sleep(1)
if __name__=='__main__':main()

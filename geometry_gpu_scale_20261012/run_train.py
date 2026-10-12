import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from adapters import *
from accelerated_config import configure,CONFIG
import faithful_trainer
from geometry_policy import search_batch
faithful_trainer.search_batch=search_batch
from src.rest import RestTreeEvaluator,build_actor
from src.checkpoints import atomic_torch_save

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial')
 with q.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 q.replace(p)
def save_npz(p,**v):
 q=p.with_suffix('.npz.partial')
 with q.open('wb') as f:np.savez_compressed(f,**v);f.flush();os.fsync(f.fileno())
 q.replace(p)
def evaluate(cp,out,degree,device,evaluator):
 c=torch.load(cp,map_location=device,weights_only=False);actor=build_actor(degree,device);actor.load_state_dict(c['actor_state_dict']);actor.eval();groups={}
 for group in ['uniform','mix_gauss','edge_bias','grid_cluster']:
  target=out/f'{group}.npz'
  if target.exists():
   with np.load(target) as z:error=z['regret']
  else:
   data,meta=datasets.load_dataset(BASE/f'experiments/cache/datasets/protocol_v1/test/{group}_d{degree}_s303_n10000.npz',verify=False)
   ref,_=datasets.load_dataset(BASE/'experiments/cache/lstar/protocol_v1'/f"{meta['array_bundle_sha256']}.npz",verify=False)
   if not np.array_equal(data['net_ids'],ref['net_ids']):raise ValueError('Reference ID mismatch')
   lengths=[]
   with torch.no_grad():
    for start in range(0,len(data['pins']),1024):
     x=data['pins'][start:start+1024];top,_=actor(x,True);lengths.append(evaluator.lengths(x,top.cpu().numpy()))
   length=np.concatenate(lengths);error=length/ref['lstar']-1
   save_npz(target,length=length,reference=ref['lstar'],regret=error,net_ids=data['net_ids'])
  groups[group]=dict(mean=float(error.mean()),p95=float(np.quantile(error,.95)),n=len(error))
 return dict(groups=groups,overall_mean=float(np.mean([v['mean'] for v in groups.values()])),worst_ood_mean=max(v['mean'] for k,v in groups.items() if k!='uniform'),checkpoint_update=c['update'],inference='greedy FP32 batch1024, unpadded tail')

def main(args):
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(float(os.environ.get('RESEARCH_CUDA_FRACTION','.55')))
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 name=f'd{args.degree}_{args.arm}_s{args.seed}';out=HERE/'artifacts/train'/name;stage=out/f'u{args.stop_after}';stage.mkdir(parents=True,exist_ok=True)
 if (stage/'complete.json').exists():print('Already completed',name,args.stop_after,flush=True);return
 protocol=read(BASE/'wdro_rest_min/configs/protocol_v1.json');trainer=read(BASE/'wdro_rest_min/configs/stage2_trainer_v1.json');baseline=read(BASE/'wdro_rest_min/configs/stage3_baselines_v1.json');wdro=read(BASE/f'wdro_rest_min/configs/stage6/d{args.degree}_q75.json')
 configure(protocol,trainer,baseline,625);wdro.update(batch_size=4096,updates=625,research_protocol_override=True)
 SETTINGS.update(arm='geo',degree=args.degree,geometry_arm=args.arm,geometry_metric='G',device=device,pool=[],rest=RestTreeEvaluator(LIB),geosteiner_workers=4,ranking_microbatch=4096)
 actual=dict(degree=args.degree,arm=args.arm,seed=args.seed,protocol=protocol,trainer=trainer,wdro=wdro,implementation=CONFIG,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
 old=out/'run_config.json'
 if old.exists():
  prev=read(old);prev.pop('physical_gpu',None);now=actual.copy();now.pop('physical_gpu')
  if prev!=now:raise ValueError('Frozen run configuration changed')
 else:write(old,actual)
 started=time.monotonic();first=[None]
 def heartbeat(step,status,detail):
  if first[0] is None:first[0]=step
  elapsed=time.monotonic()-started;rate=elapsed/max(1,step-first[0]+1)
  h=dict(step=step,steps=625,stage_target=args.stop_after,status=status,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],time=time.time(),seconds_per_update_estimate=rate,remaining_training_seconds_estimate=rate*(625-step),detail=detail)
  write(out/'heartbeat.json',h)
  print(json.dumps(dict(job=name,step=step,target=args.stop_after,elapsed=elapsed,shortlist_seconds=detail.get('shortlist_seconds'),search_seconds=detail.get('search_seconds'))),flush=True)
 cp=out/'checkpoints/latest.pt'
 result=faithful_trainer.train_wdro(wdro_config=wdro,rho_mode='q75',baseline_config=baseline,protocol=protocol,trainer_config=trainer,degree=args.degree,training_seed=args.seed,expected_updates=625,stop_after=args.stop_after,device=device,run_id=name,checkpoint_dir=out/'checkpoints',log_dir=out/'logs',resume=cp.exists(),heartbeat_callback=heartbeat,physical_gpu_index=int(os.environ['CUDA_VISIBLE_DEVICES']),host_runtime_manifest=str(HERE/'protocol.json'),geosteiner_workers=4)
 if result.get('completed_updates')!=args.stop_after:
  write(out/'paused.json',dict(status='paused',result=result));return
 (stage/'checkpoints').mkdir(exist_ok=True)
 for which in ['latest','best']:
  dest=stage/'checkpoints'/f'{which}.pt'
  if not dest.exists():atomic_torch_save(torch.load(out/'checkpoints'/f'{which}.pt',map_location='cpu',weights_only=False),dest)
  ev=stage/f'test_{which}';ev.mkdir(exist_ok=True);write(stage/f'test_{which}.json',evaluate(dest,ev,args.degree,device,SETTINGS['rest']))
 write(stage/'complete.json',dict(status='complete',degree=args.degree,arm=args.arm,seed=args.seed,updates=args.stop_after,expected_updates=625,result=result,wall_seconds_this_process=time.monotonic()-started))
 if args.stop_after==625:write(out/'complete.json',dict(status='complete',updates=625,stage=str(stage)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--degree',type=int,choices=[10,30],required=True);p.add_argument('--arm',choices=['original','random','entropy','geometry'],required=True);p.add_argument('--seed',type=int,choices=[101,102,103],required=True);p.add_argument('--stop-after',type=int,choices=[125,625],required=True);main(p.parse_args())

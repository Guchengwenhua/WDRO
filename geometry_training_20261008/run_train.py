import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from adapters import *
from accelerated_config import configure,CONFIG
import faithful_trainer
from geometry_policy import search_batch as geometry_search
faithful_trainer.search_batch=geometry_search
from faithful_trainer import train_wdro
from src.rest import RestTreeEvaluator,build_actor
from src.checkpoints import atomic_torch_save

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def npz(path,**arrays):
 path=Path(path);tmp=path.with_suffix('.npz.partial')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def evaluate_test(cp,out,device,rest_eval):
 actor=build_actor(10,device);c=torch.load(cp,map_location=device,weights_only=False);actor.load_state_dict(c['actor_state_dict']);actor.eval();groups={}
 for g in ['uniform','mix_gauss','edge_bias','grid_cluster']:
  target=out/f'{g}.npz'
  if target.exists():
   with np.load(target) as z:v=z['regret']
  else:
   data,m=datasets.load_dataset(BASE/f'experiments/cache/datasets/protocol_v1/test/{g}_d10_s303_n10000.npz',verify=False);ref,_=datasets.load_dataset(BASE/'experiments/cache/lstar/protocol_v1'/f"{m['array_bundle_sha256']}.npz",verify=False)
   if not np.array_equal(data['net_ids'],ref['net_ids']):raise ValueError('test reference ID mismatch')
   ls=[]
   with torch.no_grad():
    for i in range(0,len(data['pins']),1024):
     x=data['pins'][i:i+1024];top,_=actor(x,True);ls.append(rest_eval.lengths(x,top.cpu().numpy()))
   length=np.concatenate(ls);v=length/ref['lstar']-1;npz(target,regret=v,length=length,reference=ref['lstar'],net_ids=data['net_ids'])
  groups[g]={'mean':float(v.mean()),'p95':float(np.quantile(v,.95)),'n':len(v)}
 return {'groups':groups,'overall_mean':float(np.mean([r['mean'] for r in groups.values()])),'worst_ood_mean':max(groups[g]['mean'] for g in groups if g!='uniform'),'checkpoint_update':c['update'],'inference':'accelerated batch1024 unpadded tail, deterministic greedy eval'}
def main(args):
 from week_common import runtime
 runtime(False)
 gate=read(HERE/'artifacts/gate.json')
 if not gate.get('allow'):raise RuntimeError('development gate denied training: '+gate.get('reason',''))
 torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(float(os.environ.get("RESEARCH_CUDA_FRACTION","0.55")),device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 out=HERE/'artifacts'/'train'/f'{args.arm}_s{args.seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():print('Already complete',out,flush=True);return
 if args.stop_after<125 and (out/f'segment_{args.stop_after:04d}.json').exists():return
 protocol=read(BASE/'wdro_rest_min/configs/protocol_v1.json');trainer=read(BASE/'wdro_rest_min/configs/stage2_trainer_v1.json');wdro=read(BASE/'wdro_rest_min/configs/stage6/d10_q75.json');baseline=read(BASE/'wdro_rest_min/configs/stage3_baselines_v1.json');configure(protocol,trainer,baseline,125)
 pool=[]
 for row in pool:
  st=Path(row['path']).stat()
  if st.st_size!=row['size_bytes'] or st.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('bank input changed '+row['path'])
 SETTINGS.update(arm='geo',geometry_arm=args.arm,geometry_metric=gate['metric'],device=device,pool=pool,rest=RestTreeEvaluator(LIB))
 write(out/'run_config.json',{'arm':args.arm,'seed':args.seed,'exploratory':True,'gate_override':gate,'protocol':protocol,'trainer':trainer,'wdro':wdro,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'resource_override':'rootcontroller selects CUDA_VISIBLE_DEVICES; memoryfraction0.55/threads1; original scientific loop preserved','bookkeeping':'no content hashes, no unsolicited inverse audit, no scalar cache; source and input metadata recorded','reference':'GeoSteiner exact binary scale v2 for every arm','geometry_metric':gate['metric'],'search_arm':args.arm,'change':'only shortlist ranking; no hard candidate radius; same32+nominal candidates and4exact budget'})
 started=time.monotonic();start_update=[None]
 def heartbeat(step,status,detail):
  if start_update[0] is None:start_update[0]=step
  elapsed=time.monotonic()-started;count=max(1,step-start_update[0]+1);write(out/'heartbeat.json',{'step':step,'steps':125,'status':status,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'time':time.time(),'elapsed_seconds':elapsed,'seconds_per_update_estimate':elapsed/count,'remaining_training_seconds_estimate':elapsed/count*(125-step),'estimate_caveat':'includes initialization until amortized, excludes final test','detail':detail});print(json.dumps({'arm':args.arm,'seed':args.seed,'step':step,'status':status,'seconds':elapsed}),flush=True)
 result=train_wdro(wdro_config=wdro,rho_mode='q75',baseline_config=baseline,protocol=protocol,trainer_config=trainer,degree=10,training_seed=args.seed,expected_updates=125,stop_after=args.stop_after,device=device,run_id=f'geometry_exploratory_20261008_{args.arm}_s{args.seed}',checkpoint_dir=out/'checkpoints',log_dir=out/'logs',resume=(out/'checkpoints/latest.pt').exists(),heartbeat_callback=heartbeat,physical_gpu_index=int(os.environ.get('CUDA_VISIBLE_DEVICES','5')),host_runtime_manifest=str(HERE/'protocol.json'),geosteiner_workers=1)
 if args.stop_after<125 and result.get('completed_updates',0)>=args.stop_after:
  write(out/f'segment_{args.stop_after:04d}.json',dict(status='prefix_completed_not_full_training',completed_updates=result['completed_updates'],expected_updates=125,wall_seconds_this_process=time.monotonic()-started,checkpoint=str(out/'checkpoints/latest.pt'),test_used=False))
  return
 if result.get('completed_updates')!=125:return
 for name in ['latest','best']:
  dest=out/f'test_{name}';dest.mkdir(exist_ok=True);write(out/f'test_{name}.json',evaluate_test(out/f'checkpoints/{name}.pt',dest,device,SETTINGS['rest']))
 write(out/'complete.json',{'status':'complete','updates':125,'result':result,'wall_seconds_this_process':time.monotonic()-started})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['original','random','entropy','geometry'],required=True);p.add_argument('--seed',type=int,choices=[101,102,103],required=True);p.add_argument('--stop-after',type=int,default=125);main(p.parse_args())

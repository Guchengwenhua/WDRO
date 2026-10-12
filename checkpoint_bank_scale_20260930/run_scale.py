import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from adapters import *
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
    for i in range(0,len(data['pins']),256):
     x=data['pins'][i:i+256];top,_=actor(x,True);ls.append(rest_eval.lengths(x,top.cpu().numpy()))
   length=np.concatenate(ls);v=length/ref['lstar']-1;npz(target,regret=v,length=length,reference=ref['lstar'],net_ids=data['net_ids'])
  groups[g]={'mean':float(v.mean()),'p95':float(np.quantile(v,.95)),'n':len(v)}
 return {'groups':groups,'overall_mean':float(np.mean([r['mean'] for r in groups.values()])),'worst_ood_mean':max(groups[g]['mean'] for g in groups if g!='uniform'),'checkpoint_update':c['update'],'inference':'original batch256 unpadded tail, deterministic greedy eval'}
def main(args):
 torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.1,device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 out=HERE/'artifacts'/f'{args.arm}_s{args.seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():print('Already complete',out,flush=True);return
 protocol=read(BASE/'wdro_rest_min/configs/protocol_v1.json');trainer=read(BASE/'wdro_rest_min/configs/stage2_trainer_v1.json');wdro=read(BASE/'wdro_rest_min/configs/stage6/d10_q75.json');baseline=read(BASE/'wdro_rest_min/configs/stage3_baselines_v1.json')
 pool=read(HERE/'inputs.json')['eligible']
 for row in pool:
  st=Path(row['path']).stat()
  if st.st_size!=row['size_bytes'] or st.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('bank input changed '+row['path'])
 SETTINGS.update(arm=args.arm,device=device,pool=pool,rest=RestTreeEvaluator(LIB))
 write(out/'run_config.json',{'arm':args.arm,'seed':args.seed,'protocol':protocol,'trainer':trainer,'wdro':wdro,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'resource_override':'new authorized experiment GPU5/memoryfraction.1/threads1; frozen scientific settings preserved','bookkeeping':'no content hashes, no unsolicited inverse audit, no scalar cache; source and input metadata recorded','reference_bank':'historical K3' if args.arm=='k3' else 'anchor+initial legacy snapshot; after50 anchor+latest2 lagged snapshots, no ranking' if args.arm=='lagged3' else 'GeoSteiner exact binary scale v2'})
 started=time.monotonic();start_update=[None]
 def heartbeat(step,status,detail):
  if start_update[0] is None:start_update[0]=step
  elapsed=time.monotonic()-started;count=max(1,step-start_update[0]+1);write(out/'heartbeat.json',{'step':step,'steps':2000,'status':status,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'time':time.time(),'elapsed_seconds':elapsed,'seconds_per_update_estimate':elapsed/count,'remaining_training_seconds_estimate':elapsed/count*(2000-step),'estimate_caveat':'includes initialization until amortized, excludes final test','detail':detail});print(json.dumps({'arm':args.arm,'seed':args.seed,'step':step,'status':status,'seconds':elapsed}),flush=True)
 result=train_wdro(wdro_config=wdro,rho_mode='q75',baseline_config=baseline,protocol=protocol,trainer_config=trainer,degree=10,training_seed=args.seed,expected_updates=2000,stop_after=2000,device=device,run_id=f'scale_{args.arm}_s{args.seed}',checkpoint_dir=out/'checkpoints',log_dir=out/'logs',resume=(out/'checkpoints/latest.pt').exists(),heartbeat_callback=heartbeat,physical_gpu_index=int(os.environ.get('CUDA_VISIBLE_DEVICES','5')),host_runtime_manifest=str(HERE/'protocol.json'),geosteiner_workers=1)
 if result.get('completed_updates')!=2000:return
 for name in ['latest','best']:
  dest=out/f'test_{name}';dest.mkdir(exist_ok=True);write(out/f'test_{name}.json',evaluate_test(out/f'checkpoints/{name}.pt',dest,device,SETTINGS['rest']))
 write(out/'complete.json',{'status':'complete','updates':2000,'result':result,'wall_seconds_this_process':time.monotonic()-started})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['geo','k3','lagged3'],required=True);p.add_argument('--seed',type=int,choices=[101,102,103],required=True);main(p.parse_args())

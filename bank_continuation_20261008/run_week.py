import argparse,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
_RUNTIME_ROOT=Path(__file__).resolve().parent/'artifacts'
for _key,_leaf in [('XDG_CACHE_HOME','cache/xdg'),('TORCH_HOME','cache/torch'),('CUDA_CACHE_PATH','cache/cuda'),('TRITON_CACHE_DIR','cache/triton'),('TMPDIR','tmp')]:
 _path=_RUNTIME_ROOT/_leaf;_path.mkdir(parents=True,exist_ok=True);os.environ[_key]=str(_path)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from adapters import *
from accelerated_config import configure,CONFIG
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
def evaluate_test(cp,out,device,rest_eval,degree):
 actor=build_actor(degree,device);c=torch.load(cp,map_location=device,weights_only=False);actor.load_state_dict(c['actor_state_dict']);actor.eval();groups={}
 for g in ['uniform','mix_gauss','edge_bias','grid_cluster']:
  target=out/f'{g}.npz'
  if target.exists():
   with np.load(target) as z:v=z['regret']
  else:
   data,m=datasets.load_dataset(BASE/f'experiments/cache/datasets/protocol_v1/test/{g}_d{degree}_s303_n10000.npz',verify=False);ref,_=datasets.load_dataset(BASE/'experiments/cache/lstar/protocol_v1'/f"{m['array_bundle_sha256']}.npz",verify=False)
   if not np.array_equal(data['net_ids'],ref['net_ids']):raise ValueError('test reference ID mismatch')
   ls=[]
   with torch.no_grad():
    for i in range(0,len(data['pins']),1024):
     x=data['pins'][i:i+1024];top,_=actor(x,True);ls.append(rest_eval.lengths(x,top.cpu().numpy()))
   length=np.concatenate(ls);v=length/ref['lstar']-1;npz(target,regret=v,length=length,reference=ref['lstar'],net_ids=data['net_ids'])
  groups[g]={'mean':float(v.mean()),'p95':float(np.quantile(v,.95)),'n':len(v)}
 return {'groups':groups,'overall_mean':float(np.mean([r['mean'] for r in groups.values()])),'worst_ood_mean':max(groups[g]['mean'] for g in groups if g!='uniform'),'checkpoint_update':c['update'],'inference':'accelerated batch1024 unpadded tail, deterministic greedy eval'}
def main(args):
 torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(float(os.environ.get("RESEARCH_CUDA_FRACTION","0.55")),device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 arm=args.arm
 if arm=='selected':
  selection=read(args.selection_file)
  if not selection.get('allow'):raise RuntimeError('Selected-arm gate is closed')
  arm=selection['selected_arm']
 out=HERE/'artifacts'/args.job_id;out.mkdir(parents=True,exist_ok=True)
 if args.evaluate_only and (out/'evaluation_complete.json').exists():print('Evaluation already complete',out,flush=True);return
 if not args.evaluate_only and (out/'complete.json').exists():print('Training already complete',out,flush=True);return
 protocol=read(BASE/'wdro_rest_min/configs/protocol_v1.json');trainer=read(BASE/'wdro_rest_min/configs/stage2_trainer_v1.json');wdro=read(BASE/f'wdro_rest_min/configs/stage6/d{args.degree}_q75.json');baseline=read(BASE/'wdro_rest_min/configs/stage3_baselines_v1.json');configure(protocol,trainer,baseline,args.updates)
 pool=read(HERE/f'inputs_d{args.degree}.json')['eligible']
 for row in pool:
  st=Path(row['path']).stat()
  if st.st_size!=row['size_bytes'] or st.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('bank input changed '+row['path'])
 SETTINGS.update(arm=arm,degree=args.degree,target_updates=args.updates,device=device,pool=pool,rest=RestTreeEvaluator(LIB),import_source=args.import_from,out=str(out))
 import_source=Path(args.import_from) if args.import_from else None
 if import_source and not (out/'checkpoints/latest.pt').exists():
  (out/'checkpoints').mkdir(exist_ok=True)
  for name in ['latest','best']:
   state=torch.load(import_source/f'checkpoints/{name}.pt',map_location='cpu',weights_only=False);atomic_torch_save(state,out/f'checkpoints/{name}.pt')
   if name=='latest':write(out/'import_cost_baseline.json',{'source':str(import_source),'source_update':state['update'],'accounting':state['reference_state']['accounting']})
 # Freeze actual command independently of source-metadata identity.
 frozen_command=vars(args).copy();frozen_command.pop('evaluate_only');frozen_command['actual_arm']=arm
 command_path=out/'command.json'
 if command_path.exists() and read(command_path)!=frozen_command:raise ValueError('Resume command differs from frozen job')
 if not command_path.exists():write(command_path,frozen_command)
 write(out/'run_config.json',{'arm':arm,'seed':args.seed,'degree':args.degree,'target_updates':args.updates,'protocol':protocol,'trainer':trainer,'wdro':wdro,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'resource_override':'controller-assigned CUDA_VISIBLE_DEVICES/memoryfraction0.55/threads1; frozen scientific settings preserved','bookkeeping':'no content hashes, no unsolicited inverse audit, no scalar cache; source and input metadata recorded','reference_bank':'historical K3' if arm=='k3' else 'anchor+initial legacy snapshot; after50 anchor+latest2 lagged snapshots, no ranking' if arm=='lagged3' else 'validation-qualified best3 refreshed every100' if arm=='qualifiedbest3' else 'GeoSteiner exact binary scale v2'})
 started=time.monotonic();start_update=[None]
 def heartbeat(step,status,detail):
  if start_update[0] is None:start_update[0]=step
  elapsed=time.monotonic()-started;count=max(1,step-start_update[0]+1);write(out/'heartbeat.json',{'step':step,'steps':args.updates,'status':status,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'time':time.time(),'elapsed_seconds':elapsed,'seconds_per_update_estimate':elapsed/count,'remaining_training_seconds_estimate':elapsed/count*(args.updates-step),'estimate_caveat':'includes initialization until amortized, excludes final test','detail':detail});print(json.dumps({'arm':arm,'seed':args.seed,'step':step,'status':status,'seconds':elapsed}),flush=True)
 if args.evaluate_only:
  if not (out/'complete.json').exists():raise RuntimeError('Evaluation requires completed target training')
  for name in ['latest','best']:
   dest=out/f'test_{name}';dest.mkdir(exist_ok=True);write(out/f'test_{name}.json',evaluate_test(out/f'checkpoints/{name}.pt',dest,device,SETTINGS['rest'],args.degree))
  write(out/'evaluation_complete.json',{'complete':True,'status':'complete','degree':args.degree,'updates':args.updates,'wall_seconds_this_process':time.monotonic()-started});return
 result=train_wdro(wdro_config=wdro,rho_mode='q75',baseline_config=baseline,protocol=protocol,trainer_config=trainer,degree=args.degree,training_seed=args.seed,expected_updates=args.updates,stop_after=args.updates,device=device,run_id=args.job_id,checkpoint_dir=out/'checkpoints',log_dir=out/'logs',resume=(out/'checkpoints/latest.pt').exists(),heartbeat_callback=heartbeat,physical_gpu_index=int(os.environ.get('CUDA_VISIBLE_DEVICES','5')),host_runtime_manifest=str(HERE/'protocol.json'),geosteiner_workers=1)
 completed=int(result.get('completed_updates',0))
 if completed<args.updates:
  write(out/'paused.json',{'complete':False,'status':'paused','completed_updates':completed,'target_updates':args.updates});print('Paused without completion marker',flush=True);return
 # Early-return resumed-at-target case must still emit a valid completion and dev record.
 checkpoint=torch.load(out/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
 last_validation=result.get('last_validation')
 if last_validation is None:
  actor=build_actor(args.degree,device);actor.load_state_dict(checkpoint['actor_state_dict']);from src.accounting import CallLedger
  last_validation=evaluate_validation(actor,SETTINGS['rest'],protocol,degree=args.degree,device=device,batch_size=trainer['validation_batch_size'],ledger=CallLedger())
 write(out/'dev.json',{'complete':True,'arm':arm,'degree':args.degree,'seed':args.seed,'updates':completed,'last_validation':last_validation,'best':checkpoint['best'],'reference_state':checkpoint['reference_state'],'wall_seconds_this_process':time.monotonic()-started})
 write(out/'complete.json',{'complete':True,'status':'complete','updates':completed,'degree':args.degree,'arm':arm,'seed':args.seed,'result':result,'wall_seconds_this_process':time.monotonic()-started})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['geo','k3','lagged3','qualifiedbest3','selected'],required=True);p.add_argument('--seed',type=int,choices=[101,102,103],required=True);p.add_argument('--degree',type=int,choices=[10,30],required=True);p.add_argument('--updates',type=int,choices=[125,625],required=True);p.add_argument('--job-id',required=True);p.add_argument('--import-from');p.add_argument('--selection-file');p.add_argument('--evaluate-only',action='store_true');main(p.parse_args())

from common import *
import torch,argparse,time
from src.rest import build_actor,RestTreeEvaluator

def main(seed):
 start=time.monotonic();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.06)
 actor=build_actor(30,torch.device('cuda'));rest=RestTreeEvaluator(CFG['library']);data=npz(ART/'data.npz');mask=np.isin(data['split'],['B_test','C_external']);testx=data['pins'][mask];valid=data['split']=='A_val';vx=data['pins'][valid];ds=data['design_ids'][valid];refs=data['lstar'][valid];rows=[];fail=[];calls=0
 @torch.no_grad()
 def score(x):
  nonlocal calls
  actor.eval();out=[]
  for b in range(0,len(x),256):
   part=np.asarray(x[b:b+256],np.float32);n=len(part)
   if n<256:part=np.concatenate([part,np.repeat(part[-1:],256-n,axis=0)])
   topo,_=actor(part,True);out.append(rest.lengths(part[:n],topo[:n].cpu().numpy()));calls+=n
  return np.concatenate(out)
 # Reproduce complete first and final original batches, including original tail padding.
 sample=np.r_[np.arange(256),np.arange((len(testx)//256)*256,len(testx))]
 for path in sorted((ART/'test').glob(f'*s{seed}*.npz')):
  f=npz(path);m=str(f['method'][0]);g=int(f['generator'][0]);b=int(f['budget'][0]);role=str(f['role'][0]);folder=ART/'runs'/f'{m}_g{g}_s{seed}'
  cpp=Path(CFG['parent_template'].format(seed=seed)) if m=='parent' else folder/f'{role}_{b}.pt'
  cp=torch.load(cpp,map_location='cpu',weights_only=False);actor.load_state_dict(cp['actor_state_dict'])
  predicted=np.r_[score(testx[:256]),score(testx[(len(testx)//256)*256:])];err=float(np.max(np.abs(predicted-f['length'][sample])))
  rows.append(dict(file=path.name,checkpoint=str(cpp),kind='natural_test',nets=len(sample),max_abs_length_error=err))
  if err>1e-7:fail.append(rows[-1])
  if m!='parent' and b==2048:
   vv=score(vx)/refs-1;macro=float(np.mean([vv[ds==d].mean() for d in np.unique(ds)]));vals=cp['validations'];v=next(v for v in vals if v['update']==cp['update']);diff=abs(macro-v['macro'])
   rows.append(dict(file=path.name,checkpoint=str(cpp),kind='full_validation',nets=len(vx),max_abs_length_error=None,macro_error=diff))
   if diff>1e-8:fail.append(rows[-1])
  if m!='parent' and b==2048 and role=='val_best':
   bank=npz(ART/'pools/attack32.npz');length=score(bank['pins'].reshape(-1,30,2)).reshape(-1,32);r=length/bank['lstar']-1;saved=npz(ART/'attack'/f'attack_{m}_g{g}_s{seed}.npz');diff=float(np.max(np.abs(r.max(1)-saved['worst_regret'])))
   rows.append(dict(file=path.name,kind='full_independent_attack',nets=bank['pins'].shape[0]*32,max_regret_error=diff))
   if diff>1e-8:fail.append(rows[-1])
  print('REPLAY',seed,path.name,'error',err,flush=True)
 # Supplemental fixed-parent evaluation was missing from the original final tables.
 parent=torch.load(CFG['parent_template'].format(seed=seed),map_location='cpu',weights_only=False);actor.load_state_dict(parent['actor_state_dict']);ids=npz(ART/'training_centers.npz')['indices'];rng=np.random.default_rng(20260926);chosen=[]
 for d in sorted(np.unique(data['design_ids'][ids])):
  local=np.flatnonzero(data['design_ids'][ids]==d);chosen.extend(np.sort(rng.choice(local,min(32,len(local)),replace=False)))
 chosen=np.array(chosen);pressure=[]
 for name in ['raw_wdro','vae_shell_wdro','feature_gate','laplacian','tile_raw','set_dae_g301','graph_dae_g301','graph_tile_g301','graph_tile_g302','graph_tile_g303']:
  bank=npz(ART/'pools'/f'{name}.npz');pins=bank['pins'][chosen];cost=bank['cost'][chosen];length=score(pins.reshape(-1,30,2)).reshape(-1,8);r=length/bank['lstar'][chosen]-1
  for lam in [0.,1.]:
   slot=(r-lam*cost).argmax(1);ix=np.arange(len(chosen));pressure.append(dict(pool=name,seed=seed,lambda_value=lam,centers=len(chosen),K=8,nominal_regret=float(r[:,0].mean()),max_regret=float(r.max(1).mean()),selected_regret=float(r[ix,slot].mean()),selected_cost=float(cost[ix,slot].mean()),nominal_fraction=float((slot==0).mean())))
  np.savez_compressed(OUT/f'fixed_parent_s{seed}_{name}.npz',center_index=ids[chosen],regret=r,cost=cost)
 csvout(f'gpu_replay_s{seed}.csv',rows);csvout(f'fixed_parent_s{seed}.csv',pressure)
 write(OUT/f'gpu_replay_s{seed}.json',dict(status='PASS' if not fail else 'FAIL',failures=fail,checks=len(rows),requested_predictions=calls,seed=seed,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),max_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,seconds=time.monotonic()-start,time=now()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);main(p.parse_args().seed)

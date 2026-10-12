"""Exploratory continued WDRO: exact reference versus frozen/refreshed checkpoint banks."""
import argparse,copy,json,os,signal,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/remote-home/Guilin/DRO/Codex')
BASE=ROOT/'wdro_precision_v2'
sys.path.insert(0,str(BASE/'wdro_rest_min'))
import src.rest as restmod
restmod.LEGACY_ROOT=Path('/remote-home/Guilin/DRO/DRO/WDRO-REST')
from src.rest import build_actor,build_critic,RestTreeEvaluator
from src.geosteiner import GeoSteinerEvaluator
from src.checkpoints import atomic_torch_save,capture_rng_state,restore_rng_state
from src.datasets import load_dataset
from research_data import materialize_training_batch,generate_support
from src.candidates import shortlist_support
from stage4.dual import transport_costs,select_wdro,update_lambda
LIB=restmod.LEGACY_ROOT/'algorithms/libeval.so'
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
def event(p,d):
 with Path(p).open('a') as f:f.write(json.dumps(d)+'\n')
def atomic_npz(path,**arrays):
 path=Path(path);tmp=path.with_suffix('.npz.partial')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def pool():
 rows=[];excluded=[]
 for p in sorted((BASE/'experiments/logs').glob('stage*/formal*d10*/result.json')):
  d=read(p);b=d['best'];run=p.parent.name
  if d.get('status')!='complete' or not b.get('feasible',True):continue
  cp=BASE/'experiments/checkpoints'/p.parent.parent.name/run/'best.pt'
  if cp.is_file():rows.append(dict(path=str(cp),run=run,score=b['selection_objective'],nominal=b['nominal_mean'],validation_manifest=b['validation_manifest_sha256'],source_result=str(p)))
 rows.sort(key=lambda x:(x['score'],x['run']));out=[];seen=set()
 for r in rows:
  if r['validation_manifest'] in seen:excluded.append(dict(**r,reason='same existing validation manifest; conservatively one representative, not a weight identity claim'))
  else:seen.add(r['validation_manifest']);out.append(r)
 return out,excluded

def dataset(split,group,n=None):
 count=2000 if split=='validation' else 10000;seed=202 if split=='validation' else 303
 path=BASE/f'experiments/cache/datasets/protocol_v1/{split}/{group}_d10_s{seed}_n{count}.npz'
 a,m=load_dataset(path,verify=False)
 l,lm=load_dataset(BASE/'experiments/cache/lstar/protocol_v1'/f"{m['array_bundle_sha256']}.npz",verify=False)
 if not np.array_equal(a['net_ids'],l['net_ids']):raise ValueError('reference IDs mismatch')
 return a['pins'][:n],l['lstar'][:n],a['net_ids'][:n]

def lengths(actor,pins,rest,batch=256):
 actor.eval();out=[]
 with torch.no_grad():
  for i in range(0,len(pins),batch):
   chunk=pins[i:i+batch];count=len(chunk)
   padded=np.concatenate([chunk,np.repeat(chunk[-1:],batch-count,axis=0)],axis=0) if count<batch else chunk
   t,_=actor(padded,True);out.append(rest.lengths(chunk,t[:count].cpu().numpy()))
 return np.concatenate(out)

def evaluate(actor,rest,split,n=None,save=None):
 groups={};arrays={}
 for g in ['uniform','mix_gauss','edge_bias','grid_cluster']:
  x,y,ids=dataset(split,g,n);z=lengths(actor,x,rest);v=z/y-1
  groups[g]={'mean':float(v.mean()),'p95':float(np.quantile(v,.95)),'n':len(v)}
  arrays[g+'_regret']=v;arrays[g+'_ids']=ids
 if save:atomic_npz(save,**arrays)
 return {'groups':groups,'score':max(groups[g]['mean'] for g in groups if g!='uniform'),'nominal':groups['uniform']['mean'],'split':split}

def offline_diagnostic(actor,rest,bank,out,device,rho,price):
 from types import SimpleNamespace
 from scipy.stats import spearmanr
 x,_,ids=dataset('test','uniform',32)
 search=read(BASE/'wdro_rest_min/configs/stage3_baselines_v1.json')['search'];search['candidate_count']=8
 support=generate_support(SimpleNamespace(pins=x,net_ids=ids),training_seed=101,update=1,search=search)
 z=support.pins;flat=z.reshape(-1,10,2);current=lengths(actor,flat,rest)
 oracle=GeoSteinerEvaluator(LIB);parts=[]
 for start in range(0,len(flat),32):
  part=out/f'offline_oracle_{start:04d}.npz'
  if part.exists():
   with np.load(part) as cached:parts.append(cached['truth'])
  else:
   values=oracle.lengths(flat[start:start+32]);atomic_npz(part,truth=values);parts.append(values)
 truth=np.concatenate(parts);cost=transport_costs(x,z)
 teacher=[]
 for r in bank:
  m=build_actor(10,device);ck=torch.load(r['path'],map_location=device,weights_only=False);m.load_state_dict(ck.get('actor_state_dict',ck.get('actor')));teacher.append(lengths(m,flat,rest))
 reference=np.min(teacher,axis=0);gap=reference/truth-1;true=current/truth-1;proxy=current/reference-1
 ft=true.reshape(32,9)-price*cost;fb=proxy.reshape(32,9)-price*cost
 correlations=[float(spearmanr(a,b).statistic) for a,b in zip(ft,fb)]
 atomic_npz(out/'offline_bank_diagnostic.npz',pins=z,ids=ids,current=current,oracle=truth,bank=reference,true_regret=true,proxy_regret=proxy,bank_gap=gap,transport=cost,oracle_f=ft,bank_f=fb)
 write(out/'offline_bank_diagnostic.json',{'scope':'best validation checkpoint, fixed test-uniform 32 centers x (8 perturbations + nominal); offline only, never returned to training','oracle_calls':len(flat),'bank_members':bank,'bank_gap_mean':float(gap.mean()),'proxy_bias_mean':float((proxy-true).mean()),'argmax_agreement':float(np.mean(ft.argmax(1)==fb.argmax(1))),'mean_within_center_spearman':float(np.nanmean(correlations)),'lambda_final':price,'rho':rho})

def run(args):
 torch.set_num_threads(1);device=torch.device(args.device)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
 if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.1,device)
 protocol=read(BASE/'wdro_rest_min/configs/protocol_v1.json');protocol['protocol_id']='checkpoint-bank-continued-20260930'
 search=read(BASE/'wdro_rest_min/configs/stage3_baselines_v1.json')['search']
 cfg=vars(args).copy();cfg.update(batch=64,degree=10,refresh_interval=50,candidate_count=32,shortlist=4,training_namespace=protocol['protocol_id'],test_per_group=256,loss='C/reference-1; negative values retained',critic='unchanged original architecture for every arm',scope='exploratory continued training, not frozen formal protocol replication')
 out=HERE/'artifacts'/f'{args.arm}_s{args.seed}';out.mkdir(parents=True,exist_ok=True)
 latest=out/'latest.pt';cfgpath=out/'config.json'
 if (out/'complete.json').exists():print('Already complete:',out,flush=True);return
 if cfgpath.exists() and read(cfgpath)!=cfg:raise ValueError('existing config mismatch')
 write(cfgpath,cfg)
 inputs_path=HERE/'inputs.json'
 if inputs_path.exists():
  frozen=read(inputs_path);rows=frozen['eligible'];excluded=frozen['excluded']
 else:
  rows,excluded=pool()
  for row in rows:
   stat=Path(row['path']).stat();row['size_bytes']=stat.st_size;row['mtime_ns']=stat.st_mtime_ns
  write(inputs_path,{'eligible':rows,'excluded':excluded,'selection':'existing validation worst OOD mean with recorded feasibility; no test ranking','historical_oracle_assisted':True})
 for row in rows:
  stat=Path(row['path']).stat()
  if stat.st_size!=row['size_bytes'] or stat.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('frozen input changed: '+row['path'])
 parent=next(r for r in rows if r['run']=='formal_hard_mining_d10_s102_u2000')
 checkpoint=torch.load(parent['path'],map_location=device,weights_only=False)
 actor=build_actor(10,device);critic=build_critic(10,device)
 actor.load_state_dict(checkpoint['actor_state_dict']);critic.load_state_dict(checkpoint['critic_state_dict'])
 opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=1e-4);opt.load_state_dict(checkpoint['optimizer_state_dict'])
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
 if 'scheduler_state_dict' in checkpoint:scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
 import random
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
 rest=RestTreeEvaluator(LIB);geo=GeoSteinerEvaluator(LIB) if args.arm=='geo' else None
 k={'geo':0,'k1':1,'k3':3,'k5':5,'refresh3':3,'lagged3':3}[args.arm]
 bank=copy.deepcopy(rows[:k]);start=1;price=0.;ledger={'geosteiner_train_nets':0,'teacher_forward_nets':0,'teacher_seconds':0.,'validation_cached_oracle':True,'refresh_seconds':0.,'bank_turnovers':0};bestscore=float('inf')
 if latest.exists():
  state=torch.load(latest,map_location=device,weights_only=False)
  if state['config']!=cfg:raise ValueError('resume configuration differs')
  actor.load_state_dict(state['actor']);critic.load_state_dict(state['critic']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);bank=state['bank'];price=state['price'];start=state['step']+1;ledger=state['ledger'];bestscore=state['bestscore'];restore_rng_state(state['rng'])
 teachers=[]
 def reload_bank():
  saved_rng=capture_rng_state()
  teachers.clear()
  for r in bank:
   m=build_actor(10,device);c=torch.load(r['path'],map_location=device,weights_only=False);m.load_state_dict(c.get('actor_state_dict',c.get('actor')));m.eval();m.requires_grad_(False);teachers.append(m)
  restore_rng_state(saved_rng)
 reload_bank()
 rho=read(BASE/'experiments/cache/stage6/calibration_d10_s101.json')['hungarian_l1_mean']['q75']
 stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
 def save(step):
  atomic_torch_save({'config':cfg,'step':step,'actor':actor.state_dict(),'critic':critic.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'rng':capture_rng_state(),'bank':bank,'price':price,'ledger':ledger,'bestscore':bestscore,'parent':parent},latest)
 if start==1:
  initial_val=evaluate(actor,rest,'validation');bestscore=initial_val['score'];atomic_torch_save({'actor_state_dict':actor.state_dict(),'step':0,'validation':initial_val},out/'best.pt');event(out/'validation.jsonl',dict(step=0,**initial_val));save(0)
 segment_start=time.monotonic()
 for step in range(start,args.steps+1):
  t0=time.monotonic();b=materialize_training_batch(protocol=protocol,degree=10,training_seed=args.seed,update=step,batch_size=64)
  support=generate_support(b,training_seed=args.seed,update=step,search=search);indices,_=shortlist_support(support,4)
  z=support.pins[np.arange(64)[:,None],indices];flat=z.reshape(-1,10,2);current=lengths(actor,flat,rest)
  if geo is not None:reference=geo.lengths(flat);ledger['geosteiner_train_nets']+=len(flat)
  else:
   bt=time.monotonic();reference=np.min([lengths(m,flat,rest) for m in teachers],axis=0);ledger['teacher_forward_nets']+=len(flat)*len(teachers);ledger['teacher_seconds']+=time.monotonic()-bt
  regrets=(current/reference-1).reshape(64,4);cost=transport_costs(b.pins,z);selected,col=select_wdro(indices,regrets,cost,price,rho);price=update_lambda(price,float(cost[np.arange(64),col].mean()),rho,1.)
  x=support.pins[np.arange(64),selected];ref=reference.reshape(64,4)[np.arange(64),col]
  actor.train();critic.train();top,lp=actor(x,False);costs=rest.lengths(x,top.detach().cpu().numpy());target=torch.as_tensor(costs/ref-1,dtype=torch.float32,device=device);pred=critic(x)
  loss=((target-pred).detach()*lp).mean()+((pred-target)**2).mean()
  if not torch.isfinite(loss):raise RuntimeError('nonfinite loss before update')
  if not np.isfinite(price) or price>100:raise RuntimeError('lambda divergence')
  opt.zero_grad(set_to_none=True);loss.backward()
  ag=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);cg=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.)
  if not (torch.isfinite(ag) and torch.isfinite(cg)):raise RuntimeError('nonfinite gradient before update')
  opt.step();scheduler.step()
  if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
  event(out/'train.jsonl',{'step':step,'seconds':time.monotonic()-t0,'loss':float(loss.detach()),'surrogate_regret':float(target.mean()),'negative_target_fraction':float((target<0).float().mean()),'lambda':price,'bank_runs':[r['run'] for r in bank],'ledger':ledger.copy()})
  if step%50==0 or step==args.steps:
   refresh_t0=time.monotonic()
   val=evaluate(actor,rest,'validation');event(out/'validation.jsonl',dict(step=step,**val))
   if val['nominal']<=read(BASE/'experiments/logs/stage2'/f'formal_erm_d10_s{args.seed}_u2000/result.json')['best']['nominal_mean']+.002 and val['score']<bestscore:bestscore=val['score'];atomic_torch_save({'actor_state_dict':actor.state_dict(),'step':step,'validation':val},out/'best.pt')
   if args.arm=='refresh3' and step<args.steps:
    # Candidate is ranked only on cached validation references; bank immutable within each update/block.
    if val['nominal']<=read(BASE/'experiments/logs/stage2'/f'formal_erm_d10_s{args.seed}_u2000/result.json')['best']['nominal_mean']+.002:
     path=out/f'bank_candidate_u{step:04d}.pt';atomic_torch_save({'actor_state_dict':actor.state_dict()},path)
     bank=sorted(bank+[{'path':str(path),'run':f'online_s{args.seed}_u{step}','score':val['score'],'nominal':val['nominal']}],key=lambda r:(r['score'],r['run']))[:3];reload_bank()
   if args.arm=='lagged3' and step<args.steps:
    path=out/f'window_u{step:04d}.pt'
    atomic_torch_save({'actor_state_dict':actor.state_dict(),'step':step},path)
    candidate={'path':str(path),'run':f'window_s{args.seed}_u{step}','score':val['score'],'nominal':val['nominal'],'step':step}
    rolling=sorted([r for r in bank if r['run'].startswith('window_')]+[candidate],key=lambda r:r['step'],reverse=True)
    proposed=[copy.deepcopy(rows[0])]+rolling+copy.deepcopy(rows[1:3])
    unique=[];states=[];seen=set()
    for r in proposed:
     if r['run'] in seen:continue
     ck=torch.load(r['path'],map_location='cpu',weights_only=False);sd=ck.get('actor_state_dict',ck.get('actor'))
     if any(sd.keys()==prior.keys() and all(torch.equal(sd[key],prior[key]) for key in sd) for prior in states):continue
     seen.add(r['run']);unique.append(r);states.append(sd)
     if len(unique)==3:break
    before=[r['run'] for r in bank];bank=unique;reload_bank()
    after=[r['run'] for r in bank];changed=before!=after
    ledger['bank_turnovers']+=int(changed)
    event(out/'bank_updates.jsonl',{'completed_window_step':step,'effective_from_step':step+1,'before':before,'after':after,'changed':changed,'frozen_independent_copies':True,'policy':'anchor plus two most recent distinct window snapshots'})
   ledger['refresh_seconds']+=time.monotonic()-refresh_t0
  if step%10==0 or step==args.steps or stop[0]:
   save(step)
   elapsed=time.monotonic()-segment_start;done=step-start+1;rate=elapsed/done
   write(out/'heartbeat.json',{'step':step,'steps':args.steps,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'time':time.time(),'segment_start_step':start,'segment_seconds':elapsed,'mean_wall_seconds_per_update':rate,'remaining_seconds_estimate':rate*(args.steps-step),'estimate_scope':'remaining training/periodic validation at observed segment wall rate; excludes final test and offline diagnostic'})
  print(json.dumps({'arm':args.arm,'seed':args.seed,'step':step,'seconds':time.monotonic()-t0}),flush=True)
  if stop[0]:return
 write(out/'final_test.json',evaluate(actor,rest,'test',256,out/'final_test.npz'))
 initial_actor=build_actor(10,device);initial_actor.load_state_dict(checkpoint['actor_state_dict']);write(out/'initial_test.json',evaluate(initial_actor,rest,'test',256,out/'initial_test.npz'));del initial_actor
 best=torch.load(out/'best.pt',map_location=device,weights_only=False);actor.load_state_dict(best['actor_state_dict']);write(out/'best_test.json',evaluate(actor,rest,'test',256,out/'best_test.npz'))
 offline_diagnostic(actor,rest,bank if bank else rows[:3],out,device,rho,price)
 write(out/'complete.json',{'status':'complete','steps':args.steps,'ledger':ledger,'best_validation':best['validation'],'parent':parent,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'test_selection_used':False})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['geo','k1','k3','k5','refresh3','lagged3'],required=True);p.add_argument('--seed',type=int,choices=[101,102],required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--device',default='cuda:0');run(p.parse_args())

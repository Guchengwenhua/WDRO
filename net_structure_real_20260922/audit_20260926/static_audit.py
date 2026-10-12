from common import *
from concurrent.futures import ThreadPoolExecutor
import torch,fcntl,itertools,time
from scipy.optimize import linear_sum_assignment

def main():
 start=time.monotonic();checks=[];failures=[]
 def check(name,ok,**detail):
  checks.append(dict(check=name,pass_=bool(ok),**detail))
  if not ok:failures.append(checks[-1])
 manifest=read(ART/'completion.json');bindings=[]
 for origin,items in [('completion',manifest['artifacts']),('current_source',read(ART/'current_source_manifest.json')['files']),('input',read(ART/'input_manifest.json')['sources']),('physical_sources',read(ART/'input_manifest.json')['physical_sources']),('parents_and_library',read(ART/'tests/additional_audit.json')['reused_source_bindings'])]:
  bindings.extend([dict(origin=origin,**r) for r in items])
 paths=list({r['path'] for r in bindings})
 def hash_one(p):return p,sha(p) if Path(p).is_file() else None
 with ThreadPoolExecutor(max_workers=4) as pool:actual=dict(pool.map(hash_one,paths))
 mismatch=[dict(path=r['path'],origin=r['origin'],expected=r['sha256'],actual=actual[r['path']]) for r in bindings if r['sha256']!=actual[r['path']]]
 check('recorded_sha256_bindings',not mismatch,bindings=len(bindings),unique_files=len(paths),mismatches=mismatch);print('HASH',len(paths),'mismatch',len(mismatch),flush=True)
 q=read(ART/'queue.json');check('DAG_complete',len(q['jobs'])==166 and all(j['status']=='done' for j in q['jobs']));check('supervisor_exit',read(ART/'supervisor_exit.json')['success'])
 active=[]
 for j in q['jobs']+[q]:
  try:
   stat=Path(f"/proc/{j['pid']}/stat").read_text().split()
   if stat[21]==j['starttime'] and stat[2]!='Z':active.append(j.get('id','supervisor'))
  except (KeyError,FileNotFoundError,ProcessLookupError):pass
 check('recorded_processes_terminated',not active,live_matching_processes=active)
 with (ART/'supervisor.lock').open('r') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);lockfree=True;fcntl.flock(f,fcntl.LOCK_UN)
  except BlockingIOError:lockfree=False
 check('supervisor_lock_released',lockfree)
 data=npz(ART/'data.npz');check('net_ids_unique',len(set(data['net_ids']))==len(data['pins']))
 splitsets={s:{x.tobytes() for x in data['pins'][data['split']==s]} for s in np.unique(data['split'])}
 overlaps={a+'|'+b:len(splitsets[a]&splitsets[b]) for a,b in itertools.combinations(splitsets,2)}
 check('train_cal_val_test_geometry_isolation',all(n==0 for pair,n in overlaps.items() if pair!='B_test|C_external'),overlap_rows=overlaps)
 check('family_split_isolation',all(len(set(data['split'][data['family_ids']==f]))==1 for f in set(data['family_ids'])))
 counts={s:int((data['split']==s).sum()) for s in splitsets};check('frozen_d30_counts',counts==CFG['splits']['expected_rows_before_cross_split_dedup'],counts=counts)
 mdrows=[]
 for d in CFG['generator']['degrees']:
  sets={}
  for s in ['A_train','A_cal','A_val','B_test']:
   f=npz(ART/'multidegree'/f'd{d}_{s}.npz');sets[s]={x.tobytes() for x in f['pins']};check(f'd{d}_{s}_designs',set(f['design'])<=set(CFG['splits'][s]));mdrows.append(dict(degree=d,split=s,nets=len(f['pins']),designs=len(set(f['design']))))
  check(f'd{d}_geometry_isolation',all(not sets[a]&sets[b] for a,b in itertools.combinations(sets,2)))
 check('pretraining_train_count',sum(r['nets'] for r in mdrows if r['split']=='A_train')==40630)
 csvout('data_counts.csv',mdrows)
 centers=npz(ART/'training_centers.npz')['indices'];check('centers_train_only',np.array_equal(centers,np.flatnonzero(data['split']=='A_train')))
 for seed in CFG['seeds']:
  seq=npz(ART/'indices'/f'seed{seed}.npz')['indices'];check(f'shared_schedule_s{seed}',np.array_equal(seq,np.random.default_rng(seed).integers(0,len(centers),(2048,64))))
 runrows=[];checkpoint_count=0;parents={s:torch.load(CFG['parent_template'].format(seed=s),map_location='cpu',weights_only=False) for s in CFG['seeds']}
 for folder in sorted((ART/'runs').iterdir()):
  latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False);s=latest['seed'];method=latest['method'];g=latest['generator'];h=latest['history'];vals=latest['validations'];ledger=latest['ledger'];tag=folder.name
  check(tag+'_updates',latest['update']==2048 and [r['update'] for r in h]==list(range(1,2049)))
  check(tag+'_sample_budget',ledger['sampled_requested']==2048*64)
  check(tag+'_search_budget',ledger.get('candidate_requested',0)==(0 if method=='erm_a' else 2048*64*8))
  check(tag+'_finite_history',all(np.isfinite(v) for row in h for v in row.values() if isinstance(v,(int,float))))
  check(tag+'_BN_stats_frozen',all(torch.equal(value,parents[s][which][key]) for which in ['actor_state_dict','critic_state_dict'] for key,value in latest[which].items() if key.endswith(('running_mean','running_var','num_batches_tracked'))))
  recurrence=max(abs(row['lambda_after']-(0 if method=='erm_a' else max(0,row['lambda_before']+row['selected_cost']-latest['rho']))) for row in h)
  check(tag+'_dual_recurrence',recurrence<1e-12,max_error=recurrence)
  check(tag+'_validation_schedule',[v['update'] for v in vals]==list(range(0,2049,128)))
  for budget in [256,1024,2048]:
   allowed=[v for v in vals if v['update']<=budget];chosen=min(allowed,key=lambda v:(v['macro'],v['update']))
   for role in ['last','val_best']:
    cp=torch.load(folder/f'{role}_{budget}.pt',map_location='cpu',weights_only=False);checkpoint_count+=1
    check(f'{tag}_{role}_{budget}_identity',(cp['method'],cp['seed'],cp['generator'])==(method,s,g))
    check(f'{tag}_{role}_{budget}_selected_step',cp['update']==(budget if role=='last' else chosen['update']))
    check(f'{tag}_{role}_{budget}_finite_parameters',all(torch.isfinite(t).all().item() for typ in ['actor_state_dict','critic_state_dict'] for t in cp[typ].values()))
  runrows.append(dict(method=method,seed=s,generator=g,updates=latest['update'],best_update=latest['best']['update'],rho=latest['rho'],lambda_final=latest['lambda'],mean_cost=np.mean([r['selected_cost'] for r in h]),budget_gap=np.mean([r['budget_gap'] for r in h]),nominal_fraction=np.mean([r['nominal_fraction'] for r in h]),candidate_calls=ledger.get('candidate_requested',0),sampled_calls=ledger['sampled_requested'],training_seconds=latest['training_seconds']))
  print('RUN',tag,flush=True)
 check('formal_run_count',len(runrows)==33);csvout('training_audit.csv',runrows)
 genrows=[]
 for folder in sorted((ART/'generators').iterdir()):
  done=read(folder/'done.json');latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False);best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);pre=6000 if done['phase']=='tune' else 18000;total=7500 if done['phase']=='tune' else 22500;h=latest['history'];chosen=min([r for r in h if r['step']>pre],key=lambda r:r['validation']['score'])
  check(folder.name+'_budget',latest['step']==total and [r['step'] for r in h]==list(range(250,total+1,250)))
  check(folder.name+'_val_selection',best['step']==chosen['step'] and abs(done['best']['score']-chosen['validation']['score'])<1e-12)
  genrows.append(dict(kind=done['spec']['kind'],width=done['spec']['width'],eta=done['spec']['eta'],phase=done['phase'],seed=done['seed'],steps=total,best_step=best['step'],validation_score=done['best']['score'],seconds=done['seconds']))
 selected=read(ART/'generator_selected.json')['selected']
 for kind in ['set','graph']:
  winner=min([r for r in genrows if r['phase']=='tune' and r['kind']==kind],key=lambda r:r['validation_score']);check(kind+'_hyperparameter_selection',selected[kind]['spec']['width']==winner['width'] and selected[kind]['spec']['eta']==winner['eta'])
 check('generator_run_count',len(genrows)==12);csvout('generator_audit.csv',genrows)
 write(OUT/'static.json',dict(time=now(),status='PASS' if not failures else 'FAIL',checks=checks,failures=failures,hash_files=len(paths),hash_bindings=len(bindings),rest_checkpoint_count=checkpoint_count,seconds=time.monotonic()-start))
 print('STATIC',len(checks),'checks',len(failures),'failures',flush=True)
if __name__=='__main__':main()

"""Independent checkpoint, lifecycle, budget-trajectory and numerical review.

Reads frozen scientific artifacts only. No training, GPU inference or oracle call.
"""
from pathlib import Path
import csv,fcntl,hashlib,json,math,os,subprocess,time
import numpy as np
import torch
torch.set_num_threads(1)

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
ERRORS=[]
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(test,msg):
    if not test:ERRORS.append(msg)
def finite(obj):
    if isinstance(obj,torch.Tensor):return bool(torch.isfinite(obj).all()) if obj.is_floating_point() else True
    if isinstance(obj,np.ndarray):return bool(np.isfinite(obj).all()) if np.issubdtype(obj.dtype,np.number) else True
    if isinstance(obj,dict):return all(finite(x) for x in obj.values())
    if isinstance(obj,(list,tuple)):return all(finite(x) for x in obj)
    if isinstance(obj,float):return math.isfinite(obj)
    return True
def digest_state(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def live(record):
    try:
        tail=Path(f'/proc/{record["pid"]}/stat').read_text().rsplit(')',1)[1].split()
        return tail[0]!='Z' and tail[19]==str(record['start_ticks']) and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
    except (FileNotFoundError,KeyError):return False
def released(path):
    with path.open('r') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
        except BlockingIOError:return False
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def csv_dump(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)

freeze=read(ROOT/'artifacts/operations/source_freeze.json')
for name,value in freeze['files'].items():require(sha(ROOT/name)==value,'Frozen file changed: '+name)
queue=read(ROOT/'artifacts/operations/queue.json');delivery=read(ROOT/'artifacts/operations/delivery.json')
require(queue['status']=='complete' and not queue['running'],'Queue incomplete/live')
require(delivery['status']=='COMPLETE','Original delivery incomplete')
require(delivery['queue_sha256']==sha(ROOT/'artifacts/operations/queue.json'),'Original queue binding changed')
require(delivery['audit_sha256']==sha(ROOT/'artifacts/report/audit.json'),'Original audit binding changed')
require(not live(read(ROOT/'artifacts/operations/launch.json')['identity']),'Original scheduler still live')
require(released(ROOT/'artifacts/operations/scheduler.lock'),'Scheduler lock held')
tmux=subprocess.run(['tmux','-S',str(ROOT/'artifacts/operations/research.sock'),'list-sessions'],capture_output=True,text=True)
require(tmux.returncode!=0,'Research tmux still exists')
expected=[]
for s in P['seeds']:
    expected.append(f'warmup_s{s}')
    expected.extend(f'fixed_{m}_s{s}' for m in P['learned_methods'])
    expected.extend(f'coev_{m}_s{s}' for m in P['coev_methods'])
require(sorted(expected)==sorted(queue['complete']),'51-task queue set mismatch')
actual_dirs={d.name for d in (ROOT/'artifacts/runs').iterdir() if d.is_dir()}
require(actual_dirs==set(expected),'Unexpected/missing formal run directories')
partial=[str(p.relative_to(ROOT)) for p in (ROOT/'artifacts/runs').rglob('*') if '.partial' in p.name]
require(not partial,'Unpublished formal partials remain')
parent_digests={};runs=[];budgets=[];call_totals={};runtime=[];arrays={};all_completes={}
for rid in expected:
    out=ROOT/'artifacts/runs'/rid;m=read(out/'complete.json');all_completes[rid]=m
    c=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    require(c['step']==P[m['kind']+'_updates']==256,rid+': checkpoint step')
    require(c['binding']==m['binding'],rid+': checkpoint/manifest binding')
    require(finite(c),rid+': nonfinite checkpoint')
    require(not live(m['identity']) and released(out/'run.lock'),rid+': process or lock still active')
    require(m['device']=='cuda:0' and m['peak_allocated_bytes']>0,rid+': missing GPU evidence')
    exits=list((ROOT/'artifacts/operations').glob(rid+'_exit_a*.json'))
    require(len(exits)==queue['attempts'][rid]==1,rid+': missing/non-single exit receipt')
    if exits:require(read(exits[0])['returncode']==0,rid+': nonzero exit')
    for k in ['search_greedy_scores','sampled_actor_scores']:
        require(c['calls'][k]==m['calls'][k],rid+': checkpoint training ledger '+k)
    require(c['calls']['evaluation_greedy_scores']==0,rid+': training checkpoint contains test calls')
    for key,value in m['calls'].items():call_totals[key]=call_totals.get(key,0)+value
    actor_digest=digest_state(c['actor'])
    seed=m['seed'];kind=m['kind'];method=m['method']
    if kind=='warmup':parent_digests[seed]=actor_digest
    else:
        if kind=='fixed':require(actor_digest==parent_digests[seed],rid+': fixed scorer weights moved')
        else:require(actor_digest!=parent_digests[seed],rid+': actor never changed')
    expected_opt=512 if kind=='coev' else 256
    opt_steps={key:sorted({int(v['step']) for v in c[key]['state'].values() if 'step' in v}) for key in ['actor_optimizer','value_optimizer']}
    require(all(v==[expected_opt] for v in opt_steps.values()),rid+': optimizer step count')
    log=[json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines()]
    require([row['step'] for row in log]==list(range(1,257)),rid+': training progress sequence')
    require(finite(log),rid+': nonfinite progress')
    require(abs(c['lambda']-log[-1]['lambda_after'])<1e-12,rid+': checkpoint lambda mismatch')
    for i,row in enumerate(log):
        expected_before=5. if i==0 else log[i-1]['lambda_after']
        require(abs(row['lambda_before']-expected_before)<1e-10,rid+': lambda continuity')
        expected_after=max(0.,row['lambda_before']+P['lambda_lr']*(row['selected_cost']-P['rho'])) if kind=='coev' and method!='erm' else row['lambda_before']
        require(abs(row['lambda_after']-expected_after)<1e-10,rid+': dual update incorrect')
    if c['attacker']:
        a=c['attacker'];factor=1 if method in ['reinforce','maddpg_bandit'] else P['attacker']['epochs']
        require(a['updates']==256*factor*(2 if kind=='coev' else 1),rid+': attacker update count')
        if method=='maddpg_bandit':require(len(a['replay']['reward'])==(224 if kind=='coev' else 8192),rid+': stale/missing replay')
    if method=='population_ppo':require(len(c['history'])==3,rid+': history count')
    if kind=='coev' and method!='erm':
        costs=np.array([row['selected_cost'] for row in log]);prices=np.array([row['lambda_after'] for row in log])
        budgets.append(dict(run_id=rid,method=method,seed=seed,mean_cost=float(costs.mean()),last64_mean_cost=float(costs[-64:].mean()),rho=P['rho'],mean_gap=float(costs.mean()-P['rho']),last64_gap=float(costs[-64:].mean()-P['rho']),final_lambda=float(prices[-1]),no_move_fraction=float(np.mean([row['no_move_fraction'] for row in log]))))
    runs.append(dict(run_id=rid,actor_sha256=actor_digest,optimizer_steps=opt_steps,step=c['step'],live=False,lock_released=True,finite_checkpoint=True))
    runtime.append(dict(run_id=rid,kind=kind,method=method,seed=seed,elapsed_seconds=m['elapsed_seconds'],training_seconds=log[-1]['elapsed_seconds'],evaluation_and_publication_seconds=m['elapsed_seconds']-log[-1]['elapsed_seconds'],peak_allocated_mib=m['peak_allocated_bytes']/2**20,**m['calls']))
    if kind!='warmup':
        with np.load(out/'results.npz') as z:arrays[rid]={k:z[k] for k in z.files}
    print('verified',rid,flush=True)

# Independently recompute table entries directly from stored arrays.
metric_checks=0;orig=list(csv.DictReader((ROOT/'artifacts/report/metrics.csv').open()))
for row in orig:
    rid=f'{row["kind"]}_{row["method"]}_s{row["seed"]}';a=arrays[rid]
    if row['kind']=='fixed':v=a[f'{row["distribution"]}__{row["price"]}__{row["evaluated_attacker"]}__{row["metric"]}']
    elif row['distribution']=='equal_group_aggregate':v=np.concatenate([a[f'{d}__{row["metric"]}'] for d in P['distributions']])
    else:v=a[f'{row["distribution"]}__{row["metric"]}']
    require(abs(float(row['mean'])-float(v.mean()))<1e-12,'Table mean mismatch '+rid)
    require(abs(float(row['p95'])-float(np.quantile(v,.95)))<1e-12,'Table P95 mismatch '+rid)
    require(int(row['n'])==len(v),'Table count mismatch '+rid)
    if row.get('worst_group_mean'):
        require(abs(float(row['worst_group_mean'])-max(a[f'{d}__{row["metric"]}'].mean() for d in P['distributions']))<1e-12,'Worst group mismatch '+rid)
    metric_checks+=1

coev=[];fixed=[]
for method in P['coev_methods']:
    record={'method':method}
    for suite in ['nominal_length','random32_length','es32_length']:
        seed_means=[];seed_p95=[];seed_worst=[]
        for s in P['seeds']:
            a=arrays[f'coev_{method}_s{s}'];groups=[a[f'{d}__{suite}'] for d in P['distributions']]
            seed_means.append(float(np.mean([g.mean() for g in groups])));seed_p95.append(float(np.quantile(np.concatenate(groups),.95)));seed_worst.append(float(max(g.mean() for g in groups)))
        record[suite+'_mean']=float(np.mean(seed_means));record[suite+'_p95_seedmean']=float(np.mean(seed_p95));record[suite+'_worst_group_seedmean']=float(np.mean(seed_worst));record[suite+'_seed_means']=json.dumps(seed_means)
    coev.append(record)
for method in P['learned_methods']:
    for price in P['test_lambdas']:
        for attacker in [method,'random','es']:
            values=[arrays[f'fixed_{method}_s{s}'][f'{d}__lambda{price:g}__{attacker}__gain'] for s in P['seeds'] for d in P['distributions']]
            fixed.append(dict(method=method,price=price,evaluated_attacker=attacker,mean_gain=float(np.mean([v.mean() for v in values]))))
csv_dump('training_budget.csv',budgets);csv_dump('runtime_and_calls.csv',runtime);csv_dump('coev_summary.csv',coev);csv_dump('fixed_summary.csv',fixed)

# Duplicate baseline evaluations across fixed tasks must agree; they have the same seed and router.
duplicate_checks=0
for s in P['seeds']:
    first=arrays[f'fixed_reinforce_s{s}']
    for method in P['learned_methods']:
        other=arrays[f'fixed_{method}_s{s}']
        for key,value in first.items():
            if '__random__' in key or '__es__' in key:
                require(np.array_equal(value,other[key]),'Duplicated baseline differs '+method+'/'+key)
                duplicate_checks+=1

launch=read(ROOT/'artifacts/operations/launch.json')
original_audit=read(ROOT/'artifacts/report/audit.json')
result=dict(status='PASS' if not ERRORS else 'FAIL',time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),errors=ERRORS,
    tasks=len(runs),source_freeze_files=len(freeze['files']),metric_rows_recomputed=metric_checks,duplicate_baseline_arrays=duplicate_checks,
    partials=partial,tmux_absent=tmux.returncode!=0,runs=runs,total_calls=call_totals,
    original_queue_wall_hours=(queue['time']-launch['time'])/3600,
    total_worker_wall_hours=sum(x['elapsed_seconds'] for x in runtime)/3600,
    total_peak_allocated_max_mib=max(x['peak_allocated_mib'] for x in runtime),
    original_verified_binding_count=sum(x['verified_hashes'] for x in original_audit['runs']),
    original_unique_training_inputs=original_audit['regenerated_unique_training_inputs'],
    no_new_gpu_inference=True,independent_nominal_length_recompute_in_original_report=True,
    limitations=['Attack candidate topology is not stored: candidate length scores have source/hash/ledger evidence, but this review does not re-infer them from models.',
      'Mean transport budget feasibility must be read from training_budget.csv; correct dual updates do not imply finite-training feasibility.',
      'Only degree30 and three seeds; unadjusted exploratory intervals; raw self-action length, no regret or optimum certificate.'])
dump('checkpoint_lifecycle_audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False,indent=2))
raise SystemExit(0 if not ERRORS else 1)

"""Independent, CPU-only closeout. Does not import worker.py or report.py.
Reads frozen artifacts; writes only this acceptance directory.
"""
import collections
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
GROUPS=('uniform','mix_gauss','edge_bias','grid_cluster')
HASHES={}; CHECKS=collections.Counter(); INITIAL_TIME=time.monotonic()
torch.set_num_threads(3)

def read(path): return json.loads(Path(path).read_text())
def write(name,value):
    tmp=OUT/(name+'.partial'); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(OUT/name)
def digest(path):
    path=Path(path).resolve()
    if path not in HASHES:
        with path.open('rb') as f: HASHES[path]=hashlib.file_digest(f,'sha256').hexdigest()
    return HASHES[path]
def require(condition,msg):
    if not condition: raise AssertionError(msg)
def close(a,b,msg,tol=1e-11):
    np.testing.assert_allclose(a,b,rtol=0,atol=tol,err_msg=msg)
def hashes(base,mapping):
    for path,h in mapping.items():
        target=(Path(base)/path).resolve(); require(digest(target)==h,'Hash mismatch: '+str(target))
        CHECKS['hash_bindings']+=1
def lines(path):
    with Path(path).open() as f:
        for s in f: yield json.loads(s)
def utc(s): return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def alive(record):
    if not record or record.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip(): return False
    try:
        st=Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')',1)[1].split()
        return st[0]!='Z' and st[19]==record['start_ticks']
    except FileNotFoundError: return False
def free_lock(path):
    # Open existing lock only, never create a missing receipt.
    with Path(path).open('r') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(f,fcntl.LOCK_UN)
    CHECKS['released_locks']+=1
def equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def model_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        require(bool(torch.isfinite(v).all()),'Nonfinite model state')
        h.update(k.encode()); h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def load_ck(path): return torch.load(path,map_location='cpu',weights_only=False)
def csv_write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def verify_tree_and_cost(x,t):
    """Batched union-find plus independent anchored coordinate ranges."""
    x=np.asarray(x,dtype=np.float64); t=np.asarray(t)
    b,n,_=x.shape
    require(t.shape==(b,2*(n-1)) and np.issubdtype(t.dtype,np.integer),'Invalid topology shape/type')
    require(t.min()>=0 and t.max()<n,'Topology endpoint outside input')
    edge=t.reshape(b,n-1,2); row=np.arange(b)
    parent=np.broadcast_to(np.arange(n),(b,n)).copy(); rank=np.zeros((b,n),dtype=np.int8)
    def root(v):
        while True:
            nxt=parent[row,v]
            if np.array_equal(nxt,v): return v
            v=nxt
    for j in range(n-1):
        u=root(edge[:,j,0]); v=root(edge[:,j,1]); require(np.all(u!=v),'Cycle or self-edge in model output')
        swap=rank[row,u]<rank[row,v]; u,v=np.where(swap,v,u),np.where(swap,u,v)
        ties=rank[row,u]==rank[row,v]; parent[row,v]=u; rank[row,u]+=ties
    value=np.zeros(b)
    # Each range contains the anchor and endpoints of all corresponding spans.
    for node in range(n):
        vy=np.where(edge[:,:,0]==node,x[row[:,None],edge[:,:,1],1],x[:,node,1,None])
        hx=np.where(edge[:,:,1]==node,x[row[:,None],edge[:,:,0],0],x[:,node,0,None])
        value+=np.ptp(np.column_stack((x[:,node,1],vy)),axis=1)
        value+=np.ptp(np.column_stack((x[:,node,0],hx)),axis=1)
    require(np.isfinite(value).all() and np.all(value>0),'Invalid recomputed own-action length')
    CHECKS['topologies_reduced']+=b
    return value

def ledger(path):
    pending={}; seen=set(); total=collections.Counter(); attempted=collections.Counter(); starts=[]
    for r in lines(path):
        if r['event']=='start':
            require(r['id'] not in seen,'Repeated call identifier'); seen.add(r['id']); pending[r['id']]=r
            attempted[r['phase']]+=r['rows']; starts.append(r['time'])
        else:
            require(r['event']=='complete' and r['id'] in pending,'Completion without call start')
            old=pending.pop(r['id']); require(old['phase']==r['phase'] and old['rows']==r['rows'],'Call receipt mismatch')
            total[r['phase']]+=r['rows']
    require(not pending,'Unfinished scoring call')
    require(dict(total)==dict(attempted),'Attempt accounting mismatch')
    return dict(total),min(starts),max(starts)

def bootstrap(b,c,which,rng,reps=5000):
    def reduce(group_means):
        return group_means.mean(axis=-1) if which=='mean' else group_means.max(axis=-1)
    bv=reduce(b.mean(-1)); cv=reduce(c.mean(-1)); estimates=np.empty(reps)
    # Independent implementation of the predeclared resampling law; same PRNG
    # draws allow exact numerical reconciliation rather than a new CI threshold.
    for i in range(reps):
        seeds=rng.integers(3,size=3); indices=rng.integers(b.shape[-1],size=(4,b.shape[-1]))
        bm=np.stack([b[seeds,g][:,indices[g]].mean(1) for g in range(4)],axis=1)
        cm=np.stack([c[seeds,g][:,indices[g]].mean(1) for g in range(4)],axis=1)
        estimates[i]=(reduce(bm)-reduce(cm)).mean()
    return dict(baseline_value=float(bv.mean()),candidate_value=float(cv.mean()),
        improvement=float(bv.mean()-cv.mean()),improvement_percent=float(100*(bv.mean()-cv.mean())/bv.mean()),
        improved_seeds=int(np.sum(cv<bv)),seed_improvement=(bv-cv).tolist(),
        ci95=np.quantile(estimates,[.025,.975]).tolist(),simultaneous_ci99=np.quantile(estimates,[.005,.995]).tolist())

def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Acceptance must be CPU-only')
    p=read(ROOT/'protocol.json'); ops=ROOT/'artifacts/operations'; q=read(ops/'queue.json')
    launch=read(ops/'launch.json'); exit_record=read(ops/'exit.json')
    require(exit_record['status']=='complete' and len(exit_record['completed'])==36,'Queue not completely closed')
    require(not q['running'] and not q['pending'] and not q['failures'],'Queue has unresolved tasks')
    hashes(ROOT,read(ops/'source_hashes.json')['hashes'])
    preflight=read(ROOT/'artifacts/tests/preflight.json'); hashes(ROOT,preflight['source_sha256']); hashes(ROOT,preflight['evidence_sha256'])
    require(preflight['status']=='PASS','Preflight not passed')
    require(not alive(launch['identity']),'Scheduler still active'); CHECKS['exited_process_identities']+=1
    free_lock(ops/'queue.lock')
    for f in ops.glob('gpu_*.lock'): free_lock(f)
    results=[read(f) for f in ops.glob('result_*.json')]
    exited={r['task_id']:r for r in results if r['status']=='exited'}
    deferred=[r for r in results if r['status']=='deferred']
    require(len(exited)==36 and len(exited)+len(deferred)==len(results),'Unexpected worker result')
    for r in exited.values():
        require(r['returncode']==0,'Nonzero worker exit')
        for ident in (r,r['child_identity']):
            require(not alive(ident),'Worker/child still active'); CHECKS['exited_process_identities']+=1
        require(len(r['cpu_affinity'])==3,'Unexpected CPU affinity count')
    specs=[read(f) for f in ops.glob('attempt_*.json')]
    for spec in specs:
        require(not alive(spec['worker_identity']),'Attempt wrapper still alive')
        hist=spec['admission_history']; require(len(hist)==3,'Missing three admission snapshots')
        require(all(v['utilization']<=35 and v['free_mib']>=8192 for v in hist),'Admission requirement violated')
    dataset_manifest=read(ROOT/'artifacts/data/manifest.json'); hashes(ROOT/'artifacts/data',dataset_manifest['artifacts'])
    val=dict(np.load(ROOT/'artifacts/data/validation.npz')); test=dict(np.load(ROOT/'artifacts/data/test.npz'))
    seen=set()
    for split,count in [(val,2048),(test,4096)]:
        for g in GROUPS:
            x=split[g]; require(x.shape==(count,30,2) and np.isfinite(x).all(),'Invalid frozen data')
            require(x.min()>=.1 and x.max()<=.9,'Input outside frozen window')
            for net in x:
                v=net.tobytes(); require(v not in seen,'Duplicate held-out net'); seen.add(v)
    lineage=read(ROOT/'artifacts/inputs/lineage.json')
    hashes(ROOT/'artifacts/inputs',{'lineage.json':dataset_manifest['lineage_sha256']})
    report_complete=read(ROOT/'artifacts/report/complete.json'); hashes(ROOT/'artifacts/report',report_complete['artifacts'])
    hashes(ROOT,report_complete['run_bindings'])
    runs={}; evaluations={}; data_hashes={}; weights=[]; run_rows=[]; model_values={}; metric_rows=[]; curve_rows=[]
    test_start=[]; train_end=[]; max_cost_error=0.; parent_checks=0
    print('Frozen hashes, data, admission, queue and lifecycle checks passed',flush=True)
    for arm in p['arms']:
        for seed in p['seeds']:
            key=f'{arm}_s{seed}'; folder=ROOT/'artifacts/runs'/key; evdir=ROOT/'artifacts/eval'/key
            c=read(folder/'complete.json'); e=read(evdir/'complete.json'); runs[key]=c; evaluations[key]=e
            for directory,comp,kind in [(folder,c,'training_complete'),(evdir,e,'evaluation_complete')]:
                require(comp['status']=='complete' and comp['acceptance']==kind,'Completion type mismatch')
                require(comp['identity']['device']=='cuda:0','Run not executed on CUDA')
                require(comp['binding']['arm']==arm and comp['binding']['seed']==seed,'Run identity mismatch')
                require(comp['binding']['protocol_sha256']==digest(ROOT/'protocol.json'),'Protocol mismatch')
                require(comp['binding']['data_manifest_sha256']==digest(ROOT/'artifacts/data/manifest.json'),'Data binding mismatch')
                hashes(ROOT,comp['binding']['code_sha256']); hashes(ROOT,comp['binding']['dependencies'])
                hashes(directory,comp['artifacts']); free_lock(directory/'run.lock')
                require(not alive(comp['identity']),'Completed worker still active')
                require(not read(directory/'run.json')['diagnostic'],'Diagnostic counted as formal run')
                require(comp['ledger']['oracle_calls']==0,'Reference call counter nonzero')
                require([x['event'] for x in lines(directory/'segments.jsonl')]==['start','exit'],'Incomplete process segment lifecycle')
            require(c['initial_actor_sha256']==lineage[str(seed)]['initial_actor_sha256'],'Initial actor hash differs')
            updates=p['targets'][arm]-p['start_steps'][arm]
            require(c['new_updates']==updates and c['step']==p['targets'][arm],'Update budget mismatch')
            log=list(lines(folder/'progress.jsonl')); upd=[r for r in log if r['type']=='update']; valid=[r for r in log if r['type']=='validation']
            require([r['step'] for r in upd]==list(range(p['start_steps'][arm]+1,p['targets'][arm]+1)),'Update sequence not contiguous')
            require([r['step'] for r in valid]==list(range(max(2048,p['start_steps'][arm]),p['targets'][arm]+1,512)),'Validation schedule changed')
            qprev=np.ones(4)/4; chain=''
            for row in upd:
                step=row['step']; expected_lr=1e-4 if step<=8192 else 5e-5
                require(row['learning_rate']==expected_lr,'Learning-rate schedule changed')
                require(np.isfinite([row[k] for k in ('loss','gradient_norm','sampled_mean','baseline_mean')]).all(),'Nonfinite training log')
                close(row['q_used'],qprev,'Group weights did not use previous batch')
                if arm in ('D','MG','SG'):
                    unnorm=qprev*np.exp(.01*np.asarray(row['group_risks']))
                    qnext=unnorm/unnorm.sum()
                else: qnext=qprev
                close(row['q_next'],qnext,'Group weight update incorrect')
                qprev=np.asarray(row['q_next'])
                close(row['q_entropy'],-sum(v*math.log(v) for v in qprev if v>0),'Group entropy mismatch')
                chain=hashlib.sha256((chain+row['data_sha256']).encode()).hexdigest()
                if arm!='U':
                    data_key=(seed,step)
                    require(data_hashes.setdefault(data_key,row['data_sha256'])==row['data_sha256'],'Mixed branches did not use paired inputs')
                CHECKS['training_updates_checked']+=1
            if arm in ('D','MG','SG'):
                weights.append(dict(arm=arm,seed=seed,uniform=qprev[0],mix_gauss=qprev[1],edge_bias=qprev[2],grid_cluster=qprev[3],entropy=upd[-1]['q_entropy']))
            parent_mean=valid[0]['metrics']['mean'] if arm in ('ME','MG','SG') else None
            eligible=[]
            for row in valid:
                means=[row['metrics']['groups'][g]['mean'] for g in GROUPS]
                close(np.mean(means),row['metrics']['mean'],'Validation average mismatch')
                close(max(means),row['metrics']['worst'],'Validation worst-group mismatch')
                allowed=parent_mean is None or row['metrics']['mean']<=parent_mean*1.01
                require(row['eligible']==allowed,'Wrong validation eligibility')
                if allowed: eligible.append(row)
                curve_rows.append(dict(arm=arm,seed=seed,step=row['step'],mean=row['metrics']['mean'],worst=row['metrics']['worst']))
            best=min(eligible,key=lambda v:(v['metrics']['worst'],v['step']))
            require(c['best_step']==best['step'],'Validation checkpoint selection mismatch')
            actual,first,last=ledger(folder/'actual_calls.jsonl'); eval_actual,tfirst,tlast=ledger(evdir/'actual_calls.jsonl')
            require(actual.get('test',0)==0,'Training read/scored test data')
            require(actual['sampled']==updates*64 and actual['rollout']==updates*64,'Sampled/rollout budget mismatch')
            require(actual['validation']==(len(valid)+1)*8192,'Validation budget mismatch')
            require(c['ledger']['sampled_rows']==actual['sampled'] and c['ledger']['backward_updates']==updates,'Committed budget differs from actual calls')
            require(eval_actual=={'test':len(c['models'])*16384},'Test call ledger differs from model matrix')
            train_end.append(utc(c['finished'])); test_start.append(utc(tfirst))
            latest=load_ck(folder/'latest.pt')
            require(latest['step']==c['step'] and latest['data_chain']==chain,'Committed checkpoint/log mismatch')
            require(latest['progress_bytes']==(folder/'progress.jsonl').stat().st_size,'Checkpoint log boundary mismatch')
            close(latest['logq'].exp().numpy(),qprev,'Final q checkpoint mismatch')
            if c['parent']:
                parent_path=Path(c['parent']['path']); require(digest(parent_path)==c['parent']['sha256'],'Changed parent')
                parent=load_ck(parent_path); start=load_ck(folder/f"step{p['start_steps'][arm]}.pt")
                for child_key,parent_key in [('actor','actor'),('optimizer','optimizer'),('rng','rng')]:
                    expected=(parent['networks']['actor'] if child_key=='actor' else parent['optimizers']['actor'] if child_key=='optimizer' else parent['rng']) if arm=='U' else parent[parent_key]
                    require(equal(start[child_key],expected),f'{key} did not exactly inherit {child_key}')
                require(model_hash(start['actor'])==c['parent']['actor_sha256'],'Parent actor digest mismatch')
                parent_checks+=1
                del parent,start
            expected_steps=[s for s in p['endpoints'] if max(2048,p['start_steps'][arm])<=s<=p['targets'][arm]]
            require(set(c['models'])=={f'step{s}' for s in expected_steps}|{'best'},'Missing fixed-budget endpoint')
            for label,model in c['models'].items():
                ck=load_ck(folder/model['file'])
                require(ck['step']==model['step'] and model_hash(ck['actor'])==model['actor_sha256'],'Saved model content mismatch')
                require(digest(folder/model['file'])==model['sha256'],'Model SHA mismatch')
                del ck
                receipt=read(evdir/f'{label}.json'); require(receipt['model_sha256']==model['sha256'],'Evaluated model mismatch')
                lengths=[]
                with np.load(evdir/f'{label}.npz') as z:
                    for g in GROUPS:
                        value=verify_tree_and_cost(test[g],z[g+'__topology'])
                        error=float(np.max(np.abs(value-z[g+'__length']))); max_cost_error=max(error,max_cost_error)
                        close(value,z[g+'__length'],'Test length mismatch',1e-12)
                        close(value.mean(),receipt['metrics']['groups'][g]['mean'],'Group mean mismatch')
                        close(np.quantile(value,.95),receipt['metrics']['groups'][g]['p95'],'Group P95 mismatch')
                        lengths.append(value)
                        metric_rows.append(dict(arm=arm,seed=seed,model=label,step=model['step'],group=g,mean=float(value.mean()),p95=float(np.quantile(value,.95))))
                model_values[(arm,label,seed)]=np.stack(lengths)
                CHECKS['evaluated_models']+=1
            with np.load(folder/'results.npz') as z:
                for g in GROUPS:
                    value=verify_tree_and_cost(val[g],z[g+'__topology'])
                    close(value,z[g+'__length'],'Selected validation length mismatch',1e-12)
                    close(value.mean(),best['metrics']['groups'][g]['mean'],'Selected validation does not reproduce chosen checkpoint')
            run_rows.append(dict(arm=arm,seed=seed,updates=updates,sampled=actual['sampled'],rollout=actual['rollout'],
                validation=actual['validation'],test=eval_actual['test'],best_step=c['best_step'],
                train_worker_seconds=exited[key]['elapsed_seconds'],eval_worker_seconds=exited['eval_'+key]['elapsed_seconds']))
            del latest,log,upd,valid
            print(key,'hashes, training, inheritance, model selection, all saved predictions PASS',flush=True)
    require(max(train_end)<min(test_start),'Test started before final training freeze')
    for key,e in evaluations.items():
        selection=read(ROOT/'artifacts/eval'/key/'selection.json')
        require(len(selection['training'])==18,'Incomplete frozen training matrix')
        for train_key,item in selection['training'].items():
            require(item['completion_sha256']==digest(ROOT/'artifacts/runs'/train_key/'complete.json'),'Selection lineage changed')
            require(item['models']==runs[train_key]['models'],'Frozen model list changed')
    require(sum(r['updates'] for r in run_rows)==135168,'Total training budget changed')
    # Rebuild initial random state and data probes on CPU, never infer any model.
    sys.path.insert(0,str(ROOT)); import common
    for seed in p['seeds']:
        common.seed_all(seed); model=common.G.build_actor(30,torch.device('cpu'))
        require(model_hash(model.state_dict())==lineage[str(seed)]['initial_actor_sha256'],'Random initialization cannot be reproduced')
        for arm in ('U','D'):
            path=ROOT/'artifacts/runs'/f'{arm}_s{seed}'/'progress.jsonl'
            by_step={r['step']:r['data_sha256'] for r in lines(path) if r['type']=='update'}
            for step in (2049,4096,6144,8192):
                x,_=common.training_batch(arm,seed,step)
                require(hashlib.sha256(x.tobytes()).hexdigest()==by_step[step],'Data probe failed replay')
                require(not any(net.tobytes() in seen for net in x),'Train/held-out overlap in probe')
                CHECKS['data_batches_regenerated']+=1
    def array(spec):
        arm,label=spec.split('/'); return np.stack([model_values[(arm,label,s)] for s in p['seeds']])
    saved=read(ROOT/'artifacts/report/statistics.json')['comparisons']; rng=np.random.default_rng(p['bootstrap_seed']); comparisons=[]
    for old in saved:
        a=array(old['baseline']); b=array(old['candidate']); new=bootstrap(a,b,old['metric'],rng,p['bootstrap_replicates'])
        for field in new: close(new[field],old[field],'Statistics disagreement: '+field,1e-10)
        mean_change=100*(b.mean()/a.mean()-1); close(mean_change,old['mean_degradation_percent'],'Mean degradation mismatch')
        supports=new['improvement_percent']>=.5 and new['improved_seeds']>=2 and new['simultaneous_ci99'][0]>0
        if old['name']=='robust_finetuning': supports=supports and mean_change<=1
        if old['primary']:
            require(supports==old['conclusion'].startswith('supports'),'Predeclared acceptance threshold mismatch')
        new.update(name=old['name'],baseline=old['baseline'],candidate=old['candidate'],metric=old['metric'],
            primary=old['primary'],mean_degradation_percent=float(mean_change),supports=bool(supports) if old['primary'] else None,
            direction=('candidate_worse_with_corrected_interval' if old['primary'] and new['simultaneous_ci99'][1]<0 else
                       'candidate_better_with_corrected_interval' if old['primary'] and new['simultaneous_ci99'][0]>0 else 'exploratory_or_uncertain'))
        comparisons.append(new); print('Independent 5000-replicate statistics:',new['name'],'PASS',flush=True)
    csv_write('metrics_independent.csv',metric_rows); csv_write('runtime_and_calls.csv',run_rows)
    csv_write('group_weights_final.csv',weights); csv_write('validation_curves_independent.csv',curve_rows)
    rankings=[]
    for arm in p['arms']:
        for label in ('best',f"step{p['targets'][arm]}"):
            x=array(arm+'/'+label); per_group=x.mean(-1)
            rankings.append(dict(arm=arm,model=label,mean=float(per_group.mean()),worst=float(per_group.max(1).mean()),
                mean_p95=float(np.mean([np.quantile(s,.95) for s in x])),steps=[runs[f'{arm}_s{s}']['models'][label]['step'] for s in p['seeds']]))
    csv_write('model_summary.csv',rankings)
    # Extra budget curves are descriptive and do not create new confirmatory tests.
    descriptive=[]
    for arm,label1,label2 in [('U','step4096','step8192'),('M','step4096','step8192'),('ME','step8192','step12288')]:
        a=array(arm+'/'+label1); b=array(arm+'/'+label2)
        descriptive.append(dict(arm=arm,baseline=label1,candidate=label2,
            mean_improvement_percent=100*(a.mean()-b.mean())/a.mean(),
            per_seed_mean_improvement_percent=(100*(a.mean((1,2))-b.mean((1,2)))/a.mean((1,2))).tolist()))
    write('statistics_independent.json',dict(comparisons=comparisons,model_summary=rankings,descriptive=descriptive))
    audit=dict(status='PASS_FOR_FROZEN_THREE_SEED_EXPLORATORY_PROTOCOL',time=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='CPU-only independent acceptance; no training, no new model inference, no GPU or oracle calls',
        training_tasks=18,evaluation_tasks=18,counts=dict(CHECKS),unique_files_hashed=len(HASHES),
        exact_parent_state_inheritances=parent_checks,total_new_updates=135168,total_sampled=sum(r['sampled'] for r in run_rows),
        total_rollout=sum(r['rollout'] for r in run_rows),total_validation_scores=sum(r['validation'] for r in run_rows),
        total_test_scores=sum(r['test'] for r in run_rows),worker_successful_exits=len(exited),admission_deferrals=len(deferred),
        sampled_replays=0,unfinished_calls=0,maximum_cost_reduction_error=max_cost_error,
        queue_wall_hours=(exit_record['at']-launch['at'])/3600,
        training_worker_hours=sum(r['train_worker_seconds'] for r in run_rows)/3600,
        evaluation_worker_hours=sum(r['eval_worker_seconds'] for r in run_rows)/3600,
        last_training_finished_utc=dt.datetime.fromtimestamp(max(train_end),dt.timezone.utc).isoformat(),
        first_test_scoring_utc=dt.datetime.fromtimestamp(min(test_start),dt.timezone.utc).isoformat(),
        verification_seconds=time.monotonic()-INITIAL_TIME,
        limits=['Three training seeds; bootstrap uncertainty is exploratory','Degree 30 synthetic fixed-window inputs only',
                'REST span length is not optimal regret','No same-test DAC21 equivalence claim','No learned Wasserstein adversary experiment'],
        numerical_differences=[])
    write('audit.json',audit)
    write('input_hashes.json',{str(path):h for path,h in HASHES.items()})
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        write('FAILED.json',dict(status='FAIL',traceback=traceback.format_exc()))
        raise

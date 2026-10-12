"""Independent CPU acceptance. Never import the training/reporting implementations."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv
import collections
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import datetime as dt
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
GROUPS=('uniform','mix_gauss','edge_bias','grid_cluster')
P=json.loads((ROOT/'protocol.json').read_text())
EXPECTED={}; REFERENCES=0; CACHE={}

def read(path): return json.loads(Path(path).read_text())
def save(name,value):
    p=OUT/name; tmp=p.with_suffix(p.suffix+'.partial'); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); os.replace(tmp,p)
def csvwrite(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bind(path,h):
    global REFERENCES
    path=Path(path).resolve(); REFERENCES+=1
    if path in EXPECTED:assert EXPECTED[path]==h,('inconsistent binding',path)
    EXPECTED[path]=h
def metric(lengths):
    means=np.array([lengths[g].mean() for g in GROUPS])
    return dict(mean=float(means.mean()),worst=float(means.max()),p95=float(np.quantile(np.concatenate(list(lengths.values())),.95)),
                groups={g:dict(mean=float(lengths[g].mean()),p95=float(np.quantile(lengths[g],.95))) for g in GROUPS})
def close(a,b):assert np.allclose(a,b,rtol=0,atol=1e-10),(a,b)
def alive(r):
    try:
        s=Path(f"/proc/{r['pid']}/stat").read_text().rsplit(')',1)[1].split()
        return s[0]!='Z' and s[19]==str(r['start_ticks']) and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id']
    except FileNotFoundError:return False
def lock_free(p):
    with p.open('r') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
def all_tasks():
    tasks={}
    for stage in ('screen','parents','final','evaluation'):
        q=read(ROOT/f'artifacts/queues/{stage}.json')
        for x in q['tasks']:
            path=ROOT/x['args'][2]; t=read(path)
            if t['id'] in tasks:assert tasks[t['id']]==t
            tasks[t['id']]=t
    return list(tasks.values())

def metadata(tasks):
    ids=[]; locks=[]; results=[]; admissions=[]
    pre=read(ROOT/'artifacts/tests/preflight.json');assert pre['status']=='PASS'
    bind(ROOT/'protocol.json',pre['protocol_sha256'])
    for key in ('source_sha256','evidence_sha256'):
        for p,h in pre[key].items():bind(ROOT/p,h)
    for stage in ('screen','parents','final','evaluation'):
        op=ROOT/f'artifacts/operations/{stage}'
        for p,h in read(op/'source_hashes.json')['hashes'].items():bind(ROOT/p,h)
        state=read(op/'queue.json'); ex=read(op/'exit.json');launch=read(op/'launch.json')
        assert ex['status']=='complete' and ex['launch_id']==launch['launch_id']
        assert not state['running'] and not state['pending'] and not state['failures']
        ids.append(launch['identity']); locks.append(op/'queue.lock')
        for attempt in op.glob('attempt_*.json'):
            a=read(attempt); ids.append(a['worker_identity'])
            for g in a['admission_history']:
                assert g['utilization']<=a['max_gpu_util'] and g['free_mib']>=a['admission_free_mib']
            admissions.append(a)
    for t in tasks:
        directory=ROOT/t['outdir'];c=read(directory/'complete.json');r=read(directory/'run.json');b=c['binding']
        assert c['status']=='complete' and r['task']==t
        assert c['acceptance']==('training_complete' if t['mode']=='train' else 'evaluation_complete')
        assert b['task_id']==t['id'] and b['seed']==t['seed'] and b['arm']==t['arm']
        assert c['identity']['device'].startswith('cuda') and not r['diagnostic']
        bind(ROOT/f"artifacts/tasks/{t['id']}.json",b['task_sha256']);bind(ROOT/'protocol.json',b['protocol_sha256'])
        bind(ROOT/'artifacts/data/manifest.json',b['data_manifest_sha256'])
        for key in ('code_sha256','dependencies'):
            for path,h in b[key].items():bind(path,h)
        for file,h in c['artifacts'].items():bind(directory/file,h)
        if t['parent']:bind(ROOT/t['parent']['path'],t['parent']['sha256'])
        ids.append(c['identity']);locks.append(directory/'run.lock')
    for path in (ROOT/'artifacts/operations').glob('result_*.json'):
        r=read(path); results.append(r)
        if r['status']=='exited':assert r['returncode']==0;ids.extend([r,r['child_identity']])
        else:assert r['status']=='deferred'
    good=[r for r in results if r['status']=='exited'];counts=collections.Counter(r['task_id'] for r in good)
    assert set(counts)=={t['id'] for t in tasks} and all(n==1 for n in counts.values())
    intervals=[]
    for r in good:
        for x in intervals:
            if max(x['started_at'],r['started_at'])<min(x['finished_at'],r['finished_at']):
                assert x['gpu']!=r['gpu'];assert not set(x['cpu_affinity'])&set(r['cpu_affinity'])
        intervals.append(r)
    manifest=read(ROOT/'artifacts/data/manifest.json')
    for file,h in manifest['artifacts'].items():bind(ROOT/'artifacts/data'/file,h)
    bind(ROOT/'artifacts/inputs/lineage.json',manifest['lineage_sha256'])
    for x in read(ROOT/'artifacts/inputs/lineage.json').values():bind(ROOT/x['path'],x['sha256']);bind(x['source'],x['sha256'])
    for name in ('promotion','selection'):
        obj=read(ROOT/f'artifacts/{name}.json')
        for path,h in obj['training_completions'].items():bind(ROOT/path,h)
    selection=read(ROOT/'artifacts/selection.json')
    assert len(selection['training_completions'])==58
    for x in selection['models'].values():
        for m in x['models'].values():bind(ROOT/m['path'],m['sha256'])
    for x in tasks:
        if x['mode']=='eval':
            bind(ROOT/'artifacts/selection.json',x['selection_sha256'])
            assert read(ROOT/x['outdir']/'selection.json')['global_sha256']==x['selection_sha256']
    rc=read(ROOT/'artifacts/report/complete.json')
    for name,h in rc['artifacts'].items():bind(ROOT/'artifacts/report'/name,h)
    life=read(ROOT/'artifacts/lifecycle.json');assert life['status']=='complete';ids.append(life['identity'])
    bind(ROOT/'artifacts/report/complete.json',life['report_complete_sha256'])
    locks.extend((ROOT/'artifacts/operations').glob('gpu_*.lock'));locks.append(ROOT/'artifacts/operations/pipeline.lock')
    identities={(r['pid'],r['start_ticks'],r['boot_id']):r for r in ids}
    assert not any(alive(r) for r in identities.values())
    for p in locks:lock_free(p)
    print('Hash verification',len(EXPECTED),'files',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for p,h in zip(EXPECTED,pool.map(sha,EXPECTED)):assert h==EXPECTED[p],('hash mismatch',p)
    save('input_hashes.json',{str(p):h for p,h in EXPECTED.items()})
    start=min(r['started_at'] for r in good);finish=dt.datetime.fromisoformat(life['time'].replace('Z','+00:00')).timestamp()
    totals=dict(hash_references=REFERENCES,unique_hashed_files=len(EXPECTED),released_locks=len(locks),exited_process_identities=len(identities),
        successful_workers=len(good),admission_deferrals=len(results)-len(good),gpu_cpu_overlap=0,
        total_hours=(finish-start)/3600,start_utc=dt.datetime.fromtimestamp(start,dt.timezone.utc).isoformat(),end_utc=life['time'],
        worker_hours=sum(r['elapsed_seconds'] for r in good)/3600,training_worker_hours=sum(r['elapsed_seconds'] for r in good if not r['task_id'].startswith('eval_'))/3600)
    save('metadata.json',totals);return totals

def calls(path,c):
    pending={};totals=collections.Counter();first=None
    for line in path.open():
        r=json.loads(line)
        if r['event']=='start':assert r['id'] not in pending;pending[r['id']]=r
        else:
            s=pending.pop(r['id']);assert (s['phase'],s['rows'],s['step'])==(r['phase'],r['rows'],r['step'])
            totals[r['phase']]+=r['rows']
            if s['phase']=='test':first=min(first or s['time'],s['time'])
    assert not pending
    for k in ('sampled','rollout','validation','calibration','test'):assert totals[k]==c['ledger'][k+'_rows']
    assert c['ledger']['oracle_calls']==0
    return dict(totals),first

def train_log(t):
    d=ROOT/t['outdir'];c=read(d/'complete.json');budget=t['target']-t['start'];cfg=t['config']
    assert c['new_updates']==budget and c['step']==t['target'] and c['ledger']['backward_updates']==budget
    totals,first=calls(d/'actual_calls.jsonl',c)
    assert totals.get('test',0)==0 and totals['sampled']==totals['rollout']==budget*cfg['batch']
    import torch
    ck=torch.load(d/f"step{t['start']}.pt",map_location='cpu',weights_only=False)
    logr=ck['logr'].numpy().copy();anchor=ck['anchors'].numpy().copy();prev_var=ck['adv_variance']
    step=t['start']; chain='';vals=[]; hashes=[];parent_mean=None
    for line in (d/'progress.jsonl').open():
        r=json.loads(line)
        if r['type']=='validation':
            parent_mean=r['metrics']['mean'] if parent_mean is None else parent_mean
            eligible=r['metrics']['mean']<=parent_mean*1.01
            assert r['eligible']==eligible
            vals.append(r);continue
        step+=1;assert r['step']==step
        lr=cfg['lr']
        if cfg['schedule']=='original':lr=1e-4 if step<=8192 else 5e-5
        elif cfg['schedule']=='cosine':lr=5e-6+(5e-5-5e-6)*.5*(1+math.cos(math.pi*min(1,max(0,(step-12288)/(49152-12288)))))
        close(r['learning_rate'],lr)
        q=(1-cfg['uniform_mix'])*np.exp(logr)+cfg['uniform_mix']/4 if cfg['objective']=='group' else np.full(4,.25)
        close(q,r['q_used']);risks=np.asarray(r['group_risks']);obj=risks/anchor if cfg['normalize_risk'] else risks
        close(obj,r['objective_risks'])
        if cfg['objective']=='group':
            logr+=cfg['eta']*obj;maximum=logr.max();logr-=maximum+np.log(np.exp(logr-maximum).sum())
        close((1-cfg['uniform_mix'])*np.exp(logr)+cfg['uniform_mix']/4,r['q_next'])
        last_q=r['q_next']
        close(r['advantage_scale_used'],max(math.sqrt(prev_var),.05) if cfg['advantage_ema'] else 1)
        prev_var=r['advantage_variance_next'];assert np.isfinite(prev_var) and prev_var>=0
        chunks=cfg['batch']//64;first_step=step if chunks==1 else 12288+(step-12289)*chunks+1
        assert r['data_info']['chunks']==chunks and r['data_info']['first_base_step']==first_step
        hashes.append((step,r['data_info']['unaugmented_sha256']))
        chain=hashlib.sha256((chain+r['data_sha256']).encode()).hexdigest()
        assert all(math.isfinite(r[k]) for k in ('loss','sampled_mean','baseline_mean','gradient_norm'))
    assert step==t['target']
    best=min((v for v in vals if v['eligible']),key=lambda v:(v['metrics']['worst'],v['step']))
    assert best['step']==c['best_step']
    expected=[t['start']]+[i for i in range(t['start']+1,t['target']+1) if i%1024==0 or i==t['target']]
    assert [v['step'] for v in vals]==expected
    end=torch.load(d/'latest.pt',map_location='cpu',weights_only=False)
    assert end['step']==step and end['data_chain']==chain
    close(end['logr'].numpy(),logr);close(end['adv_variance'],prev_var)
    rows=[dict(id=t['id'],arm=t['arm'],seed=t['seed'],step=v['step'],mean=v['metrics']['mean'],worst=v['metrics']['worst']) for v in vals]
    return dict(id=t['id'],new_updates=budget,batch=cfg['batch'],sampled=totals['sampled'],validation=totals['validation'],
        calibration=totals.get('calibration',0),best_step=best['step'],finished=c['finished'],final_q=last_q,parent_anchors=anchor.tolist()),rows,hashes

def get_data(split):
    if split not in CACHE:
        with np.load(ROOT/'artifacts/data'/f'{split}.npz') as z:CACHE[split]={g:z[g] for g in GROUPS}
    return CACHE[split]

def independently_score(x,topology):
    b,n,_=x.shape;assert topology.shape==(b,2*(n-1));assert topology.min()>=0 and topology.max()<n
    assert np.issubdtype(topology.dtype,np.integer)
    result=[]
    for begin in range(0,b,256):
        pins=x[begin:begin+256].astype(np.float64);e=topology[begin:begin+256].reshape(len(pins),n-1,2)
        for edges in e.tolist():
            parent=list(range(n))
            for u,v in edges:
                while parent[u]!=u:u=parent[u]
                while parent[v]!=v:v=parent[v]
                assert u!=v,'cycle or self-edge';parent[u]=v
        ri=np.arange(len(pins))[:,None];a=np.arange(n)[None,:,None]
        vy=pins[ri,e[:,:,1],1][:,None,:]; hx=pins[ri,e[:,:,0],0][:,None,:]
        vertical=np.where(e[:,None,:,0]==a,vy,pins[:,:,1,None])
        horizontal=np.where(e[:,None,:,1]==a,hx,pins[:,:,0,None])
        span=(np.maximum(vertical.max(-1),pins[:,:,1])-np.minimum(vertical.min(-1),pins[:,:,1])+
              np.maximum(horizontal.max(-1),pins[:,:,0])-np.minimum(horizontal.min(-1),pins[:,:,0])).sum(-1)
        result.append(span)
    return np.concatenate(result)

def numerical(t):
    d=ROOT/t['outdir'];c=read(d/'complete.json');rows=[];maxerr=0;count=0
    if t['mode']=='train':items=[('selected_validation','results.npz',c['selected_validation'])];split='validation'
    else:
        calls(d/'actual_calls.jsonl',c)
        items=[(label,label+'.npz',read(d/(label+'.json'))['metrics']) for label in t['models']]
        split='test' if t['degree']==30 else f"test_degree{t['degree']}"
    data=get_data(split)
    for label,file,ref in items:
        lengths={}
        with np.load(d/file) as z:
            for g in GROUPS:
                l=independently_score(data[g],z[g+'__topology']);stored=z[g+'__length'];close(l,stored)
                maxerr=max(maxerr,float(np.max(np.abs(l-stored))));count+=len(l);lengths[g]=l
        m=metric(lengths)
        for key in ('mean','worst','p95'):close(m[key],ref[key])
        for g in GROUPS:
            for key in ('mean','p95'):close(m['groups'][g][key],ref['groups'][g][key])
        if t['mode']=='eval':rows.append(dict(run=t['id'],stage=t['stage'],arm=t['arm'],seed=t['seed'],degree=t['degree'],
            checkpoint=label,step=t['models'][label]['step'],**{k:m[k] for k in ('mean','worst','p95')},**{g:m['groups'][g]['mean'] for g in GROUPS}))
    return dict(id=t['id'],mode=t['mode'],count=count,max_abs_error=maxerr),rows

def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def checkpoint_check(t):
    import torch
    d=ROOT/t['outdir'];r=read(d/'run.json');c=read(d/'complete.json')
    initial=torch.load(d/f"step{t['start']}.pt",map_location='cpu',weights_only=False)
    count=0
    for label,m in c['models'].items():
        ck=torch.load(d/m['file'],map_location='cpu',weights_only=False);h=hashlib.sha256()
        for k,v in sorted(ck['actor'].items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
        assert h.hexdigest()==m['actor_sha256'];count+=1
    if t['parent']:
        parent=torch.load(ROOT/t['parent']['path'],map_location='cpu',weights_only=False)
        for key in ('actor','rng'):assert equal(parent[key],initial[key]),(t['id'],key)
        assert equal(parent['optimizer']['state'],initial['optimizer']['state'])
        for p,q in zip(parent['optimizer']['param_groups'],initial['optimizer']['param_groups']):
            assert equal({k:v for k,v in p.items() if k!='weight_decay'},{k:v for k,v in q.items() if k!='weight_decay'})
            assert q['weight_decay']==t['config']['weight_decay']
        if t['inherit_training']:
            for key in ('logr','anchors','adv_variance'):assert equal(parent[key],initial[key])
    if t['config']['freeze_bn']:
        end=torch.load(d/'latest.pt',map_location='cpu',weights_only=False)
        for k in initial['actor']:
            if k.endswith(('running_mean','running_var','num_batches_tracked')):assert equal(initial['actor'][k],end['actor'][k])
    return dict(id=t['id'],models_verified=count,parent_inheritance_verified=bool(t['parent']),parameter_count=r['parameter_count'])

def selection_and_data(train,logs):
    rec={};gates=[]
    for t in train:
        if t['stage']=='screen':rec.setdefault(t['arm'],[]).append(read(ROOT/t['outdir']/'complete.json')['endpoint_validation'])
    promotion=read(ROOT/'artifacts/promotion.json');assert promotion['endpoint_validation']==rec
    winners={}
    for key,other in [('mean','worst'),('worst','mean')]:
        valid=[];base=rec['E50'];bm=np.mean([x[key] for x in base]);bo=np.mean([x[other] for x in base])
        for a in P['arms']:
            if a in ('E50','B256'):continue
            cm=np.mean([x[key] for x in rec[a]]);co=np.mean([x[other] for x in rec[a]])
            n=sum(x[key]<y[key] for x,y in zip(rec[a],base));gain=100*(bm-cm)/bm
            eligible=gain>=.2 and n>=2 and co<=bo*1.01
            gates.append(dict(family='variant',arm=a,metric=key,gain_percent=gain,improved_seeds=n,other_deterioration_percent=100*(co-bo)/bo,eligible=eligible))
            if eligible:valid.append((cm,a))
        winners[key]=min(valid)[1] if valid else None
    assert winners==promotion['winners']
    arch=[]
    for a in ('WIDE','DEEP'):
        b=rec['BASE'];c=rec[a];bm=np.mean([x['worst'] for x in b]);cm=np.mean([x['worst'] for x in c]);gain=100*(bm-cm)/bm
        n=sum(x['worst']<y['worst'] for x,y in zip(c,b));ratio=np.mean([x['mean'] for x in c])/np.mean([x['mean'] for x in b])
        ok=gain>=.2 and n>=2 and ratio<=1.01
        gates.append(dict(family='architecture',arm=a,metric='worst',gain_percent=gain,improved_seeds=n,other_deterioration_percent=100*(ratio-1),eligible=ok))
        if ok:arch.append((cm,a))
    assert (min(arch)[1] if arch else None)==promotion['architecture']
    csvwrite('promotion_independent.csv',gates)
    streams={};paired=0
    for t,(_,_,hh) in zip(train,logs):
        if t['config']['batch']!=64:continue
        for step,h in hh:
            key=t['seed'],step
            if key in streams:assert streams[key]==h;paired+=1
            else:streams[key]=h
    seen=set();nets=0
    for split in ('calibration','validation','test','test_degree20','test_degree40'):
        for x in get_data(split).values():
            for row in x:
                h=hashlib.sha256(row.tobytes()).digest();assert h not in seen;seen.add(h);nets+=1
    old=ROOT.parent/'rl_budget_scaling_20260921/artifacts/data'
    for split in ('validation','test'):
        with np.load(old/f'{split}.npz') as z:
            for g in z.files:
                for row in z[g]:assert hashlib.sha256(row.tobytes()).digest() not in seen
    return dict(paired_batch64_log_entries=paired,exact_new_split_nets=nets,previous_split_overlap=0,selection=winners,architecture=promotion['architecture'])

def main():
    started=time.time();tasks=all_tasks();train=[t for t in tasks if t['mode']=='train'];ev=[t for t in tasks if t['mode']=='eval']
    assert len(train)==58 and len(ev)==71
    meta=metadata(tasks);print('Metadata PASS',flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        logs=[]
        for i,result in enumerate(pool.map(train_log,train)):
            logs.append(result)
            if i%8==7:print('Training log audit',i+1,'/58',flush=True)
        ck=list(pool.map(checkpoint_check,train));print('Checkpoint inheritance PASS',flush=True)
        numeric=[];metrics=[]
        for i,(n,rows) in enumerate(pool.map(numerical,tasks)):
            numeric.append(n);metrics.extend(rows)
            if i%16==15:print('Numerical audit',i+1,'/129',flush=True)
    extra=selection_and_data(train,logs)
    csvwrite('training_audit.csv',[x[0] for x in logs]);csvwrite('validation_curves_independent.csv',sum((x[1] for x in logs),[]))
    csvwrite('checkpoint_audit.csv',ck);csvwrite('numerical_audit.csv',numeric);csvwrite('metrics_independent.csv',metrics)
    latest=max(x[0]['finished'] for x in logs);first=min(calls(ROOT/t['outdir']/'actual_calls.jsonl',read(ROOT/t['outdir']/'complete.json'))[1] for t in ev)
    assert latest<first
    original=list(csv.DictReader((ROOT/'artifacts/report/models.csv').open()));lookup={(r['run'],r['checkpoint']):r for r in original}
    for r in metrics:
        ref=lookup[(r['run'],r['checkpoint'])]
        for key in ('mean','worst','p95',*GROUPS):close(r[key],float(ref[key]))
    wd=[]
    for seed in P['screening_seeds']:
        a=read(ROOT/f'artifacts/runs/screen_E50_s{seed}/complete.json');b=read(ROOT/f'artifacts/runs/screen_WD_s{seed}/complete.json')
        same=all(a['models'][k]['actor_sha256']==b['models'][k]['actor_sha256'] for k in a['models'])
        wd.append(dict(seed=seed,all_saved_actor_hashes_identical=same))
    assert all(x['all_saved_actor_hashes_identical'] for x in wd)
    stats=dict(status='PASS_CORE_WITH_DEGENERATE_WEIGHT_DECAY_ABLATION',scope='Independent CPU audit; no new GPU inference/training',
        metadata=meta,training_tasks=len(train),evaluation_tasks=len(ev),audited_updates=sum(x[0]['new_updates'] for x in logs),
        sampled=sum(x[0]['sampled'] for x in logs),calibration_rows=sum(x[0]['calibration'] for x in logs),validation_rows=sum(x[0]['validation'] for x in logs),
        independent_test_lengths=sum(x['count'] for x in numeric if x['mode']=='eval'),
        independent_selected_validation_lengths=sum(x['count'] for x in numeric if x['mode']=='train'),max_length_error=max(x['max_abs_error'] for x in numeric),
        exact_parent_inheritances=sum(x['parent_inheritance_verified'] for x in ck),verified_saved_actors=sum(x['models_verified'] for x in ck),
        last_training_finished=latest,first_test_call=first,weight_decay_degenerate=wd,
        float32_decay_factor=float(np.float32(1-5e-5*1e-4)),oracle_calls=0,**extra,elapsed_audit_seconds=time.time()-started)
    save('audit.json',stats);print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()

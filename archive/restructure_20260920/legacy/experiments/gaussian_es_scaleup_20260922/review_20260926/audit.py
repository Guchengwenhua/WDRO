"""Independent read-only acceptance. Writes only to this review directory.

Does not call the original finalizer, selector, or statistical functions.
Reuses the frozen data specification and geometric primitive implementations.
"""
from pathlib import Path
import collections
import csv
import datetime
import fcntl
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path('/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922')
ART=ROOT/'artifacts';OUT=ART/'review_20260926'
sys.path.insert(0,str(ROOT))
from common import data,pin_set_sha256,costs,proc_ticks
from adv_v3.search import _check_pins

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(name,v):
    p=OUT/name;q=p.with_name(p.name+'.partial')
    q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False));q.replace(p)
def csvwrite(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def progress(stage,**kw):
    v=dict(at=time.time(),stage=stage,**kw);write('progress.json',v);print(json.dumps(v),flush=True)
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
def released(p):
    with Path(p).open('r') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)

def main():
    started=time.time();torch.set_num_threads(1)
    P=read(ROOT/'protocol.json');D=P['distributions'];status=read(ART/'status.json')
    completion=read(ART/'completion.json');freeze=read(ART/'manifests/source_freeze.json')
    source_id=digest(freeze);records={};bindings={}
    def bind(path,expected):
        p=str(Path(path).resolve())
        if p in bindings:assert bindings[p]==expected,('inconsistent binding',p)
        else:assert sha(p)==expected,('hash mismatch',p);bindings[p]=expected
    def walk(v):
        if isinstance(v,dict):
            if 'sha256' in v and ('path' in v or 'logical_path' in v):bind(v.get('path',v.get('logical_path')),v['sha256'])
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    for p,h in freeze['sources'].items():bind(p,h)
    walk(completion);walk(freeze)
    for p in (ART/'manifests').glob('*.json'):walk(read(p))
    assert P==freeze['protocol']
    assert status['phase']=='COMPLETE' and not status['failed'] and len(status['tasks'])==97
    assert all(t['status']=='complete' for t in status['tasks'])
    assert proc_ticks(status['pid'])!=status['start_ticks']
    released(ART/'pipeline.lock')
    for row in status['tasks']:
        ident=row['id'];out=ART/'runs'/ident;r=read(out/'complete.json');t=r['task']
        assert ident==t['id'];assert read(ART/'tasks'/f'{ident}.json')==t
        launch=read(out/'launch.json');exit_record=read(out/'exit.json');hb=read(out/'heartbeat.json')
        assert launch['source_freeze']==source_id
        assert proc_ticks(launch['pid'])!=launch['start_ticks']
        assert exit_record['status']=='complete' and exit_record['pid']==launch['pid'] and exit_record['start_ticks']==launch['start_ticks']
        assert hb['pid']==launch['pid'] and hb['start_ticks']==launch['start_ticks'] and exit_record['at']>=hb['at']-1
        released(out/'run.lock');walk(r)
        assert read(out/'oracle/oracle_ledger.json')==r['ledger']
        for k,v in r['ledger'].items():
            if k.endswith('_requested') and k[:-10]+'_completed' in r['ledger']:
                assert v==r['ledger'][k[:-10]+'_completed'],(ident,k)
        assert r['ledger']['geosteiner_misses_started']==r['ledger']['geosteiner_misses_completed']
        assert not list(out.glob('failure_*.json'))
        records[ident]=(t,r,out,launch,exit_record)
    kinds=collections.Counter(t['kind'] for t,_,_,_,_ in records.values())
    assert kinds=={'train':49,'screen':15,'eval':33}
    progress('sources_and_lifecycle_pass',source_files=len(freeze['sources']),unique_bindings=len(bindings),released_locks=98)

    # Independently recover selected search parameters and outer hyperparameters.
    screen_rows=[]
    for ident,(t,r,out,_,_) in records.items():
        if t['kind']!='screen':continue
        gains=[]
        for dist in D:
            with np.load(out/f'{dist}.npz') as f:
                assert len(f['centers'])==64
                np.testing.assert_array_equal(f['centers'],data(30,64,dist,t.get('split','search-validation')))
                _check_pins(f['pins'][:,None],30)
                close(f['regret'],f['length']/f['lstar']-1)
                actual=costs(f['centers'],f['pins'][:,None])[:,0]
                close(f['cost'],actual)
                price=freeze['dual_by_degree']['30'][0]
                close(f['score'],f['regret']-price*actual)
                assert np.all(f['score']>=f['original_score']-1e-12)
                gain=float(np.mean(f['score']-f['original_score']));gains.append(gain)
                close(gain,r['results'][dist]['gain'])
                screen_rows.append(dict(task=ident,method=t['method'],budget=t['budget'],distribution=dist,gain=gain,seconds=float(f['seconds'])))
        close(np.mean(gains),r['mean_gain'])
        assert r['ledger']['actor_completed']==r['ledger']['reference_requests']==256*t['budget']
    chosen=min([v for k,v in records.items() if k.startswith('screen_es_')],key=lambda v:(-v[1]['mean_gain'],v[0]['id']))[0]
    selection=read(ART/'manifests/search_selection.json')
    assert selection['task']==chosen['id'] and selection['mu']==chosen['mu'] and selection['search_step']==chosen['search_step']
    csvwrite('search_cells.csv',screen_rows)

    train_rows=[];center_streams={};init_bindings={};budget_sums=collections.Counter();audit_centers=0;train_replays=0
    for ident,(t,r,out,launch,ex) in records.items():
        if t['kind']!='train':continue
        pre=t['phase']=='pre';pilot=t['phase']=='pilot';d=t['degree'];seed=t['seed'];b=t['batch'];u=t['updates']
        assert b==(P['pre_batch'] if pre else P['post_batch'])
        assert u==(P['pilot_updates'] if pilot else P['pre_updates' if pre else 'post_updates'][str(d)])
        assert seed==901 if pilot else seed in (P['seeds'][str(d)]+([901] if pre and d==30 else []))
        identity=read(out/'identity.json');assert identity['task']==t and identity['source_freeze']==source_id
        assert digest(identity)==r['identity_hash'];walk(identity)
        if t.get('parent'):
            parent=read(Path(t['parent'])/'complete.json');assert identity['initialization']==parent['best']
            if not pilot:assert r['initial_validation']==parent['best_validation']
        if t['phase']=='post':
            key=(d,seed);value=identity['initialization']['sha256']
            assert key not in init_bindings or init_bindings[key]==value;init_bindings[key]=value
        vals={int(p.stem.split('_')[1]):read(p) for p in out.glob('validation_*.json')}
        assert set(vals)==set(range(0,u+1,P['validation_every']))
        initial=vals[0];assert r['initial_validation']==initial
        def criterion(v):
            if pre:return v['uniform']['mean']
            if v['uniform']['mean']>initial['uniform']['mean']+P['nominal_tolerance']:return float('inf')
            return max(v[g]['mean'] for g in D[1:])
        best=min(vals,key=lambda step:(criterion(vals[step]),step))
        assert best==r['best_step'] and vals[best]==r['best_validation']
        for name,target in [('latest',u),('best',best)]:
            cp=torch.load(out/f'{name}.pt',map_location='cpu',weights_only=False)
            assert cp['step']==target and cp['task_hash']==r['identity_hash']
            assert cp['best_validation']==r['best_validation'] and cp['initial_validation']==initial
            assert (target==0 or cp['optimizer_state_dict']['state']) and cp['rng']['torch'] is not None
            for model in ['actor_state_dict','critic_state_dict']:
                assert all(torch.isfinite(x).all() for x in cp[model].values())
            del cp
        rows=[json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines()]
        assert [x['step'] for x in rows]==list(range(1,u+1)),('missing or duplicate steps',ident)
        stream=tuple(x['center_hash'] for x in rows)
        if t['phase']=='post':
            key=(d,seed)
            assert key not in center_streams or center_streams[key]==stream;center_streams[key]=stream
        running_price=freeze['dual_by_degree'][str(d)][0];rho=freeze['dual_by_degree'][str(d)][1]
        for x in rows:
            assert x['search_calls']==(0 if t['method']=='erm' else b*64)
            close(x['price_before'],running_price);close(x['rho'],rho);close(x['budget_gap'],x['cost']-rho)
            expected_price=running_price if t['method']=='erm' else max(0,running_price+P['lambda_lr']*(x['cost']-rho))
            close(x['price'],expected_price);running_price=x['price']
            assert np.isfinite([x['cost'],x['gain'],x['seconds']]).all() and x['gain']>=-1e-12
            for key in ('sampled_actor_requested','sampled_actor_completed','sampled_rest_requested','sampled_rest_completed','sampled_reference_reuses'):
                assert x['ledger_delta'][key]==b
            for key in ('actor_requested','actor_completed','rest_requested','rest_completed'):
                assert x['ledger_delta'].get(key,0)==x['search_calls']
            assert x['ledger_delta']['reference_requests']==(b if t['method']=='erm' else b*64+round(t.get('nominal_fraction',0)*b))
        for step in sorted({1,u//2,u}):
            phase='train-pre' if pre else ('train-pilot' if pilot else 'train-post')
            x=data(d,b,'uniform',f'{phase}:u{step}',seed)
            assert ah(x)==rows[step-1]['center_hash'];train_replays+=1
        for path in sorted((out/'search_audit').glob('*.npz')):
            with np.load(path) as f:
                step=int(path.stem[1:]);_check_pins(f['support'],d)
                scores=f['regrets']-float(f['price'])*costs(f['centers'],f['support'])
                indices=scores.argmax(axis=1)
                close(f['score'],scores.max(axis=1));close(f['original'],scores[:,0])
                np.testing.assert_array_equal(f['selected'],f['support'][np.arange(2),indices])
                close(float(f['price']),rows[step-1]['price_before']);audit_centers+=2
        val_calls=len(vals)*P['validation_count']*4
        expected_greedy=val_calls+(0 if t['method']=='erm' else u*b*64)
        expected_ref=expected_greedy+u*(b if t['method']=='erm' else round(t.get('nominal_fraction',0)*b))
        assert r['ledger']['actor_completed']==expected_greedy and r['ledger']['reference_requests']==expected_ref
        assert r['ledger']['sampled_actor_completed']==u*b
        for k,v in r['ledger'].items():budget_sums[k]+=v
        train_rows.append(dict(task=ident,phase=t['phase'],degree=d,seed=seed,method=t['method'],updates=u,batch=b,best_step=best,
            train_hours=r['train_seconds']/3600,wall_hours=(ex['at']-launch['at'])/3600,
            rho=rho,mean_cost=float(np.mean([x['cost'] for x in rows])),last_quarter_cost=float(np.mean([x['cost'] for x in rows[-u//4:]])),
            last_price=running_price,zero_price_fraction=float(np.mean([x['price']==0 for x in rows])),
            initial_uniform=initial['uniform']['mean'],best_uniform=vals[best]['uniform']['mean'],last_uniform=vals[u]['uniform']['mean'],
            best_ood_worst=max(vals[best][g]['mean'] for g in D[1:]),last_ood_worst=max(vals[u][g]['mean'] for g in D[1:])))
        progress('training_audit',done=len(train_rows),total=49)
    csvwrite('training_audit.csv',train_rows)
    pilots=[(t,r) for t,r,_,_,_ in records.values() if t.get('phase')=='pilot']
    eligible=[(t,r) for t,r in pilots if r['best_step']>0 and r['best_validation']['uniform']['mean']<=r['initial_validation']['uniform']['mean']+.002]
    picked=min(eligible,key=lambda v:(max(v[1]['best_validation'][g]['mean'] for g in D[1:]),v[0]['id']))[0]
    outer=read(ART/'manifests/outer_selection.json')
    assert outer['task']==picked['id'] and outer['lr']==picked['lr'] and outer['nominal_fraction']==picked['nominal_fraction']
    assert all(t['seed']==901 for t,r in pilots)
    for t,_,_,_,_ in records.values():
        if t.get('phase')=='post':
            assert t['lr']==outer['lr'] and t['nominal_fraction']==outer['nominal_fraction']
            assert t['mu']==selection['mu'] and t['search_step']==selection['search_step']

    # All declared tests are after model freeze, which is after all post-training exits.
    model_freeze=ART/'manifests/model_freeze.json';models=read(model_freeze)['models']
    assert len(models)==33
    post_end=max((out/'complete.json').stat().st_mtime for t,_,out,_,_ in records.values() if t.get('phase')=='post')
    test_start=min(la['at'] for t,_,_,la,_ in records.values() if t['kind']=='eval')
    assert post_end<=model_freeze.stat().st_mtime<=test_start
    for m in models:assert records[m['id']][1]['best']==m['best']

    values={};metric_rows=[];reference_centers={};row_count=0;shard_count=0;replay_specs=[]
    original_metrics={tuple(str(r[k]) for k in ('degree','seed','method','suite','distribution')):r for r in csv.DictReader((ART/'metrics.csv').open())}
    for ident,(t,r,out,_,_) in records.items():
        if t['kind']!='eval':continue
        assert t['model']==records[ident[5:]][1]['best']
        for kind,count,chunk in [('nominal',4096,128),('pressure',128,32)]:
            for dist in D:
                chunks=[];centers=[];files=sorted(out.glob(f'{dist}_{kind}_*.npz'))
                assert [int(p.stem.rsplit('_',1)[1]) for p in files]==list(range(0,count,chunk))
                for path in files:
                    with np.load(path) as f:
                        assert str(f['model_sha256'])==t['model']['sha256']
                        assert f['pins'].shape==(chunk,t['degree'],2)
                        _check_pins(f['pins'][:,None],t['degree'])
                        assert np.isfinite(f['regret']).all() and (f['regret']>=-1e-7).all() and (f['lstar']>0).all()
                        close(f['regret'],f['length']/f['lstar']-1)
                        if kind=='nominal':np.testing.assert_array_equal(f['pins'],f['centers'])
                        else:
                            close(f['cost'],costs(f['centers'],f['pins'][:,None])[:,0]);close(f['regret'],f['score'])
                            assert np.all(f['score']>=f['original_score']-1e-12)
                        chunks.append(f['regret'].copy());centers.append(f['centers'].copy());row_count+=chunk;shard_count+=1
                    if path in (files[0],files[-1]):replay_specs.append(dict(task=ident,path=str(path),indices=[0,chunk-1]))
                array=np.concatenate(chunks);pins=np.concatenate(centers);key=(t['degree'],kind,dist)
                if key not in reference_centers:
                    generated=data(t['degree'],count,dist,'test' if kind=='nominal' else 'pressure')
                    np.testing.assert_array_equal(generated,pins);reference_centers[key]=pins
                else:np.testing.assert_array_equal(reference_centers[key],pins)
                values[(t['degree'],t['seed'],t['method'],kind,dist)]=array
                row=dict(degree=t['degree'],seed=t['seed'],method=t['method'],suite=kind,distribution=dist,n=count,
                    mean=float(array.mean()),p95=float(np.quantile(array,.95)))
                original=original_metrics[tuple(str(row[k]) for k in ('degree','seed','method','suite','distribution'))]
                close(row['mean'],float(original['mean']));close(row['p95'],float(original['p95']));metric_rows.append(row)
        assert r['ledger']['actor_completed']==4*(4096+128*64)==r['ledger']['reference_requests']
        for k,v in r['ledger'].items():budget_sums[k]+=v
    csvwrite('metrics_recomputed.csv',metric_rows);write('replay_specs.json',replay_specs)
    progress('all_evaluation_rows_pass',prediction_rows=row_count,shards=shard_count)

    # Reconstruct every held-out split and compare canonical net identities.
    overlaps=[]
    for d in P['degrees']:
        sets={}
        for split,count in [('search-validation',64),('budget-validation',64),('model-validation',256),('pilot-validation',256),('test',4096),('pressure',128)]:
            pins=np.concatenate([data(d,count,dist,split) for dist in D]);keys=[pin_set_sha256(x) for x in pins]
            assert len(set(keys))==len(keys),(d,split,'within split duplicate');sets[split]=set(keys)
        for l,r in itertools.combinations(sets,2):
            assert not sets[l]&sets[r];overlaps.append(dict(degree=d,left=l,right=r,overlap=0))
    write('split_audit.json',overlaps)

    left=np.asarray([[values[(30,s,'es','nominal',dist)] for dist in D[1:]] for s in P['seeds']['30']])
    right=np.asarray([[values[(30,s,'random','nominal',dist)] for dist in D[1:]] for s in P['seeds']['30']])
    ltail=np.quantile(left,.95,axis=2).mean(axis=1);rtail=np.quantile(right,.95,axis=2).mean(axis=1);delta=ltail-rtail
    all_signs=np.asarray(list(itertools.product([-1,1],repeat=5)))
    p=float(np.mean((all_signs@delta)/5<=delta.mean()+1e-15))
    # Exact frozen RNG protocol, implemented independently from report.py.
    rng=np.random.default_rng(20260922);boot=[]
    for _ in range(2000):
        si=rng.integers(0,5,5);gs=[]
        for group in range(3):
            ni=rng.integers(0,4096,4096)
            a=np.quantile(left[si,group,:][:,ni],.95,axis=1)
            b=np.quantile(right[si,group,:][:,ni],.95,axis=1)
            gs.append(float((a-b).mean()))
        boot.append(float(np.mean(gs)))
    ci=np.quantile(boot,[.025,.975]);reduction=float(1-ltail.mean()/rtail.mean())
    nominal=float(np.mean([values[(30,s,'es','nominal','uniform')].mean()-values[(30,s,'erm','nominal','uniform')].mean() for s in P['seeds']['30']]))
    attackgap=float(np.mean([values[(30,s,'es','pressure',d)].mean()-values[(30,s,'random','pressure',d)].mean() for s in P['seeds']['30'] for d in D]))
    primary=dict(seed_differences=delta.tolist(),mean_difference=float(delta.mean()),descriptive_crossed_bootstrap_95ci=ci.tolist(),one_sided_seed_signflip_p=p,
        relative_reduction=reduction,improving_seeds=int((delta<0).sum()),nominal_mean_gap_vs_erm=nominal,cem64_mean_gap_vs_random=attackgap,
        primary_gain_supported=bool(p<=.05 and reduction>=.01 and (delta<0).sum()>=4 and nominal<=.002 and attackgap<=0))
    for k,v in primary.items():close(v,read(ART/'statistics.json')[k])
    write('statistics_recomputed.json',primary)
    comparisons=[]
    for d in P['degrees']:
        for kind in ('nominal','pressure'):
            for base in ('erm','random'):
                for metric in ('mean','p95'):
                    a=[];b=[]
                    for seed in P['seeds'][str(d)]:
                        fun=(lambda x:float(np.mean(x))) if metric=='mean' else (lambda x:float(np.quantile(x,.95)))
                        a.append(np.mean([fun(values[(d,seed,'es',kind,g)]) for g in D[1:]]))
                        b.append(np.mean([fun(values[(d,seed,base,kind,g)]) for g in D[1:]]))
                    comparisons.append(dict(degree=d,suite=kind,metric=metric,baseline=base,es=float(np.mean(a)),baseline_value=float(np.mean(b)),
                        relative_improvement=float(1-np.mean(a)/np.mean(b)),improving_seeds=int(np.sum(np.asarray(a)<np.asarray(b))),seeds=len(a)))
    csvwrite('comparisons.csv',comparisons)
    for t,r,_,_,_ in records.values():
        if t['kind']=='screen':
            for k,v in r['ledger'].items():budget_sums[k]+=v
    write('input_bindings.json',[dict(path=p,sha256=h) for p,h in sorted(bindings.items())])
    result=dict(status='PASS_ARTIFACTS_STATISTICS_PENDING_GPU_REPLAY',at=time.time(),elapsed_seconds=time.time()-started,
        tasks=len(records),task_kinds=dict(kinds),training_updates=sum(r['updates'] for r in train_rows),
        sampled_training_nets=sum(r['updates']*r['batch'] for r in train_rows),prediction_rows=row_count,evaluation_shards=shard_count,
        source_files=len(freeze['sources']),unique_verified_bindings=len(bindings),released_locks=98,
        train_batch_reconstructions=train_replays,audited_support_centers=audit_centers,primary=primary,ledger_totals=dict(budget_sums),
        model_freeze_after_training_before_tests=True,completed_utc=datetime.datetime.fromtimestamp(completion['at'],datetime.timezone.utc).isoformat(),
        total_wall_hours=(completion['at']-min(la['at'] for _,_,_,la,_ in records.values()))/3600,
        provenance_limits=['Per-step training input/selection hashes are complete; 147 batches reconstructed, not every training net.',
            'Held-out data disjointness fully recomputed; no complete train-vs-test content intersection claimed.',
            'All selected outputs checked; candidate supports audited for 1108 stored centers, not all 5e8 evaluated candidates.',
            'Training receipts complete; scientific primary criterion not met.'])
    write('audit.json',result);progress('artifact_statistics_audit_complete',status=result['status'])

if __name__=='__main__':
    try:main()
    except BaseException:
        write('audit_failure.json',dict(at=time.time(),traceback=traceback.format_exc()));raise

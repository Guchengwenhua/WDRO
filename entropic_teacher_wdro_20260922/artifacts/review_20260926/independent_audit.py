"""Acceptance from immutable raw artifacts; never writes experiment sources/results."""
import argparse,collections,csv,datetime,fcntl,hashlib,itertools,json,math,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path('/remote-home/Guilin/DRO/Codex/entropic_teacher_wdro_20260922')
ART=ROOT/'artifacts';OUT=ART/'review_20260926'
P=json.loads((ROOT/'protocol.json').read_text())
MAN=json.loads((ART/'operations/manifest.json').read_text())
DIST=P['distributions'];METHODS=P['methods']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,path)
def csvwrite(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def exact_cvar(x):
    s=np.sort(x);mass=len(s)/10;whole=int(mass);frac=mass-whole
    return float((s[-whole:].sum()+frac*s[-whole-1])/mass)
def old_cvar(x):
    q=np.quantile(x,.9);return float(q+np.maximum(x-q,0).mean()/.1)
def pid_same(record):
    try:
        fields=Path(f'/proc/{record["pid"]}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0] not in ('Z','X') and fields[19]==record['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
    except FileNotFoundError:return False
def unlocked(path):
    with Path(path).open() as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        fcntl.flock(f,fcntl.LOCK_UN);return True
def close(a,b,tol=1e-11):return bool(np.allclose(a,b,rtol=0,atol=tol))
def recompute_comparisons(metrics,variant='cvar90'):
    rows=[]
    for a,b in P['primary_pairs']:
        delta=[];rel=[];nominal=[];attacks={s:[] for s in ('random64','es64')}
        for seed in P['degrees']['30']['seeds']:
            ma=metrics[f'post_d30_{a}_s{seed}_best'];mb=metrics[f'post_d30_{b}_s{seed}_best']
            va=np.mean([ma[d+'_natural'][variant] for d in DIST[1:]]);vb=np.mean([mb[d+'_natural'][variant] for d in DIST[1:]])
            delta.append(float(vb-va));rel.append(float((vb-va)/vb));nominal.append(ma['uniform_natural']['mean']/mb['uniform_natural']['mean']-1)
            for suite in attacks:attacks[suite].append(np.mean([ma[d+'_'+suite]['mean']-mb[d+'_'+suite]['mean'] for d in DIST]))
        d=np.array(delta);p=float(sum(sum(s*x for s,x in zip(sign,d))/8>=d.mean()-1e-14 for sign in itertools.product([-1,1],repeat=8))/256)
        boot=d[np.random.default_rng(20260922).integers(0,8,(10000,8))].mean(1)
        rows.append(dict(method=a,reference=b,improvement=float(d.mean()),relative_improvement=float(np.mean(rel)),improving_seeds=int((d>0).sum()),nominal_change=float(np.mean(nominal)),attack_change={s:float(np.mean(v)) for s,v in attacks.items()},p=p,ci95=np.quantile(boot,[.025,.975]).tolist(),seed_differences=delta))
    ordered=sorted(range(len(rows)),key=lambda i:rows[i]['p']);prior=0
    for rank,i in enumerate(ordered):
        prior=max(prior,(len(rows)-rank)*rows[i]['p']);rows[i]['holm_p']=min(1.,prior)
    for r in rows:r['scientific_pass']=bool(r['holm_p']<.05 and r['relative_improvement']>=.005 and r['improving_seeds']>=6 and r['nominal_change']<=.005 and all(x<=0 for x in r['attack_change'].values()))
    return rows

def cpu_audit():
    import torch
    torch.set_num_threads(1)
    errors=[];warnings=[];hashed={};budget=[];transport_rows=[];metricrows=[];metrics={};selection=[]
    def check(ok,msg):
        if not ok:errors.append(msg)
    def hashcheck(path,digest):
        actual=sha(path);hashed[str(path)]=actual;check(actual==digest,'hash mismatch '+str(path));return actual
    frozen=read(ART/'operations/source_freeze.json');model_freeze=read(ART/'operations/model_freeze.json')
    for name,digest in {**frozen['binding'],**frozen['additional_files']}.items():hashcheck(ROOT/name,digest)
    check(model_freeze['source']==sha(ART/'operations/source_freeze.json'),'source freeze chain')
    check(len(MAN['training'])==116 and len(MAN['evaluation'])==224,'task counts')
    check(len({t['id'] for t in MAN['training']})==116,'training task uniqueness')
    check(len({t['id'] for t in MAN['evaluation']})==224,'evaluation task uniqueness')
    expected=set()
    for ds,c in P['degrees'].items():
        for s in c['seeds']:
            expected.add(f'pre_d{ds}_erm_s{s}')
            for m in METHODS:expected.add(f'post_d{ds}_{m}_s{s}')
    actual={t['id'] for t in MAN['training'] if t['seed']!=801}
    check(expected==actual,'formal factorial matrix')
    check(set(p.name for p in (ART/'runs').iterdir())=={t['id'] for t in MAN['training']},'extra/missing training directories')
    check(set(p.name for p in (ART/'evaluation').iterdir())=={t['id'] for t in MAN['evaluation']},'extra/missing evaluation directories')
    totals=collections.Counter();max_training_time=0.;earliest_training=float('inf');logs_with_extra=[]
    parent_best_validation={t['id']:read(ART/'runs'/t['id']/'complete.json') for t in MAN['training'] if t['stage']=='pre'}
    for number,t in enumerate(MAN['training']):
        out=ART/'runs'/t['id'];comp=read(out/'complete.json');run=read(out/'run.json');hb=read(out/'heartbeat.json')
        check(comp['status']=='complete' and comp['task']==t and comp['step']==t['updates'],t['id']+' completion')
        check(run['task']==t and run['binding']==frozen['binding'] and comp['binding']==frozen['binding'],t['id']+' source/task binding')
        check(not pid_same(run['identity']) and unlocked(out/'run.lock'),t['id']+' process/lock still active')
        check(hb['status']=='complete' and hb['step']==t['updates'],t['id']+' heartbeat terminal state')
        for fn,k in [('best.pt','best_sha256'),('latest.pt','latest_sha256')]:hashcheck(out/fn,comp[k])
        if t.get('parent'):
            parent=ART/'runs'/t['parent'];check(comp['parent_sha256']==sha(parent/'best.pt'),t['id']+' parent binding')
        c=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        check(c['binding']==frozen['binding'] and best['binding']==frozen['binding'],t['id']+' checkpoint binding')
        check(c['step']==t['updates'] and c['calls']==comp['calls'] and c['task']==t,t['id']+' checkpoint state')
        check(c['parent_sha256']==comp['parent_sha256'],t['id']+' checkpoint parent')
        def tensor_finite(v):
            if isinstance(v,torch.Tensor):return not (v.is_floating_point() and not torch.isfinite(v).all().item())
            if isinstance(v,dict):return all(tensor_finite(x) for x in v.values())
            if isinstance(v,(list,tuple)):return all(tensor_finite(x) for x in v)
            return True
        check(tensor_finite(c) and tensor_finite(best),t['id']+' nonfinite tensor')
        for opt in ('actor_optimizer','value_optimizer'):
            check(all(int(s['step'].item())==t['updates'] for s in c[opt]['state'].values()),t['id']+' Adam steps')
        history=[json.loads(l) for l in (out/'validation.jsonl').read_text().splitlines()]
        interval=t.get('val_every',P['validation_every']);expected_steps=[0]+list(range(interval,t['updates']+1,interval))
        check([h['step'] for h in history]==expected_steps,t['id']+' validation schedule/duplicates')
        base=history[0]['values']['uniform'];chosen=None;best_value=math.inf
        for h in history:
            eligible=t['stage']=='pre' or h['values']['uniform']<=base*1.005
            val=h['values']['uniform'] if t['stage']=='pre' else max(h['values'][d] for d in DIST[1:])
            if eligible and val<best_value:best_value=val;chosen=h
            check(eligible==h['eligible'] and close(val,h['metric']) and h['best_step']==chosen['step'],t['id']+' selection rule')
        check(best['step']==comp['best_step']==chosen['step'] and best['validation']==chosen['values'],t['id']+' chosen checkpoint')
        if t.get('parent'):
            pp=torch.load(ART/'runs'/t['parent']/'best.pt',map_location='cpu',weights_only=False)
            check(history[0]['values']==pp['validation'],t['id']+' common-parent validation')
        selection.append(dict(task=t['id'],degree=t['degree'],method=t['method'],seed=t['seed'],best_step=best['step'],last_step=c['step'],initial_uniform=base,best_uniform=chosen['values']['uniform'],last_uniform=history[-1]['values']['uniform']))
        actual_calls=collections.Counter();started=collections.Counter();padded=0;first=float('inf');last=0.
        with (out/'actual_calls.jsonl').open() as stream:
            for line in stream:
                v=json.loads(line);first=min(first,v['time']);last=max(last,v['time'])
                if v['event']=='start':started[v['phase']]+=v['rows']
                if v['event']=='complete':
                    actual_calls[v['phase']]+=v['rows']
                    if v['phase']!='sampled':padded+=math.ceil(v['rows']/256)*256
        check(first>=frozen['time'],t['id']+' launch before source freeze')
        check(comp['time']<model_freeze['time'],t['id']+' training after model freeze')
        earliest_training=min(earliest_training,first);max_training_time=max(max_training_time,comp['time'])
        expected_calls=dict(search=0 if t['method']=='erm' else t['updates']*128*16,teacher=t['updates']*128*16 if t['method']=='excess' else 0,validation=len(history)*4*256,sampled=t['updates']*128,test=0)
        for phase,value in expected_calls.items():
            check(comp['calls'][phase]==value and actual_calls[phase]==value,t['id']+' budget '+phase)
            if phase not in ('sampled',):check(started[phase]==actual_calls[phase],t['id']+' start/complete '+phase)
        check(padded==comp['calls']['padded'],t['id']+' padded rows')
        totals.update(expected_calls);totals['updates']+=t['updates'];totals['padded']+=padded
        budget.append(dict(task=t['id'],degree=t['degree'],method=t['method'],stage=t['stage'],seed=t['seed'],updates=t['updates'],worker_hours=comp['elapsed_seconds']/3600,**expected_calls,padded=padded))
        progress=[];finite_bad=False
        with (out/'progress.jsonl').open() as stream:
            for i,line in enumerate(stream,1):
                v=json.loads(line);progress.append(v)
                if v['step']!=i or any(isinstance(x,(float,int)) and not math.isfinite(x) for x in v.values()):finite_bad=True
        check(not finite_bad and len(progress)==t['updates'],t['id']+' progress continuity/nonfinite')
        check(c['progress_bytes']==(out/'progress.jsonl').stat().st_size,t['id']+' log/checkpoint byte binding')
        if t['method']!='erm':
            ec=np.array([v['expected_cost'] for v in progress]);sc=np.array([v['selected_cost'] for v in progress]);lam=np.array([v['lambda_'] for v in progress])
            check((lam>=0).all(),t['id']+' negative lambda')
            for i,v in enumerate(progress):
                tau=t['tau']*(4**(1-2*i/(t['updates']-1)) if t['method']=='anneal' else 1)
                check(close(tau,v['tau']),t['id']+' temperature schedule')
            if t['method'] not in ('hard','soft_dual'):
                check(ec.max()<=.04+1e-10,t['id']+' expected transport violation')
                check(all(abs(v['gap'])<=v['gap_tolerance'] and close(v['dual']-v['primal'],v['gap']) for v in progress),t['id']+' logged primal-dual gap')
            else:
                value=1.
                for v in progress:
                    check(close(v['lambda_'],value),t['id']+' lambda recursion');value=max(0.,value+2*(v['selected_cost']-.04))
                check(close(value,c['lam']),t['id']+' final lambda')
            transport_rows.append(dict(task=t['id'],degree=t['degree'],method=t['method'],seed=t['seed'],expected_mean=float(ec.mean()),expected_max=float(ec.max()),selected_mean=float(sc.mean()),sampled_over_budget_fraction=float((sc>.04).mean()),last_quarter_selected_mean=float(sc[-len(sc)//4:].mean()),lambda_mean=float(lam.mean()),lambda_last=float(lam[-1]),eta_last=c['eta']))
            z=np.load(out/'candidate_audit.npz');centers=z['centers'];cand=z['candidates'];cost=np.abs(centers.astype('float64')[:,None]-cand).sum(-1).mean(-1);pr=z['probability']
            check(close(cost,z['cost']) and np.array_equal(centers,cand[:,0]),t['id']+' independent cost/anchor')
            check((pr>=0).all() and close(pr.sum(1),np.ones(len(pr))),t['id']+' probability simplex')
            check(np.isfinite(cand).all() and (cand>=np.float32(.1)).all() and (cand<=np.float32(.9)).all(),t['id']+' geometry window')
            flat=cand.reshape(-1,t['degree'],2);dist=np.abs(flat[:,:,None].astype('float64')-flat[:,None]).sum(-1);dist[:,np.arange(t['degree']),np.arange(t['degree'])]=np.inf
            check(dist.min()>=1e-6,t['id']+' pin spacing')
            idx=((t['updates']-1)//256)*256;logged=progress[idx]
            check(close(float((pr*cost).sum(1).mean()),logged['expected_cost']),t['id']+' stored candidate expectation')
            check(close(float(cost[np.arange(128),z['index']].mean()),logged['selected_cost']),t['id']+' sampled transport')
            if t['method']=='excess':
                # Teacher-only forward is covered by frozen parent and GPU spot-check; historical candidate teacher outputs were not saved.
                warnings.append(t['id']+': historical teacher denominator not stored; full historical probability replay unavailable')
            else:
                ell=z['length']/math.sqrt(t['degree'])
                if t['method']=='tail':
                    eta_before=progress[idx-1]['eta'];ell=.5*ell+.5*(eta_before+np.maximum(ell-eta_before,0)/.1)
                if t['method']=='hard':expected_pr=np.eye(16)[np.argmax(ell-logged['lambda_']*cost,axis=1)]
                else:
                    logits=(ell-logged['lambda_']*cost)/logged['tau'];logits-=logits.max(1,keepdims=True);expected_pr=np.exp(logits);expected_pr/=expected_pr.sum(1,keepdims=True)
                check(close(pr,expected_pr),t['id']+' independent probability reconstruction')
            if t['method'] not in ('hard','soft_dual'):check((pr*cost).sum(1).mean()<=.04+1e-10,t['id']+' independent stored-budget check')
        for fn in ('complete.json','run.json','heartbeat.json','progress.jsonl','validation.jsonl','actual_calls.jsonl'):
            hashed[str(out/fn)]=sha(out/fn)
        if number%20==0:print('CPU checked training',number+1,flush=True)
    pilot_results=[]
    for t in MAN['training']:
        if t['seed']==801 and t['stage']=='post':
            h=json.loads((ART/'runs'/t['id']/'validation.jsonl').read_text().splitlines()[-1]);pilot_results.append((not h['eligible'],h['metric'],t['tau']))
    chosen=min(pilot_results)[2] if not all(v[0] for v in pilot_results) else .02
    check(chosen==read(ART/'operations/selection.json')['tau'],'pilot temperature selection')
    check(all(t.get('tau')==chosen for t in MAN['training'] if t['stage']=='post' and t['seed']!=801),'formal common temperature')
    correction={'natural':0.,'attack':0.};evaluation_calls=collections.Counter();earliest_eval=float('inf')
    for t in MAN['evaluation']:
        out=ART/'evaluation'/t['id'];comp=read(out/'complete.json');hb=read(out/'heartbeat.json');metrics[t['id']]={}
        check(comp['task']==t and comp['status']=='complete',t['id']+' eval completion')
        check(not pid_same(hb) and unlocked(out/'run.lock'),t['id']+' eval process/lock')
        model=ART/'runs'/t['run_id']/t['checkpoint'];check(comp['model_sha256']==model_freeze['models'][str(model)]==sha(model),t['id']+' frozen model')
        check(set(comp['shards'])=={d+'_'+s+'.npz' for d in DIST for s in ('natural','random64','es64')},t['id']+' shard set')
        orig={}
        for name,digest in comp['shards'].items():
            path=out/name;hashcheck(path,digest);v=np.load(path);arr=v['length'];original=v['original'];key=path.stem
            n=4096 if key.endswith('_natural') else 128
            check(arr.shape==original.shape==(n,) and np.isfinite(arr).all() and (arr>0).all(),t['id']+' invalid array '+name)
            check((arr>=original).all(),t['id']+' attack worse than original '+name)
            if n==4096:check(np.array_equal(arr,original),t['id']+' natural original')
            else:orig[key]=original
            m=dict(mean=float(arr.mean()),p95=float(np.quantile(arr,.95)),cvar90=exact_cvar(arr),cvar90_legacy=old_cvar(arr),n=n)
            saved=comp['metrics'][key]
            check(close([m['mean'],m['p95'],m['cvar90_legacy']],[saved['mean'],saved['p95'],saved['cvar90']]) and saved['n']==n,t['id']+' original metric mismatch '+key)
            correction['natural' if n==4096 else 'attack']=max(correction['natural' if n==4096 else 'attack'],m['cvar90_legacy']-m['cvar90'])
            metrics[t['id']][key]=m;metricrows.append(dict(run_id=t['run_id'],checkpoint=t['checkpoint'],degree=t['degree'],seed=t['seed'],method=t['method'],suite=key,**m))
        for d in DIST:check(np.array_equal(orig[d+'_random64'],orig[d+'_es64']),t['id']+' attacks have differing anchor outputs')
        counts=collections.Counter();starts=0;first=float('inf');padded=0
        with (out/'actual_calls.jsonl').open() as f:
            for line in f:
                v=json.loads(line);first=min(first,v['time'])
                if v['event']=='complete':counts[v['phase']]+=v['rows'];padded+=math.ceil(v['rows']/256)*256
                else:starts+=v['rows']
        check(first>=model_freeze['time'],t['id']+' evaluated before model freeze');earliest_eval=min(earliest_eval,first)
        check(counts['test']==starts==81920 and comp['calls']['test']==81920 and comp['calls']['padded']==padded,t['id']+' eval calls')
        evaluation_calls.update(counts)
        for fn in ('complete.json','heartbeat.json','actual_calls.jsonl'):hashed[str(out/fn)]=sha(out/fn)
    original_comparisons=recompute_comparisons(metrics,'cvar90_legacy');corrected=recompute_comparisons(metrics)
    old=read(ART/'report/audit.json')['comparisons']
    for r,s in zip(original_comparisons,old):
        for k in ('improvement','relative_improvement','nominal_change','p','holm_p','ci95','seed_differences'):check(close(r[k],s[k]),'original statistics mismatch '+r['method']+' '+k)
        check(r['scientific_pass']==s['scientific_pass'],'original decision mismatch '+r['method'])
    check(unlocked(ART/'operations/scheduler.lock') and not pid_same(read(ART/'operations/controller.json')),'controller still active')
    save('file_hashes.json',hashed);save('metrics_by_model.json',metrics);save('primary_comparisons.json',dict(corrected_exact_empirical_cvar=corrected,reproduced_original_formula=original_comparisons))
    csvwrite('metrics_corrected.csv',metricrows);csvwrite('budgets.csv',budget);csvwrite('transport_diagnostics.csv',transport_rows);csvwrite('checkpoint_selection.csv',selection)
    summary=dict(status='PASS' if not errors else 'FAIL',errors=errors,warnings=warnings,files_hashed=len(hashed),training_tasks=len(MAN['training']),evaluations=len(MAN['evaluation']),metric_rows=len(metricrows),totals=dict(totals),evaluation_calls=dict(evaluation_calls),selected_tau=chosen,max_cvar_correction=correction,exact_cvar_decisions_unchanged=all(a['scientific_pass']==b['scientific_pass'] for a,b in zip(original_comparisons,corrected)),earliest_training=earliest_training,model_freeze=model_freeze['time'],earliest_evaluation=earliest_eval,delivered=read(ART/'operations/delivery.json')['time'],worker_hours=sum(x['worker_hours'] for x in budget),best_at_zero=sum(x['best_step']==0 for x in selection),time=time.time())
    save('cpu_audit.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='warnings'},indent=2),flush=True)

def replay(index,parts):
    import torch
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
    import geometry as g
    from models.actor_critic import Actor
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.cuda.set_per_process_memory_fraction(.12)
    g.install_oracle_guard()
    # Independent terminal-span evaluator and union-find topology validation.
    def length(x,t):
        out=[]
        for pins,edges in zip(x.astype('float64'),t.reshape(len(t),-1,2)):
            parent=list(range(len(pins)));ys=[[float(v[1])] for v in pins];xs=[[float(v[0])] for v in pins]
            def root(k):
                while parent[k]!=k:k=parent[k]
                return k
            for u,v in edges:
                u=int(u);v=int(v);ru=root(u);rv=root(v)
                if ru==rv:raise ValueError('cycle')
                parent[ru]=rv;ys[u].append(float(pins[v,1]));xs[v].append(float(pins[u,0]))
            out.append(sum(max(v)-min(v) for v in ys)+sum(max(v)-min(v) for v in xs))
        return np.array(out)
    totals=collections.Counter();results=[];errors=[];max_error=0;data_cache={}
    def score(actor,x):
        chunks=[]
        with torch.no_grad():
            for i in range(0,len(x),256):
                chunk=x[i:i+256];n=len(chunk)
                if n<256:chunk=np.concatenate((chunk,np.repeat(chunk[-1:],256-n,axis=0)))
                top,_=actor(chunk,True);chunks.append(length(chunk[:n],top[:n].cpu().numpy()));totals['forward_rows']+=256;totals['predictions']+=n
        return np.concatenate(chunks)
    def dataset(degree,dist,namespace,n):
        key=(degree,dist,namespace,n)
        if key not in data_cache:data_cache[key]=g.make_data(n,degree,dist,P['namespace']+'/'+namespace,0)
        return data_cache[key]
    def verify(label,got,expected):
        nonlocal max_error
        error=float(np.max(np.abs(got-expected)));max_error=max(max_error,error)
        if not np.array_equal(got,expected):errors.append(dict(label=label,max_abs=error,mismatches=int((got!=expected).sum())))
        totals['arrays']+=1
    tasks=MAN['evaluation'][index::parts]
    start=time.time()
    for it,t in enumerate(tasks):
        model=ART/'runs'/t['run_id']/t['checkpoint'];actor=Actor(t['degree'],torch.device('cuda:0'));actor.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['actor']);actor.eval()
        for dist in DIST:
            x=dataset(t['degree'],dist,'test',4096)[:256];v=score(actor,x)
            target=np.load(ART/'evaluation'/t['id']/f'{dist}_natural.npz')['length'][:256];verify(t['id']+'/'+dist,v,target)
        # One full attack distribution for all seven methods at all three degrees, seed 901 best.
        if t['seed']==901 and t['checkpoint']=='best.pt' and t['run_id'].startswith('post_'):
            dist='uniform';x=dataset(t['degree'],dist,'attack',128);b,n,_=x.shape
            for method in ('random','es'):
                rng=np.random.default_rng(g.derived_seed(P['namespace'],'independent_attack',n,dist,method))
                if method=='random':
                    delta=np.clip(rng.normal(0,.04,(b,63,n,2)),-.08,.08);z=np.concatenate((x[:,None],g.legalize(x,delta)),1)
                    vals=score(actor,z.reshape(-1,n,2)).reshape(b,64)
                else:
                    u=rng.normal(size=(b,31,n,2));u/=np.maximum(np.sqrt((u*u).mean((-1,-2),keepdims=True)),1e-12)
                    delta=np.clip(.04*np.stack((u,-u),2).reshape(b,62,n,2),-.08,.08);z=np.concatenate((x[:,None],g.legalize(x,delta)),1)
                    vals=score(actor,z.reshape(-1,n,2)).reshape(b,63)
                    pairs=vals[:,1:].reshape(b,31,2);grad=((pairs[:,:,0]-pairs[:,:,1])[:,:,None,None]*u).mean(1);grad/=np.maximum(np.sqrt((grad*grad).mean((-1,-2),keepdims=True)),1e-12)
                    last=g.legalize(x,np.clip(.04*grad,-.08,.08));vals=np.column_stack((vals,score(actor,last)))
                target=np.load(ART/'evaluation'/t['id']/f'{dist}_{method}64.npz')
                verify(t['id']+'/'+method,vals.max(1),target['length']);verify(t['id']+'/'+method+'/anchor',vals[:,0],target['original'])
        totals['models']+=1
        if it%8==0:print('replay',index,it+1,'/',len(tasks),flush=True)
        save(f'gpu_replay_{index}.json',dict(status='RUNNING',gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),totals=dict(totals),max_abs_error=max_error,errors=errors,elapsed=time.time()-start))
    save(f'gpu_replay_{index}.json',dict(status='PASS' if not errors else 'FAIL',gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),totals=dict(totals),max_abs_error=max_error,errors=errors,elapsed=time.time()-start,time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','replay']);p.add_argument('--index',type=int,default=0);p.add_argument('--parts',type=int,default=4);a=p.parse_args()
    if a.mode=='cpu':cpu_audit()
    else:replay(a.index,a.parts)

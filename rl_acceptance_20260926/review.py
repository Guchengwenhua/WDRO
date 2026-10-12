"""Read-only experiment reacceptance; outputs never replace frozen run artifacts."""
import argparse,csv,hashlib,importlib,itertools,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np

BASE=Path('/remote-home/Guilin/DRO/Codex')
OUT=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)
def table(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def progress(mode,stage,**kw):
    record=dict(time_utc=stamp(),stage=stage,**kw);save(OUT/mode/'progress.json',record);print(json.dumps(record),flush=True)
def artifacts(root,manifest,key,errors):
    for name,digest in manifest.get(key,{}).items():
        p=root/name
        if not p.is_file() or sha(p)!=digest:errors.append('Historical artifact hash: '+str(p))
def independent_signflip(values):
    x=np.asarray(values);observed=float(x.mean());hits=0
    for mask in range(1<<len(x)):
        value=sum(float(v)*(1 if mask&(1<<i) else -1) for i,v in enumerate(x))/len(x)
        hits+=value>=observed-1e-12
    return hits/(1<<len(x))
def independent_holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.]*len(values);last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,values[i]*(len(values)-rank)));out[i]=last
    return out

def old():
    root=BASE/'rl_scaleup_20260921';dest=root/'artifacts/acceptance_20260926';dest.mkdir(exist_ok=True)
    sys.path.insert(0,str(root));r=importlib.import_module('report');s=importlib.import_module('scheduler');c=importlib.import_module('confirmatory')
    errors=[];assert s.verify_source_freeze(root)
    delivery=read(root/'artifacts/operations/delivery.json');audit=read(root/'artifacts/report/audit.json');confirm=read(root/'artifacts/report/confirmatory.json')
    if delivery['status']!='COMPLETE':errors.append('Historical delivery is incomplete')
    if delivery['queue_sha256']!=sha(root/'artifacts/operations/queue.json'):errors.append('Historical queue hash')
    if delivery['audit_sha256']!=sha(root/'artifacts/report/audit.json'):errors.append('Historical audit hash')
    artifacts(root/'artifacts/report',audit,'report_artifacts_sha256',errors)
    artifacts(root/'artifacts/report',confirm,'artifact_sha256',errors)
    save(OUT/'old/historical_status.json',dict(delivery=delivery,audit_status=audit['status'],confirmatory_status=confirm['status'],source_freeze_verified=True,errors=errors))
    progress('old','full_independent_report_replay')
    sys.argv=['report.py','--outdir',str(dest)]
    if r.main():errors.append('Full artifact/numerical/statistical replay failed')
    progress('old','checkpoint_dual_primary_replay')
    c.OUT=dest
    if c.main():errors.append('Checkpoint/primary replay failed')
    compared={}
    for name in ['metrics.csv','comparisons.csv','primary_comparisons.csv','training_budget.csv','runtime_and_calls.csv']:
        same=sha(dest/name)==sha(root/'artifacts/report'/name);compared[name]=same
        if not same:errors.append('Recomputed table differs: '+name)
    primary=read(dest/'confirmatory.json')['primary_comparisons']
    raw=[independent_signflip(list(json.loads(row['seed_improvements']).values())) for row in primary]
    adjusted=independent_holm(raw)
    for row,p,ph in zip(primary,raw,adjusted):
        if abs(p-row['p_one_sided'])>1e-14 or abs(ph-row['p_holm'])>1e-14:errors.append('Independent sign-flip/Holm mismatch')
    p=read(root/'protocol.json');lifecycle=[]
    for t in s.tasks(p):
        ok,reason=s.validate_complete(t,p,root=root,expected=s.expected_binding(root));lifecycle.append(dict(task=t['id'],passed=ok,reason=reason))
        if not ok:errors.append('Lifecycle '+t['id']+': '+reason)
    if s.discover_live_workers(root) or s.lock_held(root/'artifacts/operations/scheduler.lock'):errors.append('Old experiment still active')
    table(OUT/'old/lifecycle.csv',lifecycle)
    result=dict(status='PASS' if not errors else 'FAIL',time_utc=stamp(),tasks=80,errors=errors,recomputed_tables_byte_identical=compared,
        independent_signflip_holm_verified=True,primary=primary,replay_directory=str(dest),audit_sha256=sha(dest/'audit.json'),confirmatory_sha256=sha(dest/'confirmatory.json'))
    save(OUT/'old/acceptance.json',result);progress('old','finished',status=result['status'],errors=errors)

def new():
    root=BASE/'rl_prepost_scaleup_20260922';sys.path.insert(0,str(root))
    r=importlib.import_module('report');s=importlib.import_module('scheduler');o=importlib.import_module('ops');a=importlib.import_module('audit_utils')
    p=s.config();q=read(root/'artifacts/operations/queue.json');snap=o.snapshot();cap,cards,reasons=o.admission(snap,[snap],q['running'],p['resources']);errors=[]
    assert o.verify_source_freeze(root)
    launch=read(root/'artifacts/operations/launch.json');live=o.same_identity(o.ident(launch['identity']['pid']),launch['identity'])
    gates=read(root/'artifacts/operations/start_gate.json')
    for name,digest in gates['checks_sha256'].items():
        if sha(root/name)!=digest or read(root/name)['status']!='PASS':errors.append('Engineering gate '+name)
    tasks=[t for t in p['tasks'] if t['id'] in q['complete']];missing=[t for t in p['tasks'] if t['id'] not in q['complete']]
    save(OUT/'new/snapshot.json',dict(time_utc=stamp(),queue=q,resources=snap,capacity=cap,eligible_gpu_count=len(cards),exclusions=reasons,
        source_freeze_verified=True,scheduler_live=live,scheduler_lock_held=o.lock_held(root/'artifacts/operations/scheduler.lock'),workers=o.discover_live_workers(root),engineering_gate=gates))
    table(OUT/'new/missing_tasks.csv',[dict(task=t['id'],kind=t['kind'],method=t['method'],seed=t['seed'],start=t['start'],end=t['end']) for t in missing])
    progress('new','completed_task_audits',expected=104,available=len(tasks));results=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(r.audit_task,t):t for t in tasks}
        for future in as_completed(futures):
            t=futures[future]
            try:result=future.result();results.append(result);errors.extend(t['id']+': '+e for e in result['errors'])
            except Exception as exc:errors.append(t['id']+': '+repr(exc))
            progress('new','completed_task_audits',complete=len(results),available=len(tasks),errors=len(errors))
    common={};heldout=set();byid={x['task']:x for x in results}
    for x in results:
        for key,digest in x['input_hashes'].items():
            if key in common and common[key]!=digest:errors.append('Common input order '+key)
            common[key]=digest
        heldout.update(x.pop('heldout'))
    progress('new','independent_training_input_regeneration',audited=len(results))
    training=a.regenerate_training_ids(p);overlap=len(heldout&training);training_count=len(training);del training
    if overlap:errors.append('Training/heldout overlap')
    lineage=[]
    for receipt in sorted((root/'lineage').glob('*/receipt.json')):
        v=read(receipt);source=Path(v['source_directory']);local=receipt.parent
        ok=sha(local/'latest.pt')==v['checkpoint_sha256']==sha(source/'latest.pt') and sha(local/'complete.json')==v['source_complete_sha256']==sha(source/'complete.json')
        lineage.append(dict(source_run=local.name,passed=ok))
        if not ok:errors.append('Lineage '+local.name)
    primary=[];groupmeans=[];attacker_secondary=[]
    def vec(rid,step,d,suite='robust'):
        z=byid[rid]['stats'][str(step)]
        return (z[d+'__random32_length']+z[d+'__es32_length'])/2 if suite=='robust' else z[d+'__'+suite]
    specs=[('post_only','post_pre2048_{method}_s{seed}',8192,'post_pre2048_{method}_s{seed}',2048),
        ('pre_only_at_long_post','post_pre8192_{method}_s{seed}',8192,'post_pre2048_{method}_s{seed}',8192),
        ('ppo_vs_random_at_long_pre_long_post','post_pre8192_{method}_s{seed}',8192,'post_pre8192_random_s{seed}',8192)]
    for method in p['statistics']['primary_methods']:
        for name,left,ls,right,rs in specs:
            row=dict(contrast=name,method=method,expected_seeds=8,conclusion='PENDING_FULL_EXPERIMENT')
            available=[seed for seed in p['seeds'] if left.format(method=method,seed=seed) in byid and right.format(method=method,seed=seed) in byid]
            row['available_seeds']=len(available)
            if len(available)==8:
                groups={};ref=[];diff=[]
                for seed in available:
                    groups[seed]={}
                    for d in p['distributions']:
                        l=vec(left.format(method=method,seed=seed),ls,d);rr=vec(right.format(method=method,seed=seed),rs,d)
                        groups[seed][d]=rr-l;ref.append(float(rr.mean()))
                    diff.append(float(np.mean([x.mean() for x in groups[seed].values()])))
                bs=a.bootstrap(groups,p['statistics']['replicates']);base=float(np.mean(ref));delta=float(np.mean(diff))
                row.update(mean_improvement=delta,relative_improvement=delta/base,reference_mean=base,improving_seeds=sum(x>0 for x in diff),
                    p_one_sided=independent_signflip(diff),seed_improvements=json.dumps(dict(zip(available,diff))),ci_low=bs['ci_low'],ci_high=bs['ci_high'])
            primary.append(row)
    for pre in [2048,8192]:
        for method in p['main_methods']:
            for step in [2048,4096,8192]:
                ids=[f'post_pre{pre}_{method}_s{seed}' for seed in p['seeds']]
                available=[x for x in ids if x in byid]
                if len(available)!=8:continue
                for suite in ['nominal_length','random32_length','es32_length','robust']:
                    groupmeans.append(dict(pretraining=pre,posttraining=step,method=method,suite=suite,seeds=8,
                        mean=float(np.mean([vec(rid,step,d,suite).mean() for rid in available for d in p['distributions']]))))
    for method in p['statistics']['primary_methods']:
        available=[seed for seed in p['seeds'] if f'attacker8192_{method}_s{seed}' in byid]
        row=dict(method=method,available_seeds=len(available),scope='Secondary descriptive fixed-Actor lambda5 gain')
        if len(available)==8:
            groups={};refs=[];deltas=[]
            for seed in available:
                z=byid[f'attacker8192_{method}_s{seed}']['stats'];groups[seed]={}
                for d in p['distributions']:
                    key=f'{d}__lambda5__{method}__gain'
                    groups[seed][d]=z['8192'][key]-z['2048'][key];refs.append(float(z['2048'][key].mean()))
                deltas.append(float(np.mean([v.mean() for v in groups[seed].values()])))
            bs=a.bootstrap(groups,p['statistics']['replicates']);reference=float(np.mean(refs));delta=float(np.mean(deltas))
            row.update(mean_improvement=delta,reference_mean=reference,relative_improvement=delta/reference,
                improving_seeds=sum(x>0 for x in deltas),ci_low=bs['ci_low'],ci_high=bs['ci_high'])
        attacker_secondary.append(row)
    table(OUT/'new/attacker_secondary.csv',attacker_secondary)
    if not missing and all(row['available_seeds']==8 for row in primary):
        adjusted=independent_holm([row['p_one_sided'] for row in primary])
        for row,ph in zip(primary,adjusted):
            row['p_holm']=ph
            row['conclusion']='SUPPORTED' if ph<=.05 and row['relative_improvement']>=.005 and row['improving_seeds']>=6 else 'NOT_ESTABLISHED'
        official=root/'artifacts/report'
        if (official/'audit.json').exists():
            oa=read(official/'audit.json');artifacts(official,oa,'artifact_sha256',errors)
            if oa['status']!='PASS':errors.append('Official full numerical audit did not pass')
            with (official/'primary_comparisons.csv').open() as f:original=list(csv.DictReader(f))
            for row in primary:
                other=next(x for x in original if x['method']==row['method'] and x['contrast']==row['contrast'])
                for key in ['mean_improvement','reference_mean','relative_improvement','p_one_sided','p_holm','ci_low','ci_high']:
                    if abs(float(other[key])-row[key])>1e-12:errors.append('Official primary table differs: '+row['contrast']+'/'+key)
                if other['conclusion']!=row['conclusion']:errors.append('Official conclusion differs')
            with (official/'secondary_comparisons.csv').open() as f:secondary_original=list(csv.DictReader(f))
            for row in attacker_secondary:
                other=next(x for x in secondary_original if x['method']==row['method'] and x['contrast']=='attacker_8192_vs_2048')
                for key in ['mean_improvement','reference_mean','relative_improvement','ci_low','ci_high']:
                    if abs(float(other[key])-row[key])>1e-12:errors.append('Official attacker comparison differs: '+row['method']+'/'+key)
        else:errors.append('Official full audit is missing')
        delivery=root/'artifacts/operations/delivery.json'
        if not delivery.exists() or read(delivery)['status']!='COMPLETE':errors.append('Final lifecycle delivery incomplete')
        if o.discover_live_workers(root) or o.lock_held(root/'artifacts/operations/scheduler.lock'):errors.append('Final worker/scheduler lifecycle still active')
    table(OUT/'new/available_primary_descriptive.csv',primary);table(OUT/'new/full_seed_group_means.csv',groupmeans)
    table(OUT/'new/metrics.csv',[v for x in results for v in x['metrics']]);table(OUT/'new/training_budget.csv',[x['budget'] for x in results if x['budget']])
    table(OUT/'new/runtime_and_calls.csv',[x['runtime'] for x in results]);table(OUT/'new/lineage.csv',lineage)
    save(OUT/'new/task_audits.json',[{k:v for k,v in x.items() if k not in ['stats','metrics']} for x in results])
    current=read(root/'artifacts/operations/queue.json')
    result=dict(status='INCOMPLETE' if missing and not errors else 'FAIL' if errors else 'PASS',completed_subset_status='PASS' if not errors else 'FAIL',
        time_utc=stamp(),audited_tasks=len(results),snapshot_completed=len(tasks),current_completed=len(current['complete']),expected_tasks=104,
        missing_tasks=[t['id'] for t in missing],errors=errors,training_unique_inputs=training_count,heldout_unique_inputs=len(heldout),overlap=overlap,
        added_updates_verified=sum(x['runtime']['added_updates'] for x in results),expected_additional_updates=sum(t['end']-t['start'] for t in p['tasks']),
        primary=primary,attacker_secondary=attacker_secondary,scientific_status='PENDING_FULL_EXPERIMENT' if missing else 'REVIEW_REQUIRED' if errors else 'FULL_PRIMARY_FAMILY_RECOMPUTED',resource_capacity_at_snapshot=cap,free_gib=snap['free_bytes']/2**30,
        scheduler_live=live,source_freeze_verified=True,source_freeze_sha256=sha(root/'artifacts/operations/source_freeze.json'),
        read_only_scope='No training, resource thresholds, source, checkpoint, or formal report was altered.')
    save(OUT/'new/acceptance.json',result);progress('new','finished',status=result['status'],subset=result['completed_subset_status'],errors=errors)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['old','new']);parser.add_argument('--output-dir',type=Path);arg=parser.parse_args()
    if arg.output_dir:
        OUT=arg.output_dir.resolve()
        if not OUT.is_relative_to(BASE):parser.error('Acceptance output must stay in DRO/Codex')
    (OUT/arg.mode).mkdir(parents=True,exist_ok=True)
    try:old() if arg.mode=='old' else new()
    except BaseException as exc:
        save(OUT/arg.mode/'execution_error.json',dict(time_utc=stamp(),error=repr(exc)));raise

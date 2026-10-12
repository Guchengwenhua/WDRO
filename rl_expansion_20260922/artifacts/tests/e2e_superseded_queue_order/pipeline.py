"""Frozen stage generation and validation-only selection; no test read during selection."""
from pathlib import Path
import geometry as G
from common import ROOT,read,verify_complete,now
import engine as E

def publish(path,value):
    path=ROOT/path
    if path.exists():
        if read(path)!=value: raise RuntimeError('Immutable artifact differs: '+str(path))
    else: G.atomic_json(path,value)

def task(stage,arm,seed,c,start,target,parent=None,inherit=False):
    key=f'{stage}_{arm}_s{seed}'
    endpoints=sorted({target,*([14336] if arm=='B256' and target>=14336 else []),
                      *([32768] if start<32768<target else [])})
    return dict(id=key,mode='train',stage=stage,arm=arm,seed=seed,config=c,start=start,target=target,
                outdir=f'artifacts/runs/{key}',endpoints=endpoints,parent=parent,inherit_training=inherit)

def parent_from(t):
    path=ROOT/t['outdir']; c=verify_complete(path/'complete.json')
    info=c['models'][f"step{t['target']}"]
    return dict(path=str((path/info['file']).relative_to(ROOT)),sha256=info['sha256'])

def training_tasks(stage):
    return [read(ROOT/x['args'][2]) for x in read(ROOT/f'artifacts/queues/{stage}.json')['tasks']]

def queue(stage,tasks):
    rows=[]
    for t in tasks:
        name=f"artifacts/tasks/{t['id']}.json"; publish(name,t); out=t['outdir']; train=t['mode']=='train'
        rows.append(dict(id=t['id'],kind='training' if train else 'evaluation',
            args=['worker.py','--task-json',name],complete=out+'/complete.json',checkpoint=out+'/latest.pt' if train else '',
            resume_args=['--resume'],depends_on=[],cpu_threads=3,required_free_mib=6144,
            verify_artifacts=True,required_artifacts=['latest.pt','best.pt','results.npz','run.json'] if train else ['selection.json','run.json'],
            completion_checks=dict(acceptance='training_complete' if train else 'evaluation_complete'),
            binding_checks=dict(arm=t['arm'],seed=t['seed'])))
    q=dict(schema_version=1,tasks=rows,cpu_diagnostic_gate=dict(report='artifacts/tests/preflight.json'),
        source_paths=['protocol.json','docs/THEORY_PROTOCOL.md','artifacts/data/manifest.json','artifacts/inputs/lineage.json',
                      *[f"artifacts/tasks/{t['id']}.json" for t in tasks],
                      *[str(x.relative_to(ROOT)) for x in (ROOT/'vendor/models').glob('*.py')]])
    publish(f'artifacts/queues/{stage}.json',q)
    return tasks

def screen():
    p=read(ROOT/'protocol.json'); line=read(ROOT/'artifacts/inputs/lineage.json'); ts=[]
    # Interleave seeds/configurations so every hypothesis begins early.
    for seed in p['screening_seeds']:
        for arm,c in p['arms'].items():
            par={k:line[str(seed)][k] for k in ('path','sha256')}
            ts.append(task('screen',arm,seed,c,p['parent_step'],p['screen_target'],par))
    for seed in p['architecture_seeds']:
        for arm,c in p['architectures'].items(): ts.append(task('screen',arm,seed,c,0,p['screen_target']))
    return queue('screen',ts)

def promote():
    p=read(ROOT/'protocol.json'); ts=training_tasks('screen'); rec={}; source={}
    for t in ts:
        path=ROOT/t['outdir']/'complete.json'; c=verify_complete(path)
        if c['step']!=t['target']: raise RuntimeError('Incomplete screening')
        rec.setdefault(t['arm'],[]).append(c['endpoint_validation']); source[str(path.relative_to(ROOT))]=G.sha256(path)
    winners=E.select_variants({a:rec[a] for a in p['arms']},p)
    arch=E.select_architecture({a:rec[a] for a in p['architectures']},p)
    selected=['E50',*sorted(set(x for x in winners.values() if x))]
    row=dict(status='validation_selection_frozen',selected=selected,winners=winners,architecture=arch,
             endpoint_validation=rec,training_completions=source,protocol_sha256=G.sha256(ROOT/'protocol.json'))
    publish('artifacts/promotion.json',row)
    return row

def parents():
    p=read(ROOT/'protocol.json')
    return queue('parents',[task('parents','ME',s,p['architectures']['BASE'],0,p['parent_step']) for s in p['new_seeds']])

def final():
    p=read(ROOT/'protocol.json'); choice=read(ROOT/'artifacts/promotion.json'); ts=[]
    screen_map={(t['arm'],t['seed']):t for t in training_tasks('screen')}
    par_map={t['seed']:t for t in training_tasks('parents')}
    for seed in p['screening_seeds']+p['new_seeds']:
        for arm in choice['selected']:
            old=seed in p['screening_seeds']; src=screen_map[(arm,seed)] if old else par_map[seed]
            ts.append(task('final',arm,seed,p['arms'][arm],src['target'],p['final_target'],parent_from(src),old))
    if choice['architecture']:
        for seed in p['architecture_seeds']:
            for arm in ('BASE',choice['architecture']):
                src=screen_map[(arm,seed)]
                ts.append(task('final',arm,seed,p['architectures'][arm],src['target'],p['architecture_final_target'],parent_from(src),True))
    return queue('final',ts)

def freeze_and_evaluations():
    p=read(ROOT/'protocol.json'); all_tasks=sum((training_tasks(s) for s in ('screen','parents','final')),[])
    frozen={}; completions={}
    for t in all_tasks:
        path=ROOT/t['outdir']; c=verify_complete(path/'complete.json')
        if c['step']!=t['target']: raise RuntimeError('All-training gate not satisfied')
        completions[str((path/'complete.json').relative_to(ROOT))]=G.sha256(path/'complete.json')
        models={k:dict(path=str((path/v['file']).relative_to(ROOT)),**{x:v[x] for x in ('sha256','step','actor_sha256')})
                for k,v in c['models'].items() if k=='best' or v['step'] in t['endpoints']}
        frozen[t['id']]=dict(task=t,models=models)
    # The accepted parents and new-seed parents are explicit old-budget controls.
    line=read(ROOT/'artifacts/inputs/lineage.json')
    for seed in p['screening_seeds']:
        par=line[str(seed)]; import torch
        ck=torch.load(ROOT/par['path'],map_location='cpu',weights_only=False)
        from common import digest
        actor=G.build_actor(30,torch.device('cpu')); actor.load_state_dict(ck['actor'])
        key=f'origin_ME_s{seed}'; t=task('origin','ME',seed,p['arms']['E50'],p['parent_step'],p['parent_step'])
        frozen[key]=dict(task=t,models={f"step{p['parent_step']}":dict(path=par['path'],sha256=par['sha256'],step=p['parent_step'],actor_sha256=digest(actor))})
    selection=dict(status='all_training_frozen',training_completions=completions,models=frozen,
                   promotion_sha256=G.sha256(ROOT/'artifacts/promotion.json'),protocol_sha256=G.sha256(ROOT/'protocol.json'))
    publish('artifacts/selection.json',selection)
    selection_hash=G.sha256(ROOT/'artifacts/selection.json'); evaluations=[]
    for key,item in frozen.items():
        t=item['task']; degrees=[30,*p['secondary_degrees']] if t['stage']=='final' else [30]
        for degree in degrees:
            eid=f'eval_{key}_d{degree}'
            models=item['models'] if degree==30 else {k:v for k,v in item['models'].items() if k in ('best',f"step{t['target']}")}
            evaluations.append(dict(t,id=eid,mode='eval',outdir=f'artifacts/eval/{key}_d{degree}',
                                    models=models,degree=degree,selection_sha256=selection_hash))
    return queue('evaluation',evaluations)

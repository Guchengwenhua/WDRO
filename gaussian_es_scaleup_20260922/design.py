"""Task graph; all tuning is validation-only and all test tasks require a freeze."""
import json
from common import ART, PROTOCOL as P, atomic_json, artifact, checked, best_key, DISTS

def pre_id(d,s):return f'pre_d{d}_s{s}'
def post_id(d,s,m):return f'post_{m}_d{d}_s{s}'
def done(task_id):
    path=ART/'runs'/task_id/'complete.json'
    return json.loads(path.read_text()) if path.exists() else None

def training_task(d,s,m,phase,updates,batch,lr,parent=None,**kw):
    ident=pre_id(d,s) if phase=='pre' else post_id(d,s,m)
    return dict(id=ident,kind='train',phase=phase,degree=d,seed=s,method=m,updates=updates,batch=batch,lr=lr,parent=parent,**kw)

def initial_tasks():
    tasks=[training_task(30,901,'erm','pre',P['pre_updates']['30'],P['pre_batch'],P['pre_lr'])]
    for mu in P['search_mus']:
        for step in P['search_steps']:
            tasks.append(dict(id=f'screen_es_mu{mu}_step{step}',kind='screen',method='es',degree=30,seed=101,mu=mu,search_step=step,budget=64))
    for method in ('random','cem'):
        tasks.append(dict(id=f'screen_{method}',kind='screen',method=method,degree=30,seed=101,budget=64))
    # Degree30 s101 first to unblock outer tuning as early as possible.
    for d in (30,20,40):
        for s in P['seeds'][str(d)]:
            tasks.append(training_task(d,s,'erm','pre',P['pre_updates'][str(d)],P['pre_batch'],P['pre_lr']))
    return tasks

def search_selection(tasks):
    candidates=[t for t in tasks if t['id'].startswith('screen_es_')]
    if not all(done(t['id']) for t in candidates):return None
    path=ART/'manifests/search_selection.json'
    if path.exists():return json.loads(path.read_text())
    chosen=min(candidates,key=lambda t:(-done(t['id'])['mean_gain'],t['id']))
    rec=dict(mu=chosen['mu'],search_step=chosen['search_step'],task=chosen['id'],criterion='validation mean exact penalized score gain',
             inputs=[artifact(ART/'runs'/t['id']/'complete.json') for t in candidates])
    atomic_json(path,rec);return rec

def pilot_tasks(selection):
    tasks=[]
    for lr in P['pilot_lrs']:
        for nominal in P['pilot_nominal_fractions']:
            t=training_task(30,901,'es','pilot',P['pilot_updates'],P['post_batch'],lr,
                str(ART/'runs'/pre_id(30,901)),mu=selection['mu'],search_step=selection['search_step'],nominal_fraction=nominal)
            t['id']=f'pilot_lr{lr}_nominal{nominal}';tasks.append(t)
    return tasks

def outer_selection(tasks):
    if not all(done(t['id']) for t in tasks):return None
    path=ART/'manifests/outer_selection.json'
    if path.exists():return json.loads(path.read_text())
    eligible=[]
    for t in tasks:
        r=done(t['id'])
        # Initial best does not establish a successful pilot.
        if r['best_step']>0 and r['best_validation']['uniform']['mean']<=r['initial_validation']['uniform']['mean']+P['nominal_tolerance']:
            eligible.append(t)
    chosen=min(eligible,key=lambda t:(best_key(done(t['id'])['best_validation'],done(t['id'])['initial_validation']),t['id'])) if eligible else min(tasks,key=lambda t:(t['lr'],t['nominal_fraction']))
    rec=dict(lr=chosen['lr'],nominal_fraction=chosen['nominal_fraction'],task=chosen['id'],no_eligible_pilot=not eligible,
             inputs=[artifact(ART/'runs'/t['id']/'complete.json') for t in tasks])
    atomic_json(path,rec);return rec

def expand(tasks):
    selection=search_selection(tasks)
    if selection is None:return tasks
    existing={t['id'] for t in tasks}
    extras=[]
    for k in (64,256):
        for method in ('es','random'):
            t=dict(id=f'budget_{method}{k}',kind='screen',method=method,degree=30,seed=101,budget=k,
                mu=selection['mu'],search_step=selection['search_step'],split='budget-validation')
            extras.append(t)
    if done(pre_id(30,901)):
        pilots=pilot_tasks(selection);extras+=pilots
        outer=outer_selection(pilots)
        if outer:
            for d in P['degrees']:
                for s in P['seeds'][str(d)]:
                    if not done(pre_id(d,s)):continue
                    for method in ('erm','random','es'):
                        extras.append(training_task(d,s,method,'post',P['post_updates'][str(d)],P['post_batch'],outer['lr'],
                            str(ART/'runs'/pre_id(d,s)),mu=selection['mu'],search_step=selection['search_step'],nominal_fraction=outer['nominal_fraction']))
    tasks=tasks+[t for t in extras if t['id'] not in existing]
    expected=[post_id(d,s,m) for d in P['degrees'] for s in P['seeds'][str(d)] for m in ('erm','random','es')]
    if all(done(t) for t in expected):
        path=ART/'manifests/model_freeze.json'
        if path.exists():frozen=json.loads(path.read_text())
        else:
            models=[]
            for ident in expected:
                r=done(ident);checked(r['best'])
                models.append(dict(id=ident,degree=r['task']['degree'],seed=r['task']['seed'],method=r['task']['method'],best=r['best']))
            frozen=dict(models=models,source=artifact(ART/'manifests/source_freeze.json'),selection=artifact(ART/'manifests/outer_selection.json'))
            atomic_json(path,frozen)
        existing={t['id'] for t in tasks}
        for model in frozen['models']:
            ident='eval_'+model['id']
            if ident not in existing:tasks.append(dict(id=ident,kind='eval',degree=model['degree'],seed=model['seed'],method=model['method'],model=model['best']))
    return tasks

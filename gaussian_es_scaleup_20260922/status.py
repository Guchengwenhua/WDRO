import json
from common import ART,PROTOCOL as P
from design import initial_tasks

def estimate():
    path=ART/'tests/gpu_gate.json'
    if not path.exists():return dict(status='awaiting_gpu_profile')
    gate=json.loads(path.read_text());rates={(r['degree'],r['method'],r['batch'],r['budget']):r['seconds'] for r in gate['measurements']}
    # Replace startup-heavy ERM measurements with live production training when available.
    observed={}
    for path in (ART/'runs').glob('*/progress.jsonl'):
        ident=path.parent/'identity.json'
        if not ident.exists() or path.parent.name.startswith('engineering_'):continue
        t=json.loads(ident.read_text())['task']
        records=[json.loads(line) for line in path.read_text().splitlines()[-32:]]
        if len(records)<8:continue
        key=(t['degree'],t['method'],t['batch'],0 if t['method']=='erm' else 64)
        observed.setdefault(key,[]).extend(r['seconds'] for r in records)
    import statistics
    for key,seconds in observed.items():rates[key]=statistics.median(seconds)
    work=[]
    for t in initial_tasks():
        if t['kind']=='train':work.append((t['id'],t['updates']*rates[(t['degree'],'erm',128,0)]))
    chosen_max=max(rates[(30,'es',64,64)],rates[(30,'random',64,64)])
    work += [(f'pilot_{i}',P['pilot_updates']*rates[(30,'es',64,64)]) for i in range(4)]
    for d in P['degrees']:
        for s in P['seeds'][str(d)]:
            for m in ('erm','random','es'):
                sec=rates[(d,m,128,0)] if m=='erm' else rates[(d,m,64,64)]
                work.append((f'post_{m}_d{d}_s{s}',P['post_updates'][str(d)]*sec))
    total=sum(w for _,w in work)
    # Sum/8 is only a capacity lower bound: pilot parent and pilot tuning are dependencies.
    critical=P['pre_updates']['30']*rates[(30,'erm',128,0)]+P['pilot_updates']*rates[(30,'es',64,64)]+P['post_updates']['30']*chosen_max
    lower=max(total/8,critical)/3600
    return dict(status='profile_based_projection',worker_hours=total/3600,capacity_8_gpu_lower_hours=lower,
        planning_range_hours=[lower*1.15,lower*2.0],queue_wait='additional; cannot bound future contention',
        assumptions='production-shape single-update profile, 8 admitted GPUs, 15-100 percent evaluation and contention allowance; startup included')

if __name__=='__main__':
    path=ART/'status.json'
    if path.exists():
        s=json.loads(path.read_text());counts={}
        for t in s['tasks']:counts[t['status']]=counts.get(t['status'],0)+1
        print(json.dumps(dict(phase=s['phase'],counts=counts,running=[dict(id=t['id'],**t.get('heartbeat',{})) for t in s['tasks'] if t['status']=='running']),ensure_ascii=False,indent=2))
    print(json.dumps(estimate(),ensure_ascii=False,indent=2))

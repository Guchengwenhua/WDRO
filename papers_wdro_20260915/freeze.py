"""Bind prepared inputs, implemented methods and the written theory before CUDA."""
import json,os,sys,time
import numpy as np,scipy,torch
from core import *

def main():
    protocol=read_protocol();manifest=json.loads((ROOT/'artifacts/data/manifest.json').read_text())
    if manifest['status']!='complete' or manifest['protocol']!=digest(protocol):raise ValueError('data preparation is incomplete or mismatched')
    files=dict(manifest['files'])
    for path,hash_value in files.items():
        if sha(path)!=hash_value:raise ValueError('data hash changed: '+path)
    sources=list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))+[ROOT/'protocol.json',ROOT/'docs/THEORY.md',ROOT/'docs/PAPER_23_DAC25.md',ROOT/'docs/PAPER_ICCAD25.md',ROOT/'artifacts/data/manifest.json']
    for base in [ROOT/'vendor_harrl',ROOT/'sources/Timing-RST/models',ROOT/'sources/Timing-RST/utils',ROOT/'sources/An-Adaptive-Learning-Approach-for-Hub-Node-Selection/models',ROOT/'sources/An-Adaptive-Learning-Approach-for-Hub-Node-Selection/cpp',MAIN/'src']:
        sources += [p for p in base.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.hpp','.c')]
    sources+=list((ROOT/'artifacts/build').glob('*.so'))
    from src.rest import LEGACY_ROOT,DEFAULT_LIBRARY
    sources += [DEFAULT_LIBRARY,LEGACY_ROOT/'save/DAC21'/f'rsmt{protocol["degree"]}b.pt']+list((LEGACY_ROOT/'models').glob('*.py'))
    for p in sorted(set(sources)):files[str(p)]=sha(p)
    state={'schema_version':1,'protocol_sha256':sha(ROOT/'protocol.json'),'files':files,
        'software':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'torch_cuda':torch.version.cuda},
        'created_at':time.time(),'scope':'24 paired degree10 solver-task feasibility runs; no test-based changes'}
    target=ROOT/'artifacts/frozen.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old['files']!=files or old['protocol_sha256']!=state['protocol_sha256']:raise ValueError('frozen experiment differs; explicit new version required')
    else:atomic_json(target,state)
    freeze_hash=sha(target);tasks=[]
    for family in protocol['families']:
        task_id='gpu_smoke_'+family
        tasks.append({'id':task_id,'command':['bash','{root}/run_env.sh','{root}/runner.py','--family',family,'--method','wdro','--seed','9101','--device','cuda','--smoke','--smoke-batch-size','64','--run-id',task_id],
            'complete':f'artifacts/runs/{task_id}/complete.json','checkpoint':f'artifacts/runs/{task_id}/latest.pt','progress':f'artifacts/runs/{task_id}/heartbeat.json',
            'cpu_threads':2,'required_free_mib':4096,'completion_checks':{'status':'complete','scope':'smoke','completed_updates':6}})
    # Seed-interleaved order exposes all paired methods early when capacity varies.
    for seed in protocol['seeds']:
        for family in protocol['families']:
            for method in protocol['methods']:
                task_id=f'{family}_{method}_s{seed}'
                tasks.append({'id':task_id,'command':['bash','{root}/run_env.sh','{root}/runner.py','--family',family,'--method',method,'--seed',str(seed),'--device','cuda'],
                    'complete':f'artifacts/runs/{task_id}/complete.json','checkpoint':f'artifacts/runs/{task_id}/latest.pt','progress':f'artifacts/runs/{task_id}/heartbeat.json',
                    'depends_on':['gpu_smoke_'+family],'cpu_threads':2,'required_free_mib':4096,
                    'completion_checks':{'status':'complete','scope':'degree10_pilot','completed_updates':protocol['updates'],'protocol_sha256':sha(ROOT/'protocol.json'),'frozen_sha256':freeze_hash}})
    queue={'schema_version':1,'source_paths':[str(p.relative_to(ROOT)) for p in sorted(set(sources)) if p.is_relative_to(ROOT)]+['artifacts/frozen.json'],
        'tasks':tasks,'postprocess':{'command':['bash',str(ROOT/'run_env.sh'),str(ROOT/'report.py')],'complete':'artifacts/report/complete.json'}}
    q=ROOT/'queue.json'
    if q.exists() and json.loads(q.read_text())!=queue:raise ValueError('queue changed after freeze')
    if not q.exists():atomic_json(q,queue)
    print(json.dumps({'status':'frozen','files':len(files),'tasks':len(tasks),'frozen_sha256':freeze_hash}))

if __name__=='__main__':main()

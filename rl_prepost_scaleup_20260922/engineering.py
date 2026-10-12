"""Isolated CPU recovery proof and bounded production-batch CUDA pilot."""
import argparse,copy,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
import ops as o
ROOT=Path(__file__).resolve().parent
FILES=['runner.py','engine.py','policies.py','env_geometry.py','ops.py','scheduler.py','audit_utils.py','report.py']

def prepare(mode):
    root=ROOT/'artifacts'/mode
    if root.exists():
        if mode=='gpu_pilot':
            for n in ['runner.py','engine.py','policies.py','env_geometry.py']:
                if o.sha256(root/n)!=o.sha256(ROOT/n):raise RuntimeError('Existing pilot training source differs')
            for n in ['report.py','audit_utils.py','scheduler.py','ops.py']:shutil.copyfile(ROOT/n,root/n)
            return root,json.loads((root/'protocol.json').read_text())
        raise RuntimeError('Inspect existing engineering output instead of overwriting')
    root.mkdir(parents=True);p=json.loads((ROOT/'protocol.json').read_text())
    p.update(experiment='ENGINEERING_ONLY_'+mode,fixed_test_per_distribution=4 if mode.startswith('cpu') else 16,
             test_per_distribution=4 if mode.startswith('cpu') else 64,attack_per_distribution=4 if mode.startswith('cpu') else 16,
             evaluation_namespace='rl-prepost-engineering-only',checkpoint_every=1 if mode.startswith('cpu') else 4)
    ts=[]
    for original in p['tasks']:
        if original['seed']!=301:continue
        t=copy.deepcopy(original)
        if mode.startswith('cpu'):
            if not t['id'].startswith('post_pre2048_') or t['method'] not in ['bestof_ppo','sequential_ppo']:continue
            t['end']=2050;t['milestones']=[2050] if mode=='cpu_noeval' else [2048,2049,2050]
        else:
            t['end']=2064 if t['start']==2048 else 16
            t['milestones']=[2048,2056,2064] if t['start']==2048 else [8,16]
            for v in t['parents'].values():
                if v['source']=='new':v['step']=2064 if v['run'].startswith('pre8192') else 16
        ts.append(t)
    p['tasks']=ts;p['resources']['wait_for_origin_training']=False
    (root/'protocol.json').write_text(json.dumps(p,indent=2))
    for n in FILES:shutil.copyfile(ROOT/n,root/n)
    (root/'run_env.sh').write_text((ROOT/'run_env.sh').read_text().replace(str(ROOT),str(root)))
    for t in ts:
        for v in t['parents'].values():
            if v['source']=='legacy':
                dest=root/'lineage'/v['run']
                if not dest.exists():shutil.copytree(ROOT/'lineage'/v['run'],dest)
    return root,p

def run(root,task,gpu=None,extra=()):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu['index']))
    cmd=['nice','-n','15','bash',str(root/'run_env.sh'),str(root/'runner.py'),'--task-id',task['id'],*extra]
    if gpu is None:cmd+=['--allow-cpu']
    else:cmd+=['--physical-gpu-index',str(gpu['index']),'--gpu-uuid',gpu['uuid'],'--duty-fraction','.5']
    folder=root/'artifacts/runs'/task['id'];folder.mkdir(parents=True,exist_ok=True)
    with (root/'console.log').open('a') as log:
        return subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode

def same(x,y):
    if isinstance(x,torch.Tensor):return torch.equal(x.cpu(),y.cpu())
    if isinstance(x,np.ndarray):return np.array_equal(x,y)
    if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
    return x==y

def cpu():
    cached=ROOT/'artifacts/cpu_resume/acceptance.json'
    if cached.exists():
        assert json.loads(cached.read_text())['status']=='PASS';return
    # A bootstrap only invokes this after prior test processes have exited.
    # Preserve partial attempts and rerun this bounded proof from its source.
    for mode in ['cpu_resume','cpu_noeval']:
        old=ROOT/'artifacts'/mode
        if old.exists():old.rename(old.with_name(mode+'_interrupted_'+str(time.time_ns())))
    root,p=prepare('cpu_resume');other,op=prepare('cpu_noeval');checks={}
    for t in p['tasks']:
        assert run(root,t)==0
        path=root/'artifacts/runs'/t['id'];continuous=root/'continuous'/t['id'];continuous.parent.mkdir(exist_ok=True);shutil.move(path,continuous)
        assert run(root,t,extra=['--stop-after','2049'])==75
        assert run(root,t)==0
        assert run(other,next(x for x in op['tasks'] if x['id']==t['id']))==0
        a=torch.load(continuous/'latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
        c=torch.load(other/'artifacts/runs'/t['id']/'latest.pt',map_location='cpu',weights_only=False)
        fields=['actor','value','actor_optimizer','value_optimizer','attacker','rng','calls','step','lambda','history']
        results={'resume_'+k:same(a[k],b[k]) for k in fields}
        results.update({'evaluation_neutral_'+k:same(a[k],c[k]) for k in fields if k!='calls'})
        for step in t['milestones']:
            with np.load(continuous/'evaluations'/f'step_{step}.npz') as x,np.load(path/'evaluations'/f'step_{step}.npz') as y:
                results[f'evaluation_{step}']=x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
        checks[t['id']]=results
    passed=all(all(v.values()) for v in checks.values())
    o.atomic(root/'acceptance.json',dict(status='PASS' if passed else 'FAIL',checks=checks,scope='CPU continuation and evaluation-neutrality; not scientific GPU results'))
    print(json.dumps(checks));assert passed

def gpu():
    root,p=prepare('gpu_pilot');history=[];done=[];rates={}
    for t in p['tasks']:
        out=root/'artifacts/runs'/t['id'];existing=out/'complete.json'
        if existing.exists():
            record=json.loads(existing.read_text())
            if record['step']!=t['end'] or o.lock_held(out/'run.lock') or o.same_identity(o.ident(record['identity']['pid']),record['identity']):raise RuntimeError('Prior pilot worker is still active or incomplete')
            for name,digest in record['artifacts_sha256'].items():
                if o.sha256(out/name)!=digest:raise RuntimeError('Existing pilot artifact changed')
            done.append(t['id']);continue
        failures=0
        while True:
            while True:
                try:snap=o.snapshot()
                except (OSError,subprocess.SubprocessError,ValueError) as exc:
                    o.atomic(root/'state.json',dict(status='resource_probe_failed',error=repr(exc),complete=done,time=time.time()));time.sleep(15);continue
                history=(history+[snap])[-3:]
                resources=dict(p['resources'],max_admission_util=65)
                cap,cards,_=o.admission(snap,history,{},resources)
                if len(history)>=3 and cap>0 and cards:break
                o.atomic(root/'state.json',dict(status='waiting_resources',complete=done,time=time.time()));time.sleep(5)
            device=cards[0]
            o.atomic(root/(t['id']+'_admission.json'),dict(snapshot=snap,history=history,device=device,duty_fraction=.5,scope='One bounded engineering worker'))
            o.atomic(root/'state.json',dict(status='running',task=t['id'],gpu=device['index'],complete=done,time=time.time()))
            result=run(root,t,device)
            if result==0:break
            if result!=75:failures+=1
            if failures>=3:raise RuntimeError('GPU pilot failed: '+t['id'])
            o.atomic(root/'state.json',dict(status='retry_wait',task=t['id'],returncode=result,failures=failures,complete=done,time=time.time()))
            time.sleep(30)
        done.append(t['id'])
    for t in p['tasks']:
        out=root/'artifacts/runs'/t['id'];m=json.loads((out/'complete.json').read_text())
        rows=[json.loads(x) for x in (out/'progress.jsonl').read_text().splitlines()]
        # Use deltas that do not cross a milestone evaluation.
        costs=[rows[i]['elapsed_seconds']-rows[i-1]['elapsed_seconds'] for i in range(1,len(rows)) if rows[i-1]['step'] not in t['milestones']]
        rates[t['kind']+'_'+t['method']]=dict(seconds_per_update=float(np.median(costs)),duty_fraction=.5,peak_allocated_mib=m['peak_allocated_bytes']/2**20)
    audit_code="import json; import report; import scheduler; import ops; r=[report.audit_task(t) for t in scheduler.config()['tasks']]; errors={x['task']:x['errors'] for x in r if x['errors']}; ops.atomic('audit.json',{'status':'PASS' if not errors else 'FAIL','errors':errors,'tasks':len(r)}); assert not errors, errors"
    with (root/'audit.log').open('w') as f:
        result=subprocess.run(['bash',str(root/'run_env.sh'),'-c',audit_code],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('GPU pilot numerical audit failed')
    o.atomic(root/'acceptance.json',dict(status='PASS',tasks=len(done),rates=rates,source_sha256={n:o.sha256(root/n) for n in FILES},scope='Production-batch engineering CUDA pilot; separate from formal 104 tasks'))
    o.atomic(ROOT/'artifacts/operations/calibration.json',rates)
    print(json.dumps(rates,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['cpu','gpu']);args=parser.parse_args()
    cpu() if args.mode=='cpu' else gpu()

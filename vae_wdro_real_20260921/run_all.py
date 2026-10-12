"""Detached single-instance pipeline; each GPU job is admitted on an empty GPU."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from exp_common import *
from data import prepare

def gpu_snapshot():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    occupied=set(apps.splitlines());rows=[]
    for line in output.splitlines():
        idx,uuid,used,total,util=[v.strip() for v in line.split(',')]
        rows.append(dict(index=int(idx),uuid=uuid,used_mib=int(used),total_mib=int(total),utilization=int(util),compute_process=uuid in occupied))
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config.json');parser.add_argument('--resume',action='store_true');parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    lock=(ART/'pipeline.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('This experiment already has a live pipeline owner')
    if (ART/'done'/'pipeline.json').exists(): return
    atomic_json(ART/'launcher.json',dict(pid=os.getpid(),command=sys.argv,time=utc(),cwd=str(ROOT)))
    prepare();started=read_json(ART/'config.executed.json')['started_at'];active={};attempts={};failures={};last_report=0.;last_snapshot=[]
    def done(key): return (ART/'done'/f'{key}.json').exists()
    def failed(key): return key in failures
    def launch(key,script,args=(),gpu=False):
        if done(key) or key in active or failed(key): return False
        if not (ROOT/script).exists(): return False
        if script=='train.py' and sum(k.startswith('train_') for k in active)>=CFG['runtime']['max_gpu_jobs']:return False
        device=None
        if gpu:
            if sum(v['device'] is not None for v in active.values())>=CFG['runtime']['max_gpu_jobs']: return False
            owned={v['device']['uuid'] for v in active.values() if v['device'] is not None}
            candidates=[g for g in gpu_snapshot() if g['uuid'] not in owned and not g['compute_process'] and g['used_mib']<64 and g['utilization']<=5]
            if not candidates:return False
            device=candidates[0]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device['uuid'] if device else ''
        (ART/'logs').mkdir(exist_ok=True)
        handle=(ART/'logs'/f'{key}.log').open('a')
        cmd=[CFG['python'],'-u',str(ROOT/script),*map(str,args)]
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        active[key]=dict(process=p,log=handle,device=device,start=time.time(),command=cmd)
        attempts[key]=attempts.get(key,0)+1
        event=dict(event='launch',job=key,pid=p.pid,device=device,time=utc(),command=cmd,attempt=attempts[key])
        append_json(ART/'events.jsonl',event);print(event,flush=True)
        return True
    def report(final=False):
        if not (ROOT/'report.py').exists():return
        cmd=[CFG['python'],str(ROOT/'report.py')]+(['--final'] if final else [])
        with (ART/'logs'/'report.log').open('a') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:print('REPORT ERROR: see report.log',flush=True)
    def train_launch(method,seed):
        key=f'train_{method}_s{seed}'
        if done(key) or key in active or failed(key):return
        path=ART/'devices'/f'seed{seed}.json'
        if path.exists():backend=read_json(path)['backend']
        else:
            idle=[g for g in gpu_snapshot() if not g['compute_process'] and g['used_mib']<64 and g['utilization']<=5]
            backend='cuda' if idle else 'cpu'
            atomic_json(path,dict(seed=seed,backend=backend,time=utc(),reason='empty GPU available' if idle else 'all GPUs occupied; use idle CPU capacity; same backend across all methods for this seed'))
        launch(key,'train.py',['--method',method,'--seed',seed,'--device',backend],gpu=backend=='cuda')
    main_keys=[f'train_{m}_s{s}' for s in CFG['seeds'] for m in CFG['methods_required']]
    secondary_keys=[f'train_{m}_s{s}' for s in CFG['seeds'] for m in CFG['methods_secondary']]
    while True:
        changed=False
        for key,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del active[key];changed=True
            event=dict(event='exit',job=key,returncode=code,time=utc(),wall_seconds=time.time()-job['start'])
            append_json(ART/'events.jsonl',event);print(event,flush=True)
            if code!=0 or not done(key):
                if attempts[key]>=2:
                    failures[key]=dict(returncode=code,time=utc(),log=str(ART/'logs'/f'{key}.log'))
                    atomic_json(ART/'failures.json',failures)
        launch('raw_geometry','candidates.py',['raw'])
        launch('vae','vae.py',gpu=True)
        if done('raw_geometry'):launch('raw_wdro_reference','oracle.py',['raw_wdro'])
        if done('vae'):launch('latent_geometry','candidates.py',['latent'])
        ref_active=any(k.endswith('_reference') for k in active)
        if done('latent_geometry') and not ref_active and (done('raw_wdro_reference') or failed('raw_wdro_reference')):
            if not done('vae_shell_wdro_reference') and not failed('vae_shell_wdro_reference'):
                launch('vae_shell_wdro_reference','oracle.py',['vae_shell_wdro'])
            elif done('vae_shell_wdro_reference') or failed('vae_shell_wdro_reference'):
                launch('vae_ball_wdro_reference','oracle.py',['vae_ball_wdro'])
        implementation=(ROOT/'IMPLEMENTATION_READY').exists()
        if implementation:
            for method in CFG['methods_required']:
                if done(method+'_reference'):train_launch(method,101)
            pair101=all(done(f'train_{m}_s101') for m in CFG['methods_required'])
            if pair101 and not (ROOT/'INTERIM_REPORT.md').exists():report()
            # Release later seeds only after both full seed101 tests have completed.
            if pair101:
                for seed in CFG['seeds'][1:]:
                    for method in CFG['methods_required']:
                        if done(method+'_reference'):train_launch(method,seed)
            if all(done(k) or failed(k) for k in main_keys) or (failures and not any(k in active for k in main_keys)):
                for seed in CFG['seeds']:
                    for method in CFG['methods_secondary']:
                        if method=='erm_a' or done(method+'_reference'):
                            train_launch(method,seed)
        if changed and implementation:report();last_report=time.time()
        if implementation and time.time()-last_report>=3600:report();last_report=time.time()
        try:last_snapshot=gpu_snapshot()
        except Exception as exc:print('GPU snapshot error',repr(exc),flush=True)
        status=dict(time=utc(),pipeline_pid=os.getpid(),started_at=started,
            active={k:dict(pid=v['process'].pid,device=v['device'],seconds=time.time()-v['start']) for k,v in active.items()},
            completed=[k for k in main_keys+secondary_keys if done(k)],failures=failures,gpus=last_snapshot,
            implementation_ready=implementation)
        atomic_json(ART/'status.json',status)
        if implementation and all(done(k) for k in main_keys+secondary_keys):
            report(final=True);mark_done('pipeline',status='completed',runs=12,started_at=started);return
        if implementation and failures and not active:
            report(final=True);mark_done('pipeline',status='incomplete_with_failures',failures=failures,started_at=started);return
        time.sleep(10)

if __name__=='__main__':main()

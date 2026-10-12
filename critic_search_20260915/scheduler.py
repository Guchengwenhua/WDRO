"""Persistent local queue: one bounded task per GPU, resumable task artifacts."""
import fcntl,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXP=ROOT/'artifacts';OPS=EXP/'operations'
VARIANTS=['refresh32_raw','refresh8_raw','refresh32_ema','refresh8_ema']

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in raw.splitlines():
        idx,uuid,free,util=[v.strip() for v in line.split(',')]
        gpus.append(dict(index=int(idx),uuid=uuid,free_mib=int(free),util=int(util)))
    return dict(at=time.time(),gpus=gpus,disk_free=shutil.disk_usage(EXP).free,
        artifact_logical=str(EXP),artifact_resolved=str(EXP.resolve()))

def tasks():
    items=[dict(id='fixed_validation_s101',args=['fixed.py','--phase','validation','--seed','101'],
                complete=EXP/'fixed/validation_s101/complete.json',kind='validation')]
    for seed in (101,102,103):
        for variant in VARIANTS:
            items.append(dict(id=f'training_{variant}_{seed}',args=['training.py','--variant',variant,'--seed',str(seed)],
                complete=EXP/'training'/f'{variant}_d30_s{seed}'/'complete.json',kind='training'))
    for seed in (101,102,103):
        items.append(dict(id=f'fixed_test_s{seed}',args=['fixed.py','--phase','test','--seed',str(seed)],
            complete=EXP/'fixed'/f'test_s{seed}'/'complete.json',kind='test'))
    return items

def main():
    OPS.mkdir(parents=True,exist_ok=True)
    lock=(OPS/'queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # A lost scheduler must never launch duplicate workers still running from it.
    prior=OPS/'queue.json'
    if prior.exists():
        for row in json.loads(prior.read_text()).get('running',[]):
            stat=Path(f"/proc/{row['pid']}/stat")
            if stat.exists() and stat.read_text().split()[21]==row['start_ticks']:
                raise RuntimeError(f"Prior worker still active: {row['id']} pid {row['pid']}")
    queue=tasks();running={};failures=[];completed=[]
    atomic(OPS/'launch.json',dict(pid=os.getpid(),at=time.time(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    while queue or running:
        for gpu,item in list(running.items()):
            rc=item['process'].poll()
            if rc is None:continue
            item['log'].close();task=item['task']
            if rc==0 and task['complete'].exists():completed.append(task['id'])
            else:failures.append(dict(id=task['id'],exit_code=rc,at=time.time()))
            del running[gpu]
        snap=snapshot();atomic(OPS/'resource_latest.json',snap)
        if snap['disk_free']<30*2**30:raise RuntimeError('artifact disk below 30GiB')
        for gpu in sorted(snap['gpus'],key=lambda x:-x['free_mib']):
            g=gpu['index']
            if g in running or gpu['free_mib']<6144:continue
            eligible=[t for t in queue if t['kind']!='test' or (EXP/'fixed/validation_s101/complete.json').exists()]
            if not eligible:continue
            task=eligible[0];queue.remove(task)
            if task['complete'].exists():completed.append(task['id']);continue
            log=(OPS/(task['id']+'.log')).open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g))
            # Enter the established environment, then override caches into this branch.
            stage3=Path('/remote-home/Guilin/DRO/Codex/adversarial_v3_1_20260912/wdro_rest_min/scripts/stage3_env.sh')
            cmd=['bash',str(stage3),'bash',str(ROOT/'run_env.sh')]+task['args']+['--physical-gpu-index',str(g)]
            atomic(OPS/f"preflight_{task['id']}_{time.time_ns()}.json",dict(snap,command=cmd,gpu=g))
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21]
            running[g]=dict(process=p,task=task,log=log,start_ticks=ticks)
        state=dict(at=time.time(),pid=os.getpid(),pending=[t['id'] for t in queue],completed=completed,failures=failures,
            running=[dict(id=v['task']['id'],gpu=g,pid=v['process'].pid,start_ticks=v['start_ticks']) for g,v in running.items()])
        atomic(OPS/'queue.json',state)
        # Failed validation makes its dependent tests unlaunchable; preserve exact failure.
        if not running and queue and all(t['kind']=='test' for t in queue) and any(f['id']=='fixed_validation_s101' for f in failures):break
        time.sleep(10)
    atomic(OPS/'exit.json',dict(at=time.time(),pid=os.getpid(),failures=failures,pending=[t['id'] for t in queue],completed=completed))
    if not failures and not queue:
        p=subprocess.run(['bash',str(ROOT/'run_env.sh'),'report.py'],cwd=ROOT)
        if p.returncode:raise RuntimeError('report/audit failed')

if __name__=='__main__':main()

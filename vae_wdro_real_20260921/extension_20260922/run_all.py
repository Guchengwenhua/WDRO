"""Small detached resumable queue; occupied GPUs are never admitted."""
import fcntl
import os
import subprocess
import sys
import time
from exp_common import *
from prepare import main as prepare

def gpu_snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    return [dict(index=int(a),uuid=b.strip(),memory_mib=int(c),utilization=int(d),occupied=b.strip() in apps)
        for a,b,c,d in [line.split(',') for line in raw.splitlines()]]

class AdoptedProcess:
    def __init__(self,pid,key):self.pid=pid;self.key=key
    def poll(self):
        p=Path(f'/proc/{self.pid}/cmdline');stat=Path(f'/proc/{self.pid}/stat')
        if p.exists() and str(ROOT).encode() in p.read_bytes() and stat.read_text().split(')')[-1].strip().split()[0]!='Z':return None
        return 0 if (ART/'done'/f'{self.key}.json').exists() else 1

def main():
    ART.mkdir(parents=True,exist_ok=True);lock=(ART/'pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A pipeline already owns this experiment')
    if (ART/'done'/'pipeline.json').exists():return
    active={};attempts={};failures={};last_report=0.;snap=[]
    if (ART/'status.json').exists():
        prior=read_json(ART/'status.json');failures=prior.get('failures',{})
        for key,s in prior.get('active',{}).items():
            adopted=AdoptedProcess(s['pid'],key)
            if adopted.poll() is None:active[key]=dict(process=adopted,handle=None,device=s.get('device'),compute=s.get('compute',True),started=s.get('started',time.time()),command=s.get('command',[]))
    if (ART/'events.jsonl').exists():
        for line in (ART/'events.jsonl').read_text().splitlines():
            e=json.loads(line)
            if e['event']=='launch':attempts[e['job']]=max(attempts.get(e['job'],0),e['attempt'])
    atomic_json(ART/'launcher.json',dict(pid=os.getpid(),time=utc(),command=sys.argv,cwd=str(ROOT)))
    prepare();started=read_json(ART/'config.executed.json')['started_at'];started_epoch=__import__('datetime').datetime.fromisoformat(started).timestamp()
    def done(key):return (ART/'done'/f'{key}.json').exists()
    def launch(key,script,args=(),compute=False,gpu_allowed=False,backend=None):
        if done(key) or key in active or key in failures or not (ROOT/script).exists():return False
        if compute and sum(j['compute'] for j in active.values())>=CFG['runtime']['max_compute_jobs']:return False
        device=None
        if gpu_allowed and backend!='cpu':
            owned={j['device']['uuid'] for j in active.values() if j['device']}
            empty=[g for g in gpu_snapshot() if not g['occupied'] and g['memory_mib']<64 and g['utilization']<=5 and g['uuid'] not in owned]
            if empty:device=empty[0]
            elif backend=='cuda':return False
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device['uuid'] if device else ''
        command=[CFG['python'],'-u',str(ROOT/script),*map(str,args)]
        if script in ['train.py','cvae.py']:command+=['--device','cuda' if device else 'cpu']
        (ART/'logs').mkdir(exist_ok=True);handle=(ART/'logs'/f'{key}.log').open('a')
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        active[key]=dict(process=p,handle=handle,device=device,compute=compute,started=time.time(),command=command)
        attempts[key]=attempts.get(key,0)+1
        event=dict(event='launch',job=key,pid=p.pid,device=device,command=command,time=utc(),attempt=attempts[key]);append_json(ART/'events.jsonl',event);print(event,flush=True)
        return True
    def report(final=False):
        if not (ROOT/'report.py').exists():return
        with (ART/'logs'/'report.log').open('a') as f:
            r=subprocess.run([CFG['python'],str(ROOT/'report.py')]+(['--final'] if final else []),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:print('REPORT failed; retained logs',flush=True)
    def start_main(method,seed,g,target):
        key=f'main_{method}_g{g}_s{seed}_u{target}'
        if done(key) or key in active or key in failures:return
        bank=f'{method}_g{g}' if method in ['local_shell','local_ball','mix_local'] else method
        if method!='erm_a' and not (ART/'pools'/f'{bank}.npz').exists():return
        if any(k.startswith(f'main_{method}_g{g}_s{seed}_u') for k in active):return
        path=ART/'devices'/f'seed{seed}.json'
        if path.exists():backend=read_json(path)['backend']
        else:
            available=[x for x in gpu_snapshot() if not x['occupied'] and x['memory_mib']<64 and x['utilization']<=5]
            backend='cuda' if available else 'cpu';atomic_json(path,dict(seed=seed,backend=backend,time=utc()))
        launch(key,'train.py',['--method',method,'--seed',seed,'--generator',g,'--target',target],compute=True,gpu_allowed=True,backend=backend)
    diagkeys=[f'diag_{r}_{m}_s{s}' for r in CFG['recipes'] for s in [101,102] for m in ['erm_a','raw_wdro']]
    primary=[(m,s,0) for s in CFG['seeds'] for m in CFG['primary_methods']]
    extra=[('local_shell',s,g) for g in [1,2] for s in CFG['seeds']]
    while True:
        changed=False
        for key,j in list(active.items()):
            code=j['process'].poll()
            if code is None:continue
            if j['handle']:j['handle'].close()
            del active[key];changed=True
            append_json(ART/'events.jsonl',dict(event='exit',job=key,returncode=code,time=utc(),wall_seconds=time.time()-j['started']))
            print('EXIT',key,code,flush=True)
            if (code or not done(key)) and attempts.get(key,1)>=2:
                failures[key]=dict(returncode=code,time=utc(),log=str(ART/'logs'/f'{key}.log'));atomic_json(ART/'failures.json',failures)
        launch('bn_diagnosis','bn_diagnosis.py',compute=True)
        launch('cvae_g0','cvae.py',['--generator',0],compute=True,gpu_allowed=True)
        # All diagnostic variants are formal 256-step runs. Standard_1e5 is read-only reuse.
        for seed in [101,102]:
            for recipe in CFG['recipes']:
                for method in ['erm_a','raw_wdro']:
                    launch(f'diag_{recipe}_{method}_s{seed}','train.py',
                        ['--phase','diag','--recipe',recipe,'--method',method,'--seed',seed,'--target',256],compute=True)
        if all(done(k) for k in diagkeys):launch('recipe_selection','select_recipe.py')
        if done('cvae_g0'):launch('geometry_g0','pools.py',['--generator',0],compute=True)
        # Exactly one 8-worker reference queue at a time across all banks.
        if not any(k.endswith('_reference') for k in active):
            for g in [0,1,2]:
                if done(f'geometry_g{g}'):
                    found=False
                    for m in ['local_shell','local_ball']:
                        bank=f'{m}_g{g}'
                        if not done(bank+'_reference') and bank+'_reference' not in failures:
                            launch(bank+'_reference','oracle.py',[bank]);found=True;break
                    if found:break
        launch('mix_legacy','pools.py',['--mix','legacy'])
        for g in [0,1,2]:
            if done(f'local_shell_g{g}_reference'):launch(f'mix_local_g{g}','pools.py',['--mix','local','--generator',g])
        ready=(ROOT/'IMPLEMENTATION_READY').exists()
        if ready and done('recipe_selection'):
            for m in ['raw_wdro','local_shell']:start_main(m,101,0,256)
            first=all(done(f'main_{m}_g0_s101_u256') for m in ['raw_wdro','local_shell'])
            if first and not (ROOT/'INTERIM_REPORT.md').exists():report()
            if first:
                for s in [102,103]:
                    for m in ['raw_wdro','local_shell']:start_main(m,s,0,256)
            core=all(done(f'main_{m}_g0_s{s}_u256') for s in CFG['seeds'] for m in ['raw_wdro','local_shell'])
            if core:
                for m,s,g in primary:start_main(m,s,g,256)
            for target,previous in [(512,256),(1024,512)]:
                if all(done(f'main_{m}_g{g}_s{s}_u{previous}') for m,s,g in primary):
                    for m,s,g in primary:start_main(m,s,g,target)
            primary_done=all(done(f'main_{m}_g{g}_s{s}_u1024') for m,s,g in primary)
            hours=(time.time()-started_epoch)/3600
            # Additional generators and independent attacks are predeclared optional scale-up.
            if primary_done:
                if not (ART/'optional_schedule.json').exists():
                    atomic_json(ART/'optional_schedule.json',dict(extra_generators=hours<CFG['runtime']['extra_generator_admission_hours'],
                        attack=hours<CFG['runtime']['attack_admission_hours'],time=utc(),elapsed_hours=hours))
                optional=read_json(ART/'optional_schedule.json')
                if optional['extra_generators']:
                    for g in [1,2]:
                        launch(f'cvae_g{g}','cvae.py',['--generator',g],compute=True,gpu_allowed=True)
                        if done(f'cvae_g{g}'):launch(f'geometry_g{g}','pools.py',['--generator',g],compute=True)
                    for m,s,g in extra:
                        for target,prev in [(256,0),(512,256),(1024,512)]:
                            if not prev or done(f'main_{m}_g{g}_s{s}_u{prev}'):start_main(m,s,g,target)
                if optional['attack']:
                    launch('attack_geometry','attack.py',['--prepare'])
                    if done('attack_geometry') and not any(k.endswith('_reference') for k in active):launch('attack_raw32_reference','oracle.py',['attack_raw32'])
                    if done('attack_raw32_reference'):launch('attack_evaluation','attack.py',['--evaluate'],compute=True)
                extra_done=not optional['extra_generators'] or all(done(f'main_{m}_g{g}_s{s}_u1024') for m,s,g in extra)
                attack_done=not optional['attack'] or done('attack_evaluation')
                if extra_done and attack_done and not active:
                    mark_done('pipeline',status='completed',started_at=started,completed_at=utc(),optional=optional)
                    report(final=True);return
        if changed and ready:report();last_report=time.time()
        if ready and time.time()-last_report>=3600:report();last_report=time.time()
        try:snap=gpu_snapshot()
        except Exception as exc:print('GPU snapshot',repr(exc),flush=True)
        atomic_json(ART/'status.json',dict(time=utc(),pipeline_pid=os.getpid(),started_at=started,
            active={k:dict(pid=v['process'].pid,device=v['device'],compute=v['compute'],started=v['started'],command=v['command']) for k,v in active.items()},
            failures=failures,gpus=snap,completed_markers=len(list((ART/'done').glob('*.json'))),implementation_ready=ready))
        if failures and not active and ready:
            report(final=True);atomic_json(ART/'pipeline_blocked.json',dict(time=utc(),failures=failures));return
        time.sleep(10)

if __name__=='__main__':main()

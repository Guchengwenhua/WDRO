"""Persistent DAG queue, fresh GPU admission, per-job locks and resumable checkpoints."""
from exp_common import *
import fcntl,subprocess,shutil,signal

def jobs():
    out=[]
    def add(name,args,deps=(),gpu=True,expect=None):out.append(dict(id=name,args=args,deps=list(deps),gpu=gpu,status='pending',attempts=0,expect=expect))
    add('data',['prepare.py'],gpu=False,expect=str(ART/'done/data.json'))
    from generator import configs
    tunes=[]
    for c in configs():
        name='tune_'+c['key'];tunes.append(name);add(name,['generator.py','--phase','tune','--key',c['key'],'--seed','301'],['data'],expect=str(ART/'generators'/f"gen_tune_{c['key']}_s301"/'done.json'))
    add('select',['generator.py','--select'],tunes,gpu=False,expect=str(ART/'generator_selected.json'))
    for kind in ['set','graph']:
        for seed in [301,302,303]:add(f'formal_{kind}_{seed}',['generator.py','--phase','formal','--kind',kind,'--seed',str(seed)],['select'])
    for method in ['feature_gate','laplacian','tile_raw']:
        add('pool_'+method,['pools.py',method],['data'],gpu=False,expect=str(ART/'done'/f'{method}_geometry.json'))
        add('ref_'+method,['oracle.py',method],['pool_'+method],gpu=False,expect=str(ART/'done'/f'{method}_reference.json'))
    for method in ['set_dae','graph_dae','graph_tile']:
        for g in ([301,302,303] if method=='graph_tile' else [301]):
            kind='set' if method=='set_dae' else 'graph';name=f'{method}_g{g}'
            add('pool_'+name,['pools.py',method,'--generator',str(g)],[f'formal_{kind}_{g}'],expect=str(ART/'done'/f'{name}_geometry.json'))
            add('ref_'+name,['oracle.py',name],['pool_'+name],gpu=False,expect=str(ART/'done'/f'{name}_reference.json'))
    add('attack_geometry',['attack.py','--prepare'],['data'],gpu=False,expect=str(ART/'done/attack_geometry.json'))
    add('attack_reference',['oracle.py','attack32'],['attack_geometry'],gpu=False,expect=str(ART/'done/attack32_reference.json'))
    final=[]
    # Budgets outermost so all available methods reach an early comparison first.
    for budget in CFG['budgets']:
        for method in CFG['methods']:
            for g in ([301,302,303] if method=='graph_tile' else [301]):
                for seed in CFG['seeds']:
                    base=f'{method}_g{g}_s{seed}';name=f'main_{base}_u{budget}'
                    if budget==256:
                        if method in ['erm_a','raw_wdro','vae_shell_wdro']:deps=['data']
                        else:deps=['ref_'+(f'{method}_g{g}' if method in ['set_dae','graph_dae','graph_tile'] else method)]
                    else:deps=[f'main_{base}_u{256 if budget==1024 else 1024}']
                    add(name,['rest_train.py','--method',method,'--seed',str(seed),'--generator',str(g),'--target',str(budget),'--device','cuda'],deps,expect=str(ART/'done'/f'{name}.json'))
                    if budget==2048:
                        attack=f'attack_{base}';add(attack,['attack.py','--method',method,'--seed',str(seed),'--generator',str(g)],[name,'attack_reference'],expect=str(ART/'done'/f'{attack}.json'));final.append(attack)
    add('generator_diagnostics',['diagnostics.py'],[f'formal_{k}_{s}' for k in ['set','graph'] for s in [301,302,303]],expect=str(ART/'done/generator_diagnostics.json'))
    add('final_report',['report.py','--final'],final+['generator_diagnostics'],gpu=False,expect=str(ART/'completion.json'))
    return out

def proc_start(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except (FileNotFoundError,ProcessLookupError):return None

def alive(job):
    if not job.get('pid') or proc_start(job['pid'])!=job.get('starttime'):return False
    try:return Path(f"/proc/{job['pid']}/stat").read_text().split()[2]!='Z'
    except FileNotFoundError:return False

def gpus():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(a),uuid=b.strip(),free=int(c),util=int(d)) for a,b,c,d in [s.split(',') for s in raw.strip().splitlines()]]

def estimate(queue):
    progresses=[]
    for p in (ART/'progress').glob('*.json'):
        try:progresses.append(read_json(p))
        except (ValueError,OSError):pass
    rest=[p['seconds']/p['update'] for p in progresses if p.get('phase')=='main' and p.get('method')!='erm_a' and p.get('update',0)>0]
    gen=[p['seconds_per_step'] for p in progresses if p.get('seconds_per_step')]
    rest_rate=float(np.median(rest)) if rest else None;gen_rate=float(np.median(gen)) if gen else None
    remaining_rest=0;remaining_gen=0
    byid={j['id']:j for j in queue}
    for method in CFG['methods']:
        for g in ([301,302,303] if method=='graph_tile' else [301]):
            for s in CFG['seeds']:
                prefix=f'main_{method}_g{g}_s{s}_u';ps=[p.get('update',0) for p in progresses if p.get('method')==method and p.get('generator')==g and p.get('seed')==s and p.get('phase')=='main']
                reached=max(ps,default=0)
                if byid[prefix+'2048']['status']=='done':reached=2048
                remaining_rest+=2048-reached
    for j in queue:
        if j['status']=='done' or not j['id'].startswith(('tune_','formal_')):continue
        if j['id'].startswith('tune_'):
            key=j['id'][5:]
            path=ART/'progress'/f'gen_tune_{key}_s301.json';step=read_json(path).get('step',0) if path.exists() else 0;remaining_gen+=7500-step
        else:
            _,kind,seed=j['id'].split('_');step=0
            for p in (ART/'progress').glob(f'gen_formal_{kind}_*_s{seed}.json'):step=max(step,read_json(p).get('step',0))
            remaining_gen+=22500-step
    pilot=read_json(ART/'tests/gpu_pilot.json') if (ART/'tests/gpu_pilot.json').exists() else {}
    gpu_seconds=(remaining_rest*(rest_rate or pilot.get('rest_wdro_seconds_per_update',2.0))+remaining_gen*(gen_rate or pilot.get('graph_w160_seconds_per_step',.1)))
    # Reference queues/test overhead and sequential dependencies widen the interval.
    ideal=gpu_seconds/8
    return dict(time=utc(),measured_rest_seconds_per_update=rest_rate,measured_generator_seconds_per_step=gen_rate,remaining_rest_updates=remaining_rest,remaining_generator_updates=remaining_gen,compute_only_8gpu_hours=ideal/3600,estimated_remaining_hours_low=ideal/3600*1.25,estimated_remaining_hours_high=ideal/3600*2.5+1,not_guaranteed='includes rough test/reference slack; shared load and unavailable GPUs may delay',progress_files=len(progresses))

def main():
    ART.mkdir(parents=True,exist_ok=True);(ART/'logs').mkdir(exist_ok=True)
    with (ART/'supervisor.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Supervisor already active')
        qpath=ART/'queue.json';queue=read_json(qpath)['jobs'] if qpath.exists() else jobs();children={};stop=False;last_report=0
        def interrupt(sig,frame):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        for j in queue:
            if j['status']=='running' and not alive(j):j['status']='pending'
        while not stop:
            now=time.time()
            for j in queue:
                if j['status']!='running':continue
                p=children.get(j['id']);rc=p.poll() if p else (None if alive(j) else -99)
                if rc is None:continue
                expect=j.get('expect');ok=rc==0 and (not expect or Path(expect).exists())
                # Orphaned completed worker recovery relies on its atomic completion artifact.
                if rc==-99 and expect and Path(expect).exists():ok=True
                j.update(status='done' if ok else ('pending' if j['attempts']<CFG['runtime']['max_attempts'] else 'failed'),finished_at=utc(),returncode=rc)
                children.pop(j['id'],None);print('FINISH',j['id'],j['status'],rc,flush=True)
            lookup={j['id']:j for j in queue}
            for j in queue:
                if j['status']=='pending' and any(lookup[d]['status'] in ['failed','blocked'] for d in j['deps']):j['status']='blocked'
            running=[j for j in queue if j['status']=='running'];gpu_busy={j.get('gpu_uuid') for j in running if j['gpu']};cpu_busy=sum(not j['gpu'] for j in running)
            try:devices=gpus()
            except Exception as exc:devices=[];append_json(ART/'admission_errors.jsonl',dict(time=utc(),error=repr(exc)))
            free=shutil.disk_usage(ART).free;diskok=free>10*1024**3
            for j in queue:
                if j['status']!='pending' or not all(lookup[d]['status']=='done' for d in j['deps']) or not diskok:continue
                if j['gpu']:
                    avail=[g for g in devices if g['uuid'] not in gpu_busy and g['free']>=(CFG['runtime']['generator_min_free_mb'] if j['args'][0] in ['generator.py','pools.py','diagnostics.py'] else CFG['runtime']['gpu_min_free_mb'])]
                    if not avail:continue
                    gpu=min(avail,key=lambda g:(g['util'],-g['free']));gpu_busy.add(gpu['uuid']);j['gpu_uuid']=gpu['uuid'];j['gpu_index']=gpu['index']
                else:
                    if cpu_busy>=1:continue
                    cpu_busy+=1;j['gpu_uuid']=None
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=j['gpu_uuid'] or '';env['NET_RUN_RESOURCE']='1'
                if j['id'].startswith('formal_'):
                    _,kind,seed=j['id'].split('_');key=read_json(ART/'generator_selected.json')['selected'][kind]['key']
                    j['expect']=str(ART/'done'/f'gen_formal_{key}_s{seed}.json')
                log=(ART/'logs'/f"{j['id']}.log").open('a');p=subprocess.Popen([CFG['python'],'worker.py',j['id'],*j['args']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
                j.update(status='running',pid=p.pid,starttime=proc_start(p.pid),started_at=utc(),attempts=j['attempts']+1);children[j['id']]=p
                append_json(ART/'admissions.jsonl',dict(time=utc(),job=j['id'],gpu=j.get('gpu_index'),gpu_snapshot=devices,free_disk_bytes=free,pid=p.pid,starttime=j['starttime']))
                print('START',j['id'],'gpu',j.get('gpu_index'),flush=True)
            atomic_json(qpath,dict(time=utc(),pid=os.getpid(),starttime=proc_start(os.getpid()),jobs=queue))
            atomic_json(ART/'heartbeat.json',dict(time=utc(),pid=os.getpid(),counts={s:sum(j['status']==s for j in queue) for s in ['pending','running','done','failed','blocked']},gpus=devices,disk_free_bytes=free))
            atomic_json(ART/'runtime_estimate.json',estimate(queue))
            if now-last_report>120 and not (ART/'completion.json').exists() and lookup['final_report']['status'] not in ['running','done']:
                subprocess.run([CFG['python'],'report.py'],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});last_report=now
            if all(j['status'] in ['done','failed','blocked'] for j in queue):
                atomic_json(ART/'supervisor_exit.json',dict(time=utc(),success=all(j['status']=='done' for j in queue),failed=[j['id'] for j in queue if j['status']!='done']));break
            time.sleep(15)
        if stop:atomic_json(ART/'supervisor_paused.json',dict(time=utc(),reason='signal; children keep running; restart adopts exact PID/starttime'))
if __name__=='__main__':main()

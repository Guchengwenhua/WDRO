"""Capacity-admitted dependent experiment queue; all state is local and resumable."""
import fcntl, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'

def read(p): return json.loads(p.read_text())
def write(p, obj):
    tmp = p.with_name(p.name+'.partial')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def identity(pid):
    try: s = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError: return None
    tail = s[s.rfind(')')+2:].split()
    if tail[0] == 'Z': return None
    return {'pid': pid, 'start_ticks': int(tail[19]), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

def gpu_sample():
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    result = {}
    for line in r.stdout.splitlines():
        i, free, util = map(int, line.split(',')); result[i] = {'free': free, 'util': util}
    return result

def event(row):
    with (ART/'events.jsonl').open('a') as f: f.write(json.dumps(dict(time=time.time(), **row))+'\n')

def summary():
    subprocess.run([sys.executable, str(HERE/'summarize.py')], cwd=HERE, check=False)

def main():
    lock = (ART/'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = read(HERE/'queue.json'); jobs = cfg['jobs']; byid = {j['id']:j for j in jobs}
    if (ART/'state.json').exists():
        for row in read(ART/'state.json').get('active', []):
            if identity(row['identity']['pid']) == row['identity']: raise RuntimeError('Prior child still alive: '+row['id'])
    def marker(job): return ART/job['relative_output']/'complete.json'
    pending = [j for j in jobs if not marker(j).exists()]
    failed = []; paused = []; gated = []; active = {}; retries = {}; samples = []; stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    while pending or active:
        changed = False
        for gpu, row in list(active.items()):
            code = row['process'].poll()
            if code is None: continue
            row['log'].close(); done = marker(byid[row['id']]).exists()
            event({'event':'exit', 'job':row['id'], 'exit_code':code, 'complete':done})
            if not done:
                receipt = {'id':row['id'], 'exit_code':code, 'log_path':row['log_path']}
                if code == 75:
                    paused.append(receipt)
                    if not stopping[0] and retries.get(row['id'],0) < 1:
                        retries[row['id']] = 1; pending.append(byid[row['id']])
                else: failed.append(receipt)
            del active[gpu]; changed = True
        if stopping[0]:
            for row in active.values():
                if not row.get('stop_sent'): row['process'].terminate(); row['stop_sent'] = True
            pending = []
        for job in list(pending):
            rejected = None
            for dep in job['depends']:
                p = marker(byid[dep])
                if p.exists() and not read(p).get('qualified', True): rejected = dep+' did not meet calibration gate'
                if any(f['id'] == dep for f in failed): rejected = dep+' failed'
            if rejected:
                gated.append({'id':job['id'],'reason':rejected}); pending.remove(job); event({'event':'gated','job':job['id'],'reason':rejected}); changed = True
        if changed: summary()
        try:
            sample = gpu_sample(); samples.append(sample); samples = samples[-3:]
        except Exception as error:
            event({'event':'capacity_sample_unavailable','message':str(error)}); samples = []; sample = {}
        if not stopping[0] and len(samples) == 3 and shutil.disk_usage(ART.resolve()).free > 20*2**30:
            for gpu in sorted(sample, key=lambda g:(-sample[g]['free'],g)):
                if len(active) >= cfg['max_concurrent']: break
                if gpu in active: continue
                if not all(s.get(gpu,{}).get('free',0) >= 16384 and s.get(gpu,{}).get('util',100) <= 35 for s in samples): continue
                job = next((j for j in pending if all(marker(byid[d]).exists() and read(marker(byid[d])).get('qualified',True) for d in j['depends'])), None)
                if job is None: continue
                pending.remove(job); log_path = ART/f"{job['id']}.attempt_{time.time_ns()}.log"; log = log_path.open('w')
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                cmd = [sys.executable,'-u','-B',str(HERE/'run_scale.py'),'--cohort',job['cohort'],'--seed',str(job['seed'])]
                if job['cohort'] != 'calibration': cmd += ['--family',job['family'],'--arm',job['arm']]
                process = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = {'id':job['id'], 'gpu':gpu, 'identity':identity(process.pid), 'process':process, 'log':log, 'log_path':str(log_path), 'started':time.time()}
                event({'event':'launch','job':job['id'],'gpu':gpu,'identity':active[gpu]['identity'],'admission_samples':samples,'command':cmd})
                print('launched',job['id'],'GPU',gpu,flush=True)
        # Only unavailable dependencies are gated; ready jobs may wait for GPU capacity.
        live_ids = {j['id'] for j in pending} | {row['id'] for row in active.values()}
        for job in list(pending):
            if any(not marker(byid[dep]).exists() and dep not in live_ids for dep in job['depends']):
                gated.append({'id':job['id'],'reason':'dependency incomplete after queue exits'})
                pending.remove(job)
        write(ART/'state.json', {'status':'stopping' if stopping[0] else 'running' if active else 'waiting_for_capacity', 'time':time.time(),'identity':identity(os.getpid()),
                               'total':len(jobs),'completed':[j['id'] for j in jobs if marker(j).exists()], 'pending':[j['id'] for j in pending], 'failed':failed,'paused':paused,'gated':gated,
                               'active':[{k:v for k,v in row.items() if k not in ['process','log']} for row in active.values()], 'gpu_sample':sample})
        if pending or active: time.sleep(20)
    done = [j['id'] for j in jobs if marker(j).exists()]
    status = 'complete' if len(done) == len(jobs) else 'complete_with_gates' if len(done)+len(gated) == len(jobs) and not failed else 'incomplete'
    write(ART/'state.json', {'status':status,'time':time.time(),'total':len(jobs),'completed':done,'failed':failed,'paused':paused,'gated':gated,'active':[],
                           'pending':[j['id'] for j in jobs if j['id'] not in done and j['id'] not in [g['id'] for g in gated]]})
    summary()

if __name__ == '__main__': main()

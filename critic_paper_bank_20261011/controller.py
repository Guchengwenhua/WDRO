"""Persistent capacity-admitted queue; no foreign process is signalled."""
import fcntl, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'

def write(path, value):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2)); os.replace(tmp, path)

def identity(pid):
    p = Path(f'/proc/{pid}/stat')
    if not p.exists(): return None
    s = p.read_text(); fields = s[s.rfind(')') + 2:].split()
    return {'pid': pid, 'start_ticks': int(fields[19]), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

def gpu_sample():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return {int(parts[0]): {'free': int(parts[1]), 'util': int(parts[2])} for line in result.stdout.splitlines() if len(parts := line.split(',')) == 3}

def main():
    lock = (ART / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    # Different families begin together; within each family each seed is paired.
    queues = {family: [(family, arm, seed) for seed in protocol['seeds'] for arm in arms] for family, arms in protocol['families'].items()}
    jobs = []
    while any(queues.values()):
        for q in queues.values():
            if q: jobs.append(q.pop(0))
    def name(job): return f'{job[0]}_{job[1]}_s{job[2]}'
    old = json.loads((ART / 'state.json').read_text()) if (ART / 'state.json').exists() else {}
    # A controller restart may not duplicate a detached child.
    for row in old.get('active', []):
        if identity(row['identity']['pid']) == row['identity']:
            raise RuntimeError('Previous child still alive; retain existing queue: ' + row['name'])
    pending = [j for j in jobs if not (ART / name(j) / 'complete.json').exists()]
    active = {}; failed = []; stopping = [False]; samples = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    while pending or active:
        for gpu, row in list(active.items()):
            code = row['process'].poll()
            if code is None: continue
            row['log'].close()
            complete = (ART / row['name'] / 'complete.json').exists()
            if not complete: failed.append({'name': row['name'], 'exit_code': code, 'log': row['log_path']})
            with (ART / 'events.jsonl').open('a') as f: f.write(json.dumps({'event': 'exit', 'name': row['name'], 'code': code, 'complete': complete, 'time': time.time()}) + '\n')
            del active[gpu]
            subprocess.run([sys.executable, str(HERE / 'summarize.py')], cwd=HERE, check=False)
        if stopping[0]:
            for row in active.values():
                if not row.get('stop_sent'): row['process'].terminate(); row['stop_sent'] = True
            pending = []
        try:
            sample = gpu_sample(); samples.append(sample); samples = samples[-3:]
        except Exception as error:
            sample = {}; samples = []; print('capacity sample unavailable:', error, flush=True)
        if len(samples) == 3 and not stopping[0] and shutil.disk_usage(ART.resolve()).free > 10 * 2**30:
            for gpu in sorted(sample, key=lambda g: (-sample[g]['free'], g)):
                if not pending or len(active) >= 3: break
                if gpu in active: continue
                if not all(s.get(gpu, {}).get('free', 0) >= 16384 and s.get(gpu, {}).get('util', 100) <= 35 for s in samples): continue
                job = pending.pop(0); task = name(job)
                log_path = ART / (task + f'.attempt_{time.time_ns()}.log'); log = log_path.open('w')
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                command = [sys.executable, '-u', '-B', str(HERE / 'run_experiment.py'), '--family', job[0], '--arm', job[1], '--seed', str(job[2])]
                process = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = {'process': process, 'log': log, 'name': task, 'identity': identity(process.pid), 'log_path': str(log_path), 'gpu': gpu, 'started': time.time()}
                with (ART / 'events.jsonl').open('a') as f: f.write(json.dumps({'event': 'launch', 'name': task, 'gpu': gpu, 'identity': active[gpu]['identity'], 'admission_samples': samples, 'command': command, 'time': time.time()}) + '\n')
                print('launched', task, 'GPU', gpu, flush=True)
        write(ART / 'state.json', {'status': 'stopping' if stopping[0] else 'running' if active else 'waiting_for_capacity', 'time': time.time(), 'identity': identity(os.getpid()), 'total': len(jobs),
                'completed': [name(j) for j in jobs if (ART / name(j) / 'complete.json').exists()], 'pending': [name(j) for j in pending], 'failed': failed,
                'active': [{k: v for k, v in row.items() if k not in ['process', 'log']} for row in active.values()], 'gpu_sample': sample})
        if pending or active: time.sleep(20)
    subprocess.run([sys.executable, str(HERE / 'summarize.py')], cwd=HERE, check=True)
    completed = [name(j) for j in jobs if (ART / name(j) / 'complete.json').exists()]
    write(ART / 'state.json', {'status': 'complete' if len(completed) == len(jobs) else 'incomplete', 'time': time.time(), 'total': len(jobs), 'completed': completed, 'failed': failed, 'active': [], 'pending': [name(j) for j in jobs if name(j) not in completed]})

if __name__ == '__main__': main()

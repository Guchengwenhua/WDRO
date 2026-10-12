"""Persistent setup gate: finish/adopt engineering checks, then run the frozen queue."""
import json,os,subprocess,time
from pathlib import Path
import ops as o
ROOT=Path(__file__).resolve().parent;OPS=ROOT/'artifacts/operations'
def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def status(stage,**details):
    o.atomic(OPS/'bootstrap.json',dict(stage=stage,time=time.time(),identity=o.ident(os.getpid()),**details))
    o.atomic(OPS/'queue.json',dict(status='setup_'+stage,complete=[],total=104,running={},time=time.time(),**details))
def run(name,*args):
    with (OPS/(name+'.log')).open('a') as log:
        result=subprocess.run(['bash',str(ROOT/'run_env.sh'),str(ROOT/(name+'.py')),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(name+' failed: '+str(result.returncode))
def main():
    lock=o.acquire_scheduler_lock(OPS/'bootstrap.lock')
    if not o.verify_source_freeze(ROOT):raise RuntimeError('Source freeze missing')
    eta_stream=(OPS/'setup_eta.log').open('a')
    eta=subprocess.Popen(['bash',str(ROOT/'run_env.sh'),str(ROOT/'eta.py'),'--watch'],cwd=ROOT,stdout=eta_stream,stderr=subprocess.STDOUT)
    def stop_eta():
        if eta.poll() is None:
            eta.terminate()
            try:eta.wait(timeout=15)
            except subprocess.TimeoutExpired:eta.kill();eta.wait()
        eta_stream.close()
    try:
        # Existing tool-launched tests may finish normally; if their coordinator
        # disappears, wait for owned children, then resume/replay bounded checks.
        prior=read(OPS/'existing_engineering.json',{'identities':[]})
        while True:
            alive=[x for x in prior['identities'] if o.same_identity(o.ident(x['pid']),x)]
            workers=[]
            for mode in ['gpu_pilot','cpu_resume','cpu_noeval']:
                workers+=o.discover_live_workers(ROOT/'artifacts'/mode)
            if not alive and not workers:break
            status('waiting_existing_engineering',coordinators=alive,workers=workers)
            time.sleep(10)
        status('cpu_recovery');run('engineering','cpu')
        status('gpu_calibration');run('engineering','gpu')
        status('allocator_cap');run('allocator_probe')
        files=['artifacts/cpu_resume/acceptance.json','artifacts/gpu_pilot/acceptance.json','artifacts/gpu_pilot/audit.json','artifacts/allocator_probe/acceptance.json']
        for name in files:
            if read(ROOT/name,{}).get('status')!='PASS':raise RuntimeError('Gate did not pass: '+name)
        for mode in ['gpu_pilot','cpu_resume','cpu_noeval','allocator_probe']:
            for name in ['runner.py','engine.py','policies.py','env_geometry.py']:
                if o.sha256(ROOT/name)!=o.sha256(ROOT/'artifacts'/mode/name):raise RuntimeError('Engineering training code changed: '+mode+'/'+name)
        if not o.verify_source_freeze(ROOT):raise RuntimeError('Source freeze changed')
        o.atomic(OPS/'start_gate.json',dict(status='PASS',time=time.time(),checks_sha256={n:o.sha256(ROOT/n) for n in files},
            protocol_sha256=o.sha256(ROOT/'protocol.json'),source_freeze_sha256=o.sha256(OPS/'source_freeze.json'),snapshot=o.snapshot()))
        status('passed');lock.close();stop_eta()
        return subprocess.run(['bash',str(ROOT/'run_session.sh')],cwd=ROOT).returncode
    except BaseException as exc:
        status('failed',error=repr(exc))
        o.atomic(OPS/'delivery.json',dict(status='NEEDS_REVIEW',stage='engineering_gate',error=repr(exc),time=time.time()))
        raise
    finally:stop_eta();lock.close()
if __name__=='__main__':raise SystemExit(main())

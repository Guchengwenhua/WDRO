"""Tmux controller for the complete authorized external-test extension."""
import concurrent.futures
import shlex
import subprocess
import time
from common import *

def run(script, *args):
    command=['bash',str(ENV),PYTHON,str(BASE/script),*map(str,args)]
    logpath=BASE/'logs'/f'{Path(script).stem}_pipeline.log'
    with logpath.open('a') as log:
        log.write('\n'+now()+' '+shlex.join(command)+'\n')
        log.flush()
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=BASE)
        while proc.poll() is None:
            write(BASE/'logs/pipeline'/f'{Path(script).stem}.json',dict(status='running',pid=proc.pid,at=now(),command=command,log=str(logpath)))
            time.sleep(3)
        result=dict(status='pass' if proc.returncode==0 else 'failed',pid=proc.pid,at=now(),command=command,log=str(logpath),exit_code=proc.returncode)
        write(BASE/'logs/pipeline'/f'{Path(script).stem}.json',result)
        if proc.returncode:
            raise RuntimeError(f'{script} exited {proc.returncode}: {logpath}')
        return result

def main():
    with lease('pipeline'):
        write(BASE/'logs/pipeline/launch.json',dict(at=now(),pid=os.getpid(),command=shlex.join(['bash',str(ENV),PYTHON,str(BASE/'pipeline.py')]),
            recovery='Rerun this same pipeline in the documented tmux session after checking existing process identities and locks.',
            tmux_session=os.environ.get('WDRO_TMUX_SESSION'),socket=str(BASE/'logs/public_external.sock'),
            smoke=artifact(BASE/'logs/evaluation_smoke/audit.json'),plan=artifact(BASE/'docs/EXPERIMENT_PLAN.md')))
        run('prepare.py','--source-manifest',BASE/'sources/contests/designs_manifest.json','--source-manifest',BASE/'sources/real_designs/design_manifest.json','--workers','12')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,'reference.py','--workers','24'),pool.submit(run,'audit_inputs.py')]
            for f in futures:
                f.result()
        run('run_queue.py','--slots-per-gpu','4')
        assert (BASE/'summarize.py').exists(), 'statistics implementation not ready'
        run('summarize.py')
        write(BASE/'logs/pipeline/completion.json',dict(status='pass',at=now(),queue=artifact(BASE/'logs/queue/completion.json'),summary=artifact(BASE/'summary.json')))

if __name__=='__main__':
    main()

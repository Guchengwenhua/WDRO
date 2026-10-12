"""Keep the detached queue alive and publish its terminal report/status."""
import fcntl
import subprocess
import time
from exp_common import *


def queue_alive(pid):
    try:
        base=Path(f'/proc/{pid}')
        command=base.joinpath('cmdline').read_bytes().split(b'\0')
        return (base.joinpath('cwd').resolve()==ROOT
                and any(Path(os.fsdecode(x)).name=='run_all.py' for x in command if x)
                and base.joinpath('stat').read_text().split(')')[-1].strip().split()[0]!='Z')
    except (OSError,ValueError):
        return False


def main():
    lock=(ART/'supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    restarts=0
    while True:
        completed=(ART/'done'/'pipeline.json').exists()
        blocked=(ART/'pipeline_blocked.json').exists()
        if completed or blocked:
            with (ART/'logs'/'report.log').open('a') as log:
                result=subprocess.run([CFG['python'],str(ROOT/'report.py'),'--final'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status=read_json(ART/'status.json')
            status.update(time=utc(),state='completed' if completed else 'blocked',active={},report_exit_code=result.returncode)
            atomic_json(ART/'status.json',status)
            cfg=read_json(ART/'config.executed.json');cfg.update(status=status['state'],finished_at=utc());atomic_json(ART/'config.executed.json',cfg)
            atomic_json(ART/'supervisor.json',dict(time=utc(),pid=os.getpid(),state=status['state'],restarts=restarts,report_exit_code=result.returncode))
            return
        launcher=read_json(ART/'launcher.json') if (ART/'launcher.json').exists() else {}
        pid=launcher.get('pid',0)
        if not queue_alive(pid):
            if restarts>=6:
                atomic_json(ART/'supervisor.json',dict(time=utc(),pid=os.getpid(),state='restart_limit',restarts=restarts));return
            with (ART/'launcher.log').open('a') as log:
                child=subprocess.Popen([CFG['python'],'-u',str(ROOT/'run_all.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            restarts+=1;pid=child.pid
            append_json(ART/'supervisor_events.jsonl',dict(time=utc(),event='restart_queue',pid=pid,restart=restarts))
        atomic_json(ART/'supervisor.json',dict(time=utc(),pid=os.getpid(),state='watching',queue_pid=pid,restarts=restarts))
        time.sleep(30)


if __name__=='__main__':main()

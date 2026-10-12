"""Durable, duplicate-protected, recoverable command runner for the correction."""
import argparse,json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import JobLease,utc_now

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--name');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.run:
        m=json.loads(a.run.read_text());job=Path(m['job_root']);child=None
        with JobLease(job,run_id=m['name'],exact_command=m['runner_command'],resume_command=m['recovery_command'],
                      monitor_command=m['monitor_command'],expected_updates=1,physical_gpu_index=None,host_runtime_manifest=m['host_runtime']) as lease:
            with Path(m['log']).open('a') as log:
                child=subprocess.Popen(m['command'],cwd=m['cwd'],stdout=log,stderr=subprocess.STDOUT)
                atomic_write_json(job/'child.json',dict(at_utc=utc_now(),pid=child.pid,command=m['command']))
                try:
                    while child.poll() is None:
                        lease.update(update=0,status='running',detail=dict(child_pid=child.pid,log=m['log']));time.sleep(5)
                    atomic_write_json(job/'exit.json',dict(at_utc=utc_now(),child_pid=child.pid,exit_code=child.returncode))
                    if child.returncode:raise RuntimeError(f'child exit {child.returncode}; see {m["log"]}')
                finally:
                    if child.poll() is None:child.terminate();child.wait()
            lease.finish(update=1,status='complete')
        return
    assert a.name and a.name.replace('_','').isalnum() and a.command
    command=a.command[1:] if a.command[0]=='--' else a.command
    job=EXP/'logs/precision_v2/jobs'/a.name;job.mkdir(parents=True,exist_ok=True)
    socket=EXP/'tmp/precision_v2.sock';session='precision_'+a.name
    if subprocess.run(['tmux','-S',str(socket),'has-session','-t',session],capture_output=True).returncode==0:
        raise SystemExit('Existing persistent job; inspect heartbeat')
    pre=subprocess.run(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),
                        '--purpose',a.name,'--measured-peak-mib','4626','--safety-margin-mib','2048',
                        '--samples','10','--sample-interval-seconds','1',
                        '--output',str(ROOT/'manifests'/f'host_runtime_{a.name}_p{os.getpid()}_{time.time_ns()}.json')],capture_output=True,text=True)
    if pre.returncode:raise RuntimeError(pre.stdout+pre.stderr)
    host=json.loads(pre.stdout.strip().splitlines()[-1])
    runner=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__)), '--run',str(job/'launch.json')]
    recovery=shlex.join(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__)),'--name',a.name,'--',*command])
    manifest=dict(at_utc=utc_now(),name=a.name,job_root=str(job),command=command,cwd=str(CODEX),runner_command=shlex.join(runner),
                  recovery_command=recovery,monitor_command=f'cat {job}/heartbeat.json',host_runtime=host['manifest'],
                  session=session,socket=str(socket),socket_resolved=str(socket.resolve()),log=str(job/'output.log'),
                  log_resolved=str((job/'output.log').resolve()))
    atomic_write_json(job/'launch.json',manifest)
    shell=f'export WDRO_TMUX_SESSION={shlex.quote(session)}; exec {shlex.join(runner)} >> {shlex.quote(str(job/"runner.log"))} 2>&1'
    subprocess.run(['tmux','-S',str(socket),'new-session','-d','-s',session,shell],check=True)
    print(json.dumps(dict(status='launched',session=session,launch=str(job/'launch.json'),preflight=host)))
if __name__=='__main__':main()

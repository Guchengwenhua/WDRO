"""Stop dependent stage-6 work immediately if the early formal audit fails."""
import argparse,json,os,signal,sys,time,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import JobLease,utc_now
from src.rest import file_sha256
from stage6_pipeline import process_is_alive,gate

def own_process(pid,script):
    try:return str(ROOT/'scripts'/script).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False

def stop_dependents(reason):
    stopped=[];controller=EXP/'logs/stage6_pipeline/job.json'
    if controller.exists():
        pid=json.loads(controller.read_text())['pid']
        if process_is_alive(pid) and own_process(pid,'stage6_pipeline.py'):
            os.kill(pid,signal.SIGTERM);stopped.append(pid)
    for path in (EXP/'logs/jobs').glob('formal_wdro_*/job.json'):
        pid=json.loads(path.read_text())['pid']
        if process_is_alive(pid) and own_process(pid,'train_stage6.py'):
            os.kill(pid,signal.SIGTERM);stopped.append(pid)
    atomic_write_json(EXP/'logs/stage6_continuation/early_audit_failure.json',dict(status='blocked',at_utc=utc_now(),reason=reason,stop_requested_pids=stopped,
        recovery='Resolve the audit error and keep test locked before launch_stage6.py; training resumes from latest.'))

def observe(run_id):
    report=EXP/'logs/stage6_audit/runs'/f'{run_id}.json';job=EXP/'logs/stage6_audit/jobs'/run_id/'job.json'
    if report.exists():
        r=json.loads(report.read_text())
        if r['status']!='pass':return 'failed',r
        assert r['binding']['result_sha256']==file_sha256(EXP/'logs/stage6'/run_id/'result.json')
        return 'complete',r
    j=json.loads(job.read_text())
    if j['status']=='failed' or not process_is_alive(j['pid']):return 'failed',j
    assert own_process(j['pid'],'audit_stage6.py'),'audit PID identity changed'
    return 'running',j

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--once',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    gate();root=EXP/'logs/stage6_single_audit'/a.run_id/'watchdog';root.mkdir(parents=True,exist_ok=True)
    command=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__)),'--run-id',a.run_id]
    if a.launch:
        socket=EXP/'tmp/stage6_single_audit.sock';session='stage6_watch_'+a.run_id
        if subprocess.run(['tmux','-S',str(socket),'has-session','-t',session],capture_output=True).returncode==0:raise RuntimeError('watchdog already running')
        log=root/'watchdog.log';shell=f'export WDRO_TMUX_SESSION={shlex.quote(session)}; exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1'
        tmux=['tmux','-S',str(socket),'new-session','-d','-s',session,shell]
        atomic_write_json(root/'launch.json',dict(at_utc=utc_now(),session=session,socket=str(socket),socket_resolved=str(socket.resolve()),command=tmux,
            exact_command=shlex.join(tmux),log=str(log),log_resolved=str(log.resolve()),recovery_command=shlex.join(command+['--launch']),monitor_command=f'cat {root}/heartbeat.json'))
        subprocess.run(tmux,check=True);print('watchdog launched');return
    with JobLease(root,run_id='watch_'+a.run_id,exact_command=shlex.join(command),resume_command=shlex.join(command+['--launch']),
        monitor_command=f'cat {root}/heartbeat.json',expected_updates=2000,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        while True:
            state,detail=observe(a.run_id)
            if state=='failed':stop_dependents(detail);raise RuntimeError('early audit failed; stage6 dependent training stopped')
            if state=='complete':lease.finish(update=2000,status='complete');return
            cursor=EXP/'logs/stage6_audit/cursors'/f'{a.run_id}.json';u=json.loads(cursor.read_text())['completed_updates'] if cursor.exists() else 0
            lease.update(update=u,status='watching',detail={'audit_pid':detail['pid'],'main_pipeline_pid':json.loads((EXP/'logs/stage6_pipeline/job.json').read_text())['pid']})
            if a.once:
                atomic_write_json(root/'foreground_smoke.json',dict(status='pass',at_utc=utc_now(),audit_pid=detail['pid'],audit_update=u,dependent_processes_signaled=False))
                lease.finish(update=u,status='foreground_smoke_complete');return
            time.sleep(5)
if __name__=='__main__':main()

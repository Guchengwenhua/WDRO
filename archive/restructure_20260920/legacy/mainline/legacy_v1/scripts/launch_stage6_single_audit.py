"""One independently resumable formal audit, adoptable by the main pipeline."""
import argparse,json,subprocess,sys,shlex,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.job_control import utc_now
from stage6_pipeline import gate,preflight,observe,cleanup_dead_lock,process_is_alive

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--foreground-smoke',action='store_true');a=p.parse_args()
    gate();assert a.run_id in [t['run_id'] for t in json.loads((ROOT/'configs/stage6_plan_v1.json').read_text())['tasks'] if not t['reused_stage5']]
    result=EXP/'logs/stage6'/a.run_id/'result.json';r=json.loads(result.read_text());assert r['status']=='complete' and r['completed_updates']==2000
    job=EXP/'logs/stage6_single_audit'/a.run_id;job.mkdir(parents=True,exist_ok=True)
    report=EXP/'logs/stage6_audit/runs'/f'{a.run_id}.json'
    if report.exists():assert json.loads(report.read_text())['status']=='pass';print('audit already complete');return
    root=EXP/'logs/stage6_audit/jobs'/a.run_id;old=root/'job.json'
    if old.exists() and process_is_alive(json.loads(old.read_text())['pid']):raise RuntimeError('audit already running; inspect its heartbeat')
    runtime=preflight('stage6_single_audit_'+a.run_id);resource=observe();assert resource['disk'][1]['free_gib']>=100
    cleanup_dead_lock(root)
    command=['bash',str(ROOT/'scripts/stage3_env.sh'),'nice','-n','19',sys.executable,str(ROOT/'scripts/audit_stage6.py'),'--run-id',a.run_id,'--stability']
    cursor=EXP/'logs/stage6_audit/cursors'/f'{a.run_id}.json';log=job/('foreground.log' if a.foreground_smoke else 'audit.log')
    base=dict(at_utc=utc_now(),run_id=a.run_id,input_result_sha256=file_sha256(result),auditor_sha256=file_sha256(ROOT/'scripts/audit_stage6.py'),
        command=command,exact_command=shlex.join(command),preflight=runtime,resource=resource,log=str(log),log_resolved=str(log.resolve()),
        recovery_command=shlex.join(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__)),'--run-id',a.run_id]),
        monitor_command=f'cat {root}/heartbeat.json',main_pipeline_can_adopt=True)
    if a.foreground_smoke:
        with log.open('a') as out:proc=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
        atomic_write_json(job/'foreground_launch.json',dict(base,pid=proc.pid))
        while proc.poll() is None:
            if cursor.exists() and json.loads(cursor.read_text())['completed_updates']>=10:
                proc.send_signal(signal.SIGTERM);proc.wait(timeout=30);break
            time.sleep(2)
        state=json.loads(cursor.read_text());assert state['completed_updates']>=10
        assert not report.exists() and proc.returncode==-signal.SIGTERM
        atomic_write_json(job/'foreground_smoke.json',dict(status='pass',at_utc=utc_now(),cursor_update=state['completed_updates'],cursor_sha256=file_sha256(cursor),
            pid=proc.pid,exit_code=proc.returncode,complete_report_absent=True))
        print(json.dumps({'status':'pass','cursor_update':state['completed_updates'],'report':str(job/'foreground_smoke.json')}));return
    assert json.loads((job/'foreground_smoke.json').read_text())['status']=='pass'
    # The wrapper launches exactly one audit; there is no second scheduler competing
    # to admit follow-up runs when the main controller reaches its audit phase.
    socket=EXP/'tmp/stage6_single_audit.sock';session='stage6_audit_'+a.run_id
    if subprocess.run(['tmux','-S',str(socket),'has-session','-t',session],capture_output=True).returncode==0:raise RuntimeError('tmux session already exists')
    shell=f'export WDRO_TMUX_SESSION={shlex.quote(session)}; exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1'
    launch=['tmux','-S',str(socket),'new-session','-d','-s',session,shell]
    atomic_write_json(job/'launch.json',dict(base,session=session,socket=str(socket),socket_resolved=str(socket.resolve()),tmux_command=launch,
        input_cursor_sha256=file_sha256(cursor),controller_command=shlex.join(launch)))
    subprocess.run(launch,check=True);print(json.dumps({'status':'launched','session':session,'launch':str(job/'launch.json')}))
if __name__=='__main__':main()

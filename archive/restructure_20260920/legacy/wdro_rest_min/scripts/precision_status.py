"""Read current correction jobs, model cursors and live host capacity."""
import json,sys,subprocess,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments'
NEW=CODEX/'revisions/reference_precision_v2/Codex';NEXP=NEW/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json

def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None
def alive(pid):
    try:
        return int(pid)>0 and Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
    except (OSError,TypeError,ValueError):return False

def queue_tasks(stage, queue_id):
    q=read(NEXP/f'logs/stage{stage}_queue'/queue_id/'queue.json') or {}
    rows=[]
    for t in q.get('tasks',[]):
        hb=read(NEXP/'logs/jobs'/t['run_id']/'heartbeat.json') or {}
        ck=NEXP/f'checkpoints/stage{stage}'/t['run_id']
        rows.append(dict(run_id=t['run_id'],status=t['status'],gpu=t.get('physical_gpu_index'),pid=t.get('pid'),
            process_alive=alive(t.get('pid')),updates=hb.get('completed_updates',0),last_heartbeat=hb.get('last_update_utc'),
            latest_exists=(ck/'latest.pt').exists(),best_exists=(ck/'best.pt').exists()))
    return rows

def validation_tasks():
    plan=read(NEXP/'logs/stage5_pipeline/all_tasks.json') or {}
    rows=[]
    for task in plan.get('tasks',[]):
        rid=task['run_id'];root=NEXP/'logs/jobs'/rid
        hb=read(root/'heartbeat.json') or {};job=read(root/'job.json') or {}
        result=read(NEXP/'logs/stage5'/rid/'result.json') or {}
        live=alive(job.get('pid'))
        complete=result.get('status')=='complete' and result.get('completed_updates')==2000
        audit=read(NEXP/'logs/stage5_audit/runs'/f'{rid}.json') or {}
        audit_hb=read(NEXP/'logs/stage5_audit/jobs'/rid/'heartbeat.json') or {}
        validation=read(NEXP/'logs/stage5_best_validation'/f'{rid}.json') or {}
        rows.append(dict(run_id=rid,status='complete' if complete else 'running' if live else 'resume_required',
            gpu=job.get('physical_gpu_index'),pid=job.get('pid'),process_alive=live,
            updates=hb.get('completed_updates',0),update_source='heartbeat; may exceed the last atomic checkpoint',
            last_heartbeat=hb.get('last_update_utc'),full_audit_passed=audit.get('status')=='pass',
            audit_updates=audit_hb.get('completed_updates',0),audit_pid=audit_hb.get('pid'),
            audit_process_alive=alive(audit_hb.get('pid')),audit_last_heartbeat=audit_hb.get('last_update_utc'),
            best_validation_passed=validation.get('status')=='pass' and validation.get('exact_metric_equality') is True))
    return rows

def stage6_tasks():
    plan=read(NEW/'wdro_rest_min/configs/stage6_plan_v1.json') or {}
    rows=[]
    for task in plan.get('tasks',[]):
        rid=task['run_id'];result=read(task['result']) or {}
        hb=read(NEXP/'logs/jobs'/rid/'heartbeat.json') or {}
        job=read(NEXP/'logs/jobs'/rid/'job.json') or {}
        complete=result.get('status')=='complete' and result.get('completed_updates')==2000
        live=alive(job.get('pid'))
        rows.append(dict(run_id=rid,degree=task['degree'],seed=task['seed'],reused_stage5=task['reused_stage5'],
            status='complete' if complete else 'running' if live else 'pending_or_resume_required',
            gpu=job.get('physical_gpu_index'),pid=job.get('pid'),process_alive=live,
            updates=2000 if complete else hb.get('completed_updates',0),last_heartbeat=hb.get('last_update_utc')))
    return rows

def main():
    now=datetime.now(timezone.utc);jobs={}
    for name in sorted(p.name for p in (EXP/'logs/precision_v2/jobs').iterdir() if p.is_dir()):
        p=EXP/'logs/precision_v2/jobs'/name
        launch=read(p/'launch.json');hb=read(p/'heartbeat.json');ex=read(p/'exit.json')
        current_exit=bool(ex and launch and ex['at_utc']>=launch['at_utc'])
        jobs[name]=dict(launch=launch,heartbeat=hb,exit=ex,exit_belongs_to_current_launch=current_exit,
                       process_alive=alive((hb or {}).get('pid')))
    tasks=queue_tasks(2,'stage2_erm_formal_v1')
    baselines=queue_tasks(3,'stage3_baselines_formal_v1')
    wdro_validation=validation_tasks()
    wdro_multiseed=stage6_tasks()
    completed_batches={p.parent.name:dict(heartbeat=read(p),result=read(p.parent/'result.json'),
        process_alive=alive((read(p) or {}).get('pid'))) for p in (NEXP/'logs/stage5_completed_batches').glob('*/heartbeat.json')}
    pipelines={f'stage{s}':read(NEXP/f'logs/stage{s}_pipeline/heartbeat.json') for s in (3,4,5,6)}
    early_root=NEXP/'logs/stage3_completed_audit'
    early_heartbeat=read(early_root/'heartbeat.json')
    early_result=read(early_root/'result.json')
    early_active=read(early_root/'active_input.json')
    early_cursor=read(early_active['cursor']) if early_active and 'cursor' in early_active else None
    early_audit=dict(heartbeat=early_heartbeat,result=early_result,active=early_active,
        process_alive=alive((early_heartbeat or {}).get('pid')),
        cursor_completed_updates=(early_cursor or {}).get('completed_updates'),
        failure=read(early_root/'failure.json'),handoff=read(early_root/'handoff.json'))
    test_complete=sum((read(p) or {}).get('status')=='complete' for p in (NEXP/'logs/stage6_test').glob('*/*/result.json'))
    g=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    data=dict(at_utc=now.isoformat(),stage=6,status='DOING',jobs=jobs,erm_tasks=tasks,baseline_tasks=baselines,wdro_validation_tasks=wdro_validation,
        wdro_multiseed_tasks=wdro_multiseed,completed_validation_batches=completed_batches,
        pipelines=pipelines,baseline_completed_audit=early_audit,corrected_test_cells_complete=test_complete,
        controllers={n:read(NEXP/'logs'/n/'heartbeat.json') for n in ('precision_replay','precision_wdro_replay','precision_stage6_completion')},
        disk=[dict(logical_path=str(p),resolved_path=str(p.resolve()),free_gib=shutil.disk_usage(p).free/2**30) for p in (CODEX,EXP,NEW,NEXP)],
        gpu_csv=g.stdout,compute_processes_csv=processes.stdout,stage7_started=False)
    if (ROOT/'manifests/precision_v2_replay_completion.json').exists():
        assert read(ROOT/'manifests/precision_v2_replay_completion.json')['status']=='pass';data['status']='DONE'
    target=EXP/'logs/precision_v2/status'/f"{now.strftime('%Y%m%dT%H%M%S%fZ')}.json";atomic_write_json(target,data)
    atomic_write_json(target.parent/'latest.json',dict(path=str(target),resolved_path=str(target.resolve()),at_utc=data['at_utc']))
    def counts(rows):
        return dict(total=len(rows),complete=sum(t['status']=='complete' for t in rows),
                    running=sum(t['status']=='running' for t in rows),updates=sum(t['updates'] for t in rows))
    print(json.dumps(dict(at_utc=data['at_utc'],stage6=data['status'],manifest=str(target),erm=counts(tasks),baselines=counts(baselines),
        wdro_validation=dict(counts(wdro_validation),full_audits=sum(t['full_audit_passed'] for t in wdro_validation),
            audits_running=sum(t['audit_process_alive'] and not t['full_audit_passed'] for t in wdro_validation),
            best_validation_runs=sum(t['best_validation_passed'] for t in wdro_validation)),wdro_multiseed=counts(wdro_multiseed),
        corrected_test_cells_complete=test_complete,pipelines={k:v['status'] if v else None for k,v in pipelines.items()},
        controllers={k:v['status'] if v else None for k,v in data['controllers'].items()},disk=data['disk']),ensure_ascii=False))
    if early_heartbeat:
        print(json.dumps(dict(baseline_completed_audit=early_heartbeat['status'],
            audited_runs=len((early_result or {}).get('runs_passed',[])),
            process_alive=early_audit['process_alive'],cursor_completed_updates=early_audit['cursor_completed_updates']),ensure_ascii=False))
if __name__=='__main__':main()

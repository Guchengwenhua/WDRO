#!/usr/bin/env python3
"""Single stage-4 diagnostic, resource-aware recovery, then full CPU audit/export."""
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments'
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.job_control import JobLease,utc_now
from src.rest import file_sha256
from capture_host_runtime import gpu_sample,compute_processes,disk_snapshot
JOB=EXP/'logs/stage4_pipeline';RUN='stage4_stability_d30_s101_u500_v1'

def gate():
    contract=json.loads((ROOT/'manifests/stage4_training_contract_v1.json').read_text())
    for path,digest in contract['files'].items(): assert file_sha256(CODEX/path)==digest,path
    for path in [EXP/'logs/stage4_smoke/audit.json',
        EXP/'logs/stage4_audit/runs/stage4_smoke_q50_continuous_v1.json',
        EXP/'logs/stage4_audit/runs/stage4_smoke_zero_v1.json']:
        assert json.loads(path.read_text())['status']=='pass',path

def preflight(purpose):
    profile=json.loads((EXP/'logs/stage4_smoke/memory_profile.json').read_text())
    command=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose',purpose,
             '--measured-peak-mib',str(profile['measured_peak_reserved_mib']+1024),'--safety-margin-mib','2048']
    p=subprocess.run(command,capture_output=True,text=True)
    append_jsonl(JOB/'preflights.jsonl',{'at_utc':utc_now(),'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    if p.returncode: return None
    return json.loads(p.stdout.strip().splitlines()[-1])

def observe():
    return {'at_utc':utc_now(),'gpus':gpu_sample(),'compute_processes':compute_processes(),
            'disk':[disk_snapshot(CODEX),disk_snapshot(EXP)]}

def update_progress(message, action):
    import re
    p=CODEX/'01_WDRO-REST_项目进度.md';s=p.read_text();now=utc_now()
    s=re.sub(r'\| 更新时间 \| .*? \|',f'| 更新时间 | {now} |',s,count=1)
    s=re.sub(r'\| 实验状态 \| .*? \|',f'| 实验状态 | {message} |',s,count=1)
    s=re.sub(r'\| 下一动作 \| .*? \|',f'| 下一动作 | {action} |',s,count=1)
    s=s.replace('## 8. 变更记录\n\n',f'## 8. 变更记录\n\n### {now}\n\n- {message}。证据 `experiments/logs/stage4_pipeline/heartbeat.json`、`experiments/logs/stage4/{RUN}/`。下一步：{action}。\n\n')
    temp=p.with_suffix('.md.partial');temp.write_text(s);temp.replace(p)
    from precision_replay_pipeline import progress as publish
    publish(message,str(JOB/'heartbeat.json'))

def main():
    JOB.mkdir(parents=True,exist_ok=True);gate()
    command=shlex.join(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__))])
    if not (JOB/'launch.json').exists():
        parent=EXP/'logs/precision_replay/steps/wdro_diagnostic_train_audit_export_active.json'
        origin=json.loads(parent.read_text())
        assert origin['pid']==os.getpid()
        outer=Path('/remote-home/Guilin/DRO/Codex/experiments/logs/precision_v2/jobs/wdro_replay/launch.json')
        durable=json.loads(outer.read_text())
        atomic_write_json(JOB/'launch.json',dict(at_utc=utc_now(),pid=os.getpid(),session=os.environ.get('WDRO_TMUX_SESSION'),
            socket=durable['socket'],socket_resolved=durable['socket_resolved'],exact_command=origin['exact_command'],
            pipeline_command=command,recovery_command=durable['recovery_command'],monitor_command=f'cat {JOB}/heartbeat.json',
            log=origin['log'],log_resolved=origin['log_resolved'],preflight=origin['preflight'],
            parent_step_manifest=str(parent),outer_launch_manifest=str(outer)))
    with JobLease(JOB,run_id='stage4_pipeline',exact_command=command,
        resume_command=f'bash {ROOT}/scripts/stage3_env.sh {sys.executable} {ROOT}/scripts/launch_stage4.py',
        monitor_command=f'cat {JOB}/heartbeat.json',expected_updates=500,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        excluded={};attempt=0;last_progress=-1
        result_path=EXP/'logs/stage4'/RUN/'result.json'
        while True:
            if result_path.exists():
                result=json.loads(result_path.read_text())
                if result['status']=='complete' and result['completed_updates']==500: break
            gate();summary=preflight('stage4_stability_start_or_resume')
            if summary is None:
                lease.update(update=0,status='waiting_resources');time.sleep(30);continue
            manifest=json.loads(Path(summary['manifest']).read_text())
            eligible=[r for r in manifest['gpu_decisions'] if r['eligible'] and time.time()>excluded.get(r['index'],0)]
            if not eligible: time.sleep(30);continue
            gpu=max(eligible,key=lambda r:r['minimum_free_memory_mib'])['index']
            attempt+=1
            if attempt>5: raise RuntimeError('five recovery attempts exhausted; retain latest and failure evidence')
            args=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/train_wdro.py'),
                  '--degree','30','--seed','101','--scope','stability','--run-id',RUN,
                  '--physical-gpu-index',str(gpu),'--host-runtime-manifest',summary['manifest'],'--geosteiner-workers','3']
            if (EXP/'checkpoints/stage4'/RUN/'latest.pt').exists(): args+=['--resume']
            log=JOB/f'training_attempt_{attempt}_{int(time.time())}.log'
            with log.open('a') as out:
                process=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT)
                record={'at_utc':utc_now(),'pid':process.pid,'gpu':gpu,'attempt':attempt,'command':args,
                        'exact_command':shlex.join(args),'preflight':summary,'log':str(log),'log_resolved':str(log.resolve())}
                append_jsonl(JOB/'launches.jsonl',record)
                atomic_write_json(JOB/'active_training.json',record)
                lease.update(update=0,status='training',detail=record)
                saturated=[];stopping=False
                while process.poll() is None:
                    snapshot=observe();append_jsonl(JOB/'resource_monitor.jsonl',snapshot)
                    row=next(r for r in snapshot['gpus'] if r['index']==gpu)
                    saturated=(saturated+[row['utilization_gpu_percent']>=95])[-6:]
                    reason=None
                    if row['memory_free_mib']<2048:reason='free memory below safety margin'
                    if len(saturated)==6 and all(saturated):reason='sustained GPU saturation for six observations'
                    if snapshot['disk'][1]['free_gib']<100:reason='artifact disk below 100 GiB'
                    heartbeat=EXP/'logs/jobs'/RUN/'heartbeat.json'
                    if heartbeat.exists():
                        status=json.loads(heartbeat.read_text());u=status['completed_updates']
                        lease.update(update=u,status='training',detail={'gpu':gpu,'training_pid':process.pid,'training_heartbeat':str(heartbeat)})
                        if u//100>last_progress and u>=100:
                            last_progress=u//100
                            update_progress(f'阶段 4.3 WDRO 稳定性运行已达到 {u}/500 updates，仍待完整审计', '继续同一 run；通过完整预算/数据/LP 审计后导出，阶段 5 不启动')
                    if reason and not stopping:
                        append_jsonl(JOB/'recoveries.jsonl',{'at_utc':utc_now(),'reason':reason,'pid':process.pid,'gpu':gpu})
                        process.send_signal(signal.SIGTERM);stopping=True;excluded[gpu]=time.time()+600
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: pass
                code=process.returncode
            append_jsonl(JOB/'segments.jsonl',{'at_utc':utc_now(),'attempt':attempt,'exit_code':code,'gpu':gpu,'log':str(log)})
            if code:
                text=log.read_text()
                if 'out of memory' not in text.lower(): raise RuntimeError(f'trainer failed; inspect {log}')
                excluded[gpu]=time.time()+600
        update_progress('阶段 4.3 WDRO 连续目标 500 updates 已完成；正在全量输入、cost、lambda 和实际 HM 区间审计', '完成审计与表图/报告后验收阶段 4')
        preflight('stage4_full_cpu_audit')
        args=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/audit_stage4.py'),'--run-id',RUN,'--stability']
        with (JOB/'full_audit.log').open('a') as out:
            proc=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT)
            append_jsonl(JOB/'launches.jsonl',{'at_utc':utc_now(),'task':'audit','pid':proc.pid,'command':args,'log':str(JOB/'full_audit.log')})
            while proc.poll() is None:
                cursor=EXP/'logs/stage4_audit/cursors'/f'{RUN}.json'
                u=json.loads(cursor.read_text())['completed_updates'] if cursor.exists() else 0
                lease.update(update=u,status='auditing',detail={'audit_pid':proc.pid,'cursor':str(cursor)})
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
            if proc.returncode:raise RuntimeError('full stage4 audit failed')
        subprocess.run([sys.executable,str(ROOT/'scripts/run_precision_gpu_check.py'),'--script','verify_stage4_best_validation.py'],check=True)
        subprocess.run(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/export_stage4.py')],check=True)
        lease.finish(update=500,status='complete',detail={'stage':4,'stage5_started':False})

if __name__=='__main__': main()

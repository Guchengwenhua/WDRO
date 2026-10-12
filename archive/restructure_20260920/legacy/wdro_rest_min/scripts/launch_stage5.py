#!/usr/bin/env python3
"""Persistent tmux launcher with an in-Codex socket and fresh runtime gate."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments'
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from stage5_pipeline import gate,preflight,JOB

def main():
    JOB.mkdir(parents=True,exist_ok=True);gate()
    socket=EXP/'tmp/stage5.sock';session='wdro_stage5_v1'
    if subprocess.run(['tmux','-S',str(socket),'has-session','-t',session],capture_output=True).returncode==0:
        raise SystemExit('stage5 tmux session already exists; inspect heartbeat')
    heartbeat=JOB/'heartbeat.json'
    if heartbeat.exists() and json.loads(heartbeat.read_text())['status']=='complete':
        raise SystemExit('stage5 pipeline is already complete; no duplicate work started')
    summary=preflight('stage5_tmux_launch')
    if summary is None:raise SystemExit('fresh host-runtime preflight did not pass')
    runtime={'schema_version':1,'runtime_id':'stage5-'+utc_now(),'research_protocol_override':False,
         'python':sys.executable,'artifact_root':'Codex/experiments','allowed_gpu_indices':summary['eligible_gpu_indices'],
         'forbidden_gpu_indices':[],'allowed_gpu_idle_memory_mib_max':49140,'minimum_free_disk_gib_for_batch':100,
         'capacity_gate_manifest':summary['manifest']}
    config=ROOT/'configs'/('runtime_stage5_'+Path(summary['manifest']).stem.removeprefix('host_runtime_')+'.json')
    atomic_write_json(config,runtime,mode=0o444)
    envcheck=subprocess.run([sys.executable,str(ROOT/'scripts/run_preflight.py'),'--runtime-config',str(config),'--require-batch-ready'],capture_output=True,text=True)
    atomic_write_json(JOB/'environment_preflight.json',{'at_utc':utc_now(),'exit_code':envcheck.returncode,'stdout':envcheck.stdout,'stderr':envcheck.stderr})
    if envcheck.returncode:raise RuntimeError(envcheck.stderr+envcheck.stdout)
    log=JOB/'pipeline.log'
    args=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/stage5_pipeline.py')]
    shell=f'export WDRO_TMUX_SESSION={shlex.quote(session)}; exec {shlex.join(args)} >> {shlex.quote(str(log))} 2>&1'
    command=['tmux','-S',str(socket),'new-session','-d','-s',session,shell]
    recovery=shlex.join(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__))])
    record={'at_utc':utc_now(),'session':session,'socket':str(socket),'socket_resolved':str(socket.resolve()),
       'command':command,'exact_command':shlex.join(command),'pipeline_command':shlex.join(args),
       'recovery_command':recovery,'monitor_command':f'cat {JOB}/heartbeat.json','log':str(log),'log_resolved':str(log.resolve()),
       'host_runtime':summary,'environment_preflight':str(JOB/'environment_preflight.json'),
       'tasks':['three rho runs and full validation-only audit','six selected-rho ablations','best validation recompute','ablation table and budget plot','freeze configuration and stop at stage5'],
       'runtime_environment':{k:os.environ.get(k) for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','MPLCONFIGDIR','TRITON_CACHE_DIR']}}
    atomic_write_json(JOB/'launch.json',record)
    subprocess.run(command,check=True)
    print(json.dumps({'launched':session,'launch':str(JOB/'launch.json'),'monitor':record['monitor_command']}))

if __name__=='__main__':main()

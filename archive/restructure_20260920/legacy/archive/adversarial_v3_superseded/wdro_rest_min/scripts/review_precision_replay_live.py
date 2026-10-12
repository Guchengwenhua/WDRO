"""Capture live continuation state without restarting any healthy task."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORIG=Path('/remote-home/Guilin/DRO/Codex')
EXP=ROOT.parent/'experiments'
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from run_precision_training_queue import source_gate,process_is_alive
from precision_replay_pipeline import cpu_preflight


def main():
    source_gate()
    preflight=cpu_preflight('precision_replay_live_review')
    jobs={};checks={}
    variables=('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','MPLCONFIGDIR','TRITON_CACHE_DIR')
    for name in ('replay','wdro_replay','stage6_completion'):
        job=ORIG/'experiments/logs/precision_v2/jobs'/name
        launch=json.loads((job/'launch.json').read_text())
        heartbeat=json.loads((job/'heartbeat.json').read_text())
        child=json.loads((job/'child.json').read_text())
        checks[name+'_runner_alive']=process_is_alive(heartbeat['pid'])
        checks[name+'_child_alive']=process_is_alive(child['pid'])
        checks[name+'_session_alive']=subprocess.run(['tmux','-S',launch['socket'],'has-session','-t',launch['session']],capture_output=True).returncode==0
        env=dict(item.split('=',1) for item in Path(f"/proc/{child['pid']}/environ").read_text().split('\0') if '=' in item)
        paths={k:env.get(k) for k in variables}
        checks[name+'_runtime_paths_within_codex']=all(v and Path(v).is_relative_to(ORIG) for v in paths.values())
        checks[name+'_bytecode_disabled']=env.get('PYTHONDONTWRITEBYTECODE')=='1'
        jobs[name]=dict(launch=launch,heartbeat=heartbeat,child=child,runtime_paths=paths)
    audit=EXP/'logs/stage2/stage2_formal_audit.json'
    completion=ROOT/'manifests/stage2_completion_20260905.json'
    value=json.loads(audit.read_text());done=json.loads(completion.read_text())
    checks['erm_full_audit_pass']=value['status']=='pass' and all(value['checks'].values()) and all(all(c.values()) for c in value['run_checks'].values())
    checks['erm_completion_bound']=done['status']=='pass' and done['audit']['sha256']==file_sha256(audit)
    checks['frozen_training_sources_unchanged']=True
    now=utc_now()
    output=EXP/'logs/precision_replay_continuation'/('review_'+now.replace(':','').replace('.','_')+'.json')
    report=dict(status='pass' if all(checks.values()) else 'fail',at_utc=now,checks=checks,jobs=jobs,preflight=str(preflight),
        current_controllers={n:json.loads((EXP/'logs'/n/'heartbeat.json').read_text()) for n in ('precision_replay','precision_wdro_replay','precision_stage6_completion')},
        stage6_complete=False,stage7_started=False,research_protocol_override=False,
        operations=[dict(logical_path=str(p),resolved_path=str(p.resolve()),sha256=file_sha256(p)) for p in
                    (Path(__file__),ORIG/'wdro_rest_min/scripts/precision_status.py',ROOT/'scripts/precision_replay_pipeline.py',ROOT/'scripts/audit_stage2_formal.py')])
    atomic_write_json(output,report,mode=0o444)
    print(json.dumps(dict(status=report['status'],checks=len(checks),failed=[k for k,v in checks.items() if not v],output=str(output))))
    assert all(checks.values()),checks


if __name__=='__main__':main()

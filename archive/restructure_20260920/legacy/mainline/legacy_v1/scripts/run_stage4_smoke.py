#!/usr/bin/env python3
"""Foreground WDRO memory, resume, zero-radius and real HM-pair smoke."""
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CODEX=ROOT.parent
EXP=CODEX/'experiments'
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import utc_now
from src.rest import file_sha256
from run_stage3_smoke import assert_same

LOGS=EXP/'logs/stage4_smoke'

def launch(run_id, stop, resume=False, rho_mode='q50', measured_peak=None):
    capture=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose',f'stage4_{run_id}_stop{stop}']
    if measured_peak:
        capture+=['--measured-peak-mib',str(measured_peak+1024),'--safety-margin-mib','2048']
    captured=subprocess.run(capture,capture_output=True,text=True,check=True)
    summary=json.loads(captured.stdout.strip().splitlines()[-1])
    manifest=json.loads(Path(summary['manifest']).read_text())
    eligible=[r for r in manifest['gpu_decisions'] if r['eligible']]
    gpu=max(eligible,key=lambda r:r['minimum_free_memory_mib'])['index']
    command=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/train_wdro.py'),
             '--degree','30','--seed','101','--scope','stage4_smoke','--run-id',run_id,
             '--stop-after',str(stop),'--rho-mode',rho_mode,'--physical-gpu-index',str(gpu),
             '--host-runtime-manifest',summary['manifest'],'--geosteiner-workers','3']
    if resume: command+=['--resume']
    LOGS.mkdir(parents=True,exist_ok=True)
    log=LOGS/f'{run_id}_to{stop}.log'
    with log.open('a') as out:
        p=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
        record={'at_utc':utc_now(),'run_id':run_id,'pid':p.pid,'gpu':gpu,'command':command,
                'exact_command':shlex.join(command),'preflight':summary,'log':str(log),'log_resolved':str(log.resolve())}
        append_jsonl(LOGS/'launches.jsonl',record)
        print(json.dumps(record),flush=True)
        # Post-launch GPU/process and disk observation; no new allocation is made.
        subprocess.run([sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),
                        '--purpose',f'stage4_postlaunch_{run_id}','--measured-peak-mib',str((measured_peak or 3602)+1024),
                        '--safety-margin-mib','2048'],stdout=out,stderr=subprocess.STDOUT)
        code=p.wait()
    if code: raise RuntimeError(f'foreground smoke failed: {log}, exit {code}')
    result=json.loads((EXP/'logs/stage4'/run_id/'result.json').read_text())
    print(json.dumps({'run_id':run_id,'updates':result['completed_updates'],
          'peak_reserved_mib':result['peak_memory_reserved_bytes']/2**20,'seconds':result['segment_wall_seconds']}),flush=True)
    return result

def main():
    import torch
    continuous='stage4_smoke_q50_continuous_v1'
    split='stage4_smoke_q50_split_v1'
    zero='stage4_smoke_zero_v1'
    a=launch(continuous,12)
    peak=math.ceil(a['peak_memory_reserved_bytes']/2**20)
    atomic_write_json(LOGS/'memory_profile.json',{'status':'pass','degree':30,'run_id':continuous,
        'measured_peak_reserved_mib':peak,'context_allowance_mib':1024,'safety_margin_mib':2048,
        'result':str(EXP/'logs/stage4'/continuous/'result.json')})
    launch(split,10,measured_peak=peak)
    b=launch(split,12,resume=True,measured_peak=peak)
    ca=torch.load(a['latest_checkpoint'],map_location='cpu',weights_only=False)
    cb=torch.load(b['latest_checkpoint'],map_location='cpu',weights_only=False)
    fields=['actor_state_dict','critic_state_dict','optimizer_state_dict','scheduler_state_dict',
            'rng_state','training_stream_hash','method_state','call_ledger','initialization','best']
    for field in fields: assert_same(ca[field],cb[field],field)
    audit={'status':'running','resume':{'status':'pass','fields':fields,'continuous':a['latest_checkpoint'],
            'split':b['latest_checkpoint'],'crosses_refresh_update_11':True},'memory_profile':str(LOGS/'memory_profile.json')}
    atomic_write_json(LOGS/'audit.json',audit)
    z=launch(zero,2,rho_mode='zero',measured_peak=peak)
    cz=torch.load(z['latest_checkpoint'],map_location='cpu',weights_only=False)
    erm_path=EXP/'logs/stage2/stage2_resume_reference_erm_d30_s101_u2/result.json'
    erm=json.loads(erm_path.read_text())
    ce=torch.load(erm['latest_checkpoint'],map_location='cpu',weights_only=False)
    zero_fields=['actor_state_dict','critic_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state','training_stream_hash','initialization']
    for field in zero_fields: assert_same(cz[field],ce[field],field)
    assert z['method_state']['cost_history']==[0.,0.]
    audit.update(status='pass',completed_at_utc=utc_now(),zero_radius={'status':'pass','fields':zero_fields,
         'wdro':z['latest_checkpoint'],'erm':erm['latest_checkpoint'],'erm_result_sha256':file_sha256(erm_path),
         'mean_cost':0.0,'claim':'exact model/optimizer/RNG equality for 2 real updates; zero-cost feasible limit'})
    atomic_write_json(LOGS/'audit.json',audit)
    print(json.dumps({'status':'pass','audit':str(LOGS/'audit.json')}),flush=True)

if __name__=='__main__': main()

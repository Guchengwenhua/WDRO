#!/usr/bin/env python3
"""Foreground real updates, cross-refresh deterministic recovery and 64-call audit."""
import json,sys,subprocess,shlex,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.job_control import utc_now
from src.rest import file_sha256

from run_stage3_smoke import assert_same
LOG=EXP/'logs/stage6_smoke'

def launch(name,config,stop,resume=False,degree=30,seed=101):
    run_id='stage6_smoke_'+name+'_v1';rp=EXP/'logs/stage6'/run_id/'result.json'
    if rp.exists():
        old=json.loads(rp.read_text())
        if old['completed_updates']>=stop:return old
        resume=True
    pre=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose',run_id+'_to'+str(stop)]
    profile=LOG/'memory_profile.json'
    peak=max(3602,json.loads(profile.read_text())['maximum_reserved_mib'] if profile.exists() else 0)
    pre+=['--measured-peak-mib',str(peak+1024),'--safety-margin-mib','4096']
    p=subprocess.run(pre,capture_output=True,text=True,check=True);summary=json.loads(p.stdout.strip().splitlines()[-1])
    snapshot=json.loads(Path(summary['manifest']).read_text());gpu=max([r for r in snapshot['gpu_decisions'] if r['eligible']],key=lambda r:r['minimum_free_memory_mib'])['index']
    cmd=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/train_stage6.py'),'--degree',str(degree),'--seed',str(seed),'--scope','stage6_smoke',
         '--wdro-config',config['config'],'--rho-mode',config['rho_mode'],'--run-id',run_id,'--stop-after',str(stop),'--physical-gpu-index',str(gpu),
         '--host-runtime-manifest',summary['manifest'],'--geosteiner-workers','3']
    if resume:cmd+=['--resume']
    log=LOG/f'{run_id}_to{stop}.log';LOG.mkdir(parents=True,exist_ok=True)
    with log.open('a') as out:
        proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
        record={'at_utc':utc_now(),'pid':proc.pid,'run_id':run_id,'gpu':gpu,'command':cmd,'exact_command':shlex.join(cmd),'preflight':summary,'log':str(log),'log_resolved':str(log.resolve())}
        append_jsonl(LOG/'launches.jsonl',record);print(json.dumps(record),flush=True)
        from capture_host_runtime import gpu_sample,compute_processes,disk_snapshot
        append_jsonl(LOG/'postlaunch.jsonl',{'at_utc':utc_now(),'run_id':run_id,'gpus':gpu_sample(),'processes':compute_processes(),'disk':[disk_snapshot(CODEX),disk_snapshot(EXP)]})
        if proc.wait():raise RuntimeError(f'smoke failed: {log}')
    result=json.loads(rp.read_text());peak=math.ceil(result['peak_memory_reserved_bytes']/2**20)
    old=json.loads(profile.read_text()) if profile.exists() else {'runs':{},'maximum_reserved_mib':0}
    old['runs'][run_id]={'reserved_mib':peak,'result':str(rp),'result_sha256':file_sha256(rp)}
    old['maximum_reserved_mib']=max(old['maximum_reserved_mib'],peak);old['context_allowance_mib']=1024;old['safety_margin_mib']=2048
    atomic_write_json(profile,old)
    print(json.dumps({'run_id':run_id,'updates':stop,'peak_mib':peak,'seconds':result['segment_wall_seconds']}),flush=True)
    return result

def main():
    import torch
    LOG.mkdir(parents=True,exist_ok=True)
    tasks=json.loads((ROOT/'configs/stage6_plan_v1.json').read_text())['tasks']
    fields=['actor_state_dict','critic_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state','training_stream_hash','method_state','call_ledger','initialization','best']
    d30=next(t for t in tasks if t['degree']==30 and t['seed']==101)
    a=launch('d30_continuous',d30,12)
    launch('d30_split',d30,10);b=launch('d30_split',d30,12,True)
    ca=torch.load(a['latest_checkpoint'],map_location='cpu',weights_only=False);cb=torch.load(b['latest_checkpoint'],map_location='cpu',weights_only=False)
    for field in fields:assert_same(ca[field],cb[field],field)
    if d30['rho_mode']=='q50':
        old=json.loads((EXP/'logs/stage5/stage5_smoke_main_continuous_v1/result.json').read_text())
    else:
        # Compare against the same corrected validation-selected setting.
        # This is a 12-update implementation smoke and never enters selection.
        from run_stage5_smoke import launch as stage5_launch
        chosen=json.loads((ROOT/'configs/stage5_selected_wdro_v1.json').read_text())
        old=stage5_launch('selected_equivalence_continuous',dict(config=chosen['selected_config']['logical_path'],rho_mode=d30['rho_mode']),12)
    ck=torch.load(old['latest_checkpoint'],map_location='cpu',weights_only=False)
    for field in fields:assert_same(ca[field],ck[field],field)
    runs=[a]
    for degree in [50,40,20,10]:
        cfg=next(t for t in tasks if t['degree']==degree and t['seed']==102)
        runs.append(launch('d'+str(degree)+'_s102_peak',cfg,2,degree=degree,seed=102))
    reports=[]
    for run in runs:
        cmd=[sys.executable,str(ROOT/'scripts/audit_stage6.py'),'--run-id',run['run_id']]
        with (LOG/(run['run_id']+'_audit.log')).open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        reports.append({'path':str(EXP/'logs/stage6_audit/runs'/f"{run['run_id']}.json")})
    atomic_write_json(LOG/'audit.json',dict(status='pass',at_utc=utc_now(),resume_fields=fields,
        exact_stage5_update_equivalence=True,cross_refresh_resume=True,full_audits=reports,degrees=[10,20,30,40,50]))
    print(json.dumps({'status':'pass','report':str(LOG/'audit.json')}))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Foreground real updates, cross-refresh deterministic recovery and 64-call audit."""
import json,sys,subprocess,shlex,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.job_control import utc_now
from src.rest import file_sha256
from stage5.configuration import write_config,base_config
from run_stage3_smoke import assert_same
from precision_capacity_wait import wait_capture
LOG=EXP/'logs/stage5_smoke'

def launch(name,config,stop,resume=False):
    run_id='stage5_smoke_'+name+'_v1';rp=EXP/'logs/stage5'/run_id/'result.json'
    if rp.exists():
        old=json.loads(rp.read_text())
        if old['completed_updates']>=stop:return old
        resume=True
    pre=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose',run_id+'_to'+str(stop)]
    profile=LOG/'memory_profile.json'
    if profile.exists():
        peak=json.loads(profile.read_text())['maximum_reserved_mib']
        pre+=['--measured-peak-mib',str(peak+1024),'--safety-margin-mib','2048']
    else:
        peak=json.loads((EXP/'logs/stage4_smoke/memory_profile.json').read_text())['measured_peak_reserved_mib']
        pre+=['--measured-peak-mib',str(peak+1024),'--safety-margin-mib','8192']
    p=wait_capture(pre,context=run_id+'_to'+str(stop));summary=json.loads(p.stdout.strip().splitlines()[-1])
    snapshot=json.loads(Path(summary['manifest']).read_text());gpu=max([r for r in snapshot['gpu_decisions'] if r['eligible']],key=lambda r:r['minimum_free_memory_mib'])['index']
    cmd=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/train_stage5.py'),'--degree','30','--seed','101','--scope','stage5_smoke',
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
    fields=['actor_state_dict','critic_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state','training_stream_hash','method_state','call_ledger','initialization','best']
    audit={'status':'running','at_utc':utc_now(),'resume':{},'variants':{}}
    configs={'main':write_config('smoke_main','q50')}
    for name in ['random_64','cem_64']:
        search=dict(base_config()['search_overrides'],searcher=name,candidate_count=63,exact_rescore_top_k=64)
        configs[name]=write_config('smoke_'+name,'q50',search_overrides=search)
    for name,cfg in configs.items():
        a=launch(name+'_continuous',cfg,12)
        launch(name+'_split',cfg,10);b=launch(name+'_split',cfg,12,True)
        ca=torch.load(a['latest_checkpoint'],map_location='cpu',weights_only=False);cb=torch.load(b['latest_checkpoint'],map_location='cpu',weights_only=False)
        for field in fields:assert_same(ca[field],cb[field],field)
        if name!='main':assert a['method_state']['budget']['exact_rescore_nets']==12*256*64 and a['method_state']['budget']['surrogate_nets']==0
        audit['resume'][name]={'status':'pass','fields':fields,'continuous':a['latest_checkpoint'],'split':b['latest_checkpoint'],'crosses_refresh11':True}
        atomic_write_json(LOG/'audit.json',audit)
    for name,cfg in [('zero',write_config('smoke_zero','zero')),
      ('fixed',write_config('smoke_fixed','q50',lambda_mode='fixed',lambda_initial=.2)),
      ('k1',write_config('smoke_k1','q50',search_overrides=dict(base_config()['search_overrides'],candidate_count=1,exact_rescore_top_k=2))),
      ('k8',write_config('smoke_k8','q50',search_overrides=dict(base_config()['search_overrides'],candidate_count=8))),
      ('chamfer',write_config('smoke_chamfer','q50',ground_cost='symmetric_chamfer_l1'))]:
        r=launch(name,cfg,2);audit['variants'][name]={'result':str(EXP/'logs/stage5'/r['run_id']/'result.json'),'status':'pass'}
        if name=='fixed':assert r['method_state']['lambda']==.2
        if name=='zero':
            z=torch.load(r['latest_checkpoint'],map_location='cpu',weights_only=False)
            e=json.loads((EXP/'logs/stage2/precision_erm_continuous_d30_s101/result.json').read_text())
            er=torch.load(e['latest_checkpoint'],map_location='cpu',weights_only=False)
            for field in ['actor_state_dict','critic_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state','training_stream_hash']:assert_same(z[field],er[field],field)
        atomic_write_json(LOG/'audit.json',audit)
    audit['status']='pass';audit['completed_at_utc']=utc_now();atomic_write_json(LOG/'audit.json',audit)
    print(json.dumps({'status':'pass','audit':str(LOG/'audit.json')}),flush=True)
if __name__=='__main__':main()

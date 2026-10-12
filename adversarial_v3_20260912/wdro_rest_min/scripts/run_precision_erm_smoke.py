"""Foreground repair smoke, checkpoint equivalence and fresh five-degree peaks."""
import argparse,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.job_control import utc_now
from src.geosteiner import reference_identity
from src.rest import file_sha256
from audit_resume_determinism import first_difference
from capture_host_runtime import gpu_sample,compute_processes,disk_snapshot
import torch

LOG=EXP/'logs/precision_erm_smoke';LOG.mkdir(parents=True,exist_ok=True)

def run(degree,rid,stop,resume=False):
    path=EXP/'logs/stage2'/rid/'result.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['completed_updates']==stop:
            ck=torch.load(old['latest_checkpoint'],map_location='cpu',weights_only=False)
            assert ck['run_identity']['reference_solver']==reference_identity()
            return old
    pre=subprocess.run([sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose',f'precision_smoke_{rid}_{stop}',
                        '--measured-peak-mib','4626','--safety-margin-mib','2048'],capture_output=True,text=True,check=True)
    host=json.loads(pre.stdout.strip().splitlines()[-1]);m=json.loads(Path(host['manifest']).read_text())
    gpu=max((r for r in m['gpu_decisions'] if r['eligible']),key=lambda r:r['minimum_free_memory_mib'])['index']
    command=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/train_erm.py'),'--degree',str(degree),'--seed','101',
             '--scope','stage2_smoke','--run-id',rid,'--stop-after',str(stop),'--physical-gpu-index',str(gpu),
             '--host-runtime-manifest',host['manifest'],'--geosteiner-workers','3']
    if resume:command.append('--resume')
    log=LOG/f'{rid}_{stop}.log'
    with log.open('a') as out:
        proc=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
        append_jsonl(LOG/'launches.jsonl',dict(at_utc=utc_now(),pid=proc.pid,gpu=gpu,command=command,preflight=host,log=str(log),resolved=str(log.resolve())))
        time.sleep(2)
        append_jsonl(LOG/'postlaunch.jsonl',dict(at_utc=utc_now(),pid=proc.pid,gpus=gpu_sample(),compute_processes=compute_processes(),disk=[disk_snapshot(CODEX),disk_snapshot(EXP)]))
        code=proc.wait()
    assert code==0,f'smoke failed: {log}'
    value=json.loads(path.read_text());assert value['completed_updates']==stop
    print(json.dumps(dict(run=rid,updates=stop,peak_mib=value['peak_memory_reserved_bytes']/2**20)),flush=True)
    return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--one',action='store_true');a=parser.parse_args()
    cache=json.loads((EXP/'cache/lstar/protocol_v1/lstar_index.json').read_text());assert cache['status']=='pass' and cache['dataset_count']==40 and cache['reference_identity']==reference_identity()
    c=run(30,'precision_erm_continuous_d30_s101',2)
    if a.one:return
    run(30,'precision_erm_resumed_d30_s101',1)
    r=run(30,'precision_erm_resumed_d30_s101',2,True)
    left=torch.load(c['latest_checkpoint'],map_location='cpu',weights_only=False);right=torch.load(r['latest_checkpoint'],map_location='cpu',weights_only=False)
    fields=['run_identity','update','actor_state_dict','critic_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state','best','call_ledger','training_stream_hash','initialization','inverse_length_audit']
    errors={f:first_difference(left[f],right[f],f) for f in fields};assert not any(errors.values()),errors
    atomic_write_json(LOG/'resume_equivalence.json',dict(status='pass',at_utc=utc_now(),fields=fields,left=c['latest_checkpoint'],right=r['latest_checkpoint']))
    profiles={}
    for d in [10,20,30,40,50]:
        result=c if d==30 else run(d,f'precision_erm_peak_d{d}_s101',2)
        peak=math.ceil(result['peak_memory_reserved_bytes']/2**20)
        profiles[str(d)]=dict(measured_torch_reserved_mib=peak,capacity_peak_mib=peak+1024,evidence=result['event_log'],
                             smoke_seconds_per_update=result['segment_wall_seconds']/result['completed_updates'])
    profile=dict(schema_version=2,profile_id='precision_v2_erm_memory',created_at_utc=utc_now(),research_protocol_override=False,
                 batch_size=256,per_gpu_safety_margin_mib=2048,profiles=profiles,reference_identity=reference_identity())
    atomic_write_json(ROOT/'manifests/stage2_memory_profile_v1.json',profile,mode=0o444)
    atomic_write_json(LOG/'short_gate.json',dict(status='pass',at_utc=utc_now(),resume_fields=len(fields),profiles=profiles))
    run(30,'stage2_smoke_erm_d30_s101_u200',200)
    atomic_write_json(LOG/'completion.json',dict(status='pass',at_utc=utc_now(),resume_fields=len(fields),profiles=profiles,full_smoke_updates=200))
    print(json.dumps(dict(status='pass',full_smoke_updates=200)),flush=True)
if __name__=='__main__':main()

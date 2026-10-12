"""Bind new measured memory and original-scope baseline smoke before replay."""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from src.geosteiner import reference_identity
from run_precision_training_queue import source_gate

def main():
    source_gate()
    smoke=EXP/'logs/stage3_smoke/audit_final.json';smoke_audit=json.loads(smoke.read_text());assert smoke_audit['status']=='pass'
    profile=dict(schema_version=2,reference_identity=reference_identity(),profiles={},per_gpu_safety_margin_mib=2048,research_protocol_override=False)
    gates=[]
    for method in ('mixed_erm','group_dro','hard_mining'):
        profile['profiles'][method]={}
        for d in (10,20,30,40,50):
            rid=f'smoke_final_{method}_d{d}_'+('continuous' if d==30 else 'memory')
            rp=EXP/'logs/stage3'/rid/'result.json';r=json.loads(rp.read_text())
            assert r['run_identity']['reference_solver']==reference_identity()
            # Include the resume segment: restoring Adam may change reserved memory.
            peak=max(math.ceil(r['peak_memory_reserved_bytes']/2**20),
                     smoke_audit['profiles'][method][str(d)]['peak_reserved_mib'])
            profile['profiles'][method][str(d)]=dict(measured_torch_reserved_mib=peak,capacity_peak_mib=peak+1024,
                smoke_seconds_per_update=r['segment_wall_seconds']/r['completed_updates'],evidence=str(rp),result_sha256=file_sha256(rp),
                resume_peak_evidence=str(smoke),resume_peak_evidence_sha256=file_sha256(smoke))
        p=EXP/'logs/stage3_audit/runs'/f'smoke_final_{method}_d30_continuous.json'
        a=json.loads(p.read_text());assert a['status']=='pass';gates.append(dict(path=str(p),sha256=file_sha256(p)))
    pp=ROOT/'manifests/stage3_memory_profile_v1.json';atomic_write_json(pp,profile,mode=0o444)
    atomic_write_json(EXP/'logs/stage3_audit/final_smoke_gate.json',dict(status='pass',at_utc=utc_now(),runs=gates,full_input_rebuild=True))
    source=json.loads((ROOT/'manifests/precision_v2_training_source_v1.json').read_text())
    files={'wdro_rest_min/'+p:h for p,h in source['files'].items()}
    extras=[pp,smoke,EXP/'logs/stage3_audit/final_smoke_gate.json',EXP/'logs/stage3_unit_tests_final.log']
    extras += [ROOT/'scripts'/n for n in ('train_baseline.py','run_precision_training_queue.py','run_stage3_baseline_queue.py',
        'run_stage3_smoke.py','stage3_pipeline.py','audit_stage3.py','export_stage3.py','stage3_env.sh','capture_host_runtime.py')]
    for p in extras:files[str(p.relative_to(CODEX))]=file_sha256(p)
    init=json.loads((ROOT/'manifests/precision_v2_initial_preflight.json').read_text())
    env=json.loads(Path(init['manifest']).read_text())
    legacy={str(CODEX.parent/p):h for p,h in (env['code']['legacy_assets']|env['checkpoints']).items()}
    contract=dict(schema_version=2,created_at_utc=utc_now(),research_protocol_override=False,reference_identity=reference_identity(),
        source_and_config_hashes=files,legacy_hashes=legacy,formal_run_count=45,
        artifacts=dict(logical=str(EXP),resolved=str(EXP.resolve())),smoke_gates=gates)
    target=ROOT/'manifests/stage3_launch_contract_v1.json'
    if target.exists():assert json.loads(target.read_text())['source_and_config_hashes']==files
    else:atomic_write_json(target,contract,mode=0o444)
    print(json.dumps(dict(status='pass',manifest=str(target),profiles=15)))
if __name__=='__main__':main()

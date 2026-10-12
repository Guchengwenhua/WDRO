"""Freeze the corrected common implementation before the first formal replay."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.geosteiner import reference_identity
from src.rest import file_sha256

def main():
    smoke=json.loads((CODEX/'experiments/logs/precision_erm_smoke/completion.json').read_text())
    assert smoke['status']=='pass'
    proof=Path('/remote-home/Guilin/DRO/Codex/shared/experiments/logs/precision_v2/reference_verification/acceptance.json')
    assert json.loads(proof.read_text())['status']=='pass'
    files={}
    for directory in ('src','stage4','stage5'):
        for p in sorted((ROOT/directory).glob('*.py')):files[str(p.relative_to(ROOT))]=file_sha256(p)
    for name in ('protocol_v1','data_contract_v1','stage2_trainer_v1','stage3_baselines_v1','stage4_wdro_v1'):
        p=ROOT/'configs'/f'{name}.json';files[str(p.relative_to(ROOT))]=file_sha256(p)
        assert file_sha256(p)==file_sha256(Path('/remote-home/Guilin/DRO/Codex/wdro_legacy_v1/wdro_rest_min/configs')/p.name)
    target=ROOT/'manifests/precision_v2_training_source_v1.json'
    if target.exists():assert json.loads(target.read_text())['files']==files
    else:atomic_write_json(target,dict(schema_version=1,at_utc=utc_now(),reference_identity=reference_identity(),
        research_protocol_override=False,files=files,proof=str(proof),proof_sha256=file_sha256(proof),
        artifact_root=dict(logical_path=str(CODEX/'experiments'),resolved_path=str((CODEX/'experiments').resolve()))),mode=0o444)
    import run_precision_training_queue as q
    for stage,count in ((2,15),(3,45)):
        q.STAGE=stage;assert len(q.build_tasks())==count
    q.source_gate()
    host_path=Path(sys.argv[1]);host=json.loads(host_path.read_text());assert host['preflight_passed']
    runtime=ROOT/'configs'/('runtime_precision_'+host_path.stem+'.json')
    atomic_write_json(runtime,dict(schema_version=1,runtime_id=runtime.stem,research_protocol_override=False,
        python=sys.executable,artifact_root='Codex/experiments',allowed_gpu_indices=host['eligible_gpu_indices'],
        forbidden_gpu_indices=[],allowed_gpu_idle_memory_mib_max=49140,minimum_free_disk_gib_for_batch=100,
        host_runtime_manifest=str(host_path),allocation_is_permanent=False),mode=0o444)
    command=[sys.executable,str(ROOT/'scripts/run_preflight.py'),'--runtime-config',str(runtime),'--require-batch-ready']
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    summary=json.loads(result.stdout.strip().splitlines()[-1])
    atomic_write_json(ROOT/'manifests/precision_v2_initial_preflight.json',summary,mode=0o444)
    print(json.dumps(dict(status='pass',source=str(target),preflight=summary)))
if __name__=='__main__':main()

"""Bind checked operations and smoke artifacts before stage-6 admission."""
import json,sys,ast,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.job_control import utc_now
from stage6_pipeline import gate

def main():
    gate(smoke=False)
    test=EXP/'logs/stage6_pipeline/unit_tests_final_pass.log';value=test.read_text()
    assert 'Ran 87 tests' in value and value.rstrip().endswith('OK')
    smoke=EXP/'logs/stage6_smoke/audit.json';evaluation=EXP/'logs/stage6_evaluation_smoke/audit.json'
    a=json.loads(smoke.read_text());e=json.loads(evaluation.read_text());assert a['status']==e['status']=='pass'
    assert a['exact_stage5_update_equivalence'] and a['cross_refresh_resume']
    files=[*sorted((ROOT/'scripts').glob('*stage6*.py')),*sorted((ROOT/'stage6').glob('*.py')),ROOT/'tests/test_stage6.py',
        ROOT/'scripts/capture_host_runtime.py',ROOT/'scripts/run_preflight.py',ROOT/'scripts/stage3_env.sh',test,smoke,evaluation,
        EXP/'logs/stage6_smoke/memory_profile.json',EXP/'logs/stage6_evaluation_smoke/interruption.json',
        EXP/'logs/stage6_pipeline/input_gate.json',EXP/'logs/stage6_pipeline/statistics_validation_benchmark.json']
    total=0
    for row in a['full_audits']:
        p=Path(row['path']);r=json.loads(p.read_text());assert r['status']=='pass' and all(r['checks'].values());files.append(p);total+=len(r['checks'])
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text())
    diff=''.join(difflib.unified_diff((ROOT/'stage5/trainer.py').read_text().splitlines(True),(ROOT/'stage6/trainer.py').read_text().splitlines(True),fromfile='stage5/trainer.py',tofile='stage6/trainer.py'))
    dp=EXP/'logs/stage6_pipeline/trainer_scope_diff.patch';dp.write_text(diff);files.append(dp)
    report=dict(status='frozen',stage=6,research_protocol_override=False,files={str(p.relative_to(CODEX)):file_sha256(p) for p in files},
        unit_tests=87,smoke_checks=total,scheduler=dict(max_runs=8,max_per_gpu=2,geosteiner_workers=3,context_mib=1024,safety_mib=2048,
            memory_history_seconds=600,per_degree_measured_peak=True,cpu_audit_workers=8,stagger_seconds=15),
        dependencies=['training/evaluation smoke','14 new WDRO training','full candidate/cost/dual/HM audits','freeze75','shared test300','paired CI and all-prediction verification','DONE stage6 only'])
    p=ROOT/'manifests/stage6_operations_contract_v1.json'
    if p.exists():assert json.loads(p.read_text())==report
    else:atomic_write_json(p,report,mode=0o444)
    print(json.dumps({'status':'pass','smoke_checks':total,'bound_files':len(files)}))
if __name__=='__main__':main()

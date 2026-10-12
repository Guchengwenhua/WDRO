#!/usr/bin/env python3
"""Bind tested operations and smoke evidence before any formal stage-5 batch."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from stage5_pipeline import gate

def main():
    gate()
    test=EXP/'logs/stage5_unit_tests_final.log';text=test.read_text();assert 'Ran 80 tests' in text and text.rstrip().endswith('OK')
    inputs=[*sorted((ROOT/'scripts').glob('*stage5*.py')),ROOT/'tests/test_stage5.py',ROOT/'tests/test_stage5_operations.py',
        ROOT/'scripts/capture_host_runtime.py',ROOT/'scripts/run_preflight.py',ROOT/'scripts/stage3_env.sh',
        ROOT/'manifests/stage4_completion_v1.json',ROOT/'manifests/stage4_delivery_verification_v1.json',
        test,EXP/'logs/stage5_smoke/audit.json',EXP/'logs/stage5_smoke/memory_profile.json',EXP/'logs/stage5_audit/smoke_gate.json']
    gate_data=json.loads((EXP/'logs/stage5_audit/smoke_gate.json').read_text())
    inputs.extend(Path(row['path']) for row in gate_data['runs']+gate_data['independent_scores'])
    files={str(path.relative_to(CODEX)):file_sha256(path) for path in inputs}
    record={'status':'frozen','stage':5,'research_protocol_override':False,'files':files,'unit_tests':80,
      'full_smoke_checks':gate_data['total_checks'],'scheduler':{'max_runs':6,'max_runs_per_gpu':2,'context_mib':1024,'safety_mib':2048,
          'memory_low_watermark_seconds':600,'normal_geosteiner_workers':3,'equal64_geosteiner_workers':8},
      'phase_dependencies':['smoke gate','rho training','rho full audit','validation-only selection','six ablation runs','all full audits','best validation recompute','export','independent delivery verification','DONE at stage5 only']}
    target=ROOT/'manifests/stage5_operations_contract_v1.json'
    if target.exists():assert json.loads(target.read_text())==record
    else:atomic_write_json(target,record,mode=0o444)
    print(json.dumps({'status':'pass','files':len(files),'smoke_checks':gate_data['total_checks'],'manifest':str(target)}))
if __name__=='__main__':main()

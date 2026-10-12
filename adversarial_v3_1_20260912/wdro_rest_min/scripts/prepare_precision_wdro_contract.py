"""Rebind the original WDRO implementation contract to corrected references."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from run_precision_training_queue import source_gate

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=[4,5],required=True);a=p.parse_args()
    source_gate()
    previous=ROOT/'manifests'/f'stage{a.stage-1}_delivery_verification_v1.json'
    assert json.loads(previous.read_text())['status']=='pass'
    original=Path('/remote-home/Guilin/DRO/Codex/wdro_legacy_v1/wdro_rest_min/manifests')/f'stage{a.stage}_training_contract_v1.json'
    old=json.loads(original.read_text());files={p:file_sha256(CODEX/p) for p in old['files']}
    value=dict(stage=a.stage,status='frozen',research_protocol_override=False,files=files,
        authorized_reference_correction=True,original_contract_sha256=file_sha256(original),previous_delivery_sha256=file_sha256(previous))
    dst=ROOT/'manifests'/original.name
    if dst.exists():assert json.loads(dst.read_text())==value
    else:atomic_write_json(dst,value,mode=0o444)
    if a.stage==4:
        import numpy as np
        from stage4.dual import primal_dual_lp
        # Same tiny case as the original diagnostic, independently recomputed.
        regrets=np.array([[0.,2.],[0.,3.]])
        costs=np.array([[0.,1.],[0.,3.]])
        cases=[primal_dual_lp(regrets,costs,rho) for rho in (0.,.05,.5,2.)]
        assert all(c['status']=='pass' for c in cases)
        atomic_write_json(EXP/'logs/stage4/math_lp_toy.json',dict(status='pass',at_utc=utc_now(),cases=cases,
            source_file_sha256=file_sha256(ROOT/'stage4/dual.py')))
    print(json.dumps(dict(status='pass',manifest=str(dst),stage=a.stage)))
if __name__=='__main__':main()

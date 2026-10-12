"""Freeze 75 audited validation-selected models before any test prediction."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from stage6.evaluation import artifact
from stage6_pipeline import gate
from src.stage2_audit_checks import all_finite

def main():
    gate();plan=json.loads((ROOT/'configs/stage6_plan_v1.json').read_text());models=list(plan['reused_models']);audits=[]
    for t in plan['tasks']:
        if t['reused_stage5']:continue
        r=json.loads(Path(t['result']).read_text());ap=EXP/'logs/stage6_audit/runs'/f"{t['run_id']}.json";a=json.loads(ap.read_text())
        assert a['status']=='pass' and all(a['checks'].values()) and a['completed_updates']==2000
        assert a['binding']['result_sha256']==file_sha256(Path(t['result'])) and a['training_nets_checked']==512000
        assert a['candidate_nets_checked']==16896000 and a['actual_hard_mining_pair']['paired_updates']==[1,2000]
        assert r['status']=='complete' and r['completed_updates']==2000 and r['selection_feasible'] and not r['test_or_ispd_used']
        models.append(dict(method='wdro_rest',degree=t['degree'],seed=t['seed'],result=artifact(t['result']),
            best=artifact(r['best_checkpoint']),latest=artifact(r['latest_checkpoint'])))
        audits.append(artifact(ap))
    keys=set()
    for m in models:
        key=(m['method'],m['degree'],m['seed']);assert key not in keys;keys.add(key)
        for k in ['result','best','latest']:
            x=m[k];assert file_sha256(Path(x['logical_path']))==x['sha256'] and str(Path(x['logical_path']).resolve())==x['resolved_path']
        r=json.loads(Path(m['result']['logical_path']).read_text())
        assert r['method']==m['method'] and r['degree']==m['degree'] and r['training_seed']==m['seed'] and r['completed_updates']==2000 and r['batch_size']==256
        best=torch.load(m['best']['logical_path'],map_location='cpu',weights_only=False)
        latest=torch.load(m['latest']['logical_path'],map_location='cpu',weights_only=False)
        assert best['update']==r['best']['update'] and latest['update']==2000 and all_finite(best) and all_finite(latest)
        for ck in [best,latest]:
            for k in ['optimizer_state_dict','scheduler_state_dict','rng_state']:assert k in ck
        assert not r['test_or_ispd_used'] and r['best_checkpoint_sha256']==m['best']['sha256']
    expected={(m,d,s) for m in ['erm','mixed_erm','hard_mining','group_dro','wdro_rest'] for d in [10,20,30,40,50] for s in [101,102,103]}
    assert keys==expected
    index=ROOT/'manifests/stage6_models_frozen_v1.json'
    if index.exists():
        old=json.loads(index.read_text());assert old['models']==models;print(json.dumps({'status':'pass','models':75,'reused_freeze':True}));return
    assert not list((EXP/'logs/stage6_test').glob('*/*/result.json')),'test outputs exist before first model freeze'
    code=json.loads((ROOT/'manifests/stage6_operations_contract_v1.json').read_text())['files']
    atomic_write_json(index,dict(status='frozen',at_utc=utc_now(),models=models,full_wdro_audits=audits,
        test_evaluated_at_freeze=False,selection='synthetic_validation_only',test_used_for_selection=False,stage7_started=False,
        plan=artifact(ROOT/'configs/stage6_plan_v1.json'),code_files=code),mode=0o444)
    print(json.dumps({'status':'pass','models':75,'index':str(index)}))
if __name__=='__main__':main()

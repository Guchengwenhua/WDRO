"""Verify completed stage-6 training artifacts without opening model test metrics."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.io_utils import atomic_write_json
from src.rest import file_sha256
from src.job_control import utc_now
from src.stage2_audit_checks import all_finite,optimizer_checks
from src.trainer import load_trainer_config
from audit_stage3 import effective_events
from stage6_pipeline import gate

def main():
    gate();plan=json.loads((ROOT/'configs/stage6_plan_v1.json').read_text());rows=[]
    for task in plan['tasks']:
        if task['reused_stage5']:continue
        path=Path(task['result'])
        if not path.exists():continue
        r=json.loads(path.read_text())
        if r['status']!='complete':continue
        latest=torch.load(r['latest_checkpoint'],map_location='cpu',weights_only=False)
        best=torch.load(r['best_checkpoint'],map_location='cpu',weights_only=False)
        train,validation,_=effective_events(r['event_log'])
        checks=dict(updates=r['completed_updates']==latest['update']==2000,
            budget=r['batch_size']==256 and r['expected_updates']==2000,
            identity=latest['run_identity']==best['run_identity']==r['run_identity'],
            degree_seed=r['degree']==task['degree'] and r['training_seed']==task['seed'],
            rho=r['run_identity']['rho']==task['rho'],
            checkpoint_hashes=all(file_sha256(Path(r[k+'_checkpoint']))==r[k+'_checkpoint_sha256'] for k in ['latest','best']),
            full_log=sorted(train)==list(range(1,2001)) and sorted(validation)==list(range(0,2001,100)),
            validation_only=not r['test_or_ispd_used'] and all(v['result']['split']=='validation' for v in validation.values()),
            finite=all_finite(latest) and all_finite(best) and all_finite(list(train.values())),
            restored_state=all(k in latest and k in best for k in ['optimizer_state_dict','scheduler_state_dict','rng_state']),
            nominal_feasible=r['selection_feasible'] and r['best']['nominal_mean']<=r['nominal_limit'],
            selected_update=best['update']==r['best']['update'],
            ledger=latest['call_ledger']==r['call_ledger'],
            method_state=latest['method_state']==r['method_state'],
            stream=latest['training_stream_hash']==r['training_stream_hash'],
            no_partial=not list(Path(r['latest_checkpoint']).parent.glob('*.partial')),
            no_live_lock=not (EXP/'logs/jobs'/task['run_id']/'run.lock').exists())
        checks.update(optimizer_checks(latest,load_trainer_config()))
        assert all(checks.values()),(task['run_id'],[k for k,v in checks.items() if not v])
        rows.append(dict(run_id=task['run_id'],status='pass',result_sha256=file_sha256(path),checks=checks,
            best_update=best['update'],best_checkpoint_sha256=r['best_checkpoint_sha256']))
    output=dict(status='pass',at_utc=utc_now(),completed_runs=len(rows),checks_passed=sum(len(r['checks']) for r in rows),runs=rows,
        full_candidate_audit_still_required=True,test_or_ispd_model_metrics_used=False)
    atomic_write_json(EXP/'logs/stage6_continuation/completed_training_review.json',output)
    print(json.dumps({'status':'pass','completed_runs':len(rows),'checks':output['checks_passed']}))
if __name__=='__main__':main()

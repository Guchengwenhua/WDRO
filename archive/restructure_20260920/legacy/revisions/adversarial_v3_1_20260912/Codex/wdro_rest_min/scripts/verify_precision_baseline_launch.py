"""Verify the formal replay launch; this is not final training acceptance."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments'
sys.path.insert(0,str(ROOT))
from src.checkpoints import load_checkpoint
from src.geosteiner import reference_identity
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from src.stage2_audit_checks import all_finite
from src.trainer import load_trainer_config
from audit_stage2_formal import optimizer_checks
from stage3_pipeline import verify_sources
from run_precision_training_queue import source_gate,process_is_alive
from precision_replay_pipeline import cpu_preflight


def main():
    source_gate();verify_sources()
    pre=cpu_preflight('baseline_formal_launch_review')
    queue=EXP/'logs/stage3_queue/stage3_baselines_formal_v1/queue.json'
    value=json.loads(queue.read_text());tasks=value['tasks']
    expected={(m,d,s) for m in ('mixed_erm','group_dro','hard_mining') for d in (10,20,30,40,50) for s in (101,102,103)}
    checks={'exact_45_run_matrix':len(tasks)==45 and {(t['method'],t['degree'],t['seed']) for t in tasks}==expected,
            'source_and_operations_gates_pass':True}
    profile=json.loads((ROOT/'manifests/stage3_memory_profile_v1.json').read_text())
    smoke=json.loads((EXP/'logs/stage3_smoke/audit_final.json').read_text())
    checks['all_15_profiles_include_resume_peak']=all(
        profile['profiles'][m][str(d)]['measured_torch_reserved_mib']>=smoke['profiles'][m][str(d)]['peak_reserved_mib']
        for m,d in {(m,d) for m,d,_ in expected})
    active=[t for t in tasks if t['status']=='running'];checks['active_training_present']=bool(active)
    reviewed=[]
    for t in active:
        rid=t['run_id'];p=EXP/'checkpoints/stage3'/rid/'latest.pt'
        if not p.exists():continue  # A newly staggered launch may still be initializing.
        ck=load_checkpoint(p,map_location='cpu');ident=ck['run_identity']
        hb=json.loads((EXP/'logs/jobs'/rid/'heartbeat.json').read_text())
        prefix=rid+'_'
        checks[prefix+'process_alive']=process_is_alive(t['pid'])
        checks[prefix+'identity']=ident['reference_solver']==reference_identity() and ident['expected_updates']==2000 and ident['batch_size']==256 and ident['method']==t['method'] and ident['degree']==t['degree'] and ident['training_seed']==t['seed']
        checks.update({prefix+k:v for k,v in optimizer_checks(ck,load_trainer_config()).items()})
        checks[prefix+'finite_checkpoint']=all_finite(ck)
        host=json.loads(Path(t['host_runtime_manifest']).read_text())
        checks[prefix+'admitted_by_fresh_preflight']=host['preflight_passed'] and t['physical_gpu_index'] in host['eligible_gpu_indices']
        checks[prefix+'actual_gpu_matches_admission']=hb['physical_gpu_index']==t['physical_gpu_index'] if 'physical_gpu_index' in hb else json.loads((EXP/'logs/jobs'/rid/'job.json').read_text())['physical_gpu_index']==t['physical_gpu_index']
        reviewed.append(dict(run_id=rid,pid=t['pid'],gpu=t['physical_gpu_index'],checkpoint_update=ck['update'],heartbeat_update=hb['completed_updates'],
            checkpoint_logical_path=str(p),checkpoint_resolved_path=str(p.resolve()),preflight=t['host_runtime_manifest']))
    checks['at_least_one_committed_positive_update']=any(r['checkpoint_update']>0 for r in reviewed)
    output=EXP/'logs/precision_replay_continuation/baseline_launch_verification.json'
    report=dict(at_utc=utc_now(),status='pass' if all(checks.values()) else 'fail',scope='launch verification only',
        checks=checks,reviewed=reviewed,preflight=str(pre),queue=str(queue),training_complete=False,stage6_complete=False,
        source_sha256=file_sha256(Path(__file__)),memory_profile_sha256=file_sha256(ROOT/'manifests/stage3_memory_profile_v1.json'))
    atomic_write_json(output,report)
    print(json.dumps(dict(status=report['status'],checks=len(checks),failed=[k for k,v in checks.items() if not v],reviewed=reviewed,output=str(output))))
    assert all(checks.values()),checks


if __name__=='__main__':main()

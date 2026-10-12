"""Continue the approved correction only after the 60 baselines pass delivery."""
import argparse,json,shlex,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.job_control import JobLease
from src.io_utils import atomic_write_json
from run_precision_training_queue import source_gate
from precision_replay_pipeline import step,read,cpu_preflight,progress

def main():
    source_gate();job=EXP/'logs/precision_wdro_replay'
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only:
        pre=cpu_preflight('wdro_controller_foreground_smoke')
        atomic_write_json(job/'foreground_smoke.json',dict(status='pass',preflight=str(pre),guard='wait for hash-bound baseline delivery before any WDRO run'))
        print(json.dumps(dict(status='pass',preflight=str(pre))));return
    exact=shlex.join(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__))])
    recovery='bash /remote-home/Guilin/DRO/Codex/wdro_legacy_v1/wdro_rest_min/scripts/stage3_env.sh '+sys.executable+' /remote-home/Guilin/DRO/Codex/wdro_legacy_v1/wdro_rest_min/scripts/precision_job.py --name wdro_replay -- '+exact
    with JobLease(job,run_id='precision_wdro_replay',exact_command=exact,resume_command=recovery,
        monitor_command=f'cat {job}/heartbeat.json',expected_updates=1,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        while not (EXP/'logs/precision_replay/steps/baseline_delivery.json').exists():
            lease.update(update=0,status='waiting_baseline_delivery',detail=dict(prerequisite=str(EXP/'logs/precision_replay/heartbeat.json')));time.sleep(15)
        assert read(ROOT/'manifests/stage3_delivery_verification_v1.json')['status']=='pass'
        step(lease,'wdro_diagnostic_contract','prepare_precision_wdro_contract.py',['--stage','4'],[ROOT/'manifests/stage4_training_contract_v1.json'])
        step(lease,'wdro_diagnostic_regression','run_precision_unit_tests.py',['--through-stage4','--output',EXP/'logs/stage4_all_unit_tests_v1.log'],[EXP/'logs/stage4_all_unit_tests_v1.log'])
        step(lease,'wdro_diagnostic_smoke','run_stage4_smoke.py',outputs=[EXP/'logs/stage4_smoke/audit.json'])
        for rid in ('stage4_smoke_q50_continuous_v1','stage4_smoke_zero_v1'):
            step(lease,'wdro_diagnostic_audit_'+rid,'audit_stage4.py',['--run-id',rid],[EXP/'logs/stage4_audit/runs'/f'{rid}.json'])
        step(lease,'wdro_diagnostic_independent_scores','run_precision_gpu_check.py',['--script','audit_stage4_initial_scores.py'],[EXP/'logs/stage4/independent_initial_score_audit.json'])
        step(lease,'wdro_diagnostic_train_audit_export','stage4_pipeline.py',outputs=[ROOT/'manifests/stage4_completion_v1.json'])
        step(lease,'wdro_diagnostic_delivery','verify_stage4_delivery.py',outputs=[ROOT/'manifests/stage4_delivery_verification_v1.json'])
        assert read(ROOT/'manifests/stage4_delivery_verification_v1.json')['status']=='pass'
        step(lease,'wdro_validation_contract','prepare_precision_wdro_contract.py',['--stage','5'],[ROOT/'manifests/stage5_training_contract_v1.json'])
        step(lease,'wdro_validation_regression','run_precision_unit_tests.py',['--output',EXP/'logs/stage5_unit_tests_final.log'],[EXP/'logs/stage5_unit_tests_final.log'])
        step(lease,'wdro_validation_smoke','run_stage5_smoke.py',outputs=[EXP/'logs/stage5_smoke/audit.json'])
        step(lease,'wdro_validation_smoke_audit','run_stage5_smoke_audit.py',outputs=[EXP/'logs/stage5_audit/smoke_gate.json'])
        step(lease,'wdro_validation_operations','freeze_stage5_operations.py',outputs=[ROOT/'manifests/stage5_operations_contract_v1.json'])
        step(lease,'wdro_validation_search_audit_freeze','precision_stage5_capacity_pipeline.py',outputs=[ROOT/'manifests/stage5_delivery_verification_v1.json',ROOT/'configs/stage5_selected_wdro_v1.json'])
        assert read(ROOT/'manifests/stage5_delivery_verification_v1.json')['status']=='pass'
        lease.finish(update=1,status='complete',detail=dict(completed_replay_stages=[4,5],next='stage6 using corrected validation freeze'))
if __name__=='__main__':main()

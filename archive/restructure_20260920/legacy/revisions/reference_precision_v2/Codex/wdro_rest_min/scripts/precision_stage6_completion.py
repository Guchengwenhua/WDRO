"""Finish Stage 6 after corrected validation freeze, with no test feedback loop."""
import argparse,json,shlex,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.job_control import JobLease
from src.io_utils import atomic_write_json
from run_precision_training_queue import source_gate
from precision_replay_pipeline import step,read,cpu_preflight

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    source_gate();job=EXP/'logs/precision_stage6_completion'
    if args.check_only:
        pre=cpu_preflight('stage6_controller_foreground_smoke')
        atomic_write_json(job/'foreground_smoke.json',dict(status='pass',preflight=str(pre),guard='corrected validation delivery before Stage6 plan or model test'))
        print(json.dumps(dict(status='pass',preflight=str(pre))));return
    exact=shlex.join(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__))])
    recovery='bash /remote-home/Guilin/DRO/Codex/wdro_rest_min/scripts/stage3_env.sh '+sys.executable+' /remote-home/Guilin/DRO/Codex/wdro_rest_min/scripts/precision_job.py --name stage6_completion -- '+exact
    with JobLease(job,run_id='precision_stage6_completion',exact_command=exact,resume_command=recovery,
        monitor_command=f'cat {job}/heartbeat.json',expected_updates=1,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        while not (EXP/'logs/precision_replay/steps/wdro_validation_search_audit_freeze.json').exists():
            lease.update(update=0,status='waiting_corrected_validation_freeze',detail=dict(prerequisite=str(EXP/'logs/precision_wdro_replay/heartbeat.json')));time.sleep(15)
        assert read(ROOT/'manifests/stage5_delivery_verification_v1.json')['status']=='pass'
        step(lease,'stage6_corrected_plan','prepare_stage6.py',outputs=[ROOT/'configs/stage6_plan_v1.json',ROOT/'manifests/stage6_training_contract_v1.json'])
        step(lease,'stage6_corrected_training_smoke','precision_stage6_isolated_smoke.py',outputs=[EXP/'logs/stage6_smoke/audit.json'])
        step(lease,'stage6_corrected_evaluation_smoke','run_stage6_evaluation_smoke.py',outputs=[EXP/'logs/stage6_evaluation_smoke/audit.json'])
        step(lease,'stage6_corrected_regression','run_precision_unit_tests.py',['--include-stage6','--output',EXP/'logs/stage6_pipeline/unit_tests_final_pass.log'],[EXP/'logs/stage6_pipeline/unit_tests_final_pass.log'])
        step(lease,'stage6_corrected_statistics_smoke','benchmark_precision_statistics.py',outputs=[EXP/'logs/stage6_pipeline/statistics_validation_benchmark.json'])
        step(lease,'stage6_corrected_operations','freeze_stage6_operations.py',outputs=[ROOT/'manifests/stage6_operations_contract_v1.json'])
        step(lease,'stage6_complete_training_test_delivery','stage6_pipeline.py',outputs=[ROOT/'manifests/stage6_completion_v1.json',ROOT/'manifests/stage6_delivery_verification_v1.json'])
        step(lease,'stage6_publish_authoritative_completion','publish_precision_completion.py',outputs=[Path('/remote-home/Guilin/DRO/Codex/wdro_rest_min/manifests/precision_v2_replay_completion.json')])
        lease.finish(update=1,status='complete',detail=dict(stage6='DONE',stage7_started=False))
if __name__=='__main__':main()

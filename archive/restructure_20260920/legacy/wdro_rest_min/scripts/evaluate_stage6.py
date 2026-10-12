"""One isolated GPU evaluation worker; smoke uses validation exclusively."""
import argparse,json,os,sys,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--degree',type=int,required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--preflight',type=Path,required=True);p.add_argument('--smoke',action='store_true')
p.add_argument('--smoke-name',default='resume');p.add_argument('--max-new-batches',type=int)
args=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.job_control import JobLease,utc_now
from src.rest import file_sha256
from stage6.evaluation import evaluate_cell
from capture_host_runtime import gpu_sample
from datetime import datetime,timezone

def main():
    host=json.loads(args.preflight.read_text());assert host['preflight_passed'] and args.gpu in host['eligible_gpu_indices']
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(host['captured_at_utc'].replace('Z','+00:00'))).total_seconds()<900
    hardware=next(g for g in gpu_sample() if g['index']==args.gpu)
    # Same GPU model and numerical implementation for all cells. Each degree worker
    # additionally keeps all methods/seeds on one UUID unless capacity recovery requires migration.
    hardware=dict(name=hardware['name'],torch_version=__import__('torch').__version__,device_capability=list(__import__('torch').cuda.get_device_capability()))
    plan=json.loads((ROOT/'configs/stage6_plan_v1.json').read_text())
    if args.smoke:
        models=[m for m in plan['reused_models'] if m['method']=='erm' and m['degree']==args.degree and m['seed']==101]
        split='validation';job=EXP/'logs/stage6_evaluation_smoke'/f'd{args.degree}_{args.smoke_name}'
    else:
        index=ROOT/'manifests/stage6_models_frozen_v1.json';frozen=json.loads(index.read_text())
        assert frozen['status']=='frozen' and len(frozen['models'])==75 and frozen['test_evaluated_at_freeze'] is False
        for path,digest in frozen['code_files'].items():assert file_sha256(ROOT.parent/path)==digest
        models=[m for m in frozen['models'] if m['degree']==args.degree];assert len(models)==15
        split='test';job=EXP/'logs/stage6_evaluation_jobs'/f'd{args.degree}'
    command=shlex.join([sys.executable,str(Path(__file__)),*sys.argv[1:]])
    with JobLease(job,run_id=f'eval_{split}_d{args.degree}',exact_command=command,
        resume_command=f'bash {ROOT}/scripts/stage3_env.sh {sys.executable} {ROOT}/scripts/launch_stage6.py',
        monitor_command=f'cat {job}/heartbeat.json',expected_updates=len(models)*4*(2000 if args.smoke else 10000),
        physical_gpu_index=args.gpu,host_runtime_manifest=str(args.preflight)) as lease:
        completed=[]
        for model in models:
            for dist in plan['evaluation']['distributions']:
                out=job/'cells'/dist if args.smoke else EXP/'logs/stage6_test'/f"{model['method']}_d{args.degree}_s{model['seed']}"/dist
                append_jsonl(job/'attempts.jsonl',dict(at_utc=utc_now(),gpu=args.gpu,model=model,distribution=dist,output=str(out),resolved=str(out.resolve())))
                result=evaluate_cell(model,dist,split,out,hardware,max_new_batches=args.max_new_batches,
                    heartbeat=lambda n:lease.update(update=sum(c['completed_nets'] for c in completed)+n,status='evaluating',detail={'cell':str(out)}))
                if result['status']=='paused':lease.finish(update=result['completed_nets'],status='paused');return
                completed.append(result)
        atomic_write_json(job/'result.json',dict(status='complete',at_utc=utc_now(),degree=args.degree,split=split,cells=len(completed),
            completed_nets=sum(c['completed_nets'] for c in completed),peak_reserved_mib=max(c['peak_reserved_mib'] for c in completed)))
        lease.finish(update=sum(c['completed_nets'] for c in completed),status='complete')
if __name__=='__main__':main()

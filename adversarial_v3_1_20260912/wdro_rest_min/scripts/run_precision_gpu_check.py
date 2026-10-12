"""Fresh admission and post-launch evidence for a bounded GPU verification."""
import argparse,json,subprocess,sys,time,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import append_jsonl
from src.job_control import utc_now
from capture_host_runtime import gpu_sample,compute_processes,disk_snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--script',choices=['audit_stage4_initial_scores.py','verify_stage4_best_validation.py'],required=True);a=p.parse_args()
    profile=json.loads((EXP/'logs/stage4_smoke/memory_profile.json').read_text())
    command=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose','precision_'+a.script,
        '--measured-peak-mib',str(profile['measured_peak_reserved_mib']+1024),'--safety-margin-mib','2048']
    while True:
        result=subprocess.run(command,capture_output=True,text=True)
        append_jsonl(EXP/'logs/precision_gpu_checks/preflights.jsonl',dict(at_utc=utc_now(),command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode==0:break
        time.sleep(15)
    summary=json.loads(result.stdout.strip().splitlines()[-1]);host=json.loads(Path(summary['manifest']).read_text())
    gpu=max((r for r in host['gpu_decisions'] if r['eligible']),key=lambda r:r['minimum_free_memory_mib'])['index']
    flag='--physical-gpu-index' if a.script.startswith('audit') else '--gpu'
    cmd=[sys.executable,str(ROOT/'scripts'/a.script),flag,str(gpu),'--preflight',summary['manifest']]
    proc=subprocess.Popen(cmd)
    record=dict(at_utc=utc_now(),pid=proc.pid,gpu=gpu,command=cmd,exact_command=shlex.join(cmd),preflight=summary)
    append_jsonl(EXP/'logs/precision_gpu_checks/launches.jsonl',record)
    time.sleep(1)
    append_jsonl(EXP/'logs/precision_gpu_checks/postlaunch.jsonl',dict(record,gpus=gpu_sample(),compute_processes=compute_processes(),disk=[disk_snapshot(CODEX),disk_snapshot(EXP)]))
    if proc.wait():raise SystemExit(1)
if __name__=='__main__':main()

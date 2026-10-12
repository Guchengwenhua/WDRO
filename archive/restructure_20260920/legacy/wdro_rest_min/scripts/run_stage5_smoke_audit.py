#!/usr/bin/env python3
"""Wait for foreground smoke, then independently reconstruct every smoke input."""
import json,sys,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.job_control import utc_now
from src.rest import file_sha256

def main():
    source=EXP/'logs/stage5_smoke/audit.json'
    # This controller is launched in foreground only after the smoke is ready.
    smoke=json.loads(source.read_text());assert smoke['status']=='pass'
    names=['main_continuous','random_64_continuous','cem_64_continuous','zero','fixed','k1','k8','chamfer']
    records=[]
    for name in names:
        rid='stage5_smoke_'+name+'_v1';path=EXP/'logs/stage5_audit/runs'/f'{rid}.json'
        if not path.exists() or json.loads(path.read_text())['status']!='pass':
            pre=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose','stage5_smoke_audit_'+name,'--measured-peak-mib','2654','--safety-margin-mib','2048']
            p=subprocess.run(pre,capture_output=True,text=True)
            append_jsonl(EXP/'logs/stage5_smoke/audit_launches.jsonl',{'at_utc':utc_now(),'command':pre,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
            cmd=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/audit_stage5.py'),'--run-id',rid]
            log=EXP/'logs/stage5_smoke'/f'{rid}_full_audit.log'
            with log.open('a') as out:
                proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
                append_jsonl(EXP/'logs/stage5_smoke/audit_launches.jsonl',{'at_utc':utc_now(),'pid':proc.pid,'command':cmd,'log':str(log),'log_resolved':str(log.resolve())})
                if proc.wait():raise RuntimeError('smoke audit failed '+str(log))
        result=json.loads(path.read_text());assert result['status']=='pass'
        records.append({'run_id':rid,'path':str(path),'sha256':file_sha256(path),'checks':len(result['checks'])})
    score_reports=[]
    for name in ['random_64_continuous','cem_64_continuous']:
        rid='stage5_smoke_'+name+'_v1';out=EXP/'logs/stage5_initial_score_audit'/f'{rid}.json'
        if not out.exists() or json.loads(out.read_text())['status']!='pass' or json.loads(out.read_text()).get('geosteiner_workers')!=8:
            pre=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose','stage5_initial_true_score_'+name,'--measured-peak-mib','2654','--safety-margin-mib','2048']
            p=subprocess.run(pre,capture_output=True,text=True,check=True);summary=json.loads(p.stdout.strip().splitlines()[-1])
            snapshot=json.loads(Path(summary['manifest']).read_text());gpu=max([r for r in snapshot['gpu_decisions'] if r['eligible']],key=lambda r:r['minimum_free_memory_mib'])['index']
            cmd=[sys.executable,str(ROOT/'scripts/audit_stage5_initial_scores.py'),'--run-id',rid,'--gpu',str(gpu),'--preflight',summary['manifest']]
            log=EXP/'logs/stage5_smoke'/f'{rid}_independent_scores.log'
            with log.open('a') as handle:
                proc=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT)
                append_jsonl(EXP/'logs/stage5_smoke/audit_launches.jsonl',{'at_utc':utc_now(),'pid':proc.pid,'command':cmd,'preflight':summary,'log':str(log),'log_resolved':str(log.resolve())})
                if proc.wait():raise RuntimeError('independent true scoring failed '+str(log))
        score_reports.append({'path':str(out),'sha256':file_sha256(out)})
    atomic_write_json(EXP/'logs/stage5_audit/smoke_gate.json',{'status':'pass','at_utc':utc_now(),'runs':records,'total_checks':sum(r['checks'] for r in records),
        'independent_scores':score_reports,'smoke_sha256':file_sha256(source),'full_input_rebuild':True,'test_ispd_model_metrics_read':False})
    print(json.dumps({'status':'pass','runs':len(records),'checks':sum(r['checks'] for r in records)}))
if __name__=='__main__':main()

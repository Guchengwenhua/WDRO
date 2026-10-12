"""Prove inference interruption/recovery on validation before test is unlocked."""
import json,sys,subprocess,shlex
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.rest import file_sha256
from src.job_control import utc_now
from capture_host_runtime import gpu_sample,compute_processes,disk_snapshot
LOG=EXP/'logs/stage6_evaluation_smoke'

def launch(name,limit=None):
    command=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose','stage6_evaluation_validation_smoke',
        '--measured-peak-mib','4626','--safety-margin-mib','4096']
    summary=json.loads(subprocess.run(command,check=True,capture_output=True,text=True).stdout.strip().splitlines()[-1])
    host=json.loads(Path(summary['manifest']).read_text());gpu=max((g for g in host['gpu_decisions'] if g['eligible']),key=lambda g:g['minimum_free_memory_mib'])['index']
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_stage6.py'),'--degree','50','--gpu',str(gpu),'--preflight',summary['manifest'],'--smoke','--smoke-name',name]
    if limit:cmd+=['--max-new-batches',str(limit)]
    with (LOG/f'{name}_{limit}.log').open('a') as out:
        p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
        append_jsonl(LOG/'launches.jsonl',dict(at_utc=utc_now(),pid=p.pid,command=cmd,exact_command=shlex.join(cmd),preflight=summary))
        append_jsonl(LOG/'postlaunch.jsonl',dict(at_utc=utc_now(),gpus=gpu_sample(),compute_processes=compute_processes(),disk=[disk_snapshot(ROOT.parent),disk_snapshot(EXP)]))
        assert p.wait()==0,str(LOG/f'{name}_{limit}.log')

def main():
    LOG.mkdir(parents=True,exist_ok=True)
    rp=LOG/'d50_resume/result.json'
    if not rp.exists():
        if not (LOG/'interruption.json').exists():
            cursor=LOG/'d50_resume/cells/uniform/cursor.json'
            if not cursor.exists() or json.loads(cursor.read_text())['complete_nets']==0:launch('resume',1)
            state=json.loads(cursor.read_text())
            assert state['complete_nets']==256 and not (LOG/'d50_resume/cells/uniform/result.json').exists()
            atomic_write_json(LOG/'interruption.json',dict(status='pass',committed_nets=256,complete_result_absent=True))
        launch('resume')
    launch('continuous')
    model=json.loads((EXP/'logs/stage2/formal_erm_d50_s101_u2000/result.json').read_text())
    expected=None
    for line in Path(model['event_log']).read_text().splitlines():
        e=json.loads(line)
        if e['event']=='validation' and e['update']==model['best']['update']:expected=e['result']
    checks={}
    for dist in ['uniform','mix_gauss','edge_bias','grid_cluster']:
        a=LOG/'d50_resume/cells'/dist;b=LOG/'d50_continuous/cells'/dist
        with np.load(a/'predictions.npz') as x,np.load(b/'predictions.npz') as y:
            for k in x.files:np.testing.assert_array_equal(x[k],y[k])
        result=json.loads((a/'result.json').read_text());assert result['summary']==expected['groups'][dist]['summary']
        checks[dist]=dict(exact_recovery=True,exact_frozen_validation_metrics=True,nets=2000)
    atomic_write_json(LOG/'audit.json',dict(status='pass',at_utc=utc_now(),checks=checks,test_model_metrics_used=False,
        maximum_reserved_mib=json.loads(rp.read_text())['peak_reserved_mib'],evaluator_sha256=file_sha256(ROOT/'stage6/evaluation.py')))
    print(json.dumps({'status':'pass','audit':str(LOG/'audit.json')}))
if __name__=='__main__':main()

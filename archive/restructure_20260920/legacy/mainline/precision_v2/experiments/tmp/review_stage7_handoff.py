"""Read-only result/exit review; optional progress handoff, no model calls."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path('/remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min')
sys.path.insert(0,str(ROOT))
from stage7.common import *
from stage7.evaluation import model_id


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--finalize-progress',action='store_true')
    args=parser.parse_args()
    plan=gate()
    completion_path=ROOT/'manifests/stage7_completion_v1.json'
    delivery_path=ROOT/'manifests/stage7_delivery_verification_v1.json'
    assert completion_path.exists() and delivery_path.exists(),'Stage7 delivery is not complete'
    delivery=read(delivery_path);completion=read(checked(delivery['completion']))
    assert delivery['status']=='pass' and completion['status']=='pass'
    assert (completion['models'],completion['predictions'],completion['unique_nets'])==(75,337470,22498)
    preflight,host=capture('Stage7 final handoff '+('smoke' if args.smoke else 'full')+' (CPU)')
    out=LOG/'handoff'
    out.mkdir(parents=True,exist_ok=True)
    checkfile=out/('write_probe_'+stamp())
    checkfile.write_bytes(b'stage7 ordinary-user write/readback')
    assert checkfile.read_bytes()==b'stage7 ordinary-user write/readback'
    checkfile.unlink()
    documents=[read(PLAN),read(ROOT/'manifests/stage7_operations_v1.json'),completion,delivery,
        read(LOG/'references_gate.json'),read(LOG/'smoke/reference_gate.json'),
        read(LOG/'smoke/evaluation_gate.json'),read(LOG/'audit/result.json'),read(LOG/'statistics.json')]
    total=0
    for model in plan['models']:
        p=LOG/'evaluation'/model_id(model)
        result=read(p/'result.json');cursor=read(p/'cursor.json')
        assert result['binding']['model']==model and cursor['complete_nets']==result['completed_nets']
        pred=load_npz(checked(result['predictions']))
        assert len(pred['net_ids'])==result['completed_nets'] and len(set(pred['net_ids']))==len(pred['net_ids'])
        assert np.isfinite(pred['regret']).all() and pred['regret'].min()>=-1e-12
        documents.extend([result,cursor,read(LOG/'audit/models'/f'{model_id(model)}.json')])
        total+=result['completed_nets']
    assert total==337470
    for group in plan['groups']:
        path=CACHE/'references'/f"degree_{group['degree']}"
        documents.extend([read(path/'result.json'),read(path/'cursor.json')])
    found={}
    def collect(value):
        if isinstance(value,dict):
            if {'logical_path','resolved_path','sha256'}.issubset(value):
                if value['logical_path'] in found:
                    assert found[value['logical_path']]['sha256']==value['sha256'],value['logical_path']
                found[value['logical_path']]=value
            for v in value.values():collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for value in documents:collect(value)
    chosen=list(found.values())[:10] if args.smoke else list(found.values())
    for value in chosen:checked(value)
    exits=[]
    for launch_path in sorted((LOG/'launches').glob('*/launch.json')):
        launch=read(launch_path)
        code=read(launch_path.parent/'exit.json')['exit_code']
        assert code==0,(launch_path,code)
        r=subprocess.run(['tmux','-S',launch['socket'],'has-session','-t',launch['session']],capture_output=True)
        assert r.returncode!=0,'Stage7 session still alive'
        exits.append(dict(phase=launch['phase'],session=launch['session'],exit_code=code,
                          launch=artifact(launch_path)))
    assert {x['phase'] for x in exits}=={'references','evaluation'}
    locked=[]
    for path in LOG.rglob('run.lock'):
        with path.open('a+') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:locked.append(str(path))
            else:fcntl.flock(f,fcntl.LOCK_UN)
    assert not locked,locked
    assert not list((LOG/'evaluation').rglob('*.partial'))
    assert not list((CACHE/'references').rglob('*.partial'))
    starts=[json.loads(x) for x in (LOG/'pipeline/evaluation/launches.jsonl').read_text().splitlines()]
    runtime_jobs=0
    for path in list((LOG/'evaluation').glob('*/job.json'))+list((LOG/'references').glob('*/job.json')):
        job=read(path);pid=job['process'];current=pid_identity(pid['pid'])
        assert not(current and current['state'] not in ('Z','X') and current['start_ticks']==pid['start_ticks'])
        assert job['lock_released']
        for k in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','MPLCONFIGDIR','TRITON_CACHE_DIR'):
            assert Path(job['runtime_environment'][k]).is_relative_to(TOP)
        runtime_jobs+=1
    target=out/('smoke.json' if args.smoke else 'review.json')
    result=dict(status='pass',at_utc=utc_now(),scope='smoke' if args.smoke else 'full',
        preflight=artifact(preflight),code=artifact(Path(__file__)),models=75,predictions=total,
        hashes_checked=len(chosen),artifacts_available=len(found),runtime_jobs=runtime_jobs,
        gpu_indices_used=sorted({r['gpu'] for r in starts}),launches=len(starts),exits=exits,
        all_locks_released=True,no_incomplete_prediction_or_reference=True,
        ordinary_user_write_readback_cleanup=True,uid=os.getuid(),actor_calls=0,rest_calls=0,geosteiner_calls=0,
        completion=artifact(completion_path),delivery=artifact(delivery_path),files=chosen)
    if args.finalize_progress:
        assert not args.smoke
        path=TOP/'01_WDRO-REST_项目进度.md';content=path.read_text();now=utc_now()
        content='\n'.join(f'| 更新时间 | {now} |' if line.startswith('| 更新时间 |') else line for line in content.splitlines())+'\n'
        entry=f'### {now} 阶段7最终交接复核\n\n- `DONE` 75模型/337470预测、{len(chosen)}个文件hash、2070行表格与20份配对差值的绑定再次核验通过；两个tmux流水线exit0、全部任务退出且锁释放，无不完整reference/预测。证据 `revisions/reference_precision_v2/Codex/experiments/logs/stage7/handoff/review.json`。\n- 实测使用GPU{result["gpu_indices_used"]}；最终三类调用新增均为0，普通用户写入/回读/清理及逻辑/resolved磁盘复核通过。无剩余阶段7任务，阶段8–9未启动。\n\n'
        content=content.replace('## 8. 变更记录\n\n','## 8. 变更记录\n\n'+entry,1)
        tmp=path.with_suffix('.md.partial');tmp.write_text(content);os.replace(tmp,path)
        result['progress']=artifact(path)
    result['report']=artifact(TOP/'阶段7_ISPD外部测试验收报告.md')
    atomic_write_json(target,result,mode=0o444)
    print({k:v for k,v in result.items() if k not in ('files','exits')},flush=True)


if __name__=='__main__':main()

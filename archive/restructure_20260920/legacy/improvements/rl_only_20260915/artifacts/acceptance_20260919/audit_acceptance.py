"""Independent CPU acceptance; reads completed artifacts, never executes a model forward."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import geometry as G
import report as R
import scheduler as S

def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def finite_tensors(obj):
    if torch.is_tensor(obj):
        if (obj.is_floating_point() or obj.is_complex()) and not torch.isfinite(obj).all():
            raise AssertionError('nonfinite checkpoint tensor')
        return 1
    if isinstance(obj,dict): return sum(finite_tensors(v) for v in obj.values())
    if isinstance(obj,(tuple,list)): return sum(finite_tensors(v) for v in obj)
    return 0

def independent_lengths(pins,topology):
    """Adjacency-mask reduction, independent of production scatter/union-find evaluator."""
    x=np.asarray(pins,np.float64);t=np.asarray(topology);out=[];n=x.shape[1]
    for start in range(0,len(x),256):
        p=x[start:start+256];a=t[start:start+256];b=len(p)
        links=np.broadcast_to(np.eye(n,dtype=bool),(b,n,n)).copy()
        ii=np.arange(b)[:,None];links[ii,a[:,::2],a[:,1::2]]=True
        vertical=np.where(links,p[:,None,:,1],np.nan)
        horizontal=np.where(links.transpose(0,2,1),p[:,None,:,0],np.nan)
        value=(np.nanmax(vertical,2)-np.nanmin(vertical,2)+np.nanmax(horizontal,2)-np.nanmin(horizontal,2)).sum(1)
        out.append(value)
    return np.concatenate(out)

def calls_audit(path,c):
    starts={};ends={}
    for line in (path/'actual_calls.jsonl').read_text().splitlines():
        row=json.loads(line); dest=starts if row['event']=='start' else ends
        if row['id'] in dest: raise AssertionError('duplicate actual call id')
        dest[row['id']]=row
    if not ends.keys()<=starts.keys(): raise AssertionError('orphan call completion')
    for k in ends:
        for field in ('phase','real_rows','forward_rows','step'):
            if starts[k][field]!=ends[k][field]: raise AssertionError('call identity mismatch')
    actual={}
    for phase in ('search','evaluation','rollout','sampled_training'):
        entries=[v for v in starts.values() if v['phase']==phase]
        actual[phase]={
            'completed_real_rows':sum(v['real_rows'] for v in entries if v['id'] in ends),
            'attempted_real_rows_upper_bound':sum(v['real_rows'] for v in entries),
            'completed_forward_rows':sum(v['forward_rows'] for v in entries if v['id'] in ends),
            'attempted_forward_rows_upper_bound':sum(v['forward_rows'] for v in entries)}
    if actual!=c['actual_call_bounds']: raise AssertionError('actual call accounting mismatch')
    ledger=c['ledger']; replay={}
    for phase,key in [('search','search_greedy_scores'),('evaluation','evaluation_greedy_scores'),('rollout','rollout_greedy_scores'),('sampled_training','sampled_actor_scores')]:
        replay[phase]=actual[phase]['completed_real_rows']-ledger[key]
        if replay[phase]<0: raise AssertionError('physical calls below committed calls')
    return dict(events=len(starts)+len(ends),unresolved_calls=len(starts)-len(ends),replayed_real_rows=replay,actual=actual)

def history_checkpoint_audit(path,c):
    p=c['binding']['effective_protocol'];rows=[json.loads(x) for x in (path/'progress.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['type']=='update'];vals=[r for r in rows if r['type']=='validation']
    assert [r['step'] for r in updates]==list(range(1,p['updates']+1))
    expected=list(range(0,p['updates']+1,p['validation_every']))
    if expected[-1]!=p['updates']:expected.append(p['updates'])
    assert [r['step'] for r in vals]==expected
    for r in vals:
        assert abs(max(v['mean_length'] for v in r['groups'].values())-r['worst_group_mean_length'])<1e-12
    chosen=min(vals,key=lambda v:(v['worst_group_mean_length'],v['step']))
    assert c['best_step']==chosen['step']
    assert abs(c['best_validation_worst_group_length']-chosen['worst_group_mean_length'])<1e-12
    price=p['lambda_init'];method=c['binding']['method']
    for r in updates:
        if method.startswith('wdro_'):price=max(0.,price+p['lambda_lr']*(r['mean_selected_cost']-p['rho']))
        assert abs(price-r['lambda'])<1e-11
        for k,v in r.items():
            if isinstance(v,(int,float)) and not np.isfinite(v):raise AssertionError('nonfinite training log')
    assert abs(c['lambda_final']-price)<1e-11
    latest=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
    best=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
    assert latest['binding']==best['binding']==c['binding']
    assert latest['step']==p['updates'] and best['step']==latest['best_step']==c['best_step']
    assert best['metric']==latest['best_metric']==c['best_validation_worst_group_length']
    assert state_hash(best['actor'])==c['final_actor_sha256']
    assert hashlib.sha256(latest['best_checkpoint_bytes']).hexdigest()==G.sha256(path/'best.pt')
    assert latest['progress_bytes']==(path/'progress.jsonl').stat().st_size
    for name,digest in latest['checkpoint_artifacts'].items():assert G.sha256(path/name)==digest
    cost=np.array([r['mean_selected_cost'] for r in updates])
    return dict(checkpoint_tensors_checked=finite_tensors(latest)+finite_tensors(best),validation_count=len(vals),
        selected_update=c['best_step'],mean_cost=float(cost.mean()),last128_mean_cost=float(cost[-128:].mean()),
        mean_cost_minus_rho=float(cost.mean()-p['rho']),lambda_final=price)

def runtime_final_gate(tasks):
    q=json.loads((ROOT/'artifacts/operations/queue.json').read_text());ex=json.loads((ROOT/'artifacts/operations/exit.json').read_text())
    assert ex['status']=='complete' and not q['running'] and not q['pending'] and not q['failures']
    assert set(q['completed'])=={t['id'] for t in tasks}
    assert not S.is_alive(q['scheduler_identity'])
    for p in (ROOT/'artifacts/operations').glob('attempt_*.json'):
        a=json.loads(p.read_text())
        if a.get('worker_identity'):assert not S.is_alive(a['worker_identity'])
        hb=S.read_json(S.local_path(a['heartbeat_path']),{})
        if hb.get('child_identity'):assert not S.is_alive(hb['child_identity'])
    locks=list((ROOT/'artifacts/operations').glob('*.lock'))
    locks+=list((ROOT/'artifacts/runs').glob('*/run.lock'))+list((ROOT/'artifacts/gpu_smoke').glob('*/run.lock'))
    for p in locks:
        with p.open('r') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
    return dict(locks_released=len(locks),scheduler_exited=True,all_recorded_workers_exited=True)

def audit(full=False):
    G.install_oracle_guard();torch.set_num_threads(1)
    protocol=json.loads((ROOT/'protocol.json').read_text());queue=json.loads((ROOT/'queue.json').read_text())
    frozen=json.loads((ROOT/'artifacts/operations/source_hashes.json').read_text())
    for name,digest in frozen['hashes'].items():assert G.sha256(ROOT/name)==digest, name
    report=dict(status='PARTIAL',time=time.time(),scope='rl_only_20260915',protocol_sha256=G.sha256(ROOT/'protocol.json'),
        auditor_sha256=G.sha256(__file__),frozen_artifacts_checked=len(frozen['hashes']),runs={},smokes={},missing=[],
        independent_length_rows=0,artifact_bindings={},runtime=None,statistics_recompute=None)
    lengths_by_run={};init_hashes={};metric_rows=[]
    for seed in protocol['seeds']:
        torch.manual_seed(seed);actor=G.build_actor(protocol['degree'],torch.device('cpu'))
        init_hashes[seed]=state_hash(actor.state_dict());del actor
    for task in queue['tasks']:
        path=S.local_path(task['complete']).parent
        if not (path/'complete.json').exists():report['missing'].append(task['id']);continue
        smoke=task['kind']=='gpu_smoke';method=task['binding_checks']['method'];seed=task['binding_checks']['seed']
        c,a=R.verify_run(path,protocol,method,seed,smoke,False)
        assert c['initial_actor_sha256']==init_hashes[seed]
        p=c['binding']['effective_protocol']
        item=dict(binding_sha256=G.sha256(path/'complete.json'),history=history_checkpoint_audit(path,c),calls=calls_audit(path,c))
        report['artifact_bindings'][str(path/'complete.json')]=item['binding_sha256']
        for name,digest in c['artifacts'].items():report['artifact_bindings'][str(path/name)]=digest
        count=0
        for d in protocol['distributions']:
            for phase in ('initial','final','attack'):
                prefix=f'{phase}__{d}__';x=a[prefix+'pins'];top=a[prefix+'topology'];values=a[prefix+'length']
                computed=independent_lengths(x.reshape(-1,p['degree'],2),top.reshape(-1,2*(p['degree']-1)))
                np.testing.assert_allclose(computed,values.reshape(-1),rtol=0,atol=2e-12);count+=len(computed)
                if not smoke and phase!='attack':
                    metric_rows.append(dict(method=method,seed=seed,distribution=d,stage=phase,count=len(values),mean=float(np.mean(computed)),p95=float(np.quantile(computed,.95))))
            if not smoke:lengths_by_run[method,seed,d]=(a[f'initial__{d}__length'],a[f'final__{d}__length'])
        item['independent_length_rows']=count;report['independent_length_rows']+=count
        report['smokes' if smoke else 'runs'][task['id']]=item
        print(json.dumps({'audited':task['id'],'independent_rows':count}),flush=True)
    report['completed_formal_runs']=len(report['runs']);report['required_formal_runs']=18
    report['all_initial_weights_independently_recreated']=True
    report['new_model_forward_calls']=0;report['new_training_updates']=0;report['new_reference_solver_calls']=0
    R.write_csv(OUT/('metrics_final_independent.csv' if full else 'metrics_partial_independent.csv'),metric_rows)
    if full:
        assert not report['missing'],'final audit requires all24 task completions'
        report['runtime']=runtime_final_gate(queue['tasks'])
        original=ROOT/'artifacts/report';c=json.loads((original/'complete.json').read_text())
        assert c['status']=='complete' and c['acceptance']=='accepted_gpu' and c['task_count']==18
        for name,digest in c['artifacts'].items():assert G.sha256(original/name)==digest
        for name,digest in c['run_bindings'].items():
            p=Path(name);p=p if p.is_absolute() else ROOT/p
            assert G.sha256(p)==digest
        rerun=OUT/'recomputed_report'
        result=subprocess.run(['bash',str(ROOT/'run_env.sh'),str(ROOT/'report.py'),'--outdir',str(rerun)],cwd=ROOT,capture_output=True,text=True)
        (OUT/'report_recompute.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr
        compared={}
        for name in ('metrics.csv','paired_comparisons.csv','attacks.csv','budget_and_runtime.csv','REPORT.md'):
            assert (original/name).read_bytes()==(rerun/name).read_bytes(),name
            compared[name]=G.sha256(original/name)
        report['statistics_recompute']={'byte_identical_files':compared}
        independent_metrics={(r['method'],r['seed'],r['distribution'],r['stage']):r for r in metric_rows}
        with (original/'metrics.csv').open() as f:
            published=list(csv.DictReader(f))
        assert len(published)==len(independent_metrics)
        for row in published:
            key=(row['method'],int(row['seed']),row['distribution'],row['stage'])
            expected=independent_metrics[key]
            assert int(row['count'])==expected['count']
            for field in ('mean','p95'):assert abs(float(row[field])-expected[field])<1e-12
        # Independently recreate every reported paired interval from saved per-net lengths.
        with (original/'paired_comparisons.csv').open() as f:
            for row in csv.DictReader(f):
                method,baseline,d=row['method'],row['baseline'],row['distribution']
                delta=np.stack([lengths_by_run[method,s,d][1]-(lengths_by_run[method,s,d][0] if baseline=='initial' else lengths_by_run[baseline,s,d][1]) for s in protocol['seeds']])
                rng=np.random.default_rng(20260915);boots=[]
                for _ in range(protocol['statistics']['bootstrap_replicates']):
                    selected_seeds=rng.integers(3,size=3);selected_nets=rng.integers(delta.shape[1],size=delta.shape[1])
                    boots.append(np.mean([delta[k,selected_nets].mean() for k in selected_seeds]))
                lo,hi=np.quantile(boots,[.025,.975])
                for field,v in [('difference',delta.mean()),('ci_low',lo),('ci_high',hi)]:assert abs(float(row[field])-v)<1e-12
        statistics_dir=OUT/'final_statistics'
        statistics_script=OUT/'statistics_review.py'
        result=subprocess.run(['bash',str(ROOT/'run_env.sh'),str(statistics_script),'--outdir',str(statistics_dir),'--require-complete'],cwd=ROOT,capture_output=True,text=True)
        (OUT/'final_statistics.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr
        statistics=json.loads((statistics_dir/'statistics_review.json').read_text())
        assert statistics['status']=='NUMERICAL_REVIEW_PASS' and statistics['completed_runs']==18
        assert not statistics['missing'] and not statistics['errors']
        assert statistics['protocol_sha256']==report['protocol_sha256']
        report['independent_statistics']={'source_sha256':G.sha256(statistics_script),
            'artifacts':{str(p.relative_to(OUT)):G.sha256(p) for p in sorted(statistics_dir.iterdir()) if p.is_file()}}
        report['status']='PASS'
    filename='acceptance_final.json' if full else 'acceptance_partial.json'
    title='完整验收通过' if full else '阶段验收：整批尚未完成'
    lines=[f'# {title}','',f"已核验 {len(report['runs'])}/18 个正式任务、{len(report['smokes'])}/6 个GPU smoke。",
        f"独立重算 {report['independent_length_rows']:,} 条保存拓扑的记账线长；模型/优化器状态、选模、lambda递推、调用日志、来源绑定及初始随机权重均复核。",
        '此次验收没有新增训练、模型前向或参考求解器调用。','',
        ('完整独立统计见 [final_statistics/statistics_review.md](final_statistics/statistics_review.md)；源码方法审查见 [method_review.md](method_review.md)，后者为17/18时的快照。' if full else '阶段统计与方法审查见同目录 statistics_review.md、method_review.md；partial 文档只覆盖其审查时已完成任务。'),
        '关键范围：degree30/三个种子/固定合成布局；指标是自身记账线长，不是精确regret。Random32是局部窗口压力测试，不是rho预算内全局证书。',
        'WDRO的全程平均成本可能略超过rho；不据有限步对偶更新声称严格预算可行。',
        '训练候选未逐个持久化，因此没有离线重放全部训练argmax；检查覆盖其预算、递推和最终测试/攻击拓扑。']
    if report['missing']:lines+=['','尚缺：'+', '.join(report['missing'])+'。不能标记整批DONE。']
    if full:
        lines+=['','最终报告及四张CSV已独立重生成并逐字节相等，配对bootstrap区间还经独立实现核对。所有记录的科研进程退出、锁释放。',
            '','## 完整结果摘要','','四分布组及三个训练种子等权；长度越低越好。此表为描述统计，方法优劣需结合配对区间。','',
            '| 方法 | 初始平均线长 | 最终平均线长 | 窗口攻击后线长 |','|---|---:|---:|---:|']
        for row in statistics['method_summaries']:
            lines.append(f"| {row['method']} | {row['initial_mean']:.6f} | {row['final_mean']:.6f} | {row['mean_attacked']:.6f} |")
        lines+=['','完整验收通过表示冻结任务、数值和产物符合协议，不表示所有改良方法均优于对照，也不表示达到最优解。']
    (OUT/('ACCEPTANCE_FINAL.md' if full else 'ACCEPTANCE_PARTIAL.md')).write_text('\n'.join(lines)+'\n')
    if full:
        (ROOT/'STATUS.md').write_text('# RL-only 当前状态\n\n2026-09-16：18/18正式GPU模型及6/6smoke完成，独立验收通过。\n\n[最终验收](artifacts/acceptance_20260916/ACCEPTANCE_FINAL.md) · [完整结果](artifacts/report/REPORT.md) · [详细验收凭证](artifacts/acceptance_20260916/acceptance_final.json)\n\n验收通过表示冻结任务和产物符合协议，不代表各改良方法均优于对照，不提供最优误差或连续Wasserstein证书。\n')
    G.atomic_json(OUT/filename,report)
    print(json.dumps({'status':report['status'],'formal_runs':len(report['runs']),'missing':report['missing'],'output':str(OUT/filename)}),flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');args=ap.parse_args()
    audit(args.full)

"""Independent artifact/metric audit and paired statistics, with no model execution."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import geometry as G

def write_csv(path, rows):
    path=Path(path); temp=path.with_suffix('.csv.partial')
    with temp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    temp.replace(path)

def metrics(length,lb,mst):
    return dict(length=length,q_lb=length/lb-1,q_mst=length/mst-1,
                evaluator_regret_lower=np.maximum(0,length/mst-1))

def paired_ci(delta, seed=20260915, repeats=2000):
    """Hierarchical paired bootstrap: resample training seeds, then shared net IDs.

    Three seeds give limited information about training variability; this is not
    a distribution-free confidence guarantee or multiple-testing correction.
    """
    a=np.asarray(delta,dtype=np.float64)
    if a.ndim!=2: raise ValueError('expected [training_seed, net]')
    rng=np.random.default_rng(seed); values=np.empty(repeats)
    for i in range(repeats):
        seeds=rng.integers(len(a),size=len(a)); nets=rng.integers(a.shape[1],size=a.shape[1])
        values[i]=a[seeds][:,nets].mean()
    lo,hi=np.quantile(values,[.025,.975])
    return dict(difference=float(a.mean()),ci_low=float(lo),ci_high=float(hi),
                seed_differences=';'.join(f'{x:.12g}' for x in a.mean(1)),
                all_seed_negative=bool(np.all(a.mean(1)<0)),all_seed_positive=bool(np.all(a.mean(1)>0)))

def verify_run(path, protocol, expected_method, expected_seed, smoke=False, allow_cpu=False):
    complete=path/'complete.json'; c=json.loads(complete.read_text())
    if c['status']!='complete': raise AssertionError('run not complete')
    if allow_cpu:
        if not smoke: raise AssertionError('CPU acceptance only allowed for diagnostic smoke report')
    elif c['acceptance']!=('gpu_smoke' if smoke else 'accepted_gpu'):
        raise AssertionError('GPU acceptance missing')
    b=c['binding']
    if (b['method'],int(b['seed']),bool(b['smoke']))!=(expected_method,expected_seed,smoke):
        raise AssertionError('task identity mismatch')
    if b['protocol_sha256']!=G.sha256(G.ROOT/'protocol.json'): raise AssertionError('protocol changed')
    for collection in (b['code_sha256'],b['dependencies']):
        for name,digest in collection.items():
            if G.sha256(name)!=digest: raise AssertionError(f'source changed: {name}')
    for name,digest in c['artifacts'].items():
        target=(path/name).resolve()
        if not target.is_relative_to(path.resolve()) or G.sha256(target)!=digest:
            raise AssertionError(f'artifact changed: {name}')
    with (path/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_UN)
    if c['ledger']['oracle_calls']!=0 or c.get('forbidden_modules_loaded'):
        raise AssertionError('oracle-free audit failed')
    p=b['effective_protocol']; updates=int(p['updates']); batch=int(p['batch_size'])
    if c['step']!=updates or c['ledger']['sampled_actor_scores']!=updates*batch:
        raise AssertionError('policy budget incomplete')
    expected_search=updates*batch*int(p['candidates']) if expected_method.startswith('wdro_') else 0
    if c['ledger']['search_greedy_scores']!=expected_search: raise AssertionError('candidate budget mismatch')
    with np.load(path/'results.npz',allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    for dist in protocol['distributions']:
        for stage in ('initial','final'):
            prefix=f'{stage}__{dist}__'; pins=arrays[prefix+'pins']
            if pins.shape!=(p['test_per_distribution'],p['degree'],2): raise AssertionError('test count mismatch')
            if not np.array_equal(pins,G.make_data(p['test_per_distribution'],p['degree'],dist,'test/'+dist,seed=0)):
                raise AssertionError('test data mismatch')
            lb,mst,hpwl=G.bounds(pins)
            for name,expected in [('lb',lb),('mst',mst),('hpwl',hpwl)]:
                np.testing.assert_allclose(arrays[prefix+name],expected,rtol=0,atol=1e-12)
            length=arrays[prefix+'length']
            np.testing.assert_allclose(length,G.lengths(pins,arrays[prefix+'topology']),rtol=0,atol=1e-12)
            if not np.isfinite(length).all() or np.any(length<lb-1e-6): raise AssertionError('invalid length')
            np.testing.assert_allclose(arrays[prefix+'q'],length/lb-1,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(arrays['initial__'+dist+'__pins'],arrays['final__'+dist+'__pins'])
        prefix='attack__'+dist+'__'; score=arrays[prefix+'q']; index=arrays[prefix+'selected_index']
        if score.shape!=(p['attack_per_distribution'],p['attack_candidates']): raise AssertionError('attack budget mismatch')
        np.testing.assert_array_equal(index,np.argmax(score,axis=1))
        attack_pins=arrays[prefix+'pins']; centers=attack_pins[:,0]
        np.testing.assert_allclose(arrays[prefix+'cost'],G.costs(centers,attack_pins),rtol=0,atol=1e-12)
        alb,am,ah=G.bounds(attack_pins.reshape(-1,p['degree'],2))
        for key,val in [('lb',alb),('mst',am),('hpwl',ah)]:
            np.testing.assert_allclose(arrays[prefix+key],val.reshape(score.shape),rtol=0,atol=1e-12)
        np.testing.assert_allclose(score,arrays[prefix+'length']/arrays[prefix+'lb']-1,rtol=0,atol=1e-12)
        recomputed=G.lengths(attack_pins.reshape(-1,p['degree'],2),arrays[prefix+'topology'].reshape(-1,2*(p['degree']-1)))
        np.testing.assert_allclose(arrays[prefix+'length'],recomputed.reshape(score.shape),rtol=0,atol=1e-12)
    return c,arrays

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs-root',default=str(G.ROOT/'artifacts/runs'))
    parser.add_argument('--outdir',default=str(G.ROOT/'artifacts/report'))
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--allow-cpu',action='store_true')
    args=parser.parse_args(); G.install_oracle_guard()
    out=Path(args.outdir).resolve()
    if not out.is_relative_to(G.ROOT): raise ValueError('report must stay under experiment')
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((G.ROOT/'protocol.json').read_text())
    seeds=[101] if args.smoke else protocol['seeds']; methods=protocol['methods']
    data={}; runs={}; summary=[]; attacks=[]; bindings={}; diagnostics=[]; budgets=[]
    for method in methods:
        for seed in seeds:
            path=Path(args.runs_root)/f'{method}_s{seed}'
            c,a=verify_run(path,protocol,method,seed,args.smoke,args.allow_cpu)
            runs[method,seed]=c; data[method,seed]=a
            bindings[str(path/'complete.json')]=G.sha256(path/'complete.json')
            events=[json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines()]
            updates=[r for r in events if r['type']=='update']
            if [r['step'] for r in updates]!=list(range(1,c['step']+1)): raise AssertionError('update history gap/duplicate')
            costs=np.asarray([r['mean_selected_cost'] for r in updates]); rho=protocol['rho']
            budgets.append(dict(method=method,seed=seed,rho=rho,mean_cost=float(costs.mean()),
                last128_mean_cost=float(costs[-128:].mean()),mean_cost_minus_rho=float(costs.mean()-rho),
                lambda_final=c['lambda_final'],committed_search_scores=c['ledger']['search_greedy_scores'],
                committed_sampled_scores=c['ledger']['sampled_actor_scores'],elapsed_seconds=c['elapsed_seconds'],
                peak_reserved_bytes=c['peak_reserved_bytes'],actual_call_bounds=json.dumps(c.get('actual_call_bounds',{}),sort_keys=True)))
            for dist in protocol['distributions']:
                for stage in ('initial','final'):
                    prefix=f'{stage}__{dist}__'
                    for metric,values in metrics(a[prefix+'length'],a[prefix+'lb'],a[prefix+'mst']).items():
                        summary.append(dict(method=method,seed=seed,distribution=dist,stage=stage,metric=metric,
                                            count=len(values),mean=float(values.mean()),p95=float(np.quantile(values,.95))))
                    diagnostics.append(dict(method=method,seed=seed,distribution=dist,stage=stage,
                        mst_win_rate=float(np.mean(a[prefix+'length']<a[prefix+'mst'])),
                        mean_bound_width_ratio=float(np.mean(a[prefix+'mst']/a[prefix+'lb'])),
                        max_bound_width_ratio=float(np.max(a[prefix+'mst']/a[prefix+'lb']))))
                prefix='attack__'+dist+'__'; selected=a[prefix+'selected_index']; rows=np.arange(len(selected))
                attacks.append(dict(method=method,seed=seed,distribution=dist,count=len(selected),
                    candidates=protocol['attack_candidates'],mean_q_lb=float(a[prefix+'q'][rows,selected].mean()),
                    mean_center_q_lb=float(a[prefix+'q'][:,0].mean()),mean_selected_cost=float(a[prefix+'cost'][rows,selected].mean())))
    # Same fresh initialization and identical test nets are explicit paired-design gates.
    for seed in seeds:
        if len({runs[m,seed]['initial_actor_sha256'] for m in methods})!=1: raise AssertionError('unmatched initialization')
        for dist in protocol['distributions']:
            reference=data[methods[0],seed]['initial__'+dist+'__pins']
            for method in methods:
                np.testing.assert_array_equal(data[method,seed]['final__'+dist+'__pins'],reference)
    comparisons=[]
    for method in methods:
        for dist in protocol['distributions']:
            for metric in ('length','q_mst','q_lb'):
                def values(m,s,stage):
                    a=data[m,s]; prefix=f'{stage}__{dist}__'
                    return metrics(a[prefix+'length'],a[prefix+'lb'],a[prefix+'mst'])[metric]
                for baseline in ('initial','erm_lb','wdro_random'):
                    if method==baseline: continue
                    delta=np.stack([values(method,s,'final')-values(method,s,'initial') if baseline=='initial'
                        else values(method,s,'final')-values(baseline,s,'final') for s in seeds])
                    comparisons.append(dict(method=method,baseline=baseline,distribution=dist,metric=metric,
                        training_seeds=len(seeds),nets_per_seed=delta.shape[1],**paired_ci(delta,repeats=200 if args.smoke else 2000)))
    write_csv(out/'metrics.csv',summary); write_csv(out/'paired_comparisons.csv',comparisons); write_csv(out/'attacks.csv',attacks)
    write_csv(out/'bound_diagnostics.csv',diagnostics); write_csv(out/'budget_and_runtime.csv',budgets)
    lines=['# 无 GeoSteiner 学习：自动验收报告','',
        '**CPU/小样本工程 smoke，不能作为正式 GPU 科学结果。**' if args.smoke else '**18 个 degree30 GPU 任务的产物及数值审计通过。性能优劣以以下配对结果为准。**','',
        f'训练种子：{seeds}。全部任务 oracle_calls=0；源码、配置、产物哈希及锁均核验；使用保存的拓扑独立重算全部测试/攻击树长。',
        '指标 q_lb 是 evaluator 相对最优值的保守上界，q_mst 是相对终端 MST 的可负指标；二者都不是实测真实 regret。',
        f'相同初始 Actor 和相同测试 net 已逐项核对。区间使用种子与 net 的配对分层 bootstrap；本报告 {len(seeds)} 个训练种子，未做多重比较校正。','',
        '| 方法 | 测试最差组 mean q_lb（种子平均） | mean q_mst（四组等权） |','|---|---:|---:|']
    for method in methods:
        per_group=[np.mean([runs[method,s]['final_groups'][d]['mean_q_lb'] for s in seeds]) for d in protocol['distributions']]
        mean_mst=np.mean([runs[method,s]['final_groups'][d]['mean_q_mst'] for s in seeds for d in protocol['distributions']])
        lines.append(f'| {method} | {max(per_group):.6f} | {mean_mst:.6f} |')
    lines+=['','[逐方法/种子/分布 mean 和 P95](metrics.csv) · [训练前后及相对 ERM 配对差](paired_comparisons.csv) · [独立 Random32 压力测试](attacks.csv)','',
        '负配对差表示该指标降低。不能将有限候选攻击称为连续 Wasserstein 证书，也不能把未排除零的区间解释为等效。',
        '运输 rho 是平均预算，训练的逐步 λ 更新不保证每批或全程精确满足；检查各任务 progress.jsonl 中的 mean_selected_cost。',
        '全部结果保留，包括没有改善或退化的任务。此次范围为 degree30 从头训练可行性验证，不能据此比较历史预训练模型的优劣。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    artifacts={name:G.sha256(out/name) for name in ('metrics.csv','paired_comparisons.csv','attacks.csv','bound_diagnostics.csv','budget_and_runtime.csv','REPORT.md')}
    G.atomic_json(out/'complete.json',dict(status='complete',acceptance='diagnostic_only' if args.smoke else 'accepted_gpu',
        created_at=time.time(),task_count=len(runs),protocol_sha256=G.sha256(G.ROOT/'protocol.json'),
        run_bindings=bindings,artifacts=artifacts,all_oracle_calls_zero=True,all_run_locks_released=True))
    print(json.dumps(dict(status='complete',tasks=len(runs),report=str(out/'REPORT.md'))))

if __name__=='__main__': main()

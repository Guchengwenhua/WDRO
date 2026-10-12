"""RL-only artifact and numerical audit; no model inference or external reference."""
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

def verify_run(path,protocol,method,seed,smoke,allow_cpu):
    c=json.loads((path/'complete.json').read_text());b=c['binding'];p=b['effective_protocol']
    if c['status']!='complete': raise AssertionError('incomplete run')
    if allow_cpu:
        if not smoke or c['acceptance']!='cpu_smoke': raise AssertionError('CPU only allowed for diagnostic report')
    elif c['acceptance']!=('gpu_smoke' if smoke else 'accepted_gpu'):
        raise AssertionError('GPU acceptance absent')
    if (b['method'],int(b['seed']),bool(b['smoke']))!=(method,seed,smoke): raise AssertionError('task mismatch')
    if b['protocol_sha256']!=G.sha256(G.ROOT/'protocol.json'): raise AssertionError('protocol changed')
    for collection in (b['code_sha256'],b['dependencies']):
        for name,digest in collection.items():
            if G.sha256(name)!=digest: raise AssertionError('source changed: '+name)
    for name,digest in c['artifacts'].items():
        target=(path/name).resolve()
        if not target.is_relative_to(path.resolve()) or G.sha256(target)!=digest: raise AssertionError('artifact changed: '+name)
    with (path/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
    if c['ledger']['oracle_calls']!=0 or c['forbidden_modules_loaded']: raise AssertionError('external reference dependency')
    n=int(p['updates'])*int(p['batch_size'])
    if c['step']!=p['updates'] or c['ledger']['sampled_actor_scores']!=n: raise AssertionError('training budget')
    if c['ledger']['search_greedy_scores']!=(n*p['candidates'] if method.startswith('wdro_') else 0): raise AssertionError('search budget')
    if c['ledger']['rollout_greedy_scores']!=(n if method=='erm_rollout' else 0): raise AssertionError('rollout budget')
    with np.load(path/'results.npz',allow_pickle=False) as z: a={k:z[k] for k in z.files}
    permitted={'pins','length','topology','cost','selected_index'}
    if any(key.split('__')[-1] not in permitted for key in a): raise AssertionError('unexpected reference metric in RL-only artifact')
    for d in protocol['distributions']:
        for stage in ('initial','final'):
            k=f'{stage}__{d}__'; pins=a[k+'pins']; lengths=a[k+'length']
            if pins.shape!=(p['test_per_distribution'],p['degree'],2): raise AssertionError('test shape')
            np.testing.assert_array_equal(pins,G.make_data(p['test_per_distribution'],p['degree'],d,'test/'+d,seed=0))
            np.testing.assert_allclose(lengths,G.lengths(pins,a[k+'topology']),rtol=0,atol=1e-12)
            if not np.isfinite(lengths).all() or np.any(lengths<=0): raise AssertionError('invalid own length')
            group=c['initial_groups' if stage=='initial' else 'final_groups'][d]
            np.testing.assert_allclose(group['mean_length'],lengths.mean(),rtol=0,atol=1e-12)
            np.testing.assert_allclose(group['p95_length'],np.quantile(lengths,.95),rtol=0,atol=1e-12)
        np.testing.assert_array_equal(a['initial__'+d+'__pins'],a['final__'+d+'__pins'])
        k='attack__'+d+'__'; shape=(p['attack_per_distribution'],p['attack_candidates'])
        score=a[k+'length']; pins=a[k+'pins']; top=a[k+'topology']
        if score.shape!=shape: raise AssertionError('attack shape')
        centers=G.make_data(p['attack_per_distribution'],p['degree'],d,'attack/'+d,seed=0)
        np.testing.assert_array_equal(pins[:,0],centers)
        rng=np.random.default_rng(G.derived_seed('heldout_attack',d,0))
        delta=rng.normal(0,p['search_sigma'],(len(centers),p['attack_candidates']-1,p['degree'],2)).astype(np.float32)
        candidates=G.legalize(centers,np.clip(delta,-p['epsilon'],p['epsilon']))
        np.testing.assert_array_equal(pins,np.concatenate((centers[:,None],candidates),axis=1))
        np.testing.assert_array_equal(a[k+'selected_index'],np.argmax(score,axis=1))
        np.testing.assert_allclose(score,G.lengths(pins.reshape(-1,p['degree'],2),top.reshape(-1,2*(p['degree']-1))).reshape(shape),rtol=0,atol=1e-12)
        np.testing.assert_allclose(a[k+'cost'],G.costs(pins[:,0],pins),rtol=0,atol=1e-12)
    return c,a

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs-root',default=str(G.ROOT/'artifacts/runs'))
    parser.add_argument('--outdir',default=str(G.ROOT/'artifacts/report'))
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--allow-cpu',action='store_true')
    args=parser.parse_args();G.install_oracle_guard()
    out=Path(args.outdir).resolve()
    if not out.is_relative_to(G.ROOT): raise ValueError('report path must remain in new branch')
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((G.ROOT/'protocol.json').read_text());methods=protocol['methods'];seeds=[101] if args.smoke else protocol['seeds']
    records={};data={};rows=[];attacks=[];budgets=[];comparisons=[];bindings={}
    for m in methods:
        for s in seeds:
            path=Path(args.runs_root)/f'{m}_s{s}'
            c,a=verify_run(path,protocol,m,s,args.smoke,args.allow_cpu);records[m,s]=c;data[m,s]=a
            bindings[str(path/'complete.json')]=G.sha256(path/'complete.json')
            updates=[json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines() if json.loads(line)['type']=='update']
            if [r['step'] for r in updates]!=list(range(1,c['step']+1)): raise AssertionError('update sequence')
            cost=np.array([r['mean_selected_cost'] for r in updates])
            budgets.append(dict(method=m,seed=s,updates=c['step'],rho=protocol['rho'],mean_cost=float(cost.mean()),
                last128_mean_cost=float(cost[-128:].mean()),mean_cost_minus_rho=float(cost.mean()-protocol['rho']),
                lambda_final=c['lambda_final'],committed_search_scores=c['ledger']['search_greedy_scores'],
                committed_sampled_scores=c['ledger']['sampled_actor_scores'],committed_rollout_scores=c['ledger']['rollout_greedy_scores'],
                elapsed_seconds=c['elapsed_seconds'],peak_reserved_bytes=c['peak_reserved_bytes'],
                actual_call_bounds=json.dumps(c.get('actual_call_bounds',{}),sort_keys=True)))
            for d in protocol['distributions']:
                for stage in ('initial','final'):
                    v=a[f'{stage}__{d}__length']
                    rows.append(dict(method=m,seed=s,distribution=d,stage=stage,count=len(v),mean=float(v.mean()),p95=float(np.quantile(v,.95))))
                k=f'attack__{d}__';idx=a[k+'selected_index'];ii=np.arange(len(idx));v=a[k+'length'][ii,idx]
                attacks.append(dict(method=m,seed=s,distribution=d,count=len(idx),candidates=protocol['attack_candidates'],
                    mean_length=float(v.mean()),p95_length=float(np.quantile(v,.95)),mean_center_length=float(a[k+'length'][:,0].mean()),
                    mean_lift=float((v-a[k+'length'][:,0]).mean()),mean_selected_cost=float(a[k+'cost'][ii,idx].mean())))
    for s in seeds:
        if len({records[m,s]['initial_actor_sha256'] for m in methods})!=1: raise AssertionError('unmatched initial Actor')
        for d in protocol['distributions']:
            for m in methods:
                np.testing.assert_array_equal(data[m,s][f'final__{d}__pins'],data[methods[0],s][f'initial__{d}__pins'])
                np.testing.assert_array_equal(data[m,s][f'attack__{d}__pins'],data[methods[0],seeds[0]][f'attack__{d}__pins'])
    for m in methods:
        for d in protocol['distributions']:
            for baseline in ('initial','erm_critic','wdro_random'):
                if baseline==m: continue
                delta=np.stack([data[m,s][f'final__{d}__length']-(data[m,s][f'initial__{d}__length'] if baseline=='initial' else data[baseline,s][f'final__{d}__length']) for s in seeds])
                comparisons.append(dict(method=m,baseline=baseline,distribution=d,training_seeds=len(seeds),nets_per_seed=delta.shape[1],
                    **paired_ci(delta,repeats=200 if args.smoke else 2000)))
    write_csv(out/'metrics.csv',rows);write_csv(out/'paired_comparisons.csv',comparisons);write_csv(out/'attacks.csv',attacks);write_csv(out/'budget_and_runtime.csv',budgets)
    lines=['# 仅使用强化学习自身反馈：自动报告','',
        '**CPU 工程 smoke；不能作为 GPU 科学结论。**' if args.smoke else '**18 个 degree30 GPU 任务的产物与数值审计通过；性能结论以配对统计为准。**','',
        '训练、基线、候选评分和选模只使用当前学习模型自身动作的线长。没有最优标签、MST、几何界、匹配求解器或外部预训练权重。',
        f'本报告 {len(seeds)} 个训练种子；逐 net 坐标配对、相同初始权重、源码/协议/产物哈希和锁均核验。所有测试/攻击线长从保存的 topology 独立重算。','',
        '| 方法 | 四组等权 mean 长度（种子平均） | 最差分布组 mean 长度 |','|---|---:|---:|']
    for m in methods:
        groups=[np.mean([records[m,s]['final_groups'][d]['mean_length'] for s in seeds]) for d in protocol['distributions']]
        lines.append(f'| {m} | {np.mean(groups):.6f} | {max(groups):.6f} |')
    lines+=['','## 解释范围','',
        '- 负配对差表示线长降低。分层 bootstrap 同时重采样训练种子和共享 net；种子数少，未做多重比较校正。',
        '- 线长下降可以直接验证；精确最优误差、最优率和近似比没有被测得。不能称作已经证明 regret 鲁棒性。',
        '- ERM-rollout 多用自身 Actor 一次贪心前向/训练样本；调用、耗时单列，不能只按更新数宣称计算效率胜出。',
        '- 主动生成完全依赖学习策略的路线为 WDRO-RL；Random、ES、梯度代理是无外部参考的对照提案机制。',
        '- 扰动成本为对应 pin 的直接平均 L1；它描述有标签坐标的运输集合，与旧无序集合匹配距离不同。',
        '- 单次有限 Random32 压力测试不是连续最坏情况证明；逐步对偶价格更新不保证全程平均成本严格合规。',
        '- evaluator 保留历史 REST 按 pin 段长记账规则；共线重叠可能双计，不能混称物理线段并集长度。','',
        '[mean/P95 逐组统计](metrics.csv) · [配对差及区间](paired_comparisons.csv) · [独立攻击](attacks.csv) · [预算与实际耗时](budget_and_runtime.csv)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    artifacts={name:G.sha256(out/name) for name in ('metrics.csv','paired_comparisons.csv','attacks.csv','budget_and_runtime.csv','REPORT.md')}
    G.atomic_json(out/'complete.json',dict(status='complete',acceptance='diagnostic_only' if args.smoke else 'accepted_gpu',created_at=time.time(),
        task_count=len(records),protocol_sha256=G.sha256(G.ROOT/'protocol.json'),run_bindings=bindings,artifacts=artifacts,
        all_oracle_calls_zero=True,all_reference_solver_calls_zero=True,all_run_locks_released=True))
    print(json.dumps(dict(status='complete',tasks=len(records),report=str(out/'REPORT.md'))))

if __name__=='__main__': main()

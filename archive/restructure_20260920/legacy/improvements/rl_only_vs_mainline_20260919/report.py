"""Independently validate shared inference and compare frozen model cohorts."""
import csv
import fcntl
import json
from pathlib import Path
import time
import numpy as np
from common import ROOT, G, GROUPS, SEEDS, read, sha, checked, atomic, independent_lengths, check_frozen


def write_csv(path, rows):
    tmp=path.with_suffix('.csv.partial')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)


def bootstrap(values, families, seed=20260919, repeats=2000):
    """[method, seed, group, net], independent seeds between training cohorts."""
    a=np.asarray(values,np.float64);m,s,g,n=a.shape
    assert s==3 and g==4
    rng=np.random.default_rng(seed);boot=np.empty((repeats,m,g),np.float64)
    family_ids={f:np.array([i for i,x in enumerate(families) if x==f]) for f in sorted(set(families))}
    for b in range(repeats):
        choices={f:rng.integers(3,size=3) for f in family_ids}
        for group in range(g):
            idx=rng.integers(n,size=n)
            mean=a[:,:,group,:][:,:,idx].mean(-1)
            for family,ids in family_ids.items():
                boot[b,ids,group]=mean[ids][:,choices[family]].mean(-1)
    return boot


def main():
    manifest=read(ROOT/'manifest.json');p=read(ROOT/'protocol.json')
    check_frozen(manifest)
    frozen=read(ROOT/'artifacts/operations/source_hashes.json')
    for name,digest in frozen['hashes'].items():checked(ROOT/name,digest)
    data={}
    for d in manifest['data']:
        with np.load(checked(d['path'],d['sha256']),allow_pickle=False) as z:
            data[d['suite'],d['group']]={k:z[k] for k in z.files}
    records={};values={};bindings={};metrics=[];budgets=[];independent_count=0
    for m in manifest['models']:
        folder=ROOT/'artifacts/runs'/m['id'];c=read(folder/'complete.json')
        assert c['status']=='complete' and c['acceptance']=='accepted_gpu'
        assert c['binding']['manifest_sha256']==sha(ROOT/'manifest.json') and c['binding']['model']==m
        assert not c['binding']['smoke'] and not c['forbidden_modules_loaded']
        assert c['identity']['device']=='cuda:0' and c['reference_solver_calls']==c['new_training_updates']==0
        for path,digest in c['binding']['sources'].items():checked(path,digest)
        for path,digest in c['binding']['dependencies'].items():checked(path,digest)
        assert c['binding']['geometry_sha256']==sha(Path(G.__file__))
        with (folder/'run.lock').open() as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
        bindings[str(folder/'complete.json')]=sha(folder/'complete.json')
        for name,digest in c['artifacts'].items():
            checked(folder/name,digest);bindings[str(folder/name)]=digest
        key=m['family']+':'+m['method'];seed=m['seed'];records[key,seed]=c
        assert len(c['cells'])==12
        assert {(v['suite'],v['group']) for v in c['cells']}==set(data)
        for cell in c['cells']:
            suite,group=cell['suite'],cell['group'];d=data[suite,group]
            assert cell['binding']['run']==c['binding']
            datum=next(v for v in manifest['data'] if (v['suite'],v['group'])==(suite,group))
            assert cell['binding']['data_sha256']==datum['sha256']
            with np.load(checked(cell['path'],cell['sha256']),allow_pickle=False) as z:
                length=z['length'];top=z['topology']
            pins=d['pins'].reshape(-1,30,2)
            assert cell['count']==len(pins)==len(length)
            np.testing.assert_allclose(G.lengths(pins,top),length,rtol=0,atol=1e-12)
            np.testing.assert_allclose(independent_lengths(pins,top),length,rtol=0,atol=2e-12)
            independent_count+=len(length)
            if suite=='rl_attack32':
                score=length.reshape(128,32);idx=score.argmax(1);value=score[np.arange(128),idx]
                center_mean=float(score[:,0].mean());cost=float(d['cost'][np.arange(128),idx].mean())
            else:value=length;center_mean=None;cost=None
            values[key,seed,suite,group]=value
            metrics.append(dict(model_id=m['id'],family=m['family'],method=m['method'],seed=seed,suite=suite,group=group,
                                count=len(value),mean=float(value.mean()),p95=float(np.quantile(value,.95)),
                                attack_center_mean=center_mean,attack_selected_cost=cost))
        events=[json.loads(x) for x in (folder/'calls.jsonl').read_text().splitlines()]
        starts={v['id']:v for v in events if v['event']=='start'};ends={v['id']:v for v in events if v['event']=='complete'}
        assert len(starts)+len(ends)==len(events) and ends.keys()<=starts.keys()
        for k,v in ends.items():
            for field in ('phase','real_rows','forward_rows'):assert v[field]==starts[k][field]
        expected=dict(completed_real_rows=sum(v['real_rows'] for v in ends.values()),
                      attempted_real_rows_upper_bound=sum(v['real_rows'] for v in starts.values()),
                      completed_forward_rows=sum(v['forward_rows'] for v in ends.values()),unresolved_calls=len(starts)-len(ends))
        assert c['actual_calls']==expected
        expected_new=40000 if m['family']=='rl_only' else 60480
        assert c['unique_new_inference_rows']==expected_new
        assert c['reused_rows']==(20480 if m['family']=='rl_only' else 0)
        assert expected['completed_real_rows']>=expected_new+(512 if m['family']=='rl_only' else 0)
        budgets.append(dict(model_id=m['id'],selected_update=c['selected_update'],new_rows=c['unique_new_inference_rows'],
                            reused_rows=c['reused_rows'],**expected,elapsed_seconds=c['elapsed_seconds'],
                            peak_reserved_bytes=c['peak_reserved_bytes']))
        print(json.dumps({'verified_model':m['id'],'saved_rows':60480}),flush=True)
    assert len(records)==33 and independent_count==1995840
    methods=[f'mainline:{x}' for x in p['mainline_methods']]+[f'rl_only:{x}' for x in p['rl_only_methods']]
    families=[m.split(':')[0] for m in methods]
    out=ROOT/'artifacts/report';out.mkdir(parents=True,exist_ok=True)
    summaries=[];comparisons=[]
    for suite_i,suite in enumerate(('rl_test','mainline_test','rl_attack32')):
        array=np.stack([np.stack([np.stack([values[m,s,suite,g] for g in GROUPS]) for s in SEEDS]) for m in methods])
        boots=bootstrap(array,families,seed=p['statistics']['seed']+suite_i,repeats=p['statistics']['replicates'])
        for i,m in enumerate(methods):
            per_group=array[i].mean((0,2))
            summaries.append(dict(suite=suite,method=m,mean=float(array[i].mean()),
                                  p95_mean=float(np.quantile(array[i],.95,axis=-1).mean()),
                                  worst_group_mean=float(per_group.max()),worst_group=GROUPS[int(per_group.argmax())],
                                  seed_means=';'.join(f'{v:.12g}' for v in array[i].mean((1,2)))))
            for baseline in p['statistics']['baselines']:
                if m==baseline:continue
                j=methods.index(baseline)
                for gi,g in enumerate(['equal_weight_4_groups']+GROUPS):
                    if gi==0:
                        v=float(array[i].mean()-array[j].mean());b=boots[:,i].mean(1)-boots[:,j].mean(1);base=float(array[j].mean())
                    else:
                        v=float(array[i,:,gi-1].mean()-array[j,:,gi-1].mean());b=boots[:,i,gi-1]-boots[:,j,gi-1];base=float(array[j,:,gi-1].mean())
                    lo,hi=np.quantile(b,[.025,.975])
                    comparisons.append(dict(suite=suite,method=m,baseline=baseline,group=g,difference=v,
                        relative_mean_difference_percent=100*v/base,ci_low=float(lo),ci_high=float(hi),
                        seed_resampling='paired_within_cohort' if families[i]==families[j] else 'independent_between_cohorts',
                        replicates=p['statistics']['replicates'],nets_per_group_per_seed=array.shape[-1]))
        print(json.dumps({'statistics_complete':suite,'methods':len(methods)}),flush=True)
    write_csv(out/'per_seed_metrics.csv',metrics);write_csv(out/'summary.csv',summaries)
    write_csv(out/'comparisons.csv',comparisons);write_csv(out/'inference_budget.csv',budgets)
    lines=['# 纯 RL 与 GroupDRO 等主线模型：统一评测','',
        '**33 个冻结 Actor 的统一评测与独立数值核验通过。** 11 类方法各 3 个 seed，degree 30；本次不重训、不调用 GeoSteiner 或读取最优标签。','',
        '## 比较口径','',
        '- 主线五方法：ERM、Mixed ERM、Hard Mining、GroupDRO、WDRO-REST；新方案六方法：ERM-Critic、ERM-rollout、Random/ES/Critic/RL-WDRO。',
        '- 所有方法在每套数据上收到逐字节相同的坐标，使用相同 REST Actor 架构、T=1 greedy、64 行固定前向及重复末行 padding。没有额外归一化或重新选模。',
        '- 新方案测试集为全部 4,096 个固定窗口 nets；主线测试集为全部 40,000 个 bbox 归一化 nets。两套数据生成器/尺度处理不同，分别报告。',
        '- Random32 是相同 512 个中心 × 32 个候选上的自身线长最大值；不能解释为 ρ 约束内最坏风险。',
        '- 主线模型使用 DAC21 预训练起点、2,000×256 微调及 relative-regret 参考反馈；新模型为随机起点、2,048×64 自身线长训练。**这是现有模型比较，不能把差异全部归因于算法或去掉 GeoSteiner。**','',
        '## 汇总','',
        '均值对四分布及三 seed 等权，越低越好。P95 先在各 seed/分布内计算再平均；最差组取种子均值后四组最大值。']
    titles={'rl_test':'新方案完整测试集（4,096 nets）','mainline_test':'主线完整测试集（40,000 nets）','rl_attack32':'新方案窗口攻击（512 centers × 32 candidates）'}
    for suite in titles:
        lines+=['',f'### {titles[suite]}','','| 方法 | mean 线长 | P95 平均 | 最差组 mean |','|---|---:|---:|---:|']
        for row in summaries:
            if row['suite']==suite:lines.append(f'| {row["method"]} | {row["mean"]:.6f} | {row["p95_mean"]:.6f} | {row["worst_group_mean"]:.6f} |')
    lines+=['','## 新方案相对 GroupDRO 的配对坐标比较','',
        '下表为方法减 GroupDRO 的四组平均差及探索性 95% 区间。跨两训练体系独立重采样各自的三个训练 seed，同时使用共同 net 重采样；没有把同名 seed 当成相同初始化。2,000 次 bootstrap，未做多重比较校正。','',
        '| 测试套件 | 新方法 | mean 差 | 相对均值差 | 95% 区间 |','|---|---|---:|---:|---|']
    for row in comparisons:
        if row['method'].startswith('rl_only:') and row['baseline']=='mainline:group_dro' and row['group']=='equal_weight_4_groups':
            lines.append(f'| {row["suite"]} | {row["method"]} | {row["difference"]:+.6f} | {row["relative_mean_difference_percent"]:+.2f}% | [{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}] |')
    lines+=['','## 验收与边界','',
        '- 33 个验证集选出的 checkpoint 来源哈希、输入数据、源码、输出拓扑、连通性和调用账本均核验；保存的 1,995,840 条长度由另一种 NumPy 实现独立重算。',
        '- 原 RL-only 两套结果复用前，对每个模型、每个分布/套件重放首 64 行并核对拓扑与线长；新主线测试集则对全部 33 个模型重新推理。',
        f'- 新增真实推理行 {sum(b["completed_real_rows"] for b in budgets):,}（包含兼容性重放及可能的恢复重放）；复用保存结果 {sum(b["reused_rows"] for b in budgets):,} 行。物理前向含 padding 另见预算表。',
        '- 线长为历史 REST span 记账值，共线重叠可能重复计入；没有测量精确 regret、最优率或近似比。',
        '- 本轮未更新任何 Actor 参数；未使用测试结果选 checkpoint、超参或方法。旧模型的训练历史仍包含参考算法，本次无求解器推理不改变其训练来源。',
        '- 同架构/同推理/同样本消除了评测口径差异；预训练、训练样本量、学习率、目标及选模等差异仍存在。需另有同起点同预算训练消融才能回答方法本身的因果优劣。',
        '- 三个训练 seed 数量有限；合成结果不自动推广至其他 degree 或公开芯片；窗口攻击不构成连续鲁棒性证明。','',
        '[逐 seed 指标](per_seed_metrics.csv) · [汇总](summary.csv) · [相对 GroupDRO/Mixed ERM/WDRO 的全部区间](comparisons.csv) · [实际推理预算](inference_budget.csv)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    artifacts={name:sha(out/name) for name in ('per_seed_metrics.csv','summary.csv','comparisons.csv','inference_budget.csv','REPORT.md')}
    atomic(out/'complete.json',dict(status='complete',acceptance='accepted_gpu',task_count=33,created_at=time.time(),
        manifest_sha256=sha(ROOT/'manifest.json'),protocol_sha256=sha(ROOT/'protocol.json'),source_hashes=frozen['hashes'],
        independent_length_rows=independent_count,run_bindings=bindings,artifacts=artifacts,
        new_inference_completed_real_rows=sum(v['completed_real_rows'] for v in budgets),
        reused_rows=sum(v['reused_rows'] for v in budgets),new_training_updates=0,reference_solver_calls=0))
    print(json.dumps({'status':'complete','models':33,'independent_rows':independent_count,'report':str(out/'REPORT.md')}),flush=True)

if __name__=='__main__':main()

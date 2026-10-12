"""Publish the independent acceptance report without modifying frozen evidence."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
p=read(ROOT/'protocol.json');base=read(OUT/'checkpoint_lifecycle_audit.json');rerun=read(OUT/'recomputed/audit.json')
assert base['status']=='PASS' and rerun['status']=='PASS'
test=(OUT/'tests.log').read_text();assert 'Ran 20 tests' in test and test.rstrip().endswith('OK')
byte_checks={name:sha(ROOT/'artifacts/report'/name)==sha(OUT/'recomputed'/name) for name in ['REPORT.md','metrics.csv','comparisons.csv']}
assert all(byte_checks.values()),byte_checks
stats=rows(OUT/'recomputed/comparisons.csv');coev=rows(OUT/'coev_summary.csv');fixed=rows(OUT/'fixed_summary.csv');budget=rows(OUT/'training_budget.csv')
def comparison(kind,suite,method,reference):
    for r in stats:
        if r['kind']!=kind or r['suite']!=suite or r['seed']!='aggregate':continue
        if (r['method'],r['reference'])==(method,reference):sign=1
        elif (r['method'],r['reference'])==(reference,method):sign=-1
        else:continue
        lo,hi=float(r['ci_low']),float(r['ci_high'])
        return dict(method=method,reference=reference,kind=kind,suite=suite,mean=sign*float(r['mean_difference']),low=lo if sign==1 else -hi,high=hi if sign==1 else -lo,all_seed_signs_agree=r['all_seed_signs_agree']=='True',seed_differences={k:sign*v for k,v in json.loads(r['seed_differences']).items()})
    raise KeyError((kind,suite,method,reference))
selected=[]
for m in p['learned_methods']:
    if m!='reinforce':selected.append(comparison('fixed','lambda5__gain',m,'reinforce'))
for method,ref,suites in [('sequential_ppo','random',['nominal_length','random32_length','es32_length']),('random','erm',['random32_length','es32_length']),('reinforce','erm',['nominal_length']),('population_ppo','ppo',['nominal_length','random32_length','es32_length'])]:
    for suite in suites:selected.append(comparison('coev',suite,method,ref))
with (OUT/'selected_comparisons.json').open('w') as f:json.dump(selected,f,ensure_ascii=False,indent=2)
actual_calls=[x['actual_call_log_totals'] for x in rerun['runs']]
overhead=sum(sum(x['retry_or_uncommitted_completed_overhead'].values()) for x in actual_calls)
inflight=sum(x['unpaired_inflight_starts'] for x in actual_calls)
require_zero=overhead==0 and inflight==0
assert require_zero
maximum=max(float(x['mean_cost']) for x in budget);late_max=max(float(x['last64_mean_cost']) for x in budget)
full_excess=sum(float(x['mean_gap'])>0 for x in budget);late_excess=sum(float(x['last64_gap'])>0 for x in budget)
report=[
'# RL 攻击器研究：2026-09-19 独立验收',
'',
'## 1. 验收结论',
'',
'**PASS：冻结探索协议的任务交付、来源绑定、检查点、统计复现与进程退出验收通过。** 共51任务，degree=30，seeds=201/202/203。该结论不表示每个方法有效，也不表示严格WDRO预算可行或得到连续空间鲁棒保证。',
'',
'原流水线于2026-09-16 04:15:13 UTC完成，队列墙钟约6小时49分钟，51任务均一次成功、无失败重试。此次验收没有重训、追加超参选择或新增GPU推理；保留原始报告及其哈希绑定。',
'',
'## 2. 本次实际检查',
'',
'| 检查 | 结果 |',
'|---|---|',
'| 正式任务与CUDA凭证 | 51/51，3共同起点 + 18固定攻击 + 30共同训练 |',
f'| 冻结文件与绑定 | {base["source_freeze_files"]}个冻结文件一致；原报告{base["original_verified_binding_count"]}项绑定重新核验通过（含重复引用） |',
'| 检查点内容 | 51个均有限；Actor/价值网络/优化器/攻击器更新数正确；固定阶段Actor未变；共同训练Actor实际变化 |',
'| 对偶和历史状态 | 全部256步序列连续，λ递推正确；MADDPG共同训练回放清空策略与3个历史对手状态符合协议 |',
'| 评分与调用 | 日志和账本一致；无额外恢复评分、无未配对调用、无外部oracle调用 |',
f'| 数据和数值 | 重建{base["original_unique_training_inputs"]:,}个独立训练中心检查held-out重合；原始数据几何/argmax/成本复算通过 |',
f'| 独立数值复算 | {base["metric_rows_recomputed"]:,}行指标；{base["duplicate_baseline_arrays"]:,}个重复对照数组逐项相同 |',
f'| 表与统计重现 | metrics.csv和comparisons.csv及REPORT.md均逐字节相同；{len(stats):,}行配对统计，2000次seed×共同中心交叉bootstrap |',
'| 工程测试 | 20/20再次通过；已有逐位恢复一致性证据仍完整 |',
'| 生命周期 | tmux已退出；原scheduler与51worker身份均不再活跃；52把调度/运行锁均释放；无formal partial文件 |',
'',
'评分预算合计：搜索greedy 2,949,120；评价greedy 1,769,472；sampled Actor 270,336。greedy含padding前向行合计6,193,152。51任务worker墙钟合计21.49小时（含主动等待和CPU阶段，不等于GPU活跃计算时间）。最大PyTorch allocated约230.96 MiB，未包含CUDA上下文/驱动总显存。',
'',
'## 3. 固定主模型的攻击能力',
'',
'以训练价格λ=5、四分布等权、三seed汇总；每中心8次自身Actor真实评分含原中心。指标是最大惩罚线长相对原中心的增益，越大表示更强的有限预算搜索。',
'',
'| 攻击器 | 平均gain | 相对REINFORCE均值差与95% CI |',
'|---|---:|---|'
]
for m in ['random','es',*p['learned_methods']]:
    r=next(x for x in fixed if float(x['price'])==5 and x['evaluated_attacker']==m and (x['method']==m or x['method']=='reinforce'))
    text='对照' if m in ('random','es','reinforce') else (lambda c:f'{c["mean"]:+.6f} [{c["low"]:+.6f}, {c["high"]:+.6f}]')(comparison('fixed','lambda5__gain',m,'reinforce'))
    report.append(f'| {m} | {float(r["mean_gain"]):.6f} | {text} |')
report += ['',
'PPO、best-of-K、序列PPO和MAPPO相对REINFORCE的区间均跨零，不能宣称改进成立，也不能宣称方法等价。MADDPG bandit变体相对REINFORCE的gain差约−0.739936，95% CI [−0.983301,−0.350020]，三个seed方向一致，当前配置表现更弱。λ=1和10属于训练价格外测试，完整结果保留在复算报告。',
'',
'## 4. 共同训练后的REST结果',
'',
'下表为四分布等权、再对三训练seed取均值的自身动作线长；越低越好。`常规测试`对应原表nominal_length，但这里包含全部四分布，不仅是uniform nominal分布。',
'',
'| 训练分支 | 常规测试均值 | Random32攻击均值 | ES32攻击均值 |',
'|---|---:|---:|---:|']
for x in coev:report.append(f'| {x["method"]} | {float(x["nominal_length_mean"]):.6f} | {float(x["random32_length_mean"]):.6f} | {float(x["es32_length_mean"]):.6f} |')
report += ['',
'- **序列PPO**：常规测试相对Random降低0.060031，95% CI [−0.103534,−0.027403]，三seed同向；约降低1.47%。但Random32/ES32下相对Random的区间均跨零，不能扩展为稳健性全面提升。',
'- **best-of-K PPO**：独立攻击点估计最低，但相对Random、ERM、REINFORCE的相关区间跨零；没有建立稳定胜出。',
'- **Random**：Random32/ES32相对ERM分别降低0.123864/0.132119，95% CI分别为[−0.217154,−0.019049]/[−0.226078,−0.038629]，三seed同向；应保留这一廉价对照。',
'- **REINFORCE**：常规四分布均值相对ERM降低0.043125，95% CI [−0.080879,−0.019882]，但独立攻击下没有明确改善。',
'- **MAPPO / MADDPG / 历史对手池**：没有建立相对对应简单对照的稳定综合优势。population相对PPO在三个评价套件的区间均跨零。',
'',
'以上为探索性、未做多重比较校正的区间。只有三个训练seed，所有不利结果保留；P95和worst-group仍为描述性统计，详见coev_summary.csv和原指标表。',
'',
'## 5. 运输预算：需要保留的限制',
'',
f'λ更新全部按冻结公式执行，但27个对抗共同训练任务中，**{full_excess}/27**的全程平均选中成本高于ρ=.04，最大为{maximum:.9f}（高约{(maximum/.04-1)*100:.2f}%）。最后64步仍有{late_excess}/27高于ρ，最大{late_max:.9f}（高约{(late_max/.04-1)*100:.2f}%）。',
'',
'这是有限训练的预算跟踪现象；不能把λ递推正确当成分布可行性证明。冻结协议没有规定严格有限时域可行性容差，因此保留该事实后可验收本轮探索实验；若按“严格满足Wasserstein平均预算的WDRO求解器”验收，目前证据不足。未为通过验收而调整ρ、λ或训练步数。逐run数据见training_budget.csv。',
'',
'独立Random32/ES32使用λ=0，搜索逐坐标±.08范围内的大线长；它们不是ρ=.04的可行分布求解，也不是全空间最坏值。历史对手方案是采样旧REST的困难输入课程，不等于实时REST的精确WDRO内层。',
'',
'## 6. 证据范围及下一步判断',
'',
'- 纯RL指标为自身动作线长，没有精确最优参考，不能报告relative regret或最优性差距。',
'- 已存nominal topology，本次全量复算其线长。攻击候选存了坐标/评分/选中索引，未存所有topology；本次核验来源、账本、几何、成本和argmax，没有重新推理攻击树。因此不把候选得分称为重新运行模型独立确认。',
'- 三seed与degree30、256步共同起点+256步继续训练属于短程机制验证，不足以判断最终收敛、其他degree或芯片泛化。',
'- 当前证据支持保留Random/REINFORCE基准、优先关注序列PPO和best-of-K的可检验机制；不支持直接把整个项目迁移为MAPPO或MADDPG。任何扩展需要新协议与新实验，本轮未擅自追加。',
'',
'## 7. 复核产物',
'',
'- `acceptance.json`：本次最终验收状态和文件哈希。',
'- `checkpoint_lifecycle_audit.json`：独立检查点/进程/λ/账本检查。',
'- `recomputed/`：重新运行冻结报告得到的完整指标、配对统计和审计。',
'- `training_budget.csv`、`runtime_and_calls.csv`、`coev_summary.csv`、`fixed_summary.csv`：本次独立汇总。',
'- `tests.log`、`recompute.log`、`independent_verify.log`：执行证据。',
'']
(OUT/'REPORT.md').write_text('\n'.join(report))
acceptance={'status':'PASS_FOR_FROZEN_EXPLORATORY_PROTOCOL','reviewed_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'task_count':51,'tests_passed':20,'recomputed_report_status':rerun['status'],'byte_identical_original_outputs':byte_checks,
    'strict_wdro_budget_feasibility':'NOT_ESTABLISHED','training_mean_cost_above_rho_runs':full_excess,'last64_mean_cost_above_rho_runs':late_excess,
    'new_training_or_gpu_inference':False,'scientific_scope':'own-action wirelength; degree30;3seeds; finite candidate search; unadjusted exploratory CIs',
    'errors':[],'artifacts_sha256':{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='acceptance.json'}}
(OUT/'acceptance.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in acceptance.items() if k!='artifacts_sha256'},ensure_ascii=False,indent=2))

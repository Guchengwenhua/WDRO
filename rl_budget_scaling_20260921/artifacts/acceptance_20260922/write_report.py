"""Publish acceptance findings from the independent audit, without changing frozen runs."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

a=read(OUT/'audit.json'); stats=read(OUT/'statistics_independent.json'); p=read(ROOT/'protocol.json')
assert a['status']=='PASS_FOR_FROZEN_THREE_SEED_EXPLORATORY_PROTOCOL'
comparisons=stats['comparisons']
assert len(comparisons)==6
for observed,expected in zip(comparisons[:5],p['primary_comparisons']):
    assert all(observed[k]==expected[k] for k in ('name','baseline','candidate','metric'))
assert comparisons[5]['baseline']=='SG/step6144' and comparisons[5]['candidate']=='SG/step12288' and not comparisons[5]['primary']
ops=ROOT/'artifacts/operations'; exit=read(ops/'exit.json'); launch=read(ops/'launch.json')
attempts=[read(f) for f in ops.glob('result_*.json')]
success=sorted([r for r in attempts if r['status']=='exited'],key=lambda r:r['started_at'])
for i,first in enumerate(success):
    for second in success[i+1:]:
        if second['started_at']<first['finished_at']:
            assert first['gpu']!=second['gpu'], 'Overlapping GPU leases'
            assert not set(first['cpu_affinity'])&set(second['cpu_affinity']), 'Overlapping experiment CPU assignments'
for f,h in read(OUT/'input_hashes.json').items():
    assert sha(f)==h,'Frozen input changed during acceptance'
supplement=dict(status='PASS',exact_primary_comparison_list=True,no_within_experiment_gpu_lease_overlap=True,
                no_within_experiment_cpu_assignment_overlap=True,all_frozen_inputs_unchanged_after_verification=True)
write(OUT/'final_delivery_checks.json',supplement)

labels=['More training: U 2048 → 8192 (mean)', 'Mixed vs uniform at 8192 (mean)',
        'GroupDRO vs continued ERM (worst)', 'Pretrain then GroupDRO vs direct (worst)',
        'Long vs short pretrain, equal total (worst)']
fig,ax=plt.subplots(figsize=(10.5,4.8),layout='constrained')
for i,c in enumerate(comparisons[:5]):
    y=4-i; lo,hi=c['simultaneous_ci99']; value=c['improvement']
    color='#1b7837' if c['supports'] else '#b2182b' if hi<0 else '#777777'
    ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',color=color,ecolor=color,capsize=5,linewidth=2)
    ax.scatter(c['seed_improvement'],[y]*3,s=18,color=color,alpha=.35)
    ax.text(.32,y,f"{c['improvement_percent']:+.2f}%",va='center',color=color,fontweight='bold')
ax.axvline(0,color='#333333',linestyle='--',linewidth=1)
ax.set_yticks(range(5),labels[::-1]); ax.set_xlim(-.1,.36)
ax.set_xlabel('Baseline − candidate own-action length (positive = improvement)')
ax.set_title('Independent acceptance: five preregistered comparisons\n99% paired hierarchical-bootstrap intervals; three training seeds (exploratory)',fontsize=11)
ax.grid(axis='x',alpha=.2)
for ext in ('png','svg','pdf'): fig.savefig(OUT/f'primary_comparisons.{ext}',dpi=180)
plt.close(fig)

names={'training_budget':'基础训练量：U 2,048→8,192','data_coverage':'数据覆盖：M8,192 vs U8,192',
 'robust_finetuning':'稳健后训练：MG12,288 vs ME12,288','pretraining_order':'顺序：MG12,288 vs D12,288',
 'stage_allocation':'阶段分配：MG12,288 vs SG12,288','posttraining_scale_exploratory':'后训练延长：SG6,144→12,288'}
weights=list(csv.DictReader((OUT/'group_weights_final.csv').open()))
qmin=min(float(r['edge_bias']) for r in weights)*100; qmax=max(float(r['edge_bias']) for r in weights)*100
start=dt.datetime.fromtimestamp(launch['at'],dt.timezone.utc).isoformat(); finish=dt.datetime.fromtimestamp(exit['at'],dt.timezone.utc).isoformat()
text=['# 纯 RL 训练规模实验：独立验收报告','',
 '**验收通过：`PASS_FOR_FROZEN_THREE_SEED_EXPLORATORY_PROTOCOL`。** 工程执行与数值有效性通过；科学结论限定于 degree30、三个种子和本次冻结的合成数据协议。',
 '', '核心结论：扩大基础训练预算确实改善了本轮结果，支持“训练不足是旧结果的原因之一”。相同总预算下，先做等权预训练再做 GroupDRO 优于直接 GroupDRO；但继续 ERM 在本轮比 GroupDRO 后训练更好。没有证据把全部历史差距归因于训练量或 GeoSteiner 的有无。',
 '', '## 1. 工程与数据验收', '',
 '|检查|独立核验结果|','|---|---|',
 '|任务完整性|18/18 训练，18/18 新测试，36 个成功退出；另有1次资源准入延后，无训练重跑|',
 f"|耗时|{start} 至 {finish}，队列墙钟 {a['queue_wall_hours']:.4f} 小时；训练 worker 合计 {a['training_worker_hours']:.4f} 小时，测试 worker 合计 {a['evaluation_worker_hours']:.4f} 小时|",
 f"|哈希|核对 {a['counts']['hash_bindings']} 项绑定，涉及 {a['unique_files_hashed']} 个独立文件；验收结束再次核对全部输入未变|",
 f"|生命周期|核验 {a['counts']['exited_process_identities']} 个正式进程身份已退出、{a['counts']['released_locks']} 个锁已释放；延后尝试的 wrapper 也已退出|",
 '|资源|准入三次利用率≤35%、空闲显存≥8GiB；实验内GPU和CPU分配无时间重叠|',
 f"|训练预算|完整核对 {a['total_new_updates']:,} 步；新增 {a['total_sampled']:,} sampled 和 {a['total_rollout']:,} rollout 评分；无未完成调用、无重放|",
 f"|验证/测试评分|验证 {a['total_validation_scores']:,} 行；测试 {a['total_test_scores']:,} 行，96 个固定端点或选中模型|",
 f"|独立数值复算|检查连通树拓扑并重算 {a['counts']['topologies_reduced']:,} 条自身线长；与保存数组最大差 {a['maximum_cost_reduction_error']}|",
 '|权重谱系|3个随机初始化可重建；12次父checkpoint的Actor（含BatchNorm）、Adam、RNG逐项相等|',
 '|数据配对|全部对应训练步的四分布批次哈希匹配；另外重新生成24批输入核对哈希|',
 '|数据隔离|验证/测试24,576个net无重复，训练probe无重合；训练进程禁止打开测试文件，调用账本无测试评分|',
 f"|测试时序|最后训练结束 {a['last_training_finished_utc']}；第一笔测试评分 {a['first_test_scoring_utc']}；所有测试选择清单绑定全部18个已完成训练|",
 '|选模与目标|完整复核每512步验证、最坏组最小及均值退化容忍规则；GroupDRO始终用历史组权重，更新与记录一致|',
 '|依赖|冻结的奖励与策略代码无外部构造求解器调用；oracle计数为0。独立验收仅CPU读取产物，没有新训练、推理或GPU调用|',
 '', '审核没有修改训练源码、协议、checkpoint、原始日志或原始自动报告。独立实现见 [verify.py](verify.py)，逐项凭证见 [audit.json](audit.json)、[verify.log](verify.log) 与 [input_hashes.json](input_hashes.json)。数据probe不是对所有历史训练坐标逐net重新枚举，完整配对检查以逐步批次哈希为证据。',
 '', '## 2. 五项预注册比较与一项补充分析','',
 '以下均为相同新测试集上的固定预算端点。改善=(基线−候选)/基线；正数表示线长下降。主要比较的99%区间来自5000次种子及net配对的双层bootstrap，对五项比较采用Bonferroni名义族覆盖95%。独立重算与原报告数值一致（核对容差1e-10）。',
 '', '|比较|指标|基线→候选|相对改善|绝对改善99%区间|改善种子|判断|','|---|---|---|---:|---|---:|---|']
for c in comparisons:
    verdict='达到预注册探索支持标准' if c['supports'] else '候选更差，区间完全为负' if c['direction']=='candidate_worse_with_corrected_interval' else '尚未确立收益'
    if not c['primary']: verdict='补充探索，未纳入五项主要检验'
    text.append(f"|{names[c['name']]}|{'平均' if c['metric']=='mean' else '最坏组'}|{c['baseline_value']:.6f} → {c['candidate_value']:.6f}|{c['improvement_percent']:+.3f}%|[{c['simultaneous_ci99'][0]:.6f}, {c['simultaneous_ci99'][1]:.6f}]|{c['improved_seeds']}/3|{verdict}|")
text+=['', '“支持”的阈值仍是冻结协议：改善≥0.5%、至少2/3种子改善、校正区间下界>0；稳健后训练还要求均值恶化≤1%。MG 相比 ME 的均值反而恶化 **2.014%**，因此也没有满足该条件。',
 '', '原自动报告将所有未达到正向标准的比较统一写为“本预算下尚未确立”。本验收补充更明确的解释：MG 对 ME 不是单纯区间跨零，而是三个种子均更差，校正区间全负。这是报告解释的细化，数值不需要更正。',
 '', '![主要比较](primary_comparisons.png)',
 '', '## 3. 固定端点和验证选模结果','',
 'U=uniform ERM；M=四分布 ERM 预训练；D=直接 GroupDRO；ME=M8192 后继续 ERM；MG=M8192 后 GroupDRO；SG=M2048 后 GroupDRO。均值为四组等权、再平均三个种子；最坏组先对每个模型取四组最大均值，再平均种子。P95先在每个模型的16,384个测试net上取95%分位，再平均种子。',
 '', '|路线|checkpoint|均值|最坏组|P95|三个种子的选模步数|','|---|---|---:|---:|---:|---|']
for r in stats['model_summary']:
    text.append(f"|{r['arm']}|{r['model']}|{r['mean']:.6f}|{r['worst']:.6f}|{r['mean_p95']:.6f}|{r['steps']}|")
text+=['', '**本轮冻结方案中，ME 的测试均值和最坏组均最低**：四分布等权预训练8,192步，再继续ERM4,096步。它的三个验证最佳checkpoint都在12,288步。该排名是本轮观察，不把跨方案测试排名当作新的确认性试验。',
 '', '描述性预算曲线：']
for r in stats['descriptive']:
    text.append(f"- {r['arm']}：{r['baseline']}→{r['candidate']}，测试平均线长下降 {r['mean_improvement_percent']:.3f}%，三个种子均改善。")
text+=['', '这说明增加训练仍有价值，但不保证未来按相同速度改善，也没有确立充分训练后的误差下限。ME延长训练同时跨过预设学习率阶段，不能把这部分收益完全拆解成更新次数的独立效应。',
 '', '## 4. GroupDRO权重与解释边界','',
 f'9个 GroupDRO 运行的终点 `edge_bias` 权重范围为 **{qmin:.5f}%–{qmax:.7f}%**，其余分布的权重接近零。逐步复算确认这符合冻结的原始线长乘法权重更新，不是使用当前批次新权重回传的实现错误。详见 [group_weights_final.csv](group_weights_final.csv)。',
 '', '该现象与“原始长度混入各组固有难度、权重可能高度集中”的事前预警一致，可以作为后续机制消融的线索；尚不能据此证明权重集中就是退化的唯一原因。不要在本次测试之后修改η、加权重下限或换目标，再沿用本次测试作盲测。',
 '', '结论限于三种子探索。三个种子全部同向也不足以构成大量独立训练重复；例如仅对3/3方向一致做精确双侧符号检验，p=0.25。bootstrap区间与预注册门槛在这里用于冻结的探索性判断，不能代替更多训练种子的确认。',
 '', '本轮没有使用最优参考，因此指标是自身拓扑 REST span 线长，不是真实最优相对误差；没有重新评测 DAC21 主线、真实芯片或连续Wasserstein最坏风险，不能声称已追平旧主线或证明学习型WDRO优于ERM。长/短预训练同总预算的最坏组差异只有0.110%且区间跨零，不能写成“预训练越长越好”。',
 '', '## 5. 验收产物','',
 '- [机器验收结论](audit.json)、[最终交付核查](final_delivery_checks.json)、[输入SHA256](input_hashes.json)。',
 '- [独立统计](statistics_independent.json)、[逐组指标](metrics_independent.csv)、[预算与耗时](runtime_and_calls.csv)、[模型摘要](model_summary.csv)。',
 '- 比较图：[PNG](primary_comparisons.png)、[SVG](primary_comparisons.svg)、[PDF](primary_comparisons.pdf)。',
 '- [原始训练曲线](../report/learning_curves.png)与[原自动报告](../report/REPORT.md)保留不变。',
 '', '本次验收已完成，未启动额外实验。']
(OUT/'ACCEPTANCE.md').write_text('\n'.join(text)+'\n')
artifacts={f.name:sha(f) for f in OUT.iterdir() if f.is_file() and f.name not in ('complete.json','write_report.log')}
write(OUT/'complete.json',dict(status=a['status'],completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    protocol_sha256=sha(ROOT/'protocol.json'),original_report_completion_sha256=sha(ROOT/'artifacts/report/complete.json'),
    artifacts=artifacts,scope='Independent acceptance; three-seed exploratory conclusions only',new_gpu_work=False))
print('Acceptance report, figures, final source recheck, and completion receipt written.')

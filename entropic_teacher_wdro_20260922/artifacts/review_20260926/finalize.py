"""Produce acceptance report; original frozen experiment is kept intact."""
import csv,hashlib,json,math,time,datetime
from pathlib import Path
import numpy as np
ROOT=Path('/remote-home/Guilin/DRO/Codex/entropic_teacher_wdro_20260922');ART=ROOT/'artifacts';OUT=ART/'review_20260926'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cpu=read(OUT/'cpu_audit.json');gpu=[read(OUT/f'gpu_replay_{i}.json') for i in range(4)];teacher=read(OUT/'teacher_replay.json')
data=read(OUT/'data_isolation.json');comparisons=read(OUT/'primary_comparisons.json')['corrected_exact_empirical_cvar'];sensitivity=read(OUT/'sensitivity_remove_engineering_overlap.json')
summary=list(csv.DictReader((OUT/'method_summary.csv').open()));budgets=list(csv.DictReader((OUT/'budgets.csv').open()))
assert cpu['status']=='PASS' and all(g['status']=='PASS' for g in gpu) and teacher['status']=='PASS'
assert sensitivity['scientific_pass_count']==0 and not any(c['scientific_pass'] for c in comparisons)
predictions=sum(g['totals']['predictions'] for g in gpu);arrays=sum(g['totals']['arrays'] for g in gpu)
models=sum(g['totals']['models'] for g in gpu)
finished=datetime.datetime.fromtimestamp(cpu['delivered'],datetime.timezone.utc).isoformat()
wall=(cpu['delivered']-cpu['earliest_training'])/3600
pretty={'soft_budget':'熵预算 SOFT-BUDGET','hard':'HARD','anneal':'温度延续 ANNEAL','excess':'教师相对损失 EXCESS','tail':'尾部训练 TAIL'}
lines=[
'# 熵预算、教师相对损失与尾部风险实验：独立验收',
'',
'验收日期：2026-09-26 UTC。范围仅为 `entropic_teacher_wdro_20260922`。本报告从原始检查点、日志和逐样本数组复核，未重训、未依据测试调参，也未覆盖原始冻结报告。',
'',
'## 验收结论',
'',
'**有条件通过：执行、预算、选模和数值复现通过；四个新方法均未达到预注册稳定改善门槛。** 报告的CVaR计算已在本验收副本中修正；启动前工程自检曾复用少量正式测试坐标，严格的“测试从未提前访问”条件有例外。另有一对训练批次发生32位派生种子碰撞。这些事实均保留，不能把验收写成无保留的协议全部通过。',
'',
'教师方法是四个主比较中唯一正向趋势：相对SOFT-BUDGET，三OOD分布平均CVaR90改善约0.298%，7/8种子改善；但原始单侧p=0.05078125、Holm p=0.203125，且小于预设0.5%实际收益门槛，未建立稳定优势。',
'',
'## 完成与证据',
'',
f'- 116/116项训练、224/224项best/last模型评价已完成；完成时间 `{finished}`，从首次正式计算至交付约{wall:.2f}小时。训练worker累计{cpu["worker_hours"]:.2f}小时，非整机占用时长。',
f'- 独立核对{cpu["files_hashed"]:,}个文件哈希，包括冻结源码/协议、232个训练检查点、2,688个逐样本分片及日志/凭证；模型冻结链一致。所有正式评价都晚于模型冻结。',
'- 116个训练checkpoint的Actor/Value/Adam状态均有限，更新计数一致；逐步日志连续，没有重复/回退训练记录；从验证日志重新执行选模和温度选择，结果一致。共同温度为0.08，所有best均在update0之后。',
'- 所有记录的训练/评价进程已经退出，340个worker内核锁及scheduler锁已释放；队列为空，原tmux服务已退出。未发现失败重试记录。',
f'- 验收重新加载全部{models}个正式评价模型，每模型回放四分布各256个自然样本；另对三degree×七方法、seed901的best模型回放完整Uniform Random64/ES64攻击。共重新计算{predictions:,}次动作预测，{arrays}组保存输出完全一致，最大绝对误差0。中间攻击候选未全部留存，因此比较的是保存的最大值及原点输出。',
f'- 另重算14个EXCESS模型的冻结教师，对最后留存候选执行{teacher["predictions"]:,}次教师前向；重建候选概率与保存值完全一致，期望成本、原始目标亦通过。',
'',
'| 训练与评价账本 | 实测数量 |',
'|---|---:|',
f'| Actor更新 | {cpu["totals"]["updates"]:,} |',
f'| sampled Actor训练样本 | {cpu["totals"]["sampled"]:,} |',
f'| 对抗候选Actor greedy评分 | {cpu["totals"]["search"]:,} |',
f'| 额外冻结教师greedy评分 | {cpu["totals"]["teacher"]:,} |',
f'| 训练期validation评分 | {cpu["totals"]["validation"]:,} |',
f'| 最终自然/攻击评分 | {cpu["evaluation_calls"]["test"]:,} |',
'',
'这些是计算样本数；同seed方法间配对重复，不是独立新数据规模。EXCESS每中心多出16次教师评分，ERM省去候选搜索，不能声称各方法总算力相等。逐任务账本见 `budgets.csv`。',
'',
'## 预注册主比较：degree30、8个训练种子、best模型',
'',
'终点为三OOD分布精确经验CVaR90的等权平均。改善为“对照−方法”，正值表示更好；相对改善先逐seed计算再平均。p值为256种符号的单侧枚举，四比较Holm校正。符号检验依赖零假设下差值对称/符号可交换；bootstrap CI是描述性seed级区间。',
'',
'| 方法 | 对照 | 尾部相对改善 | 改善seed | Holm p | 稳定改善门槛 |',
'|---|---|---:|---:|---:|---|'
]
for c in comparisons:lines.append(f'| {pretty[c["method"]]} | {pretty.get(c["reference"],c["reference"])} | {c["relative_improvement"]:+.3%} | {c["improving_seeds"]}/8 | {c["holm_p"]:.6f} | 未通过 |')
lines += ['',
'SOFT-BUDGET和ANNEAL在该协议下没有带来额外收益；TAIL的8个seed方向均较对照差。不能从未通过推导所有未来设置无效，也不能把主比较外的某个均值优势升格为显著性结论。',
'',
'## 七方法和共同父模型的完整主度数概览',
'',
'以下均为degree30八seed均值、best模型，全部指标越小越好。Natural OOD和CVaR90平均三个OOD分布；Random64/ES64平均全部四分布。',
'',
'| 方法 | Uniform mean | OOD mean | OOD CVaR90 | Random64 mean | ES64 mean |',
'|---|---:|---:|---:|---:|---:|']
for r in summary:
    if r['degree']=='30' and r['checkpoint']=='best':lines.append('| '+r['method']+' | '+' | '.join(f'{float(r[k]):.6f}' for k in ['uniform_mean','ood_mean','ood_cvar90','random64_mean','es64_mean'])+' |')
lines += ['',
'共同父模型到继续训练ERM已有明显描述性下降；不能把相对父模型的全部收益归因于新WDRO机制。EXCESS与HARD接近，现有证据不足以宣布教师方法优于HARD。degree20/40各只有3个seed，作为探索性外推报告；所有degree×方法×best/last汇总见 `method_summary.csv`，逐分布/逐seed/逐测试套件见 `metrics_corrected.csv`，没有按结果筛掉负值。',
'',
'## 运输预算与目标语义',
'',
'对全部训练日志核查期望成本与对偶间隙，并独立重算每个对抗run留存候选的成本、概率归一化与抽样成本。熵预算、ANNEAL、EXCESS、TAIL的逐批期望成本均≤0.04+1e−10。原始HARD/SOFT-DUAL仅靠递推价格，不保证每批可行，degree30的最大批期望成本分别约0.05105/0.05394。精确预算机制实现了其预定数值约束，但不等于Actor性能改善。',
'',
'抽样后的单批成本仍可能超过0.04，这是有限候选概率分布的期望约束，不是每次抽样的硬预算；两者在 `transport_diagnostics.csv` 分开报告。训练目标是固定窗口中的Actor自身拓扑线长；EXCESS分母是冻结父模型的greedy线长，不是真正最优解。无GeoSteiner、无最优regret保证，也没有连续Wasserstein最坏风险证书。',
'',
'## 本次发现和处理',
'',
'1. **CVaR报告修正。** 原实现将`np.quantile(x,0.9)`的线性插值结果直接代入CVaR变分式；有限样本下，该插值不总是经验CVaR的最小化阈值。现按最高10%样本的分数权重均值精确计算：令m=n/10，取最大的floor(m)个样本，再加下一样本的剩余权重，除以m。原始源码、数组、报告和冻结哈希全部保留，修正结果仅写本验收目录。',
f'   自然测试单格最大修正{cpu["max_cvar_correction"]["natural"]:.10g}，攻击测试单格最大修正{cpu["max_cvar_correction"]["attack"]:.10g}。四个主比较的p值、校正结果和最终判定全部不变。原公式的全部2,688行指标和四个比较也已独立复算一致。',
'',
'2. **预先访问测试坐标的工程例外。** 启动前的小规模seed17评价自检使用了正式test/attack命名空间；与正式Uniform自然测试重叠8个net、与Uniform攻击中心重叠4个net。其他三分布未发现该自检坐标重叠。该自检模型不是正式模型，调温与选模代码仅读取validation；未发现测试驱动的调参，但本实验不能表述为测试数据从未被任何自检提前访问。',
'   额外作了剔除这8/4个重叠坐标的敏感性分析，四个主比较仍全部未通过；三OOD主终点本身不受这些Uniform重叠影响。详见 `sensitivity_remove_engineering_overlap.json`。未来新协议应使用独立的engineering namespace。',
'',
'3. **训练种子碰撞。** 32位派生seed使degree30的post seed902、零基step259，与pre seed904、零基step1814生成相同128个中心；已逐坐标核实。首次采样的136,703个不同RNG键中未发现与heldout命名空间的键碰撞，留存训练审计坐标也未与validation/test/attack重复。该重复属于训练内部，不能把全部采样次数称为全部唯一新样本。未来应使用更宽的派生种子。',
'',
'数据隔离检查覆盖已生成的正式holdout集合、留存候选中心、全训练首轮RNG键及heldout各重试键；没有重新生成全部约8,684万训练样本，不能将它表述为对所有在线几何输入的穷尽去重证明。',
'',
'## 产物和复核入口',
'',
'- `acceptance.json`：本轮结论、边界与证据哈希；原自动delivery的COMPLETE只表明执行完成。',
'- `independent_audit.py`：CPU全量审计与GPU回放代码；`teacher_replay.py`：教师概率重建。',
'- `cpu_audit.json`、`gpu_replay_0..3.json`、`teacher_replay.json`：数值和生命周期证据；原CPU摘要对教师历史分母的警告已由单独教师回放补足。',
'- `primary_comparisons.json`：精确CVaR修正与原公式复现并列，包含seed差值、CI、p值与门槛。',
'- `data_isolation.json`、`sensitivity_remove_engineering_overlap.json`：隔离范围、重复样本和敏感性结果。',
'- `method_summary.csv`、`metrics_corrected.csv`、`budgets.csv`、`transport_diagnostics.csv`、`checkpoint_selection.csv`：完整结果表。',
'',
'本轮验收没有追加训练、扩大检验族、修改显著性阈值或清理其他人的作业。所有新文件保存在当前方法的附加盘artifacts入口中。'
]
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
# Keep an easy-to-find entry without modifying any frozen document.
entry=ROOT/'验收报告_20260926.md'
if not entry.exists():entry.symlink_to('artifacts/review_20260926/REPORT.md')
bindings={str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='acceptance.json' and p.suffix in ('.py','.json','.csv','.md','.log')}
result=dict(status='ACCEPTED_WITH_CAVEATS',engineering_and_numerical='PASS',strict_never_accessed_holdout='EXCEPTION_DOCUMENTED',reporting='CVaR_CORRECTED_IN_ACCEPTANCE_COPY',scientific_superiority='NOT_ESTABLISHED',all_primary_comparisons_pass=False,original_delivery_sha256=sha(ART/'operations/delivery.json'),source_freeze_sha256=sha(ART/'operations/source_freeze.json'),counts=dict(training=116,evaluation=224,files_hashed=cpu['files_hashed'],updates=cpu['totals']['updates'],gpu_replay_predictions=predictions,gpu_replay_arrays=arrays,teacher_replay_predictions=teacher['predictions']),caveats=['prelaunch engineering reuse: 8 Uniform natural and 4 attack centers','one internal training batch duplicate from 32-bit seed collision','CVaR percentile interpolation corrected; decisions unchanged'],comparisons=comparisons,evidence_sha256=bindings,time=time.time())
(OUT/'acceptance.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_sha256','comparisons')},indent=2,ensure_ascii=False))

"""Publish the fresh acceptance summary after all numerical checks pass."""
from pathlib import Path
import csv
import hashlib
import json
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'artifacts/acceptance_20260916'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_text(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(value)
    temporary.replace(path)


audit = read(OUT / 'acceptance_final.json')
original = read(OUT / 'original_completion_recheck.json')
stats = read(OUT / 'final_statistics/statistics_review.json')
assert audit['status'] == original['status'] == 'PASS'
assert audit['completed_formal_runs'] == 18 and len(audit['smokes']) == 6
assert not audit['missing'] and audit['independent_length_rows'] == 444096
assert stats['status'] == 'NUMERICAL_REVIEW_PASS' and not stats['errors'] and not stats['missing']
assert sha(OLD / 'acceptance_final.json') == original['original_final_sha256']
assert sha(OLD / 'watcher_exit.json') == original['original_watcher_exit_sha256']
assert audit['auditor_sha256'] == sha(OUT / 'audit_acceptance.py')
for name, digest in audit['independent_statistics']['artifacts'].items():
    assert sha(OUT / name) == digest, name

# The two reviews use identical paired statistics. Their metadata timestamps and
# output-directory fields differ, while the four numerical CSV files must agree.
independent_csv = {}
for suffix in ('runs', 'groups', 'attacks', 'pairwise'):
    name = f'final_statistics/statistics_review_{suffix}.csv'
    assert (OLD / name).read_bytes() == (OUT / name).read_bytes(), name
    independent_csv[name] = sha(OUT / name)

protocol = read(ROOT / 'protocol.json')
budgets = []
with (ROOT / 'artifacts/report/budget_and_runtime.csv').open() as stream:
    budgets = list(csv.DictReader(stream))
totals = {field: sum(int(row[field]) for row in budgets)
          for field in ('updates', 'committed_search_scores', 'committed_sampled_scores', 'committed_rollout_scores')}
assert totals == dict(updates=36864, committed_search_scores=12582912,
                     committed_sampled_scores=2359296, committed_rollout_scores=393216)

now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
pairs = [row for row in stats['paired_comparisons'] if row['distribution'] == 'equal_weight_4_groups']
methods = {row['method']: row for row in stats['method_summaries']}
minimum = min(row['reduction_percent'] for row in stats['runs'])
maximum = max(row['reduction_percent'] for row in stats['runs'])
transport = [row['mean_training_cost'] for row in stats['method_summaries'] if row['mean_training_cost'] is not None]
cost_percent = [(value / protocol['rho'] - 1) * 100 for value in transport]
attack_fraction = [row['attack_selected_over_rho_fraction'] * 100 for row in stats['method_summaries']]

lines = [
    '# RL-only 实验整批验收报告', '',
    f'复验时间：{now}。范围：`improvements/rl_only_20260915`，六方法 × 三个训练种子，degree 30。', '',
    '**结论：冻结实验整批验收通过。18/18 个正式 GPU 模型及 6/6 个 GPU smoke 已完成；本轮重新运行完整验收，数值错误为 0。**',
    '原后台验收已于 2026-09-16 完成。本轮单独保存新凭证，并确认原结果与原验收凭证未变化。', '',
    '## 1. 工程与数值证据', '',
    f'- 复核原完成链的 {original["files_checked"]} 个文件哈希；当前全量验收绑定 {len(audit["artifact_bindings"])} 个任务产物/完成标记。',
    f'- 独立重算 {audit["independent_length_rows"]:,} 条保存拓扑的记账线长，包含初始、最终、全部攻击候选及 smoke。',
    '- 初始 Actor 权重在 CPU 上从三个种子重新构造并核对；检查点及优化器张量有限，validation 选模和最早并列规则一致，λ 递推与调用账本一致。',
    '- 原报告与四张正式 CSV 重新生成后逐字节相等；逐分布配对 bootstrap 再由独立实现复算，均通过。',
    '- 另一独立统计程序也重新运行，其四张数值 CSV 与 9 月 16 日完整统计逐字节相等。',
    f'- 原调度器、记录的训练/包装进程均已退出，{audit["runtime"]["locks_released"]} 个实验锁可重新获取；后台验收进程也退出，其独立锁已释放。',
    '- 本轮只执行 CPU 验收，没有新增训练更新、模型前向或参考求解器调用。', '',
    '正式训练预算合计：36,864 次更新、2,359,296 个 sampled Actor 评分、12,582,912 个训练候选贪心评分、393,216 个额外 rollout 评分。',
    '正式最终测试共 73,728 次预测，使用各模型共享的 4,096 个唯一测试中心；攻击共 294,912 次候选预测，使用共享的 512 个唯一攻击中心。重复模型预测不计作独立新数据。', '',
    '## 2. 完整结果', '',
    '以下对四种合成分布及三个训练种子等权，数值越低越好。所有方法的初始平均线长均为 10.626327。', '',
    '| 方法 | 最终平均线长 | 窗口攻击后平均线长 |',
    '|---|---:|---:|',
]
for row in stats['method_summaries']:
    lines.append(f'| {row["method"]} | {row["final_mean"]:.6f} | {row["mean_attacked"]:.6f} |')
lines += [
    '', f'18 个单次训练运行相对各自随机初始化的四组平均线长下降 {minimum:.2f}%–{maximum:.2f}%。ERM-rollout 的测试均值和本次窗口攻击后均值最低。', '',
    '## 3. 配对统计与科学结论', '',
    '下表给出四组等权线长差（方法减对照）及分层配对 bootstrap 95% 区间；负值表示方法线长更低。三训练种子、2,000 次重采样，未做多重比较校正。', '',
    '| 方法与对照 | 平均差 | 95% 区间 |', '|---|---:|---|',
]
for method, baseline in [('erm_rollout', 'erm_critic'), ('erm_rollout', 'wdro_random'),
                         ('wdro_rl', 'erm_critic'), ('wdro_rl', 'wdro_random')]:
    row = next(x for x in pairs if x['method'] == method and x['baseline'] == baseline)
    lines.append(f'| {method} − {baseline} | {row["mean_difference"]:.6f} | [{row["ci_low"]:.6f}, {row["ci_high"]:.6f}] |')
lines += [
    '',
    '- **无参考算法的 RL 学习具备本实验范围内的可行性证据。** 实现使用自身生成动作的实测线长，未发现 GeoSteiner、MST、几何界、匹配求解器、最优标签或旧教师权重依赖。长度测量与合法性检查属于环境反馈。',
    '- 纯 RL 主线为 ERM-Critic、ERM-rollout、WDRO-RL；Random/ES/代理梯度分支是无参考解的对照，不应全部描述为学习坐标对手。',
    '- ERM-rollout 相对 ERM-Critic 和 Random 的上述区间均低于零，支持这组实验中的线长改善。它额外使用贪心 Actor 前向，不能仅据相同更新数声称计算效率胜出。',
    '- WDRO-RL 相对 ERM-Critic 和 Random 的区间都跨零，且种子 103 的比较方向反转；尚未建立其稳定优势。其他改良方法也不能只按均值差给出普遍优劣结论。',
    '- 本实验没有同预算 GeoSteiner 监督对照，没有测量最优值，因此没有确定去掉参考算法后训练代价和最优误差如何变化。', '',
    '## 4. 必须保留的适用边界', '',
    f'- 四类 WDRO 的方法平均运输成本为 {min(transport):.6f}–{max(transport):.6f}，均略超 ρ=0.04（约 {min(cost_percent):.2f}%–{max(cost_percent):.2f}%）。有限步 λ 更新没有严格满足全程平均运输预算。',
    f'- Random32 按 λ=0 最大化自身线长；各方法所选攻击样本有 {min(attack_fraction):.2f}%–{max(attack_fraction):.2f}% 的运输成本超过 ρ。它是位移窗口内的有限压力测试，不能解释为 ρ 球内最坏风险或连续 Wasserstein 证书。',
    '- 指标沿用 REST 的 span 记账规则；共线重叠可能重复计入，不能混称去重后的物理线段并集长度。运输距离是对应 pin 的平均 L1，训练目标为 raw 线长；均不同于旧相对 regret/最小匹配设定。',
    '- degree 30、三个训练种子及固定合成分布限定了结论范围；mix_gauss 每个 split 的样本共享一套生成的簇中心，区间不涵盖跨簇布局的不确定性。',
    '- test 未用于代码内 checkpoint 选择，但在训练前用于随机初始 Actor 的评价。',
    '- 未逐个保存训练候选，无法离线重放全部训练 argmax；未做 advantage 方差、独立代理方向相关等机制诊断。验收不将这些未执行诊断记为完成。',
    '- 无参考依赖结论来自冻结源码数据流、imports/loads、guard、随机权重重建及产物绑定；零调用计数字段自身不构成未知外部调用均被拦截的证明。', '',
    '## 5. 验收文件', '',
    '- [完整机器验收凭证](acceptance_final.json)',
    '- [原完成链本轮复核](original_completion_recheck.json)',
    '- [完整独立统计](final_statistics/statistics_review.md)',
    '- [全部配对区间](final_statistics/statistics_review_pairwise.csv)',
    '- [本轮重新生成的正式报告](recomputed_report/REPORT.md)',
    '- [方法源码审查快照](method_review.md)',
    '- [本轮完整验收日志](full_audit.log)',
]
atomic_text(OUT / 'ACCEPTANCE_REVIEW.md', '\n'.join(lines) + '\n')
release = dict(status='PASS', reviewed_at_utc=now, formal_runs=18, gpu_smokes=6,
               independent_length_rows=444096, original_artifacts_unchanged=True,
               independent_statistics_csv_byte_identical=independent_csv,
               new_training_updates=0, new_model_forward_calls=0, new_reference_solver_calls=0,
               artifacts={name: sha(OUT / name) for name in
                          ('acceptance_final.json', 'original_completion_recheck.json',
                           'ACCEPTANCE_REVIEW.md', 'full_audit.log', 'publish_review.py')})
atomic_text(OUT / 'release.json', json.dumps(release, indent=2) + '\n')
atomic_text(ROOT / 'STATUS.md', f'''# RL-only 当前状态

{now}：**18/18 个正式 GPU 模型、6/6 个 GPU smoke 已完成，整批验收及本轮复验通过。**

本轮重核原完成链、全部任务产物/检查点、444,096 条保存解的线长、选模及调用预算；重新生成的正式报告、四张正式 CSV 和四张独立统计 CSV 均与原结果逐字节一致。原实验及后台验收进程已退出，锁已释放，无需继续等待 GPU 实验。

[本轮验收报告](artifacts/acceptance_20260919/ACCEPTANCE_REVIEW.md) · [验收凭证](artifacts/acceptance_20260919/release.json) · [完整独立统计](artifacts/acceptance_20260919/final_statistics/statistics_review.md)

纯 RL 学习降低自身线长的可行性得到本实验支持；ERM-rollout 平均线长最低，WDRO-RL 尚未显示相对 Random 的稳定优势。平均运输成本略超 ρ；有限窗口攻击不提供严格 ρ 球最坏风险或最优误差证书。

本轮没有新增训练、模型前向或参考求解器调用。原 2026-09-16 验收及实验结果保留在原目录。
''')
print(json.dumps({key: value for key, value in release.items() if key != 'artifacts'}, indent=2))

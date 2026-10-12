"""Close this review only after all independent receipts pass; preserve originals."""
from pathlib import Path
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CODEX = ROOT.parents[1]

def load(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def atomic(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(tmp, p)

i = load(OUT / 'integrity/integrity_summary.json')
s = load(OUT / 'statistics/comparison_receipt.json')
c = load(OUT / 'statistics/compact_summary.json')
m = load(OUT / 'semantic/summary.json')
g = load(OUT / 'gpu_spotcheck/receipt.json')
assert i['status'] == s['status'] == g['status'] == 'PASS'
assert m['status'] == 'PASS_WITH_SCIENTIFIC_LIMITATIONS'
assert i['formal_runs_passed'] == s['runs'] == len(g['runs']) == 24
assert i['formal_effective_updates'] == 49152
assert i['effective_scores'] == i['actual_started_scores'] == i['actual_completed_scores'] == 10457088
assert i['all_locks_released'] and i['all_recorded_processes_exited']
assert all(i['original_report_unchanged'].values())
assert c['cost_mean_seed_CI_excludes_zero'] == c['cost_P95_seed_CI_excludes_zero'] == c['finite_LP_seed_CI_excludes_zero'] == 0
assert load(OUT / 'semantic/buffer_cpu_acceptance.json')['status'] == 'pass'
assert all(v['byte_identical'] and v['field_differences_count'] == 0 for v in s['csv_comparison'].values())

# Recheck original bindings after the new audits, without rewriting originals.
frozen = load(ROOT / 'artifacts/frozen.json')
for filename, digest in frozen['files'].items():
    assert sha(filename) == digest, filename
old_complete = load(ROOT / 'artifacts/report/complete.json')
for filename, digest in old_complete['artifacts'].items():
    assert sha(ROOT / 'artifacts/report' / filename) == digest, filename
assert old_complete['frozen_sha256'] == sha(ROOT / 'artifacts/frozen.json') == i['frozen_sha256']
for run in g['runs']:
    assert run['status'] == 'PASS'
    assert all(v['exact_equal'] for v in run['comparisons'])
    for filename, digest in run['input_sha256'].items():
        assert sha(ROOT / 'artifacts/runs' / run['run_id'] / filename) == digest
    assert sha(OUT / 'gpu_spotcheck' / (run['run_id'] + '.npz')) == run['output_sha256']
stats_complete = load(OUT / 'statistics/complete.json')
for filename, digest in stats_complete['artifacts'].items():
    assert sha(OUT / 'statistics' / filename) == digest, filename
assert sha(OUT / 'gpu_spotcheck.py') == g['script_sha256']

audit_script_names = {'gpu_spotcheck.py', 'review_integrity.py', 'recompute.py', 'audit.py'}
live = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = proc.joinpath('cmdline').read_bytes().split(b'\0')
        if args and b'python' in args[0]:
            if any(Path(a.decode()).name in audit_script_names and b'review_20260919' in a for a in args[1:] if a):
                live.append({'pid': int(proc.name), 'args': [a.decode() for a in args if a]})
    except (OSError, UnicodeError):
        pass
assert not live, live
now = datetime.now(timezone.utc).isoformat()
atomic(OUT / 'final_runtime.json', {
    'checked_at_utc': now, 'live_audit_workers': live,
    'original_queue_status': load(ROOT / 'artifacts/operations/exit.json')['status'],
    'gpu_snapshot': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader'], text=True),
    'note': 'All original task identities and locks were checked by integrity/runtime.json. This final scan checks the new audit worker scripts; no training was started.'})

names = {'rest': '纯线长 REST 对照', 'timing': '23 ICCAD / Timing-RST', 'hub': '25 DAC / Hub', 'buffer': '25 ICCAD / HarRL 组件'}
findings = {
    'rest': '三个 WDRO best 均为初始化；没有选出训练带来的改进模型。',
    'timing': '四分布均值下降，但每个分布仅 1/3 训练 seed 改善。',
    'hub': '两分布略好、两分布略差；多数最佳策略选择零个 hub。',
    'buffer': '有局部平均改善信号；尾部与有限候选风险并非全面改善。'}
table = '\n'.join(f"| {names[r['family']]} | {r['relative_percent_min']:+.3f}% 至 {r['relative_percent_max']:+.3f}% | {findings[r['family']]} |" for r in c['families'])
report = f'''# 三篇论文 × 基础 WDRO：实验验收

验收日期：2026-09-19。完成时间（UTC）：{now}。

## 1. 验收结论

**工程与数值验收 PASS；本轮尚未建立 WDRO 相对 ERM 的稳定优势。**

4 个任务家族 × 2 种方法 × 3 个训练 seed 的 24 个正式运行全部完成并通过复核。实验验证了三类论文组件可接入基础 Random/Sobol WDRO 并执行训练；没有证明它们已经改善现有纯线长 WDRO 主线。负结果和收益不稳定不构成执行失败。

本轮没有新增训练或调整科研参数。新验收证据全部保存在本目录，原冻结源码、数据、模型和原报告保持不变。

## 2. 完整性、数值和退出检查

| 检查 | 结果 |
|---|---|
| 冻结源码/数据 | 288/288 SHA256 一致，调度来源另 48/48 一致 |
| 正式实验 / GPU 预检 | 24/24 正式、4/4 预检通过 |
| 有效更新 | 每次 2048，共 49,152；无回滚、重复训练行 |
| 评分账本 | 有效、实际开始、实际完成均为 10,457,088 次网络实例评分 |
| 配对与选模 | 12/12 初始 Actor 和初始逐样本指标一致；按验证集重算最早最优选择一致 |
| 原始成本与组件 | 196,608 条 best/initial 成本重构通过；缓冲 9 项 CPU 组件探针通过 |
| 统计复算 | {s['metric_rows']:,} 行指标、{s['comparison_rows']:,} 行比较与原 CSV 逐字段及逐字节一致 |
| 有限候选风险 | 96 个 LP 重新优化，保存的质量分配、预算与目标值通过 |
| GPU 模型抽查 | 24 个 best checkpoint、12,288 次推理；792 个成本/指标数组逐元素一致 |
| 退出与锁 | 原任务全部退出，66 把锁释放，原 tmux 已结束；新验收 worker 也已退出 |

统计检查重新读取原始 NPZ，并按冻结统计定义重新计算描述统计、2000 次 paired bootstrap 和 LP。测试 best 与 initial 各 98,304 条成本预测；攻击 405,504 条成本预测。数据几何在模型间复用，不能把预测条数当成独立训练样本数。

GPU 抽查固定取每个分布的原始首个 64 条测试 batch 以及展平攻击候选的首 64 条，覆盖全部 24 个 best。它验证了 checkpoint 到已存预测的对应关系，**不是全部预测的神经网络重放**。使用空闲 GPU 1，耗时 {g['elapsed_seconds']:.1f} 秒，PyTorch 峰值 reserved 为 {g['max_cuda_reserved_bytes']/1048576:.0f} MiB。

两次额外启动尝试因即时 GPU 准入变化在训练子进程启动前退出 75，随后成功；没有业务失败、丢失训练或未结清评分。原实验从 9 月 15 日 22:01:47 到 9 月 16 日 05:34:26 UTC，墙时 7.544 小时。累计 GPU-worker 墙时 39.052 小时（正式 38.929、预检 0.124），含 CPU 评分和等待，不能解释为 CUDA kernel 净计算时。

## 3. 科学结果

下表是对冻结主成本的描述性百分比换算：每个分布先对三个 seed 求均值，再算 `(WDRO−ERM)/ERM`；范围覆盖四个分布。负数表示成本较低。各家族物理目标不同，不能跨家族按这些数值排名。

| 任务 | 四分布成本均值相对变化 | 实际含义 |
|---|---:|---|
{table}

16 个“家族 × 分布”均值比较中，固定模型后的 net 条件 95% CI 有 **{c['cost_mean_net_CI_excludes_zero']}/16** 排除零；反映训练重复差异的 seed CI 为 **0/16**。P95 的 net CI 为 {c['cost_P95_net_CI_excludes_zero']}/16，seed CI 为 0/16；有限候选 LP 风险的 seed CI 也为 0/16。因此不能用较多测试 net 掩盖只有三个训练重复的限制。这不是等价性证明，也不能据此断言 WDRO 无效。

### 纯线长 REST：没有选出 WDRO 训练收益

WDRO 三个 best 都是 update 0，权重及预测与初始化相同。ERM 的 best 为 0、0、256。表面上的 WDRO 平均优势全部来自 ERM seed103 的选中模型在测试上比共同初始化略差；不能归因为 WDRO 把 Actor 训练得更好。三个 WDRO 轨迹中 λ 为零的比例约为 64.8%、71.6%、67.2%。

### 23 ICCAD：可训练，但收益受 seed 影响

Timing 的四个分布均值差都为负，net 条件区间也为负，但各分布只有一个训练 seed 优于 ERM。mix_gauss 的 P95 上升，uniform/grid_cluster 的 LP 风险上升。支持可运行性与局部改善信号，不支持稳定全面优势。

### 25 DAC：需解释零 hub 行为

三个 ERM 最佳模型及 WDRO seeds102/103 在全部测试样本上选择零个 hub，使用无 hub 的 MSPD 路径。只有 WDRO seed101 少量选择 hub，分布平均为 0.049–0.110 个/net。这里关闭了原论文的额外 hub bonus；这是预先冻结的选择，但不能把当前结果解释为完整论文配方或高质量非平凡 hub 集合的验证。

### 25 ICCAD：组件实验有改善信号，尚无归因与最优性结论

Buffer 在 mix_gauss/grid_cluster 上均值下降的 net 条件区间排除零，但训练 seed 区间包含零；uniform/grid_cluster 的 P95 上升，edge_bias 的 LP 风险上升。它使用 16 类离散相位、固定合成 RC/负载、局部 greedy 插入与树投影，不是论文完整最优缓冲 DP。末相位梯度存在的检查通过，但没有固定/随机相位消融，不能单独归因谐波模块的贡献。

Buffer 三个训练轨迹全程平均运输成本比 rho 高约 0.90%–0.97%，末 256 步高约 0.23%–0.50%；两个最佳模型在 2048 步终点。它们是有限步对偶优化的残差，不是目标接线错误，也不支持“已经收敛并严格满足训练预算”的说法。

## 4. 结论适用范围

- 这是 degree10、2048 updates、batch64、3 seeds、四个合成分布的任务迁移试验。共同新 MLP baseline、初始化和选模规则按本分支冻结；不是 precision_v2 原主线的逐项复跑。
- 使用基础 Random/Sobol 候选与真实评分，没有 ES、Critic-gradient、RL 或 CEM 攻击器。内层 greedy 评分与外层 sampled-policy REINFORCE，以及中心加三个预筛候选，构成工程近似；未精确求解连续 min-max。
- 等更新次数不等计算量：ERM 每次 173,568 次评分，WDRO 每次 697,856 次，约 4.02 倍；训练部分为 5 倍。Hub 每个真实评分还含 11 次 PD 扫描。
- 测试 LP 只对每中心指定的 33 个候选运输边、共享平均 rho 精确求解；不包含全部中心到候选并集的跨行边，不是连续 Wasserstein 证书。训练的对偶预算残差与测试 LP 的严格可行性分别报告。
- P95 汇总是各 seed P95 的均值；三 seed 区间是探索性结果，多重比较没有校正。没有真实芯片测试、工业 signoff、跨 degree 验证或论文完整配方复现。

## 5. 交付入口

- [工程完整性与运行态](integrity/INTEGRITY.md)
- [独立统计与全部 CI](statistics/STATISTICAL_REVIEW.md)、[指标表](statistics/metrics.csv)、[比较表](statistics/comparisons.csv)
- [目标、预算与论文组件语义审查](semantic/REVIEW.md)
- [GPU 前向抽查凭证](gpu_spotcheck/receipt.json)
- [总验收与产物摘要](acceptance.json)

**最终状态：本轮实验与验收完成，后续无需监控；稳定 WDRO 优势与现有纯线长主线改进尚未得到本轮证据支持。**
'''
(OUT / 'REPORT.md').write_text(report)

# Update only the mutable status/navigation documents, preserving their old text.
docs = [ROOT / 'STATUS.md', ROOT / 'README.md', CODEX / 'docs/STATUS.md']
backup = OUT / 'before_closeout'
backup.mkdir(exist_ok=True)
for path, name in zip(docs, ['branch_STATUS.md', 'branch_README.md', 'docs_STATUS.md']):
    assert str(path) not in frozen['files']
    if not (backup / name).exists():
        (backup / name).write_bytes(path.read_bytes())
docs[0].write_text(f'''# 实验与验收状态

更新：{now}。

**DONE：24 次正式训练和 4 次 GPU 预检已完成，2026-09-19 独立验收 PASS。**

- 正式实验于 2026-09-16 05:34:26 UTC 完成；24×2048=49,152 有效更新，原任务退出、66 把锁释放，tmux 已结束，无需继续监控。
- 288 冻结文件、初始化配对、选模、评分账本和 96 LP 重验通过；1,984 行指标和 1,072 行比较复算与原表逐字节一致。
- 新增 GPU 抽查 24 best / 12,288 次推理，792 个数组全部精确复现；未新增训练或更改科研参数。
- 科学结论：未建立跨训练 seed 稳定的 WDRO 优势。REST-WDRO 三个 best 全为初始化；Hub 多数 best 选择零 hub；Timing/Buffer 仅有局部改善信号。
- degree10 组件迁移结果不能解释为原纯线长主线改善、完整论文复现或连续 Wasserstein 证书。

以[2026-09-19 验收报告](artifacts/review_20260919/REPORT.md)及[总凭证](artifacts/review_20260919/acceptance.json)为准。原始自动报告见 [artifacts/report/REPORT.md](artifacts/report/REPORT.md)。旧交接快照保存在验收目录 before_closeout/。
''')
readme = docs[1].read_text()
readme = readme.replace('5. 最终报告将由队列生成至 `artifacts/report/REPORT.md`；是否完成应以 `artifacts/operations/exit.json` 和独立统计验收为准。', '5. [2026-09-19 独立验收与结论](artifacts/review_20260919/REPORT.md)、[验收凭证](artifacts/review_20260919/acceptance.json)：24 次正式训练完成，工程与数值验收 PASS；稳定 WDRO 优势尚未建立。\n6. [原始自动报告](artifacts/report/REPORT.md)、[最新状态](STATUS.md)。')
docs[1].write_text(readme)
status = docs[2].read_text()
newrow = '| 三篇论文 × 基础 WDRO | DONE；degree10、24 次正式训练；9 月 19 日独立验收 PASS，尚未建立稳定 WDRO 优势 | [验收与统计](../improvements/papers_wdro_20260915/artifacts/review_20260919/REPORT.md) |\n'
if '| 三篇论文 × 基础 WDRO |' not in status:
    status = status.replace('| 阶段 9 论文同步 |', newrow + '| 阶段 9 论文同步 |')
note = '\n最新分支更新：2026-09-19，仅新增三篇论文 × 基础 WDRO 验收记录；其余完成项未在本次重新验收。\n'
if note.strip() not in status:
    status = status.replace('\n| 研究 |', note + '\n| 研究 |', 1)
docs[2].write_text(status)

artifacts = {}
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name not in {'acceptance.json', 'acceptance.sha256', 'finalize.log'} and not path.name.endswith('.partial'):
        artifacts[str(path.relative_to(OUT))] = sha(path)
receipt = {
    'schema_version': 1, 'status': 'PASS', 'completed_at_utc': now,
    'scope': 'degree10 paper components plus basic WDRO pilot; independent acceptance',
    'scientific_conclusion': 'NO_STABLE_WDRO_SUPERIORITY_DEMONSTRATED',
    'new_training_runs': 0, 'new_gpu_forward_predictions': g['forward_predictions'],
    'frozen_files_checked': len(frozen['files']), 'formal_runs': 24,
    'formal_effective_updates': i['formal_effective_updates'], 'effective_score_calls': i['effective_scores'],
    'metrics_rows': s['metric_rows'], 'comparison_rows': s['comparison_rows'],
    'original_csv_byte_identical': True, 'original_report_preserved': True,
    'frozen_sha256': sha(ROOT / 'artifacts/frozen.json'),
    'protocol_sha256': sha(ROOT / 'protocol.json'),
    'original_completion_sha256': sha(ROOT / 'artifacts/report/complete.json'),
    'scientific_limits': m['critical_limitations'],
    'artifacts': artifacts, 'updated_status_documents': {str(path): sha(path) for path in docs}}
atomic(OUT / 'acceptance.json', receipt)
(OUT / 'acceptance.sha256').write_text(sha(OUT / 'acceptance.json') + '  acceptance.json\n')
print(json.dumps({'status': 'PASS', 'receipt': str(OUT / 'acceptance.json'), 'artifacts_bound': len(artifacts), 'updated_docs': [str(x) for x in docs]}, ensure_ascii=False))

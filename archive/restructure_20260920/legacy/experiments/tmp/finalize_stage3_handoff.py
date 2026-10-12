"""Publish documentation only after independent delivery verification passes."""
import csv
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

CODEX = Path('/remote-home/Guilin/DRO/Codex')
EXP = CODEX / 'experiments'
ROOT = CODEX / 'wdro_rest_min'
now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_text(path, content):
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    assert temporary.read_text() == content
    os.replace(temporary, path)


verification_path = ROOT / 'manifests/stage3_delivery_verification_v1.json'
v = json.loads(verification_path.read_text())
assert v['status'] == 'pass' and v['reference_cache_checks_passed'] == 180
assert not v['stage4_started'] and not v['test_or_ispd_models_evaluated']
completion_path = ROOT / 'manifests/stage3_completion_v1.json'
assert digest(completion_path) == v['completion_manifest']['sha256']
c = json.loads(completion_path.read_text())
assert c['status'] == 'pass' and c['model_count'] == 45
table_path = Path(c['tables']['runs']['logical_path'])
assert digest(table_path) == c['tables']['runs']['sha256']
with table_path.open(newline='') as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 45
names = {'mixed_erm': 'Mixed ERM', 'group_dro': 'GroupDRO', 'hard_mining': 'Hard Mining'}
lines = [
    '独立交付复核与 validation 描述统计：', '',
    f"- 完整审计 {v['full_audit_checks_passed']} 项通过；重建 {v['training_nets_rebuilt']:,} 个训练输入与 {v['candidate_nets_rebuilt']:,} 个候选，held-out 内容交集为 0。",
    '- 15 个 Hard Mining 实际 reference cache/调用账本另有 180 项检查通过；90 个 latest/best checkpoint hash 再核验一致。',
    '- 四张导出表行数依次为 45、180、3,780、15；模型索引双路径/hash、验证键唯一性、选中行、nominal 门禁一致。',
    '- 独立复核证据：`wdro_rest_min/manifests/stage3_delivery_verification_v1.json`；实际 cache 审计：`experiments/logs/stage3_hard_mining_budget_precheck_20260905_all15/result.json`。',
    '',
    '| 方法 | nominal mean regret (%) | OOD worst-group regret (%) | 相对 ERM 的 OOD 差值（百分点） | OOD 优于配对 ERM |',
    '|---|---:|---:|---:|---:|',
]
summaries = []
for method, label in names.items():
    group = [row for row in rows if row['method'] == method]
    assert len(group) == 15
    summary = {'method': method, 'runs': 15,
        'nominal_regret_percent': 100 * statistics.mean(float(r['best_nominal_mean']) for r in group),
        'ood_worst_group_regret_percent': 100 * statistics.mean(float(r['best_ood_worst_group']) for r in group),
        'ood_delta_vs_erm_percentage_points': 100 * statistics.mean(float(r['ood_delta_vs_erm']) for r in group),
        'ood_improved_runs': sum(float(r['ood_delta_vs_erm']) < 0 for r in group)}
    summaries.append(summary)
    lines.append(f"| {label} | {summary['nominal_regret_percent']:.6f} | {summary['ood_worst_group_regret_percent']:.6f} | {summary['ood_delta_vs_erm_percentage_points']:+.6f} | {summary['ood_improved_runs']}/15 |")
lines += ['', '上表对 5 个 degree × 3 个 seed 的所选 validation 指标等权平均；OOD 列先取每个 run 的 worst group，再跨 run 平均，差值按相同 degree/seed 配对 ERM。数值越低越好。这是用于选模的 validation 描述统计，不是 test 泛化结果或显著性结论。', '']
report_path = CODEX / '阶段3_三基线与验收报告.md'
marker = '\n独立交付复核与 validation 描述统计：\n'
base = report_path.read_text().split(marker)[0].rstrip() + '\n\n'
atomic_text(report_path, base + '\n'.join(lines))

progress_path = CODEX / '01_WDRO-REST_项目进度.md'
progress = progress_path.read_text()
progress_lines = progress.splitlines()
for i, line in enumerate(progress_lines):
    if line.startswith('| 更新时间 |'):
        progress_lines[i] = f'| 更新时间 | {now} |'
    elif line.startswith('| 实验状态 |'):
        progress_lines[i] = f"| 实验状态 | 阶段 3 的 45/45 基线与 {v['full_audit_checks_passed']} 项全量审计完成；180 项实际 cache 检查和独立交付复核通过；连同阶段 2 共 60 个正式模型，仅使用 validation 选模 |"
    elif line.startswith('- 3.2 运行记录'):
        progress_lines[i] = '- `DONE` 3.2：45/45 正式训练完成，各 2,000 updates、batch 256，90 个 latest/best checkpoint hash 再核验一致；证据 `experiments/logs/stage3_pipeline/training_completion_verification_20260905T121703Z.json`。启动、动态资源预检和恢复历史保留在正式队列与逐 run manifests。'
    elif line.startswith('- 3.3 自动验收步骤'):
        progress_lines[i] = f"- `DONE` 3.3：45/45 全量输入/候选重建审计、{v['full_audit_checks_passed']} 项检查、180 项实际 cache 检查及独立交付复核通过；模型索引、四张表和最终报告已导出。证据 `wdro_rest_min/manifests/stage3_completion_v1.json` 与 `stage3_delivery_verification_v1.json`。"
    elif line.startswith('- 完成项提前审计：'):
        progress_lines[i] = '- 完成项提前审计于 11:26 UTC 交接原子游标并退出；最终 8-worker 审计已通过全部 45 个 run，取代此前 24/45 的中途计数。交接证据 `experiments/logs/stage3_completed_audit/handoff.json`，总审计 `experiments/logs/stage3_audit/stage3_formal_audit.json`。'
progress = '\n'.join(progress_lines) + '\n'
entry = f'''### {now}

- 阶段 3 最终交付独立复核通过：45 个模型、90 个 checkpoint hash、四张表的行数/唯一键/选中行与模型索引双路径一致；全部训练输入和候选已重建，held-out 交集 0。证据 `wdro_rest_min/manifests/stage3_delivery_verification_v1.json`。
- 15 个 Hard Mining 的实际 reference-cache/调用账本 180 项检查通过，补充到最终报告；按 15 个 degree/seed 配对汇总 validation 描述统计，未将其作为 test 或显著性结论。
- 复核命令：`bash /remote-home/Guilin/DRO/Codex/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/experiments/tmp/verify_stage3_delivery.py`；文档重建同一 wrapper 执行 `experiments/tmp/finalize_stage3_handoff.py`。训练/审计恢复命令保留在 `阶段3_子步骤与运行说明.md`，完成状态无需重启任务。
- 本次授权的阶段 3 全部完成并停止；protocol v1、data contract v1 与冻结训练范围未改动。下一动作仅为后续获准阶段 4 时执行 WDRO 对偶训练与实际预算配对；test/ISPD 模型评价未启动。

'''
progress = progress.replace('## 8. 变更记录\n\n', '## 8. 变更记录\n\n' + entry, 1)
atomic_text(progress_path, progress)

runbook_path = CODEX / '阶段3_子步骤与运行说明.md'
runbook = runbook_path.read_text().splitlines()
runbook[2] = f'更新时间：{now}。阶段 3.1、3.2、3.3 全部 DONE；45/45 正式模型、全量审计、四张表、最终报告与独立交付复核均已通过。持久训练/审计已完成，保留以下命令与记录供复核和故障恢复使用。'
for i, line in enumerate(runbook):
    if line.startswith('- 3.3 输入：'):
        runbook[i] = f"- 3.3 DONE：45 个完整 run 通过 {v['full_audit_checks_passed']} 项全量审计；15 个实际 reference cache 另通过 180 项检查。四张表、模型 hash 索引和 `阶段3_三基线与验收报告.md` 已发布；独立复核见 `wdro_rest_min/manifests/stage3_delivery_verification_v1.json`。"
    elif line.startswith('12:32 UTC 当前运行的是'):
        runbook[i] = '12:32 UTC 的中途记录：当时运行最终 8-worker CPU 审计，训练已完成；截至本文件顶部时间，完整审计及独立交付复核均已通过。15-run cache 证据为 `experiments/logs/stage3_hard_mining_budget_precheck_20260905_all15/result.json`。以下记录保留作运行与恢复历史。'
atomic_text(runbook_path, '\n'.join(runbook) + '\n')

artifacts = [verification_path, completion_path, report_path, progress_path, runbook_path, table_path,
    EXP / 'tmp/verify_stage3_delivery.py', EXP / 'tmp/stage3_hard_mining_budget_precheck_all15.py',
    Path(__file__)]
manifest = {'status': 'pass', 'at_utc': now, 'stage': 3, 'validation_descriptive_summary': summaries,
    'stage4_started': False, 'test_or_ispd_models_evaluated': False,
    'artifacts': [{'logical_path': str(p), 'resolved_path': str(p.resolve()), 'sha256': digest(p)} for p in artifacts]}
manifest_path = ROOT / 'manifests/stage3_handoff_v1.json'
atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
os.chmod(manifest_path, 0o444)
print(json.dumps({'status': 'pass', 'handoff_manifest': str(manifest_path), 'validation_summary': summaries}))

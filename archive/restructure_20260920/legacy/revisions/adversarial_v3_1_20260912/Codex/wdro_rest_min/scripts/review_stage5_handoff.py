#!/usr/bin/env python3
"""Read back the completed stage-5 delivery and clarify its report wording."""
import csv
import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CODEX = Path('/remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex')
ROOT = CODEX / 'wdro_rest_min'
EXP = CODEX / 'experiments'
OUT = EXP / 'logs/stage5_final_delivery_review'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    assert path.is_relative_to(CODEX)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with partial.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    assert partial.read_bytes() == data
    os.replace(partial, path)


def artifact(path):
    return dict(logical_path=str(path), resolved_path=str(path.resolve()), sha256=digest(path))


def main():
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    completion_path = ROOT / 'manifests/stage5_completion_v1.json'
    delivery_path = ROOT / 'manifests/stage5_delivery_verification_v1.json'
    completion, delivery = read(completion_path), read(delivery_path)
    assert completion['status'] == delivery['status'] == 'pass'
    assert delivery['completion_sha256'] == digest(completion_path)
    assert len(delivery['checks']) == 17 and all(delivery['checks'].values())
    assert not completion['stage6_started'] and not completion['test_or_ispd_used']
    contract_count = 0
    for name in ['stage5_training_contract_v1.json', 'stage5_operations_contract_v1.json']:
        for path, expected in read(ROOT / 'manifests' / name)['files'].items():
            assert digest(CODEX / path) == expected, path
            contract_count += 1
    entries = []

    def walk(value):
        if isinstance(value, dict):
            if {'logical_path', 'resolved_path', 'sha256'} <= value.keys():
                path = Path(value['logical_path'])
                assert path.is_relative_to(CODEX) and str(path.resolve()) == value['resolved_path']
                assert path.stat().st_size == value['bytes']
                item = dict(value)
                if path.stat().st_size <= 64 * 1024**2:
                    assert digest(path) == value['sha256'], str(path)
                    item['handoff_check'] = 'sha256_recomputed'
                else:
                    # The completed full verifier already hashed these large files.
                    assert path.stat().st_mtime <= datetime.fromisoformat(delivery['at_utc'].replace('Z', '+00:00')).timestamp()
                    item['handoff_check'] = 'path_size_mtime; full_sha256_from_bound_delivery_verification'
                entries.append(item)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(completion)
    assert len(entries) == delivery['artifact_count'] == 84
    audits, nets = [], 0
    for model in completion['models']:
        audit = read(model['audit']['logical_path'])
        replay = read(model['best_validation_recompute']['logical_path'])
        assert audit['status'] == 'pass' and all(audit['checks'].values())
        assert audit['completed_updates'] == 2000
        assert replay['status'] == 'pass' and replay['exact_metric_equality'] and not replay['test_ispd_used']
        assert replay['checkpoint_sha256'] == model['best_checkpoint']['sha256']
        assert replay['result_sha256'] == model['result']['sha256']
        nets += replay['validation_nets']
        audits.append(dict(run_id=model['run_id'], checks=len(audit['checks']), updates=2000))
    assert len(audits) == 9 and sum(a['checks'] for a in audits) == 396 and nets == 72000
    tables = {}
    for name, expected in [('stage5_ablation.csv', 9), ('stage5_validation_long.csv', 756), ('stage5_budget_trajectory.csv', 18000)]:
        with (EXP / 'tables' / name).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected
        tables[name] = len(rows)
    frozen_path = ROOT / 'configs/stage5_selected_wdro_v1.json'
    frozen = read(frozen_path)
    assert stat.S_IMODE(frozen_path.stat().st_mode) == 0o444
    assert (frozen['selected_rho_mode'], frozen['searcher'], frozen['candidate_count'], frozen['ground_cost'], frozen['lambda_mode']) == ('q50', 'random_sobol', 32, 'hungarian_l1_mean', 'learned')
    heartbeat = read(EXP / 'logs/stage5_pipeline/heartbeat.json')
    assert heartbeat['status'] == 'complete' and heartbeat['completed_updates'] == 18000
    assert not (EXP / 'logs/stage5_pipeline/run.lock').exists()
    processes = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(Path(arg).name in {'train_stage5.py', 'audit_stage5.py', 'stage5_pipeline.py', 'run_stage5_pipeline.py', 'launch_stage5.py'} for arg in argv if arg):
                processes.append(dict(pid=int(proc.name), argv=argv))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    assert not processes, processes
    tmux = subprocess.run(['tmux', '-S', str(EXP / 'tmp/stage5.sock'), 'list-sessions'], capture_output=True, text=True)
    assert 'wdro_stage5_v1:' not in tmux.stdout
    report = CODEX / '阶段5_Validation搜索与消融验收报告.md'
    backup = OUT / 'report_before_clarification.md'
    old = '主配置选择只比较 rho=0/q50/q75 三个规范主方法配置；nominal 约束为同 degree/seed ERM best validation mean +0.002，选择最早可行 checkpoint，再取最小 OOD worst-group。消融候选不得替换 Random/Sobol K32、Hungarian、learned lambda 主方法。'
    new = '主配置选择只比较 rho=0/q50/q75 三个规范主方法配置；nominal 约束为同 degree/seed ERM best validation mean +0.002。每个 rho run 在满足 nominal 约束的全部 validation checkpoints 中选择 OOD worst-group mean 最小者，目标相同时取最早 checkpoint；再比较三个 rho run 的所选目标，目标相同时按预声明顺序 zero、q50、q75 选择。消融候选不得替换 Random/Sobol K32、Hungarian、learned lambda 主方法。'
    body = report.read_text()
    if old in body:
        assert body.count(old) == 1
        if backup.exists():
            assert backup.read_bytes() == report.read_bytes()
        else:
            atomic(backup, report.read_bytes())
        body = body.replace(old, new)
        body += '\n交接文字校正：本报告的 checkpoint 选择顺序说明已按冻结的 `stage5_plan_v1.json` tie_rule、`stage5/trainer.py` 和完整审计实现澄清；训练、选模结果、配置与表图未变。原报告及校正证据见 `experiments/logs/stage5_final_delivery_review/`。只读导出器未改动，重新生成报告后需再次运行 `scripts/review_stage5_handoff.py` 保留此澄清。\n'
        atomic(report, body.encode())
    assert new in report.read_text() and backup.exists()
    command = 'bash /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/revisions/reference_precision_v2/Codex/wdro_rest_min/scripts/review_stage5_handoff.py'
    progress = CODEX / '01_WDRO-REST_项目进度.md'
    text = progress.read_text()
    text = text.replace('阶段5完成项validation重算：7/7', '阶段5中途证据（最终已完成9/9）：7/7')
    text = text.replace('这是各run验收的提前执行；5.4最终导出仍要求9/9完整训练/审计和全部best重算通过。', '这是各run验收的提前执行记录；最终9/9完整训练/审计、全部best重算及5.4导出已通过，见最终验收报告。')
    text = text.replace('阶段5完成项提前审计：4/4', '阶段5中途提前审计记录（最终已完成9/9）：4/4')
    text = text.replace('单进程 nice=19，原训练继续；最终仍须 9/9 训练/审计和 5.4 交付复核。', '当时为单进程 nice=19、原训练继续；最终 9/9 训练/审计和 5.4 交付复核现已通过。')
    text = text.replace('- 当前代码离 WDRO 尚有完整实现距离；进度不能依据目录名 `WDRO-REST` 判断。', '- 阶段 4/5 的 WDRO 实现、完整审计与单 degree/seed validation 消融已完成；多 degree/seed 正式 WDRO 及共享 test 评价属于尚未启动的阶段 6。阶段 5 结果不能当作 test 泛化或统计显著性证据。')
    text = text.replace('WDRO 正式模型与五方法共享 test 主评价待后续冻结配置后执行，阶段 6 未启动', 'WDRO 正式模型与五方法共享 test 主评价待后续获准后按阶段 5 已冻结配置执行，阶段 6 未启动')
    text = re.sub(r'\| 更新时间 \| [^\n]+', f'| 更新时间 | {now} |', text, count=1)
    marker = '### 2026-09-06 阶段 5 最终交接复核'
    note = f'''{marker}

- `DONE` 主队列于 12:54:31 UTC 完成；本次交接复核时间 {now}。9/9 runs、18,000 updates、396/396 完整审计检查、9 个 best 独立 validation 重算共 72,000 nets 均通过；原最终交付复核 17/17 检查、84 个产物通过。
- `DONE` 最终配置只读，选中 q50（d30 rho=0.01381177343428135），主方法保持 Random/Sobol K32、Hungarian-L1、learned lambda。三张表分别为 9、756、18,000 行；预算图已回看，图例和曲线完整可读。
- `DONE` 校正验收报告中“最早可行 checkpoint”歧义：先在 nominal 可行点中最小化 OOD worst-group，仅目标相同时选最早点；rho 间并列按 zero/q50/q75 顺序。冻结实现和数值结果不变。原稿、修订稿 hash、运行状态及证据在 `experiments/logs/stage5_final_delivery_review/handoff_review.json`；原稿为同目录 `report_before_clarification.md`。
- 资源证据：`wdro_rest_min/manifests/host_runtime_20260906T130730Z.json` 三次采样 preflight 通过，GPU 2/3/4/5/6 当时有容量和计算余量，产物逻辑目录及附加盘 resolved path 已记录，产物盘可用 2835.159 GiB。本次只做 CPU 文件交接复核，未启动新批跑。
- 无剩余阶段 5 阻塞或在训/在审任务。复核/重复文档校正命令：`{command}`。完整交付验证入口仍为 `scripts/verify_stage5_delivery.py`；本次回读不重复大文件全 hash 或训练数学重放，复用已绑定 completion hash 的完整交付证据，并重算小产物与 70 项冻结文件 hash。
- 下一步：按本次授权止于阶段 5；阶段 6 未启动，test/ISPD 模型指标未用于选模。

'''
    if marker not in text:
        text = text.replace('## 8. 变更记录\n', '## 8. 变更记录\n\n' + note, 1)
        assert marker in text
    atomic(progress, text.encode())
    guide = CODEX / '阶段5_子步骤与运行说明.md'
    text = guide.read_text()
    status = '> 当前状态：阶段 5 全部 DONE（2026-09-06 12:54 UTC），5.1–5.4 均通过；以下运行/恢复与续跑记录保留供追溯。最终证据见 `阶段5_Validation搜索与消融验收报告.md` 和 `experiments/logs/stage5_final_delivery_review/handoff_review.json`。本次止于阶段 5。\n\n'
    if status not in text:
        text = text.replace('# 阶段 5：子步骤与恢复说明\n\n', '# 阶段 5：子步骤与恢复说明\n\n' + status, 1)
        atomic(guide, text.encode())
    result = dict(status='pass', at_utc=now, scope='final handoff readback and report wording clarification; no new model evaluation', command=command, resume_command=command, pid=os.getpid(), stage=5, stage6_started=False, test_or_ispd_used=False, completion=artifact(completion_path), original_delivery=artifact(delivery_path), host_runtime=artifact(ROOT / 'manifests/host_runtime_20260906T130730Z.json'), contract_entries_verified=contract_count, artifact_count=len(entries), artifacts=entries, audits=audits, full_audit_checks=396, best_validation_nets=nets, table_rows=tables, active_stage5_workers=processes, heartbeat=heartbeat, tmux=dict(returncode=tmux.returncode, stdout=tmux.stdout, stderr=tmux.stderr), selection_wording=dict(before=old, after=new, original_report=artifact(backup), updated_report=artifact(report)), documents=[artifact(progress), artifact(guide)], script=artifact(Path(__file__)), figures_visual_review='Both PNGs inspected; labels, legends and complete curves readable; no graphic edits.')
    atomic(OUT / 'handoff_review.json', (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({k: result[k] for k in ['status', 'at_utc', 'artifact_count', 'contract_entries_verified', 'full_audit_checks', 'best_validation_nets', 'stage6_started']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

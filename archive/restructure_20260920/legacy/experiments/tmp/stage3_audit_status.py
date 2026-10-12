"""Read current acceptance progress without launching or changing any run."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

CODEX = Path('/remote-home/Guilin/DRO/Codex')
EXP = CODEX / 'experiments'
now = datetime.now(timezone.utc)
heartbeat = json.loads((EXP / 'logs/stage3_audit/stage3_completion_audit/heartbeat.json').read_text())
pipeline = json.loads((EXP / 'logs/stage3_pipeline/heartbeat.json').read_text())
cursors = [json.loads(p.read_text()) for p in (EXP / 'logs/stage3_audit/cursors').glob('formal_*.json')]
candidate_count = sum(c['candidate_nets_checked'] for c in cursors)
active = [(c['binding']['identity']['degree'], c['binding']['identity']['training_seed'], c['completed_updates'])
          for c in cursors if c['status'] != 'pass' and c['binding']['identity']['method'] == 'hard_mining']
record = {'at_utc': now.isoformat().replace('+00:00', 'Z'), 'pipeline_status': pipeline['status'],
          'audit_status': heartbeat['status'], 'final_runs_passed': heartbeat['completed_updates'],
          'candidate_nets_rebuilt': candidate_count, 'candidate_nets_required': 253440000,
          'candidate_percent': round(candidate_count / 253440000 * 100, 2),
          'heldout_intersections': sum(c['heldout_intersections'] for c in cursors),
          'active_hard_cursors': sorted(active),
          'heartbeat_age_seconds': round((now - datetime.fromisoformat(heartbeat['last_update_utc'].replace('Z', '+00:00'))).total_seconds(), 1)}
target = EXP / 'logs/stage3_pipeline/live_audit_progress.json'
temporary = target.with_suffix('.json.partial')
temporary.write_text(json.dumps(record, indent=2) + '\n')
os.replace(temporary, target)
progress = CODEX / '01_WDRO-REST_项目进度.md'
if pipeline['status'] != 'complete' and heartbeat['status'] == 'running':
    lines = progress.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith('| 更新时间 |'):
            lines[i] = f"| 更新时间 | {record['at_utc']} |"
        elif line.startswith('| 实验状态 |'):
            lines[i] = (f"| 实验状态 | 45/45 正式训练完成；最终全量审计 {heartbeat['completed_updates']}/45 通过；"
                        f"候选重建 {record['candidate_percent']}%，已重建部分 held-out 交集 {record['heldout_intersections']} |")
    # Completion is published before the exporter's own final progress update.
    if not (CODEX / 'wdro_rest_min/manifests/stage3_completion_v1.json').exists():
        temporary = progress.with_name(progress.name + '.monitor.partial')
        temporary.write_text('\n'.join(lines) + '\n')
        os.replace(temporary, progress)
print(json.dumps(record), flush=True)

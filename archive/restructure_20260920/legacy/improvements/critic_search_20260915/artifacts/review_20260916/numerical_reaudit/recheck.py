"""Re-execute the frozen numerical gates without rewriting any original output."""
import json,sys,time,hashlib
from pathlib import Path
ROOT=Path('/remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915')
sys.path.insert(0,str(ROOT))
import report

OUT=ROOT/'artifacts/review_20260916/numerical_reaudit'
started=time.monotonic()
a=report.Audit()
for path,kind in report.expected_tasks(ROOT/'artifacts'):
    print(json.dumps(dict(task=path.name,kind=kind,phase='checking')),flush=True)
    (a.fixed_task if kind=='fixed' else a.training_task)(path)
    print(json.dumps(dict(task=path.name,status='pass')),flush=True)
leakage=a.no_leakage()
selection=a.selection()
payload=dict(status='pass',at=time.time(),tasks=a.task_rows,
    unique_bound_files=len(a.hash_cache),bindings_checked=len(a.artifacts),
    leakage=leakage,selection=selection,seconds=time.monotonic()-started,
    original_report_sha256=hashlib.sha256((ROOT/'report.py').read_bytes()).hexdigest(),
    scope='Repeated frozen per-candidate numeric gates; independent bootstrap reimplementation is a separate review component.',
    greedy_requested=sum(t['actual_greedy_requested'] for t in a.task_rows),
    greedy_committed=sum(t['committed_greedy'] for t in a.task_rows),
    sampled_requested=sum(t['actual_sampled_requested'] for t in a.task_rows))
(OUT/'result.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
print(json.dumps({k:payload[k] for k in ('status','unique_bound_files','bindings_checked','seconds','greedy_requested','sampled_requested')}),flush=True)

"""Read-only supplement: find task/orphan GeoSteiner processes by exact script/cwd."""
from datetime import datetime, timezone
import json
from pathlib import Path
import os

ROOT=Path('/remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915')
OUT=Path(__file__).resolve().parent
WORKERS={'fixed.py','training.py','scheduler.py','report.py','autofinish.py','closeout.py'}
rows=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        if not argv:continue
        cwd=(proc/'cwd').resolve()
        worker=any(Path(a).name in WORKERS and (str(ROOT) in a or cwd==ROOT) for a in argv[1:])
        multiprocessing_child=cwd==ROOT and any('multiprocessing.spawn' in a or 'multiprocessing.resource_tracker' in a for a in argv)
        if worker or multiprocessing_child:
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            rows.append(dict(pid=int(proc.name),state=fields[0],parent_pid=int(fields[1]),
                start_ticks=fields[19],cwd=str(cwd),argv=argv,worker=worker,
                multiprocessing_child=multiprocessing_child))
    except (FileNotFoundError,PermissionError,ProcessLookupError):pass
autofinish=json.loads((ROOT/'artifacts/operations/autofinish_status.json').read_text())
auto_pid=autofinish['pid']
result=dict(status='pass' if not rows else 'fail',at=datetime.now(timezone.utc).isoformat(),
    relevant_live_processes=rows,autofinish_pid=auto_pid,autofinish_pid_exists=Path(f'/proc/{auto_pid}').exists(),
    scope='exact task script basename under research root, plus multiprocessing children with research cwd; other users not modified')
path=OUT/'remaining_processes.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))

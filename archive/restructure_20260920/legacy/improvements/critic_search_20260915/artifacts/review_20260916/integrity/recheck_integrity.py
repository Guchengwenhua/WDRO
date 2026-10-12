"""Independent, read-only scientific integrity and live lifecycle recheck.

Only files in this script's directory are created. Hash every canonical file
once, retain every original manifest reference and every mismatch separately.
No model inference, training, source changes, or original-manifest rewrites.
"""
from __future__ import annotations
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

ROOT = Path('/remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915')
EXP = ROOT / 'artifacts'
OUT = Path(__file__).resolve().parent
LEGACY = ROOT.parents[1] / 'revisions/adversarial_v3_1_20260912/Codex'
SEEDS = (101, 102, 103)
VARIANTS = ('refresh32_raw', 'refresh8_raw', 'refresh32_ema', 'refresh8_ema')
DYNAMIC = {str((ROOT.parents[1] / 'docs/STATUS.md').resolve())}
GREEDY = ('actor_requested', 'actor_completed', 'rest_requested', 'rest_completed', 'reference_requests')
SAMPLED = ('sampled_actor_requested', 'sampled_actor_completed', 'sampled_rest_requested',
           'sampled_rest_completed', 'sampled_reference_reuses')


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.partial')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def process(record):
    pid = record.get('pid')
    expected = record.get('start_ticks', record.get('process_start_ticks'))
    row = dict(pid=pid, expected_start_ticks=str(expected), exists=False,
               same_process_alive=False, state=None, current_start_ticks=None)
    try:
        proc = Path(f'/proc/{pid}')
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        row.update(exists=True, state=fields[0], current_start_ticks=fields[19],
                   same_process_alive=fields[19] == str(expected),
                   command=(proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return row


def check_lock(path):
    row = dict(path=str(path), exists=path.is_file(), released=False)
    if not row['exists']:
        return row
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            row['released'] = True
            fcntl.flock(stream, fcntl.LOCK_UN)
        except BlockingIOError:
            pass
    return row


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(args=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as exc:
        return dict(args=args, error=repr(exc))


def main():
    started = time.monotonic()
    controls = [EXP/'audit/completion_audit.json', EXP/'freeze.json',
                EXP/'automation_manifest.json', EXP/'audit/final_delivery.json']
    errors, references, control_hashes = [], [], {}
    def add(path, expected, source, field):
        p = Path(path)
        references.append(dict(path=str(p), canonical_path=str(p.resolve()), expected_sha256=expected,
                               source_manifest=str(source), field=field))
    def walk(value, source, field='root'):
        if isinstance(value, dict):
            if 'sha256' in value and ('path' in value or 'logical_path' in value):
                add(value.get('path', value.get('logical_path')), value['sha256'], source, field)
            for key, item in value.items():
                walk(item, source, field+'.'+key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, source, field+f'[{i}]')
    documents = {}
    for p in controls:
        documents[str(p)] = read(p); control_hashes[str(p)] = sha(p)
        walk(documents[str(p)], p)
    audit, frozen, automation, final = [documents[str(p)] for p in controls]
    add(controls[0], final['audit_sha256'], controls[3], 'audit_sha256')
    add(ROOT/'report.py', audit['report_source_sha256'], controls[0], 'report_source_sha256')
    if audit.get('status') != 'pass' or audit.get('formal_tasks') != 16:
        errors.append('completion audit status/count')
    if final.get('status') != 'pass' or final.get('formal_tasks') != 16:
        errors.append('final delivery status/count')

    tasks = [(EXP/'fixed/validation_s101', 'fixed')]
    tasks += [(EXP/'fixed'/f'test_s{s}', 'fixed') for s in SEEDS]
    tasks += [(EXP/'training'/f'{v}_d30_s{s}', 'training') for s in SEEDS for v in VARIANTS]
    parent_registry = read(LEGACY/'manifests/parent_models_frozen.json')['models']
    task_rows, process_rows, lock_rows = [], [], []
    for out, kind in tasks:
        complete_path, identity_path = out/'complete.json', out/'identity.json'
        done, identity = read(complete_path), read(identity_path)
        control_hashes[str(complete_path)] = sha(complete_path)
        control_hashes[str(identity_path)] = sha(identity_path)
        walk(done, complete_path)
        if done.get('status') != 'pass' or done.get('identity') != identity:
            errors.append(f'{out}: completion/identity mismatch')
        if kind == 'training':
            for name, expected in identity['source_sha256'].items():
                add(LEGACY/name if name.startswith('adv_v3/') else ROOT/name, expected, identity_path, 'source_sha256.'+name)
            if identity.get('smoke') or identity['settings']['updates'] != 128:
                errors.append(f'{out}: formal settings mismatch')
            for method in ('erm', 'wdro_rest'):
                expected = next(r['best'] for r in parent_registry if (r['degree'],r['seed'],r['method']) == (30,identity['seed'],method))
                got = identity['dependencies'][method]
                if Path(got['path']).resolve() != Path(expected['logical_path']).resolve() or got['sha256'] != expected['sha256']:
                    errors.append(f'{out}: parent {method} differs from frozen registry')
            committed_greedy, committed_sampled = 128*16*4 + 5*4*64 + 3*4*256, 128*16
            summary = read(out/'summary.json')
            if len(summary['history']) != 128 or summary['updates'] != 128:
                errors.append(f'{out}: training updates incomplete')
            history_updates = [r['update'] for r in summary['history']]
            if history_updates != list(range(1,129)):
                errors.append(f'{out}: nonconsecutive committed steps')
            for step in (0,32,64,96,128):
                if not (out/'validation'/f'u{step:06d}').is_dir():
                    errors.append(f'{out}: missing validation step {step}')
        else:
            for name, expected in identity['source'].items():
                add(ROOT/name, expected, identity_path, 'source.'+name)
            add(ROOT/'protocol.json', identity['protocol_sha256'], identity_path, 'protocol_sha256')
            if identity['phase'] == 'test':
                add(EXP/'selection.json', identity['selection_sha256'], identity_path, 'selection_sha256')
            expected = next(r['best'] for r in parent_registry if (r['degree'],r['seed'],r['method']) == (30,identity['seed'],'erm'))
            got = identity['bindings']['actor']
            if got['sha256'] != expected['sha256']:
                errors.append(f'{out}: fixed Actor differs from parent registry')
            summary = read(out/'summary.json')
            count = 32 if identity['phase'] == 'validation' else 64
            methods = {r['method'] for r in summary}
            if len(summary) != len(methods)*4 or (identity['phase']=='test' and len(methods)!=11):
                errors.append(f'{out}: fixed result coverage')
            committed_greedy = 4*len(methods)*count*64+(2560 if identity['phase']=='test' else 0)
            committed_sampled = 0

        ledger = read(out/'oracle/oracle_ledger.json')
        differences = {}
        for key in GREEDY:
            differences[key] = ledger.get(key,0)-committed_greedy
            if differences[key] < 0:
                errors.append(f'{out}: ledger {key} below committed')
        for key in SAMPLED:
            differences[key] = ledger.get(key,0)-committed_sampled
            if differences[key] < 0:
                errors.append(f'{out}: ledger {key} below committed')
        for family in ('actor','rest','actor_forward_rows','actor_padding_rows','sampled_actor','sampled_rest'):
            if ledger.get(family+'_requested',0) < ledger.get(family+'_completed',0):
                errors.append(f'{out}: completed > requested {family}')
        for ending in ('requested','completed'):
            if ledger.get('actor_forward_rows_'+ending,0) != ledger.get('actor_'+ending,0)+ledger.get('actor_padding_rows_'+ending,0):
                errors.append(f'{out}: forward/padding identity {ending}')
        dbpath = out/'oracle/reference.sqlite'
        with sqlite3.connect(f'file:{dbpath}?mode=ro', uri=True) as db:
            sqlite_calls = list(db.execute('SELECT COALESCE(SUM(requested),0),COALESCE(SUM(misses_started),0),COALESCE(SUM(misses_completed),0) FROM calls').fetchone())
            sqlite_integrity = db.execute('PRAGMA quick_check').fetchall()
        ledger_calls = [ledger[k] for k in ('reference_requests','geosteiner_misses_started','geosteiner_misses_completed')]
        if sqlite_calls != ledger_calls or sqlite_integrity != [('ok',)]:
            errors.append(f'{out}: SQLite/ledger integrity')
        launch, exited = read(out/'launch.json'), read(out/'exit.json')
        proc = process(launch); proc.update(task=str(out.relative_to(EXP)))
        lock = check_lock(out/'task.lock')
        if proc['same_process_alive'] or not lock['released'] or exited.get('complete') is not True:
            errors.append(f'{out}: incomplete live lifecycle')
        process_rows.append(proc); lock_rows.append(lock)
        task_rows.append(dict(task=str(out.relative_to(EXP)), kind=kind, status=done['status'],
            committed_greedy=committed_greedy, committed_sampled=committed_sampled,
            ledger=ledger, actual_minus_committed=differences, sqlite_calls=sqlite_calls,
            sqlite_quick_check=sqlite_integrity, exited=exited, bound_artifact_count=len(done['artifacts'])))

    unique = sorted({r['canonical_path'] for r in references})
    print(f'Hashing {len(unique)} canonical files for {len(references)} manifest references',flush=True)
    def check(path):
        try:
            p=Path(path);before=p.stat();actual=sha(p);after=p.stat()
            stable=(before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
            return path,dict(actual_sha256=actual,size=after.st_size,stable_during_read=stable,error=None)
        except Exception as exc:
            return path,dict(actual_sha256=None,size=None,stable_during_read=False,error=repr(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes=dict(pool.map(check, unique))
    mismatches=[]
    for r in references:
        observed=hashes[r['canonical_path']]
        r.update(observed)
        r['matches']=observed['actual_sha256']==r['expected_sha256'] and observed['stable_during_read']
        r['classification']='dynamic_navigation' if r['canonical_path'] in DYNAMIC else 'scientific_or_operational_evidence'
        if not r['matches']:mismatches.append(r)
    scientific_mismatches=[r for r in mismatches if r['classification']!='dynamic_navigation']
    if scientific_mismatches:errors.append(f'{len(scientific_mismatches)} scientific/operational binding mismatches')

    queue, queue_exit=read(EXP/'operations/queue.json'),read(EXP/'operations/exit.json')
    if queue['running'] or queue['pending'] or queue['failures'] or len(set(queue['completed']))!=16:
        errors.append('scheduler queue not successfully drained')
    if queue_exit['pending'] or queue_exit['failures'] or len(set(queue_exit['completed']))!=16:
        errors.append('scheduler exit incomplete')
    scheduler_process=process(read(EXP/'operations/launch.json'))
    process_rows.append(dict(task='scheduler',**scheduler_process))
    if scheduler_process['same_process_alive']:errors.append('scheduler still active')
    for name in ('queue.lock','autofinish.lock'):
        lock=check_lock(EXP/'operations'/name);lock_rows.append(lock)
        if not lock['released']:errors.append(f'{name} not released')
    autofinish=read(EXP/'operations/autofinish_status.json')
    if autofinish.get('status') not in ('complete','done','pass'):
        errors.append('autofinish did not record successful completion')
    socket=EXP/'operations/tmux.sock'
    tmux=command(['tmux','-S',str(socket),'list-sessions','-F','#{session_name} #{session_attached} #{session_windows}'])
    live=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])
    # The GPU snapshot can include other users. Check only the recorded research PIDs.
    owned={r['pid'] for r in process_rows}
    owned_gpu_rows=[]
    if live.get('returncode')==0:
        for line in live['stdout'].splitlines():
            try:pid=int(line.split(',')[0].strip())
            except ValueError:continue
            if pid in owned:owned_gpu_rows.append(line)
    if owned_gpu_rows:errors.append('a recorded experiment PID remains in GPU process snapshot')
    live.update(recorded_experiment_pid_rows=owned_gpu_rows)
    if tmux.get('returncode')==0 and tmux.get('stdout','').strip():
        errors.append('dedicated experiment tmux session still exists')
    for p in controls:
        if sha(p)!=control_hashes[str(p)]:errors.append(f'control changed during review {p}')
    totals={key:sum(row['ledger'].get(key,0) for row in task_rows) for key in GREEDY+SAMPLED+('geosteiner_misses_started','geosteiner_misses_completed','actor_forward_rows_requested','actor_padding_rows_requested')}
    result=dict(status='pass' if not errors else 'fail', scientific_integrity_pass=not errors,
        exact_all_bindings_match=not mismatches, at=datetime.now(timezone.utc).isoformat(),
        formal_tasks=len(tasks), canonical_files_hashed=len(hashes), manifest_binding_references=len(references),
        total_bytes_hashed=sum(v['size'] or 0 for v in hashes.values()),
        mismatch_reference_count=len(mismatches), mismatch_canonical_files=len({r['canonical_path'] for r in mismatches}),
        scientific_mismatch_count=len(scientific_mismatches), dynamic_navigation_mismatch_count=len(mismatches)-len(scientific_mismatches),
        bound_control_sha256=control_hashes, original_completion_audit_bindings=len(audit['audited_artifacts']),
        original_published_outputs=len(audit['published_outputs']),freeze_files=len(frozen['files']),automation_files=len(automation['files']),
        errors=errors, ledger_totals=totals, committed_greedy=sum(r['committed_greedy'] for r in task_rows),
        committed_sampled=sum(r['committed_sampled'] for r in task_rows),
        queue=queue,queue_exit=queue_exit,autofinish=autofinish,live_processes=process_rows,
        locks=lock_rows,tmux=tmux,resource_snapshot=live,
        elapsed_seconds=time.monotonic()-started,
        scope='Full hash bindings, parent identities, task coverage, Oracle/SQLite counters, and live lifecycle; no rerun of inference or numerical statistics')
    write('binding_checks.json',references);write('mismatches.json',mismatches)
    write('tasks_and_ledgers.json',task_rows);write('integrity_report.json',result)
    print(json.dumps({k:result[k] for k in ('status','formal_tasks','canonical_files_hashed','manifest_binding_references','mismatch_reference_count','scientific_mismatch_count','dynamic_navigation_mismatch_count','errors','ledger_totals','elapsed_seconds')},ensure_ascii=False,indent=2),flush=True)
    return 0 if not errors else 1


if __name__=='__main__':
    raise SystemExit(main())

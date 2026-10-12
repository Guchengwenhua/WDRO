"""Read-only independent closeout checks; only writes this review directory."""
from __future__ import annotations
import copy
from collections import Counter, defaultdict
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('frozen_report_review', ROOT / 'report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
import numpy as np
import torch
torch.set_num_threads(2)


def save(name, value):
    report.atomic_json(HERE / name, value)


def identity_now(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), start_ticks=str(fields[19]), state=fields[0],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return None


def inspect_identity(saved, origin):
    if not saved:
        return None
    current = identity_now(saved['pid'])
    matches = bool(current and current['state'] not in ('Z', 'X', 'x')
                   and str(saved.get('start_ticks')) == current['start_ticks']
                   and (not saved.get('boot_id') or saved['boot_id'] == current['boot_id']))
    return dict(origin=origin, recorded=saved, current=current, same_process_still_live=matches,
                boot_identity_available=bool(saved.get('boot_id')))


def inspect_lock(path):
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(path=str(path), released=False)
        fcntl.flock(stream, fcntl.LOCK_UN)
    return dict(path=str(path), released=True)


def smoke_audit(task, p):
    out = ROOT / Path(task['complete']).parent
    done = report.read(out / 'complete.json')
    for key, expected in task['completion_checks'].items():
        report.require(done.get(key) == expected, 'smoke completion field mismatch ' + key)
    report.require(report.REQUIRED_ARTIFACTS <= done['artifacts'].keys(), 'smoke artifact set incomplete')
    for name, expected in done['artifacts'].items():
        path = (out / name).resolve();path.relative_to(out.resolve())
        report.require(report.sha(path) == expected, 'smoke artifact drift ' + str(path))
    heartbeat, metrics, run = [report.read(out / name) for name in ('heartbeat.json', 'metrics.json', 'run.json')]
    report.require(str(heartbeat['device']).startswith('cuda') and heartbeat['phase'] == 'complete', 'smoke CUDA completion missing')
    report.require(metrics['cuda_max_reserved_mib'] > 0 and metrics['cuda_max_allocated_mib'] > 0, 'smoke memory measurement missing')
    protocol = copy.deepcopy(p)
    protocol.update(updates=6, batch_size=64, validation_every=2, checkpoint_every=2,
                    validation_count=64, test_count=64, attack_count=32, greedy_microbatch=64)
    expected_binding = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    report.require(done['frozen_sha256'] == expected_binding and done['protocol_sha256'] == report.sha(ROOT / 'protocol.json'), 'smoke effective protocol mismatch')
    checkpoints = [torch.load(out / name, map_location='cpu', weights_only=False) for name in ('initial.pt','latest.pt','best.pt')]
    report.require(all(c['identity'] == done['identity'] == metrics['identity'] == run['identity'] for c in checkpoints), 'smoke checkpoint identity mismatch')
    latest = checkpoints[1]
    report.require(latest['update'] == 6 and heartbeat['update'] == 6, 'smoke final update missing')
    calls = report.audit_calls(out, protocol, 'wdro', metrics, latest)
    progress = report.json_lines(out / 'progress.jsonl')
    report.require([r['update'] for r in progress] == list(range(1,7)), 'smoke training update sequence mismatch')
    validations = report.json_lines(out / 'validation.jsonl')
    report.require([r['update'] for r in validations] == [0,2,4,6], 'smoke validation schedule mismatch')
    selected = min(validations, key=lambda row: max(row['summary'][d]['cost']['mean'] for d in p['distributions'] if d != 'uniform'))
    report.require(checkpoints[2]['best']['update'] == selected['update'] == metrics['best']['update'], 'smoke selected checkpoint mismatch')
    exit_record = report.read(out / f"exit_{heartbeat['attempt']}.json")
    report.require(exit_record['exit_code'] == 0 and exit_record['scope'] == 'smoke', 'smoke exit missing')
    return dict(task_id=task['id'], status='PASS', completion_sha256=report.sha(out/'complete.json'),
                artifact_count=len(done['artifacts']), calls=calls, selected_update=selected['update'],
                cuda_max_allocated_mib=metrics['cuda_max_allocated_mib'], cuda_max_reserved_mib=metrics['cuda_max_reserved_mib'],
                seconds=metrics['seconds'])


def runtime_audit(queue, p):
    ops = ROOT / 'artifacts/operations'
    state = report.read(ops/'queue_state.json');end = report.read(ops/'exit.json')
    task_map = {t['id']:t for t in queue['tasks']}
    expected = set(task_map)
    report.require(end['status']=='complete' and set(end['completed'])==expected, 'queue exit incomplete')
    report.require(state['status']=='tasks_complete' and set(state['completed'])==expected, 'queue state incomplete')
    report.require(not state['pending'] and not state['running'] and not state['failures'] and not state['exhausted'], 'queue has pending/running/failed tasks')
    sources = report.read(ops/'source_freeze.json')['sha256']
    source_checks = {name:report.sha(name)==digest for name,digest in sources.items()}
    report.require(all(source_checks.values()), 'scheduler source freeze drift')
    identities = [inspect_identity(state['scheduler_identity'], 'queue_state.scheduler')]
    attempts=[];by_task=defaultdict(list);by_gpu=defaultdict(list)
    for path in sorted(ops.glob('attempt_*.json')):
        attempt=report.read(path);result_path=ROOT/attempt['result_path'];result=report.read(result_path)
        key=attempt['task_id'];task=task_map[key]
        report.require(attempt['source_hashes']==sources, 'attempt source binding mismatch '+key)
        report.require(result['returncode'] in (0,75), 'attempt unexpected exit '+key)
        report.require(result['finished_at']>=result['started_at'], 'attempt reversed times')
        histories=attempt['admission_history']
        report.require(len(histories)==3 and sum(x['utilization'] for x in histories)/3<=80 and
                       min(x['utilization'] for x in histories)<=70 and max(x['utilization'] for x in histories)<=95,
                       'multi-sample GPU utilization rule violated '+key)
        required=max(4096,task.get('required_free_mib',4096))
        evidence=attempt['resource_requirement_evidence']
        for dep in task.get('depends_on',[]):
            smoke_path=ROOT/Path(task_map[dep]['complete']).parent/'metrics.json'
            peak=report.read(smoke_path)['cuda_max_reserved_mib'];required=max(required,math.ceil(peak+1024))
            row=next(x for x in evidence if x['task_id']==dep)
            report.require(row['cuda_max_reserved_mib']==peak and row['metrics_sha256']==report.sha(smoke_path), 'measured memory evidence mismatch '+key)
        report.require(attempt['required_free_mib']==required and attempt['admission_free_mib']==required+2048, 'admission memory threshold mismatch '+key)
        report.require(min(x['free_mib'] for x in histories)>=attempt['admission_free_mib'], 'GPU memory rule violated '+key)
        report.require(len(attempt['cpu_affinity'])==task['cpu_threads']==2, 'CPU budget mismatch '+key)
        for owner in ('worker_identity','child_identity'):
            for source,record in [('attempt',attempt),('result',result)]:
                row=inspect_identity(record.get(owner),f'{path.name}:{source}:{owner}')
                if row:identities.append(row)
        if result['returncode']==75:
            report.require(result['status']=='capacity_deferred' and result.get('reason')=='immediate_admission_changed' and result.get('child_identity') is None,
                           'capacity retry included unreported training '+key)
        record=dict(task_id=key, attempt=attempt['attempt'], gpu=attempt['gpu'], cpu_affinity=attempt['cpu_affinity'],
                    started_at=result['started_at'], finished_at=result['finished_at'], elapsed_seconds=result['elapsed_seconds'],
                    returncode=result['returncode'], status=result['status'], reason=result.get('reason',result.get('stop_reason')),
                    child_started=result.get('child_identity') is not None, required_free_mib=required,
                    admission_free_mib=attempt['admission_free_mib'], spec_sha256=report.sha(path),result_sha256=report.sha(result_path))
        attempts.append(record);by_task[key].append((attempt,result));by_gpu[attempt['gpu']].append(record)
    for key,entries in by_task.items():
        entries.sort(key=lambda pair:pair[0]['attempt'])
        report.require(len(entries)==state['attempts'][key] and entries[-1][1]['returncode']==0, 'latest successful attempt missing '+key)
        done=report.read(ROOT/task_map[key]['complete'])
        report.require(state['completed'][key]['sha256']==report.sha(ROOT/task_map[key]['complete']), 'queue completion binding changed '+key)
        report.require(entries[-1][1]['finished_at']>=done['time'], 'completion written after successful wrapper exit '+key)
        out=(ROOT/task_map[key]['complete']).parent
        hb=report.read(out/'heartbeat.json')
        identities.append(inspect_identity(hb['identity'],str(out/'heartbeat.json')))
        child=entries[-1][1]['child_identity']
        report.require(int(hb['identity']['pid'])==int(child['pid']) and str(hb['identity']['start_ticks'])==str(child['start_ticks']), 'runner/wrapper identity mismatch '+key)
    overlaps=[]
    for gpu,rows in by_gpu.items():
        rows.sort(key=lambda row:row['started_at'])
        for left,right in zip(rows,rows[1:]):
            if left['finished_at']>right['started_at']:
                overlaps.append(dict(gpu=gpu,left=left['task_id'],right=right['task_id']))
    report.require(not overlaps,'same-GPU overlapping branch workers')
    report.require(not any(x['same_process_still_live'] for x in identities if x), 'recorded task process still alive')
    locks=[inspect_lock(path) for path in sorted(ops.glob('*.lock'))]
    locks += [inspect_lock((ROOT/task['complete']).parent/'run.lock') for task in queue['tasks']]
    locks.append(inspect_lock(ROOT/'artifacts/report/report.lock'))
    report.require(all(row['released'] for row in locks), 'branch locks still held')
    tmux=subprocess.run(['tmux','-S',str(ops/'tmux.sock'),'list-sessions'],capture_output=True,text=True)
    report.require(tmux.returncode != 0, 'tmux session remains')
    live=[]
    needles={str(ROOT/'runner.py'),str(ROOT/'scheduler.py'),str(ROOT/'report.py')}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            argv=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if needles.intersection(argv):live.append(dict(pid=int(entry.name),argv=argv))
        except (OSError,ProcessLookupError):pass
    report.require(not live,'unrecorded branch workers still live')
    events=[]
    for row in attempts:
        if row['child_started']:
            events.extend([(row['started_at'],1),(row['finished_at'],-1)])
    count=peak=0
    for _,change in sorted(events):count+=change;peak=max(peak,count)
    started=min(row['started_at'] for row in attempts)
    frozen_at=report.read(ops/'source_freeze.json')['at']
    return dict(status='PASS', attempts=attempts, attempt_count=len(attempts), successful_tasks=len(expected),
                source_files_checked=len(source_checks), source_checks=source_checks, identities=identities,locks=locks,
                tmux=dict(returncode=tmux.returncode,stdout=tmux.stdout,stderr=tmux.stderr),
                live_unrecorded_workers=live, same_gpu_overlaps=overlaps, peak_concurrent_workers=peak,
                capacity_deferrals=[row for row in attempts if row['returncode']==75],
                time=dict(first_attempt_at=started, scheduler_freeze_at=frozen_at, queue_complete_at=end['at'],
                          wall_from_first_attempt_seconds=end['at']-started, wall_from_scheduler_freeze_seconds=end['at']-frozen_at,
                          all_attempt_worker_seconds=sum(row['elapsed_seconds'] for row in attempts),
                          cuda_child_worker_seconds=sum(row['elapsed_seconds'] for row in attempts if row['child_started']),
                          formal_worker_seconds=sum(row['elapsed_seconds'] for row in attempts if row['child_started'] and not row['task_id'].startswith('gpu_smoke_')),
                          smoke_worker_seconds=sum(row['elapsed_seconds'] for row in attempts if row['child_started'] and row['task_id'].startswith('gpu_smoke_'))))


def main():
    started=time.time();p=report.read(ROOT/'protocol.json');queue=report.read(ROOT/'queue.json')
    originals={str(path):report.sha(path) for path in (ROOT/'artifacts/report').iterdir() if path.is_file()}
    frozen=report.read(ROOT/'artifacts/frozen.json')
    frozen_checks=[dict(path=name,expected=digest,observed=report.sha(name)) for name,digest in frozen['files'].items()]
    save('frozen_hashes.json',dict(files=frozen_checks,checked=len(frozen_checks),mismatches=[x for x in frozen_checks if x['expected']!=x['observed']]))
    binding,resolved=report.audit_frozen(ROOT,p)
    runtime=runtime_audit(queue,p);save('runtime.json',runtime)
    smokes=[smoke_audit(task,p) for task in queue['tasks'] if task['id'].startswith('gpu_smoke_')]
    save('gpu_smokes.json',dict(status='PASS',runs=smokes))
    results={};failures=[]
    for family in p['families']:
        for method in p['methods']:
            for seed in p['seeds']:
                key=(family,method,seed)
                try:
                    results[key]=report.audit_run(ROOT,p,binding,*key)
                    print(f"PASS {family}_{method}_s{seed}",flush=True)
                except Exception:
                    failures.append(dict(run_id=f'{family}_{method}_s{seed}',error=traceback.format_exc()))
                    print(f"FAIL {family}_{method}_s{seed}",flush=True)
                save('audit_progress.json',dict(status='running',passed=len(results),failures=failures,at=time.time()))
    pairs=[]
    for family in p['families']:
        for seed in p['seeds']:
            er=results.get((family,'erm',seed));wd=results.get((family,'wdro',seed))
            if not er or not wd:continue
            same_actor=er[0]['initial_actor_digest']==wd[0]['initial_actor_digest']
            same_initial=all(np.array_equal(er[1]['test_initial'][k],wd[1]['test_initial'][k]) for k in er[1]['test_initial'])
            pairs.append(dict(family=family,seed=seed,same_initial_actor=same_actor,same_initial_metrics=same_initial))
            if not same_actor or not same_initial:failures.append(dict(run_id=f'{family}_pair_s{seed}',error='initialization mismatch'))
    pickle_path=HERE/'audited_results.pkl'
    with pickle_path.open('wb') as stream:pickle.dump(results,stream,protocol=pickle.HIGHEST_PROTOCOL)
    save('audited_results_manifest.json',dict(path=str(pickle_path),sha256=report.sha(pickle_path),runs=len(results),
         format='trusted local pickle: dict[(family,method,seed)] = (evidence,datasets,risks,rows)',frozen_sha256=binding))
    report.require(report.audit_frozen(ROOT,p)[0]==binding,'source freeze changed during review')
    unchanged={path:report.sha(path)==digest for path,digest in originals.items()}
    report.require(all(unchanged.values()),'original report changed during review')
    evidence=[value[0] for value in results.values()]
    summary=dict(status='PASS' if not failures and len(results)==24 else 'FAIL',started_at=started,finished_at=time.time(),
                 audit_seconds=time.time()-started,frozen_sha256=binding,frozen_files_checked=len(resolved),
                 formal_runs_passed=len(results),gpu_smokes_passed=len(smokes),failures=failures,
                 formal_effective_updates=sum(r['trajectory']['effective_updates'] for r in evidence),
                 logged_training_rows=sum(r['trajectory']['logged_rows'] for r in evidence),
                 discarded_or_duplicate_training_rows=sum(r['trajectory']['discarded_or_duplicate_rows'] for r in evidence),
                 effective_scores=sum(r['calls']['effective_scores'] for r in evidence),
                 actual_started_scores=sum(r['calls']['actual_started_scores'] for r in evidence),
                 actual_completed_scores=sum(r['calls']['actual_completed_scores'] for r in evidence),
                 paired_initializations=pairs,original_report_unchanged=unchanged,runtime_time=runtime['time'],
                 formal_reported_runtime_seconds=sum(r['seconds'] for r in evidence),
                 formal_effective_training_seconds=sum(r['trajectory']['effective_update_seconds'] for r in evidence),
                 capacity_deferrals=runtime['capacity_deferrals'],all_locks_released=True,all_recorded_processes_exited=True,
                 boundary='No model forward pass or GPU computation rerun; validates frozen artifacts, checkpoints, recorded training chains, selection, call ledgers and independently resolved finite-support LPs.',runs=evidence)
    save('integrity_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('runs','paired_initializations','original_report_unchanged','capacity_deferrals')},indent=2),flush=True)
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        save('fatal.json',dict(status='FAIL',error=traceback.format_exc(),at=time.time()))
        raise

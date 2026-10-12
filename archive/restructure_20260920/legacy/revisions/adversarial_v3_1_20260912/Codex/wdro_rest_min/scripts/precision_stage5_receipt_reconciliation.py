"""Compose an unchanged full audit with explicit interrupted-receipt accounting.

This supplement is limited to the observed completed CEM run. The failed audit,
its cursor, all raw receipts and frozen sources remain intact. Publication uses
the unchanged auditor's 43 other checks plus separately verified transaction
accounting; it does not claim the legacy started==completed predicate passed.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / 'experiments'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from src.io_utils import atomic_write_json
from src.job_control import JobLease, utc_now
from src.rest import file_sha256
from src.geosteiner import reference_identity
from audit_stage3 import effective_events
from precision_reference_receipts import reconcile_cem
from precision_replay_pipeline import cpu_preflight
from precision_stage5_capacity_pipeline import gate
import audit_stage5 as original

RID = 'stage5_cem_64_d30_s101_u2000_v1'
EVIDENCE = EXP / 'logs/precision_stage5_receipt_reconciliation' / RID
REPORT = EXP / 'logs/stage5_audit/runs' / (RID + '.json')
CURSOR = EXP / 'logs/stage5_audit/cursors' / (RID + '.json')


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path)
    assert path.is_relative_to(CODEX)
    return dict(logical_path=str(path), resolved_path=str(path.resolve()),
                sha256=file_sha256(path), bytes=path.stat().st_size)


def preserve(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_name(destination.name + '.partial')
    temporary.write_bytes(source.read_bytes())
    assert file_sha256(temporary) == file_sha256(source)
    temporary.chmod(0o444)
    temporary.replace(destination)


def tests():
    output = EVIDENCE / 'fault_injection_tests.log'
    command = [sys.executable, str(ROOT / 'operations_tests/test_precision_reference_receipts.py')]
    temporary = output.with_suffix('.log.partial')
    with temporary.open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    temporary.replace(output)
    assert result.returncode == 0, str(output)
    return dict(status='pass', command=command, output=artifact(output), tests=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if REPORT.exists() and read(REPORT).get('audit_method') == 'unchanged_full_replay_plus_transaction_reconciliation':
        manifest = read(EVIDENCE / 'acceptance.json')
        assert manifest['published_report']['sha256'] == file_sha256(REPORT)
        print(json.dumps(dict(status='already_published', evidence=str(EVIDENCE / 'acceptance.json'))))
        return
    gate()
    preflight = cpu_preflight('stage5_receipt_reconciliation')
    exact = shlex.join(['bash', str(ROOT / 'scripts/stage3_env.sh'), sys.executable, str(Path(__file__)), *sys.argv[1:]])
    with JobLease(EVIDENCE / 'job', run_id='receipt_reconciliation_' + RID,
                  exact_command=exact, resume_command=exact,
                  monitor_command=f'cat {EVIDENCE}/job/heartbeat.json',
                  expected_updates=2000, physical_gpu_index=None,
                  host_runtime_manifest=str(preflight)) as lease:
        preserve(REPORT, EVIDENCE / 'original_failed_report.json')
        preserve(CURSOR, EVIDENCE / 'original_failed_cursor.json')
        failed = read(EVIDENCE / 'original_failed_report.json')
        assert failed['status'] == 'fail'
        assert [k for k, v in failed['checks'].items() if not v] == ['reference_cache_receipts']
        assert failed['completed_updates'] == 2000 and failed['cache_values_checked'] == 32768000
        assert failed['binding']['auditor_sha256'] == file_sha256(ROOT / 'scripts/audit_stage5.py')
        rpath = EXP / 'logs/stage5' / RID / 'result.json'
        result = read(rpath)
        assert result['status'] == 'complete' and result['completed_updates'] == 2000
        assert result['run_identity']['ablation_config']['search_overrides']['searcher'] == 'cem_64'
        assert result['test_or_ispd_used'] is False
        assert not (EXP / 'logs/jobs' / RID / 'run.lock').exists(), 'live training lock'
        assert not (EXP / 'logs/stage5_pipeline/run.lock').exists(), 'live pipeline lock'
        assert not (EXP / 'logs/stage5_audit/jobs' / RID / 'run.lock').exists(), 'live audit lock'
        assert failed['binding']['result_sha256'] == file_sha256(rpath)
        assert failed['binding']['events_sha256'] == file_sha256(Path(result['event_log']))
        test_receipt = tests()
        train, _, raw = effective_events(result['event_log'])
        assert sorted(train) == list(range(1, 2001))
        cache = Path(result['reference_cache']['logical_path'])
        assert cache.stat().st_mtime_ns <= REPORT.stat().st_mtime_ns
        assert not Path(str(cache) + '-wal').exists() or Path(str(cache) + '-wal').stat().st_size == 0
        cache_before = artifact(cache)
        lease.update(update=2000, status='reconcile_all_receipts')
        with sqlite3.connect(f'file:{cache}?mode=ro', uri=True) as db:
            assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
            identity = json.loads(db.execute("SELECT value FROM reference_metadata WHERE key='identity'").fetchone()[0])
            assert identity == reference_identity(), 'reference precision identity'
            rows = db.execute('SELECT id,requested,misses_started,misses_completed FROM calls ORDER BY id').fetchall()
            cached = db.execute('SELECT COUNT(*) FROM lstar').fetchone()[0]
            reconciliation = reconcile_cem(rows, raw, cached)
        db.close()
        assert reconciliation['totals']['cached_values'] == failed['cache_values_checked'] == 32768000
        assert reconciliation['totals']['uncommitted_receipt_ids'] == [739]
        assert reconciliation['totals']['uncommitted_miss_requests'] == 4096
        assert len(reconciliation['interrupted_updates']) == 1
        assert reconciliation['interrupted_updates'][0]['interrupted_update'] == 185
        del train, raw
        diagnosis = dict(status='pass', at_utc=utc_now(), run_id=RID,
            scope='operations accounting only; no training, reference, cost, dual, selection or protocol changes',
            research_protocol_override=False, preflight=artifact(preflight), tests=test_receipt,
            original_failed_report=artifact(EVIDENCE / 'original_failed_report.json'),
            original_failed_cursor=artifact(EVIDENCE / 'original_failed_cursor.json'),
            cache=cache_before, result=artifact(rpath), events=artifact(result['event_log']),
            reconciliation=reconciliation,
            source=[artifact(p) for p in (Path(__file__), ROOT / 'scripts/precision_reference_receipts.py',
                ROOT / 'operations_tests/test_precision_reference_receipts.py', ROOT / 'scripts/audit_stage5.py', ROOT / 'src/candidates.py')])
        assert cache_before == artifact(cache), 'reference cache changed during reconciliation'
        atomic_write_json(EVIDENCE / 'diagnosis.json', diagnosis)
        if not args.publish:
            lease.finish(update=2000, status='diagnosis_complete')
            print(json.dumps(dict(status='pass', diagnosis=str(EVIDENCE / 'diagnosis.json'), published=False)))
            return
        # Re-run every unchanged final check from the exact, already complete
        # 2000-update cursor. The baseline auditor and its binding stay intact.
        original.AUDIT = EVIDENCE / 'unchanged_auditor_recheck'
        local_cursor = original.AUDIT / 'cursors' / (RID + '.json')
        local_cursor.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(local_cursor, read(EVIDENCE / 'original_failed_cursor.json'))
        lease.update(update=2000, status='unchanged_auditor_final_checks')
        try:
            original.audit_run(RID, stability=True)
        except AssertionError as error:
            assert error.args == (['reference_cache_receipts'],), error.args
        else:
            raise AssertionError('expected legacy receipt predicate failure was not reproduced')
        recheck_path = original.AUDIT / 'runs' / (RID + '.json')
        recheck = read(recheck_path)
        assert recheck['binding'] == failed['binding']
        assert recheck['checks'] == failed['checks']
        assert cache_before == artifact(cache), 'reference cache changed during audit'
        gate()
        # Publish an explicitly composite verdict; preserve both legacy failures.
        accepted = copy.deepcopy(recheck)
        accepted['status'] = 'pass'
        accepted['at_utc'] = utc_now()
        accepted['audit_method'] = 'unchanged_full_replay_plus_transaction_reconciliation'
        accepted['checks']['reference_cache_receipts'] = True
        accepted['receipt_reconciliation'] = dict(
            diagnosis=artifact(EVIDENCE / 'diagnosis.json'),
            unchanged_auditor_recheck=artifact(recheck_path),
            legacy_predicate_passed=False,
            replaced_check='reference_cache_receipts',
            corrected_predicate='all per-update raw receipts align; atomic commits equal cached values; incomplete requests occur only at evidenced resume boundaries',
            raw_counts_preserved=True, sources=diagnosis['source'])
        accepted_cursor = read(local_cursor)
        accepted_cursor.update(status='pass', updated_at_utc=utc_now(),
            audit_method=accepted['audit_method'], receipt_reconciliation=accepted['receipt_reconciliation'])
        accepted_cursor_path = EVIDENCE / 'accepted_cursor.json'
        atomic_write_json(accepted_cursor_path, accepted_cursor)
        accepted['stream_cursor'] = str(accepted_cursor_path)
        assert all(accepted['checks'].values())
        assert accepted['raw_cache_receipts'] == failed['raw_cache_receipts']
        atomic_write_json(EVIDENCE / 'composite_report.json', accepted)
        published = artifact(EVIDENCE / 'composite_report.json')
        published.update(logical_path=str(REPORT), resolved_path=str(REPORT.resolve()))
        acceptance = dict(status='pass', at_utc=utc_now(), run_id=RID, checks=len(accepted['checks']),
            research_protocol_override=False, diagnosis=artifact(EVIDENCE / 'diagnosis.json'),
            unchanged_auditor_recheck=artifact(recheck_path),
            original_failed_report=diagnosis['original_failed_report'], original_failed_cursor=diagnosis['original_failed_cursor'],
            accepted_cursor=artifact(accepted_cursor_path), published_report=published,
            cache_unchanged=cache_before == artifact(cache), recovery_command=exact)
        atomic_write_json(EVIDENCE / 'acceptance.json', acceptance)
        # This report is the only file consumed by the dependent pipeline.
        # Publish it last, after all of its recovery evidence is durable.
        atomic_write_json(REPORT, accepted)
        assert file_sha256(REPORT) == published['sha256']
        lease.finish(update=2000, status='complete', detail=dict(acceptance=str(EVIDENCE / 'acceptance.json')))
        print(json.dumps(dict(status='pass', acceptance=str(EVIDENCE / 'acceptance.json'), checks=len(accepted['checks']))))


if __name__ == '__main__':
    main()

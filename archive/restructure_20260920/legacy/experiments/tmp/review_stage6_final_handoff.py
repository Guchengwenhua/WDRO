"""Recheck a completed precision_v2 delivery without rewriting frozen evidence."""
import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path('/remote-home/Guilin/DRO/Codex')
REV = BASE / 'revisions/reference_precision_v2/Codex'
ROOT = REV / 'wdro_rest_min'
EXP = REV / 'experiments'
SCRIPT = BASE / 'experiments/tmp/review_stage6_final_handoff.py'
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from src.io_utils import atomic_write_json
from src.job_control import JobLease, utc_now
from src.metrics import relative_regret, summarize_regret
from src.rest import file_sha256
from run_precision_training_queue import source_gate
from stage6_pipeline import gate


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    p = Path(path)
    return dict(logical_path=str(p), resolved_path=str(p.resolve()), sha256=file_sha256(p))


def process_state(pid):
    try:
        p = Path('/proc') / str(int(pid))
        state = (p / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
        return dict(pid=pid, state=state, active=state not in ('Z', 'X'),
                    cwd=str((p / 'cwd').resolve()) if state not in ('Z', 'X') else None)
    except (OSError, ValueError, TypeError):
        return dict(pid=pid, state='absent', active=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.absolute()
    assert out.is_relative_to(BASE), out
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + f'_{os.getpid()}'
    peak = read(EXP / 'logs/stage6_smoke/memory_profile.json')['maximum_reserved_mib']
    host = out / f'host_runtime_{stamp}.json'
    capture = [sys.executable, str(ROOT / 'scripts/capture_host_runtime.py'), '--purpose',
               'stage6_final_delivery_handoff_cpu_only_no_experiment_launch', '--samples', '3',
               '--sample-interval-seconds', '2', '--measured-peak-mib', str(peak + 1024),
               '--safety-margin-mib', '2048', '--output', str(host)]
    captured = subprocess.run(capture, text=True, capture_output=True)
    atomic_write_json(out / f'preflight_command_{stamp}.json', dict(command=capture,
        returncode=captured.returncode, stdout=captured.stdout, stderr=captured.stderr))
    assert captured.returncode == 0, captured.stdout + captured.stderr
    command = ['bash', str(BASE / 'wdro_rest_min/scripts/stage3_env.sh'), sys.executable,
               str(SCRIPT), '--output-dir', str(out)]
    actual = command + (['--smoke'] if args.smoke else [])
    checks = {}
    files = {}

    def gather(value):
        if isinstance(value, dict):
            if {'logical_path', 'resolved_path', 'sha256'} <= value.keys():
                p = Path(value['logical_path'])
                assert p.is_absolute() and p.is_relative_to(BASE), p
                assert str(p.resolve()) == value['resolved_path'], p
                record = {k: value[k] for k in ('logical_path', 'resolved_path', 'sha256')}
                assert str(p) not in files or files[str(p)] == record, p
                files[str(p)] = record
            for item in value.values():
                gather(item)
        elif isinstance(value, list):
            for item in value:
                gather(item)

    job = out / ('smoke_job' if args.smoke else 'review_job')
    with JobLease(job, run_id='stage6_final_handoff', exact_command=shlex.join(actual),
                  resume_command=shlex.join(command), monitor_command=f'cat {job}/heartbeat.json',
                  expected_updates=300, physical_gpu_index=None, host_runtime_manifest=str(host)) as lease:
        started = time.monotonic()
        source_gate()
        gate()
        checks['existing_frozen_source_and_stage6_gates'] = True
        authority_path = BASE / 'wdro_rest_min/manifests/precision_v2_replay_completion.json'
        authority = read(authority_path)
        cp = ROOT / 'manifests/stage6_completion_v1.json'
        complete = read(cp)
        delivery = read(ROOT / 'manifests/stage6_delivery_verification_v1.json')
        frozen = read(ROOT / 'manifests/stage6_models_frozen_v1.json')
        stats = read(EXP / 'logs/stage6_statistics/result.json')
        selected = read(ROOT / 'configs/stage5_selected_wdro_v1.json')
        assert authority['status'] == complete['status'] == delivery['status'] == stats['status'] == 'pass'
        assert (authority['models'], authority['test_cells'], authority['test_predictions']) == (75, 300, 3000000)
        assert delivery['completion']['sha256'] == file_sha256(cp)
        assert len(complete['checks']) == delivery['checks_passed'] == 310 and all(complete['checks'].values())
        assert frozen['status'] == selected['status'] == 'frozen'
        assert not frozen['test_evaluated_at_freeze'] and not frozen['test_used_for_selection']
        assert not complete['test_used_for_selection'] and not selected['test_or_ispd_used']
        assert selected['source'] == frozen['selection'] == 'synthetic_validation_only'
        assert selected['frozen_at_utc'] < frozen['at_utc']
        assert all(not d['stage7_started'] for d in (authority, complete, delivery, frozen))
        for document in (authority, complete, delivery, frozen, stats, selected):
            gather(document)
        gather(artifact(authority_path))
        for relative, digest in frozen['code_files'].items():
            p = REV / relative
            gather(dict(logical_path=str(p), resolved_path=str(p.resolve()), sha256=digest))
        for name in ('protocol_v1.json', 'data_contract_v1.json'):
            assert (BASE / 'wdro_rest_min/configs' / name).read_bytes() == (ROOT / 'configs' / name).read_bytes()
        checks['authoritative_completion_and_310_check_delivery_binding'] = True
        checks['validation_freeze_and_original_protocol_unchanged'] = True

        models = {(m['method'], m['degree'], m['seed']): m for m in frozen['models']}
        methods = ['erm', 'mixed_erm', 'hard_mining', 'group_dro', 'wdro_rest']
        degrees = [10, 20, 30, 40, 50]
        seeds = [101, 102, 103]
        distributions = ['uniform', 'mix_gauss', 'edge_bias', 'grid_cluster']
        assert len(frozen['models']) == 75 and set(models) == set(itertools.product(methods, degrees, seeds))
        for m in models.values():
            result = read(m['result']['logical_path'])
            assert result['status'] == 'complete' and result['completed_updates'] == 2000
            for ck in ('best', 'latest'):
                assert not list(Path(m[ck]['logical_path']).parent.glob('*.partial'))
            run_id = Path(m['result']['logical_path']).parent.name
            assert not (EXP / 'logs/jobs' / run_id / 'run.lock').exists(), run_id
        audits = [read(r['logical_path']) for r in frozen['full_wdro_audits']]
        assert len(audits) == 14
        for a in audits:
            assert a['status'] == 'pass' and a['completed_updates'] == 2000 and all(a['checks'].values())
            p = EXP / 'logs/stage6' / a['run_id'] / 'result.json'
            assert file_sha256(p) == a['binding']['result_sha256']
        checks['75_unique_complete_models_and_14_full_audit_bindings'] = True

        for r in stats['bootstrap']:
            meta = read(r['logical_path'])
            assert meta['status'] == 'pass'
            gather(meta)
        checks['bootstrap_records_bound_to_original_statistics'] = True
        rows = {Path(r['logical_path']).stem.removeprefix('stage6_'):
                len(list(csv.DictReader(Path(r['logical_path']).open()))) for r in stats['tables']}
        assert rows == stats['rows'] == dict(main=20, per_degree=100, per_seed=300, paired_ci=676, worst_group=30)
        checks['five_tables_and_676_paired_ci_rows'] = True

        auditor_hash = file_sha256(ROOT / 'scripts/verify_stage6_delivery.py')
        evaluator_hash = file_sha256(ROOT / 'stage6/evaluation.py')
        values, hardware, cell_keys = {}, set(), set()
        groups = {}
        paths = sorted((EXP / 'logs/stage6_test').glob('*/*/result.json'))
        assert len(paths) == 300
        for i, p in enumerate(paths[:1] if args.smoke else paths, 1):
            r = read(p)
            b = r['binding']
            m = b['model']
            key = (m['method'], m['degree'], m['seed'], b['distribution'])
            assert key not in cell_keys and models[key[:3]] == m
            cell_keys.add(key)
            assert r['status'] == 'complete' and r['completed_nets'] == 10000
            assert not r['test_used_for_selection'] and b['split'] == 'test'
            assert b['T'] == 1 and b['batch_size'] == 256 and b['mode'] == 'deterministic_greedy'
            assert b['evaluator_sha256'] == evaluator_hash and frozen['at_utc'] < r['at_utc']
            assert r['call_ledger'] == dict(actor_forward_nets=10000, exact_rest_evaluator_nets=10000, geosteiner_nets=0)
            receipt_path = EXP / 'logs/stage6_delivery/cells' / f'{key[0]}_d{key[1]}_s{key[2]}_{key[3]}.json'
            receipt = read(receipt_path)
            assert receipt['status'] == 'pass' and receipt['rest_nets_recomputed'] == 10000
            assert receipt['binding'] == dict(result_sha256=file_sha256(p),
                prediction_sha256=r['predictions']['sha256'], sources=b['sources'], auditor_sha256=auditor_hash)
            gather(b['sources'])
            gather(artifact(receipt_path))
            group = (key[1], key[3])
            if group not in groups:
                with np.load(b['sources']['dataset']['logical_path'], allow_pickle=False) as d:
                    ids = d['net_ids']
                with np.load(b['sources']['reference']['logical_path'], allow_pickle=False) as d:
                    refs = d['lstar']
                groups[group] = ids, refs
            ids, refs = groups[group]
            with np.load(r['predictions']['logical_path'], allow_pickle=False) as a:
                np.testing.assert_array_equal(a['net_ids'], ids)
                np.testing.assert_array_equal(a['lstar'], refs)
                regret = relative_regret(a['rest_lengths'], refs)
                np.testing.assert_array_equal(regret, a['regret'])
                assert r['summary'] == summarize_regret(regret)
            values[key] = regret
            hardware.add(json.dumps(b['hardware'], sort_keys=True))
            if i % 25 == 0:
                lease.update(update=i, status='rechecking_prediction_bindings')
        if not args.smoke:
            assert cell_keys == set(itertools.product(methods, degrees, seeds, distributions)) and len(hardware) == 1
            checks['all_300_prediction_receipts_ids_references_regret_and_hardware'] = True
            for base, degree, dist in itertools.product(methods[:-1], degrees, distributions):
                p = EXP / 'tables/stage6_paired_nets' / f'wdro_minus_{base}_d{degree}_{dist}.npz'
                with np.load(p, allow_pickle=False) as a:
                    np.testing.assert_array_equal(a['seeds'], seeds)
                    np.testing.assert_array_equal(a['relative_regret_difference'], np.stack([
                        values[('wdro_rest', degree, s, dist)] - values[(base, degree, s, dist)] for s in seeds]))
            checks['all_80_paired_net_difference_files'] = True

        atomic_write_json(out / ('smoke_inputs.json' if args.smoke else 'inputs.json'), list(files.values()))
        hashed = []
        for i, row in enumerate(files.values(), 1):
            p = Path(row['logical_path'])
            assert file_sha256(p) == row['sha256'], f'changed artifact: {p}'
            hashed.append(row)
            if args.smoke and i == 3:
                break
            if i % 100 == 0:
                lease.update(update=len(cell_keys), status='rehashing_delivery_artifacts', detail={'files_hashed': i})
        checks['artifact_sha256_and_logical_resolved_paths'] = True
        assert not list((EXP / 'logs/stage6_test').glob('**/*.partial'))
        controllers = {}
        for name in ('replay', 'wdro_replay', 'stage6_completion'):
            root = BASE / 'experiments/logs/precision_v2/jobs' / name
            launch, ex, hb = (read(root / n) for n in ('launch.json', 'exit.json', 'heartbeat.json'))
            assert ex['exit_code'] == 0 and ex['at_utc'] >= launch['at_utc']
            state = process_state(ex['child_pid'])
            assert not state['active'], state
            result = subprocess.run(['tmux', '-S', launch['socket'], 'has-session', '-t', launch['session']], capture_output=True, text=True)
            assert result.returncode == 1, (name, result.stderr)
            assert not (root / 'run.lock').exists(), root
            controllers[name] = dict(exit=ex, process=state, heartbeat_status=hb['status'],
                session=launch['session'], tmux_session_absent=True, launch=artifact(root / 'launch.json'))
        for name in ('stage6_pipeline', 'stage6_statistics', 'stage6_delivery'):
            hb = read(EXP / 'logs' / name / 'heartbeat.json')
            assert hb['status'] == 'complete' and not process_state(hb['pid'])['active']
            assert not (EXP / 'logs' / name / 'run.lock').exists()
        checks['completed_controllers_exited_no_live_locks_or_partial_test_outputs'] = True
        probe = out / f'write_readback_{stamp}.partial'
        probe.write_text(stamp)
        assert probe.read_text() == stamp
        probe.unlink()
        checks['ordinary_user_storage_write_readback_cleanup'] = True
        report = dict(status='smoke_pass' if args.smoke else 'pass', stage=6, at_utc=utc_now(),
            scope='completed delivery handoff; existing formal audits reused through verified hashes',
            authoritative_completion=artifact(authority_path), original_delivery=artifact(ROOT / 'manifests/stage6_delivery_verification_v1.json'),
            script=artifact(SCRIPT), host_runtime=artifact(host), exact_command=shlex.join(actual),
            monitor_command=f'cat {job}/heartbeat.json', reproduce_command=shlex.join(command),
            output_dir=dict(logical_path=str(out), resolved_path=str(out.resolve())),
            checks=checks, model_count=75, checkpoint_hashes_verified=0 if args.smoke else 150,
            test_cells_rechecked=len(cell_keys), predictions_rechecked=10000 * len(cell_keys),
            original_delivery_checks=310, tables=rows, files_hashed=len(hashed), artifacts=hashed,
            original_new_wdro_audit_checks=sum(len(a['checks']) for a in audits),
            controllers=controllers, stage7_started=False,
            additional_call_ledger=dict(actor_forward_nets=0, exact_rest_evaluator_nets=0, geosteiner_nets=0),
            limitation='No new training, inference, topology evaluation, bootstrap resampling or full training-stream reconstruction; original mathematical audits and CI recomputation retained and hash-bound.',
            elapsed_seconds=time.monotonic()-started)
        dest = out / ('smoke.json' if args.smoke else 'review.json')
        atomic_write_json(dest, report)
        assert read(dest) == report
        lease.finish(update=len(cell_keys), status='complete')
        print(json.dumps({k: report[k] for k in ('status', 'model_count', 'checkpoint_hashes_verified',
            'test_cells_rechecked', 'predictions_rechecked', 'files_hashed', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
        print(str(dest), flush=True)


if __name__ == '__main__':
    main()

"""Independent, read-only delivery audit; never trains, scores, or signals processes.

Default dry-run reads only metadata. Full mode requires the production all-phase
manifest and exited controllers, then hashes each referenced physical file once.
It does not repeat the production report's per-net geometry/statistics audit.
Historical controller hashes may resolve only to exact archived operations copies.
"""
from __future__ import annotations
import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
OLD = ROOT.parent.parent / 'adversarial_v3_20260912' / 'Codex'
OUT = EXP / 'diagnostics'
SEEDS = (101, 102, 103)
METHODS = ('random_top4', 'critic_calibrated4', 'rl_frozen4', 'rl_joint4', 'random64', 'es64')
REFERENCE_FIELDS = ('reference_requests', 'geosteiner_misses_started', 'geosteiner_misses_completed')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.partial.{os.getpid()}')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temp.replace(path)


def expected_tasks():
    return {
        'fixed': {f'{p}_d{d}_s{s}': (p, d, s) for p in ('erm', 'wdro_rest')
                  for d in (10, 20, 30, 40, 50) for s in SEEDS},
        'joint': {f'{m}_d30_s{s}': (m, 30, s) for m in METHODS for s in SEEDS},
    }


def historical_roots():
    return [
        ROOT / 'manifests/revision_reason.json', ROOT / 'manifests/source_archive.json',
        ROOT / 'manifests/reference_cache_seed.json',
        OUT / 'network_recovery_audit.json', OUT / 'queue_recovery_proof.json',
        OUT / 'first_formal_task_audit.json', OUT / 'first_formal_task_audit_initial_formula.json',
        OUT / 'actor_batchshape_probe/result.json', OUT / 'actor_batchshape_probe/archived_correspondence.json',
        OLD / 'manifests/source_snapshot.json', OLD / 'manifests/parent_snapshot.json',
        OLD / 'experiments/diagnostics/first_formal_task_audit.json',
        OLD / 'experiments/diagnostics/batchshape_run_stop.json',
        ROOT / 'operations/admission_share_20260912/restart.json',
        ROOT / 'operations/worker_session_isolation_20260912/recovery.json',
    ]


def prerequisites():
    paths = [ROOT/'manifests/all_completion.json', ROOT/'manifests/parent_snapshot.json',
             ROOT/'manifests/source_snapshot.json', EXP/'pipeline/complete.json', EXP/'pipeline/exit.json',
             OUT/'delivery_audit_selftest.json']
    for phase in ('fixed', 'joint'):
        paths += [EXP/f'queue/{phase}/complete.json', EXP/f'queue/{phase}/controller_exit.json']
    if (EXP/'monitor/launch.json').exists():
        paths.append(EXP/'monitor/exit.json')
    return paths + historical_roots()


def process(launch):
    pid = int(launch['pid'])
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return {'pid': pid, 'active': False, 'state': 'exited'}
    start = launch.get('process_start_ticks')
    same = start is None or str(start) == fields[19]
    return {'pid': pid, 'active': same and fields[0] not in ('Z', 'X'),
            'state': fields[0], 'identity_matches': same, 'start_ticks': fields[19]}


class Graph:
    """Hash closure, with one physical-file read per unchanged inode snapshot."""
    def __init__(self, root=ROOT):
        self.root = root
        self.cache, self.bindings, self.visited, self.historical_resolutions = {}, {}, set(), []
        self.bytes_hashed = 0

    def hash(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stamp not in self.cache:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(block)
            end = path.stat()
            require((stat.st_size, stat.st_mtime_ns) == (end.st_size, end.st_mtime_ns), f'file changed during audit: {path}')
            self.cache[stamp] = digest.hexdigest()
            self.bytes_hashed += stat.st_size
        return self.cache[stamp]

    def bind(self, path, digest=None, historical=False):
        path = Path(path)
        if not path.is_absolute():
            path = self.root/path
        original = path
        require(path.is_file(), f'missing artifact: {path}')
        actual = self.hash(path)
        if digest is not None and actual != digest:
            eligible = historical and path.parent == self.root/'adv_v3' and path.name in ('queue.py', 'pipeline.py', 'report.py', 'monitor.py')
            alternatives = [p for p in (self.root/'operations').rglob(path.name) if self.hash(p) == digest] if eligible else []
            require(bool(alternatives), f'hash mismatch: {path}, expected {digest}, got {actual}')
            path = sorted(alternatives)[0]
            actual = digest
            resolution = {'historical_logical_path': str(original), 'sha256': digest, 'archive_path': str(path)}
            if resolution not in self.historical_resolutions:
                self.historical_resolutions.append(resolution)
        key = (str(path), actual)
        self.bindings[key] = {'path': str(path), 'resolved_path': str(path.resolve()), 'sha256': actual}
        return path

    def document(self, path, digest=None, historical=False):
        path = self.bind(path, digest, historical)
        key = (str(path.resolve()), historical)
        if key in self.visited:
            return
        self.visited.add(key)
        if path.suffix == '.json':
            self.walk(read(path), historical)
        elif path.suffix == '.jsonl':
            for line in path.read_text().splitlines():
                if line.strip():
                    self.walk(json.loads(line), historical)

    def walk(self, value, historical):
        if isinstance(value, list):
            for row in value:
                self.walk(row, historical)
        elif isinstance(value, dict):
            digest = value.get('sha256')
            if isinstance(digest, str) and len(digest) == 64:
                path = value.get('path', value.get('source', value.get('logical_path')))
                if isinstance(path, str):
                    self.document(path, digest, historical)
            # Hash-labelled references used by operation receipts/cache manifests.
            for key, digest in value.items():
                if not key.endswith('_sha256') or not isinstance(digest, str) or len(digest) != 64:
                    continue
                stem = key[:-7]
                candidate = value.get(stem)
                if stem == 'seed':
                    candidate = value.get('immutable_seed', candidate)
                if isinstance(candidate, str) and (candidate.startswith('/') or candidate.endswith(('.json', '.jsonl', '.log', '.py', '.npz', '.sqlite', '.pt'))):
                    self.document(candidate, digest, historical)
            for item in value.values():
                if isinstance(item, (dict, list)):
                    self.walk(item, historical)


def lock_released(path):
    require(path.is_file(), f'lock provenance missing: {path}')
    with path.open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'lock still held: {path}') from exc
        fcntl.flock(handle, fcntl.LOCK_UN)
    return str(path)


def research_process_kind(executable, argv):
    """Identify the program Python executes, not command text retained by tmux."""
    if not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', Path(executable).name) or not argv:
        return None
    modules = {f'adv_v3.{name}' for name in ('worker', 'joint', 'queue', 'pipeline', 'monitor')}
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument.startswith(('-m', '-c')):
            payload = argument[2:] if len(argument) > 2 else (argv[index+1] if index+1 < len(argv) else '')
            if argument.startswith('-m'):
                return payload if payload in modules else None
            remainder = argv[index+1:] if len(argument) > 2 else argv[index+2:]
            if '--multiprocessing-fork' not in remainder:
                return None
            try:
                statements = ast.parse(payload).body
            except SyntaxError:
                return None
            spawn_names = set()
            for statement in statements:
                if isinstance(statement, ast.ImportFrom) and statement.module == 'multiprocessing.spawn':
                    spawn_names.update(alias.asname or alias.name for alias in statement.names if alias.name == 'spawn_main')
                elif isinstance(statement, ast.Import):
                    spawn_names.update((alias.asname or alias.name) + '.spawn_main'
                                       for alias in statement.names if alias.name == 'multiprocessing.spawn')
                elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    if ast.unparse(statement.value.func) in spawn_names:
                        return 'multiprocessing.spawn'
            return None
        if argument in ('-W', '-X', '--check-hash-based-pycs'):
            index += 2
        elif argument in ('-', '--') or not argument.startswith('-'):
            return None  # stdin/script arguments cannot select a Python module.
        else:
            index += 1
    return None


def quiet_gate():
    """Check all process identities first; never acquire a live task's lock."""
    states, lock_paths = [], []
    for phase, tasks in expected_tasks().items():
        for name in tasks:
            folder = EXP/phase/name
            require((folder/'complete.json').is_file() and (folder/'exit.json').is_file(), f'incomplete task: {folder}')
            state = process(read(folder/'launch.json'))
            require(not state['active'], f'task still running: {folder}')
            require(read(folder/'exit.json').get('complete') is True, f'task exit not complete: {folder}')
            states.append(dict(state, name=f'{phase}/{name}'));lock_paths.append(folder/'task.lock')
    for folder, lockname in [(EXP/'pipeline', 'pipeline.lock'), (EXP/'queue/fixed', 'queue.lock'), (EXP/'queue/joint', 'queue.lock')]:
        state = process(read(folder/'launch.json'))
        require(not state['active'], f'controller still running: {folder}')
        require(read(folder/'complete.json').get('status') == 'pass', f'controller not complete: {folder}')
        exit_record = read(folder/('exit.json' if folder == EXP/'pipeline' else 'controller_exit.json'))
        require(exit_record['pid'] == state['pid'], f'controller exit identity mismatch: {folder}')
        require(exit_record.get('completed') is True if folder == EXP/'pipeline' else not exit_record.get('active_pids'), f'controller exit incomplete: {folder}')
        states.append(dict(state, name=str(folder.relative_to(EXP))));lock_paths.append(folder/lockname)
    if (EXP/'monitor/launch.json').exists():
        state = process(read(EXP/'monitor/launch.json'))
        require(not state['active'], 'observer still running; wait for its final sample and exit')
        states.append(dict(state, name='monitor'));lock_paths.append(EXP/'monitor/monitor.lock')
    leftovers = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cwd = (entry/'cwd').resolve()
            argv = (entry/'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            executable = str((entry/'exe').resolve(strict=True))
            status = (entry/'stat').read_text().rsplit(') ',1)[1].split()[0]
        except (OSError, RuntimeError):
            continue
        kind = research_process_kind(executable, argv)
        if cwd in (ROOT, OLD) and status not in ('Z', 'X') and kind is not None:
            leftovers.append({'pid': int(entry.name), 'cwd': str(cwd), 'command': ' '.join(argv),
                              'executable': executable, 'argv': argv, 'kind': kind})
    require(not leftovers, f'experiment processes remain: {leftovers}')
    return {'processes': states, 'released_locks': [lock_released(p) for p in lock_paths], 'leftover_processes': leftovers}


def committed_counts(phase, method):
    if phase == 'fixed':
        return {'greedy': 388096, 'forward_rows': 454656, 'padding_rows': 66560, 'sampled': 0}
    budget = 64 if method in ('random64', 'es64') else 4
    online = 16384 if method == 'rl_joint4' else 0
    greedy = 512*64*budget + 4608 + 4096 + 16384 + online
    forward = 512*(5632 if method == 'es64' else 64*budget) + 51200 + (32768 if online else 0)
    return {'greedy': greedy, 'forward_rows': forward, 'padding_rows': forward-greedy, 'sampled': 32768}


def accounting(folder, phase, method, done, graph):
    ledger = done['ledger']
    require(read(folder/'oracle/oracle_ledger.json') == ledger, f'complete/persisted ledger mismatch: {folder}')
    graph.document(folder/'oracle/oracle_ledger.json')
    expected = committed_counts(phase, method)
    pairs = [('actor', expected['greedy']), ('rest', expected['greedy']),
             ('actor_forward_rows', expected['forward_rows']), ('actor_padding_rows', expected['padding_rows'])]
    if phase == 'joint':
        require(done['updates'] == 512 and len(done['history']) == 512, f'joint update count: {folder}')
        pairs += [('sampled_actor', expected['sampled']), ('sampled_rest', expected['sampled'])]
        for index, row in enumerate(done['history'], 1):
            require(row['update'] == index and row['search_calls_per_center'] == (64 if method in ('random64','es64') else 4), f'committed update budget changed: {folder}/{index}')
    counters = {}
    for prefix, committed in pairs:
        low, high = ledger.get(prefix+'_completed', 0), ledger.get(prefix+'_requested', 0)
        require(0 <= committed <= low <= high, f'ledger below committed work: {folder}/{prefix}')
        counters[prefix] = {'committed': committed, 'completed_lower_bound': low, 'requested_upper_bound': high,
                            'extra_completed': low-committed, 'uncertain': high-low}
    require(ledger['reference_requests'] >= expected['greedy'], f'reference requests below committed: {folder}')
    low, high = ledger.get('geosteiner_misses_completed', 0), ledger.get('geosteiner_misses_started', 0)
    require(0 <= low <= high, f'invalid reference miss interval: {folder}')
    counters['reference'] = {'committed_requests': expected['greedy'], 'actual_requests': ledger['reference_requests'],
        'extra_requests': ledger['reference_requests']-expected['greedy'], 'completed_misses': low,
        'started_misses': high, 'uncertain_misses': high-low}
    balances = {}
    for suffix in ('requested', 'completed'):
        balance = ledger['actor_forward_rows_'+suffix] - ledger['actor_'+suffix] - ledger['actor_padding_rows_'+suffix]
        balances[suffix] = balance
        allowance = sum(counters[k]['uncertain'] for k in ('actor','actor_forward_rows','actor_padding_rows'))
        require(abs(balance) <= allowance, f'unexplained padding/forward balance: {folder}')
    with sqlite3.connect((folder/'oracle/reference.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as database:
        values = database.execute('SELECT COALESCE(SUM(requested),0), COALESCE(SUM(misses_started),0), COALESCE(SUM(misses_completed),0) FROM calls').fetchone()
    require(tuple(ledger[k] for k in REFERENCE_FIELDS) == values, f'SQLite receipt mismatch: {folder}')
    reconciliations = []
    for path in sorted((folder/'oracle/reconciliations').glob('*.after.json')):
        receipt = read(path);before, after = receipt['ledger_before'], receipt['ledger_after']
        require(set(receipt['replaced_fields']) == set(REFERENCE_FIELDS), f'unexpected reconciled fields: {path}')
        require(all(before.get(k) == after.get(k) for k in set(before)|set(after) if k not in REFERENCE_FIELDS), f'recovery rewrote actor/REST counters: {path}')
        require(all(after[k] >= before.get(k,0) for k in REFERENCE_FIELDS), f'reference counter rollback: {path}')
        graph.document(path, historical=True)
        reconciliations.append(str(path))
    return {'phase': phase, 'task': folder.name, 'committed': expected, 'counters': counters,
            'forward_minus_real_minus_padding': balances, 'reference_reconciliations': reconciliations,
            'padding_is_not_a_real_score_request': True}


def dry_run():
    checks = [{'path': str(p), 'exists': p.is_file()} for p in prerequisites()]
    counts = {phase: {'expected': len(tasks), 'complete_files': len(list((EXP/phase).glob('*/complete.json')))}
              for phase, tasks in expected_tasks().items()}
    return {'status': 'dry_run_only', 'at_unix': time.time(), 'full_audit_performed': False,
            'required_items': checks, 'missing_items': [r['path'] for r in checks if not r['exists']],
            'phase_counts': counts, 'locks_acquired': 0, 'oracle_calls': 0, 'hashes_recomputed': 0,
            'full_mode_gates': ['all_completion pass with A=30/1080 and B=18/216', 'all tasks/controllers/observer exited before any lock probe',
                'recursive artifact hashes once per unchanged physical file', '801 original sources + 810 old frozen sources + new frozen snapshot preserved',
                'current versus archived operational versions explicitly resolved by exact hashes', 'committed budgets separated from recovery extra/uncertain work and padding',
                'old batchshape results excluded and all recovery history retained'],
            'report_binding_gap': ['operations archives', 'queue_recovery_proof', 'first_formal_task_audit',
                'recursive bindings/evidence/source_files inside the already-bound network recovery evidence',
                'per-task reference reconciliation receipts'],
            'note': 'This independent delivery manifest supplements production all_completion without editing it. Full mode does not rerun geometry, reference solving, model inference or bootstrap.'}


def full():
    missing = [str(p) for p in prerequisites() if not p.is_file()]
    require(not missing, f'full mode prerequisites missing: {missing}')
    completion = read(ROOT/'manifests/all_completion.json')
    require(completion.get('status') == 'pass' and completion.get('phase') == 'all' and completion.get('all_experiments_complete') is True, 'production all_completion is not passing')
    require(completion['phases'] == {'fixed': {'tasks': 30, 'cells': 1080}, 'joint': {'tasks': 18, 'cells': 216}}, 'wrong final experiment coverage')
    quiet = quiet_gate()
    graph = Graph()
    graph.document(ROOT/'manifests/all_completion.json')
    require(completion['protocol_sha256'] == graph.hash(ROOT/'protocol.json'), 'final protocol hash mismatch')
    original_protocol, current_protocol = read(OLD/'protocol.json'), read(ROOT/'protocol.json')
    require(all(current_protocol.get(k) == v for k,v in original_protocol.items() if k != 'plan_sha256'), 'original experiment grid/budget changed')
    require(set(current_protocol)-set(original_protocol) == {'implementation_revision','greedy_inference'}, 'unplanned protocol amendments')
    require(current_protocol['plan_sha256'] == graph.hash(ROOT/'docs/EXPERIMENT_PLAN.md'), 'new plan hash mismatch')
    require(original_protocol['plan_sha256'] == graph.hash(OLD/'docs/EXPERIMENT_PLAN.md'), 'old plan hash mismatch')
    expected_plan = (OLD/'docs/EXPERIMENT_PLAN.md').read_text().replace('adversarial_v3_20260912', 'adversarial_v3_1_20260912').strip()
    actual_plan = (ROOT/'docs/EXPERIMENT_PLAN.md').read_text().split('## v3.1 数值实现修订（正式重跑前冻结）', 1)[0].strip()
    require(actual_plan == expected_plan, 'original plan content changed beyond version paths and appended numerical repair')
    graph.bind(OLD/'docs/EXPERIMENT_PLAN.md')
    require(read(OUT/'delivery_audit_selftest.json')['status'] == 'pass', 'independent auditor self-test failed')
    graph.document(OUT/'delivery_audit_selftest.json')
    coverage = {}
    for label, file, key, count in [('original_sources', ROOT/'manifests/parent_snapshot.json', 'source_files', 801),
                                  ('old_frozen_sources', OLD/'manifests/source_snapshot.json', 'files', 810),
                                  ('new_frozen_sources', ROOT/'manifests/source_snapshot.json', 'files', 810)]:
        document = read(file);require(len(document[key]) == count, f'unexpected source inventory: {file}')
        graph.document(file, historical=label=='old_frozen_sources')
        coverage[label] = count
    for file in historical_roots():
        graph.document(file, historical=True)
    # Bind all actual operational source generations, including paths listed without hashes in historical receipts.
    for file in sorted((ROOT/'operations').rglob('*')):
        if file.is_file():
            graph.document(file, historical=True)
    old_audit = read(OUT/'network_recovery_audit.json')
    require(old_audit['status'] == 'recovery_audit_pass_old_study_ineligible', 'old v3 exclusion is missing')
    require(old_audit['totals']['locally_complete_tasks'] == 11 and old_audit['totals']['cross_method_original_mismatch_rows'] == 14, 'old evidence scope changed')
    require(read(OUT/'first_formal_task_audit.json')['status'] == 'pass', 'first corrected formal task audit failed')
    proof = read(OUT/'queue_recovery_proof.json')
    require(proof['status'] == 'pass' and proof['scientific_modules_changed'] is False, 'queue recovery proof failed')
    accounting_rows = []
    top_artifacts = {str(Path(r['path'])) for r in completion['artifacts']}
    for phase, expected in expected_tasks().items():
        actual = {p.parent.name for p in (EXP/phase).glob('*/complete.json')}
        require(actual == set(expected), f'unexpected/missing formal task set: {phase}')
        table = {r['task']: r for r in csv.DictReader((EXP/'tables'/f'{phase}_efficiency.csv').open())}
        require(set(table) == set(expected), f'efficiency table task coverage: {phase}')
        for name, (method, degree, seed) in expected.items():
            folder = EXP/phase/name;done = read(folder/'complete.json')
            require(str(folder/'complete.json') in top_artifacts, f'unbound task completion: {folder}')
            require(done['status'] == 'pass' and not done.get('smoke') and done['degree'] == degree and done['seed'] == seed, f'task identity: {folder}')
            require(done['parent' if phase=='fixed' else 'method'] == method, f'method identity: {folder}')
            identity = done['identity'];source = identity['source_hashes' if phase=='fixed' else 'source']
            for name_, digest in source.items():
                graph.bind(ROOT/'adv_v3'/name_, digest)
            protocol_hash = identity['protocol_sha256'] if phase=='fixed' else identity['protocol']['sha256']
            require(protocol_hash == completion['protocol_sha256'], f'task protocol drift: {folder}')
            displayed = json.loads(table[name]['actual_ledger'])
            require(all(displayed.get(k) == v for k,v in done['ledger'].items()), f'efficiency ledger conceals actual work: {folder}')
            row = accounting(folder, phase, method, done, graph)
            if phase == 'fixed':
                require(int(table[name]['extra_completed_requests']) == row['counters']['actor']['extra_completed'], f'fixed recovery extra mismatch: {folder}')
            accounting_rows.append(row)
            for file in [folder/'launch.json', folder/'exit.json']:
                graph.document(file)
            for pattern in ('recovery/*.json', 'oracle/reconciliations/*.json'):
                for file in sorted(folder.glob(pattern)):
                    graph.document(file, historical=True)
            for file in (folder/'failure.json', folder/'paused.json'):
                if file.is_file():
                    graph.document(file, historical=True)
    # Closed process/event logs are bound after the quiescence gate, never while writers are running.
    for folder in (EXP/'pipeline', EXP/'queue/fixed', EXP/'queue/joint'):
        launch = read(folder/'launch.json')
        if 'code_sha256' in launch:
            graph.bind(ROOT/'adv_v3'/('pipeline.py' if folder == EXP/'pipeline' else 'queue.py'),
                       launch['code_sha256'], historical=True)
        for file in sorted(folder.iterdir()):
            if file.is_file() and file.suffix in ('.json', '.jsonl', '.log'):
                graph.document(file, historical=True)
    if (EXP/'monitor/launch.json').exists():
        for name in ('launch.json', 'exit.json', 'heartbeat.json'):
            graph.document(EXP/'monitor'/name, historical=True)
    graph.bind(Path(__file__))
    totals = {}
    for row in accounting_rows:
        for prefix, values in row['counters'].items():
            target = totals.setdefault(prefix, {})
            for key, value in values.items():
                target[key] = target.get(key, 0) + value
    return {'status': 'pass', 'at_unix': time.time(), 'full_audit_performed': True, 'oracle_calls': 0,
            'production_completion': {'path': str(ROOT/'manifests/all_completion.json'), 'sha256': graph.hash(ROOT/'manifests/all_completion.json')},
            'phase_coverage': completion['phases'], 'source_coverage': coverage, 'quiescence': quiet,
            'recursive_bindings_verified': len(graph.bindings), 'bytes_hashed_once_per_physical_file': graph.bytes_hashed,
            'historical_operational_resolutions': graph.historical_resolutions,
            'old_v3_results': {'status': 'excluded', 'locally_completed_tasks_retained': 11,
                'known_cross_method_mismatch_rows': 14, 'exact_reference_cache_reuse_is_actor_independent': True},
            'accounting': accounting_rows, 'accounting_totals': totals,
            'interpretation': 'Declared committed search budgets remain unchanged. Recovery work and interrupted-call uncertainty are additional actual work; fixed-shape padding is compute overhead, not a true-score request. No counters or original artifacts were rewritten by this audit.',
            'scope_limit': 'Relies on hash-bound production all_completion for per-net geometry, paired statistics and split verification; those expensive computations are not repeated here.',
            'artifacts': sorted(graph.bindings.values(), key=lambda r:r['path'])}


def self_test():
    count = 0
    with tempfile.TemporaryDirectory(prefix='delivery_audit_selftest_', dir=OUT) as temporary:
        root=Path(temporary);(root/'adv_v3').mkdir();(root/'operations/v0').mkdir(parents=True)
        old=root/'operations/v0/queue.py';old.write_text('old queue\n')
        current=root/'adv_v3/queue.py';current.write_text('new queue\n')
        graph=Graph(root);digest=graph.hash(old)
        graph.document(current,digest,historical=True);require(len(graph.historical_resolutions)==1,'historical resolution test');count+=1
        try:graph.document(current,digest,historical=False)
        except RuntimeError:count+=1
        else:raise RuntimeError('current-source mismatch did not fail')
        leaf=root/'leaf.txt';leaf.write_text('bound leaf\n')
        child=root/'child.json';child.write_text(json.dumps({'evidence':[{'path':str(leaf),'sha256':graph.hash(leaf)}]}))
        main=root/'main.json';main.write_text(json.dumps({'artifacts':[{'path':str(child),'sha256':graph.hash(child)}]}))
        graph.document(main);require(any(r['path']==str(leaf) for r in graph.bindings.values()),'recursive evidence test');count+=1
        before=graph.bytes_hashed;graph.document(main);require(graph.bytes_hashed==before,'duplicate hash read');count+=1
        require(committed_counts('fixed','erm')=={'greedy':388096,'forward_rows':454656,'padding_rows':66560,'sampled':0},'fixed expected counts');count+=1
        require(committed_counts('joint','es64')['greedy']==2122240 and committed_counts('joint','es64')['padding_rows']==812544,'joint ES counts');count+=1
        require(committed_counts('joint','rl_joint4')['greedy']==172544 and committed_counts('joint','rl_joint4')['sampled']==32768,'joint RL counts');count+=1
    return {'status':'pass','tests':count,'oracle_calls':0,'production_locks_acquired':0,'at_unix':time.time()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('dry-run','full','self-test'), default='dry-run')
    args=parser.parse_args();started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        result={'dry-run':dry_run,'full':full,'self-test':self_test}[args.mode]()
    except Exception as exc:
        result={'status':'fail','at_unix':time.time(),'mode':args.mode,'error':repr(exc),'traceback':traceback.format_exc(),'oracle_calls':0}
    result['mode']=args.mode;result['wall_seconds']=time.monotonic()-started
    name={'dry-run':'delivery_preflight.json','full':'delivery_audit.json','self-test':'delivery_audit_selftest.json'}[args.mode]
    path=OUT/name;atomic(path,result)
    print(json.dumps({'path':str(path),'status':result['status'],'missing_items':result.get('missing_items'),
                      'recursive_bindings_verified':result.get('recursive_bindings_verified'), 'error':result.get('error')},indent=2),flush=True)
    raise SystemExit(result['status']=='fail')


if __name__=='__main__':
    main()

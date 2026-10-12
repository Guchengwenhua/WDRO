"""Foreground gates and persistent Stage 7 reference/evaluation queues."""
from __future__ import annotations

import signal
import subprocess
import time

from stage7.common import *
from stage7.evaluation import reference_gate, model_id, python_tree_lengths


def foreground(args, tag, *, peak=3602, gpu_required=False):
    preflight, host = capture('Stage7 foreground ' + tag, peak=peak)
    argv = list(args)
    gpu = None
    if gpu_required:
        candidates = [x for x in host['gpu_decisions'] if x['eligible']]
        assert candidates, 'No GPU capacity for foreground smoke; preserve existing output'
        gpu = max(candidates, key=lambda x: x['minimum_free_memory_mib'])['index']
        argv += ['--gpu', gpu]
    argv += ['--preflight', preflight]
    cmd = command(*argv)
    path = LOG / 'smoke' / f'{tag}_{stamp()}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        proc = subprocess.Popen(cmd, cwd=CODEX, stdout=handle, stderr=subprocess.STDOUT)
        append_jsonl(LOG / 'smoke/launches.jsonl', dict(command=cmd, pid=proc.pid, gpu=gpu,
            preflight=artifact(preflight), log=str(path), at_utc=utc_now()))
        code = proc.wait()
    assert code == 0, f'foreground {tag} failed; {path}'
    return artifact(path)


def smoke():
    plan = gate()
    reference_smoke = LOG / 'smoke/reference_gate.json'
    if not reference_smoke.exists():
        records = []
        for number in (1, 2):
            records.append(foreground(['reference', '--degree', 50, '--max-new-batches', 1], f'reference_segment{number}'))
            cursor = read(CACHE / 'references/degree_50/cursor.json')
            assert cursor['complete_nets'] == number * 128
            assert not (CACHE / 'references/degree_50/result.json').exists()
        from stage7.evaluation import read_shards
        values = read_shards(cursor)
        np.testing.assert_array_equal(values['lstar'], values['verification_lstar'])
        assert len(values['net_ids']) == 256
        with Lease(LOG / 'smoke/duplicate_guard', 'stage7_smoke_lock'):
            rejected = False
            try:
                with Lease(LOG / 'smoke/duplicate_guard', 'stage7_smoke_lock'):
                    pass
            except RuntimeError:
                rejected = True
            assert rejected
        cursor_snapshot = LOG / 'smoke/reference_cursor_256.json'
        atomic_write_json(cursor_snapshot, cursor, mode=0o444)
        atomic_write_json(reference_smoke, dict(status='pass', at_utc=utc_now(), logs=records,
            committed_nets=256, geosteiner_calls=512, two_scale_equal=True,
            interruption_cursor=256, no_premature_result=True, duplicate_rejected=True,
            cursor=artifact(cursor_snapshot)), mode=0o444)
        progress('7.1 reference前台分片恢复smoke通过', 'DONE', str(reference_smoke),
                 '在tmux完成五degree参考缓存；之后前台评价smoke')
        print({'reference_smoke': 'pass', 'next': 'launch reference queue'}, flush=True)
        return
    reference_gate()
    output = LOG / 'smoke/evaluation_gate.json'
    if output.exists():
        assert read(output)['status'] == 'pass'
        print({'evaluation_smoke': 'already_passed'}, flush=True)
        return
    logs, profiles, speeds = [], {}, {}
    model = next(m for m in plan['models'] if m['method'] == 'erm' and m['degree'] == 50 and m['seed'] == 101)
    for name, maximum in [('resumed', 1), ('resumed', None), ('continuous', None)]:
        out = LOG / 'smoke' / name
        args = ['evaluate', '--model', model_id(model), '--limit', 513, '--output', out]
        if maximum:
            args += ['--max-new-batches', maximum]
        logs.append(foreground(args, 'evaluation_' + name, gpu_required=True))
        if maximum:
            assert read(out / 'cursor.json')['complete_nets'] == 256 and not (out / 'result.json').exists()
    resumed = read(LOG / 'smoke/resumed/result.json')
    continuous = read(LOG / 'smoke/continuous/result.json')
    a, b = load_npz(checked(resumed['predictions'])), load_npz(checked(continuous['predictions']))
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert resumed['summary'] == continuous['summary']
    profiles['50'] = max(resumed['peak_reserved_mib'], continuous['peak_reserved_mib'])
    speeds['50'] = (continuous['actor_seconds'] + continuous['rest_seconds'])/513
    data = load_npz(checked(next(g for g in plan['groups'] if g['degree'] == 50)['input']))
    np.testing.assert_array_equal(python_tree_lengths(data['pins'][:513], a['topology']), a['rest_lengths'])
    for degree in (10, 20, 30, 40):
        model = next(m for m in plan['models'] if m['method'] == 'erm' and m['degree'] == degree and m['seed'] == 101)
        out = LOG / 'smoke' / f'peak_d{degree}'
        logs.append(foreground(['evaluate', '--model', model_id(model), '--limit', 256,
                               '--output', out], f'peak_d{degree}', gpu_required=True))
        r = read(out / 'result.json')
        profiles[str(degree)] = r['peak_reserved_mib']
        speeds[str(degree)] = (r['actor_seconds'] + r['rest_seconds'])/256
        data = load_npz(checked(next(g for g in plan['groups'] if g['degree'] == degree)['input']))
        pred = load_npz(checked(r['predictions']))
        np.testing.assert_array_equal(python_tree_lengths(data['pins'][:256], pred['topology']), pred['rest_lengths'])
    atomic_write_json(output, dict(status='pass', at_utc=utc_now(), logs=logs,
        profiles_reserved_mib=profiles, smoke_seconds_per_net=speeds,
        context_allowance_mib=1024, safety_margin_mib=2048,
        exact_resume_nets=513, all_prediction_fields_equal=True, independent_tree_nets=1537,
        actor_calls=2050, rest_calls=2050, model_or_input_changed=False), mode=0o444)
    progress('7.1 五degree显存、513-net精确恢复及独立拓扑评价smoke通过', 'DONE', str(output),
             '冻结阶段7运行代码并启动75模型正式ISPD评价')
    print({'evaluation_smoke': 'pass', 'memory_profile': profiles}, flush=True)


def post_snapshot(tag):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from capture_host_runtime import gpu_sample, compute_processes, disk_snapshot
    row = dict(at_utc=utc_now(), monotonic=time.monotonic(), tag=tag, gpus=gpu_sample(),
               processes=compute_processes(), disk=[disk_snapshot(CODEX), disk_snapshot(EXP)])
    append_jsonl(LOG / 'resources/postlaunch.jsonl', row)
    return row


def process_matches(row):
    current = pid_identity(row['pid'])
    if not current or current['state'] in ('Z', 'X'):
        return False
    if current['start_ticks'] != row['process']['start_ticks'] or current['cwd'] != str(CODEX):
        return False
    args = Path(f"/proc/{row['pid']}/cmdline").read_bytes().split(b'\0')
    return str(SCRIPT).encode() in args


def task_queue(phase, lease):
    plan = gate()
    if phase == 'references':
        tasks = [dict(id=f"degree_{g['degree']}", degree=g['degree'],
                      command=['reference', '--degree', g['degree']],
                      output=str(CACHE / 'references' / f"degree_{g['degree']}" / 'result.json'))
                 for g in sorted(plan['groups'], key=lambda x: -x['degree'])]
        peaks = None
    else:
        peaks = read(LOG / 'smoke/evaluation_gate.json')['profiles_reserved_mib']
        groups = {g['degree']: g for g in plan['groups']}
        tasks = [dict(id=model_id(m), degree=m['degree'], command=['evaluate', '--model', model_id(m)],
                      output=str(LOG / 'evaluation' / model_id(m) / 'result.json'))
                 for m in sorted(plan['models'], key=lambda x: (-groups[x['degree']]['count']*x['degree']**2,
                                                               x['seed'], x['method']))]
    job = LOG / 'pipeline' / phase
    job.mkdir(parents=True, exist_ok=True)
    task_manifest = job / 'tasks.json'
    immutable_json(task_manifest, tasks)
    queue_path = job / 'queue.json'
    old = read(queue_path) if queue_path.exists() else {}
    active, children, history, cooldown, failures = {}, {}, [], {}, []
    for key, row in old.get('active', {}).items():
        if process_matches(row):
            active[key] = row
    stopping = False
    def stop_requested(*unused):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop_requested)
    def stop_all():
        for row in active.values():
            if process_matches(row):
                os.kill(row['pid'], signal.SIGTERM)
        while any(process_matches(row) for row in active.values()):
            lease.update(0, 'waiting_for_worker_checkpoints')
            time.sleep(2)
    try:
        while True:
            completed = set()
            for task in tasks:
                path = Path(task['output'])
                if path.exists():
                    r = read(path)
                    assert r['status'] == 'complete'
                    checked(r['reference' if phase == 'references' else 'predictions'])
                    completed.add(task['id'])
            for key, row in list(active.items()):
                proc = children.get(key)
                code = proc.poll() if proc else None
                alive = code is None and process_matches(row)
                if alive:
                    continue
                active.pop(key)
                if key in completed:
                    continue
                log = Path(row['log']).read_text()
                if row.get('capacity_pause') or 'CUDA out of memory' in log:
                    cooldown[row['gpu']] = time.monotonic() + 60
                    append_jsonl(job / 'recoveries.jsonl', dict(task=key, pid=row['pid'], exit_code=code,
                        reason=row.get('capacity_pause', 'OOM'), at_utc=utc_now()))
                    continue
                raise RuntimeError(f'non-resource failure in {key}; inspect {row["log"]}')
            atomic_write_json(queue_path, dict(phase=phase, at_utc=utc_now(), tasks=len(tasks),
                complete=sorted(completed), active=active, status='complete' if len(completed)==len(tasks) else 'running'))
            lease.update(len(completed), phase, total=len(tasks), active=list(active))
            if stopping:
                stop_all()
                return False
            if len(completed) == len(tasks) and not active:
                return True
            pending = [t for t in tasks if t['id'] not in completed and t['id'] not in active]
            if not pending:
                time.sleep(2)
                continue
            if phase == 'evaluation':
                speeds = read(LOG/'smoke/evaluation_gate.json')['smoke_seconds_per_net']
                for key, row in active.items():
                    cursor_path = Path(row['output']).parent/'cursor.json'
                    if cursor_path.exists() and not row.get('capacity_pause'):
                        state = read(cursor_path)
                        parts = state['shards'][-3:]
                        elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(row['started_at_utc'].replace('Z','+00:00'))).total_seconds()
                        degree = state['binding']['model']['degree']
                        if len(parts)==3 and elapsed>30 and all(
                            (s['actor_seconds']+s['rest_seconds'])/(s['end']-s['start']) > 4*speeds[str(degree)] for s in parts):
                            row['capacity_pause']='sustained_throughput_drop'
                            if process_matches(row):
                                os.kill(row['pid'],signal.SIGTERM)
                            cooldown[row['gpu']]=time.monotonic()+60
            peak = max(peaks.values()) if peaks else 3602
            preflight, host = capture('Stage7 ' + phase + ' queue admission', peak=peak)
            history.append(post_snapshot('queue_monitor'))
            history = history[-150:]
            # Only this queue's workers may be paused, after an atomic batch.
            for gpu in {r['gpu'] for r in active.values() if r['gpu'] is not None}:
                rows = [next(g for g in h['gpus'] if g['index'] == gpu) for h in history[-3:]]
                low_memory = rows[-1]['memory_free_mib'] < 2048
                saturated = len(rows) == 3 and all(r['utilization_gpu_percent'] >= 95 for r in rows)
                if low_memory or saturated:
                    key = next(k for k, r in reversed(list(active.items())) if r['gpu'] == gpu)
                    row = active[key]
                    if not row.get('capacity_pause') and process_matches(row):
                        row['capacity_pause'] = 'memory_margin' if low_memory else 'sustained_saturation'
                        os.kill(row['pid'], signal.SIGTERM)
                        cooldown[gpu] = time.monotonic() + 60
            decisions = {g['index']: dict(g) for g in host['gpu_decisions']}
            batch_started = 0
            for task in pending:
                if phase == 'references':
                    if len(active) >= 2:
                        break
                    gpu = None
                else:
                    eligible = []
                    for gpu, row in decisions.items():
                        occupied = sum(x['gpu'] == gpu for x in active.values())
                        recent = [g for h in history if h['monotonic'] >= time.monotonic()-600
                                  for g in h['gpus'] if g['index'] == gpu]
                        floor = min([row['minimum_free_memory_mib']] + [x['memory_free_mib'] for x in recent])
                        stable = not recent or np.mean([x['utilization_gpu_percent'] for x in recent]) < 85
                        need = peaks[str(task['degree'])] + 1024 + 2048
                        if row['eligible'] and stable and floor > need and occupied < 2 and time.monotonic() > cooldown.get(gpu, 0):
                            eligible.append((occupied, -floor, gpu))
                    if not eligible:
                        break
                    gpu = min(eligible)[2]
                argv = task['command'] + ['--preflight', str(preflight)]
                if gpu is not None:
                    argv += ['--gpu', gpu]
                cmd = command(*argv)
                logfile = job / f"{task['id']}_{stamp()}.log"
                with logfile.open('w') as log:
                    proc = subprocess.Popen(cmd, cwd=CODEX, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row = dict(pid=proc.pid, process=pid_identity(proc.pid), gpu=gpu, command=cmd,
                    preflight=artifact(preflight), log=str(logfile), started_at_utc=utc_now(), output=task['output'])
                active[task['id']] = row
                children[task['id']] = proc
                append_jsonl(job / 'launches.jsonl', dict(id=task['id'], **row))
                atomic_write_json(queue_path, dict(phase=phase, at_utc=utc_now(), tasks=len(tasks),
                    complete=sorted(completed), active=active, status='running'))
                time.sleep(1)
                snapshot = post_snapshot('after_launch_' + task['id'])
                history.append(snapshot)
                if gpu is not None:
                    nowrow = next(x for x in snapshot['gpus'] if x['index'] == gpu)
                    decisions[gpu]['minimum_free_memory_mib'] = min(decisions[gpu]['minimum_free_memory_mib'], nowrow['memory_free_mib'])
                batch_started += 1
                if batch_started >= (2 if phase == 'references' else 8):
                    break
            time.sleep(2)
    except BaseException:
        stop_all()
        raise


def pipeline():
    plan = gate()
    phase = 'evaluation' if (LOG / 'references_gate.json').exists() else 'references'
    preflight, _ = capture('Stage7 persistent ' + phase + ' entry')
    with Lease(LOG / 'pipeline', 'stage7_pipeline_' + phase, str(preflight)) as lease:
        if phase == 'references':
            assert read(LOG / 'smoke/reference_gate.json')['status'] == 'pass'
            if not task_queue(phase, lease):
                return
            refs = reference_gate()
            atomic_write_json(LOG / 'references_gate.json', dict(status='pass', at_utc=utc_now(),
                references=refs, unique_nets=22498, committed_primary_calls=22498,
                committed_verification_calls=22498), mode=0o444)
            progress('7.1 全部22498-net双尺度参考缓存通过', 'DONE', str(LOG / 'references_gate.json'),
                     '前台验证评价恢复及五degree显存，之后正式外部测试')
            lease.update(5, 'references_complete_awaiting_foreground_evaluation_smoke')
            return
        assert read(LOG / 'smoke/evaluation_gate.json')['status'] == 'pass'
        assert (ROOT / 'manifests/stage7_operations_v1.json').exists()
        reference_gate()
        if not task_queue(phase, lease):
            return
        progress('7.2 全部75模型ISPD预测已提交，待独立审计', 'DONE',
                 str(LOG / 'pipeline/evaluation/queue.json'), '执行逐net拓扑与reference审计、统计及交付复核')
        from stage7.delivery import audit, export, verify
        audit()
        export()
        verify()
        lease.update(75, 'complete')


def launch():
    gate()
    finished = ROOT / 'manifests/stage7_delivery_verification_v1.json'
    if finished.exists():
        delivery = read(finished)
        assert delivery['status'] == 'pass'
        checked(delivery['completion'])
        print({'stage7': 'already_complete', 'delivery': artifact(finished)}, flush=True)
        return
    phase = 'evaluation' if (LOG / 'references_gate.json').exists() else 'references'
    if phase == 'evaluation':
        assert read(LOG / 'smoke/evaluation_gate.json')['status'] == 'pass'
        assert read(ROOT / 'manifests/stage7_operations_v1.json')['status'] == 'frozen'
    else:
        assert read(LOG / 'smoke/reference_gate.json')['status'] == 'pass'
    session = f'wdro_stage7_{phase}_v1'
    sock = TOP / 'experiments/tmp/stage7.sock'
    launchdir = LOG / 'launches' / stamp()
    with Lease(LOG / 'launch_guard', 'stage7_launch_guard'):
        exists = subprocess.run(['tmux', '-S', str(sock), 'has-session', '-t', session], capture_output=True)
        if exists.returncode == 0:
            raise RuntimeError('Stage7 tmux session already exists: ' + session)
        job = LOG / 'pipeline/job.json'
        if job.exists():
            prior = read(job).get('process')
            current = pid_identity(prior['pid']) if prior else None
            assert not (current and current['state'] not in ('Z', 'X') and
                        current['start_ticks'] == prior['start_ticks']), 'Stage7 pipeline already running'
        preflight, host = capture('Stage7 tmux launch ' + phase)
        if phase == 'evaluation':
            assert host['preflight_passed'], 'GPU capacity unavailable; retain smoke/checkpoints'
        launchdir.mkdir(parents=True, exist_ok=True)
        shell = launchdir / 'run.sh'
        log = launchdir / 'pipeline.log'
        cmd = command('pipeline')
        shell.write_text('#!/bin/bash\nset -u\ncd ' + shlex.quote(str(CODEX)) + '\n' +
            'export WDRO_TMUX_SESSION=' + shlex.quote(session) + '\n' +
            shlex.join(cmd) + ' > ' + shlex.quote(str(log)) + ' 2>&1\n' +
            'stage7_exit=$?\nprintf \'{"exit_code":%s}\\n\' "$stage7_exit" > ' +
            shlex.quote(str(launchdir / 'exit.json.partial')) + '\n' +
            'mv ' + shlex.quote(str(launchdir / 'exit.json.partial')) + ' ' +
            shlex.quote(str(launchdir / 'exit.json')) + '\nexit "$stage7_exit"\n')
        start = ['tmux', '-S', str(sock), 'new-session', '-d', '-s', session, '-c', str(CODEX), 'bash', str(shell)]
        subprocess.run(start, check=True)
        info = dict(at_utc=utc_now(), phase=phase, session=session, socket=str(sock), command=cmd,
            tmux_command=start, preflight=artifact(preflight), shell=artifact(shell),
            log=dict(logical_path=str(log), resolved_path=str(log.resolve())),
            recovery_command=shlex.join(command('launch')), monitor_command=shlex.join(command('status')),
            launch_directory=str(launchdir))
        atomic_write_json(launchdir / 'launch.json', info)
        atomic_write_json(LOG / 'latest_launch.json', info)
        progress('7.' + ('1 reference' if phase=='references' else '2 ISPD评价') + '持久队列已启动',
                 'DOING', str(launchdir / 'launch.json'), '按冻结门禁推进当前阶段，失败保留分片并恢复')
        print(info, flush=True)


def freeze():
    plan = gate()
    reference_gate()
    smoke_report = read(LOG/'smoke/evaluation_gate.json')
    assert smoke_report['status']=='pass'
    unit = LOG/'unit_tests_correct_cwd.log'
    shared = LOG/'shared_input_regression.log'
    assert 'Ran 9 tests' in unit.read_text() and unit.read_text().rstrip().endswith('OK')
    assert 'Ran 19 tests' in shared.read_text() and shared.read_text().rstrip().endswith('OK')
    preflight, host = capture('Stage7 final environment and code freeze admission',
                               peak=max(smoke_report['profiles_reserved_mib'].values()))
    assert host['preflight_passed']
    ident = stamp()
    runtime = ROOT/'configs'/f'runtime_stage7_{ident}.json'
    maximum = max(r['memory_total_mib'] for r in host['sampling']['samples'][-1]['gpus'])
    allowed = host['eligible_gpu_indices']
    atomic_write_json(runtime, dict(schema_version=1,runtime_id='stage7_'+ident,
        research_protocol_override=False,python=sys.executable,artifact_root='Codex/experiments',
        allowed_gpu_indices=allowed,forbidden_gpu_indices=[r['index'] for r in host['gpu_decisions'] if r['index'] not in allowed],
        allowed_gpu_idle_memory_mib_max=maximum,minimum_free_disk_gib_for_batch=100,
        admission_authority=artifact(preflight),
        note='Host snapshot only; shared-memory capacity and utilization gates are enforced afresh by the Stage7 queue.'),mode=0o444)
    out=LOG/'resources'/f'environment_{ident}.json'
    partial=out.with_suffix('.json.partial')
    cmd=[sys.executable,str(ROOT/'scripts/run_preflight.py'),'--runtime-config',str(runtime),
         '--output',str(partial),'--require-batch-ready']
    run=subprocess.run(cmd,cwd=CODEX,text=True,capture_output=True)
    append_jsonl(LOG/'resources/environment_commands.jsonl',dict(command=cmd,exit_code=run.returncode,
        stdout=run.stdout,stderr=run.stderr,at_utc=utc_now()))
    assert run.returncode==0 and read(partial)['batch_ready'],run.stderr
    os.replace(partial,out)
    sidecar=partial.with_suffix(partial.suffix+'.sha256')
    if sidecar.exists():
        os.replace(sidecar,out.with_suffix('.json.original_output.sha256'))
    from src.rest import DEFAULT_LIBRARY
    files=list((ROOT/'stage7').glob('*.py')) + list((ROOT/'src').glob('*.py')) + [
        SCRIPT,ROOT/'operations_tests/test_stage7.py',ENV,ROOT/'scripts/capture_host_runtime.py',
        ROOT/'scripts/run_preflight.py',ROOT/'stage6/evaluation.py',DEFAULT_LIBRARY]
    contract=dict(status='frozen',stage=7,at_utc=utc_now(),research_protocol_override=False,
        plan=artifact(PLAN),files=[artifact(p) for p in sorted(set(files))],
        gates=[artifact(LOG/'references_gate.json'),artifact(LOG/'smoke/reference_gate.json'),
               artifact(LOG/'smoke/evaluation_gate.json'),artifact(unit),artifact(shared),artifact(out)],
        dynamic_gpu_admission='per queue expansion; all GPUs with capacity; at most two workers per GPU',
        maximum_test_batch=256,training_started=False,stage8_started=False)
    target=ROOT/'manifests/stage7_operations_v1.json'
    assert not target.exists(), 'operations already frozen'
    atomic_write_json(target,contract,mode=0o444)
    progress('7.1 环境、输入、reference、恢复与测试全部通过，运行合约冻结', 'DONE',str(target),
             '新资源预检后执行75冻结模型ISPD评价、独立审计及交付')
    print({'status':'pass','operations':str(target),'environment':str(out),'eligible_gpus':allowed},flush=True)


def status():
    for phase in ('references', 'evaluation'):
        path = LOG / 'pipeline' / phase / 'queue.json'
        if path.exists():
            q = read(path)
            print(dict(phase=phase, completed=len(q['complete']), total=q['tasks'], status=q['status'],
                       active={k:dict(pid=v['pid'], gpu=v['gpu'], alive=process_matches(v)) for k,v in q['active'].items()}))
    for path in [LOG / 'pipeline/heartbeat.json', LOG / 'references_gate.json',
                 LOG / 'smoke/evaluation_gate.json', ROOT / 'manifests/stage7_delivery_verification_v1.json']:
        if path.exists():
            r = read(path)
            print(str(path), {k:v for k,v in r.items() if k in ('status','updated_at_utc','completed_items','at_utc','checks_passed')})

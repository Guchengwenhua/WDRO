"""Foreground gates and durable Stage 8 queue."""
from __future__ import annotations
import signal
import subprocess
import time
from stage8.common import *
from stage8.attack import model_id, task_id
from stage8.audit import audit_cell

def foreground(args, tag, *, peak=None, gpu_required=False):
    preflight, host = capture('Stage8 foreground ' + tag, peak=peak)
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


def smoke():
    plan = gate()
    dest = LOG/'smoke/evaluation_gate.json'
    if dest.exists():
        assert read(dest)['status']=='pass'
        return
    logs, profiles, speeds, audits = [], {}, {}, []
    model = next(m for m in plan['models'] if model_id(m)=='erm_d50_s101')
    out = LOG/'smoke/resumed'
    args=['attack','--model',model_id(model),'--distribution','uniform','--limit',16,'--output',out]
    if not (out/'cursor.json').exists():
        logs.append(foreground(args+['--max-new-rounds',2],'round_pause',gpu_required=True))
    state=read(out/'cursor.json')
    if not (out/'result.json').exists():
        assert state['complete_nets']==0 and len(state['current']['rounds'])==2
        atomic_write_json(LOG/'smoke/paused_cursor.json',state,mode=0o444)
        logs.append(foreground(args,'round_resume',peak=state['peak_reserved_mib'],gpu_required=True))
    resumed=read(out/'result.json')
    continuous=LOG/'smoke/continuous'
    if not (continuous/'result.json').exists():
        logs.append(foreground(['attack','--model',model_id(model),'--distribution','uniform','--limit',16,
            '--output',continuous,'--workers',2],'continuous_two_workers',peak=resumed['peak_reserved_mib'],gpu_required=True))
    a,b=load_npz(checked(resumed['predictions'])),load_npz(checked(read(continuous/'result.json')['predictions']))
    for k in a:
        np.testing.assert_array_equal(a[k],b[k])
    ca,cb=read(out/'cursor.json'),read(continuous/'cursor.json')
    for ra,rb in zip(ca['batches'][0]['rounds'],cb['batches'][0]['rounds'],strict=True):
        va,vb=load_npz(checked(ra['artifact'])),load_npz(checked(rb['artifact']))
        for k in va:
            np.testing.assert_array_equal(va[k],vb[k])
    for directory in (out,continuous):
        r=read(directory/'result.json')
        assert r['call_ledger']==dict(actor_forward_nets=4096,exact_rest_evaluator_nets=4096,
            geosteiner_search_nets=4096,geosteiner_selected_verification_nets=16)
        audit_cell(directory)
        audits.append(artifact(directory/'audit.json'))
    profiles['50']=max(resumed['peak_reserved_mib'],read(continuous/'result.json')['peak_reserved_mib'])
    speeds['50']=resumed['times']['wall_seconds']/16
    for degree in (10,20,30,40):
        directory=LOG/'smoke'/f'peak_d{degree}'
        if not (directory/'result.json').exists():
            logs.append(foreground(['attack','--model',f'erm_d{degree}_s101','--distribution','uniform',
                '--limit',16,'--output',directory],f'peak_d{degree}',peak=profiles['50'],gpu_required=True))
        r=read(directory/'result.json')
        profiles[str(degree)]=r['peak_reserved_mib']
        speeds[str(degree)]=r['times']['wall_seconds']/16
        audit_cell(directory)
        audits.append(artifact(directory/'audit.json'))
    with Lease(LOG/'smoke/lock_test','stage8_lock_test'):
        try:
            with Lease(LOG/'smoke/lock_test','stage8_lock_test_duplicate'):
                pass
        except RuntimeError:
            pass
        else:
            raise AssertionError('duplicate job accepted')
    report=dict(status='pass',at_utc=utc_now(),plan=artifact(PLAN),logs=logs,audits=audits,
        profiles_reserved_mib=profiles,smoke_seconds_per_center=speeds,context_allowance_mib=1024,
        safety_margin_mib=2048,exact_round_resume=True,worker_count_equivalence=[2,3],
        centers=96,candidate_scores=24576,duplicate_lock_rejected=True,all_candidate_replay=True)
    atomic_write_json(dest,report,mode=0o444)
    progress('8.1 CEM-256全候选审计、逐轮精确恢复和五degree显存smoke通过','DONE',str(dest),
             '完成运行测试及合约冻结，再启动300格持久攻击队列')
    print(report,flush=True)


def task_queue(lease):
    plan=gate()
    smoke_report=read(LOG/'smoke/evaluation_gate.json')
    peaks=smoke_report['profiles_reserved_mib']
    speeds=smoke_report['smoke_seconds_per_center']
    tasks=[dict(id=task_id(m,d),degree=m['degree'],model=m,distribution=d,
        output=str(LOG/'attacks'/task_id(m,d)/'result.json'))
        for m in sorted(plan['models'],key=lambda m:(-m['degree'],m['seed'],m['method']))
        for d in ('uniform','mix_gauss','edge_bias','grid_cluster')]
    job=LOG/'pipeline/attacks';job.mkdir(parents=True,exist_ok=True)
    immutable_json(job/'tasks.json',tasks)
    queue_path=job/'queue.json'
    old=read(queue_path) if queue_path.exists() else {}
    active,children,history,cooldown={},{},[],{}
    # A healthy worker is adopted after controller reconnection.
    for key,row in old.get('active',{}).items():
        if process_matches(row):
            active[key]=row
    stopping=False
    def stop_requested(*unused):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop_requested)
    def stop_all():
        for row in active.values():
            if process_matches(row):
                os.kill(row['pid'],signal.SIGTERM)
        while any(process_matches(r) for r in active.values()):
            lease.update(0,'waiting_for_round_checkpoints')
            time.sleep(2)
    completed=set()  # revalidate every existing result once on controller resume
    last_progress=0
    try:
        while True:
            for task in tasks:
                dest=Path(task['output'])
                if dest.exists() and task['id'] not in completed:
                    r=read(dest)
                    assert r['status']=='complete' and r['completed_nets']==2000
                    assert r['binding']['plan']==artifact(PLAN)
                    assert r['binding']['model']==task['model']
                    checked(r['predictions'])
                    completed.add(task['id'])
            for key,row in list(active.items()):
                proc=children.get(key)
                code=proc.poll() if proc else None
                if code is None and process_matches(row):
                    continue
                active.pop(key)
                log=Path(row['log']).read_text()
                if (code not in (None,0) or 'Traceback (most recent call last)' in log) and 'CUDA out of memory' not in log:
                    raise RuntimeError(f'non-resource failure {key}: {row["log"]}')
                if key in completed:
                    continue
                if row.get('capacity_pause') or 'CUDA out of memory' in log or code is None:
                    cooldown[row['gpu']]=time.monotonic()+60
                    append_jsonl(job/'recoveries.jsonl',dict(task=key,pid=row['pid'],exit_code=code,
                        reason=row.get('capacity_pause','lost worker or OOM; recover committed rounds'),at_utc=utc_now()))
                else:
                    raise RuntimeError(f'incomplete attack exited without resource pause: {key}')
            atomic_write_json(queue_path,dict(at_utc=utc_now(),tasks=300,complete=sorted(completed),active=active,
                status='complete' if len(completed)==300 and not active else 'running'))
            lease.update(len(completed),'attacks',total=300,active=list(active))
            if time.monotonic()-last_progress>600:
                progress(f'8.2 CEM-256已完成{len(completed)}/300格，{len(active)}任务活动','DOING',str(queue_path),
                    '同一冻结队列完成攻击、全候选审计与最终统计；恢复命令见stage8/latest_launch.json')
                last_progress=time.monotonic()
            if stopping:
                stop_all();return False
            if len(completed)==300 and not active:
                return True
            preflight,host=capture('Stage8 attack queue admission',peak=max(peaks.values()))
            history.append(post_snapshot('queue_monitor'));history=history[-100:]
            for gpu in {r['gpu'] for r in active.values()}:
                rows=[next(g for g in h['gpus'] if g['index']==gpu) for h in history[-3:]]
                low=rows[-1]['memory_free_mib']<2048
                saturated=len(rows)==3 and all(r['utilization_gpu_percent']>=95 for r in rows)
                if low or saturated:
                    key=next(k for k,r in reversed(list(active.items())) if r['gpu']==gpu)
                    row=active[key]
                    if not row.get('capacity_pause') and process_matches(row):
                        row['capacity_pause']='memory_margin' if low else 'sustained_saturation'
                        os.kill(row['pid'],signal.SIGTERM);cooldown[gpu]=time.monotonic()+60
            for key,row in active.items():
                path=Path(row['output']).parent/'cursor.json'
                if not path.exists() or row.get('capacity_pause'):
                    continue
                state=read(path)
                rounds=[r for b in state['batches'][-1:] for r in b['rounds']][-4:]
                if len(rounds)==4 and sum(r['wall_seconds'] for r in rounds)>4*speeds[str(row['degree'])]*4:
                    if sum(r['gpu']==row['gpu'] for r in active.values())>1:
                        row['capacity_pause']='sustained_throughput_drop'
                        if process_matches(row):os.kill(row['pid'],signal.SIGTERM)
                        cooldown[row['gpu']]=time.monotonic()+60
            decisions={g['index']:dict(g) for g in host['gpu_decisions']}
            pending=[t for t in tasks if t['id'] not in completed and t['id'] not in active]
            launched=0
            for task in pending:
                if len(active)>=min(8,max(1,len(os.sched_getaffinity(0))//3)):
                    break
                eligible=[]
                for gpu,row in decisions.items():
                    occupied=sum(r['gpu']==gpu for r in active.values())
                    recent=[g for h in history if h['monotonic']>=time.monotonic()-600
                            for g in h['gpus'] if g['index']==gpu]
                    floor=min([row['minimum_free_memory_mib']]+[r['memory_free_mib'] for r in recent])
                    stable=not recent or np.mean([r['utilization_gpu_percent'] for r in recent])<85
                    need=peaks[str(task['degree'])]+1024+2048
                    if row['eligible'] and stable and floor>need and occupied<2 and time.monotonic()>cooldown.get(gpu,0):
                        eligible.append((occupied,-floor,gpu))
                if not eligible:break
                gpu=min(eligible)[2]
                cmd=command('attack','--model',model_id(task['model']),'--distribution',task['distribution'],
                            '--gpu',gpu,'--preflight',preflight,'--workers',3)
                log=job/f'{task["id"]}_{stamp()}.log'
                with log.open('w') as f:
                    proc=subprocess.Popen(cmd,cwd=CODEX,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                row=dict(pid=proc.pid,process=pid_identity(proc.pid),gpu=gpu,degree=task['degree'],command=cmd,
                    log=str(log),output=task['output'],preflight=artifact(preflight),started_at_utc=utc_now())
                active[task['id']]=row;children[task['id']]=proc
                append_jsonl(job/'launches.jsonl',dict(id=task['id'],**row))
                atomic_write_json(queue_path,dict(at_utc=utc_now(),tasks=300,complete=sorted(completed),active=active,status='running'))
                time.sleep(1)
                snapshot=post_snapshot('after_launch_'+task['id']);history.append(snapshot)
                nowrow=next(g for g in snapshot['gpus'] if g['index']==gpu)
                # Reserve the full measured allocation before the CUDA context
                # appears in nvidia-smi; subsequent expansions resample afresh.
                decisions[gpu]['minimum_free_memory_mib']=min(nowrow['memory_free_mib'],
                    decisions[gpu]['minimum_free_memory_mib']-peaks[str(task['degree'])]-1024)
                launched+=1
                if launched>=4:break
            time.sleep(10)
    except BaseException:
        stop_all()
        raise


def pipeline():
    gate()
    assert read(ROOT/'manifests/stage8_operations_v1.json')['status']=='frozen'
    preflight,_=capture('Stage8 persistent pipeline entry',peak=max(read(LOG/'smoke/evaluation_gate.json')['profiles_reserved_mib'].values()))
    with Lease(LOG/'pipeline','stage8_pipeline',str(preflight)) as lease:
        if not task_queue(lease):return
        from stage8.audit import audit
        from stage8.delivery import export,verify
        audit()
        export()
        verify()
        lease.update(300,'complete')


def launch():
    gate()
    assert read(ROOT/'manifests/stage8_operations_v1.json')['status']=='frozen'
    if (ROOT/'manifests/stage8_delivery_verification_v1.json').exists():
        assert read(ROOT/'manifests/stage8_delivery_verification_v1.json')['status']=='pass'
        print({'stage8':'already_complete'},flush=True);return
    session='wdro_stage8_cem256_v1';sock=TOP/'experiments/tmp/stage8.sock'
    launchdir=LOG/'launches'/stamp()
    with Lease(LOG/'launch_guard','stage8_launch_guard'):
        exists=subprocess.run(['tmux','-S',str(sock),'has-session','-t',session],capture_output=True)
        if exists.returncode==0:raise RuntimeError('Stage8 tmux session already exists: '+session)
        job=LOG/'pipeline/job.json'
        if job.exists():
            prior=read(job).get('process');current=pid_identity(prior['pid']) if prior else None
            assert not(current and current['state'] not in ('Z','X') and current['start_ticks']==prior['start_ticks']), 'Stage8 pipeline already running'
        peak=max(read(LOG/'smoke/evaluation_gate.json')['profiles_reserved_mib'].values())
        preflight,host=capture('Stage8 tmux launch',peak=peak)
        assert host['preflight_passed'],'capacity unavailable; preserve all committed rounds'
        launchdir.mkdir(parents=True,exist_ok=True)
        shell=launchdir/'run.sh';log=launchdir/'pipeline.log';cmd=command('pipeline')
        shell.write_text('#!/bin/bash\nset -u\ncd '+shlex.quote(str(CODEX))+'\n'+
            'export WDRO_TMUX_SESSION='+shlex.quote(session)+'\n'+shlex.join(cmd)+' > '+shlex.quote(str(log))+' 2>&1\n'+
            'stage8_exit=$?\nprintf \'{"exit_code":%s}\\n\' "$stage8_exit" > '+shlex.quote(str(launchdir/'exit.json.partial'))+'\n'+
            'mv '+shlex.quote(str(launchdir/'exit.json.partial'))+' '+shlex.quote(str(launchdir/'exit.json'))+'\nexit "$stage8_exit"\n')
        start=['tmux','-S',str(sock),'new-session','-d','-s',session,'-c',str(CODEX),'bash',str(shell)]
        subprocess.run(start,check=True)
        info=dict(at_utc=utc_now(),session=session,socket=str(sock),command=cmd,tmux_command=start,
            preflight=artifact(preflight),shell=artifact(shell),log=dict(logical_path=str(log),resolved_path=str(log.resolve())),
            recovery_command=shlex.join(command('launch')),monitor_command=shlex.join(command('status')),
            launch_directory=str(launchdir))
        atomic_write_json(launchdir/'launch.json',info);atomic_write_json(LOG/'latest_launch.json',info)
        progress('8.2 300格CEM-256持久队列已启动','DOING',str(launchdir/'launch.json'),
            '按冻结门禁完成攻击、审计、效率与最终统计；阶段9保持未启动')
        print(info,flush=True)


def launch_supporting():
    gate()
    if (LOG/'supporting_statistics/result.json').exists():
        assert read(LOG/'supporting_statistics/result.json')['status']=='pass'
        print({'supporting_statistics':'already_complete'});return
    assert read(LOG/'smoke/ispd_statistics.json')['status']=='pass'
    assert read(LOG/'smoke/supporting_input.json')['status']=='pass'
    session='wdro_stage8_supporting_v1';sock=TOP/'experiments/tmp/stage8.sock'
    with Lease(LOG/'supporting_launch_guard','stage8_supporting_launch_guard'):
        exists=subprocess.run(['tmux','-S',str(sock),'has-session','-t',session],capture_output=True)
        if exists.returncode==0:raise RuntimeError('supporting tmux session already exists')
        preflight,_=capture('Stage8 supporting statistics persistent launch (CPU)',peak=0)
        directory=LOG/'supporting_launch'/stamp();directory.mkdir(parents=True,exist_ok=True)
        cmd=command('supporting');log=directory/'output.log';shell=directory/'run.sh'
        shell.write_text('#!/bin/bash\nset -u\ncd '+shlex.quote(str(CODEX))+'\nexport WDRO_TMUX_SESSION='+shlex.quote(session)+'\n'+
            shlex.join(cmd)+' > '+shlex.quote(str(log))+' 2>&1\nsupporting_exit=$?\n'+
            'printf \'{"exit_code":%s}\\n\' "$supporting_exit" > '+shlex.quote(str(directory/'exit.json.partial'))+'\n'+
            'mv '+shlex.quote(str(directory/'exit.json.partial'))+' '+shlex.quote(str(directory/'exit.json'))+'\nexit "$supporting_exit"\n')
        subprocess.run(['tmux','-S',str(sock),'new-session','-d','-s',session,'-c',str(CODEX),'bash',str(shell)],check=True)
        info=dict(session=session,socket=str(sock),at_utc=utc_now(),command=cmd,shell=artifact(shell),
            log=dict(logical_path=str(log),resolved_path=str(log.resolve())),preflight=artifact(preflight),
            recovery_command=shlex.join(command('launch-supporting')),monitor_command=f'cat {LOG}/supporting_statistics/heartbeat.json')
        atomic_write_json(directory/'launch.json',info);atomic_write_json(LOG/'supporting_launch/latest.json',info)
        print(info,flush=True)


def freeze():
    gate()
    report=read(LOG/'smoke/evaluation_gate.json');assert report['status']=='pass'
    extra_gates=[LOG/'smoke/supporting_delivery_audit.json',LOG/'smoke/final_aggregation.json',
                 LOG/'smoke/geometry_and_budget_figure.json',LOG/'smoke/budget_figure_v2.json']
    for path in extra_gates:
        assert read(path)['status']=='pass',path
    from stage8.delivery import check_tree
    check_tree(read(LOG/'smoke/supporting_delivery_audit.json'))
    unit=LOG/'unit_tests.log'
    assert unit.read_text().rstrip().endswith('OK') and 'Ran ' in unit.read_text()
    preflight,host=capture('Stage8 final environment and code freeze',peak=max(report['profiles_reserved_mib'].values()))
    assert host['preflight_passed']
    ident=stamp();runtime=ROOT/'configs'/f'runtime_stage8_{ident}.json'
    allowed=host['eligible_gpu_indices'];maximum=max(g['memory_total_mib'] for g in host['sampling']['samples'][-1]['gpus'])
    atomic_write_json(runtime,dict(schema_version=1,runtime_id='stage8_'+ident,research_protocol_override=False,
        python=sys.executable,artifact_root='Codex/experiments',allowed_gpu_indices=allowed,
        forbidden_gpu_indices=[g['index'] for g in host['gpu_decisions'] if g['index'] not in allowed],
        allowed_gpu_idle_memory_mib_max=maximum,minimum_free_disk_gib_for_batch=100,admission_authority=artifact(preflight)),mode=0o444)
    env=LOG/'resources'/f'environment_{ident}.json';partial=env.with_suffix('.json.partial')
    cmd=[sys.executable,str(ROOT/'scripts/run_preflight.py'),'--runtime-config',str(runtime),
         '--output',str(partial),'--require-batch-ready']
    proc=subprocess.run(cmd,cwd=CODEX,text=True,capture_output=True)
    append_jsonl(LOG/'resources/environment_commands.jsonl',dict(command=cmd,exit_code=proc.returncode,
        stdout=proc.stdout,stderr=proc.stderr,at_utc=utc_now()))
    assert proc.returncode==0 and read(partial)['batch_ready'],proc.stderr
    os.replace(partial,env)
    side=partial.with_suffix(partial.suffix+'.sha256')
    if side.exists():os.replace(side,env.with_suffix('.json.original_output.sha256'))
    files=list((ROOT/'stage8').glob('*.py'))+list((ROOT/'src').glob('*.py'))+[
        SCRIPT,ENV,ROOT/'stage5/search.py',ROOT/'stage6/statistics.py',ROOT/'stage6/evaluation.py',
        ROOT/'stage7/evaluation.py',ROOT/'stage7/delivery.py',ROOT/'operations_tests/test_stage8.py',
        ROOT/'operations_tests/stage8_delivery_smoke.py',
        ROOT/'scripts/capture_host_runtime.py',ROOT/'scripts/run_preflight.py']
    dest=ROOT/'manifests/stage8_operations_v1.json'
    assert not dest.exists()
    atomic_write_json(dest,dict(status='frozen',stage=8,at_utc=utc_now(),plan=artifact(PLAN),
        files=[artifact(p) for p in sorted(set(files))],gates=[artifact(p) for p in
            [LOG/'smoke/evaluation_gate.json',unit,env,*extra_gates]],
        measured_peak_plus_context_and_margin=True,research_protocol_override=False,stage9_started=False),mode=0o444)
    progress('8.1 输入、真实256-call、全候选审计、恢复、测试与环境门禁通过','DONE',str(dest),
             '现场预检后启动300格CEM-256攻击及最终统计流水线')
    print({'status':'frozen','contract':str(dest)},flush=True)


def status():
    path=LOG/'pipeline/attacks/queue.json'
    if path.exists():
        q=read(path);active={}
        for key,row in q['active'].items():
            p=Path(row['output']).parent/'cursor.json';s=read(p) if p.exists() else {}
            active[key]=dict(pid=row['pid'],gpu=row['gpu'],alive=process_matches(row),
                complete_nets=s.get('complete_nets',0),round=len((s.get('current') or {}).get('rounds',[])))
        print(dict(status=q['status'],complete_cells=len(q['complete']),total=300,active=active),flush=True)
    for p in (LOG/'pipeline/heartbeat.json',ROOT/'manifests/stage8_delivery_verification_v1.json'):
        if p.exists():print(str(p),read(p),flush=True)

#!/usr/bin/env python3
"""Resume Stage 8 after attack completion, using isolated phase processes.

The original frozen auditor, statistical estimands and attack files are reused.
No stage8.operations.post_snapshot is imported: it places scripts/ before the
package root and can make scripts/stage7.py shadow the stage7 package.
"""
import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage8.common import *

SELF=Path(__file__).resolve()
JOB=LOG/'recovery_v1'
CONTRACT=ROOT/'manifests/stage8_finish_runtime_v1.json'
SESSION='wdro_stage8_finish_v1'
SOCKET=TOP/'experiments/tmp/stage8.sock'


def own_command(action):
    return ['bash',str(ENV),sys.executable,str(SELF),action]


def alive(row):
    current=pid_identity(row['pid'])
    if not current or current['state'] in ('Z','X') or current['cwd']!=str(CODEX):return False
    if current['start_ticks']!=row['process']['start_ticks']:return False
    args=Path(f"/proc/{row['pid']}/cmdline").read_bytes().split(b'\0')
    return str(SCRIPT).encode() in args or str(SELF).encode() in args


def verify_contract():
    gate();r=read(CONTRACT)
    assert r['status']=='frozen' and r['stage']==8 and not r['research_protocol_override']
    for rec in r['files']+r['gates']:checked(rec)


def set_recovery(lease):
    lease.job.update(recovery_command=shlex.join(own_command('launch')),
        monitor_command=shlex.join(own_command('status')))
    atomic_write_json(lease.directory/'job.json',lease.job)


def phase(action,lease):
    pref,_=capture('Stage8 recovery phase '+action+' (CPU)',peak=0)
    directory=JOB/'phases'/f'{action}_{stamp()}';directory.mkdir(parents=True,exist_ok=True)
    cmd=command(action);log=directory/'output.log';began=time.monotonic()
    with log.open('w') as handle:
        p=subprocess.Popen(cmd,cwd=CODEX,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(pid=p.pid,process=pid_identity(p.pid),command=cmd,preflight=artifact(pref),
            log=dict(logical_path=str(log),resolved_path=str(log.resolve())),at_utc=utc_now(),
            recovery_command=shlex.join(own_command('launch')))
        atomic_write_json(directory/'launch.json',row)
        atomic_write_json(JOB/'current_phase.json',dict(action=action,**row))
        try:
            while p.poll() is None:
                lease.update(300,action,pid=p.pid,phase_log=str(log));time.sleep(5)
            assert p.returncode==0,f'{action} failed, see {log}'
        except BaseException:
            if alive(row):p.terminate()
            p.wait();raise
    atomic_write_json(directory/'exit.json',dict(exit_code=p.returncode,wall_seconds=time.monotonic()-began,
        finished_at_utc=utc_now(),log=artifact(log)))


def audits(lease):
    inventory=read(checked(read(CONTRACT)['inventory']))
    tasks=inventory['records'];assert len(tasks)==300
    queuefile=JOB/'audit_queue.json';old=read(queuefile) if queuefile.exists() else {}
    active={k:r for k,r in old.get('active',{}).items() if alive(r)};children={};complete=set()
    last_progress=0.;last_check=0.
    def stop_owned():
        for row in active.values():
            if alive(row):os.kill(row['pid'],signal.SIGTERM)
        for p in children.values():p.wait()
    try:
        while len(complete)<300 or active:
            for item in tasks:
                key=item['task'];directory=Path(item['result']['logical_path']).parent
                dest=directory/'audit.json'
                if key not in complete and dest.exists():
                    r=read(dest)
                    assert r['status']=='pass' and r['centers']==2000 and r['candidates']==512000
                    assert r['binding']['result']==item['result'] and r['binding']['state']==item['cursor']
                    assert r['binding']['auditor_sha256']==file_sha256(ROOT/'stage8/audit.py')
                    complete.add(key)
            for key,row in list(active.items()):
                p=children.get(key);code=p.poll() if p else None
                if code is None and alive(row):continue
                active.pop(key)
                if key not in complete:raise RuntimeError(f'audit worker incomplete: {key}; {row["log"]}')
                if code not in (0,None):raise RuntimeError(f'audit worker failed: {key}; {row["log"]}')
            atomic_write_json(queuefile,dict(at_utc=utc_now(),total=300,complete=sorted(complete),active=active,
                status='complete' if len(complete)==300 and not active else 'running'))
            lease.update(len(complete),'parallel_full_audit',total=300,active=list(active))
            if time.monotonic()-last_progress>300:
                progress(f'8.3 原冻结审计器已通过{len(complete)}/300格，{len(active)}个CPU审计任务运行','DOING',str(queuefile),
                    '独立进程完成全候选审计、统计和交付复核；恢复入口stage8_finish_v1.py launch')
                last_progress=time.monotonic()
            if len(complete)==300 and not active:break
            pending=[r for r in tasks if r['task'] not in complete and r['task'] not in active]
            limit=min(read(CONTRACT)['max_audit_workers'],max(1,len(os.sched_getaffinity(0))//2))
            if pending and len(active)<limit:
                pref,_=capture('Stage8 CPU audit expansion; unchanged per-cell auditor',peak=0)
                last_check=time.monotonic()
                for item in pending[:min(4,limit-len(active))]:
                    key=item['task'];directory=Path(item['result']['logical_path']).parent
                    checked(item['result']);checked(item['cursor'])
                    cmd=command('audit-cell','--output',directory)
                    log=JOB/'workers'/f'{key}_{stamp()}.log';log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('w') as handle:
                        p=subprocess.Popen(cmd,cwd=CODEX,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    row=dict(pid=p.pid,process=pid_identity(p.pid),command=cmd,log=str(log),at_utc=utc_now(),
                        preflight=artifact(pref),result=item['result'],recovery_command=shlex.join(own_command('launch')))
                    active[key]=row;children[key]=p
                    append_jsonl(JOB/'worker_launches.jsonl',dict(task=key,**row))
                    atomic_write_json(queuefile,dict(at_utc=utc_now(),total=300,complete=sorted(complete),active=active,status='running'))
                    time.sleep(.5)
            elif time.monotonic()-last_check>120:
                capture('Stage8 CPU audit disk and host monitoring',peak=0);last_check=time.monotonic()
            time.sleep(3)
    except BaseException:
        stop_owned();raise


def run():
    verify_contract();pref,_=capture('Stage8 finalizer recovery start (CPU)',peak=0)
    def stop(*unused):raise InterruptedError('controller stop; preserve completed audit cursors')
    signal.signal(signal.SIGTERM,stop)
    with Lease(JOB/'controller','stage8_finish_v1',str(pref)) as lease:
        set_recovery(lease)
        if not (LOG/'audit/result.json').exists():audits(lease);phase('audit',lease)
        else:assert read(LOG/'audit/result.json')['status']=='pass'
        if not (LOG/'statistics/result.json').exists():phase('export',lease)
        else:assert read(LOG/'statistics/result.json')['status']=='pass'
        phase('verify',lease)
        result=ROOT/'manifests/stage8_delivery_verification_v1.json'
        assert read(result)['status']=='pass'
        atomic_write_json(JOB/'result.json',dict(status='pass',stage=8,at_utc=utc_now(),
            runtime_contract=artifact(CONTRACT),delivery=artifact(result),stage9_started=False))
        lease.update(300,'complete')


def freeze():
    gate();assert not CONTRACT.exists()
    for p in (JOB/'smoke_audit.json',JOB/'import_isolation_smoke.json'):
        assert read(p)['status']=='pass'
    r=dict(status='frozen',stage=8,at_utc=utc_now(),research_protocol_override=False,max_audit_workers=12,
        files=[artifact(SELF)],gates=[artifact(ROOT/'manifests/stage8_operations_v1.json'),
            artifact(JOB/'smoke_audit.json'),artifact(JOB/'import_isolation_smoke.json')],
        inventory=artifact(JOB/'attack_inventory.json'),
        change='Separate CPU phase processes prevent scripts/stage7.py import shadowing. Frozen scoring/auditing/statistics untouched.',
        recovery_command=shlex.join(own_command('launch')))
    atomic_write_json(CONTRACT,r,mode=0o444);print({'frozen':str(CONTRACT)},flush=True)


def launch():
    verify_contract()
    if (JOB/'result.json').exists():
        assert read(JOB/'result.json')['status']=='pass';print('Stage8 finalizer already complete');return
    with Lease(JOB/'launch_guard','stage8_finish_launch_guard') as guard:
        set_recovery(guard)
        for job in (JOB/'controller/job.json',LOG/'pipeline/job.json'):
            if job.exists():
                row=read(job);identity=row.get('process')
                if identity:assert not alive(dict(pid=identity['pid'],process=identity)),'another stage8 controller is alive'
        assert subprocess.run(['tmux','-S',str(SOCKET),'has-session','-t',SESSION],capture_output=True).returncode!=0
        pref,_=capture('Stage8 remaining audit and statistics tmux launch (CPU)',peak=0)
        directory=JOB/'launches'/stamp();directory.mkdir(parents=True,exist_ok=True)
        script=directory/'run.sh';log=directory/'output.log';cmd=own_command('run')
        script.write_text('#!/bin/bash\nset -u\ncd '+shlex.quote(str(CODEX))+'\nexport WDRO_TMUX_SESSION='+shlex.quote(SESSION)+'\n'+
            shlex.join(cmd)+' > '+shlex.quote(str(log))+' 2>&1\nfinish_exit=$?\nprintf \'{"exit_code":%s}\\n\' "$finish_exit" > '+
            shlex.quote(str(directory/'exit.json.partial'))+'\nmv '+shlex.quote(str(directory/'exit.json.partial'))+' '+
            shlex.quote(str(directory/'exit.json'))+'\nexit "$finish_exit"\n')
        subprocess.run(['tmux','-S',str(SOCKET),'new-session','-d','-s',SESSION,'-c',str(CODEX),'bash',str(script)],check=True)
        r=dict(at_utc=utc_now(),session=SESSION,socket=str(SOCKET),command=cmd,shell=artifact(script),
            log=dict(logical_path=str(log),resolved_path=str(log.resolve())),preflight=artifact(pref),
            runtime_contract=artifact(CONTRACT),recovery_command=shlex.join(own_command('launch')),
            monitor_command=shlex.join(own_command('status')))
        atomic_write_json(directory/'launch.json',r);atomic_write_json(JOB/'latest_launch.json',r)
        progress('8.3 全量攻击完成；剩余审计与统计已在独立进程tmux恢复','DOING',str(directory/'launch.json'),
            '通过全部300格审计、最终表图和交付验证；阶段9未启动')
        print(r,flush=True)


def status():
    for path in (JOB/'controller/heartbeat.json',LOG/'audit/heartbeat.json',LOG/'statistics/heartbeat.json',JOB/'result.json'):
        if path.exists():print(str(path),read(path),flush=True)
    if (JOB/'audit_queue.json').exists():
        q=read(JOB/'audit_queue.json');print({'audited_cells':len(q['complete']),'total':300,'active':{k:dict(pid=r['pid'],alive=alive(r)) for k,r in q['active'].items()}},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','launch','run','status'])
    globals()[p.parse_args().action]()

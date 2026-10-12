"""Hash-bound, resumable correction replay; each completed substep is published."""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.job_control import JobLease,utc_now
from src.rest import file_sha256
from run_precision_training_queue import source_gate,process_is_alive
from capture_host_runtime import gpu_sample,compute_processes,disk_snapshot
JOB=EXP/'logs/precision_replay';MASTER=Path('/remote-home/Guilin/DRO/Codex/01_WDRO-REST_项目进度.md')

def progress(message,evidence,done=False):
    now=utc_now();lines=MASTER.read_text().splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| 更新时间 |'):lines[i]=f'| 更新时间 | {now} |'
        elif line.startswith('| 实验状态 |'):lines[i]=f'| 实验状态 | precision_v2：{message} |'
    s='\n'.join(lines)+'\n'
    s=s.replace('## 8. 变更记录\n\n',f'## 8. 变更记录\n\n### {now} precision_v2\n\n- `{"DONE" if done else "DOING"}` {message}。证据 `{evidence}`。总阶段仍为6；恢复命令、PID、逐步骤命令和心跳见 `revisions/reference_precision_v2/Codex/experiments/logs/precision_replay/`。阶段7未启动。\n\n',1)
    p=MASTER.with_suffix('.md.partial');p.write_text(s);p.replace(MASTER)

def cpu_preflight(purpose):
    samples=[]
    for i in range(3):
        samples.append(dict(at_utc=utc_now(),gpus=gpu_sample(),compute_processes=compute_processes()))
        if i<2:time.sleep(1)
    d=[disk_snapshot(CODEX),disk_snapshot(EXP)]
    value=dict(at_utc=utc_now(),purpose=purpose,samples=samples,disk=d,preflight_passed=d[1]['free_gib']>=100,
               research_protocol_override=False,allocation='CPU step; no GPU process launched by this preflight')
    p=JOB/'preflights'/f'{purpose}_{time.time_ns()}.json';atomic_write_json(p,value)
    assert value['preflight_passed'],'disk below 100 GiB'
    return p

def read(p):return json.loads(Path(p).read_text())

def wait_queue(lease):
    q=EXP/'logs/stage2_queue/stage2_erm_formal_v1'
    while True:
        source_gate()
        if (q/'queue.json').exists() and read(q/'queue.json')['status']=='complete':return
        j=read(q/'job.json') if (q/'job.json').exists() else {}
        if not process_is_alive(j.get('pid',-1)):
            step(lease,'erm_queue_resume','run_precision_training_queue.py',
                 ['--stage','2','--max-simultaneous','8','--max-per-gpu','2','--launch-stagger-seconds','15'],[q/'queue.json'])
            assert read(q/'queue.json')['status']=='complete';return
        lease.update(update=0,status='waiting_erm_training',detail=dict(queue=str(q/'queue.json'),queue_pid=j['pid']))
        time.sleep(15)

def step(lease,name,script,args=(),outputs=()):
    source_gate();command=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]
    binding=dict(command=command,script_sha256=file_sha256(ROOT/'scripts'/script))
    receipt=JOB/'steps'/f'{name}.json';active=JOB/'steps'/f'{name}_active.json';log=JOB/f'{name}.log'
    if receipt.exists():
        old=read(receipt);assert old['binding']==binding,'step source changed: '+name
        assert all(file_sha256(Path(p))==h for p,h in old['output_hashes'].items()),'step output changed: '+name
        return
    # A previous controller may have exited while its child continued in tmux.
    if active.exists():
        old=read(active);pid=old['pid'];cmd=Path(f'/proc/{pid}/cmdline')
        if process_is_alive(pid) and cmd.exists() and command[1].encode() in cmd.read_bytes():
            while process_is_alive(pid) and cmd.exists() and command[1].encode() in cmd.read_bytes():
                lease.update(update=0,status=name,detail=dict(adopted_pid=pid,active=str(active)))
                time.sleep(15)
            # Re-enter the same idempotent command to validate completed outputs
            # or resume its existing hash-bound checkpoint/audit cursor.
    pre=cpu_preflight(name);env=dict(os.environ,WDRO_HOST_RUNTIME_MANIFEST=str(pre))
    if script=='audit_stage2_formal.py':command+=['--host-runtime-manifest',str(pre)]
    progress(name+' 执行中',str(log.relative_to(MASTER.parent)))
    with log.open('a') as out:
        child=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,env=env,cwd=CODEX)
        record=dict(at_utc=utc_now(),name=name,pid=child.pid,command=command,exact_command=shlex.join(command),
            log=str(log),log_resolved=str(log.resolve()),preflight=str(pre),binding=binding)
        atomic_write_json(active,record);append_jsonl(JOB/'launches.jsonl',record)
        try:
            while child.poll() is None:
                lease.update(update=0,status=name,detail=dict(child_pid=child.pid,log=str(log),active=str(active)))
                time.sleep(10)
            if child.returncode != 0:
                failure = JOB/'failures'/f'{name}_{time.time_ns()}.json'
                atomic_write_json(failure,dict(at_utc=utc_now(),status='failed',name=name,
                    exit_code=child.returncode,execution=record,recovery='resume precision_replay after resolving this gate'))
                progress(name+f' 失败（exit {child.returncode}），依赖步骤已停止',str(failure))
                raise RuntimeError(f'{name} failed ({child.returncode}); inspect {log}')
        finally:
            if child.poll() is None:child.terminate();child.wait()
    hashes={str(p):file_sha256(p) for p in outputs}
    atomic_write_json(receipt,dict(status='pass',at_utc=utc_now(),binding=binding,output_hashes=hashes,execution=record),mode=0o444)
    progress(name+' 已通过',str(receipt.relative_to(MASTER.parent)),True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();JOB.mkdir(parents=True,exist_ok=True)
    source_gate()
    if a.check_only:
        pre=cpu_preflight('controller_foreground_smoke')
        atomic_write_json(JOB/'foreground_smoke.json',dict(status='pass',preflight=str(pre),source_gate=True,script_sha256=file_sha256(Path(__file__))))
        print(json.dumps(dict(status='pass',preflight=str(pre))));return
    exact=shlex.join(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__))])
    recovery='bash /remote-home/Guilin/DRO/Codex/wdro_rest_min/scripts/stage3_env.sh '+sys.executable+' /remote-home/Guilin/DRO/Codex/wdro_rest_min/scripts/precision_job.py --name replay -- '+exact
    with JobLease(JOB,run_id='precision_replay',exact_command=exact,resume_command=recovery,
        monitor_command=f'cat {JOB}/heartbeat.json',expected_updates=1,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        wait_queue(lease)
        step(lease,'erm_full_audit','audit_stage2_formal.py',outputs=[EXP/'logs/stage2/stage2_formal_audit.json'])
        step(lease,'erm_export','export_stage2_summary.py',outputs=[ROOT/'manifests/stage2_completion_20260905.json'])
        assert read(ROOT/'manifests/stage2_completion_20260905.json')['status']=='pass'
        step(lease,'baseline_smoke','run_stage3_smoke.py',['--tag','final'],[EXP/'logs/stage3_smoke/audit_final.json'])
        for method in ('mixed_erm','group_dro','hard_mining'):
            rid=f'smoke_final_{method}_d30_continuous'
            step(lease,'baseline_smoke_audit_'+method,'audit_stage3.py',['--workers','1','--smoke-run',rid],[EXP/'logs/stage3_audit/runs'/f'{rid}.json'])
        step(lease,'baseline_regression','run_precision_unit_tests.py',['--output',EXP/'logs/stage3_unit_tests_final.log'],[EXP/'logs/stage3_unit_tests_final.log'])
        step(lease,'baseline_contract','prepare_precision_stage3.py',outputs=[ROOT/'manifests/stage3_launch_contract_v1.json'])
        step(lease,'baseline_formal_audit_export','stage3_pipeline.py',outputs=[ROOT/'manifests/stage3_completion_v1.json'])
        step(lease,'baseline_reference_budget','stage3_hard_mining_budget_precheck_all15.py',outputs=[EXP/'logs/stage3_hard_mining_budget_precheck_20260905_all15/result.json'])
        step(lease,'baseline_delivery','verify_stage3_delivery.py',outputs=[ROOT/'manifests/stage3_delivery_verification_v1.json'])
        lease.finish(update=1,status='complete',detail=dict(completed_replay_stages=[2,3],next='corrected WDRO diagnostic, then validation and stage6'))
if __name__=='__main__':main()

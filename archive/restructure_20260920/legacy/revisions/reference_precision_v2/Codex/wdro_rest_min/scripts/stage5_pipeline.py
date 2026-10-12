#!/usr/bin/env python3
"""Resumable stage-5 queue, fresh shared-GPU scheduling, full audits and freeze."""
from __future__ import annotations
import json,os,shlex,signal,subprocess,sys,time,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.job_control import JobLease,utc_now,process_is_alive
from src.rest import file_sha256
from stage5.configuration import initial_tasks,select_rho,remaining_tasks
from capture_host_runtime import gpu_sample,compute_processes,disk_snapshot
JOB=EXP/'logs/stage5_pipeline'

def gate(smoke=True):
    contract=json.loads((ROOT/'manifests/stage5_training_contract_v1.json').read_text())
    for path,digest in contract['files'].items():assert file_sha256(CODEX/path)==digest,'frozen file changed: '+path
    if smoke:
        assert json.loads((EXP/'logs/stage5_smoke/audit.json').read_text())['status']=='pass'
        audit=json.loads((EXP/'logs/stage5_audit/smoke_gate.json').read_text());assert audit['status']=='pass'
        for row in audit['independent_scores']:
            path=Path(row['path']);assert file_sha256(path)==row['sha256']
            result=json.loads(path.read_text());assert result['status']=='pass' and result['geosteiner_workers']==8
        ops=ROOT/'manifests/stage5_operations_contract_v1.json'
        if ops.exists():
            for path,digest in json.loads(ops.read_text())['files'].items():assert file_sha256(CODEX/path)==digest,'operation file changed: '+path

def preflight(purpose):
    profile=json.loads((EXP/'logs/stage5_smoke/memory_profile.json').read_text())
    command=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose',purpose,
        '--measured-peak-mib',str(profile['maximum_reserved_mib']+1024),'--safety-margin-mib','2048']
    p=subprocess.run(command,capture_output=True,text=True)
    append_jsonl(JOB/'preflights.jsonl',{'at_utc':utc_now(),'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    return json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode==0 else None

def observe():return {'at_utc':utc_now(),'time':time.time(),'gpus':gpu_sample(),'compute_processes':compute_processes(),'disk':[disk_snapshot(CODEX),disk_snapshot(EXP)]}

def progress(message,action):
    p=CODEX/'01_WDRO-REST_项目进度.md';s=p.read_text();now=utc_now()
    for key,value in {'更新时间':now,'实验状态':message,'下一动作':action}.items():
        s=re.sub(r'\| '+re.escape(key)+r' \| .*? \|',f'| {key} | {value} |',s,count=1)
    s=s.replace('## 8. 变更记录\n\n',f'## 8. 变更记录\n\n### {now}\n\n- {message}。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：{action}。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。\n\n')
    temp=p.with_suffix('.md.partial');temp.write_text(s);temp.replace(p)
    from precision_replay_pipeline import progress as publish
    publish(message,str(JOB/'heartbeat.json'))

def substep(number,status,message):
    path=CODEX/'01_WDRO-REST_项目进度.md';value=path.read_text()
    value=re.sub(r'- `(DONE|DOING|TODO)` '+re.escape(number)+r'：.*?\n',f'- `{status}` {number}：{message}。\n',value,count=1)
    temp=path.with_suffix('.md.partial');temp.write_text(value);temp.replace(path)
    from precision_replay_pipeline import progress as publish
    publish('WDRO validation '+number+' '+message,str(JOB/'heartbeat.json'),status=='DONE')

def result_for(task):
    p=EXP/'logs/stage5'/task['run_id']/'result.json'
    return json.loads(p.read_text()) if p.exists() else None

def choose_gpu(rows,active,history,blocked,need,now):
    eligible=[]
    for row in rows:
        gpu=row['index'];count=sum(a['gpu']==gpu for a in active.values())
        minfree=min([row['minimum_free_memory_mib']]+[g['memory_free_mib'] for h in history for g in h['gpus'] if g['index']==gpu])
        if row['eligible'] and minfree>need and count<2 and now>blocked.get(gpu,0):eligible.append((count,-minfree,gpu))
    return min(eligible)[2] if eligible else None

def run_queue(tasks,lease,phase):
    for task in tasks:
        assert file_sha256(Path(task['config']))==task['config_sha256'],'run config hash changed'
    active={};blocked={};history=[];last_progress=0;attempts={};monitor_times={}
    # Adopt surviving project children after a controller restart.
    for t in tasks:
        job=EXP/'logs/jobs'/t['run_id']/'job.json'
        if job.exists():
            j=json.loads(job.read_text());pid=j['pid']
            cmdline=Path('/proc')/str(pid)/'cmdline'
            if process_is_alive(pid) and cmdline.exists() and t['run_id'] in cmdline.read_text():
                active[t['run_id']]={'pid':pid,'gpu':j['physical_gpu_index'],'adopted':True,'saturated':[],'task':t,'log':None,'stopping':False}
    try:
        while True:
            gate();snapshot=observe();append_jsonl(JOB/'resource_monitor.jsonl',snapshot)
            history=[h for h in history if h['time']>time.time()-600]+[snapshot]
            completed=[]
            for t in tasks:
                r=result_for(t)
                if r and r['status']=='complete' and r['completed_updates']==2000:completed.append(t['run_id'])
            for rid,a in list(active.items()):
                alive=a['proc'].poll() is None if 'proc' in a else process_is_alive(a['pid'])
                if not alive:
                    code=a['proc'].returncode if 'proc' in a else None
                    append_jsonl(JOB/'segments.jsonl',{'at_utc':utc_now(),'run_id':rid,'gpu':a['gpu'],'pid':a['pid'],'exit_code':code,'log':a['log']})
                    if code and a['log']:
                        tail=Path(a['log']).read_text()[-40000:]
                        if 'out of memory' not in tail.lower():raise RuntimeError(f'training error in {a["log"]}')
                        blocked[a['gpu']]=time.time()+600
                    del active[rid];continue
                row=next(g for g in snapshot['gpus'] if g['index']==a['gpu'])
                a['saturated']=(a['saturated']+[row['utilization_gpu_percent']>=95])[-6:]
                reason=None
                if row['memory_free_mib']<2048:reason='free VRAM below measured-peak safety margin'
                if len(a['saturated'])==6 and all(a['saturated']):reason='sustained >=95 percent GPU utilization across six observations'
                if snapshot['disk'][1]['free_gib']<100:reason='artifact disk below 100 GiB'
                hb=EXP/'logs/jobs'/rid/'heartbeat.json'
                if hb.exists():
                    h=json.loads(hb.read_text());u=h['completed_updates'];a['update']=u
                    samples=monitor_times.setdefault(rid,[])
                    if not samples or u>samples[-1][0]:samples.append((u,time.time()))
                    if len(samples)>=5 and sum(x['gpu']==a['gpu'] for x in active.values())>1:
                        durations=[(y[1]-x[1])/max(1,y[0]-x[0]) for x,y in zip(samples,samples[1:])]
                        baseline=sorted(durations[:-2])[len(durations[:-2])//2]
                        if min(durations[-2:])>max(3*baseline,5):reason='per-update throughput dropped >3x across two intervals'
                if reason and not a['stopping']:
                    # Stop only one tenant of this project per GPU per cycle.
                    if not any(x['gpu']==a['gpu'] and x['stopping'] for x in active.values()):
                        os.kill(a['pid'],signal.SIGTERM);a['stopping']=True;blocked[a['gpu']]=time.time()+600
                        append_jsonl(JOB/'recoveries.jsonl',{'at_utc':utc_now(),'run_id':rid,'pid':a['pid'],'gpu':a['gpu'],'reason':reason})
            state={'phase':phase,'at_utc':utc_now(),'tasks':tasks,'complete':completed,
                'active':[{k:v for k,v in a.items() if k not in ('proc','saturated')} for a in active.values()],
                'blocked_until':blocked,'stage6_started':False}
            atomic_write_json(JOB/'queue.json',state)
            total_updates=len(completed)*2000+sum(a.get('update',0) for rid,a in active.items() if rid not in completed)
            lease.update(update=total_updates,status=phase,detail={'complete':len(completed),'total':len(tasks),'active':state['active'],'queue':str(JOB/'queue.json')})
            if len(completed)==len(tasks) and not active:return
            if time.time()-last_progress>600:
                progress(f'阶段 5 {phase}：{len(completed)}/{len(tasks)} run 训练完成，累计 {total_updates} updates', '继续当前阶段队列；完整审计后再选择/冻结')
                last_progress=time.time()
            pending=[t for t in tasks if t['run_id'] not in completed and t['run_id'] not in active]
            if pending and len(active)<6:
                summary=preflight('stage5_'+phase+'_start_or_resume')
                if summary:
                    snap=json.loads(Path(summary['manifest']).read_text());profile=json.loads((EXP/'logs/stage5_smoke/memory_profile.json').read_text())
                    need=profile['maximum_reserved_mib']+1024+2048
                    gpu=choose_gpu(snap['gpu_decisions'],active,history,blocked,need,time.time())
                    if gpu is not None:
                        t=pending[0];rid=t['run_id'];attempts[rid]=attempts.get(rid,0)+1
                        if attempts[rid]>10:raise RuntimeError('repeated recovery failures; inspect '+rid)
                        cmd=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/train_stage5.py'),
                            '--degree','30','--seed','101','--scope','ablation','--wdro-config',t['config'],'--rho-mode',t['rho_mode'],
                            '--run-id',rid,'--physical-gpu-index',str(gpu),'--host-runtime-manifest',summary['manifest'],'--geosteiner-workers',
                            '8' if json.loads(Path(t['config']).read_text())['search_overrides']['searcher'] in ('random_64','cem_64') else '3']
                        if (EXP/'checkpoints/stage5'/rid/'latest.pt').exists():cmd+=['--resume']
                        log=JOB/(rid+'_'+str(int(time.time()))+'.log')
                        with log.open('a') as out:proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
                        a={'pid':proc.pid,'gpu':gpu,'task':t,'log':str(log),'stopping':False,'saturated':[],'proc':proc,'update':0}
                        active[rid]=a
                        record={'at_utc':utc_now(),'run_id':rid,'pid':proc.pid,'gpu':gpu,'command':cmd,'exact_command':shlex.join(cmd),
                            'log':str(log),'log_resolved':str(log.resolve()),'preflight':summary,'attempt':attempts[rid]}
                        append_jsonl(JOB/'launches.jsonl',record)
                        append_jsonl(JOB/'postlaunch.jsonl',dict(observe(),run_id=rid,pid=proc.pid))
            time.sleep(15)
    except BaseException:
        for a in active.values():
            if process_is_alive(a['pid']):os.kill(a['pid'],signal.SIGTERM)
        raise

def audited(tasks,lease,phase):
    pending=list(tasks);active={}
    for t in list(pending):
        rid=t['run_id'];jp=EXP/'logs/stage5_audit/jobs'/rid/'job.json'
        if jp.exists():
            j=json.loads(jp.read_text());pid=j['pid'];cmd=Path('/proc')/str(pid)/'cmdline'
            if process_is_alive(pid) and cmd.exists() and rid in cmd.read_text():
                active[rid]={'pid':pid,'proc':None,'log':str(JOB/(rid+'_audit.log'))};pending.remove(t)
    try:
        while pending or active:
            gate()
            while pending and len(active)<2:
                t=pending.pop(0);rid=t['run_id'];report=EXP/'logs/stage5_audit/runs'/f'{rid}.json'
                if report.exists() and json.loads(report.read_text())['status']=='pass':
                    a=json.loads(report.read_text());assert a['binding']['result_sha256']==file_sha256(EXP/'logs/stage5'/rid/'result.json');continue
                preflight('stage5_cpu_audit_'+rid)
                cmd=['bash',str(ROOT/'scripts/stage3_env.sh'),'nice','-n','5',sys.executable,str(ROOT/'scripts/audit_stage5.py'),'--run-id',rid,'--stability']
                log=JOB/(rid+'_audit.log')
                with log.open('a') as out:proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
                active[rid]={'proc':proc,'pid':proc.pid,'log':str(log)}
                append_jsonl(JOB/'launches.jsonl',{'at_utc':utc_now(),'task':'audit','run_id':rid,'pid':proc.pid,'command':cmd,'log':str(log),'log_resolved':str(log.resolve())})
            for rid,a in list(active.items()):
                alive=a['proc'].poll() is None if a['proc'] else process_is_alive(a['pid'])
                if not alive:
                    report=EXP/'logs/stage5_audit/runs'/f'{rid}.json'
                    if not report.exists() or json.loads(report.read_text())['status']!='pass':raise RuntimeError('full audit failed/interrupted: '+a['log'])
                    del active[rid]
            lease.update(update=0,status=phase,detail={'pending':len(pending),'active':{rid:a['pid'] for rid,a in active.items()}})
            time.sleep(15)
    except BaseException:
        for a in active.values():
            if process_is_alive(a['pid']):os.kill(a['pid'],signal.SIGTERM)
        raise

def main():
    JOB.mkdir(parents=True,exist_ok=True);gate()
    command=shlex.join(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__))])
    with JobLease(JOB,run_id='stage5_pipeline',exact_command=command,resume_command=f'bash {ROOT}/scripts/stage3_env.sh {sys.executable} {ROOT}/scripts/launch_stage5.py',
        monitor_command=f'cat {JOB}/heartbeat.json',expected_updates=18000,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        substep('5.1','DONE','80单测、三种搜索精确恢复、八变体完整smoke审计与64-call真实重算/8-worker等价通过；证据stage5_smoke/audit.json与stage5_audit/smoke_gate.json')
        substep('5.2','DOING','rho三配置的2000-update训练及完整审计；见stage5_pipeline/queue.json')
        initial=initial_tasks();run_queue(initial,lease,'5.2_rho');audited(initial,lease,'5.2_rho_audit')
        selection=select_rho(initial);path=JOB/'rho_selection.json'
        if path.exists():assert json.loads(path.read_text())==selection
        else:atomic_write_json(path,selection,mode=0o444)
        remaining=remaining_tasks(selection);atomic_write_json(JOB/'all_tasks.json',{'tasks':initial+remaining,'selection':str(path)})
        substep('5.2','DONE','三个rho训练/完整审计通过，validation-only选择见stage5_pipeline/rho_selection.json')
        substep('5.3','DOING','selected-rho六个最小消融，完整记录见stage5_pipeline/all_tasks.json')
        progress('阶段 5.2 三个 rho 的 2000-update 训练及完整审计通过；validation 选中 '+selection['selected']['task']['rho_mode'], '5.3 执行 selected-rho 的六个最小消融，保持主方法约束')
        run_queue(remaining,lease,'5.3_ablations');audited(remaining,lease,'5.3_ablation_audit')
        substep('5.3','DONE','九个run训练和完整输入/cost/对偶审计通过，见stage5_audit/runs')
        substep('5.4','DOING','独立重算best validation、表图、只读主配置与交付复核')
        progress('阶段 5.3 九个 run 训练和完整审计通过', '5.4 独立重算 best validation、生成消融表/预算图并冻结主配置')
        for t in initial+remaining:
            rid=t['run_id'];output=EXP/'logs/stage5_best_validation'/f'{rid}.json'
            if output.exists() and json.loads(output.read_text())['status']=='pass':continue
            summary=preflight('stage5_best_validation_'+rid)
            while summary is None:time.sleep(30);summary=preflight('stage5_best_validation_'+rid)
            snap=json.loads(Path(summary['manifest']).read_text());gpu=max([r for r in snap['gpu_decisions'] if r['eligible']],key=lambda r:r['minimum_free_memory_mib'])['index']
            cmd=[sys.executable,str(ROOT/'scripts/verify_stage5_best_validation.py'),'--run-id',rid,'--gpu',str(gpu),'--preflight',summary['manifest']]
            with (JOB/'best_validation.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/export_stage5.py')],check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/verify_stage5_delivery.py')],check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/finalize_stage5.py')],check=True)
        lease.finish(update=18000,status='complete',detail={'stage':5,'stage6_started':False})
if __name__=='__main__':main()

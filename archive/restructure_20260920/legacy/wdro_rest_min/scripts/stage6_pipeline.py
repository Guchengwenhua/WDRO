#!/usr/bin/env python3
"""Resumable stage-5 queue, fresh shared-GPU scheduling, full audits and freeze."""
from __future__ import annotations
import json,os,shlex,signal,subprocess,sys,time,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json,append_jsonl
from src.job_control import JobLease,utc_now,process_is_alive
from src.rest import file_sha256

from capture_host_runtime import gpu_sample,compute_processes,disk_snapshot
JOB=EXP/'logs/stage6_pipeline'
def process_is_alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] not in ('Z','X')
    except FileNotFoundError:return False


def gate(smoke=True):
    for name in ['stage6_training_contract_v1','stage6_operations_contract_v1']:
        path=ROOT/'manifests'/f'{name}.json'
        if not path.exists():
            assert not smoke and name=='stage6_operations_contract_v1';continue
        for p,h in json.loads(path.read_text())['files'].items():assert file_sha256(CODEX/p)==h,'frozen file changed: '+p
    if smoke:
        for p in [EXP/'logs/stage6_smoke/audit.json',EXP/'logs/stage6_evaluation_smoke/audit.json']:
            assert json.loads(p.read_text())['status']=='pass',str(p)

def measured_peak(degree=None,inference=False):
    if inference:return int(json.loads((EXP/'logs/stage6_evaluation_smoke/audit.json').read_text())['maximum_reserved_mib'])+1
    profile=json.loads((EXP/'logs/stage6_smoke/memory_profile.json').read_text())
    if degree is None:return profile['maximum_reserved_mib']
    return max(v['reserved_mib'] for v in profile['runs'].values() if json.loads(Path(v['result']).read_text())['degree']==degree)

def cleanup_dead_lock(root):
    path=Path(root)/'run.lock'
    if path.exists():
        pid=json.loads(path.read_text())['pid']
        if not process_is_alive(pid):path.rename(path.with_name('run.lock.stale.'+str(time.time_ns())))

def preflight(purpose,degree=None,inference=False):
    profile=json.loads((EXP/'logs/stage6_smoke/memory_profile.json').read_text())
    command=[sys.executable,str(ROOT/'scripts/capture_host_runtime.py'),'--purpose',purpose,
        '--measured-peak-mib',str(measured_peak(degree,inference)+1024),'--safety-margin-mib','2048']
    p=subprocess.run(command,capture_output=True,text=True)
    append_jsonl(JOB/'preflights.jsonl',{'at_utc':utc_now(),'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    return json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode==0 else None

def observe():return {'at_utc':utc_now(),'time':time.time(),'gpus':gpu_sample(),'compute_processes':compute_processes(),'disk':[disk_snapshot(CODEX),disk_snapshot(EXP)]}

def progress(message,action):
    p=CODEX/'01_WDRO-REST_项目进度.md';s=p.read_text();now=utc_now()
    for key,value in {'更新时间':now,'实验状态':message,'下一动作':action}.items():
        s=re.sub(r'\| '+re.escape(key)+r' \| .*? \|',f'| {key} | {value} |',s,count=1)
    s=s.replace('## 8. 变更记录\n\n',f'## 8. 变更记录\n\n### {now}\n\n- {message}。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：{action}。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。\n\n')
    temp=p.with_suffix('.md.partial');temp.write_text(s);temp.replace(p)

def substep(number,status,message):
    path=CODEX/'01_WDRO-REST_项目进度.md';value=path.read_text()
    value=re.sub(r'- `(DONE|DOING|TODO)` '+re.escape(number)+r'：.*?\n',f'- `{status}` {number}：{message}。\n',value,count=1)
    temp=path.with_suffix('.md.partial');temp.write_text(value);temp.replace(path)

def result_for(task):
    p=Path(task['result'])
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
                'blocked_until':blocked,'stage7_started':False}
            atomic_write_json(JOB/'queue.json',state)
            total_updates=len(completed)*2000+sum(a.get('update',0) for rid,a in active.items() if rid not in completed)
            lease.update(update=total_updates,status=phase,detail={'complete':len(completed),'total':len(tasks),'active':state['active'],'queue':str(JOB/'queue.json')})
            if len(completed)==len(tasks) and not active:return
            if time.time()-last_progress>600:
                progress(f'阶段 6 {phase}：{len(completed)}/{len(tasks)} run 训练完成，累计 {total_updates} updates', '继续冻结配置训练与完整审计；75模型锁定后共享test，不调参')
                last_progress=time.time()
            pending=[t for t in tasks if t['run_id'] not in completed and t['run_id'] not in active]
            if pending and len(active)<8:
                summary=preflight('stage6_'+phase+'_start_or_resume',degree=pending[0]['degree'])
                if summary:
                    snap=json.loads(Path(summary['manifest']).read_text());profile=json.loads((EXP/'logs/stage6_smoke/memory_profile.json').read_text())
                    need=measured_peak(pending[0]['degree'])+1024+2048
                    gpu=choose_gpu(snap['gpu_decisions'],active,history,blocked,need,time.time())
                    if gpu is not None:
                        t=pending[0];rid=t['run_id'];attempts[rid]=attempts.get(rid,0)+1
                        if attempts[rid]>10:raise RuntimeError('repeated recovery failures; inspect '+rid)
                        cmd=['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(ROOT/'scripts/train_stage6.py'),
                            '--degree',str(t['degree']),'--seed',str(t['seed']),'--scope','formal','--wdro-config',t['config'],'--rho-mode',t['rho_mode'],
                            '--run-id',rid,'--physical-gpu-index',str(gpu),'--host-runtime-manifest',summary['manifest'],'--geosteiner-workers',
                            '3']
                        if (EXP/'checkpoints/stage6'/rid/'latest.pt').exists():cmd+=['--resume']
                        log=JOB/(rid+'_'+str(int(time.time()))+'.log')
                        cleanup_dead_lock(EXP/'logs/jobs'/rid)
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
        rid=t['run_id'];jp=EXP/'logs/stage6_audit/jobs'/rid/'job.json'
        if jp.exists():
            j=json.loads(jp.read_text());pid=j['pid'];cmd=Path('/proc')/str(pid)/'cmdline'
            if process_is_alive(pid) and cmd.exists() and rid in cmd.read_text():
                active[rid]={'pid':pid,'proc':None,'log':str(JOB/(rid+'_audit.log'))};pending.remove(t)
    try:
        while pending or active:
            gate()
            while pending and len(active)<8:
                t=pending.pop(0);rid=t['run_id'];report=EXP/'logs/stage6_audit/runs'/f'{rid}.json'
                if report.exists() and json.loads(report.read_text())['status']=='pass':
                    a=json.loads(report.read_text());assert a['binding']['result_sha256']==file_sha256(EXP/'logs/stage6'/rid/'result.json');continue
                preflight('stage6_cpu_audit_'+rid)
                cmd=['bash',str(ROOT/'scripts/stage3_env.sh'),'nice','-n','5',sys.executable,str(ROOT/'scripts/audit_stage6.py'),'--run-id',rid,'--stability']
                log=JOB/(rid+'_audit.log')
                cleanup_dead_lock(EXP/'logs/stage6_audit/jobs'/rid)
                with log.open('a') as out:proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
                active[rid]={'proc':proc,'pid':proc.pid,'log':str(log)}
                append_jsonl(JOB/'launches.jsonl',{'at_utc':utc_now(),'task':'audit','run_id':rid,'pid':proc.pid,'command':cmd,'log':str(log),'log_resolved':str(log.resolve())})
            for rid,a in list(active.items()):
                alive=a['proc'].poll() is None if a['proc'] else process_is_alive(a['pid'])
                if not alive:
                    report=EXP/'logs/stage6_audit/runs'/f'{rid}.json'
                    if not report.exists() or json.loads(report.read_text())['status']!='pass':raise RuntimeError('full audit failed/interrupted: '+a['log'])
                    del active[rid]
            lease.update(update=0,status=phase,detail={'pending':len(pending),'active':{rid:a['pid'] for rid,a in active.items()}})
            time.sleep(15)
    except BaseException:
        for a in active.values():
            if process_is_alive(a['pid']):os.kill(a['pid'],signal.SIGTERM)
        raise

def evaluate(lease):
    pending=[50,40,30,20,10];active={};blocked={};history=[]
    for degree in list(pending):
        path=EXP/'logs/stage6_evaluation_jobs'/f'd{degree}'/'job.json'
        if path.exists():
            old=json.loads(path.read_text());pid=old['pid']
            if process_is_alive(pid) and 'evaluate_stage6.py' in Path(f'/proc/{pid}/cmdline').read_text():
                active[degree]=dict(pid=pid,proc=None,gpu=old['physical_gpu_index'],stopping=False,saturated=[],log='adopted')
                pending.remove(degree)
    try:
        while pending or active:
            gate();snapshot=observe();append_jsonl(JOB/'evaluation_resources.jsonl',snapshot)
            history=[h for h in history if h['time']>time.time()-600]+[snapshot]
            for degree,a in list(active.items()):
                alive=a['proc'].poll() is None if a['proc'] else process_is_alive(a['pid'])
                if not alive:
                    rp=EXP/'logs/stage6_evaluation_jobs'/f'd{degree}'/'result.json'
                    good=rp.exists() and json.loads(rp.read_text())['status']=='complete'
                    if not good:
                        log=Path(a['log']);tail=log.read_text()[-30000:] if log.exists() else ''
                        if not a['stopping'] and 'out of memory' not in tail.lower():raise RuntimeError('evaluation failed: '+str(log))
                        pending.insert(0,degree);blocked[a['gpu']]=time.time()+600
                    del active[degree];continue
                gpu=next(g for g in snapshot['gpus'] if g['index']==a['gpu'])
                a['saturated']=(a['saturated']+[gpu['utilization_gpu_percent']>=95])[-6:]
                reason= gpu['memory_free_mib']<2048 or snapshot['disk'][1]['free_gib']<100 or (len(a['saturated'])==6 and all(a['saturated']))
                if reason and not a['stopping']:
                    # Inference has no optimizer; every committed batch is a recovery point.
                    os.kill(a['pid'],signal.SIGTERM);a['stopping']=True
                    append_jsonl(JOB/'recoveries.jsonl',dict(at_utc=utc_now(),task='evaluation',degree=degree,gpu=a['gpu'],pid=a['pid'],reason='GPU/disk safety gate; resume atomic batches'))
            if pending:
                degree=pending[0];rp=EXP/'logs/stage6_evaluation_jobs'/f'd{degree}'/'result.json'
                if rp.exists() and json.loads(rp.read_text())['status']=='complete':pending.pop(0);continue
                summary=preflight('stage6_test_start_or_resume',inference=True)
                if summary:
                    host=json.loads(Path(summary['manifest']).read_text())
                    rows=[g for g in host['gpu_decisions'] if g['eligible'] and all(a['gpu']!=g['index'] for a in active.values()) and time.time()>blocked.get(g['index'],0)]
                    if rows:
                        gpu=max(rows,key=lambda r:r['minimum_free_memory_mib'])['index']
                        cmd=[sys.executable,str(ROOT/'scripts/evaluate_stage6.py'),'--degree',str(degree),'--gpu',str(gpu),'--preflight',summary['manifest']]
                        log=JOB/f'eval_d{degree}_{int(time.time())}.log'
                        cleanup_dead_lock(EXP/'logs/stage6_evaluation_jobs'/f'd{degree}')
                        with log.open('a') as out:proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
                        active[degree]=dict(proc=proc,pid=proc.pid,gpu=gpu,stopping=False,saturated=[],log=str(log));pending.pop(0)
                        append_jsonl(JOB/'launches.jsonl',dict(at_utc=utc_now(),task='test',degree=degree,pid=proc.pid,gpu=gpu,command=cmd,exact_command=shlex.join(cmd),log=str(log),log_resolved=str(log.resolve()),preflight=summary))
                        append_jsonl(JOB/'postlaunch.jsonl',dict(observe(),degree=degree,pid=proc.pid))
            complete=len(list((EXP/'logs/stage6_test').glob('*/*/result.json')))
            lease.update(update=complete,status='6.3_test',detail={'complete_cells':complete,'total_cells':300,'active':{d:a['pid'] for d,a in active.items()}})
            time.sleep(15)
    except BaseException:
        for a in active.values():
            if process_is_alive(a['pid']):os.kill(a['pid'],signal.SIGTERM)
        raise

def run_cpu(script,lease,phase):
    preflight('stage6_'+phase+'_cpu');cmd=[sys.executable,str(ROOT/'scripts'/script)]
    assert disk_snapshot(EXP)['free_gib']>=100,'artifact disk below 100 GiB'
    if script=='export_stage6.py':cleanup_dead_lock(EXP/'logs/stage6_statistics')
    if script=='verify_stage6_delivery.py':cleanup_dead_lock(EXP/'logs/stage6_delivery')
    child_root={'export_stage6.py':EXP/'logs/stage6_statistics','verify_stage6_delivery.py':EXP/'logs/stage6_delivery'}.get(script)
    if child_root and (child_root/'job.json').exists():
        old=json.loads((child_root/'job.json').read_text());pid=old['pid']
        if process_is_alive(pid) and script in Path(f'/proc/{pid}/cmdline').read_text():
            while process_is_alive(pid):
                lease.update(update=0,status=phase,detail={'adopted_pid':pid,'job':str(child_root)});time.sleep(15)
            assert json.loads((child_root/'job.json').read_text())['status']=='complete','adopted CPU task did not complete'
            return
    log=JOB/(script+'.log')
    with log.open('a') as out:
        proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
        append_jsonl(JOB/'launches.jsonl',dict(at_utc=utc_now(),task=phase,pid=proc.pid,command=cmd,exact_command=shlex.join(cmd),log=str(log),log_resolved=str(log.resolve())))
        while proc.poll() is None:
            lease.update(update=0,status=phase,detail={'pid':proc.pid,'log':str(log)});time.sleep(15)
        if proc.returncode:raise RuntimeError('failed '+script+'; see '+str(log))

def main():
    JOB.mkdir(parents=True,exist_ok=True);gate()
    command=shlex.join(['bash',str(ROOT/'scripts/stage3_env.sh'),sys.executable,str(Path(__file__))])
    with JobLease(JOB,run_id='stage6_pipeline',exact_command=command,
        resume_command=f'bash {ROOT}/scripts/stage3_env.sh {sys.executable} {ROOT}/scripts/launch_stage6.py',
        monitor_command=f'cat {JOB}/heartbeat.json',expected_updates=28000,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        plan=json.loads((ROOT/'configs/stage6_plan_v1.json').read_text())
        tasks=sorted([t for t in plan['tasks'] if not t['reused_stage5']],key=lambda t:(-t['degree'],t['seed']))
        substep('6.1','DONE','原80单测与阶段6测试、跨刷新精确恢复/阶段5数值等价、五degree完整smoke审计、validation评价断点恢复通过；证据stage6_smoke/audit.json、stage6_evaluation_smoke/audit.json')
        substep('6.2','DOING','14个新增WDRO训练及完整审计；复用61已审计模型。见stage6_pipeline/queue.json')
        run_queue(tasks,lease,'6.2_training');audited(tasks,lease,'6.2_full_audit')
        run_cpu('freeze_stage6_models.py',lease,'6.2_model_freeze')
        substep('6.2','DONE','75模型完整审计与validation-only best索引冻结；证据manifests/stage6_models_frozen_v1.json')
        substep('6.3','DOING','冻结75模型的300个合成test格逐batch恢复评价')
        progress('阶段6.2完整通过，75模型已冻结', '6.3统一合成test，不再选模或调参')
        evaluate(lease)
        substep('6.3','DONE','300格/300万逐net评价已完成；等待6.4独立完整性与表格复核')
        substep('6.4','DOING','可恢复paired net bootstrap、主表/degree附表及完整交付验证')
        progress('阶段6.3评价完成，进入统计与独立验收', '保留全部结果，完成paired CI和交付报告')
        run_cpu('export_stage6.py',lease,'6.4_statistics')
        run_cpu('verify_stage6_delivery.py',lease,'6.4_delivery')
        run_cpu('finalize_stage6.py',lease,'6.4_report')
        lease.finish(update=28000,status='complete',detail={'stage':6,'stage7_started':False})
if __name__=='__main__':main()

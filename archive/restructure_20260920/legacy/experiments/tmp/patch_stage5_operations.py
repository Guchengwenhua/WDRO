from pathlib import Path
p=Path('/remote-home/Guilin/DRO/Codex/wdro_rest_min/scripts/stage5_pipeline.py');s=p.read_text()
s=s.replace("def run_queue(tasks,lease,phase):\n    active", "def run_queue(tasks,lease,phase):\n    for task in tasks:\n        assert file_sha256(Path(task['config']))==task['config_sha256'],'run config hash changed'\n    active")
a=s.index('def audited(tasks,lease,phase):');b=s.index('\ndef main():',a)
s=s[:a]+'''def audited(tasks,lease,phase):
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
''' + s[b:]
p.write_text(s)

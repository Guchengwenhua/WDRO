"""Final lifecycle gate after the scheduler and independent report have exited."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXP=ROOT/'artifacts'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def active(record):
    p=Path(f"/proc/{record['pid']}/stat")
    if not p.exists():return False
    fields=p.read_text().rsplit(')',1)[1].split()
    return fields[19]==str(record.get('start_ticks',record.get('process_start_ticks')))
def unlocked(p):
    with p.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(f,fcntl.LOCK_UN)

def main():
    auditpath=EXP/'audit/completion_audit.json';audit=read(auditpath)
    assert audit['status']=='pass' and audit['formal_tasks']==16
    queue=read(EXP/'operations/queue.json');exit_record=read(EXP/'operations/exit.json')
    assert not queue['running'] and not queue['pending'] and not queue['failures']
    assert len(set(queue['completed']))==16
    assert not exit_record['pending'] and not exit_record['failures']
    assert not active(read(EXP/'operations/launch.json')),'scheduler still active'
    unlocked(EXP/'operations/queue.lock')
    taskpaths=[EXP/'fixed'/f'test_s{s}' for s in (101,102,103)]
    taskpaths += [EXP/'fixed/validation_s101']
    taskpaths += sorted((EXP/'training').glob('*_d30_s*'))
    assert len(taskpaths)==16
    for p in taskpaths:
        assert (p/'complete.json').is_file()
        assert read(p/'exit.json')['complete'] is True
        assert not active(read(p/'launch.json'))
        unlocked(p/'task.lock')
    for item in read(EXP/'freeze.json')['files']:
        assert sha(Path(item['path']))==item['sha256'],item['path']
    automation=EXP/'automation_manifest.json'
    if automation.exists():
        for item in read(automation)['files']:
            assert sha(Path(item['path']))==item['sha256'],item['path']
    for item in audit['published_outputs']:
        assert sha(Path(item['path']))==item['sha256'],item['path']
    status=ROOT/'STATUS.md'
    status_text=('# 当前执行状态\n\n2026-09-15：**DONE（本轮机制实验已验收）**。\n\n'
        '七个方向的理论、16个正式任务、独立数值复核与统计均已完成。所有worker及scheduler已退出，任务锁和队列锁已释放。\n\n'
        '- [研究结果与边界](RESULTS.md)\n- [独立验收](artifacts/audit/completion_audit.json)\n'
        '- [最终退出与源码核验](artifacts/audit/final_delivery.json)\n'
        '- [CSV统计](artifacts/tables/)\n\n'
        '范围为degree30、3个父模型seed、固定64评分搜索与128步继续训练；不替代主线正式实验，不构成连续最坏风险保证。\n')
    docroot=ROOT.parents[1]/'docs/STATUS.md'
    text=docroot.read_text();lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| tmp2 提出的 Critic 搜索七方向 |'):
            lines[i]='| tmp2 提出的 Critic 搜索七方向 | DONE；degree30、3 seeds、16项任务及独立复核完成 | [结果与验收](../improvements/critic_search_20260915/RESULTS.md) |'
    doc_text='\n'.join(lines)+'\n'
    pending=[]
    for path,content in ((status,status_text),(docroot,doc_text)):
        temp=path.with_name(path.name+'.partial')
        with temp.open('w') as handle:
            handle.write(content);handle.flush();os.fsync(handle.fileno())
        pending.append((temp,path))
    result=dict(status='pass',at=time.time(),formal_tasks=16,all_workers_exited=True,
        scheduler_exited=True,all_locks_released=True,frozen_sources_unchanged=True,
        outputs_rehashed=True,audit_sha256=sha(auditpath),
        scope='this independent degree30 mechanism study',
        artifacts=[dict(path=str(p),sha256=sha(p)) for p in (auditpath,EXP/'freeze.json')]+[
            dict(path=str(path),sha256=sha(temp)) for temp,path in pending])
    target=EXP/'audit/final_delivery.json';tmp=target.with_suffix('.partial')
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));os.replace(tmp,target)
    # Successful scientific evidence is published before either user-facing DONE state.
    for temp,path in pending:os.replace(temp,path)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()

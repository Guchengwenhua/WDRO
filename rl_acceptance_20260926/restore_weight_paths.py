"""Keep required training weights local; completed archives and arrays stay attached."""
import ctypes,hashlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path('/remote-home/Guilin/DRO/Codex/rl_prepost_scaleup_20260922')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import ops
def sha(path):return ops.sha256(path)
def exchange(left,right):
    libc=ctypes.CDLL(None,use_errno=True)
    fn=libc.renameat2;fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(left),-100,os.fsencode(right),2):raise OSError(ctypes.get_errno(),'renameat2 exchange failed')
def main():
    assert (ROOT/'artifacts/operations/PAUSE').exists()
    assert not ops.discover_live_workers(ROOT)
    p=json.loads((ROOT/'protocol.json').read_text());q=json.loads((ROOT/'artifacts/operations/queue.json').read_text())
    assert not q['running']
    remaining=[t for t in p['tasks'] if t['id'] not in q['complete']]
    local_runs={t['id'] for t in remaining};local_lineage=set();parents=[]
    for t in remaining:
        for parent in t['parents'].values():
            if parent['source']=='legacy':
                local_lineage.add(parent['run']);parents.append(ROOT/'lineage'/parent['run']/'latest.pt')
            else:
                local_runs.add(parent['run']);parents.append(ROOT/'artifacts/runs'/parent['run']/'checkpoints'/f'step_{parent["step"]}.pt')
    copied=[];records=[]
    for logical,wanted in [(ROOT/'artifacts/runs',local_runs),(ROOT/'lineage',local_lineage)]:
        assert logical.is_symlink()
        attached=logical.resolve();stage=logical.with_name(logical.name+'.local_weights_stage_20260926')
        assert not stage.exists();stage.mkdir()
        for source in sorted(attached.iterdir()):
            target=stage/source.name
            if source.name not in wanted:
                target.symlink_to(source,target_is_directory=source.is_dir());continue
            shutil.copytree(source,target,ignore=shutil.ignore_patterns('evaluations'),copy_function=shutil.copy2)
            for file in source.rglob('*'):
                rel=file.relative_to(source)
                if file.is_file() and 'evaluations' not in rel.parts:
                    assert sha(file)==sha(target/rel),str(file)
                    copied.append(dict(logical=str(logical/source.name/rel),sha256=sha(file),bytes=file.stat().st_size))
            if logical.name=='runs':
                evaluation=source/'evaluations';evaluation.mkdir(exist_ok=True)
                (target/'evaluations').symlink_to(evaluation,target_is_directory=True)
        assert wanted.issubset({x.name for x in stage.iterdir()})
        # Exchange directory and symlink atomically so concurrent read-only
        # auditors never observe a missing logical path.
        exchange(stage,logical)
        assert stage.is_symlink();stage.unlink()
        records.append(dict(logical=str(logical),attached_archive=str(attached),local_run_directories=sorted(wanted)))
    for parent in parents:assert parent.resolve().is_relative_to(ROOT) and parent.is_file(),str(parent)
    for t in remaining:assert (ROOT/'artifacts/runs'/t['id']).resolve().is_relative_to(ROOT)
    assert ops.verify_source_freeze(ROOT)
    record=dict(status='PASS',time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),records=records,restored_files=copied,
        restored_bytes=sum(x['bytes'] for x in copied),required_parent_files=len(set(parents)),remaining_tasks=len(remaining),
        reason='Frozen oracle guard requires training weight real paths within experiment root. Localize only required parents and remaining output directories; retain attached evaluation arrays and completed archives.',
        source_guard_unchanged=True,source_freeze_verified=True,primary_free_bytes=shutil.disk_usage(ROOT).free)
    ops.atomic(OUT/'storage_compatibility.json',record)
    # Leave paused until the real frozen guard and model-load probe passes.
    print(json.dumps({k:v for k,v in record.items() if k not in ['records','restored_files']}))
if __name__=='__main__':main()

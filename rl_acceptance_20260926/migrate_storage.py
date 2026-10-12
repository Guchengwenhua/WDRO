"""Move verified, inactive experiment data to attached storage, retaining paths."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
BASE=Path('/remote-home/Guilin/DRO/Codex')
OUT=Path(__file__).resolve().parent
ROOT=BASE/'rl_prepost_scaleup_20260922'
DEST=Path('/attached/remote-home1/Guilin/DRO/Codex')
sys.path.insert(0,str(ROOT))
import ops
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(name,value):ops.atomic(OUT/name,value)
def main():
    assert (ROOT/'artifacts/operations/PAUSE').exists()
    q=json.loads((ROOT/'artifacts/operations/queue.json').read_text())
    assert not q['running'] and not ops.discover_live_workers(ROOT)
    assert not ops.discover_live_workers(BASE/'rl_scaleup_20260921')
    assert ops.verify_source_freeze(ROOT) and ops.verify_source_freeze(BASE/'rl_scaleup_20260921')
    DEST.mkdir(parents=True,exist_ok=True)
    probe=DEST/f'.rl_acceptance_write_probe_{os.getpid()}'
    payload=os.urandom(4096)
    with probe.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    assert probe.read_bytes()==payload;probe.unlink()
    assert shutil.disk_usage(DEST).free>100*2**30
    paths=['rl_prepost_scaleup_20260922/artifacts/runs','rl_prepost_scaleup_20260922/lineage','rl_scaleup_20260921/artifacts/runs']
    receipt=dict(status='COPYING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),write_read_cleanup='PASS',
        root_free_before=shutil.disk_usage(BASE).free,attached_free_before=shutil.disk_usage(DEST).free,moves=[])
    dump('storage_migration.json',receipt)
    for relative in paths:
        src=BASE/relative;dest=DEST/relative
        assert src.is_dir() and not src.is_symlink() and not dest.exists(),str(src)
        for lock in src.rglob('run.lock'):assert not ops.lock_held(lock),str(lock)
        files=sorted(x for x in src.rglob('*') if x.is_file())
        assert not any(x.is_symlink() for x in src.rglob('*'))
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(src,dest,copy_function=shutil.copy2)
        evidence=[]
        for i,path in enumerate(files):
            target=dest/path.relative_to(src);a=sha(path);b=sha(target)
            assert a==b,(str(path),'copy mismatch')
            evidence.append(dict(path=str(path.relative_to(src)),bytes=path.stat().st_size,sha256=a))
        index=OUT/f'migration_manifest_{len(receipt["moves"])}.json';dump(index,evidence)
        assert len(files)==len([x for x in dest.rglob('*') if x.is_file()])
        backup=src.with_name(src.name+'.verified_migration_backup_20260926')
        assert not backup.exists()
        src.rename(backup)
        try:src.symlink_to(dest,target_is_directory=True)
        except BaseException:backup.rename(src);raise
        assert all((src/v['path']).stat().st_size==v['bytes'] for v in evidence)
        # Delete only the original copy after all destination bytes were checked.
        shutil.rmtree(backup)
        receipt['moves'].append(dict(logical=str(src),resolved=str(src.resolve()),files=len(files),bytes=sum(x['bytes'] for x in evidence),manifest=str(index),manifest_sha256=sha(index)))
        dump('storage_migration.json',receipt)
        print(json.dumps(receipt['moves'][-1]),flush=True)
    assert ops.verify_source_freeze(ROOT) and ops.verify_source_freeze(BASE/'rl_scaleup_20260921')
    receipt.update(status='PASS',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),root_free_after=shutil.disk_usage(BASE).free,
        attached_free_after=shutil.disk_usage(DEST).free,source_freezes_unchanged=True)
    dump('storage_migration.json',receipt)
    ops.atomic(ROOT/'artifacts/operations/storage_20260926.json',receipt)
    # The existing scheduler owns continuation; no duplicate queue is launched.
    (ROOT/'artifacts/operations/PAUSE').unlink()
    print(json.dumps(dict(status='PASS',root_free_gib=receipt['root_free_after']/2**30,resumed_existing_scheduler=True)),flush=True)
if __name__=='__main__':main()

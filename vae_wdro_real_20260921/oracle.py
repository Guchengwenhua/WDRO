"""Single global reference queue; SQLite full-coordinate BLOB keys, no digest."""
import argparse
import multiprocessing as mp
import sqlite3
import time
import numpy as np
from exp_common import *

def worker(conn):
    from src.geosteiner import GeoSteinerEvaluator
    evaluator=GeoSteinerEvaluator(CFG['library'])
    while True:
        item=conn.recv()
        if item is None: break
        idx,key=item
        try:
            pins=np.frombuffer(key,dtype=np.float32).reshape(30,2)
            value=evaluator.solve(pins,require_integral=True)
            conn.send((idx,float(value),None))
        except Exception as exc: conn.send((idx,None,repr(exc)))
    conn.close()

def fill(name):
    if (ART/'done'/f'{name}_reference.json').exists(): return
    start=time.monotonic();data=load_data()
    with np.load(ART/'pools'/f'{name}.geometry.npz') as f: bank={k:f[k] for k in f.files}
    pins=bank['pins'];flat=pins.reshape(-1,30,2);keys=[x.tobytes() for x in flat]
    db=sqlite3.connect(ART/'reference_cache.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS reference (coordinates BLOB PRIMARY KEY, value REAL, error TEXT)')
    center_indices=bank['center_index']
    db.executemany('INSERT OR IGNORE INTO reference VALUES (?,?,NULL)',[(data['pins'][i].tobytes(),float(data['lstar'][i])) for i in center_indices]);db.commit()
    cache={row[0]:(row[1],row[2]) for row in db.execute('SELECT coordinates,value,error FROM reference')}
    unique=list(dict.fromkeys(keys));missing=[k for k in unique if k not in cache]
    workers=min(CFG['runtime']['total_reference_workers_default'],len(missing));context=mp.get_context('fork')
    state=[];next_idx=0;completed=0;new_failed=0;last_save=time.monotonic()
    def launch():
        parent,child=context.Pipe();proc=context.Process(target=worker,args=(child,),daemon=True);proc.start();child.close()
        return dict(proc=proc,conn=parent,index=None,start=None)
    def assign(s):
        nonlocal next_idx
        if next_idx<len(missing):
            s['index']=next_idx;s['start']=time.monotonic();s['conn'].send((next_idx,missing[next_idx]));next_idx+=1
    def commit(idx,value,error):
        nonlocal completed,new_failed
        key=missing[idx];cache[key]=(value,error);completed+=1;new_failed+=int(error is not None)
        db.execute('INSERT OR REPLACE INTO reference VALUES (?,?,?)',(key,value,error))
        if error: append_json(ART/'reference_failures.jsonl',dict(method=name,index=idx,error=error,time=utc()))
    try:
        for _ in range(workers):
            s=launch();state.append(s);assign(s)
        while completed<len(missing):
            for j,s in enumerate(state):
                if s['index'] is None: continue
                if s['conn'].poll():
                    try: idx,value,error=s['conn'].recv()
                    except EOFError:
                        idx,value,error=s['index'],None,'reference worker exited unexpectedly'
                        s['proc'].join(timeout=1);s['conn'].close();state[j]=launch();s=state[j]
                    commit(idx,value,error);s['index']=None;assign(s)
                elif time.monotonic()-s['start']>CFG['runtime']['reference_timeout_seconds'] or not s['proc'].is_alive():
                    idx=s['index'];s['proc'].terminate();s['proc'].join(timeout=2);s['conn'].close()
                    commit(idx,None,'30-second timeout or worker exit');state[j]=launch();assign(state[j])
            if time.monotonic()-last_save>=5 or completed==len(missing):
                db.commit();elapsed=time.monotonic()-start;rate=completed/max(elapsed,.001)
                progress(name+'_reference',completed=completed,total=len(missing),cache_reused=len(unique)-len(missing),
                    failed=new_failed,elapsed_seconds=elapsed,remaining_seconds=(len(missing)-completed)/max(rate,1e-9))
                print('REFERENCE',name,completed,'/',len(missing),'elapsed',round(elapsed,1),flush=True);last_save=time.monotonic()
            time.sleep(.002)
    finally:
        db.commit()
        for s in state:
            if s['proc'].is_alive():
                if s['index'] is None:
                    try:s['conn'].send(None)
                    except (BrokenPipeError,EOFError):pass
                    s['proc'].join(timeout=2)
                if s['proc'].is_alive():s['proc'].terminate();s['proc'].join(timeout=2)
            s['conn'].close()
        db.close()
    references=np.array([cache[k][0] if cache[k][0] is not None else np.nan for k in keys]).reshape(pins.shape[:2])
    valid=np.isfinite(references)&(references>0);invalid=float((~valid[:,1:]).mean())
    if not valid[:,0].all(): raise RuntimeError('nominal references missing')
    summary=dict(method=name,seconds=time.monotonic()-start,distinct_coordinates=len(unique),
        cached_distinct=len(unique)-len(missing),new_reference_requests=len(missing),new_failed=new_failed,
        invalid_non_nominal_fraction=invalid,all_requested_candidates=len(flat),reference_scale=2**27)
    atomic_json(ART/'pools'/f'{name}.reference.json',summary)
    if invalid>CFG['runtime']['maximum_invalid_reference_fraction']: raise RuntimeError('invalid reference fraction exceeds 1%; no effect claim')
    atomic_npz(ART/'pools'/f'{name}.npz',**bank,lstar=references,valid=valid)
    mark_done(name+'_reference',**summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method');fill(p.parse_args().method)

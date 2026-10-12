#!/usr/bin/env python3
"""Read-only live worker observation; a partial audit never completes Stage 8."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scipy.optimize import linear_sum_assignment
from stage8.common import *
from stage8.operations import process_matches,post_snapshot
from stage8.attack import restore_rounds,reduce_candidates,independent_tree_lengths
from src.metrics import relative_regret


def main():
    plan=gate();queue=read(LOG/'pipeline/attacks/queue.json')
    assert queue['status']=='running' and queue['active']
    rows=[]
    for key,row in queue['active'].items():
        assert process_matches(row),key
        directory=Path(row['output']).parent
        state=read(directory/'cursor.json');assert state['batches'],key
        batch=state['batches'][0];assert batch['start']==0 and batch['end']==16
        data=load_npz(checked(state['binding']['input']['input']))
        centers,ids=data['pins'][:16],data['net_ids'][:16]
        _,_,_,candidates=restore_rounds(centers,batch['rounds'],plan['attack'],ids,reconstruct=True)
        for a in candidates:
            pins=a['pins'].reshape(-1,centers.shape[1],2)
            np.testing.assert_array_equal(independent_tree_lengths(pins,a['topology'].reshape(len(pins),-1)),a['rest_lengths'].ravel())
            np.testing.assert_array_equal(relative_regret(a['rest_lengths'],a['lstar']),a['regret'])
        selected=load_npz(checked(batch['predictions']))
        rebuilt=reduce_candidates(centers,ids,candidates)
        for k in rebuilt:np.testing.assert_array_equal(rebuilt[k],selected[k])
        for i in range(16):
            distances=np.abs(centers[i,:,None,:].astype(np.float64)-selected['pins'][i,None,:,:]).sum(axis=2)
            a,b=linear_sum_assignment(distances)
            assert selected['cost'][i]==distances[a,b].mean()
        np.testing.assert_array_equal(selected['lstar'],selected['verification_lstar'])
        job=read(directory/'job.json')
        assert job['session']=='wdro_stage8_cem256_v1'
        for k,v in job['runtime_environment'].items():
            if k not in ('CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE'):
                assert v and Path(v).is_relative_to(TOP),(k,v)
        rounds=[r for b in state['batches'] for r in b['rounds']]
        stored_bytes=sum(Path(r['artifact']['logical_path']).stat().st_size for r in batch['rounds'])
        rows.append(dict(task=key,pid=row['pid'],gpu=row['gpu'],model=state['binding']['model'],
            first_batch=batch,observed_complete_nets=state['complete_nets'],
            current_round=len((state.get('current') or {}).get('rounds',[])),
            cumulative_search_seconds_per_center=sum(r['wall_seconds'] for r in rounds)/state['complete_nets'],
            stored_candidate_bytes_per_center=stored_bytes/16,worker_job=artifact(directory/'job.json'),
            log=dict(logical_path=row['log'],resolved_path=str(Path(row['log']).resolve())),
            heartbeat_at_read=read(directory/'heartbeat.json')))
    key='erm_d50_s101_uniform'
    actual=next(r for r in rows if r['task']==key)
    first=load_npz(checked(actual['first_batch']['predictions']))
    smoke=load_npz(checked(read(LOG/'smoke/continuous/result.json')['predictions']))
    for k in smoke:np.testing.assert_array_equal(first[k],smoke[k])
    report=dict(status='pass',partial_observation_only=True,stage8_complete=False,at_utc=utc_now(),
        code=artifact(Path(__file__)),plan=artifact(PLAN),operations=artifact(ROOT/'manifests/stage8_operations_v1.json'),
        launch=artifact(LOG/'latest_launch.json'),complete_cells_at_read=len(queue['complete']),
        active_workers=len(rows),audited_centers=len(rows)*16,audited_candidates=len(rows)*4096,
        formal_first_batch_exactly_matches_foreground_smoke=True,
        snapshot=post_snapshot('live_first_batch_audit'),rows=rows)
    path=LOG/'observations'/('live_'+stamp()+'.json');atomic_write_json(path,report,mode=0o444)
    atomic_write_json(LOG/'observations/latest.json',artifact(path))
    print(dict(report=str(path),workers=len(rows),centers=len(rows)*16,candidates=len(rows)*4096,
        complete_cells=len(queue['complete']),progress=[{k:r[k] for k in ('task','gpu','observed_complete_nets','cumulative_search_seconds_per_center')} for r in rows]),flush=True)


if __name__=='__main__':main()

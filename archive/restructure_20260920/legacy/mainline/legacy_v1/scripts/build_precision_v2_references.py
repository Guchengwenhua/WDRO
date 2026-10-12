"""Versioned fixed references with two-scale checks and atomic batch recovery."""
import argparse,json,multiprocessing as mp,os,sys,time
from pathlib import Path
ORIGINAL=Path(__file__).resolve().parents[2]
NEW=Path('/remote-home/Guilin/DRO/Codex/wdro_precision_v2')
ROOT=NEW/'wdro_rest_min';EXP=NEW/'experiments';sys.path.insert(0,str(ROOT))
import numpy as np
from src.datasets import load_dataset,array_bundle_sha256
from src.geosteiner import GeoSteinerEvaluator,reference_identity,SCALE
from src.rest import file_sha256
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from stage6.evaluation import atomic_npz,artifact
EVALUATOR=None

def initialize():
    global EVALUATOR
    EVALUATOR=GeoSteinerEvaluator()

def compute(pins):
    assert pins.dtype==np.float32 and np.array_equal(pins.astype(np.float64)*SCALE,np.rint(pins.astype(np.float64)*SCALE))
    a=EVALUATOR.lengths(pins);b=EVALUATOR.lengths(pins.astype(np.float64)*2)/2
    np.testing.assert_array_equal(a,b)
    return a

def publish(path,arrays,metadata):
    # The complete metadata is prepared and validated before either final rename.
    part=path.with_suffix('.npz.partial');path.parent.mkdir(parents=True,exist_ok=True)
    with part.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    with np.load(part,allow_pickle=False) as z:
        for key in arrays:np.testing.assert_array_equal(z[key],arrays[key])
    meta=dict(metadata,array_bundle_sha256=array_bundle_sha256(arrays),npz_sha256=file_sha256(part))
    mpart=path.with_suffix('.meta.json.partial');atomic_write_json(mpart,meta)
    os.replace(part,path);os.replace(mpart,path.with_suffix('.meta.json'));path.chmod(0o444);path.with_suffix('.meta.json').chmod(0o444)
    return meta

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);parser.add_argument('--max-new-chunks',type=int);args=parser.parse_args()
    accepted=json.loads((ORIGINAL/'experiments/logs/precision_v2/reference_verification/acceptance.json').read_text())
    assert accepted['status']=='pass' and len(accepted['cases'])==38
    assert accepted['reference_identity']==reference_identity()
    identity=reference_identity();output=EXP/'cache/lstar/protocol_v1';log=EXP/'logs/reference_cache_v2';log.mkdir(parents=True,exist_ok=True)
    entries=[];new_chunks=0;all_changes=[]
    sources=[EXP/f'cache/datasets/protocol_v1/{split}/{dist}_d{d}_s{s}_n{n}.npz' for split,s,n in [('validation',202,2000),('test',303,10000)] for d in [10,20,30,40,50] for dist in ['uniform','mix_gauss','edge_bias','grid_cluster']]
    with mp.get_context('spawn').Pool(args.workers,initializer=initialize) as pool:
        for source in sources:
            arrays,meta=load_dataset(source,verify=True);bundle=meta['array_bundle_sha256'];target=output/f'{bundle}.npz'
            oldpath=ORIGINAL/f'experiments/cache/lstar/protocol_v1/{bundle}.npz';old,_=load_dataset(oldpath,verify=True)
            cursor=log/'cursors'/f'{bundle}.json';binding=dict(reference_identity=identity,dataset=artifact(source),old_reference=artifact(oldpath),chunk_size=128)
            state=json.loads(cursor.read_text()) if cursor.exists() else dict(binding=binding,shards=[],complete_nets=0)
            assert state['binding']==binding
            for shard in state['shards']:assert file_sha256(Path(shard['artifact']['logical_path']))==shard['artifact']['sha256']
            if not (target.exists() and target.with_suffix('.meta.json').exists()):
                start=state['complete_nets'];stops=list(range(start,len(arrays['pins']),128))
                if args.max_new_chunks is not None:stops=stops[:max(0,args.max_new_chunks-new_chunks)]
                jobs=[arrays['pins'][i:i+128] for i in stops]
                for begin,lengths in zip(stops,pool.imap(compute,jobs),strict=True):
                    end=begin+len(lengths);path=log/'shards'/bundle/f'{begin:06d}_{end:06d}.npz'
                    atomic_npz(path,dict(lstar=lengths,net_ids=arrays['net_ids'][begin:end]));state['shards'].append(dict(start=begin,end=end,artifact=artifact(path)))
                    state.update(complete_nets=end,updated_at_utc=utc_now());atomic_write_json(cursor,state);new_chunks+=1
                    atomic_write_json(log/'heartbeat.json',dict(at_utc=utc_now(),dataset=str(source),dataset_complete_nets=end,completed_cells=len(entries),pid=os.getpid()))
                if state['complete_nets']<len(arrays['pins']):
                    print(json.dumps(dict(status='paused',cursor=str(cursor),complete_nets=state['complete_nets'])));return
                lengths=np.concatenate([np.load(s['artifact']['logical_path'],allow_pickle=False)['lstar'] for s in state['shards']])
                publish(target,dict(lstar=lengths,net_ids=arrays['net_ids']),dict(schema_version=2,kind='geosteiner_lstar',reference_identity=identity,
                    dataset_path=str(source.relative_to(NEW.parent)),dataset_array_bundle_sha256=bundle,degree=meta['degree'],count=len(lengths),
                    evaluator_calls=2*len(lengths),library_path='DRO/WDRO-REST/algorithms/libeval.so',two_scales_exactly_equal=True,formal_input_integral=True))
            fresh,rmeta=load_dataset(target,verify=True);assert rmeta['reference_identity']==identity
            np.testing.assert_array_equal(fresh['net_ids'],arrays['net_ids']);delta=old['lstar']-fresh['lstar'];changed=np.flatnonzero(delta!=0)
            row=dict(dataset_path=str(source.relative_to(NEW.parent)),dataset_array_bundle_sha256=bundle,cache_path=str(target.relative_to(NEW.parent)),
                     cache_npz_sha256=rmeta['npz_sha256'],count=len(delta),evaluator_calls=rmeta['evaluator_calls'],
                     changed_count=len(changed),old_larger=int(np.sum(delta>0)),old_smaller=int(np.sum(delta<0)),
                     maximum_absolute_delta=float(np.max(np.abs(delta))),sources=binding)
            for i in changed:all_changes.append(dict(net_id=arrays['net_ids'][i].decode(),old=float(old['lstar'][i]),corrected=float(fresh['lstar'][i]),delta=float(delta[i])))
            entries.append(row);atomic_write_json(log/'progress.json',dict(status='running',at_utc=utc_now(),records=entries));print(json.dumps({k:v for k,v in row.items() if k!='sources'}),flush=True)
    assert len(entries)==40 and sum(r['count'] for r in entries)==240000
    index=dict(schema_version=2,status='pass',created_at_utc=utc_now(),reference_identity=identity,library='DRO/WDRO-REST/algorithms/libeval.so',
               workers=args.workers,dataset_count=40,total_evaluator_calls=480000,records=entries)
    if any(r['old_smaller'] for r in entries):
        index['status']='needs_investigation_old_reference_smaller'
    atomic_write_json(log/'reference_changes.json',dict(status=index['status'],at_utc=utc_now(),changes=all_changes))
    atomic_write_json(output/'lstar_index.json',index,mode=0o444)
    atomic_write_json(log/'completion.json',dict(status=index['status'],at_utc=utc_now(),index=artifact(output/'lstar_index.json'),changed_count=len(all_changes)))
    assert index['status']=='pass','A previous reference is shorter; diagnose before training'
    print(json.dumps(dict(status='pass',datasets=40,nets=240000,changed=len(all_changes))),flush=True)
if __name__=='__main__':main()

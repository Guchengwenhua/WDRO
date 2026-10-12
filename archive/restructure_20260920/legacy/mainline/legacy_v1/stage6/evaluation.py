"""Shared greedy inference with hash-bound atomic batch recovery."""
import json, os, time
from pathlib import Path
import numpy as np
import torch
from src.datasets import load_dataset
from src.io_utils import atomic_write_json
from src.metrics import relative_regret, summarize_regret
from src.rest import file_sha256, build_actor, RestTreeEvaluator
from src.trainer import configure_determinism, load_trainer_config
from src.protocol import load_protocol
from src.job_control import utc_now
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT.parent/'experiments'

def artifact(path):
    p=Path(path);return dict(logical_path=str(p),resolved_path=str(p.resolve()),sha256=file_sha256(p))

def load_group(degree, distribution, split):
    protocol=load_protocol(); assert split in ('validation','test')
    count=protocol['data']['sizes_per_degree_distribution'][split]; seed=protocol['data']['seeds'][split]
    path=EXP/'cache/datasets/protocol_v1'/split/f'{distribution}_d{degree}_s{seed}_n{count}.npz'
    data,meta=load_dataset(path,verify=True)
    assert meta['split']==split and meta['degree']==degree and meta['base_seed']==seed and meta['distribution']==distribution
    assert len(data['pins'])==count and all(x.decode().startswith(split+':') for x in data['net_ids'])
    reference=EXP/'cache/lstar/protocol_v1'/f"{meta['array_bundle_sha256']}.npz"
    ref,rmeta=load_dataset(reference,verify=True)
    assert rmeta['dataset_array_bundle_sha256']==meta['array_bundle_sha256']
    np.testing.assert_array_equal(data['net_ids'],ref['net_ids'])
    assert np.isfinite(ref['lstar']).all() and np.all(ref['lstar']>0)
    return data,ref,dict(dataset=artifact(path),reference=artifact(reference),dataset_bundle=meta['array_bundle_sha256'])

def atomic_npz(path, arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.npz.partial')
    with temp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    with np.load(temp,allow_pickle=False) as check:
        for k,v in arrays.items():np.testing.assert_array_equal(check[k],v)
    os.replace(temp,path)

def evaluate_cell(model,distribution,split,out,hardware,*,max_new_batches=None,heartbeat=None):
    """The test caller must supply a pre-test frozen, audited model record."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(model['best']['logical_path']);assert file_sha256(checkpoint)==model['best']['sha256']
    data,reference,sources=load_group(model['degree'],distribution,split)
    binding=dict(model=model,distribution=distribution,split=split,sources=sources,
        batch_size=256,T=1,mode='deterministic_greedy',hardware=hardware,
        evaluator_sha256=file_sha256(Path(__file__)))
    cursor=out/'cursor.json'; state=json.loads(cursor.read_text()) if cursor.exists() else dict(binding=binding,shards=[],complete_nets=0)
    assert state['binding']==binding,'inference inputs or hardware changed; cursor cannot be reused'
    for shard in state['shards']:assert file_sha256(Path(shard['artifact']['logical_path']))==shard['artifact']['sha256']
    n=len(data['pins']);result=out/'result.json'
    if result.exists():
        old=json.loads(result.read_text());assert old['status']=='complete' and old['binding']==binding
        assert old['completed_nets']==n and file_sha256(Path(old['predictions']['logical_path']))==old['predictions']['sha256']
        return old
    configure_determinism(load_trainer_config());device=torch.device('cuda:0')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['run_identity']['degree']==model['degree'] and ck['run_identity']['training_seed']==model['seed']
    actor=build_actor(model['degree'],device);actor.load_state_dict(ck['actor_state_dict'],strict=True);actor.eval()
    rest=RestTreeEvaluator();new_batches=0;torch.cuda.reset_peak_memory_stats()
    for start in range(state['complete_nets'],n,256):
        end=min(n,start+256);path=out/'shards'/f'{start:06d}_{end:06d}.npz'
        # A crash after atomic shard publication but before cursor commit recomputes
        # only that uncommitted shard; attempts are recorded separately by the worker.
        pins=data['pins'][start:end];torch.cuda.synchronize();began=time.monotonic()
        with torch.no_grad(): topology,_=actor(pins,True)
        topology=topology.detach().cpu().numpy();torch.cuda.synchronize();actor_seconds=time.monotonic()-began
        began=time.monotonic();lengths=rest.lengths(pins,topology);rest_seconds=time.monotonic()-began
        lstar=reference['lstar'][start:end];regret=relative_regret(lengths,lstar)
        assert np.isfinite(regret).all() and np.min(regret)>-1e-5
        arrays=dict(net_ids=data['net_ids'][start:end],rest_lengths=lengths,lstar=lstar,regret=regret,topology=topology)
        atomic_npz(path,arrays)
        state['shards'].append(dict(start=start,end=end,artifact=artifact(path),actor_seconds=actor_seconds,rest_seconds=rest_seconds))
        state['complete_nets']=end;state['updated_at_utc']=utc_now()
        state['peak_reserved_mib']=max(state.get('peak_reserved_mib',0),torch.cuda.max_memory_reserved()/2**20)
        atomic_write_json(cursor,state)
        if heartbeat:heartbeat(end)
        new_batches+=1
        if max_new_batches is not None and new_batches>=max_new_batches and end<n:return dict(status='paused',completed_nets=end,cursor=str(cursor))
    parts=[]
    for shard in state['shards']:
        with np.load(shard['artifact']['logical_path'],allow_pickle=False) as f:parts.append({k:f[k] for k in f.files})
    arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.testing.assert_array_equal(arrays['net_ids'],data['net_ids']);np.testing.assert_array_equal(arrays['lstar'],reference['lstar'])
    np.testing.assert_array_equal(arrays['regret'],relative_regret(arrays['rest_lengths'],arrays['lstar']))
    assert len(set(arrays['net_ids']))==n
    predictions=out/'predictions.npz';atomic_npz(predictions,arrays)
    result_value=dict(status='complete',at_utc=utc_now(),binding=binding,completed_nets=n,predictions=artifact(predictions),
        selected_update=ck['update'],summary=summarize_regret(arrays['regret']),peak_reserved_mib=state['peak_reserved_mib'],
        call_ledger=dict(actor_forward_nets=n,exact_rest_evaluator_nets=n,geosteiner_nets=0),
        actor_seconds=sum(s['actor_seconds'] for s in state['shards']),rest_seconds=sum(s['rest_seconds'] for s in state['shards']),
        accounting='committed batches; failed/replayed attempts logged separately',test_used_for_selection=False)
    atomic_write_json(result,result_value,mode=0o444);return result_value

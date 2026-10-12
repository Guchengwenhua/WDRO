"""GPU inference of frozen Actors on shared coordinates, with resumable cells."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import scheduler as S
from common import ROOT, G, read, sha, checked, atomic, install_guard, state_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--physical-gpu-index', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    manifest = read(ROOT/'manifest.json')
    model = next(m for m in manifest['models'] if m['id'] == args.model_id)
    install_guard(manifest['models'])
    out = ROOT/'artifacts'/('smoke' if args.smoke else 'runs')/model['id']
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity = dict(**S.identity(os.getpid()), device='cuda:0', physical_gpu_index=args.physical_gpu_index,
                    torch_version=torch.__version__, numpy_version=np.__version__, visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    binding = dict(model_id=model['id'], manifest_sha256=sha(ROOT/'manifest.json'), protocol_sha256=sha(ROOT/'protocol.json'),
                   model=model, sources={str(p):sha(p) for p in (ROOT/'evaluate.py', ROOT/'common.py')},
                   dependencies=G.dependencies(), geometry_sha256=sha(Path(G.__file__)), smoke=args.smoke)
    if (out/'run.json').exists(): assert read(out/'run.json')['binding'] == binding
    else: atomic(out/'run.json', dict(binding=binding, first_identity=identity, started_at=time.time()))
    assert torch.cuda.is_available() and str(args.physical_gpu_index) == os.environ['CUDA_VISIBLE_DEVICES']
    torch.set_num_threads(int(os.environ.get('RL_ONLY_CPU_THREADS',3))); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.10, device=0)
    checked(model['checkpoint'], model['checkpoint_sha256'])
    cp = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
    state = cp[model['state_key']]
    for v in state.values():
        if torch.is_tensor(v) and v.is_floating_point(): assert torch.isfinite(v).all()
    if model['family'] == 'mainline':
        r = cp['run_identity']
        assert (r['method'], r['degree'], r['training_seed']) == (model['method'],30,model['seed'])
        selected = cp['update']
    else:
        r = cp['binding']
        assert (r['method'], r['seed']) == (model['method'],model['seed'])
        assert state_hash(state) == model['actor_sha256']
        selected = cp['step']; assert selected == model['selected_update']
    actor = G.build_actor(30, torch.device('cuda:0')); actor.load_state_dict(state, strict=True); actor.eval()
    actor_digest = state_hash(actor.state_dict())
    del state, cp
    started = time.monotonic(); real_done = 0; forward_done = 0
    cells = []; completed = []; reused = 0; replay = 0
    data_rows = manifest['data']
    if args.smoke: data_rows = [d for d in data_rows if d['group']=='uniform']
    planned = sum((min(d['count'],128) if args.smoke else d['count'])
                  if model['family']=='mainline' or d['suite']=='mainline_test' else 64 for d in data_rows)
    saved = None
    if model['family']=='rl_only':
        checked(model['saved_results'], model['saved_results_sha256'])
        saved = np.load(model['saved_results'], allow_pickle=False)

    def beat(phase):
        elapsed = time.monotonic()-started
        atomic(out/'heartbeat.json', dict(**identity, updated=time.time(), phase=phase, model_id=model['id'],
               inference_rows_completed=real_done, inference_rows_target=planned, reused_rows=reused,
               seconds_per_row=elapsed/max(real_done,1),
               estimated_remaining_seconds=(planned-real_done)*elapsed/real_done if real_done else None))

    def call(pins, phase):
        nonlocal real_done, forward_done
        call_id = time.time_ns(); n = len(pins); forwards = math.ceil(n/64)*64
        def event(kind):
            with (out/'calls.jsonl').open('a') as f:
                f.write(json.dumps(dict(id=call_id,event=kind,phase=phase,real_rows=n,forward_rows=forwards,time=time.time()))+'\n')
        event('start'); tops=[]; lengths=[]
        with torch.no_grad():
            for k in range(0,n,64):
                batch = pins[k:k+64]; keep = len(batch)
                if keep < 64: batch=np.concatenate((batch,np.repeat(batch[-1:],64-keep,axis=0)))
                top, logp = actor(batch, True)
                assert torch.isfinite(logp).all()
                top = top[:keep].cpu().numpy()
                lengths.append(G.lengths(batch[:keep],top)); tops.append(top.astype(np.int16))
                real_done += keep; forward_done += 64
                if k % 1024 == 0: beat(phase)
        event('complete')
        return np.concatenate(lengths), np.concatenate(tops)

    beat('ready')
    for data in data_rows:
        checked(data['path'],data['sha256'])
        with np.load(data['path'],allow_pickle=False) as z: pins=z['pins'].reshape(-1,30,2)
        if args.smoke: pins=pins[:128]
        key=data['suite']+'__'+data['group']; path=out/(key+'.npz'); marker=out/(key+'.json')
        cell_binding=dict(run=binding,data_sha256=data['sha256'],count=len(pins))
        if marker.exists():
            previous=read(marker); assert previous['binding']==cell_binding
            checked(path,previous['sha256']); cells.append(previous); completed.append(key)
            if previous['mode']=='reused': reused+=len(pins)
            else: real_done+=len(pins)
            continue
        if model['family']=='rl_only' and data['suite'] in ('rl_test','rl_attack32'):
            prefix=('final' if data['suite']=='rl_test' else 'attack')+'__'+data['group']+'__'
            original_pins=saved[prefix+'pins'].reshape(-1,30,2)[:len(pins)]
            np.testing.assert_array_equal(pins,original_pins)
            length=saved[prefix+'length'].reshape(-1)[:len(pins)]
            topology=saved[prefix+'topology'].reshape(-1,58)[:len(pins)].astype(np.int16)
            test_length,test_top=call(pins[:64],'compatibility_replay/'+key)
            np.testing.assert_array_equal(test_top,topology[:64])
            np.testing.assert_allclose(test_length,length[:64],rtol=0,atol=1e-12)
            replay+=64; reused+=len(pins); mode='reused'
        else:
            length,topology=call(pins,'inference/'+key); mode='new_gpu_inference'
        np.testing.assert_allclose(G.lengths(pins,topology),length,rtol=0,atol=1e-12)
        G.atomic_npz(path, length=length, topology=topology)
        record=dict(key=key,suite=data['suite'],group=data['group'],mode=mode,binding=cell_binding,
                    sha256=sha(path),count=len(pins),path=str(path))
        atomic(marker,record);cells.append(record);completed.append(key)
        print(json.dumps(dict(model=model['id'],cell=key,rows=len(pins),mode=mode)),flush=True)
        beat('cell_completed')
    if saved is not None:saved.close()
    assert state_hash(actor.state_dict())==actor_digest,'evaluation changed model weights'
    events=[json.loads(x) for x in (out/'calls.jsonl').read_text().splitlines()]
    starts={x['id']:x for x in events if x['event']=='start'}
    ends={x['id']:x for x in events if x['event']=='complete'}
    assert ends.keys()<=starts.keys()
    calls=dict(completed_real_rows=sum(x['real_rows'] for x in ends.values()),
               attempted_real_rows_upper_bound=sum(x['real_rows'] for x in starts.values()),
               completed_forward_rows=sum(x['forward_rows'] for x in ends.values()),
               unresolved_calls=len(starts)-len(ends))
    artifacts={p.name:sha(p) for p in sorted(out.glob('*.npz'))}
    artifacts.update({p.name:sha(p) for p in sorted(out.glob('*.json')) if p.name not in ('complete.json','heartbeat.json')})
    artifacts['calls.jsonl']=sha(out/'calls.jsonl')
    completion=dict(status='complete',acceptance='gpu_smoke' if args.smoke else 'accepted_gpu',binding=binding,
        identity=identity,selected_update=selected,actor_sha256=actor_digest,cells=cells,artifacts=artifacts,
        actual_calls=calls,reused_rows=sum(c['count'] for c in cells if c['mode']=='reused'),
        unique_new_inference_rows=sum(c['count'] for c in cells if c['mode']=='new_gpu_inference'),
        compatibility_replay_rows_per_uninterrupted_run=0 if model['family']=='mainline' else 64*sum(c['mode']=='reused' for c in cells),
        elapsed_seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        new_training_updates=0,reference_solver_calls=0,
        forbidden_modules_loaded=[n for n in __import__('sys').modules if 'geosteiner' in n.lower() or n=='scipy.optimize'])
    assert not completion['forbidden_modules_loaded']
    atomic(out/'complete.json',completion);beat('complete')
    print(json.dumps(dict(status='complete',model=model['id'],calls=calls)),flush=True)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__': main()

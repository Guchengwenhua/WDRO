"""One locked task, atomic checkpoints, true-score accounting and exact resume."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time
import traceback
import numpy as np
import torch
from common import (ART, PROTOCOL, DISTS, ExactOracle, atomic_json, atomic_npz, atomic_torch,
    append_json, artifact, checked, digest, data, array_hash, derived_seed, proc_ticks,
    build_initial, dual_settings, evaluate, best_key, verify_freeze, load_parent,
    _rng_state, _restore_rng, _cpu_copy, sampled_update, sha256, _configure)
from search import attack

class Pause(Exception): pass

def checkpoint(actor, critic, optimizer, step, price, task_hash, initial, best, seconds):
    return _cpu_copy(dict(actor_state_dict=actor.state_dict(), critic_state_dict=critic.state_dict(),
        optimizer_state_dict=optimizer.state_dict(), step=step, price=price, task_hash=task_hash,
        initial_validation=initial, best_validation=best, rng=_rng_state(), train_seconds=seconds))

def train(task, out, device, state, pause, *, stop_after=None, workers=2):
    degree, seed = task['degree'], task['seed']
    actor, critic, init = build_initial(degree, seed, device)
    if task.get('parent'):
        parent = Path(task['parent'])
        receipt = json.loads((parent/'complete.json').read_text())
        cp = torch.load(checked(receipt['best']), map_location='cpu', weights_only=False)
        actor.load_state_dict(cp['actor_state_dict']); critic.load_state_dict(cp['critic_state_dict'])
        init = receipt['best']
    identity = dict(task=task, initialization=init, source_freeze=state.get('source_freeze'))
    task_hash = digest(identity)
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text()) != identity: raise RuntimeError('run identity changed')
    else: atomic_json(out/'identity.json', identity)
    optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=task['lr'])
    price, rho = dual_settings(degree)
    oracle = ExactOracle(actor, device, out/'oracle', workers)
    start, total_seconds = 0, 0.0
    pre = task['phase'] == 'pre'
    split = 'pilot-validation' if task['phase'] == 'pilot' else 'model-validation'
    val_count = task.get('validation_count', PROTOCOL['validation_count'])
    interval = task.get('validation_every', PROTOCOL['validation_every'])
    cp_every = task.get('checkpoint_every', PROTOCOL['checkpoint_every'])
    try:
        if (out/'latest.pt').exists():
            cp = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
            if cp['task_hash'] != task_hash: raise RuntimeError('checkpoint identity drift')
            actor.load_state_dict(cp['actor_state_dict']); critic.load_state_dict(cp['critic_state_dict'])
            optimizer.load_state_dict(cp['optimizer_state_dict'])
            for values in optimizer.state.values():
                for key, value in values.items():
                    if torch.is_tensor(value): values[key] = value.to(device)
            _restore_rng(cp['rng']); start, price = cp['step'], cp['price']
            initial, best = cp['initial_validation'], cp['best_validation']
            total_seconds = cp['train_seconds']
        else:
            state.update(phase='initial_validation', step=0)
            initial = evaluate(oracle, degree, val_count, split)
            best = initial
            cp = checkpoint(actor, critic, optimizer, 0, price, task_hash, initial, best, 0)
            atomic_torch(out/'best.pt', cp); atomic_torch(out/'latest.pt', cp)
            atomic_json(out/'validation_000000.json', initial)
        state.update(phase='training', step=start, target=task['updates'])
        for step_index in range(start+1, task['updates']+1):
            if pause.is_set(): raise Pause()
            if shutil.disk_usage(out).free < task.get('disk_min_gib', PROTOCOL['resources']['min_disk_gib'])*2**30:
                raise Pause('disk admission failed')
            tick = time.monotonic(); before = dict(oracle.ledger)
            x = data(degree, task['batch'], 'uniform',
                     ('train-pre' if pre else ('train-pilot' if task['phase']=='pilot' else 'train-post'))+f':u{step_index}', seed)
            if task['method'] == 'erm':
                reference, _ = oracle.cache.lengths(x, oracle.geosteiner)
                selected, selected_ref, selected_cost = x, reference, np.zeros(len(x))
                gain = 0.0; calls = 0
            else:
                result = attack(task['method'], x, oracle, price,
                    derived_seed('scaled-search', degree, seed, step_index),
                    mu=task['mu'], step=task['search_step'], budget=64)
                for key in ('actor_completed', 'rest_completed', 'reference_requests'):
                    if oracle.ledger.get(key,0)-before.get(key,0) != len(x)*64:
                        raise RuntimeError(f'candidate budget mismatch: {key}')
                selected, selected_ref, selected_cost = result['pins'].copy(), result['lstar'].copy(), result['cost'].copy()
                mix = int(round(task.get('nominal_fraction',0)*len(x)))
                if mix:
                    reference, _ = oracle.cache.lengths(x[:mix], oracle.geosteiner)
                    selected[:mix] = x[:mix]; selected_ref[:mix] = reference; selected_cost[:mix] = 0
                gain = float((result['score']-result['original_score']).mean()); calls = len(x)*64
                if step_index == 1 or step_index % interval == 0:
                    atomic_npz(out/'search_audit'/f'u{step_index:06d}.npz', centers=x[:2],
                        support=result['evaluated_pins'][:2], regrets=result['evaluated_regrets'][:2],
                        selected=result['pins'][:2], score=result['score'][:2], cost=result['cost'][:2],
                        original=result['original_score'][:2], price=np.asarray(price))
            torch.manual_seed(derived_seed('scaled-sampled', 'pre' if pre else 'post', degree, seed, step_index))
            if device.type == 'cuda': torch.cuda.manual_seed_all(derived_seed('scaled-sampled','pre' if pre else 'post',degree,seed,step_index))
            losses = sampled_update(actor, critic, optimizer, oracle, selected, selected_ref)
            old_price = price
            if task['method'] != 'erm': price=max(0.0,price+PROTOCOL['lambda_lr']*(float(selected_cost.mean())-rho))
            elapsed = time.monotonic()-tick; total_seconds += elapsed
            row=dict(at=time.time(),step=step_index,seconds=elapsed,train_seconds=total_seconds,
                center_hash=array_hash(x),selected_hash=array_hash(selected),price_before=old_price,price=price,
                rho=rho,cost=float(selected_cost.mean()),budget_gap=float(selected_cost.mean())-rho,
                gain=gain,search_calls=calls,losses=losses,
                ledger_delta={k:oracle.ledger.get(k,0)-before.get(k,0) for k in oracle.ledger})
            append_json(out/'progress.jsonl',row)
            state.update(step=step_index,phase='training',last_progress_at=time.time(),seconds_per_step=elapsed)
            if step_index % interval == 0 or step_index == task['updates']:
                state['phase']='validation'
                val=evaluate(oracle,degree,val_count,split)
                atomic_json(out/f'validation_{step_index:06d}.json',val)
                if best_key(val,initial,pre)<best_key(best,initial,pre):
                    best=val
                    atomic_torch(out/'best.pt',checkpoint(actor,critic,optimizer,step_index,price,task_hash,initial,best,total_seconds))
                state['phase']='training'
            if step_index in (512,2048,4096,8192):
                atomic_torch(out/f'actor_u{step_index:06d}.pt',_cpu_copy(dict(actor_state_dict=actor.state_dict(),step=step_index,task_hash=task_hash)))
            if step_index % cp_every == 0 or step_index == task['updates'] or pause.is_set() or (stop_after and step_index>=stop_after):
                atomic_torch(out/'latest.pt',checkpoint(actor,critic,optimizer,step_index,price,task_hash,initial,best,total_seconds))
            if pause.is_set() or (stop_after and step_index>=stop_after and step_index<task['updates']): raise Pause()
        best_cp=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        receipt=dict(status='TRAINING_COMPLETE',task=task,identity_hash=task_hash,
            best=artifact(out/'best.pt'),latest=artifact(out/'latest.pt'),best_step=best_cp['step'],
            best_validation=best,initial_validation=initial,train_seconds=total_seconds,
            ledger=oracle.ledger,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else 0)
        atomic_json(out/'complete.json',receipt)
        return receipt
    finally: oracle.close()

def screen(task,out,device,state,pause,workers=2):
    _configure(101)
    actor,_,_,meta=load_parent('erm',30,101,device)
    oracle=ExactOracle(actor,device,out/'oracle',workers)
    price,rho=dual_settings(30); results={}; tick=time.monotonic()
    try:
        for dist in DISTS:
            state.update(phase='screen',distribution=dist)
            path=out/f'{dist}.npz'
            if path.exists():
                with np.load(path) as f: results[dist]=dict(gain=float(np.mean(f['score']-f['original_score'])),seconds=float(f['seconds']))
                continue
            if pause.is_set(): raise Pause()
            x=data(30,task.get('count',PROTOCOL['search_validation_count']),dist,task.get('split','search-validation'))
            started=time.monotonic()
            r=attack(task['method'],x,oracle,price,derived_seed('screen',dist),mu=task.get('mu',.02),step=task.get('search_step',.01),budget=task.get('budget',64))
            sec=time.monotonic()-started
            atomic_npz(path,centers=x,**{k:r[k] for k in ['pins','score','regret','cost','lstar','length','original_score']},seconds=np.asarray(sec))
            results[dist]=dict(gain=float(np.mean(r['score']-r['original_score'])),seconds=sec)
        receipt=dict(status='SCREEN_COMPLETE',task=task,results=results,mean_gain=float(np.mean([v['gain'] for v in results.values()])),seconds=time.monotonic()-tick,ledger=oracle.ledger,
            artifacts=[artifact(out/f'{d}.npz') for d in DISTS],parent=meta['best'])
        atomic_json(out/'complete.json',receipt);return receipt
    finally: oracle.close()

def evaluation(task,out,device,state,pause,workers=2):
    binding=task['model']; cp=torch.load(checked(binding),map_location='cpu',weights_only=False)
    actor,_,_=build_initial(task['degree'],task['seed'],device);actor.load_state_dict(cp['actor_state_dict'])
    oracle=ExactOracle(actor,device,out/'oracle',workers)
    artifacts=[]
    try:
        for dist in DISTS:
            for kind,count in [('nominal',task.get('test_count',PROTOCOL['test_count'])),('pressure',task.get('pressure_count',PROTOCOL['pressure_count']))]:
                state.update(phase='evaluation',distribution=dist,kind=kind)
                x=data(task['degree'],count,dist,task.get('data_prefix','')+('test' if kind=='nominal' else 'pressure'))
                for start in range(0,count,128 if kind=='nominal' else 32):
                    path=out/f'{dist}_{kind}_{start:05d}.npz'
                    if path.exists(): artifacts.append(artifact(path));continue
                    if pause.is_set(): raise Pause()
                    chunk=x[start:start+(128 if kind=='nominal' else 32)]
                    if kind=='nominal':
                        r=oracle.evaluate(chunk)
                        payload=dict(pins=chunk,**r)
                    else:
                        r=attack('cem',chunk,oracle,0.0,derived_seed('pressure',task['degree'],dist,start))
                        payload={k:r[k] for k in ['pins','regret','length','lstar','cost','score','original_score']}
                    atomic_npz(path,centers=chunk,model_sha256=np.asarray(binding['sha256']),**payload)
                    artifacts.append(artifact(path))
        receipt=dict(status='EVALUATION_COMPLETE',task=task,artifacts=artifacts,ledger=oracle.ledger)
        atomic_json(out/'complete.json',receipt);return receipt
    finally: oracle.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--physical-gpu-index',type=int,required=True)
    p.add_argument('--stop-after',type=int);p.add_argument('--engineering',action='store_true');args=p.parse_args()
    task=json.loads(Path(args.task).read_text());out=ART/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=None if args.engineering else verify_freeze()
    if (out/'complete.json').exists():
        cp=json.loads((out/'complete.json').read_text())
        if cp['task']!=task:raise RuntimeError('completed task drift')
        for key in ('best','latest'):
            if key in cp:checked(cp[key])
        for a in cp.get('artifacts',[]):checked(a)
        return
    device=torch.device('cuda:0')
    if not torch.cuda.is_available():raise RuntimeError('GPU task cannot silently fall back to CPU')
    torch.cuda.set_per_process_memory_fraction(PROTOCOL['resources']['memory_fraction'])
    state=dict(task=task['id'],pid=os.getpid(),start_ticks=proc_ticks(os.getpid()),physical_gpu_index=args.physical_gpu_index,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),source_freeze=source,phase='loading')
    atomic_json(out/'launch.json',dict(state,at=time.time(),command=sys_argv()))
    pause=threading.Event();stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:pause.set())
    def heartbeat():
        while not stop.is_set():
            atomic_json(out/'heartbeat.json',dict(state,at=time.time()));stop.wait(5)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        fn={'train':train,'screen':screen,'eval':evaluation}[task['kind']]
        kw=dict(workers=PROTOCOL['resources']['reference_workers'])
        if task['kind']=='train':kw['stop_after']=args.stop_after
        result=fn(task,out,device,state,pause,**kw)
        atomic_json(out/'exit.json',dict(status='complete',at=time.time(),pid=os.getpid(),start_ticks=state['start_ticks']))
    except Pause as exc:
        atomic_json(out/'exit.json',dict(status='paused',reason=str(exc),at=time.time(),pid=os.getpid(),start_ticks=state['start_ticks']))
        raise SystemExit(75)
    except BaseException as exc:
        atomic_json(out/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),state=state))
        raise
    finally:stop.set();thread.join(timeout=10)

def sys_argv():
    import sys
    return sys.argv

if __name__=='__main__':main()

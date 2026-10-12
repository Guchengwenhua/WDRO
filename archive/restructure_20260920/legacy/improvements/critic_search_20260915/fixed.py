"""Recoverable fixed-Actor validation and independent seven-direction study."""
import argparse, fcntl, json, os, threading, time
from pathlib import Path
import bootstrap
import numpy as np
import torch
from adv_v3.common import (Oracle, load_parent, parent_price, make_data, derived_seed,
    atomic_json, atomic_npz, sha256, EXP as OLD_EXP)
from adv_v3.critic import InputCritic
from search import configurations, FAMILIES, propose, evaluate_candidates

DISTS=('uniform','mix_gauss','edge_bias','grid_cluster')

def load_models(seed, device):
    actor,parent,cp,meta=load_parent('erm',30,seed,device)
    path=OLD_EXP/'fixed'/f'erm_d30_s{seed}'/'critic'/'best.pt'
    model=InputCritic(parent,calibrated=True).to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model_state_dict'])
    return actor,parent,model.eval(),dict(actor=meta['best'],surrogate=dict(path=str(path),sha256=sha256(path)))

def run(args):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.10)
    out=bootstrap.EXP/'fixed'/f'{args.phase}_s{args.seed}';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'task.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();stop=threading.Event();state=dict(phase=args.phase,seed=args.seed,pid=os.getpid(),physical_gpu=args.physical_gpu_index)
    def hb():atomic_json(out/'heartbeat.json',dict(state,at=time.time()))
    def loop():
        while not stop.wait(10):hb()
    threading.Thread(target=loop,daemon=True).start()
    atomic_json(out/'launch.json',dict(state,start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],at=time.time(),argv=os.sys.argv))
    oracle=None
    try:
        actor,parent,base,bindings=load_models(args.seed,device)
        price,rho=parent_price(30,args.seed)
        identity=dict(bindings=bindings,price=price,rho=rho,phase=args.phase,seed=args.seed,
            protocol_sha256=sha256(bootstrap.ROOT/'protocol.json'),
            source={p.name:sha256(p) for p in [bootstrap.ROOT/'fixed.py',bootstrap.ROOT/'search.py',bootstrap.ROOT/'surrogate.py']})
        if args.phase=='test':identity['selection_sha256']=sha256(bootstrap.EXP/'selection.json')
        ipath=out/'identity.json'
        if ipath.exists() and json.loads(ipath.read_text())!=identity:raise RuntimeError('fixed task identity drift')
        atomic_json(ipath,identity)
        if (out/'complete.json').exists():
            done=json.loads((out/'complete.json').read_text())
            if all(sha256(a['path'])==a['sha256'] for a in done['artifacts']):return
            raise RuntimeError('completed artifact drift')
        oracle=Oracle(actor,device,out/'oracle')
        configs=configurations();models={'old':base}
        if args.phase=='validation':
            methods={name:dict(config=cfg,model='old') for name,cfg in configs.items()};count=32
        elif args.phase=='test':
            selection=json.loads((bootstrap.EXP/'selection.json').read_text())
            methods={n:dict(config=configs[n],model='old') for n in ['baseline','no_move','random_direction']}
            for family,name in selection['selected'].items():methods[family]=dict(config=configs[name],model='old',selected_name=name)
            state['stage']='surrogate_refresh';hb()
            from surrogate import train_variants
            variants=train_variants(oracle,parent,base,args.seed,out/'surrogates',degree=30,price=price,heartbeat=hb)
            models.update(variants['models'])
            for name in ['uniform_mse','region_mse','region_huber','region_rankdiff']:
                methods[name]=dict(config=configs['baseline'],model=name)
            count=64
        else:
            methods={n:dict(config=configs[n],model='old') for n in ['baseline','no_move','random_direction','differentiable_normalization','trajectory21_8']};count=2
        artifacts=[ipath];summaries=[]
        for dist in DISTS if args.phase!='smoke' else ['uniform']:
            x=make_data(30,count,dist,'critic-search-20260915-'+args.phase,seed=0)
            inputpath=out/'inputs'/f'{dist}.npz'
            if not inputpath.exists():atomic_npz(inputpath,pins=x)
            else:
                with np.load(inputpath) as z:np.testing.assert_array_equal(z['pins'],x)
            artifacts.append(inputpath)
            for name,spec in methods.items():
                values=[];times=[]
                for offset in range(0,count,16):
                    state.update(stage='attack',distribution=dist,method=name,offset=offset);hb()
                    path=out/'evaluation'/f'{dist}_{name}_{offset:04d}.npz'
                    if not path.exists():
                        chunk=x[offset:offset+16];before=dict(oracle.ledger);tick=time.monotonic()
                        proposals,diag=propose(models[spec['model']],chunk,price,
                            derived_seed('critic-search-shared-starts',args.seed,dist,args.phase,offset),spec['config'])
                        result=evaluate_candidates(chunk,proposals,oracle,price)
                        for field in ['actor_requested','actor_completed','rest_requested','rest_completed','reference_requests']:
                            if oracle.ledger.get(field,0)-before.get(field,0)!=64*len(chunk):raise AssertionError(field)
                        result.update(diagnostics_json=np.asarray(json.dumps(diag)),elapsed_seconds=np.asarray(time.monotonic()-tick),
                            ledger_json=np.asarray(json.dumps({k:oracle.ledger[k]-before.get(k,0) for k in oracle.ledger})))
                        baseline=out/'evaluation'/f'{dist}_baseline_{offset:04d}.npz'
                        if baseline.exists():
                            with np.load(baseline) as old:np.testing.assert_array_equal(result['original_regret'],old['original_regret'])
                        atomic_npz(path,**result)
                    artifacts.append(path)
                    with np.load(path) as z:values.extend(z['score']);times.append(float(z['elapsed_seconds']))
                summaries.append(dict(distribution=dist,method=name,score=float(np.mean(values)),seconds=sum(times),spec=spec))
                print(json.dumps(summaries[-1]),flush=True)
        atomic_json(out/'summary.json',summaries);artifacts.append(out/'summary.json')
        if args.phase=='validation':
            means={name:float(np.mean([r['score'] for r in summaries if r['method']==name])) for name in methods}
            selected={fam:max(names,key=lambda n:(means[n],-list(configs).index(n))) for fam,names in FAMILIES.items()}
            atomic_json(bootstrap.EXP/'selection.json',dict(selected=selected,validation_macro_scores=means,
                rule='largest four-distribution mean true penalized score on seed101 validation; stable config-order tie-break',
                source_identity=identity,frozen_at=time.time()))
            artifacts.append(bootstrap.EXP/'selection.json')
        oracle.persist();oracle.close();oracle=None
        artifacts.extend(p for p in (out/'surrogates').rglob('*') if p.is_file() and p.suffix in ('.pt','.json','.npz'))
        artifacts.append(out/'oracle'/'oracle_ledger.json')
        atomic_json(out/'complete.json',dict(status='pass',phase=args.phase,seed=args.seed,count_per_distribution=count,
            elapsed_seconds=time.monotonic()-started,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            artifacts=[dict(path=str(p),sha256=sha256(p)) for p in artifacts],identity=identity))
    except Exception as e:
        atomic_json(out/'failure.json',dict(state,error=repr(e),at=time.time()));raise
    finally:
        if oracle is not None:oracle.close()
        stop.set();hb();atomic_json(out/'exit.json',dict(state,at=time.time(),complete=(out/'complete.json').exists()))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','validation','test'],required=True)
    p.add_argument('--seed',type=int,default=101);p.add_argument('--physical-gpu-index',type=int,required=True)
    run(p.parse_args())

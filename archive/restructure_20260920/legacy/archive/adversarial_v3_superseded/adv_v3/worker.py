"""One recoverable fixed-opponent experiment task."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import threading
import time
from pathlib import Path
import numpy as np
import torch
from .common import (
    ROOT, EXP, Oracle, atomic_json, atomic_npz, sha256, make_data,
    load_parent, parent_price, derived_seed, DISTRIBUTIONS, GEOMETRY_LEDGER, parent_model,
)

METHODS = [
    "random_top4","critic_raw4","critic_calibrated4","rl4",
    "random64","cem64","es64","critic_calibrated64","rl64",
]

def heartbeat_thread(path, stop, state):
    while not stop.wait(5):
        atomic_json(path,dict(state,at=time.time(),pid=os.getpid()))

def run(args):
    from .rl import train_policy, propose as rl_propose
    from .critic import InputCritic, train_surrogate, propose as critic_propose
    from .search import attack
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ident=f"{args.parent}_d{args.degree}_s{args.seed}"
    out=EXP/("smoke" if args.smoke else "fixed")/ident
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/"task.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(parent=parent_model(args.parent,args.degree,args.seed)["best"],
        price_parent=parent_model("wdro_rest",args.degree,args.seed)["best"],smoke=args.smoke,
        protocol_sha256=sha256(ROOT/"protocol.json"),
        parent_snapshot_sha256=sha256(ROOT/"manifests/parent_snapshot.json"),
        source_hashes={name:sha256(ROOT/"adv_v3"/name) for name in
            ["common.py","worker.py","search.py","critic.py","rl.py"]})
    identity_path=out/"identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=identity: raise RuntimeError("Task identity drift")
    else: atomic_json(identity_path,identity)
    if (out/"complete.json").exists():
        done=json.loads((out/"complete.json").read_text())
        if all(sha256(a["path"])==a["sha256"] for a in done["artifacts"]):
            print(json.dumps({"status":"already_complete","task":ident}));return
        raise RuntimeError("Completed task artifact changed")
    state={"task":ident,"physical_gpu_index":args.physical_gpu_index,"phase":"load","smoke":args.smoke,
        "reference_workers":int(os.environ.get("ADV_V3_REFERENCE_WORKERS","1"))}
    atomic_json(out/"launch.json",dict(state,pid=os.getpid(),at=time.time(),command=list(os.sys.argv),
        process_start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]))
    stop=threading.Event()
    thread=threading.Thread(target=heartbeat_thread,args=(out/"heartbeat.json",stop,state),daemon=True)
    thread.start(); started=time.monotonic()
    oracle=None
    try:
        actor,parent_critic,checkpoint,meta=load_parent(args.parent,args.degree,args.seed,device)
        price,rho=parent_price(args.degree,args.seed)
        oracle=Oracle(actor,device,out/"oracle",microbatch=256)
        def hb(*a,**k):
            atomic_json(out/"heartbeat.json",dict(state,at=time.time(),pid=os.getpid()))
        rlcfg={} if not args.smoke else dict(updates=2,batch_size=8,validation_interval=1,validation_count=8)
        criticcfg={} if not args.smoke else dict(train_count=32,validation_count=16,epochs=2,batch_size=8)
        for cfg in [rlcfg,criticcfg]:
            cfg.update(parent_checkpoint_sha256=meta["best"]["sha256"],
                       protocol_sha256=identity["protocol_sha256"])
        state["phase"]="critic_training"; hb()
        start=time.monotonic()
        surrogate,cs=train_surrogate(oracle,parent_critic,args.degree,
            derived_seed("critic",args.degree,args.parent,args.seed),out/"critic",criticcfg,heartbeat=hb)
        critic_seconds=time.monotonic()-start
        state["phase"]="rl_training"; hb()
        start=time.monotonic()
        policy,rs=train_policy(oracle,args.degree,
            derived_seed("rl",args.degree,args.parent,args.seed),price,out/"rl",rlcfg,heartbeat=hb)
        rl_seconds=time.monotonic()-start
        raw=InputCritic(parent_critic,calibrated=False).to(device).eval()
        state["phase"]="evaluation"; artifacts=[dict(path=str(identity_path),sha256=sha256(identity_path))]
        count=8 if args.smoke else 256
        for distribution in DISTRIBUTIONS:
            centers=make_data(args.degree,count,distribution,
                "smoke-evaluation" if args.smoke else "fixed-evaluation")
            data_path=out/"inputs"/f"{distribution}.npz"
            if not data_path.exists(): atomic_npz(data_path,pins=centers)
            else:
                with np.load(data_path) as existing: np.testing.assert_array_equal(existing["pins"],centers)
            artifacts.append(dict(path=str(data_path),sha256=sha256(data_path)))
            for method in METHODS:
                for start_i in range(0,count,64):
                    path=out/"evaluation"/f"{distribution}_{method}_{start_i:05d}.npz"
                    if not path.exists():
                        x=centers[start_i:start_i+64]
                        proposer=None
                        if method.startswith("critic"):
                            model=raw if method=="critic_raw4" else surrogate
                            proposer=lambda c,p,k,s,model=model: critic_propose(model,c,p,k,s)
                        elif method.startswith("rl"):
                            proposer=lambda c,p,k,s: rl_propose(policy,c,p,k,s)
                        state.update(distribution=distribution,searcher=method,offset=start_i); hb()
                        before=dict(oracle.ledger); before_geometry=dict(GEOMETRY_LEDGER); tick=time.monotonic()
                        result=attack(method,x,oracle,price,
                            derived_seed("attack",args.degree,args.seed,distribution,start_i),proposer=proposer)
                        elapsed=time.monotonic()-tick
                        b=4 if method in METHODS[:4] else 64
                        assert result["calls_per_center"]==b
                        assert np.all(result["score"]>=result["original_score"]-1e-9)
                        for key in ["actor_requested","actor_completed","rest_requested","rest_completed","reference_requests"]:
                            assert oracle.ledger.get(key,0)-before.get(key,0)==len(x)*b,(key,method)
                        payload={k:result[k] for k in [
                            "pins","regret","lstar","length","cost","score","original_regret","original_score"
                        ]}
                        payload.update(centers=x,price=np.full(len(x),price),rho=np.full(len(x),rho),
                            calls=np.full(len(x),b,dtype=np.int32),elapsed_seconds=np.asarray(elapsed),
                            ledger_delta_json=np.asarray(json.dumps({k:oracle.ledger.get(k,0)-before.get(k,0) for k in oracle.ledger})),
                            geometry_delta_json=np.asarray(json.dumps({k:GEOMETRY_LEDGER[k]-before_geometry[k] for k in GEOMETRY_LEDGER})),
                            diagnostics_json=np.asarray(json.dumps(result.get("diagnostics",{}),default=lambda o:o.tolist() if hasattr(o,"tolist") else str(o))))
                        atomic_npz(path,**payload)
                    artifacts.append(dict(path=str(path),sha256=sha256(path)))
        oracle.persist()
        for folder in ["rl","critic"]:
            for path in sorted((out/folder).rglob("*")):
                if path.is_file() and path.suffix in {".pt",".json",".jsonl"}:
                    artifacts.append(dict(path=str(path),sha256=sha256(path)))
        result=dict(status="pass",task=ident,smoke=args.smoke,degree=args.degree,parent=args.parent,seed=args.seed,
            parent_checkpoint=meta["best"],price=price,rho=rho,evaluation_centers=count*4,
            methods=METHODS,critic_summary=cs,rl_summary=rs,critic_seconds=critic_seconds,rl_seconds=rl_seconds,
            task_seconds=time.monotonic()-started,ledger=oracle.ledger,geometry=GEOMETRY_LEDGER,
            identity=identity,artifacts=artifacts,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device.type=="cuda" else 0)
        atomic_json(out/"complete.json",result)
        print(json.dumps({k:result[k] for k in ["status","task","task_seconds","peak_reserved_mib"]}))
    except Exception as e:
        atomic_json(out/"failure.json",dict(state,error=repr(e),at=time.time()))
        raise
    finally:
        if oracle is not None: oracle.close()
        stop.set();thread.join(timeout=6)
        atomic_json(out/"exit.json",dict(state,pid=os.getpid(),at=time.time(),complete=(out/"complete.json").exists()))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--parent",choices=["erm","wdro_rest"],required=True)
    p.add_argument("--degree",type=int,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--physical-gpu-index",type=int,required=True)
    p.add_argument("--smoke",action="store_true")
    run(p.parse_args())

if __name__=="__main__": main()

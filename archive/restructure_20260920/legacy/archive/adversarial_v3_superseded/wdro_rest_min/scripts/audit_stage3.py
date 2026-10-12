#!/usr/bin/env python3
"""Resumable stage-3 data, selection, budget and checkpoint acceptance audit.

Only coordinates/hashes of held-out data are read; no test/ISPD predictions.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / "experiments"
sys.path.insert(0, str(ROOT))
from src.baseline_data import method_training_batch, update_group_weights
from src.candidates import generate_support, shortlist_support
from src.datasets import load_dataset, array_bundle_sha256
from src.geometry import pin_set_sha256
from src.io_utils import atomic_write_json, append_jsonl
from src.job_control import JobLease, utc_now
from src.protocol import load_protocol, protocol_sha256
from src.rest import file_sha256
from src.stage2_audit_checks import all_finite, optimizer_checks, validation_checks
from src.trainer import canonical_sha256, load_trainer_config
from src.training_data import EMPTY_STREAM_SHA256, extend_stream_hash
from src.validation import validation_manifest_hash
from run_stage3_baseline_queue import build_tasks

AUDIT = EXP / "logs/stage3_audit"


def effective_events(path):
    """Keep the committed trajectory; retain the complete raw replay log on disk."""
    train, validation = {}, {}
    raw = []
    for line in Path(path).read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # An interrupted last JSONL line is not a committed result.
        raw.append(event)
        if event.get("event") == "resume":
            cursor = event["from_update"]
            train = {u:e for u,e in train.items() if u <= cursor}
            validation = {u:e for u,e in validation.items() if u <= cursor}
        if event.get("event") == "train":
            train[event["update"]] = event
        elif event.get("event") == "validation":
            validation[event["update"]] = event
    return train, validation, raw


def rebuild_inputs(task, result, train, *, smoke=False):
    protocol = load_protocol()
    config = json.loads((ROOT / "configs/stage3_baselines_v1.json").read_text())
    heldout, datasets = set(), {}
    for split in ("validation", "test"):
        n = protocol["data"]["sizes_per_degree_distribution"][split]
        seed = protocol["data"]["seeds"][split]
        for group in protocol["data"]["distributions"]:
            path=EXP / "cache/datasets/protocol_v1" / split / f"{group}_d{task['degree']}_s{seed}_n{n}.npz"
            arrays, meta = load_dataset(path, verify=True)
            heldout.update(pin_set_sha256(pins) for pins in arrays["pins"])
            datasets[str(path)] = meta["npz_sha256"]
    target = AUDIT / "cursors" / f"{task['run_id']}.json"
    binding={"identity": result["run_identity"], "expected_stream": result["training_stream_hash"],
             "heldout_files": datasets, "auditor_sha256":file_sha256(Path(__file__))}
    state={"binding":binding,"completed_updates":0,"stream_hash":EMPTY_STREAM_SHA256,
           "support_stream_hash":EMPTY_STREAM_SHA256,"selection_stream_hash":EMPTY_STREAM_SHA256,
           "heldout_intersections":0,"nets_checked":0,"candidate_nets_checked":0,"status":"running"}
    if target.exists():
        state=json.loads(target.read_text())
        if state["binding"] != binding:
            raise ValueError(f"audit cursor identity changed: {target}")
    updates=result["completed_updates"]
    for update in range(state["completed_updates"]+1, updates+1):
        batch=method_training_batch(task["method"],protocol=protocol,degree=task["degree"],
            training_seed=task["seed"],update=update,batch_size=256)
        event=train[update]
        if batch.bundle_sha256 != event["batch_bundle_sha256"]:
            raise RuntimeError(f"training pool mismatch: {task['run_id']} update {update}")
        if any(not n.decode().startswith(f"train_u{update:06d}:") for n in batch.net_ids):
            raise RuntimeError("training ID namespace violation")
        state["heldout_intersections"]+=sum(pin_set_sha256(net) in heldout for net in batch.pins)
        state["nets_checked"]+=len(batch.pins)
        state["stream_hash"]=extend_stream_hash(state["stream_hash"],batch.bundle_sha256)
        if task["method"] == "hard_mining":
            support=generate_support(batch,training_seed=task["seed"],update=update,search=config["search"])
            detail=event["method_detail"]
            if support.sha256 != detail["support_sha256"]:
                raise RuntimeError("candidate support mismatch")
            indices,_=shortlist_support(support)
            if array_bundle_sha256({"indices":indices}) != detail["shortlist_sha256"]:
                raise RuntimeError("exact shortlist mismatch")
            chosen=np.asarray(detail["selected_indices"],dtype=np.int64)
            if not np.all(np.any(indices==chosen[:,None],axis=1)):
                raise RuntimeError("selected candidate outside the exact shortlist")
            if "exact_relative_regrets" in detail:
                scores=np.asarray(detail["exact_relative_regrets"])
                if not np.array_equal(indices, np.asarray(detail["exact_shortlist_indices"])) or not np.array_equal(chosen,indices[np.arange(256),scores.argmax(axis=1)]):
                    raise RuntimeError("Hard Mining did not select per-center highest true regret")
            pins=support.pins[np.arange(256),chosen]
            ids=np.asarray([f"{n.decode()}:candidate{int(j)}:block{support.refresh_block}".encode() for n,j in zip(batch.net_ids,chosen)])
            selected_hash=array_bundle_sha256({"pins":pins,"net_ids":ids})
            if selected_hash != detail["selected_bundle_sha256"]:
                raise RuntimeError("selected training input mismatch")
            # Check every legal candidate, including candidates not selected.
            state["heldout_intersections"]+=sum(pin_set_sha256(net) in heldout for net in support.pins.reshape(-1,task["degree"],2))
            state["candidate_nets_checked"]+=int(np.prod(support.pins.shape[:2]))
            state["support_stream_hash"]=extend_stream_hash(state["support_stream_hash"],support.sha256)
            state["selection_stream_hash"]=extend_stream_hash(state["selection_stream_hash"],selected_hash)
        if state["heldout_intersections"]:
            state["status"]="fail"
            atomic_write_json(target,state)
            raise RuntimeError("training or search candidate overlaps held-out coordinates")
        if state["stream_hash"] != event["training_stream_hash"]:
            raise RuntimeError("per-update training stream differs")
        state["completed_updates"]=update
        if update % 25 == 0 or update == updates:
            state["updated_at_utc"]=utc_now()
            atomic_write_json(target,state)
    if state["stream_hash"] != result["training_stream_hash"]:
        raise RuntimeError("final training stream differs")
    if task["method"] == "hard_mining":
        for key in ("support_stream_hash","selection_stream_hash"):
            if state[key] != result["method_state"][key]:
                raise RuntimeError(f"final {key} differs")
    state["status"]="pass"
    atomic_write_json(target,state)
    return state


def audit_run(task, smoke=False):
    protocol=load_protocol(); trainer=load_trainer_config()
    config=json.loads((ROOT / "configs/stage3_baselines_v1.json").read_text())
    run_id=task["run_id"]
    logs=EXP / "logs/stage3" / run_id
    result=json.loads((logs/"result.json").read_text())
    latest=torch.load(result["latest_checkpoint"],map_location="cpu",weights_only=False)
    best=torch.load(result["best_checkpoint"],map_location="cpu",weights_only=False)
    train, validation, raw=effective_events(logs/"events.jsonl")
    updates=result["completed_updates"]
    if not smoke and updates!=2000:
        raise ValueError("formal audit requires 2000 completed updates")
    if sorted(train)!=list(range(1,updates+1)):
        raise RuntimeError("missing train events")
    stream=rebuild_inputs(task,result,train,smoke=smoke)
    nominal_limit=result["nominal_limit"]
    erm_path=EXP/"logs/stage2"/f"formal_erm_d{task['degree']}_s{task['seed']}_u2000"/"result.json"
    erm=json.loads(erm_path.read_text())
    expected_nominal_limit=erm["best"]["nominal_mean"]+protocol["training"]["selection"]["nominal_mean_regret_degradation_max_absolute"]
    _,erm_validation,_=effective_events(erm["event_log"])
    feasible=[e for u,e in sorted(validation.items()) if e["result"]["selection"]["nominal_mean"]<=nominal_limit]
    if not feasible:
        raise RuntimeError("no nominal-feasible checkpoint; keep all outputs and report failure")
    selected=min(feasible,key=lambda e:e["result"]["selection"]["worst_group_mean"])
    expected={"actor_forward_nets":updates*256+len(validation)*8000+1,
        "exact_rest_evaluator_nets":updates*256+len(validation)*8000+2,
        "surrogate_evaluator_nets":0,"geosteiner_nets":updates*256+2}
    if task["method"]=="hard_mining":
        expected["actor_forward_nets"]+=updates*256*4
        expected["exact_rest_evaluator_nets"]+=updates*256*4
        expected["surrogate_evaluator_nets"]=updates*256*33
        expected["geosteiner_nets"]=2+sum(e["method_detail"]["reference_cache_misses"] for e in train.values())
    source_files=[*sorted((ROOT/"src").glob("*.py")),ROOT/"scripts/train_baseline.py"]
    source_hash=canonical_sha256({str(p.relative_to(ROOT)):file_sha256(p) for p in source_files})
    checks={
        "complete_updates":updates==(12 if smoke else 2000) and latest["update"]==updates,
        "frozen_protocol":latest["run_identity"]["protocol_canonical_sha256"]==protocol_sha256(protocol),
        "frozen_trainer":latest["run_identity"]["trainer_config_sha256"]==canonical_sha256(trainer),
        "implementation_source":latest["run_identity"]["implementation_sha256"]==source_hash,
        "baseline_configuration":latest["run_identity"]["baseline_config_sha256"]==canonical_sha256(config),
        "schema_and_identity":latest["run_identity"]==best["run_identity"]==result["run_identity"] and result["batch_size"]==256,
        "exact_call_ledger":latest["call_ledger"]==result["call_ledger"]==expected,
        "full_training_and_candidate_stream_rebuilt":stream["status"]=="pass" and stream["completed_updates"]==updates,
        "no_heldout_content_overlap":stream["heldout_intersections"]==0,
        "no_nan_or_inf":all_finite(latest) and all_finite(best) and all_finite(list(train.values())) and all_finite(list(validation.values())),
        "best_earliest_feasible_validation_minimum":best["update"]==selected["update"] and result["best"]==latest["best"]==best["best"] and result["best"]["selection_objective"]==selected["result"]["selection"]["worst_group_mean"],
        "nominal_guard":result["best"]["nominal_mean"]<=nominal_limit and result["selection_feasible"],
        "nominal_guard_independent_erm_reference":nominal_limit==expected_nominal_limit and latest["run_identity"]["erm_reference_result_sha256"]==file_sha256(erm_path),
        "same_dac21_initial_actor_and_validation":result["initialization"]["sha256"]==erm["initialization"]["sha256"] and validation[0]["result"]==erm_validation[0]["result"],
        "validation_hash_matches_best":result["best"]["validation_manifest_sha256"]==validation_manifest_hash(selected["result"]),
        "checkpoint_hashes":file_sha256(result["latest_checkpoint"])==result["latest_checkpoint_sha256"] and file_sha256(result["best_checkpoint"])==result["best_checkpoint_sha256"],
        "inverse_length":result["inverse_length_audit"]["status"]=="pass",
        "test_ispd_metrics_locked":result["test_or_ispd_used"] is False and all(e["result"]["split"]=="validation" and not e["result"]["selection"]["test_or_ispd_used"] for e in validation.values()),
        "method_state_checkpointed":latest["method_state"]==result["method_state"],
        "only_latest_best":sorted(p.name for p in Path(result["latest_checkpoint"]).parent.glob("*.pt"))==["best.pt","latest.pt"],
        "no_checkpoint_partial":not list(Path(result["latest_checkpoint"]).parent.glob("*.partial")),
        "run_lock_released":not (EXP/"logs/jobs"/run_id/"run.lock").exists(),
    }
    config_for_validation=json.loads(json.dumps(protocol))
    config_for_validation["training"]["policy_updates"]=0 if smoke else 2000
    vc=validation_checks(list(validation.values()),best,config_for_validation,trainer,task["degree"])
    vc.pop("selected_best_is_earliest_global_validation_minimum")
    checks.update(vc)
    checks.update({"latest_"+k:v for k,v in optimizer_checks(latest,trainer).items()})
    if best["update"]:
        checks.update({"best_"+k:v for k,v in optimizer_checks(best,trainer).items()})
    if task["method"] in ("mixed_erm","group_dro"):
        checks["equal_group_sampling"]=all(e["method_detail"]["group_counts"]==[64]*4 for e in train.values())
    if task["method"]=="group_dro":
        log_q=np.full(4,-np.log(4))
        for _,event in sorted(train.items()):
            log_q,q=update_group_weights(log_q,event["method_detail"]["group_losses"],config["group_dro"]["eta"])
            if not np.array_equal(q,np.asarray(event["method_detail"]["group_weights"])):
                raise RuntimeError("GroupDRO exponential weights disagree")
        checks["group_weights_rebuilt"]=np.array_equal(log_q,np.asarray(latest["method_state"]["group_log_weights"]))
    if task["method"]=="hard_mining":
        budget=result["method_state"]["budget"]
        checks["shared_wdro_search_budget"]=budget["centers"]==updates*256 and budget["perturbations"]==updates*256*32 and budget["surrogate_nets"]==updates*256*33 and budget["exact_rescore_nets"]==budget["reference_requests"]==updates*256*4
        checks["cache_requests_balance"]=budget["reference_cache_hits"]+budget["reference_cache_misses"]==budget["reference_requests"]
    output={"run_id":run_id,"status":"pass" if all(checks.values()) else "fail","checks":checks,
        "completed_at_utc":utc_now(),"result_sha256":file_sha256(logs/"result.json"),
        "stream_cursor":str(AUDIT/"cursors"/f"{run_id}.json"),"training_nets_checked":stream["nets_checked"],
        "candidate_nets_checked":stream["candidate_nets_checked"]}
    atomic_write_json(AUDIT/"runs"/f"{run_id}.json",output)
    if output["status"]!="pass":
        raise RuntimeError(f"{run_id}: {[k for k,v in checks.items() if not v]}")
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--smoke-run")
    args=parser.parse_args()
    tasks=build_tasks()
    if args.smoke_run:
        result=json.loads((EXP/"logs/stage3"/args.smoke_run/"result.json").read_text())
        tasks=[{"run_id":args.smoke_run,"method":result["method"],"degree":result["degree"],"seed":result["training_seed"]}]
    run_id="stage3_audit_smoke" if args.smoke_run else "stage3_completion_audit"
    command=" ".join([sys.executable,str(Path(__file__)),*sys.argv[1:]])
    with JobLease(AUDIT/run_id,run_id=run_id,exact_command=command,resume_command=command,
        monitor_command=f"cat {AUDIT/run_id/'heartbeat.json'}",expected_updates=len(tasks),
        physical_gpu_index=None,host_runtime_manifest=os.environ.get("WDRO_HOST_RUNTIME_MANIFEST")) as lease:
        outputs=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(audit_run,t,bool(args.smoke_run)):t for t in tasks}
            pending=set(futures)
            while pending:
                done,pending=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    output=future.result();outputs.append(output)
                    print(json.dumps({"completed":len(outputs),"total":len(tasks),"run_id":output["run_id"],"status":output["status"]}),flush=True)
                lease.update(update=len(outputs),status="running",detail={"remaining":len(pending),"input_rebuild_cursors":str(AUDIT/"cursors")})
        report={"status":"pass","completed_at_utc":utc_now(),"runs":outputs,
            "test_ispd_models_evaluated":False,"all_inputs_rebuilt":True,
            "checks_passed":sum(len(o["checks"]) for o in outputs)}
        path=AUDIT/("smoke_audit.json" if args.smoke_run else "stage3_formal_audit.json")
        atomic_write_json(path,report)
        lease.finish(update=len(outputs),status="complete",detail={"report":str(path)})


if __name__=="__main__":
    main()

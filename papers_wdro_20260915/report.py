"""Independent artifact, trajectory, LP and paired-statistics acceptance.

This module does not import runner/core/model code. It never starts a model or
GPU process. --partial writes progress only, never a completion certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent
EMPTY_CHAIN = hashlib.sha256(b"").hexdigest()
REQUIRED_ARTIFACTS = {"initial.pt", "latest.pt", "best.pt", "test_best.npz", "test_initial.npz", "attack.npz", "metrics.json", "run.json"}
NONCOMPARATIVE = {"legal", "edge_count", "lstar", "policy_reward", "raw_policy_reward", "phase_mean"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    with temp.open("w") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def near(actual, expected, message, rtol=2e-6, atol=2e-7):
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), message)


def stat_seed(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def paired_bootstrap(before, after, repetitions=2000, confidence=.95, seed=0):
    """Paired nets; [seed, net] inputs preserve common nets across trained models.

    For multiple seeds, estimates are the mean of per-seed means/P95s. The net
    CI is conditional on these trained models; seed-only CIs are separate.
    """
    a, b = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if a.ndim == 1:
        a, b = a[None, :], b[None, :]
    require(a.shape == b.shape and a.ndim == 2 and a.shape[1] > 0, "paired bootstrap shape mismatch")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite bootstrap input")
    rng = np.random.default_rng(seed)
    seeds, n = a.shape
    mean_seeds = (b - a).mean(1)
    p95_seeds = np.quantile(b, .95, axis=1) - np.quantile(a, .95, axis=1)
    boot_mean, boot_p95 = [], []
    for start in range(0, repetitions, 64):
        indices = rng.integers(0, n, size=(min(64, repetitions-start), n))
        aa, bb = a[:, indices], b[:, indices]
        boot_mean.extend((bb-aa).mean(axis=(0, 2)).tolist())
        boot_p95.extend((np.quantile(bb, .95, axis=2)-np.quantile(aa, .95, axis=2)).mean(axis=0).tolist())
    tails = [(1-confidence)/2, (1+confidence)/2]
    result = {"mean_delta": float(mean_seeds.mean()), "p95_delta": float(p95_seeds.mean()),
              "mean_net_ci_low": float(np.quantile(boot_mean,tails[0])), "mean_net_ci_high": float(np.quantile(boot_mean,tails[1])),
              "p95_net_ci_low": float(np.quantile(boot_p95,tails[0])), "p95_net_ci_high": float(np.quantile(boot_p95,tails[1])),
              "n_nets": n, "n_training_seeds": seeds, "bootstrap_repetitions": repetitions,
              "mean_negative_seeds": int((mean_seeds<0).sum()), "p95_negative_seeds": int((p95_seeds<0).sum())}
    if seeds > 1:
        indices = rng.integers(0,seeds,size=(repetitions,seeds))
        for label, values in (("mean",mean_seeds),("p95",p95_seeds)):
            ci = np.quantile(values[indices].mean(1),tails)
            result[label+"_seed_ci_low"], result[label+"_seed_ci_high"] = map(float,ci)
    return result


def independent_lp(loss, cost, rho, witness=None):
    """Rebuild the transportation LP independently using a sparse COO matrix."""
    loss, cost = np.asarray(loss,dtype=np.float64),np.asarray(cost,dtype=np.float64)
    require(loss.shape == cost.shape and loss.ndim == 2, "LP shape mismatch")
    require(np.isfinite(loss).all() and np.isfinite(cost).all(), "LP nonfinite array")
    require(cost.min() >= -1e-12 and np.max(np.abs(cost[:,0])) < 1e-12 and rho>=0, "LP cost or center invalid")
    rows, columns = loss.shape
    equality = coo_matrix((np.ones(rows*columns),(np.repeat(np.arange(rows),columns),np.arange(rows*columns))),shape=(rows,rows*columns)).tocsr()
    result = linprog(-loss.reshape(-1), A_ub=cost.reshape(1,-1), b_ub=[rho*rows],
                     A_eq=equality,b_eq=np.ones(rows),bounds=(0,None),method="highs-ds",options={"threads":1})
    require(result.success,"Independent LP failed: "+result.message)
    risk = float(-result.fun/rows)
    record = {"risk":risk,"nominal":float(loss[:,0].mean()),"unconstrained_max":float(loss.max(1).mean())}
    require(record["nominal"]-1e-7 <= risk <= record["unconstrained_max"]+1e-7,"LP objective bounds violated")
    if witness is not None:
        mass=np.asarray(witness,dtype=np.float64)
        require(mass.shape==loss.shape and np.isfinite(mass).all(),"Invalid stored LP mass")
        row_error=float(np.max(np.abs(mass.sum(1)-1)))
        average_cost=float(np.sum(mass*cost)/rows)
        saved_risk=float(np.sum(mass*loss)/rows)
        require(row_error <= 1e-7 and mass.min() >= -1e-8 and average_cost<=rho+1e-7,"Stored LP witness is infeasible")
        near(saved_risk,risk,"Stored LP witness is not independently optimal",rtol=1e-7,atol=1e-7)
        record.update(average_cost=average_cost,row_error=row_error,witness_risk=saved_risk)
    return record


def state_digest(state):
    h=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def audit_frozen(root,p):
    path=root/"artifacts/frozen.json"
    value=read(path)
    require(isinstance(value.get("files"),dict) and value["files"],"empty frozen source manifest")
    resolved={str(Path(name).resolve()):expected for name,expected in value["files"].items()}
    for name in ["protocol.json","runner.py","core.py","adapter_rest.py","adapter_timing.py","adapter_hub.py","adapter_buffer.py","report.py"]:
        require(str((root/name).resolve()) in resolved,"Required source missing from freeze: "+name)
    for name,expected in resolved.items():
        require(Path(name).is_file() and sha(name)==expected,"Frozen file drift: "+name)
    data=read(root/"artifacts/data/manifest.json")
    require(data.get("status")=="complete","data preparation incomplete")
    for name,expected in data["files"].items():
        require(sha(name)==expected,"Data manifest hash mismatch: "+name)
        require(resolved.get(str(Path(name).resolve()))==expected,"Data file not bound by source freeze: "+name)
    for split,key in (("test","test_count"),("attack","attack_count"),("validation","validation_count")):
        for dist in p["distributions"]:
            name=root/f"artifacts/data/{split}_{dist}.npz"
            require(str(name.resolve()) in resolved,"evaluation dataset not frozen: "+str(name))
            arrays=npz(name)
            require(arrays["pins"].shape==(p[key],p["degree"],2),"frozen dataset shape mismatch")
            require(arrays["net_ids"].shape==(p[key],) and len(np.unique(arrays["net_ids"]))==p[key],"invalid net IDs")
    return sha(path),resolved


def audit_trajectory(root,out,p,latest,method,seed):
    """Follow the final chain backward, ignoring rows discarded by resume."""
    records=json_lines(out/"progress.jsonl")
    n=p["updates"]
    require(latest["update"]==n,"latest checkpoint has wrong effective update")
    require({r["update"] for r in records} >= set(range(1,n+1)),"progress misses trained updates")
    by_chain=defaultdict(list)
    for row in records:
        require(1<=row["update"]<=n,"out-of-range progress update")
        require(all(k in row for k in ("input_sha256","training_chain","previous_training_chain")),"progress lacks independently reconstructible training chain")
        by_chain[row["training_chain"]].append(row)
    current=latest["training_chain"]
    selected=[]
    for update in range(n,0,-1):
        candidates=[r for r in by_chain[current] if r["update"]==update]
        require(candidates,"Final checkpoint chain has no matching progress row at "+str(update))
        row=candidates[-1]
        selected.append(row);current=row["previous_training_chain"]
    require(current==EMPTY_CHAIN,"training chain does not terminate at initialization")
    selected.reverse()
    last_shard=None;arrays=None;chain=EMPTY_CHAIN;price=float(p["lambda_initial"])
    for row in selected:
        u=row["update"];first=1+((u-1)//p["shard_updates"])*p["shard_updates"]
        if first != last_shard:
            arrays=npz(root/f"artifacts/data/train_s{seed}_{first:04d}.npz");last_shard=first
        index=u-first;pins=arrays["pins"][index]
        transport=0.0
        if method=="wdro":
            indices=np.asarray(row.get("selected_columns"),dtype=np.int64)
            require(indices.shape==(p["batch_size"],),"missing/malformed selected shortlist indices")
            require(indices.min()>=0 and indices.max()<p["search"]["exact_rescore_top_k"],"selected shortlist index out of bounds")
            pins=arrays["shortlist"][index,np.arange(p["batch_size"]),indices]
            kind="cost_unrooted" if latest["identity"]["family"]=="rest" else "cost_rooted"
            transport=float(arrays[kind][index,np.arange(p["batch_size"]),indices].mean())
            price=max(0.,price+p["lambda_lr"]*(transport-latest["rho"]))
        raw=np.ascontiguousarray(pins).tobytes()
        require(hashlib.sha256(raw).hexdigest()==row["input_sha256"],"selected input digest mismatch at "+str(u))
        require(row["previous_training_chain"]==chain,"previous training chain mismatch")
        chain=hashlib.sha256(bytes.fromhex(chain)+raw).hexdigest()
        require(chain==row["training_chain"],"recomputed training chain mismatch at "+str(u))
        near(row["transport_mean"],transport,"transport log differs from selected frozen inputs",atol=1e-10)
        near(row["lambda"],price,"dual update does not match frozen rule",atol=1e-10)
        require(np.isfinite([row[k] for k in ("mean_cost","actor_loss","critic_loss","grad_norm","lambda","seconds")]).all(),"nonfinite progress")
    require(chain==latest["training_chain"],"final training chain mismatch")
    near(latest["lambda"],price,"checkpoint lambda mismatch",atol=1e-10)
    return {"effective_updates":n,"logged_rows":len(records),"discarded_or_duplicate_rows":len(records)-n,
            "training_chain":chain,"effective_update_seconds":float(sum(r["seconds"] for r in selected)),
            "mean_transport":float(np.mean([r["transport_mean"] for r in selected])),"final_lambda":price}


def audit_calls(out,p,method,metrics,latest):
    micro=p["greedy_microbatch"]
    padded=lambda n:math.ceil(n/micro)*micro
    expected={"train_sampled":p["updates"]*p["batch_size"]}
    if method=="wdro":expected["search_true"]=p["updates"]*padded(p["batch_size"]*p["search"]["exact_rescore_top_k"])
    validations=1+p["updates"]//p["validation_every"]
    for d in p["distributions"]:expected["validation_"+d]=validations*padded(p["validation_count"])
    require(latest["calls"]==expected,"checkpoint effective score counts mismatch")
    for d in p["distributions"]:
        expected["test_best_"+d]=padded(p["test_count"])
        expected["test_initial_"+d]=padded(p["test_count"])
        expected["attack_"+d]=padded(p["attack_count"]*(p["search"]["candidate_count"]+1))
    require(metrics["calls_effective_checkpoint_trajectory"]==expected,"final effective score counts mismatch")
    started,completed=Counter(),Counter()
    for row in json_lines(out/"actual_calls.jsonl"):
        require(row["event"] in ("started","completed") and int(row["nets"])>0,"invalid actual call entry")
        (started if row["event"]=="started" else completed)[row["category"]]+=int(row["nets"])
    for category,count in expected.items():
        require(started[category]>=completed[category]>=count,"actual score calls undercount: "+category)
    return {"effective_scores":sum(expected.values()),"actual_started_scores":sum(started.values()),
            "actual_completed_scores":sum(completed.values()),"effective_by_category":expected,
            "actual_started_by_category":dict(started),"actual_completed_by_category":dict(completed)}


def audit_run(root,p,binding,family,method,seed):
    import torch
    runid=f"{family}_{method}_s{seed}";out=root/"artifacts/runs"/runid
    completion=read(out/"complete.json");identity=completion["identity"]
    require(completion.get("status")=="complete" and completion.get("scope")=="degree10_pilot","nonformal or incomplete run")
    expected={"family":family,"method":method,"seed":seed,"scope":"degree10_pilot","run_id":runid,
              "updates":p["updates"],"protocol_sha256":sha(root/"protocol.json"),"frozen_sha256":binding}
    for k,v in expected.items():require(identity.get(k)==v,"run identity mismatch: "+k)
    require(completion["completed_updates"]==p["updates"] and completion["frozen_sha256"]==binding,"incomplete or wrong frozen update")
    require(REQUIRED_ARTIFACTS <= set(completion["artifacts"]),"incomplete artifact manifest")
    for name,expected_sha in completion["artifacts"].items():
        path=(out/name).resolve();path.relative_to(out.resolve())
        require(sha(path)==expected_sha,"run artifact hash mismatch: "+str(path))
    run=read(out/"run.json");metrics=read(out/"metrics.json")
    require(run["identity"]==identity==metrics["identity"],"run/metrics identity differs")
    calibration=read(root/"artifacts/data/calibration.json")
    near(identity["rho"],calibration["unrooted" if family=="rest" else "rooted"],"rho not from frozen calibration",atol=1e-12)
    latest=torch.load(out/"latest.pt",map_location="cpu",weights_only=False)
    initial=torch.load(out/"initial.pt",map_location="cpu",weights_only=False)
    best=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
    require(initial["identity"]==best["identity"]==latest["identity"]==identity,"checkpoint identity differs")
    init_digest=state_digest(initial["adapter"])
    trajectory=audit_trajectory(root,out,p,latest,method,seed)
    calls=audit_calls(out,p,method,metrics,latest)
    validations=json_lines(out/"validation.jsonl")
    grouped=defaultdict(list)
    for row in validations:grouped[row["update"]].append(row)
    required_updates=list(range(0,p["updates"]+1,p["validation_every"]))
    require(set(grouped)==set(required_updates),"validation schedule incomplete")
    # Repeated completed validations must be identical after recovery.
    for rows in grouped.values():
        require(all(r["summary"]==rows[0]["summary"] for r in rows),"repeated validation changed")
    chosen=min(required_updates,key=lambda u:max(grouped[u][-1]["summary"][d]["cost"]["mean"] for d in p["distributions"] if d!="uniform"))
    objective=max(grouped[chosen][-1]["summary"][d]["cost"]["mean"] for d in p["distributions"] if d!="uniform")
    require(best["best"]["update"]==metrics["best"]["update"]==latest["best"]["update"]==chosen,"selected checkpoint violates earliest validation minimum")
    near(best["best"]["objective"],objective,"best validation objective mismatch")
    del initial,best,latest
    with (out/"run.lock").open("r") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError("completed run still owns writer lock") from error
    heartbeat=read(out/"heartbeat.json")
    require(heartbeat["phase"]=="complete" and heartbeat["update"]==p["updates"],"final heartbeat incomplete")
    require(str(heartbeat.get("device","")).startswith("cuda") and float(metrics.get("cuda_max_allocated_mib") or 0)>0,"formal GPU execution evidence missing")
    exit_record=read(out/f"exit_{heartbeat['attempt']}.json")
    require(exit_record["exit_code"]==0 and exit_record.get("scope")=="degree10_pilot","run has not exited successfully")
    process=heartbeat.get("identity",{})
    if process.get("pid"):
        proc=Path(f"/proc/{int(process['pid'])}/stat")
        try:
            fields=proc.read_text().rsplit(")",1)[1].split()
            require(int(fields[19])!=int(process["start_ticks"]) or fields[0] in ("Z","X","x"),"completed run process is still live")
        except FileNotFoundError:
            pass
    datasets={name:npz(out/(name+".npz")) for name in ("test_best","test_initial")}
    require(set(datasets["test_best"])==set(datasets["test_initial"]),"initial/best per-net metrics differ")
    rows=[]
    for checkpoint,arrays in datasets.items():
        for d in p["distributions"]:
            require(d+"__cost" in arrays,"missing per-net cost")
        for key,value in arrays.items():
            d,metric=key.split("__",1)
            require(d in p["distributions"] and value.shape==(p["test_count"],) and np.isfinite(value).all(),"per-net test shape/data mismatch")
            values=value.astype(np.float64);mean=float(values.mean());p95=float(np.quantile(values,.95))
            near(metrics[checkpoint][d][metric]["mean"],mean,"stored mean differs from NPZ: "+key,atol=1e-30)
            near(metrics[checkpoint][d][metric]["p95"],p95,"stored P95 differs from NPZ: "+key,atol=1e-30)
            rows.append(dict(family=family,method=method,seed=seed,checkpoint=checkpoint,distribution=d,metric=metric,n=len(value),mean=mean,p95=p95))
    attack=npz(out/"attack.npz");risks={}
    require(set(attack)=={d+"__"+k for d in p["distributions"] for k in ("loss","cost","mass")},"unexpected attack artifact keys")
    for d in p["distributions"]:
        loss,cost,mass=(attack[d+"__"+key] for key in ("loss","cost","mass"))
        shape=(p["attack_count"],p["search"]["candidate_count"]+1)
        require(loss.shape==cost.shape==mass.shape==shape,"attack candidate shape mismatch")
        frozen=npz(root/f"artifacts/data/attack_{d}.npz")
        require(np.array_equal(cost,frozen["cost_unrooted" if family=="rest" else "cost_rooted"]),"attack costs differ from frozen matching costs")
        result=independent_lp(loss,cost,identity["rho"],mass)
        for key in ("risk","nominal","unconstrained_max","average_cost","row_error"):
            near(metrics["attack"][d][key],result[key],"stored LP summary differs: "+key)
        risks[d]=result
        rows.append(dict(family=family,method=method,seed=seed,checkpoint="best_attack",distribution=d,metric="finite_support_risk",n=shape[0],mean=result["risk"],p95=""))
    evidence={"run_id":runid,"status":"PASS","identity":identity,"complete_sha256":sha(out/"complete.json"),
              "initial_actor_digest":init_digest,"selected_update":chosen,"seconds":float(metrics["seconds"]),
              "trajectory":trajectory,"calls":calls,"attack":risks,
              "audit_files":{name:sha(out/name) for name in ("progress.jsonl","validation.jsonl","actual_calls.jsonl","heartbeat.json",f"exit_{heartbeat['attempt']}.json")}}
    require(np.isfinite(evidence["seconds"]) and evidence["seconds"]>0,"invalid elapsed time")
    return evidence,datasets,risks,rows


def aggregate_metrics(rows, seed_count):
    groups=defaultdict(list)
    for row in rows:
        row["n_training_seeds"]=1
        groups[tuple(row[k] for k in ("family","method","checkpoint","distribution","metric"))].append(row)
    aggregates=[]
    for key,values in groups.items():
        require(len(values)==seed_count,"missing training seed in descriptive aggregation")
        row=dict(zip(("family","method","checkpoint","distribution","metric"),key))
        row.update(seed="aggregate3",n=values[0]["n"],n_training_seeds=seed_count,
                   mean=float(np.mean([v["mean"] for v in values])),
                   p95=float(np.mean([v["p95"] for v in values])) if values[0]["p95"]!="" else "",
                   mean_between_seed_sd=float(np.std([v["mean"] for v in values],ddof=1)),
                   aggregation="mean_of_per_seed_statistics; same heldout nets")
        aggregates.append(row)
    return rows+aggregates


def comparisons(p,results):
    rows=[];repetitions=p["statistics"]["bootstrap_replicates"];confidence=p["statistics"]["confidence"]
    for family in p["families"]:
        for comparison in ("wdro_minus_erm","erm_best_minus_initial","wdro_best_minus_initial"):
            for d in p["distributions"]:
                per_seed=[]
                for seed in p["seeds"]:
                    er=results[(family,"erm",seed)];wd=results[(family,"wdro",seed)]
                    require(er[0]["initial_actor_digest"]==wd[0]["initial_actor_digest"],"paired Actor initialization mismatch")
                    require(set(er[1]["test_best"])==set(wd[1]["test_best"]),"paired model metric keys mismatch")
                    if comparison=="wdro_minus_erm":before,after=er[1]["test_best"],wd[1]["test_best"]
                    else:
                        run=er if comparison.startswith("erm_") else wd
                        before,after=run[1]["test_initial"],run[1]["test_best"]
                    # Same initial weights on the same net order must score identically.
                    require(np.array_equal(er[1]["test_initial"][d+"__cost"],wd[1]["test_initial"][d+"__cost"]),"paired initial greedy costs differ")
                    per_seed.append((seed,before,after))
                keys=sorted(k for k in per_seed[0][1] if k.startswith(d+"__") and k.split("__",1)[1] not in NONCOMPARATIVE)
                for key in keys:
                    metric=key.split("__",1)[1]
                    for seed,before,after in per_seed:
                        row={"comparison":comparison,"family":family,"seed":seed,"distribution":d,"metric":metric,
                             **paired_bootstrap(before[key],after[key],repetitions,confidence,stat_seed(f"{family}/{comparison}/{seed}/{key}"))}
                        rows.append(row)
                    aa=np.stack([a[key] for _,a,_ in per_seed]);bb=np.stack([b[key] for _,_,b in per_seed])
                    rows.append({"comparison":comparison,"family":family,"seed":"aggregate3","distribution":d,"metric":metric,
                                 **paired_bootstrap(aa,bb,repetitions,confidence,stat_seed(f"{family}/{comparison}/aggregate/{key}"))})
        for d in p["distributions"]:
            delta=[]
            for seed in p["seeds"]:
                value=results[(family,"wdro",seed)][2][d]["risk"]-results[(family,"erm",seed)][2][d]["risk"]
                delta.append(value)
                rows.append({"comparison":"wdro_minus_erm","family":family,"seed":seed,"distribution":d,"metric":"finite_support_risk",
                             "mean_delta":value,"n_nets":p["attack_count"],"n_training_seeds":1,"ci_scope":"LP reoptimized exactly; net CI not estimated"})
            rng=np.random.default_rng(stat_seed(f"{family}/{d}/lp_seeds"));values=np.asarray(delta)
            samples=values[rng.integers(0,len(values),size=(repetitions,len(values)))].mean(1)
            ci=np.quantile(samples,[(1-confidence)/2,(1+confidence)/2])
            rows.append({"comparison":"wdro_minus_erm","family":family,"seed":"aggregate3","distribution":d,"metric":"finite_support_risk",
                         "mean_delta":float(values.mean()),"mean_seed_ci_low":float(ci[0]),"mean_seed_ci_high":float(ci[1]),
                         "n_nets":p["attack_count"],"n_training_seeds":len(values),"mean_negative_seeds":int((values<0).sum()),"ci_scope":"training-seed bootstrap only; 3 seeds"})
    return rows


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    temp=path.with_name(path.name+f".{os.getpid()}.partial")
    with temp.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def render(p,audit,rows):
    lines=["# 三篇论文与基础 WDRO：独立验收与配对统计","",f"工程验收：**PASS**。{len(audit['runs'])} 个正式运行均完成 {p['updates']} 次有效更新，degree={p['degree']}。", "",
           "逐项验证了冻结源码/数据与模型产物 SHA256、恢复后的有效训练输入链、真实调用计数、选模记录、成功退出与释放的写锁；从 per-net NPZ 重算统计，并独立重求有限候选 LP、核验保存的质量分配与预算。",
           "","**解释范围**：家族内部比较。不同论文的物理目标与标度不同，不跨家族排成本名次。等更新次数不等计算量；Hub 每真实评分内部有 11 次 PD 参数构造。", "",
           "每个训练 seed 的 paired-net bootstrap 使用 2000 次同 net 重采样；3 seed 汇总同时区分固定这些模型的 net 条件区间与仅 3 个训练重复的 seed 区间。汇总 P95 是各 seed P95 的均值，并非混合全部预测的 P95。多重比较未校正，探索性解释；区间跨 0 或 seed 方向不一致时不建立稳定改善。", "",
           "原始成本 mean/P95 见 metrics.csv；所有指标的 WDRO−ERM 与 best−initial 配对结果见 comparisons.csv。负差值表示该指标下降；对于 hub/buffer 数量等指标，下降本身不代表物理目标改善。初始化参与选模，故 best−initial 可为零。", "",
           "有限支持 LP 使用同一 rho 与每中心 33 个候选；保存的 LP 风险已重新优化验证。此处的 LP 差没有伪装成逐行 witness 贡献的 net bootstrap；3-seed LP 区间仅重采样训练重复，不是连续 Wasserstein 证书。", ""]
    for family in p["families"]:
        lines += [f"## {family}","","WDRO−ERM，3 个训练 seed 的均值：","","| 分布 | 成本均值差 | net 条件 95% CI | seed 95% CI | 均值下降 seed | P95 差 | LP 风险差 |","|---|---:|---|---|---:|---:|---:|"]
        for d in p["distributions"]:
            row=next(r for r in rows if r["family"]==family and r["distribution"]==d and r["comparison"]=="wdro_minus_erm" and r["seed"]=="aggregate3" and r["metric"]=="cost")
            lp=next(r for r in rows if r["family"]==family and r["distribution"]==d and r["seed"]=="aggregate3" and r["metric"]=="finite_support_risk")
            lines.append(f"| {d} | {row['mean_delta']:.7g} | [{row['mean_net_ci_low']:.7g}, {row['mean_net_ci_high']:.7g}] | [{row['mean_seed_ci_low']:.7g}, {row['mean_seed_ci_high']:.7g}] | {row['mean_negative_seeds']}/3 | {row['p95_delta']:.7g} | {lp['mean_delta']:.7g} |")
        lines += ["","| 方法/seed | 最佳 update | 运行时间(s) | 有效真评分 | 实际开始评分 | 实际完成评分 |","|---|---:|---:|---:|---:|---:|"]
        for run in audit["runs"]:
            i=run["identity"]
            if i["family"]==family:
                c=run["calls"];lines.append(f"| {i['method']}/{i['seed']} | {run['selected_update']} | {run['seconds']:.1f} | {c['effective_scores']} | {c['actual_started_scores']} | {c['actual_completed_scores']} |")
        lines.append("")
    lines += ["## 尚未由本轮验证的内容","","论文完整训练配方与 Pareto frontier、大 degree 分解合并、工业 signoff timing、跨任务优势、连续空间最坏风险及更多独立训练重复均未由此 pilot 证明。验证成本摘要与轨迹不等于重新执行所有神经网络预测；本次独立工作重算统计和 LP，并核验其绑定的实际预测产物。",""]
    return "\n".join(lines)


def partial(root,p):
    rows=[]
    for family in p["families"]:
        for method in p["methods"]:
            for seed in p["seeds"]:
                runid=f"{family}_{method}_s{seed}";out=root/"artifacts/runs"/runid
                heartbeat=read(out/"heartbeat.json") if (out/"heartbeat.json").exists() else {}
                completion=read(out/"complete.json") if (out/"complete.json").exists() else {}
                rows.append({"run_id":runid,"observed_phase":heartbeat.get("phase","not_started"),"observed_update":heartbeat.get("update",0),
                             "completion_claim_present":completion.get("status")=="complete","heartbeat_time":heartbeat.get("time"),"audit_status":"not_audited"})
    out=root/"artifacts/report";out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/"progress.json",{"status":"partial","time":time.time(),"formal_expected_runs":len(rows),"runs":rows,"note":"Read-only run progress; completion claims are not acceptance."})
    lines=["# 正式实验进度（尚未验收）","","本次 --partial 只报告真实文件中的进度，未运行统计验收，也未生成完成凭证。","","| 正式 run | 已报告阶段 | 已报告 update | 存在完成声明 |","|---|---|---:|---|"]
    lines += [f"| {r['run_id']} | {r['observed_phase']} | {r['observed_update']} | {r['completion_claim_present']} |" for r in rows]
    (out/"PROGRESS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"partial","progress":str(out/"PROGRESS.md"),"runs":len(rows)}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--partial",action="store_true");parser.add_argument("--root",type=Path,default=ROOT)
    args=parser.parse_args();root=args.root.resolve();root.relative_to(ROOT.resolve());p=read(root/"protocol.json")
    if args.partial:partial(root,p);return 0
    out=root/"artifacts/report";out.mkdir(parents=True,exist_ok=True)
    lock=(out/"report.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();audit={"status":"running","started_at":start,"runs":[],"failures":[]}
    try:
        binding,_=audit_frozen(root,p);audit["frozen_sha256"]=binding
        require(p["degree"]==10 and p["updates"]==2048 and len(p["families"])*len(p["methods"])*len(p["seeds"])==24,"unexpected formal pilot scope")
        results={};metric_rows=[]
        for family in p["families"]:
            for method in p["methods"]:
                for seed in p["seeds"]:
                    key=(family,method,seed)
                    try:
                        value=audit_run(root,p,binding,*key);results[key]=value;audit["runs"].append(value[0]);metric_rows.extend(value[3])
                    except Exception as error:audit["failures"].append({"run_id":f"{family}_{method}_s{seed}","error":repr(error)})
                    atomic_json(out/"audit_progress.json",audit)
        require(not audit["failures"],"one or more formal run audits failed")
        metric_rows=aggregate_metrics(metric_rows,len(p["seeds"]))
        comparison_rows=comparisons(p,results)
        # No source can drift while statistics are being computed.
        require(audit_frozen(root,p)[0]==binding,"source freeze changed during reporting")
        audit.update(status="PASS",finished_at=time.time(),seconds=time.time()-start,protocol_sha256=sha(root/"protocol.json"),reporter_sha256=sha(Path(__file__)),
                     inference_boundary="degree10; fixed heldout nets and 3 training seeds; finite support only; no cross-family rank")
        write_csv(out/"metrics.csv",metric_rows);write_csv(out/"comparisons.csv",comparison_rows)
        (out/"REPORT.md").write_text(render(p,audit,comparison_rows))
        atomic_json(out/"audit.json",audit)
        artifact_names=["REPORT.md","metrics.csv","comparisons.csv","audit.json"]
        atomic_json(out/"complete.json",{"status":"complete","scope":"degree10_pilot_independent_acceptance","runs":len(results),
                    "frozen_sha256":binding,"protocol_sha256":sha(root/"protocol.json"),"time":time.time(),
                    "artifacts":{name:sha(out/name) for name in artifact_names}})
        print(json.dumps({"status":"complete","runs":len(results),"report":str(out/"REPORT.md")}));return 0
    except Exception as error:
        audit.update(status="FAIL",finished_at=time.time(),seconds=time.time()-start,error=repr(error))
        atomic_json(out/"audit.json",audit)
        if (out/"complete.json").exists():
            os.replace(out/"complete.json",out/f"invalidated_complete_{time.time_ns()}.json")
        (out/"REPORT.md").write_text("# 独立验收未通过\n\n未生成完成凭证。详见 audit.json。\n\n"+repr(error)+"\n")
        print(json.dumps({"status":"FAIL","error":repr(error)}));return 2
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=="__main__":
    raise SystemExit(main())

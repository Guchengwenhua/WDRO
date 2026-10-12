"""Extract research evidence from hash-bound PASS reports; no scoring or training.

Uses only the Python standard library. It neither imports Torch nor opens task
checkpoints/NPZ files. The upstream full audit is authoritative; this consumer
rechecks the frozen protocol and every CSV/audit binding it consumes.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEGREES = (10, 20, 30, 40, 50)
SEEDS = (101, 102, 103)
PARENTS = ("erm", "wdro_rest")
DISTRIBUTIONS = ("uniform", "mix_gauss", "edge_bias", "grid_cluster")
FIXED_BUDGETS = dict(random_top4=4, critic_raw4=4, critic_calibrated4=4, rl4=4,
                     random64=64, cem64=64, es64=64, critic_calibrated64=64, rl64=64)
FIXED_PAIRS = (("es64", "random64"), ("es64", "cem64"),
               ("critic_calibrated4", "random_top4"), ("rl4", "random_top4"),
               ("critic_calibrated4", "critic_raw4"),
               ("critic_calibrated64", "random64"), ("rl64", "random64"))
JOINT_METHODS = ("random_top4", "critic_calibrated4", "rl_frozen4", "rl_joint4", "random64", "es64")
JOINT_PAIRS = (("rl_joint4", "rl_frozen4"), ("rl_frozen4", "random_top4"),
               ("critic_calibrated4", "random_top4"), ("es64", "random64"))
SCHEMAS = {
    "fixed_cells": (1080, "parent degree seed distribution searcher budget price rho nets mean_regret p95_regret mean_score mean_cost mean_delta_score improved_fraction real_score_requests ledger_delta geometry selected_legal_fraction"),
    "fixed_paired": (1120, "parent degree distribution left right metric difference ci_low ci_high per_seed_differences all_seeds_positive all_seeds_negative stable_positive stable_negative seed_count centers bootstrap_replicates"),
    "fixed_efficiency": (30, "task parent degree seed critic_label_requests rl_training_requests rl_validation_requests rl_baseline_requests search_requests extra_completed_requests actual_ledger walltime_scope"),
    "fixed_diagnostics": (30, "task parent degree seed raw_validation_mse calibrated_validation_mse raw_validation_spearman calibrated_validation_spearman raw_zero_gradient_fraction calibrated_zero_gradient_fraction raw_zero_prediction_fraction calibrated_zero_prediction_fraction rl_best_update rl_best_validation_f critic_best_epoch"),
    "joint_cells": (216, "method degree seed distribution stage best_update nets mean_regret p95_regret"),
    "joint_paired": (112, "comparison left right distribution metric difference ci_low ci_high per_seed_differences seed_count centers bootstrap_replicates stable_improvement"),
    "joint_efficiency": (18, "task method degree seed best_update initial_price final_price rho actual_ledger training_search_requests online_attacker_requests sampled_main_actor_nets committed_training_wall_seconds initial_validation best_validation"),
    "joint_worst_group": (54, "method seed stage worst_group worst_group_mean group_means"),
}
LIMITATIONS = [
    "A 的绝对 f 只在相同 X、父模型和 lambda 内构成同一个攻击目标；不同 lambda 或父模型的绝对 f 不作统一排名。",
    "跨条件汇总只描述固定 parent/degree/distribution 条件集合的配对差；平均配对差不附带新总体 CI，也不构成统一 lambda 的目标。",
    "A 的 score 差越高表示惩罚目标下攻击越强；regret 与 cost 分列，不能把位移更大单独解释为优势。",
    "4/64 是真实搜索评分请求数，包含原 X；不等于包括训练、梯度、网络 padding 和参考求解在内的总计算相同。",
    "Critic 标签和 RL 奖励预训练成本单列；B 的 rl_joint4 额外奖励调用与 A 预训练成本另计，跨分支复用的预训练不重复宣称新增工作。",
    "B 的名义性能限制是独立 validation 上相对起点退化不超过 0.002；evaluation 表中的退化单独报告，不用 evaluation 反选模型。",
    "B 仅 degree30 的最小一步双策略交替，外层 sampled REINFORCE 与内层 greedy 评分存在近似；未实现完整多步 MARL。",
    "独立攻击只对最终选中模型运行 CEM64、lambda=0；不把 attacked_final 与不同中心的 plain initial 当作配对攻击前后改善。",
    "2000 次 CI 条件于三个已训练 seed 和固定中心，三个 seed 共享中心重采样，未作多重比较校正；不同条件格不是独立重复实验。",
    "共享 GPU、CPU 参考求解、恢复分段和 reference 热缓存影响耗时；保留资源账目，不计算或声称受控加速比。",
    "合法性、调用预算、恢复和审计通过是工程验收，不自动表示算法有效；失败效果和 best_update=0 均保留。",
    "此数据是新增合成数据验证，不是 ISPD 外推证明，也没有连续 Wasserstein 最坏分布证书。",
    "本工具只读取完整 PASS 审计发布的 CSV 并校验其绑定，不重新执行逐 net 科学审计；未保存 RL 动作轨迹不在完整逐内容重叠审计内。",
]


class EvidenceError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def decode(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def read_csv(path, name):
    count, required = SCHEMAS[name]
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(set(required.split()) <= set(reader.fieldnames or []), f"{name}: schema mismatch")
        rows = [{key: decode(value) for key, value in row.items()} for row in reader]
    require(len(rows) == count, f"{name}: expected {count} rows, got {len(rows)}")
    return rows


def unique(rows, keys, expected):
    indexed = {tuple(row[key] for key in keys): row for row in rows}
    require(len(indexed) == len(rows) and set(indexed) == set(expected),
            f"incomplete or duplicate coordinate grid: {keys}")
    return indexed


def load_evidence(root, phase):
    """Require a completed upstream audit; never silently use partial task files."""
    phases = ("fixed", "joint") if phase == "all" else (phase,)
    manifest_path = root / "manifests" / f"{phase}_completion.json"
    require(manifest_path.exists(), f"完整 {phase} PASS 审计尚不存在: {manifest_path}")
    completion = json.loads(manifest_path.read_text())
    require(completion.get("status") == "pass" and completion.get("phase") == phase,
            "completion is not a matching PASS")
    require(completion.get("provenance", {}).get("status") == "pass", "provenance audit is not PASS")
    require(phase != "all" or completion.get("all_experiments_complete") is True,
            "all-experiment completion not established")
    protocol = json.loads((root / "protocol.json").read_text())
    require(completion["protocol_sha256"] == sha256(root / "protocol.json"), "protocol binding changed")
    require(protocol["plan_sha256"] == sha256(root / "EXPERIMENT_PLAN.md"), "plan binding changed")
    require(protocol["degrees"] == list(DEGREES) and protocol["seeds"] == list(SEEDS)
            and protocol["parents"] == list(PARENTS), "unsupported experiment grid")
    bindings = {}
    for record in completion["artifacts"]:
        key = str(Path(record["path"]).resolve())
        require(key not in bindings or bindings[key] == record["sha256"], "conflicting artifact binding")
        bindings[key] = record["sha256"]
    used = [{"path": str(manifest_path), "sha256": sha256(manifest_path)}]
    def verify(path):
        require(path.exists(), f"missing bound artifact: {path}")
        actual = sha256(path)
        require(bindings.get(str(path.resolve())) == actual, f"unbound or modified artifact: {path}")
        used.append({"path": str(path), "sha256": actual})
    tables = {}
    for selected in phases:
        expected = {"tasks": 30, "cells": 1080} if selected == "fixed" else {"tasks": 18, "cells": 216}
        require(completion["phases"].get(selected) == expected, f"{selected}: complete grid not established")
        audit_path = root / "experiments/tables" / f"{selected}_audit.json"
        verify(audit_path)
        audit = json.loads(audit_path.read_text())
        require(audit.get("status") == "pass" and audit.get("phase") == selected,
                f"{selected}: table audit is not PASS")
        for name in SCHEMAS:
            if name.startswith(selected + "_"):
                path = root / "experiments/tables" / f"{name}.csv"
                verify(path)
                tables[name] = read_csv(path, name)
    return tables, used


def direction(value, positive_good):
    if value == 0:
        return "equal"
    return "left_better" if (value > 0) == positive_good else "left_worse"


def seed_pairs(rows, phase):
    result = []
    for row in rows:
        values = row["per_seed_differences"]
        require(len(values) == 3 and row["seed_count"] == 3 and row["bootstrap_replicates"] == 2000,
                "paired seed/CI configuration mismatch")
        require(abs(sum(values)/3 - row["difference"]) < 1e-12, "paired mean does not match seed differences")
        for seed, value in zip(SEEDS, values):
            entry = {key: row[key] for key in ("parent", "degree", "distribution", "comparison", "left", "right", "metric") if key in row}
            entry.update(seed=seed, difference=value,
                         direction="no_standalone_win_direction" if row["metric"] == "cost_mean" else direction(value, phase == "fixed"),
                         role="attack_objective" if row["metric"] == "score_mean" else "secondary_metric")
            result.append(entry)
    return result


def fixed_extract(tables):
    cells, pairs, efficiency = (tables["fixed_"+key] for key in ("cells", "paired", "efficiency"))
    indexed = unique(cells, ("parent", "degree", "seed", "distribution", "searcher"),
                     itertools.product(PARENTS, DEGREES, SEEDS, DISTRIBUTIONS, FIXED_BUDGETS))
    for key, row in indexed.items():
        require(row["budget"] == FIXED_BUDGETS[row["searcher"]] and row["real_score_requests"] == 256*row["budget"], "unequal search budget")
        baseline = indexed[key[:-1]+("random_top4",)]
        require(row["price"] == baseline["price"], "within-condition lambda differs")
    unique(pairs, ("parent", "degree", "distribution", "left", "right", "metric"),
           ((p,d,dist,l,r,m) for p,d,dist,(l,r),m in itertools.product(PARENTS,DEGREES,DISTRIBUTIONS,FIXED_PAIRS,("score_mean","regret_mean","regret_p95","cost_mean"))))
    summaries = []
    for parent, degree, (left, right) in itertools.product(PARENTS, DEGREES, FIXED_PAIRS):
        require(FIXED_BUDGETS[left] == FIXED_BUDGETS[right], "cross-budget headline comparison")
        subset = [r for r in pairs if (r["parent"],r["degree"],r["left"],r["right"],r["metric"]) == (parent,degree,left,right,"score_mean")]
        summaries.append(dict(parent=parent,degree=degree,left=left,right=right,budget=FIXED_BUDGETS[left],
            condition_count=len(subset),stable_positive_conditions=sum(r["stable_positive"] for r in subset),
            stable_negative_conditions=sum(r["stable_negative"] for r in subset),
            remaining_conditions=sum(not (r["stable_positive"] or r["stable_negative"]) for r in subset),
            mean_of_conditional_paired_differences=sum(r["difference"] for r in subset)/len(subset),
            prices_by_seed={s:indexed[(parent,degree,s,"uniform",left)]["price"] for s in SEEDS},
            interpretation="descriptive average over four distribution conditions; no pooled CI or common-lambda objective"))
    costs = []
    for row in efficiency:
        item = dict(row)
        item.update(rl_pretraining_total_requests=row["rl_training_requests"]+row["rl_validation_requests"],
                    baseline_requests_already_included_in_rl_counts=True,
                    critic_pretraining_total_requests=row["critic_label_requests"],
                    learned_search_startup_cost_shared_between_4_and_64_budget=True)
        costs.append(item)
    return dict(fixed_by_seed=cells,fixed_paired=pairs,fixed_pairs_by_seed=seed_pairs(pairs,"fixed"),
                fixed_pair_condition_summary=summaries,fixed_training_cost=costs,
                critic_diagnostics=tables["fixed_diagnostics"])


def joint_extract(tables):
    cells, pairs, efficiency = (tables["joint_"+key] for key in ("cells", "paired", "efficiency"))
    indexed = unique(cells, ("method","seed","distribution","stage"),
                     itertools.product(JOINT_METHODS,SEEDS,DISTRIBUTIONS,("initial","final","attacked_final")))
    unique(efficiency,("method","seed"),itertools.product(JOINT_METHODS,SEEDS))
    unique(tables["joint_worst_group"],("method","seed","stage"),itertools.product(JOINT_METHODS,SEEDS,("initial","final","attacked_final")))
    expected_pairs = ((comparison,left,right,dist,metric)
        for dist,metric in itertools.product(DISTRIBUTIONS,("regret_mean","regret_p95"))
        for comparison,left,right in (
            [("after_minus_before",method,"initial_erm") for method in JOINT_METHODS]
            + [(stage,left,right) for stage,(left,right) in itertools.product(("final","attacked_final"),JOINT_PAIRS)]))
    unique(pairs,("comparison","left","right","distribution","metric"),expected_pairs)
    before_after, nominal, costs = [], [], []
    startup = {(r["parent"],r["degree"],r["seed"]):r for r in tables.get("fixed_efficiency",[])}
    for method, seed, dist in itertools.product(JOINT_METHODS,SEEDS,DISTRIBUTIONS):
        initial, final = (indexed[(method,seed,dist,stage)] for stage in ("initial","final"))
        before_after.append(dict(method=method,degree=30,seed=seed,distribution=dist,best_update=final["best_update"],
            initial_mean_regret=initial["mean_regret"],final_mean_regret=final["mean_regret"],
            mean_regret_difference=final["mean_regret"]-initial["mean_regret"],
            initial_p95_regret=initial["p95_regret"],final_p95_regret=final["p95_regret"],
            p95_regret_difference=final["p95_regret"]-initial["p95_regret"],lower_is_better=True))
    for row in efficiency:
        initial, best = row["initial_validation"], row["best_validation"]
        delta = best["nominal_mean"]-initial["nominal_mean"]
        require(delta <= .002+1e-12, "audited validation gate unexpectedly fails")
        eval_initial = indexed[(row["method"],row["seed"],"uniform","initial")]
        eval_final = indexed[(row["method"],row["seed"],"uniform","final")]
        eval_delta = eval_final["mean_regret"]-eval_initial["mean_regret"]
        nominal.append(dict(method=row["method"],seed=row["seed"],best_update=row["best_update"],
            retained_initial_model=row["best_update"]==0,validation_initial_nominal=initial["nominal_mean"],
            validation_selected_nominal=best["nominal_mean"],validation_nominal_degradation=delta,
            validation_gate_pass=True,validation_tolerance=.002,
            validation_initial_ood_worst=initial["ood_worst_group_mean"],
            validation_selected_ood_worst=best["ood_worst_group_mean"],
            evaluation_nominal_degradation=eval_delta,evaluation_exceeds_validation_tolerance=eval_delta>.002,
            evaluation_was_not_used_for_selection=True))
        item = dict(row)
        item["search_budget_per_center"] = 64 if row["method"] in ("random64","es64") else 4
        require(row["training_search_requests"] == 512*64*item["search_budget_per_center"], "joint search budget mismatch")
        require(row["online_attacker_requests"] == (16384 if row["method"]=="rl_joint4" else 0), "joint online budget mismatch")
        source = startup.get(("erm",30,row["seed"]))
        learned = row["method"] in ("critic_calibrated4","rl_frozen4","rl_joint4")
        item["pretrained_attacker_source"] = "fixed ERM d30 same seed" if learned else "none"
        item["pretrained_attacker_requests"] = (source["critic_label_requests"] if row["method"]=="critic_calibrated4" else source["rl_training_requests"]+source["rl_validation_requests"]) if learned and source else (None if learned else 0)
        item["pretraining_cost_already_paid_in_phase_A"] = learned
        item["committed_search_plus_online_requests"] = row["training_search_requests"]+row["online_attacker_requests"]
        item["additional_main_sampled_requests"] = row["sampled_main_actor_nets"]
        item["validation_evaluation_and_padding_are_in_actual_ledger"] = True
        costs.append(item)
    return dict(joint_by_seed=cells,joint_paired=pairs,joint_pairs_by_seed=seed_pairs(pairs,"joint"),
                joint_before_after_by_seed=before_after,joint_nominal_gate=nominal,
                joint_training_cost=costs,joint_worst_group=tables["joint_worst_group"])


def publish(outdir, tables, bindings, phase):
    outdir.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    for name, rows in tables.items():
        path=outdir/(name+".csv")
        with tempfile.NamedTemporaryFile(mode="w",newline="",encoding="utf-8",dir=outdir,delete=False) as handle:
            fields=list(dict.fromkeys(key for row in rows for key in row))
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
            handle.flush();os.fsync(handle.fileno())
        os.replace(handle.name,path)
        artifacts.append(dict(path=str(path),sha256=sha256(path),rows=len(rows)))
    report=dict(status="pass",phase=phase,all_experiments_complete=phase=="all",source_bindings=bindings,
                script_sha256=sha256(__file__),artifacts=artifacts,limitations=LIMITATIONS,
                interpretation="Evidence extraction only. No new training, oracle calls, statistical tests, or cross-lambda absolute-score ranking.")
    path=outdir/"extraction.json"
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=outdir,delete=False) as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.flush();os.fsync(handle.fileno())
    os.replace(handle.name,path)
    return report


def schema_dry_run(root):
    """Check interface keys and parser/guard behavior, with no research results."""
    syntax=ast.parse((root/"adv_v3/report.py").read_text())
    literals={node.value for node in ast.walk(syntax) if isinstance(node,ast.Constant) and isinstance(node.value,str)}
    literals.update(node.arg for node in ast.walk(syntax) if isinstance(node,ast.keyword) and node.arg)
    missing=sorted({key for _,spec in SCHEMAS.values() for key in spec.split()}-literals)
    require(not missing,f"report.py lacks expected schema literals: {missing}")
    require(decode("True") is True and decode("False") is False and decode("") is None
            and decode("[0.1, -0.2, 0]")==[.1,-.2,0] and decode('{"nominal_mean": 0.01}')["nominal_mean"]==.01,
            "CSV scalar/JSON decoding failed")
    with tempfile.TemporaryDirectory() as temp:
        try:
            load_evidence(Path(temp),"all")
        except EvidenceError:
            pass
        else:
            raise EvidenceError("missing audit did not block extraction")
    require(all(FIXED_BUDGETS[l]==FIXED_BUDGETS[r] for l,r in FIXED_PAIRS),"cross-budget comparison")
    return dict(status="pass",mode="schema_dry_run",research_results_read=False,GPU_used=False,
                checked_tables=len(SCHEMAS),expected_rows={name:spec[0] for name,spec in SCHEMAS.items()},
                missing_audit_correctly_rejected=True,report_sha256=sha256(root/"adv_v3/report.py"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--phase",choices=("fixed","joint","all"),default="all")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--schema-dry-run",action="store_true")
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.schema_dry_run:
            print(json.dumps(schema_dry_run(root),ensure_ascii=False,indent=2));return
        tables,bindings=load_evidence(root,args.phase)
        extracted={}
        if args.phase in ("fixed","all"):
            extracted.update(fixed_extract(tables))
        if args.phase in ("joint","all"):
            extracted.update(joint_extract(tables))
        output=args.output or root/"experiments/interpretation"/args.phase
        report=publish(output,extracted,bindings,args.phase)
        print(json.dumps(dict(status=report["status"],phase=args.phase,output=str(output),
                              table_rows={key:len(value) for key,value in extracted.items()}),ensure_ascii=False))
    except (EvidenceError,KeyError,ValueError) as error:
        parser.exit(2,"Evidence extraction blocked: "+str(error)+"\n")


if __name__=="__main__":
    main()

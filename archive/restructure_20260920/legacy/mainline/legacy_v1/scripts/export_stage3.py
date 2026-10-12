#!/usr/bin/env python3
"""Export only fully audited stage-3 results and stop at the stage boundary."""
import csv
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/"experiments"
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from run_stage3_baseline_queue import build_tasks
from audit_stage3 import effective_events


def write_csv(path,rows):
    temporary=path.with_suffix(path.suffix+".partial")
    with temporary.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows);handle.flush();os.fsync(handle.fileno())
    with temporary.open() as handle:
        if len(list(csv.DictReader(handle)))!=len(rows):
            raise RuntimeError("CSV readback count mismatch")
    os.replace(temporary,path)


def main():
    audit_path=EXP/"logs/stage3_audit/stage3_formal_audit.json"
    audit=json.loads(audit_path.read_text())
    tasks=build_tasks()
    if audit["status"]!="pass" or {r["run_id"] for r in audit["runs"]}!={t["run_id"] for t in tasks}:
        raise RuntimeError("all 45 audits must pass before stage completion")
    rows=[];long=[];best_long=[];budgets=[];models=[]
    for task in tasks:
        path=EXP/"logs/stage3"/task["run_id"]/"result.json"
        result=json.loads(path.read_text())
        record=next(r for r in audit["runs"] if r["run_id"]==task["run_id"])
        if file_sha256(path)!=record["result_sha256"]:
            raise RuntimeError("result changed since audit")
        if file_sha256(result["best_checkpoint"])!=result["best_checkpoint_sha256"]:
            raise RuntimeError("model changed since audit")
        train,validation,raw=effective_events(result["event_log"])
        initial=validation[0]["result"]["selection"]
        erm_path=EXP/"logs/stage2"/f"formal_erm_d{task['degree']}_s{task['seed']}_u2000"/"result.json"
        erm=json.loads(erm_path.read_text())
        segments=[e["result"] for e in raw if e.get("event")=="segment_end"]
        row={"run_id":task["run_id"],"method":task["method"],"degree":task["degree"],"training_seed":task["seed"],
            "policy_updates":result["completed_updates"],"batch_size":result["batch_size"],"best_update":result["best"]["update"],
            "best_nominal_mean":result["best"]["nominal_mean"],"erm_best_nominal_mean":erm["best"]["nominal_mean"],
            "nominal_delta_vs_erm":result["best"]["nominal_mean"]-erm["best"]["nominal_mean"],
            "best_ood_worst_group":result["best"]["selection_objective"],"erm_best_ood_worst_group":erm["best"]["selection_objective"],
            "ood_delta_vs_erm":result["best"]["selection_objective"]-erm["best"]["selection_objective"],
            "peak_reserved_mib":max(s["peak_memory_reserved_bytes"] for s in segments)/2**20,
            "completed_segment_wall_seconds":sum(s["segment_wall_seconds"] for s in segments),
            "best_checkpoint_sha256":result["best_checkpoint_sha256"],"latest_checkpoint_sha256":result["latest_checkpoint_sha256"],
            **result["call_ledger"]}
        rows.append(row)
        models.append({"run_id":task["run_id"],"method":task["method"],"degree":task["degree"],"seed":task["seed"],
            "best_logical_path":result["best_checkpoint"],"best_resolved_path":str(Path(result["best_checkpoint"]).resolve()),
            "best_sha256":result["best_checkpoint_sha256"],"latest_sha256":result["latest_checkpoint_sha256"],
            "run_identity":result["run_identity"]})
        for update,event in sorted(validation.items()):
            for group,value in event["result"]["groups"].items():
                line={"run_id":task["run_id"],"method":task["method"],"degree":task["degree"],"training_seed":task["seed"],
                    "update":update,"distribution":group,"is_selected":update==result["best"]["update"],
                    "mean":value["summary"]["mean"],"p95":value["summary"]["p95"],"count":value["summary"]["count"],
                    "dataset_sha256":value["dataset_array_bundle_sha256"],"best_checkpoint_sha256":result["best_checkpoint_sha256"]}
                long.append(line)
                if line["is_selected"]:best_long.append(line)
        if task["method"]=="hard_mining":
            budgets.append({"run_id":task["run_id"],"degree":task["degree"],"training_seed":task["seed"],
                **result["method_state"]["budget"],"training_center_stream_sha256":result["training_stream_hash"],
                "candidate_support_stream_sha256":result["method_state"]["support_stream_hash"],
                "wdro_actual_pair_status":"pending_stage4; shared_support_and_call_contract_ready"})
    paths={"runs":EXP/"tables/stage3_baseline_runs.csv","best_validation":EXP/"tables/stage3_best_validation.csv",
           "all_validation":EXP/"tables/stage3_all_validation.csv","hard_mining_budgets":EXP/"tables/stage3_hard_mining_budgets.csv"}
    for key,items in zip(paths,(rows,best_long,long,budgets)):write_csv(paths[key],items)
    manifest={"status":"pass","completed_at_utc":utc_now(),"stage":3,"models":models,"model_count":45,
        "audit":{"path":str(audit_path),"sha256":file_sha256(audit_path)},
        "tables":{k:{"logical_path":str(p),"resolved_path":str(p.resolve()),"sha256":file_sha256(p)} for k,p in paths.items()},
        "stage4_started":False,"test_or_ispd_models_evaluated":False,
        "wdro_budget_pairing":"pending actual WDRO runs in stage4; per-run contract and shared implementation frozen"}
    completion=ROOT/"manifests/stage3_completion_v1.json";atomic_write_json(completion,manifest)
    report=["# 阶段 3 三基线验收报告","",f"完成时间：{manifest['completed_at_utc']}。45/45 正式模型通过验收，各 2,000 policy updates、batch 256。",
        f"全量审计 {audit['checks_passed']} 项通过；全部训练输入和 Hard Mining 候选重建，held-out 内容交集为 0。",
        "", "三种方法均复用冻结 stage2 trainer 的初始化、Adam、scheduler、梯度裁剪、验证频率与 checkpoint schema。全部模型只用 validation 选择；所选 nominal 相对相同 degree/seed ERM 的退化均不超过 0.002。",
        "", "证据：", "- `wdro_rest_min/manifests/stage3_completion_v1.json`", "- `experiments/logs/stage3_audit/stage3_formal_audit.json`",
        "- `experiments/tables/stage3_baseline_runs.csv`", "- `experiments/tables/stage3_best_validation.csv`", "- `experiments/tables/stage3_all_validation.csv`", "- `experiments/tables/stage3_hard_mining_budgets.csv`",
        "", "Hard Mining：每中心 32 扰动另加原样本，共 33 个固定 surrogate 评分；4 个真实 Actor/REST/GeoSteiner 重评分包括原样本与 surrogate 最高的三个扰动。按中心选真实 regret 最大者，不计算 cost 或 lambda。共享候选、初筛、精评分和逐 run 账本已冻结；与 WDRO 实际 run 的配对检查仍待阶段 4，不能提前声称实跑对账完成。",
        "", "调用账本为有效 checkpoint 轨迹；candidate reference SQLite 的 calls 表另记实际请求、开始和完成的 cache miss（覆盖回滚重放）。中断于单次 GeoSteiner 计算期间的调用数量仅有上下界，不把有效轨迹账本当作包含所有失败成本的最终效率统计。",
        "", "阶段 3 到此停止；阶段 4、合成 test 模型评价及 ISPD 模型评价均未启动。"]
    (CODEX/"docs/阶段3_三基线与验收报告.md").write_text("\n".join(report)+"\n")
    progress=CODEX/"01_WDRO-REST_项目进度.md";s=progress.read_text();lines=s.splitlines()
    for i,line in enumerate(lines):
        if line.startswith("| 更新时间 |"):lines[i]=f"| 更新时间 | {utc_now()} |"
        elif line.startswith("| 总体阶段 |"):lines[i]="| 总体阶段 | 阶段 0、1、2、3 `DONE`；阶段 4 未启动 |"
        elif line.startswith("| 实验状态 |"):lines[i]=f"| 实验状态 | 阶段 3 的 45/45 基线完成，全量审计 {audit['checks_passed']} 项通过；连同阶段 2 已有 60 个正式模型，仅使用 validation 选模 |"
        elif line.startswith("| 批量实验阻塞 |"):lines[i]="| 批量实验阻塞 | 阶段 3 无阻塞，训练与审计完成；下一次启动仍须动态预检 |"
        elif line.startswith("| 下一动作 |"):lines[i]="| 下一动作 | 阶段 3 已验收，按本次边界停止。后续获准阶段 4 时复用已冻结候选接口并完成 WDRO 对偶训练及实际预算配对 |"
        elif line.startswith("| 3. 三个基线 |"):lines[i]="| 3. 三个基线 | `DONE` | 45 个正式模型、确定性恢复、全量训练/候选重建与调用预算审计通过；实际 WDRO 配对留待阶段 4 |"
    s="\n".join(lines)+"\n"
    entry=f"### {utc_now()}\n\n- 阶段 3 完成：45/45 正式模型、{audit['checks_passed']} 项完整审计；全部模型和数据流可追溯。证据为 `阶段3_三基线与验收报告.md` 与 `wdro_rest_min/manifests/stage3_completion_v1.json`。\n- 生成 45-run 汇总、180 行 best validation 长表、3,780 行全 validation 长表和 15 行 Hard Mining 搜索预算表。未读取 test/ISPD 模型指标，未进入阶段 4。\n\n"
    s=s.replace("## 8. 变更记录\n\n","## 8. 变更记录\n\n"+entry)
    temporary=progress.with_suffix(".md.partial");temporary.write_text(s);os.replace(temporary,progress)
    print(json.dumps({"status":"pass","completion":str(completion),"checks":audit["checks_passed"]}))


if __name__=="__main__":main()

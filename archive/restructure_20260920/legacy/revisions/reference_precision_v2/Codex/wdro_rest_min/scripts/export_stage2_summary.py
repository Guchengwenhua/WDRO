#!/usr/bin/env python3
"""Export validation-only ERM evidence after the complete stage-2 audit passes."""
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CODEX = PROJECT.parent
EXP = CODEX / "experiments"
sys.path.insert(0, str(PROJECT))
from audit_stage2_formal import atomic_write_csv
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256


def descriptor(path):
    return {"logical_path": str(path), "resolved_path": str(path.resolve()), "sha256": file_sha256(path)}


def main():
    audit_path = EXP / "logs/stage2/stage2_formal_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "pass" or audit["audit_scope"] != "15_formal_runs":
        raise RuntimeError("full 15-run audit must pass before stage-2 export")
    records = audit["records"]
    long_rows, curve_rows, degree_rows, artifacts = [], [], [], []
    for record in records:
        run_id = record["run_id"]
        log_root = EXP / "logs/stage2" / run_id
        result = json.loads((log_root / "result.json").read_text())
        events = [json.loads(line) for line in (log_root / "events.jsonl").read_text().splitlines()]
        validations = {event["update"]: event["result"] for event in events if event["event"] == "validation"}
        best = validations[record["best_update"]]
        for update, validation in sorted(validations.items()):
            for group, value in validation["groups"].items():
                row = {"run_id": run_id, "method": "erm", "degree": record["degree"],
                       "training_seed": record["training_seed"], "split": "validation", "validation_seed": 202,
                       "update": update, "is_selected_best": update == record["best_update"],
                       "distribution": group, **value["summary"],
                       "dataset_array_bundle_sha256": value["dataset_array_bundle_sha256"]}
                curve_rows.append(row)
                if update == record["best_update"]:
                    long_rows.append(row | {"best_checkpoint_sha256": record["best_checkpoint_sha256"]})
        artifacts.append({"run_id": run_id, "result": descriptor(log_root / "result.json"),
                          "event_log": descriptor(log_root / "events.jsonl"),
                          "latest": descriptor(Path(result["latest_checkpoint"])),
                          "best": descriptor(Path(result["best_checkpoint"])),
                          "stream_audit": descriptor(Path(audit["stream_artifacts"][run_id]["logical_path"]))})
    for degree in sorted({row["degree"] for row in records}):
        rows = [row for row in records if row["degree"] == degree]
        summary = {"degree": degree, "seeds": "101|102|103", "seed_count": len(rows),
                   "initial_nominal_mean": rows[0]["initial_nominal_mean"],
                   "initial_ood_worst_group_mean": rows[0]["initial_ood_worst_group_mean"],
                   "best_updates": "|".join(str(row["best_update"]) for row in rows)}
        for metric in ("best_nominal_mean", "best_ood_worst_group_mean", "final_nominal_mean", "final_ood_worst_group_mean"):
            summary[f"{metric}_across_seeds"] = statistics.mean(row[metric] for row in rows)
            summary[f"{metric}_sample_sd"] = statistics.stdev(row[metric] for row in rows)
        summary["maximum_peak_reserved_mib"] = max(row["peak_memory_reserved_mib"] for row in rows)
        summary["maximum_nominal_delta_vs_initial"] = max(row["best_nominal_mean"] - row["initial_nominal_mean"] for row in rows)
        degree_rows.append(summary)
    tables = []
    for name, rows in (("stage2_erm_best_validation_long.csv", long_rows),
                       ("stage2_erm_validation_curves.csv", curve_rows),
                       ("stage2_erm_degree_summary.csv", degree_rows)):
        path = EXP / "tables" / name
        atomic_write_csv(path, rows)
        path.chmod(0o444)
        tables.append(descriptor(path) | {"rows": len(rows)})
    audit_script = PROJECT / "scripts/audit_stage2_formal.py"
    if file_sha256(audit_script) != audit["audit_code_sha256"]:
        raise RuntimeError("auditor changed after final audit")
    summary = {
        "schema_version": 1, "kind": "stage2_erm_completion_manifest", "status": "pass",
        "captured_at_utc": utc_now(), "research_protocol_override": False, "next_stage_started": False,
        "formal_runs": len(records), "policy_updates_per_run": 2000, "batch_size": 256,
        "training_nets_total": 15 * 2000 * 256, "training_streams_rebuilt": 15,
        "train_validation_test_content_intersections": 0,
        "nominal_improved_runs": sum(row["best_nominal_mean"] < row["initial_nominal_mean"] for row in records),
        "maximum_nominal_degradation": max(row["best_nominal_mean"] - row["initial_nominal_mean"] for row in records),
        "call_ledger_total_effective": {key: 15 * count for key, count in audit["expected_call_ledger_per_run"].items()},
        "accounting_limitations": audit["accounting_scope"],
        "training_pool_representation": "immutable stateless per-update pool; source/seed/ID/batch hashes fully reconstructed",
        "reference_storage": "validation/test L_star cached by dataset hash; stage2 training L_star computed exactly online, not persistently cached",
        "inference_scope": "T=1 greedy validation only; no test or ISPD model evaluation",
        "audit": descriptor(audit_path), "tables": tables, "degree_summary": degree_rows,
        "unit_test_log": descriptor(EXP / "logs/stage2/stage2_unit_tests_20260905.log"),
        "artifacts": artifacts,
        "audit_sources": [descriptor(PROJECT / relative) for relative in
                           ("src/stage2_audit_checks.py", "scripts/audit_stage2_formal.py",
                            "scripts/launch_stage2_completion_audit.py", "scripts/export_stage2_summary.py",
                            "tests/test_stage2_audit.py")],
    }
    destination = PROJECT / "manifests/stage2_completion_20260905.json"
    atomic_write_json(destination, summary, mode=0o444)
    print(json.dumps({"manifest": str(destination), "status": "pass", "degree_summary": degree_rows}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent re-read of completed stage-4 artifacts before DONE publication."""
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments'
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now,process_is_alive
from src.rest import file_sha256
from src.trainer import canonical_sha256
from audit_stage3 import effective_events
RUN='stage4_stability_d30_s101_u500_v1'

def main():
    completion_path=ROOT/'manifests/stage4_completion_v1.json'
    completion=json.loads(completion_path.read_text())
    pipeline=json.loads((EXP/'logs/stage4_pipeline/heartbeat.json').read_text())
    assert completion['status']=='pass' and completion['stage']==4 and pipeline['status']=='complete'
    checks={}
    for i,entry in enumerate(completion['artifact_index']):
        path=Path(entry['logical_path'])
        checks[f'artifact_{i:02d}']=str(path).startswith(str(CODEX)+'/') and str(path.resolve())==entry['resolved_path'] and file_sha256(path)==entry['sha256'] and path.stat().st_size==entry['bytes']
    result=json.loads((EXP/'logs/stage4'/RUN/'result.json').read_text())
    latest=torch.load(result['latest_checkpoint'],map_location='cpu',weights_only=False)
    best=torch.load(result['best_checkpoint'],map_location='cpu',weights_only=False)
    train,validation,raw=effective_events(result['event_log'])
    audit=json.loads((EXP/'logs/stage4_audit/runs'/f'{RUN}.json').read_text())
    cursor=json.loads(Path(audit['stream_cursor']).read_text())
    checks['full_audit_counts']=audit['status']=='pass' and all(audit['checks'].values()) and audit['training_nets_checked']==128000 and audit['candidate_nets_checked']==4224000 and audit['cache_values_checked']==512000 and cursor['completed_updates']==500 and cursor['heldout_intersections']==0
    checks['audit_binding_current']=audit['binding']['result_sha256']==file_sha256(EXP/'logs/stage4'/RUN/'result.json') and audit['binding']['events_sha256']==file_sha256(Path(result['event_log'])) and audit['binding']['auditor_sha256']==file_sha256(ROOT/'scripts/audit_stage4.py')
    erm_path=EXP/'logs/stage2/formal_erm_d30_s101_u2000/result.json';erm=json.loads(erm_path.read_text())
    _,ermval,_=effective_events(erm['event_log'])
    checks['same_dac21_initialization']=result['initialization']==erm['initialization'] and validation[0]['result']==ermval[0]['result']
    checks['independent_nominal_guard']=result['nominal_limit']==erm['best']['nominal_mean']+.002 and result['run_identity']['erm_reference_result_sha256']==file_sha256(erm_path)
    checks['strict_stage4_scope']=result['degree']==30 and result['training_seed']==101 and result['batch_size']==256 and result['completed_updates']==500 and latest['update']==500 and not completion['formal_2000_update_model'] and not completion['stage5_started']
    checks['no_test_ispd_selection']=not result['test_or_ispd_used'] and all(not e['result']['selection']['test_or_ispd_used'] for e in validation.values())
    contract=json.loads((ROOT/'manifests/stage4_training_contract_v1.json').read_text())
    for i,(name,digest) in enumerate(contract['files'].items()):
        checks[f'frozen_file_{i:02d}']=file_sha256(CODEX/name)==digest
    table=EXP/'tables/stage4_budget_trajectory.csv'
    with table.open() as f: rows=list(csv.DictReader(f))
    checks['trajectory_keys']=len(rows)==500 and [int(r['update']) for r in rows]==list(range(1,501))
    checks['trajectory_values']=all(float(row['lambda_after'])==train[int(row['update'])]['method_detail']['lambda_after'] and float(row['mean_selected_cost'])==train[int(row['update'])]['method_detail']['mean_selected_cost'] and row['support_sha256']==train[int(row['update'])]['method_detail']['support_sha256'] for row in rows)
    with (EXP/'tables/stage4_validation_diagnostic.csv').open() as f: vals=list(csv.DictReader(f))
    checks['validation_table']=len(vals)==6 and [int(r['update']) for r in vals]==[0,100,200,300,400,500] and [int(r['update']) for r in vals if r['selected']=='True']==[best['update']]
    with (EXP/'tables/stage4_actual_hm_prefix_budget.csv').open() as f: pairs=list(csv.DictReader(f))
    checks['actual_pair_table']=len(pairs)==2 and all(pairs[0][k]==pairs[1][k] for k in ['paired_start_update','paired_end_update','centers','perturbations','surrogate_nets','exact_rescore_nets','reference_requests']) and pairs[0]['full_run_updates']=='500' and pairs[1]['full_run_updates']=='2000'
    from PIL import Image
    with Image.open(EXP/'figures/stage4_lambda_cost_rho.png') as im:im.verify()
    checks['png_decodes']=True
    checks['pdf_header']=(EXP/'figures/stage4_lambda_cost_rho.pdf').read_bytes().startswith(b'%PDF')
    import xml.etree.ElementTree as ET
    checks['svg_parses']=ET.parse(EXP/'figures/stage4_lambda_cost_rho.svg').getroot().tag.endswith('svg')
    # Compare the actual reference values against independently audited HM cache.
    wdro_db=sqlite3.connect(f"file:{result['reference_cache']['logical_path']}?mode=ro",uri=True)
    hm_db_path=EXP/'cache/stage3_references/formal_hard_mining_d30_s101_u2000/lstar.sqlite'
    hm_db=sqlite3.connect(f'file:{hm_db_path}?mode=ro',uri=True)
    matched=0
    for digest,value in wdro_db.execute('SELECT hash,length FROM lstar'):
        old=hm_db.execute('SELECT length FROM lstar WHERE hash=?',(digest,)).fetchone()
        assert old is not None and old[0]==value,'WDRO reference differs from certified HM cache'
        matched+=1
    wdro_db.close();hm_db.close()
    checks['reference_values_match_certified_hm']=matched==512000
    job=json.loads((EXP/'logs/jobs'/RUN/'job.json').read_text())
    checks['training_exited']=not process_is_alive(job['pid'])
    locks=[EXP/'logs/jobs'/RUN/'run.lock',EXP/'logs/stage4_audit/jobs'/RUN/'run.lock',EXP/'logs/stage4_pipeline/run.lock']
    checks['locks_released']=all(not p.exists() for p in locks)
    checks['no_partial']=not list((EXP/'checkpoints/stage4'/RUN).glob('*.partial'))
    segments=list((EXP/'logs/stage4'/RUN).glob('segment_u*.json'))
    segments_data=[json.loads(p.read_text()) for p in segments]
    checks['continuous_500_update_trajectory']=bool(segments_data) and min(s['started_this_segment_at_update'] for s in segments_data)==1 and max(s['completed_updates'] for s in segments_data)==500 and sorted(train)==list(range(1,501))
    output={'status':'pass' if all(checks.values()) else 'fail','at_utc':utc_now(),'stage':4,
       'checks':checks,'completion_sha256':file_sha256(completion_path),'reference_values_matched':matched,
       'stage5_started':False,'test_ispd_models_evaluated':False}
    target=ROOT/'manifests/stage4_delivery_verification_v1.json';atomic_write_json(target,output)
    assert output['status']=='pass',[k for k,v in checks.items() if not v]
    p=CODEX/'01_WDRO-REST_项目进度.md';s=p.read_text();now=utc_now()
    s=re.sub(r'\| 更新时间 \| .*? \|',f'| 更新时间 | {now} |',s,count=1)
    s=re.sub(r'\| 总体阶段 \| .*? \|','| 总体阶段 | 阶段 0–4 `DONE`；阶段 5 未启动 |',s,count=1)
    s=re.sub(r'\| 实验状态 \| .*? \|',f"| 实验状态 | 阶段 4 的 500 updates、{completion['full_audit_checks']} 项全量审计及 {len(checks)} 项交付复核通过；既有 60 个正式基线保留，WDRO 此处仅为诊断模型 |",s,count=1)
    s=re.sub(r'\| 批量实验阻塞 \| .*? \|','| 批量实验阻塞 | 阶段 4 无阻塞，已完成；下一阶段需另行授权且重新动态预检 |',s,count=1)
    s=re.sub(r'\| 下一动作 \| .*? \|','| 下一动作 | 按本次阶段 4 边界停止。后续获准阶段 5 时仅执行原计划最小 validation 搜索与消融 |',s,count=1)
    s=re.sub(r'\| 4\. Random-WDRO/对偶审计 \| .*? \| .*? \|',f"| 4. Random-WDRO/对偶审计 | `DONE` | 69 单测、精确恢复/零半径、500 updates、{completion['full_audit_checks']} 项全量审计、实际 HM 500-update 区间预算配对与独立交付通过 |",s,count=1)
    s=s.replace('- `DOING` 4.3：仅 d30/s101 连续目标 500 updates，batch 256、K32；','- `DONE` 4.3：d30/s101 的 500 updates 与全部输入/cost/对偶审计通过，batch 256、K32；')
    s=s.replace('6. 阶段 3 已完成。本次用户仅授权阶段 4，按 4.1 → 4.2 → 4.3 子步骤验收；完成阶段 4 后停止，阶段 5 及 test/ISPD 模型评价保持关闭。','6. 阶段 4 已完成，验收依据 `stage4_completion_v1.json` 与 `stage4_delivery_verification_v1.json`。按本次授权停止，阶段 5 及 test/ISPD 模型评价保持关闭。')
    diag=completion['diagnostics']
    entry=f'''### {now}

- 阶段 4 全部验收通过：69 单测、12 vs 10+resume 精确恢复、rho=0/ERM 精确更新、500 updates、{completion['full_audit_checks']} 项完整审计及 {len(checks)} 项独立交付复核。报告 `阶段4_Random-WDRO与数学审计报告.md`；索引 `wdro_rest_min/manifests/stage4_completion_v1.json`、`stage4_delivery_verification_v1.json`。
- 全量重建 128,000 中心和 4,224,000 候选，held-out 内容交集 0；512,000 个实际 reference 值与已审计 HM cache 完全一致。逐 update 支持、shortlist、精评分/evaluator-call 预算与 HM 实际前 500 updates 配对通过；没有声称 WDRO 500 与 HM 2000 整 run 等预算。
- 正半径 rho={diag['rho']:.12f}，最终 lambda={diag['lambda_final']:.8f}，末尾两个 100-update 平均 cost={diag['tail_100_update_window_means']}，均在预先冻结的 10% 容差内。可导出 PNG/PDF/SVG 与四张 CSV 位于 `experiments/figures`、`experiments/tables/stage4_*.csv`。
- 训练、审计完成且目标锁均释放；持久 job/启动/恢复/资源监控历史保留在 `experiments/logs/stage4_pipeline`。交付复核可由环境 wrapper 重跑 `scripts/verify_stage4_delivery.py`。原协议、数据契约及冻结基线未改动；阶段 5、test/ISPD 模型评价未启动。

'''
    s=s.replace('## 8. 变更记录\n\n','## 8. 变更记录\n\n'+entry)
    temp=p.with_suffix('.md.partial');temp.write_text(s);temp.replace(p)
    print(json.dumps({'status':'pass','checks':len(checks),'manifest':str(target),'stage4':'DONE'}))

if __name__=='__main__':main()

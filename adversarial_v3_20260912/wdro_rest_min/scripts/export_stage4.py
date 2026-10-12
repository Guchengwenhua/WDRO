#!/usr/bin/env python3
"""Export stage-4 diagnostic tables, scientific plots, and evidence index."""
import csv
import io
import json
import os
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments'
sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.rest import file_sha256
from audit_stage3 import effective_events
RUN='stage4_stability_d30_s101_u500_v1'

def artifact(path):
    path=Path(path)
    return {'logical_path':str(path),'resolved_path':str(path.resolve()),'sha256':file_sha256(path),'bytes':path.stat().st_size}

def atomic_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(text);os.replace(tmp,path)

def csv_file(path,rows):
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=out.getvalue();assert len(list(csv.DictReader(io.StringIO(text))))==len(rows)
    atomic_text(path,text)

def main():
    result_path=EXP/'logs/stage4'/RUN/'result.json';result=json.loads(result_path.read_text())
    audit_path=EXP/'logs/stage4_audit/runs'/f'{RUN}.json';audit=json.loads(audit_path.read_text())
    smoke_path=EXP/'logs/stage4_smoke/audit.json';smoke=json.loads(smoke_path.read_text())
    score_audit_path=EXP/'logs/stage4/independent_initial_score_audit.json'
    score_audit=json.loads(score_audit_path.read_text());assert score_audit['status']=='pass'
    best_audit_path=EXP/'logs/stage4/best_validation_recompute.json'
    best_audit=json.loads(best_audit_path.read_text());assert best_audit['status']=='pass' and best_audit['checkpoint_sha256']==result['best_checkpoint_sha256']
    assert result['completed_updates']==500 and result['status']=='complete'
    assert audit['status']==smoke['status']=='pass' and audit['binding']['result_sha256']==file_sha256(result_path)
    train,validation,raw=effective_events(result['event_log'])
    assert len(train)==500 and sorted(validation)==[0,100,200,300,400,500]
    tables=EXP/'tables';figures=EXP/'figures'
    rows=[]
    for u,e in sorted(train.items()):
        d=e['method_detail']
        row={'run_id':RUN,'degree':30,'seed':101,'update':u,
             **{k:d[k] for k in ['lambda_before','lambda_after','mean_selected_cost','rho','budget_gap','mean_cost_last100','legal_candidate_fraction']},
             **{k:d[k] for k in ['centers','perturbations','surrogate_nets','exact_rescore_nets','reference_requests','reference_cache_misses','reference_cache_hits']},
             'mean_training_regret':e['mean_regret'],'p95_training_regret':e['p95_regret'],
             'support_sha256':d['support_sha256'],'shortlist_sha256':d['shortlist_sha256']}
        rows.append(row)
    trajectory=tables/'stage4_budget_trajectory.csv';csv_file(trajectory,rows)
    validation_rows=[]
    for u,e in sorted(validation.items()):
        v=e['result']
        validation_rows.append({'run_id':RUN,'degree':30,'seed':101,'update':u,
            'nominal_mean':v['selection']['nominal_mean'],'ood_worst_group_mean':v['selection']['worst_group_mean'],
            'nominal_limit':result['nominal_limit'],'feasible':v['selection']['nominal_mean']<=result['nominal_limit'],
            'selected':u==result['best']['update'],'split':'validation'})
    validation_table=tables/'stage4_validation_diagnostic.csv';csv_file(validation_table,validation_rows)
    paired=audit['actual_hard_mining_pair'];budget=paired['budget']
    pair_rows=[{'method':m,'run_id':rid,'paired_start_update':1,'paired_end_update':500,
       'full_run_updates':n,**budget} for m,rid,n in [('wdro_rest',RUN,500),('hard_mining',paired['run_id'],2000)]]
    paired_table=tables/'stage4_actual_hm_prefix_budget.csv';csv_file(paired_table,pair_rows)
    lp_rows=[{'update':v['update'],'rho':v['rho'],'primal_value':v['primal_value'],'dual_value':v['dual_value'],
             'gap':v['gap'],'lambda':v['lambda'],'mean_cost':v['mean_cost'],'status':v['status']} for v in audit['real_lp']]
    lp_table=tables/'stage4_real_candidate_lp.csv';csv_file(lp_table,lp_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.asarray([r['update'] for r in rows]);rho=rows[0]['rho']
    cost=np.asarray([r['mean_selected_cost'] for r in rows]);price=np.asarray([r['lambda_after'] for r in rows]);ma=np.asarray([r['mean_cost_last100'] for r in rows])
    fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True,constrained_layout=True)
    axes[0].plot(x,price,color='#375aab',lw=1.5);axes[0].set_ylabel('Transport price lambda')
    axes[0].set_title('Stage 4 implementation diagnostic | degree 30, seed 101')
    axes[1].plot(x,cost,color='#629ba0',alpha=.4,lw=.8,label='Per-update mean cost')
    axes[1].plot(x,ma,color='#13786c',lw=1.8,label='Trailing 100-update mean')
    axes[1].axhline(rho,color='#b1492d',ls='--',label=f'Training q50 rho = {rho:.6f}')
    axes[1].set_ylabel('Hungarian-L1 mean cost');axes[1].legend(fontsize=8)
    axes[2].plot(x,ma-rho,color='#75548d',lw=1.5);axes[2].axhline(0,color='black',lw=.6)
    axes[2].axhspan(-.1*rho,.1*rho,color='#8fb78a',alpha=.2,label='Predeclared +/-10% rho tolerance')
    axes[2].set_ylabel('Trailing cost minus rho');axes[2].set_xlabel('Policy updates');axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.18)
    figures.mkdir(parents=True,exist_ok=True);plot_files=[]
    for ext in ['png','pdf','svg']:
        path=figures/f'stage4_lambda_cost_rho.{ext}';tmp=path.with_suffix(path.suffix+'.partial')
        fig.savefig(tmp,format=ext,dpi=180)
        assert tmp.stat().st_size>1000;os.replace(tmp,path);plot_files.append(path)
    plt.close(fig)
    diagnostic=audit['diagnostics']
    report=CODEX/'阶段4_Random-WDRO与数学审计报告.md'
    mean_windows=diagnostic['tail_100_update_window_means']
    text=f'''# 阶段 4 Random-WDRO 与数学审计

更新时间：{utc_now()}。阶段 4 的实现与数值门禁已通过；独立交付复核完成后由动态进度文件确认 DONE。阶段 5 未启动。

## 实现与冻结边界

WDRO 入口 `wdro_rest_min/scripts/train_wdro.py`，新增代码 `wdro_rest_min/stage4/dual.py` 与 `trainer.py`。原 `protocol_v1`、`data_contract_v1`、stage2 trainer 和 stage3 baseline/search 源码及配置 hash 均保持不变。

每个中心使用阶段 3 原样的 32 个合法扰动加原样本，固定 surrogate 初筛后，原样本加 top-3 扰动接受真实 Actor/REST 与缓存 GeoSteiner 精评分。WDRO 逐中心选择 `regret - lambda * Hungarian-L1 cost` 的最大值；lambda 更新为 `[lambda + eta * (mean_cost-rho)]_+`。Critic 仅为方差降低基线。

本阶段仅 d30/s101、batch 256、500 updates；optimizer、初始化、validation 每 100 updates 和 checkpoint schema 沿用冻结实现。诊断 rho 使用训练 update 1 的 256×32 扰动 cost q50：`{rho:.14f}`；lambda 初始 0，eta=1。这不是阶段 5 的 validation 超参选择，不将本次 500-update 模型列入 2000-update 正式主实验模型。

## 数学、恢复和实际预算证据

- 69/69 单元测试通过：`experiments/logs/stage4_all_unit_tests_v1.log`。
- 连续 12 vs 10+恢复到12 的 Actor/Critic/optimizer/scheduler/RNG、训练流、方法状态、调用账本、初始化和 best 完全一致，跨 update 11 候选刷新；`experiments/logs/stage4_smoke/audit.json`。
- rho=0 使用精确零运输可行集（lambda→∞ 的边界问题），两次实际更新的模型/optimizer/RNG 与 ERM 完全一致。此分支的 lambda=0 不用来证明正半径的活跃预算行为。
- 前台两份完整候选与对偶审计 76/76 项通过；500-update 审计 {len(audit['checks'])}/{len(audit['checks'])} 项通过。全量重建 128,000 中心和 4,224,000 候选，全部合法、每中心去重，held-out 内容交集为 0；逐项重算 Hungarian cost、逐中心选择、lambda、均值/滑动均值、512,000 个 cache reference 值及完整 stream hashes。
- 小有限支持 LP 明确为每中心条件概率质量和平均运输预算，允许分数质量。4 个已知 toy LP、smoke 与正式诊断真实分数 LP 均通过；本次真实 LP 最大绝对 gap 为 `{max(abs(v['gap']) for v in audit['real_lp']):.3g}`。LP 只证明所审计有限支持集上的原始/对偶等价，不能证明连续几何空间中内层全局最优。
- 独立重新加载 DAC21 Actor，重算全部 1,024 个 update-1 shortlist 的 REST 与 GeoSteiner 长度，真实 regret、L_star 和 cost 均与训练日志精确相同；另计 1,024 次审计 GeoSteiner calls。证据 `experiments/logs/stage4/independent_initial_score_audit.json`。
- 与已验收 `formal_hard_mining_d30_s101_u2000` 的真实 updates 1–500 配对：同中心、support hash、shortlist hash、刷新编号和预算逐 update 相同。两者配对区间均为中心 {budget['centers']:,}、扰动 {budget['perturbations']:,}、surrogate {budget['surrogate_nets']:,}、精评分 {budget['exact_rescore_nets']:,}、reference requests {budget['reference_requests']:,}。比较的是实际相同区间；HM 全 run 仍为 2000 updates。正式 WDRO 的完整逐 run 配对在阶段 6 补齐。

## 500-update 预算稳定性

lambda 最后 `{diagnostic['lambda_final']:.8f}`，轨迹最大 `{diagnostic['lambda_max']:.8f}`。最后两个不重叠的 100-update 窗口平均 cost 为 `{mean_windows[0]:.10f}`、`{mean_windows[1]:.10f}`；相对 rho 的误差分别为 `{abs(mean_windows[0]-rho)/rho:.3%}`、`{abs(mean_windows[1]-rho)/rho:.3%}`，满足运行前冻结的 10% 容差。整个 run 的均值 `{diagnostic['mean_cost']:.10f}` 包含从 lambda=0 启动的过渡期；稳定门禁使用预先指定的尾部窗口。

图：`experiments/figures/stage4_lambda_cost_rho.png`（同名 PDF/SVG 可导出）。逐 update 原值见 `experiments/tables/stage4_budget_trajectory.csv`。这些诊断证明本实现的预算行为，不是 WDRO 性能优势或多 seed 稳定改善的结论。

## Validation 与调用账本

仅使用合成 validation 的 6 个时点，best update `{result['best']['update']}`；nominal mean regret `{result['best']['nominal_mean']:.10f}`，对应 ERM guard 上限 `{result['nominal_limit']:.10f}`，OOD worst-group mean `{result['best']['selection_objective']:.10f}`。test 与 ISPD 模型指标未读取，也未用来选模。输入审计读取 test 坐标/hash 仅用于检查内容交集。

选中 checkpoint 独立重新加载后，8,000 个 validation nets 的全部指标与选模日志精确一致：`experiments/logs/stage4/best_validation_recompute.json`。额外 Actor/REST 8,000 calls 为交付复核开销，独立记账。

有效轨迹调用账本：`{json.dumps(result['call_ledger'],sort_keys=True)}`。reference cache 的实际请求/开始/完成 receipts 单独见完整审计，失败重试不可混为有效轨迹。训练 wall-clock 和实际 GPU 映射见各 segment 与 pipeline 记录；实测 reserved 峰值 `{result['peak_memory_reserved_bytes']/2**20:.0f}` MiB。

## 产物与恢复

逻辑目录 `/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments`，resolved `/attached/remote-home1/Guilin/DRO/Codex/experiments`。checkpoint、cache、临时目录、CUDA/Torch/Matplotlib cache、tmux socket、PID/锁/心跳和日志均经 Codex 内路径访问。

恢复入口（完成状态会拒绝重复启动）：

```bash
bash /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/launch_stage4.py
```

审计回放：同一环境 wrapper 执行 `wdro_rest_min/scripts/audit_stage4.py --run-id {RUN} --stability`；每 10 updates 保存绑定配置/数据/hash 的原子游标。汇总重建执行 `scripts/export_stage4.py`。持久任务精确命令、PID、GPU、逐次 preflight/post-launch、恢复与监控见 `experiments/logs/stage4_pipeline/launch.json`、`launches.jsonl` 和 `resource_monitor.jsonl`。

后续仅在授权阶段 5 时执行已冻结计划中的最小 validation 搜索与消融。
'''
    atomic_text(report,text)
    files=[result_path,audit_path,smoke_path,score_audit_path,best_audit_path,Path(result['latest_checkpoint']),Path(result['best_checkpoint']),
       Path(result['event_log']),trajectory,validation_table,paired_table,lp_table,*plot_files,report,
       ROOT/'configs/stage4_wdro_v1.json',ROOT/'manifests/stage4_training_contract_v1.json',
       EXP/'cache/stage4/calibration_d30_s101.json',EXP/'logs/stage4_pipeline/launch.json',
       EXP/'logs/stage4_all_unit_tests_v1.log',EXP/'logs/stage4/math_lp_toy.json',
       *[ROOT/'scripts'/name for name in ['train_wdro.py','calibrate_stage4.py','run_stage4_smoke.py',
         'audit_stage4.py','audit_stage4_initial_scores.py','stage4_pipeline.py','launch_stage4.py',
         'export_stage4.py','verify_stage4_delivery.py','verify_stage4_best_validation.py']],
       CODEX/'阶段4_子步骤与运行说明.md']
    completion={'status':'pass','stage':4,'at_utc':utc_now(),'run_id':RUN,'updates':500,
       'formal_2000_update_model':False,'stage5_started':False,'test_ispd_models_evaluated':False,
       'unit_tests':69,'full_audit_checks':len(audit['checks']),'diagnostics':diagnostic,
       'artifact_index':[artifact(p) for p in files],
       'table_rows':{'budget_trajectory':500,'validation':6,'actual_hm_prefix_budget':2,'lp':len(lp_rows)}}
    atomic_write_json(ROOT/'manifests/stage4_completion_v1.json',completion)
    print(json.dumps({'status':'pass','manifest':str(ROOT/'manifests/stage4_completion_v1.json')}),flush=True)

if __name__=='__main__':main()

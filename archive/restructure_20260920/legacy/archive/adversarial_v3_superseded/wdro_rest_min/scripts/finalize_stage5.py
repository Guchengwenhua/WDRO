#!/usr/bin/env python3
"""Mark DONE only after independent delivery verification of all nine runs."""
import csv,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.rest import file_sha256
from src.job_control import utc_now
from export_stage4 import atomic_text

def main():
    cp=ROOT/'manifests/stage5_completion_v1.json';vp=ROOT/'manifests/stage5_delivery_verification_v1.json'
    completion=json.loads(cp.read_text());verify=json.loads(vp.read_text())
    assert completion['status']==verify['status']=='pass' and verify['completion_sha256']==file_sha256(cp)
    rows=list(csv.DictReader((EXP/'tables/stage5_ablation.csv').open()));selected=next(r for r in rows if r['selected_main']=='True')
    now=utc_now();report=f'''# 阶段 5：最小 validation 搜索与消融验收

验收时间：{now}。阶段 5 DONE；阶段 6 未启动。

全部 9 个预声明 run 均为 degree=30、seed=101、2,000 updates、batch 256，共 18,000 updates。各 run 完整重建训练中心、全部候选、cost、逐中心 argmax、lambda 和缓存 reference；独立重载 best 后重算 8,000 validation nets。原协议、数据契约及阶段 0–4 资产未改动。

主配置选择只比较 rho=0/q50/q75 三个规范主方法配置；nominal 约束为同 degree/seed ERM best validation mean +0.002，选择最早可行 checkpoint，再取最小 OOD worst-group。消融候选不得替换 Random/Sobol K32、Hungarian、learned lambda 主方法。

选中 rho={selected['rho_mode']}，degree30 数值={selected['rho']}；best update={selected['best_update']}，nominal mean={selected['nominal_mean']}，OOD worst-group mean={selected['validation_worst_group_mean']}。

| 消融 | rho | lambda | 搜索 | K扰动数 | ground cost | best update | nominal mean | OOD worst-group |
|---|---|---|---|---|---|---|---|---|
'''
    for r in rows:report+='| '+' | '.join(r[k] for k in ['ablation','rho_mode','lambda_mode','searcher','perturbation_count','ground_cost','best_update','nominal_mean','validation_worst_group_mean'])+' |\n'
    report+='''
Random-64 与 CEM-64 均每中心每 update 恰好 64 次真实 Actor/REST/reference 评分请求，各 32,768,000 次；原样本计入 64。CEM 为 4轮×16，elite4；几何 surrogate 调用均为零。policy update 和 validation 另行计数。K=1 只有原样本和一个扰动，精评分为 2；K=8/32 精评分为 4。只有未改变 K32 Random support 的 runs 与既有 Hard Mining 对账。

fixed lambda 来自选中 learned-rho run 最后 100 updates 的 lambda_after 平均，值在 fixed run 启动前冻结。Hungarian/Chamfer 各自使用相同训练候选的独立 quantile 标定。预算不活跃或较小 K 的候选支持不足以达到 rho 时，如实报告，不把 lambda=0 自动称作实现错误。

产物：

- `wdro_rest_min/configs/stage5_plan_v1.json`：预声明范围与运行前实现选择。
- `wdro_rest_min/configs/stage5_selected_wdro_v1.json`：只读最终主配置，跨 degree 仅按同一选中 quantile 做训练数据标定，不做重复网格搜索。
- `experiments/tables/stage5_ablation.csv`、`stage5_validation_long.csv`、`stage5_budget_trajectory.csv`：全部消融、756 行 validation 与18,000 行预算轨迹。
- `experiments/figures/stage5_budget_diagnostic.{png,pdf,svg}` 与 `stage5_all_ablation_budgets.png`。
- `wdro_rest_min/manifests/stage5_completion_v1.json` 与 `stage5_delivery_verification_v1.json`：hash、逻辑/resolved 路径和独立验收。
- `experiments/logs/stage5_pipeline/launch.json`、`queue.json`、`launches.jsonl`、`recoveries.jsonl`：持久会话、PID、精确命令、运行/恢复与资源证据；无恢复事件时 recoveries 文件可能不存在。

该阶段只有单 degree/seed 的 validation 消融，不构成 test 显著性或 WDRO 优越性证据。不利配置全部保留。成功轨迹 calls 与 cache receipts 中的实际重试成本分别保存；最终效率分析不得省略失败/回滚成本。没有读取 test/ISPD 模型指标，后续不得据 test 回调本配置。

复核命令：`bash wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python wdro_rest_min/scripts/verify_stage5_delivery.py`。
'''
    atomic_text(CODEX/'阶段5_Validation搜索与消融验收报告.md',report)
    p=CODEX/'01_WDRO-REST_项目进度.md';s=p.read_text()
    for key,value in {'更新时间':now,'总体阶段':'阶段 0–5 `DONE`；阶段 6 未启动','实验状态':'阶段 5 九个 run、完整候选/cost/对偶审计与独立 validation 重算及交付复核通过','批量实验阻塞':'阶段 5 无阻塞，已完成；后续阶段需另行授权和动态预检','下一动作':'按本次阶段 5 边界停止；后续获准阶段 6 时使用已冻结配置，禁止 test 后回调'}.items():
        s=re.sub(r'\| '+re.escape(key)+r' \| .*? \|',f'| {key} | {value} |',s,count=1)
    s=re.sub(r'\| 5\. Validation/消融 \| .*?\n','| 5. Validation/消融 | `DONE` | 9 个 d30/s101 runs、18,000 updates、完整审计、validation-only 选模、消融表/预算图和最终主配置冻结 |\n',s,count=1)
    entry=f'### {now}\n\n- 阶段 5 完整验收通过；9 个最小 run、18,000 updates，选中 rho={selected["rho_mode"]}。完整报告 `阶段5_Validation搜索与消融验收报告.md`；证据 `wdro_rest_min/manifests/stage5_completion_v1.json`、`stage5_delivery_verification_v1.json`。\n- 主配置保持 Random/Sobol K32、Hungarian、learned lambda；只用 validation，未启动阶段 6/test/ISPD。未来按 `stage5_selected_wdro_v1.json` 执行，不重新搜索或 test 后回调。\n\n'
    s=re.sub(r'6\. `DOING` 本次仅阶段 5 已授权。.*?\n', '6. `DONE` 阶段5完整验收，证据stage5_completion_v1.json与stage5_delivery_verification_v1.json。本次到阶段5停止，阶段6/test/ISPD未启动。\n',s,count=1)
    for sub in ['5.1','5.2','5.3','5.4']:
        s=re.sub(r'- `(DOING|TODO)` '+re.escape(sub)+r'：.*?\n', '- `DONE` '+sub+'：完成；证据与各项统计见 `阶段5_Validation搜索与消融验收报告.md` 和 stage5_completion/delivery manifests。\n',s,count=1)
    s=s.replace('## 8. 变更记录\n\n','## 8. 变更记录\n\n'+entry)
    atomic_text(p,s)
if __name__=='__main__':main()

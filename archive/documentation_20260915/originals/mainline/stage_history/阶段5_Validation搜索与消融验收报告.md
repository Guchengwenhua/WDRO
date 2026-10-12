# 阶段 5：最小 validation 搜索与消融验收

验收时间：2026-09-06T12:54:31.185150Z。阶段 5 DONE；阶段 6 未启动。

全部 9 个预声明 run 均为 degree=30、seed=101、2,000 updates、batch 256，共 18,000 updates。各 run 完整重建训练中心、全部候选、cost、逐中心 argmax、lambda 和缓存 reference；独立重载 best 后重算 8,000 validation nets。原协议、数据契约及阶段 0–4 资产未改动。

主配置选择只比较 rho=0/q50/q75 三个规范主方法配置；nominal 约束为同 degree/seed ERM best validation mean +0.002。每个 rho run 在满足 nominal 约束的全部 validation checkpoints 中选择 OOD worst-group mean 最小者，目标相同时取最早 checkpoint；再比较三个 rho run 的所选目标，目标相同时按预声明顺序 zero、q50、q75 选择。消融候选不得替换 Random/Sobol K32、Hungarian、learned lambda 主方法。

选中 rho=q50，degree30 数值=0.01381177343428135；best update=1700，nominal mean=0.007651515352657595，OOD worst-group mean=0.012895576614701535。

| 消融 | rho | lambda | 搜索 | K扰动数 | ground cost | best update | nominal mean | OOD worst-group |
|---|---|---|---|---|---|---|---|---|
| rho_zero | zero | learned | random_sobol | 32 | hungarian_l1_mean | 800 | 0.007447040644646114 | 0.012993802973162992 |
| rho_q50 | q50 | learned | random_sobol | 32 | hungarian_l1_mean | 1700 | 0.007651515352657595 | 0.012895576614701535 |
| rho_q75 | q75 | learned | random_sobol | 32 | hungarian_l1_mean | 2000 | 0.007625632368754807 | 0.012966652987708502 |
| lambda_fixed | q50 | fixed | random_sobol | 32 | hungarian_l1_mean | 1100 | 0.007539015867191993 | 0.012974768508333482 |
| k1 | q50 | learned | random_sobol | 1 | hungarian_l1_mean | 1700 | 0.007719467928299691 | 0.012940832996624109 |
| k8 | q50 | learned | random_sobol | 8 | hungarian_l1_mean | 1500 | 0.007641443412704838 | 0.012868006020881747 |
| chamfer | q50 | learned | random_sobol | 32 | symmetric_chamfer_l1 | 2000 | 0.007508959720804908 | 0.012811844465950694 |
| random_64 | q50 | learned | random_64 | 63 | hungarian_l1_mean | 1300 | 0.007458127385270957 | 0.012679249580431769 |
| cem_64 | q50 | learned | cem_64 | 63 | hungarian_l1_mean | 1300 | 0.007643547491671283 | 0.012668099058090566 |

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

交接文字校正：本报告的 checkpoint 选择顺序说明已按冻结的 `stage5_plan_v1.json` tie_rule、`stage5/trainer.py` 和完整审计实现澄清；训练、选模结果、配置与表图未变。原报告及校正证据见 `experiments/logs/stage5_final_delivery_review/`。只读导出器未改动，重新生成报告后需再次运行 `scripts/review_stage5_handoff.py` 保留此澄清。

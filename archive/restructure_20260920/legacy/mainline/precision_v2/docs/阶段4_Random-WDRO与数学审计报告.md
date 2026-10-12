# 阶段 4 Random-WDRO 与数学审计

更新时间：2026-09-07T19:17:38.482239Z。阶段 4 的实现与数值门禁已通过；独立交付复核完成后由动态进度文件确认 DONE。阶段 5 未启动。

## 实现与冻结边界

WDRO 入口 `wdro_legacy_v1/wdro_rest_min/scripts/train_wdro.py`，新增代码 `wdro_legacy_v1/wdro_rest_min/stage4/dual.py` 与 `trainer.py`。原 `protocol_v1`、`data_contract_v1`、stage2 trainer 和 stage3 baseline/search 源码及配置 hash 均保持不变。

每个中心使用阶段 3 原样的 32 个合法扰动加原样本，固定 surrogate 初筛后，原样本加 top-3 扰动接受真实 Actor/REST 与缓存 GeoSteiner 精评分。WDRO 逐中心选择 `regret - lambda * Hungarian-L1 cost` 的最大值；lambda 更新为 `[lambda + eta * (mean_cost-rho)]_+`。Critic 仅为方差降低基线。

本阶段仅 d30/s101、batch 256、500 updates；optimizer、初始化、validation 每 100 updates 和 checkpoint schema 沿用冻结实现。诊断 rho 使用训练 update 1 的 256×32 扰动 cost q50：`0.01381177343428`；lambda 初始 0，eta=1。这不是阶段 5 的 validation 超参选择，不将本次 500-update 模型列入 2000-update 正式主实验模型。

## 数学、恢复和实际预算证据

- 69/69 单元测试通过：`shared/experiments/logs/stage4_all_unit_tests_v1.log`。
- 连续 12 vs 10+恢复到12 的 Actor/Critic/optimizer/scheduler/RNG、训练流、方法状态、调用账本、初始化和 best 完全一致，跨 update 11 候选刷新；`shared/experiments/logs/stage4_smoke/audit.json`。
- rho=0 使用精确零运输可行集（lambda→∞ 的边界问题），两次实际更新的模型/optimizer/RNG 与 ERM 完全一致。此分支的 lambda=0 不用来证明正半径的活跃预算行为。
- 前台两份完整候选与对偶审计 76/76 项通过；500-update 审计 45/45 项通过。全量重建 128,000 中心和 4,224,000 候选，全部合法、每中心去重，held-out 内容交集为 0；逐项重算 Hungarian cost、逐中心选择、lambda、均值/滑动均值、512,000 个 cache reference 值及完整 stream hashes。
- 小有限支持 LP 明确为每中心条件概率质量和平均运输预算，允许分数质量。4 个已知 toy LP、smoke 与正式诊断真实分数 LP 均通过；本次真实 LP 最大绝对 gap 为 `0`。LP 只证明所审计有限支持集上的原始/对偶等价，不能证明连续几何空间中内层全局最优。
- 独立重新加载 DAC21 Actor，重算全部 1,024 个 update-1 shortlist 的 REST 与 GeoSteiner 长度，真实 regret、L_star 和 cost 均与训练日志精确相同；另计 1,024 次审计 GeoSteiner calls。证据 `shared/experiments/logs/stage4/independent_initial_score_audit.json`。
- 与已验收 `formal_hard_mining_d30_s101_u2000` 的真实 updates 1–500 配对：同中心、support hash、shortlist hash、刷新编号和预算逐 update 相同。两者配对区间均为中心 128,000、扰动 4,096,000、surrogate 4,224,000、精评分 512,000、reference requests 512,000。比较的是实际相同区间；HM 全 run 仍为 2000 updates。正式 WDRO 的完整逐 run 配对在阶段 6 补齐。

## 500-update 预算稳定性

lambda 最后 `0.20467942`，轨迹最大 `0.21583600`。最后两个不重叠的 100-update 窗口平均 cost 为 `0.0137974127`、`0.0138051178`；相对 rho 的误差分别为 `0.104%`、`0.048%`，满足运行前冻结的 10% 容差。整个 run 的均值 `0.0142211323` 包含从 lambda=0 启动的过渡期；稳定门禁使用预先指定的尾部窗口。

图：`shared/experiments/figures/stage4_lambda_cost_rho.png`（同名 PDF/SVG 可导出）。逐 update 原值见 `shared/experiments/tables/stage4_budget_trajectory.csv`。这些诊断证明本实现的预算行为，不是 WDRO 性能优势或多 seed 稳定改善的结论。

## Validation 与调用账本

仅使用合成 validation 的 6 个时点，best update `500`；nominal mean regret `0.0075539837`，对应 ERM guard 上限 `0.0098025409`，OOD worst-group mean `0.0136216863`。test 与 ISPD 模型指标未读取，也未用来选模。输入审计读取 test 坐标/hash 仅用于检查内容交集。

选中 checkpoint 独立重新加载后，8,000 个 validation nets 的全部指标与选模日志精确一致：`shared/experiments/logs/stage4/best_validation_recompute.json`。额外 Actor/REST 8,000 calls 为交付复核开销，独立记账。

有效轨迹调用账本：`{"actor_forward_nets": 688001, "exact_rest_evaluator_nets": 688002, "geosteiner_nets": 512002, "surrogate_evaluator_nets": 4224000}`。reference cache 的实际请求/开始/完成 receipts 单独见完整审计，失败重试不可混为有效轨迹。训练 wall-clock 和实际 GPU 映射见各 segment 与 pipeline 记录；实测 reserved 峰值 `1620` MiB。

## 产物与恢复

逻辑目录 `/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments`，resolved `/attached/remote-home1/Guilin/DRO/Codex/shared/experiments`。checkpoint、cache、临时目录、CUDA/Torch/Matplotlib cache、tmux socket、PID/锁/心跳和日志均经 Codex 内路径访问。

恢复入口（完成状态会拒绝重复启动）：

```bash
bash /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/launch_stage4.py
```

审计回放：同一环境 wrapper 执行 `wdro_legacy_v1/wdro_rest_min/scripts/audit_stage4.py --run-id stage4_stability_d30_s101_u500_v1 --stability`；每 10 updates 保存绑定配置/数据/hash 的原子游标。汇总重建执行 `scripts/export_stage4.py`。持久任务精确命令、PID、GPU、逐次 preflight/post-launch、恢复与监控见 `shared/experiments/logs/stage4_pipeline/launch.json`、`launches.jsonl` 和 `resource_monitor.jsonl`。

后续仅在授权阶段 5 时执行已冻结计划中的最小 validation 搜索与消融。

# 阶段 2：统一 ERM 与验收报告

更新时间：2026-09-05 02:41 UTC。状态：阶段 2 `DONE`。15/15 正式模型、48/48 单元测试及 533/533 全量审计检查通过，审计于 02:39:40 UTC 完成；无遗留训练/审计进程、锁或 `.partial` 文件。

## 1. 范围与已完成训练

本次仅处理阶段 2，没有启动阶段 3，没有更改 `protocol_v1`、`data_contract_v1` 或冻结实验范围。

15 个正式 ERM run 覆盖 degree 10/20/30/40/50 × seeds 101/102/103。每个 run 完成 2,000 policy updates、batch 256，共 7,680,000 个训练样本。队列在 2026-09-04 19:19:52 UTC 报告全部完成，原训练 tmux 会话正常退出。

训练从对应 degree 的 DAC21 Actor 开始；Critic seed、Adam 参数、constant scheduler、每 100 updates 的验证/checkpoint、每 10 updates 的结构化日志和完整 RNG 状态由共享训练器固定。正式训练代码和三个冻结配置按批次来源 manifest 逐文件核验，没有改动。

## 2. Validation 结果

以下均为 T=1 greedy、validation seed 202、每 distribution/degree 2,000 nets。按三个 OOD 分布的 worst-group mean 最小值选 best；不是 update-2000 末点，也不是 test 结果。表中 regret 单位为百分比，`±` 为三个训练 seed 的样本标准差，不是置信区间。

| degree | 初始 DAC21 nominal | ERM best nominal | ERM best OOD worst-group | best updates（101/102/103） |
|---|---:|---:|---:|---|
| 10 | 0.23355% | 0.22472 ± 0.00176% | 0.22881 ± 0.00584% | 700 / 400 / 400 |
| 20 | 0.56649% | 0.54203 ± 0.01244% | 0.82039 ± 0.01400% | 800 / 500 / 1100 |
| 30 | 0.80582% | 0.76804 ± 0.02150% | 1.34122 ± 0.06142% | 800 / 100 / 1700 |
| 40 | 0.94201% | 0.91894 ± 0.02485% | 1.58069 ± 0.01631% | 1000 / 1000 / 700 |
| 50 | 1.13095% | 1.07372 ± 0.01819% | 1.77864 ± 0.01555% | 1000 / 800 / 1000 |

14/15 个 best 模型的 nominal 优于初始 DAC21；唯一上升的是 degree 40 / seed 101：`0.0094200847 → 0.0094721074`，绝对增加 `0.0000520227`（约 0.00520227 个百分点）。所有 degree 的三 seed nominal 均值均改善。阶段 2 对 ERM/DAC21 的合理性比较不是重新定义其他方法相对 ERM 的冻结 `0.002` nominal 约束。

该结果支持 ERM 基准训练稳定、nominal 合理；不构成 WDRO 鲁棒性或泛化优越性的结论。

## 3. 验收证据与产物

- 训练恢复闭环：200-update smoke 与连续/恢复 checkpoint 完全等价审计，分别见 [smoke 审计](../../../shared/experiments/logs/stage2/stage2_smoke_audit_d30_s101_u200.json)、[RNG 恢复审计](../../../shared/experiments/logs/stage2/resume_determinism_audit_d30_s101_u2.json)。
- 并行精确 evaluator：3-worker 与串行 degree-50 checkpoint 一致，见 [等价性审计](../../../shared/experiments/logs/stage2/parallel_geosteiner_equivalence_d50_s101_u1.json)。
- 单元测试：48/48 通过，见 [完整测试日志](../../../shared/experiments/logs/stage2/stage2_unit_tests_20260905.log)。新增测试覆盖嵌套 NaN/Inf、审计游标恢复和身份变化拒绝、最早 validation 最小值及验证点缺失检测。
- 本次前台单 run 加强审计通过，见 [审计 smoke](../../../shared/experiments/logs/stage2/stage2_completion_audit_smoke.json)。
- 全量审计：[stage2_formal_audit.json](../../../shared/experiments/logs/stage2/stage2_formal_audit.json)，`status=pass`、`audit_scope=15_formal_runs`；每 run 35 项 × 15，加 8 项全局检查，共 533 项通过。7,680,000 个训练样本对应的全部 batch hash 重建一致，训练与 validation/test 内容交集为 0。
- 完成 manifest：[stage2_completion_20260905.json](../../wdro_rest_min/manifests/stage2_completion_20260905.json)，记录模型、结果、日志、游标、表格和审计源代码的路径/hash。
- [15-run 验证汇总](../../../shared/experiments/tables/stage2_erm_validation_summary.csv)、[5-degree 三 seed 汇总](../../../shared/experiments/tables/stage2_erm_degree_summary.csv)、[60 行 best 验证长表](../../../shared/experiments/tables/stage2_erm_best_validation_long.csv)、[1,260 行全部验证点长表](../../../shared/experiments/tables/stage2_erm_validation_curves.csv)。长表保留 mean、P95、样本数及 degree/distribution/seed，不丢弃 seed 维度。
- 训练池每 100 updates 的可恢复重建游标、每 batch hash、held-out 数据 hash 和内容隔离计数位于 `shared/experiments/logs/stage2/formal_stream_audit_v1/`。
- 模型位于 `shared/experiments/checkpoints/stage2/formal_erm_d{degree}_s{seed}_u2000/{latest,best}.pt`；完整 manifest 已记录各模型的逻辑路径、resolved path 和 SHA-256。

完整审计逐 run 核验：2,000 updates/512,000 nets；latest 与 best 身份、hash 和保留规则；Actor/Critic/optimizer/RNG 数值；Adam 和 scheduler 的步数；全部 21 个 validation 点及 OOD objective 重算；最早全局最小值选模；validation 文件与 L_star hash；逆变换长度等价；训练流全量重建；训练 ID/内容与 validation/test 隔离；任务完成、锁释放、无 `.partial`。

本次只为内容隔离读取 test 坐标，没有在 test 或 ISPD 上运行模型，也没有读取它们的模型指标参与选模。训练中心由冻结 seed/ID/源代码定义为可重建的虚拟固定池，而不是重启时重新随机抽样。validation/test 的 L_star 已按数据 hash 缓存；阶段 2 训练 L_star 是在线精确计算，未持久化为训练 reference cache，不应声称训练缓存命中或候选缓存已完成。

## 4. 资源、恢复与计数边界

正式批次来源：[stage2_formal_batch_20260904T1801Z.json](../../wdro_rest_min/manifests/stage2_formal_batch_20260904T1801Z.json)。GPU 只是运行时放置，不参与训练语义。最初共租户显存突增造成 update-0 OOM，GPU 0/7 后续接近饱和；受影响任务保留失败日志，并迁移后从 latest 恢复，没有改变 batch 或 updates。最终 15 个 run 在 GPU 4/5/6 完成。

本次 02:27 与 02:33 UTC 都重新做了三次 GPU/compute-process 采样、磁盘双路径检查和 preflight。02:33 时 GPU 4/5/6/7 有容量余量，产物盘约剩 2903.784 GiB；完整审计本身是 CPU-only，无需占用这些 GPU。见 [审计前 host-runtime](../../wdro_rest_min/manifests/host_runtime_20260905T023351Z.json)。

所有新产物逻辑路径均在 `/remote-home/Guilin/DRO/Codex`。`shared/experiments` 的稳定符号链接解析到 `/attached/remote-home1/Guilin/DRO/Codex/shared/experiments`；tmp、runtime cache、tmux socket、日志和锁也在 Codex 内。

有效 checkpoint 轨迹每 run 调用：Actor `680,001`、REST `680,002`、GeoSteiner `512,002`、surrogate `0`。15 run 合计分别为 `10,200,015 / 10,200,030 / 7,680,030 / 0`，包含初始验证、每 100 updates 验证及逆变换审计。失败尝试和回滚重放产生额外开销；现有账本及 `completed_segment_wall_seconds` 不覆盖全部这些开销，不得直接作为包含重试的完整 wall-clock/call 总成本用于最终效率对比。历史启动/恢复/失败证据保留于原队列日志。

各 degree 正式 run 最大 torch reserved 峰值为 `398 / 862 / 1626 / 2446 / 3602 MiB`。这是 PyTorch allocator 峰值，不等于 nvidia-smi 总显存，也不是脱离共租户负载测得的吞吐基准。

## 5. 可复现与恢复命令

训练已完成，不应再次启动训练队列。审计被中断时，从下面入口恢复；它会重新采样硬件/磁盘、创建独立 tmux 会话并继续已有游标：

```bash
cd /remote-home/Guilin/DRO/Codex/wdro_legacy_v1/wdro_rest_min
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/remote-home/Guilin/DRO/Codex/shared/experiments/tmp \
  /remote-home/Guilin/miniconda3/envs/py311/bin/python scripts/launch_stage2_completion_audit.py
```

会话 `wdro_stage2_audit_v1`，socket 名 `wdro_stage2_audit`。精确命令、PID、环境、日志、心跳、监控/恢复命令：[launch.json](../../../shared/experiments/logs/stage2/stage2_formal_audit_job/launch.json)。监控：

```bash
tail -f /remote-home/Guilin/DRO/Codex/shared/experiments/logs/stage2/stage2_formal_audit_job/stdout.log
```

全量审计通过后导出验证表与模型索引：

```bash
cd /remote-home/Guilin/DRO/Codex/wdro_legacy_v1/wdro_rest_min
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/remote-home/Guilin/DRO/Codex/shared/experiments/tmp \
  /remote-home/Guilin/miniconda3/envs/py311/bin/python scripts/export_stage2_summary.py
```

本次工作止于阶段 2。下一阶段需另行启动；不得把这里的 ERM validation 结果当成阶段 6 的共享 test 主表。

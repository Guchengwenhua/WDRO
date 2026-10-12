# precision_v2 运行状态镜像

唯一主进度源为`/remote-home/Guilin/DRO/Codex/01_WDRO-REST_项目进度.md`。本文件供隔离版本旧控制器记录其本地子步骤；由纠错总控制器将证据同步到主文件。所有DONE均仅指precision_v2新产物。

| 项目 | 当前值 |
|---|---|
| 更新时间 | 2026-09-09T09:01:39.727607Z |
| 实验状态 | 阶段6的75模型、300测试格、paired CI与完整交付验收通过 |
| 下一动作 | 本次止于阶段6；阶段7 ISPD尚未启动 |

- `TODO` 3.1：baseline smoke与原候选预算审计。
- `TODO` 3.2：45个baseline训练。
- `TODO` 3.3：完整审计与交付。
- `DONE` 5.1：80单测、三种搜索精确恢复、八变体完整smoke审计与64-call真实重算/8-worker等价通过；证据stage5_smoke/audit.json与stage5_audit/smoke_gate.json。
- `DONE` 5.2：三个rho训练/完整审计通过，validation-only选择见stage5_pipeline/rho_selection.json。
- `DONE` 5.3：九个run训练和完整输入/cost/对偶审计通过，见stage5_audit/runs。
- `DONE` 5.4：完成；证据与各项统计见 `阶段5_Validation搜索与消融验收报告.md` 和 stage5_completion/delivery manifests。
- `DONE` 6.1：原80单测与阶段6测试、跨刷新精确恢复/阶段5数值等价、五degree完整smoke审计、validation评价断点恢复通过；证据stage6_smoke/audit.json、stage6_evaluation_smoke/audit.json。
- `DONE` 6.2：75模型完整审计与validation-only best索引冻结；证据manifests/stage6_models_frozen_v1.json。
- `DONE` 6.3：300格/300万逐net评价已完成；等待6.4独立完整性与表格复核。
- `DONE` 6.4：统计、主表/附表/paired CI和完整交付复核通过；见阶段6验收报告与completion/delivery manifests。

## 8. 变更记录

### 2026-09-09T09:01:39.727607Z

- 阶段6完整验收通过：75模型、300格/300万test预测、14个新增WDRO全候选/cost/对偶/HM预算审计、主表/degree附表及paired95%CI。证据 `阶段6_多种子合成主实验验收报告.md` 与 `wdro_rest_min/manifests/stage6_completion_v1.json`、`stage6_delivery_verification_v1.json`。
- 原protocol/data-contract和阶段5冻结配置未变；没有test选模/回调，所有结果保留；本次止于阶段6。

### 2026-09-09T08:51:03.425382Z

- 阶段6.3评价完成，进入统计与独立验收。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：保留全部结果，完成paired CI和交付报告。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T08:44:20.059381Z

- 阶段6.2完整通过，75模型已冻结。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：6.3统一合成test，不再选模或调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T06:52:12.027395Z

- 阶段 6 6.2_training：11/14 run 训练完成，累计 27880 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T06:42:00.933819Z

- 阶段 6 6.2_training：11/14 run 训练完成，累计 27640 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T06:31:49.661576Z

- 阶段 6 6.2_training：9/14 run 训练完成，累计 27000 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T06:21:38.730790Z

- 阶段 6 6.2_training：7/14 run 训练完成，累计 24900 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T06:11:27.680698Z

- 阶段 6 6.2_training：7/14 run 训练完成，累计 22620 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T06:01:16.728030Z

- 阶段 6 6.2_training：7/14 run 训练完成，累计 20340 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T05:51:16.393935Z

- 阶段 6 6.2_training：3/14 run 训练完成，累计 18420 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T05:41:05.590231Z

- 阶段 6 6.2_training：3/14 run 训练完成，累计 17020 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T05:30:58.262651Z

- 阶段 6 6.2_training：2/14 run 训练完成，累计 15740 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T05:20:47.195677Z

- 阶段 6 6.2_training：2/14 run 训练完成，累计 14570 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T05:10:36.237637Z

- 阶段 6 6.2_training：2/14 run 训练完成，累计 13350 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T05:00:24.940954Z

- 阶段 6 6.2_training：2/14 run 训练完成，累计 12150 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T04:50:20.962513Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 11020 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T04:40:10.082986Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 10090 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T04:29:59.032266Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 9160 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T04:19:47.782924Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 8200 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T04:09:36.786836Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 7260 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T03:59:25.746013Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 6320 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T03:49:14.761534Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 5390 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T03:39:03.600033Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 4440 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T03:28:52.729951Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 3500 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T03:18:41.805116Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 2550 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T03:08:31.004050Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 1620 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T02:58:25.815635Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 650 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T02:48:01.140496Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 410 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T02:37:52.686120Z

- 阶段 6 6.2_training：0/14 run 训练完成，累计 0 updates。证据 `experiments/logs/stage6_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续冻结配置训练与完整审计；75模型锁定后共享test，不调参。恢复入口 `scripts/launch_stage6.py` 将重新动态预检。

### 2026-09-09T02:09:28.020674Z

- 阶段 5 完整验收通过；9 个最小 run、18,000 updates，选中 rho=q75。完整报告 `阶段5_Validation搜索与消融验收报告.md`；证据 `wdro_rest_min/manifests/stage5_completion_v1.json`、`stage5_delivery_verification_v1.json`。
- 主配置保持 Random/Sobol K32、Hungarian、learned lambda；只用 validation，未启动阶段 6/test/ISPD。未来按 `stage5_selected_wdro_v1.json` 执行，不重新搜索或 test 后回调。

### 2026-09-09T02:02:29.916159Z

- 阶段 5.3 九个 run 训练和完整审计通过。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：5.4 独立重算 best validation、生成消融表/预算图并冻结主配置。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-09T02:02:14.559172Z

- 阶段 5.2 三个 rho 的 2000-update 训练及完整审计通过；validation 选中 q75。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：5.3 执行 selected-rho 的六个最小消融，保持主方法约束。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T15:52:33.708670Z

- 阶段 5 5.3_ablations：5/6 run 训练完成，累计 11990 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T15:42:23.708595Z

- 阶段 5 5.3_ablations：5/6 run 训练完成，累计 11960 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T15:32:13.224888Z

- 阶段 5 5.3_ablations：5/6 run 训练完成，累计 11920 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T15:22:02.831126Z

- 阶段 5 5.3_ablations：5/6 run 训练完成，累计 11890 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T15:11:52.674963Z

- 阶段 5 5.3_ablations：5/6 run 训练完成，累计 11860 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T15:01:42.620982Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11800 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T14:51:32.602296Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11740 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T14:41:22.550911Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11670 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T14:31:12.462715Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11610 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T14:21:02.329973Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11530 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T14:10:52.212048Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11470 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T14:00:41.824218Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11400 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T13:50:31.902992Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11340 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T13:40:22.041461Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11270 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T13:30:11.812937Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11210 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T13:20:01.514890Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11150 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T13:09:51.284634Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11080 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T12:59:41.245230Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 11020 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T12:49:31.042489Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10950 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T12:39:20.933462Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10890 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T12:29:11.311322Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10830 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T12:18:59.159463Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10760 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T12:08:49.819703Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10700 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T11:58:36.588856Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10640 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T11:48:27.221349Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10590 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T11:38:17.387309Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10520 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T11:28:07.773524Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10460 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T11:17:57.997151Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10400 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T11:07:48.238152Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10340 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T10:57:38.435981Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10290 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T10:47:28.461314Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10220 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T10:37:18.483376Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10160 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T10:27:08.899040Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10100 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T10:16:58.571262Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 10040 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T10:06:48.457400Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9970 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T09:56:38.373821Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9910 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T09:46:28.318979Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9860 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T09:36:18.094317Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9790 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T09:26:08.025671Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9730 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T09:15:57.680364Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9670 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T09:05:48.066143Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9600 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T08:55:38.283301Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9540 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T08:45:28.257711Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9480 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T08:35:18.619123Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9420 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T08:25:09.024410Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9350 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T08:14:59.355301Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9300 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T08:04:49.613523Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9240 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T07:54:39.695039Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9180 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T07:44:30.119672Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9110 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T07:34:20.214815Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 9050 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T07:24:10.234613Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 8990 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T07:14:00.560296Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 8940 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T07:03:50.858648Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 8880 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T06:53:41.084687Z

- 阶段 5 5.3_ablations：4/6 run 训练完成，累计 8810 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T06:43:31.414147Z

- 阶段 5 5.3_ablations：2/6 run 训练完成，累计 8650 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T06:33:21.567486Z

- 阶段 5 5.3_ablations：2/6 run 训练完成，累计 8330 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T06:23:11.669643Z

- 阶段 5 5.3_ablations：2/6 run 训练完成，累计 8010 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T06:13:02.023127Z

- 阶段 5 5.3_ablations：2/6 run 训练完成，累计 7680 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T06:02:52.462523Z

- 阶段 5 5.3_ablations：2/6 run 训练完成，累计 7380 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T05:52:46.077585Z

- 阶段 5 5.3_ablations：2/6 run 训练完成，累计 7070 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T05:42:36.452099Z

- 阶段 5 5.3_ablations：2/6 run 训练完成，累计 6790 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T05:32:26.416436Z

- 阶段 5 5.3_ablations：1/6 run 训练完成，累计 6410 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T05:22:16.456949Z

- 阶段 5 5.3_ablations：1/6 run 训练完成，累计 5950 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T05:12:09.006503Z

- 阶段 5 5.3_ablations：1/6 run 训练完成，累计 4500 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T05:01:55.489782Z

- 阶段 5 5.3_ablations：1/6 run 训练完成，累计 2000 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T05:01:55.221393Z

- 阶段 5.2 三个 rho 的 2000-update 训练及完整审计通过；validation 选中 q75。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：5.3 执行 selected-rho 的六个最小消融，保持主方法约束。恢复入口 `scripts/precision_stage5_capacity_pipeline.py` 将重新动态预检。

### 2026-09-08T00:14:04.840964Z

- 阶段 5 5.3_ablations：1/6 run 训练完成，累计 4780 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-08T00:03:52.466205Z

- 阶段 5 5.3_ablations：0/6 run 训练完成，累计 4030 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T23:53:45.334023Z

- 阶段 5 5.3_ablations：0/6 run 训练完成，累计 2770 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T23:43:37.678391Z

- 阶段 5 5.3_ablations：0/6 run 训练完成，累计 1980 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T23:33:31.975903Z

- 阶段 5 5.3_ablations：0/6 run 训练完成，累计 950 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T23:23:24.666623Z

- 阶段 5 5.3_ablations：0/6 run 训练完成，累计 0 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T23:23:24.419488Z

- 阶段 5.2 三个 rho 的 2000-update 训练及完整审计通过；validation 选中 q75。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：5.3 执行 selected-rho 的六个最小消融，保持主方法约束。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T21:46:26.516693Z

- 阶段 5 5.2_rho：1/3 run 训练完成，累计 5970 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T21:36:15.624462Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 5420 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T21:26:04.888750Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 4860 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T21:15:53.958574Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 4310 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T21:05:43.174042Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 3760 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T20:55:32.298314Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 3210 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T20:45:21.438555Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 2660 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T20:35:10.728266Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 2110 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T20:24:59.934373Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 1550 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T20:14:49.386885Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 1010 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T20:04:38.784400Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 470 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T19:54:38.235673Z

- 阶段 5 5.2_rho：0/3 run 训练完成，累计 0 updates。证据 `experiments/logs/stage5_pipeline/heartbeat.json`、`queue.json`、`launches.jsonl`。下一步：继续当前阶段队列；完整审计后再选择/冻结。恢复入口 `scripts/launch_stage5.py` 将重新动态预检。

### 2026-09-07T19:17:57.307414Z

- 阶段 4 全部验收通过：69 单测、12 vs 10+resume 精确恢复、rho=0/ERM 精确更新、500 updates、45 项完整审计及 80 项独立交付复核。报告 `阶段4_Random-WDRO与数学审计报告.md`；索引 `wdro_rest_min/manifests/stage4_completion_v1.json`、`stage4_delivery_verification_v1.json`。
- 全量重建 128,000 中心和 4,224,000 候选，held-out 内容交集 0；512,000 个实际 reference 值与已审计 HM cache 完全一致。逐 update 支持、shortlist、精评分/evaluator-call 预算与 HM 实际前 500 updates 配对通过；没有声称 WDRO 500 与 HM 2000 整 run 等预算。
- 正半径 rho=0.013811773434，最终 lambda=0.20467942，末尾两个 100-update 平均 cost=[0.013797412741968098, 0.013805117815597137]，均在预先冻结的 10% 容差内。可导出 PNG/PDF/SVG 与四张 CSV 位于 `experiments/figures`、`experiments/tables/stage4_*.csv`。
- 训练、审计完成且目标锁均释放；持久 job/启动/恢复/资源监控历史保留在 `experiments/logs/stage4_pipeline`。交付复核可由环境 wrapper 重跑 `scripts/verify_stage4_delivery.py`。原协议、数据契约及冻结基线未改动；阶段 5、test/ISPD 模型评价未启动。

### 2026-09-07T19:05:12.274557Z

- 阶段 4.3 WDRO 连续目标 500 updates 已完成；正在全量输入、cost、lambda 和实际 HM 区间审计。证据 `experiments/logs/stage4_pipeline/heartbeat.json`、`experiments/logs/stage4/stage4_stability_d30_s101_u500_v1/`。下一步：完成审计与表图/报告后验收阶段 4。

### 2026-09-07T18:59:44.507671Z

- 阶段 4.3 WDRO 稳定性运行已达到 400/500 updates，仍待完整审计。证据 `experiments/logs/stage4_pipeline/heartbeat.json`、`experiments/logs/stage4/stage4_stability_d30_s101_u500_v1/`。下一步：继续同一 run；通过完整预算/数据/LP 审计后导出，阶段 5 不启动。

### 2026-09-07T18:54:24.161654Z

- 阶段 4.3 WDRO 稳定性运行已达到 300/500 updates，仍待完整审计。证据 `experiments/logs/stage4_pipeline/heartbeat.json`、`experiments/logs/stage4/stage4_stability_d30_s101_u500_v1/`。下一步：继续同一 run；通过完整预算/数据/LP 审计后导出，阶段 5 不启动。

### 2026-09-07T18:49:03.820617Z

- 阶段 4.3 WDRO 稳定性运行已达到 200/500 updates，仍待完整审计。证据 `experiments/logs/stage4_pipeline/heartbeat.json`、`experiments/logs/stage4/stage4_stability_d30_s101_u500_v1/`。下一步：继续同一 run；通过完整预算/数据/LP 审计后导出，阶段 5 不启动。

### 2026-09-07T18:43:28.058699Z

- 阶段 4.3 WDRO 稳定性运行已达到 100/500 updates，仍待完整审计。证据 `experiments/logs/stage4_pipeline/heartbeat.json`、`experiments/logs/stage4/stage4_stability_d30_s101_u500_v1/`。下一步：继续同一 run；通过完整预算/数据/LP 审计后导出，阶段 5 不启动。

### 2026-09-07T18:31:04.718157Z

- 阶段 3 完成：45/45 正式模型、1680 项完整审计；全部模型和数据流可追溯。证据为 `阶段3_三基线与验收报告.md` 与 `wdro_rest_min/manifests/stage3_completion_v1.json`。
- 生成 45-run 汇总、180 行 best validation 长表、3,780 行全 validation 长表和 15 行 Hard Mining 搜索预算表。未读取 test/ISPD 模型指标，未进入阶段 4。


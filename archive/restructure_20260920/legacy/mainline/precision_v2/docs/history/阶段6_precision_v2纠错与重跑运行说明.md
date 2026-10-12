# 阶段6 precision_v2纠错与重跑

用户已明确授权修复GeoSteiner精度、复核及必要重跑受影响的阶段1–5，然后完成阶段6。阶段7/ISPD模型评价不在本次范围。原v1代码、配置、数据、模型和失败证据保留。

## 当前证据与门禁

| 子步骤 | 状态及验收证据 |
|---|---|
| R1 参考精度修复 | 38个独立优化对照、连通整数树、置换/精确尺度、2-worker等价和暂停恢复均通过：`shared/experiments/logs/precision_v2/reference_verification/acceptance.json` |
| R2 固定缓存纠错 | 40文件、240000输入，2^27及2^28结果逐项一致，480000调用；1392个旧值偏大，0个旧值更小。新缓存和逐格差异见revision的`shared/experiments/logs/reference_cache_v2/` |
| R3 ERM smoke | 200 updates通过；12项checkpoint语义字段连续/恢复精确相同；五degree峰值388/852/1620/2438/3602 MiB；80项原回归通过 |
| R3 同设置重放 | 15 ERM、30000 updates及533项完整审计/交付已通过（07:19 UTC）；三基线恢复/15组显存、98项smoke审计、80回归均通过。45正式训练已于07:27 UTC进入队列；首批8 run/85项启动核验通过（首个latest update100），后续仍需全量候选审计及交付复核 |
| R4/R5 WDRO重放 | 控制器已持久运行，当前等待新60模型交付；通过后执行500-update诊断及原9-run validation集合，重新按原规则冻结选择 |
| R6 合成主测试 | 控制器已持久运行，当前等待纠错版validation冻结；通过后补齐75模型，重做全部300格/300万预测、主表、degree附表与paired CI |

R1大degree对照使用GeoSteiner生成的full components，但由独立HiGHS优化器求解；小degree另用独立Hanan网格MILP。该限制在验收文件中记录。没有把有限样本对照说成任意输入的数学最优性证明。新正式参考求解逐次要求optimal状态，并核验返回树连通性和整数边长；固定缓存另逐次比较两种精确尺度。Actor输入、normalization及数据文件不变，不进行坐标取整。

## 隔离版本与冻结

新逻辑工作区为`wdro_precision_v2`。该目录的`shared/experiments`稳定链接至`Codex/shared/experiments/precision_v2`，最终resolved path为`/attached/remote-home1/Guilin/DRO/Codex/shared/experiments/precision_v2`。旧合成数据和ISPD输入只读链接复用；ISPD模型指标保持关闭。

`wdro_legacy_v1/wdro_rest_min/manifests/precision_v2_workspace.json`记录初始复制/改动。新workspace的`manifests/precision_v2_training_source_v1.json`冻结共享训练实现和原5份设置；protocol_v1、data_contract_v1、stage2_trainer_v1、stage3_baselines_v1、stage4_wdro_v1与原版字节相同。reference身份绑定求解器实现、共享库和精确尺度；checkpoint、validation/test reference及候选SQLite禁止接续旧reference版本。

主目录`summaries/history/01_WDRO-REST_项目进度.md`仍是唯一动态状态源。revision内同名文件为旧控制器运行兼容镜像，不能据其历史措辞推断新授权或总阶段DONE。

## 持久任务与恢复

tmux socket：`shared/experiments/tmp/precision_v2.sock`。全部PID、精确命令、日志、心跳和恢复命令位于`shared/experiments/logs/precision_v2/jobs/{erm_formal,replay,wdro_replay,stage6_completion}/launch.json`及对应job目录；子步骤命令/游标位于revision的`shared/experiments/logs/precision_replay/`。

首个完整ERM的提前审计证据为revision的`shared/experiments/logs/precision_first_erm/acceptance.json`。原审计器的raw报告只有`queue_complete=false`，其余全通过；该状态表示整批门禁未开，不能标阶段2或阶段6完成。35项单run检查未放宽。

```bash
cat /remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments/logs/stage2_queue/stage2_erm_formal_v1/heartbeat.json
cat /remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments/logs/precision_replay/heartbeat.json
tmux -S /remote-home/Guilin/DRO/Codex/shared/experiments/tmp/precision_v2.sock list-sessions
bash /remote-home/Guilin/DRO/Codex/wdro_legacy_v1/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/wdro_legacy_v1/wdro_rest_min/scripts/precision_status.py
```

断连不需重新启动。如果任务已退出，先读对应`exit.json`、`output.log`和失败子步骤，再使用`launch.json`内的`recovery_command`；启动器拒绝重复会话，队列另有锁。恢复会重新preflight，训练从原子latest继续；已验收子步骤只有输出hash仍一致才跳过。

## 资源与验收要求

2026-09-07约03:56 UTC启动时，实验盘约2827 GiB可用。该值与当时GPU集合仅为历史快照。每个run启动/恢复重新检查全部GPU，结合本次smoke峰值、1024 MiB上下文余量、2048 MiB安全余量、多次利用率及近期显存最低值调度；每卡当前上限2个run，错峰15秒。启动后记录GPU/compute process和双路径磁盘快照。持续饱和、余量不足或吞吐骤降时暂停本任务的一项并冷却后恢复，batch256与2000updates不变。

所有中间产物、cache、临时文件和运行时缓存均在Codex逻辑边界内。每个阶段的新完整审计通过后才进入下一批跑。旧13个完整test格不进入纠错版主统计；不使用旧test表现修改候选范围、超参数或选择规则。完成全部75模型和300测试格之前，阶段6保持DOING。

## 2026-09-07 07:19 UTC 续查与修复

15 ERM在05:27 UTC完成，原审计因d40/s102按原validation规则选中update0而误判空Adam状态，依赖批跑停止。已保留原失败报告和脚本于revision的`shared/experiments/logs/precision_erm_audit_repair/before/`，仅修正审计边界：update0必须为空且参数组完整；正update仍要求实际步数匹配。22项边界/真实checkpoint检查、该run完整审计及随后15-run总审计全部通过。选中Actor与DAC21逐tensor一致，latest为2000，checkpoint和冻结源hash未改。修复与恢复证据为`wdro_legacy_v1/wdro_rest_min/manifests/precision_v2_erm_audit_operations_patch_20260907.json`（相对revision）。

`precision_replay`已重新启动，精确恢复命令仍取原Codex `shared/experiments/logs/precision_v2/jobs/replay/launch.json`。旧失败exit.json在新任务运行期间可能仍存在，必须结合launch时间判断；状态入口现已显式记录`exit_belongs_to_current_launch`，并补充三基线、WDRO及test进度。新失败将写入结构化failures并立即更新动态进度。

07:31 UTC启动交接：revision的`shared/experiments/logs/precision_replay_continuation/baseline_launch_verification.json`核对8个活跃run和85项检查通过；当前GPU映射0/2/3/5/6，动态调度不永久保留。只验证启动与恢复产物，45run尚未完成。完整变更与证据附录`wdro_legacy_v1/wdro_rest_min/manifests/precision_v2_continuation_review_20260907.json`（相对revision）。

## 2026-09-07T09:55:28.587834Z 阶段6剩余任务续跑

本次在原持久链上补充了已完成基线的提前完整审计及后续smoke的容量等待。复核时三基线9/45训练完成，9/45通过完整审计（337项检查）；阶段6仍DOING。逐项证据、故障说明、唯一预检文件名修复和恢复入口见[阶段6剩余任务续跑记录](阶段6_剩余任务续跑记录_20260907.md)，总索引 `wdro_legacy_v1/wdro_rest_min/manifests/stage6_remaining_continuation_20260907.json`。原三条控制器保持运行，阶段7未启动。

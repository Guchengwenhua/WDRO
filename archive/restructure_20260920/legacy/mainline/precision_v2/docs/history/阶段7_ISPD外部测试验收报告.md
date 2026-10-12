# 阶段7 ISPD协议A外部测试验收报告

完成时间：2026-09-09T09:56:05.085331Z。阶段7 **DONE**。

使用 precision_v2 阶段6已经冻结的75个模型；模型及超参数只来自合成训练和validation。
全部20个基础design、95个既有design×degree文件、22498个有效net已评价；5个缺失degree文件和0个无效net均逐格保留。排除6个metal5重复版本。
合计337470条预测，统一共享归一化、batch256、T=1 deterministic greedy和RTX6000 Ada硬件。

| 方法 | net-weighted mean (%) | design-macro mean (%) | P95 (%) | worst-design mean (%) |
|---|---:|---:|---:|---:|
| erm | 0.757135 | 0.702543 | 3.130610 | 0.993673 |
| mixed_erm | 0.735209 | 0.683546 | 3.076584 | 0.977534 |
| hard_mining | 0.744835 | 0.693797 | 3.104114 | 0.959932 |
| group_dro | 0.736985 | 0.683351 | 3.099490 | 0.906074 |
| wdro_rest | 0.735651 | 0.684733 | 3.045556 | 0.933528 |

表中先对每个训练seed计算指标，再平均三个seed；P95和worst-design不通过混合seed预测计算。各seed、design和degree完整数值见CSV。
这些是冻结模型的外部测试描述统计；置信区间和独立攻击属于阶段8，尚未执行。可称模型零样本迁移，不能称研究者完全未见的blind test。

## 验收与来源

- 95个源文件hash及全部22498个源net共享归一化重建，与阶段1冻结数组逐项一致。
- 全部参考长度以2^27精确尺度求解，逐解检查optimal、连通性、整数边长；2^28复算22498项精确一致。正式缓存构建共44996次GeoSteiner调用，恢复smoke的512次包含在该缓存轨迹内。
- 评价smoke：513个net连续与分段恢复的全部预测字段精确相同；五degree显存实测已保存。smoke额外Actor/REST各2050次，独立Python拓扑检查1537条。
- 新增9项聚合/拓扑/恢复/锁/中断账目测试及19项原协议、归一化、ISPD回归通过；日志为revision experiments/logs/stage7/{unit_tests_correct_cwd,shared_input_regression}.log。最初测试入口cwd错误的失败日志保留，修正入口后未修改测试断言。
- 75模型共337470条正式预测；独立C evaluator重算及Python线段长度/全树连通性均通过。2070行表格、20份逐net配对差值、3个图文件和全部冻结模型hash再次核验。
- 每个预测保留design/net ID、source index、模型hash、reference缓存绑定、拓扑、长度及regret；每批原子分片并保存游标。失败/重放的调用attempts独立记账，不混入有效推理耗时。
- 首次准备使用旧benchmark根路径而失败；改为已有precision_v2只读legacy链接后源hash一致，未改数据或研究设置。失败日志为revision experiments/logs/stage7_prepare_front.log。

## 产物与恢复

- 完成索引：`/remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/manifests/stage7_completion_v1.json`。
- 独立交付复核：`/remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/manifests/stage7_delivery_verification_v1.json`。
- 原始预测、审计、任务清单、PID/锁、心跳、精确启动命令：`/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments/logs/stage7`。
- 表格：`/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments/tables/stage7`；图：`/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments/figures/stage7_design_mean.*`。
- 逻辑产物根：`/remote-home/Guilin/DRO/Codex/wdro_precision_v2/experiments`；resolved：`/attached/remote-home1/Guilin/DRO/Codex/shared/experiments/precision_v2`。
- 监控：`bash /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage7.py status`。
- 如有中断，恢复：`bash /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage7.py launch`；启动前重新动态预检，活动会话或锁会拒绝重复执行。
训练没有启动，latest/best、optimizer/scheduler与RNG由原冻结checkpoint只读保留。评价使用确定性greedy与原子批游标恢复。

阶段8–9保持未启动；本阶段不根据ISPD结果重新选模或调参。

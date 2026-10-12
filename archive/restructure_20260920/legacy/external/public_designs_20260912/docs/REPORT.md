# 公开芯片物理设计外部测试报告

完成日期：2026-09-12T06:47:20.350287+00:00。交付审计：**PASS**。

新增下载29个设计；原始普通信号网络共14,392,871条。28个设计有目标degree样本，纳入134,550条合法网络；75个冻结模型产生2,018,250条逐net预测。

模型来自reference_precision_v2，仅用合成数据训练/选模；本次没有新增训练或外部测试调参。旧阶段0–8产物保持原样。

本轮整体net-weighted mean：ERM 0.827848% → WDRO 0.819455%；P95：3.554503% → 3.521469%。WDRO相对ERM有小幅改善，但整体mean和P95仍落后于Mixed ERM和GroupDRO。

分数据集看，WDRO在ISPD08的mean/design-macro相对ERM差值区间低于0；Nangate45的WDRO mean和P95略高于ERM，且mean差值区间跨0。ASAP7的WDRO mean点估计较低但只有两个设计家族，区间跨0；SKY130HD仅43条net、一个设计，不能据此声称稳定跨芯片优势。结果没有支持WDRO全面优于所有基线。

## 数据与覆盖

- [ISPD08官方来源](https://www.ispd.cc/contests/08/ispd08rc.html)：16个工业ASIC衍生全局布线设计；8个与ISPD07共享，此处只计一次。
- [NVlabs/CircuitOps作者仓库](https://github.com/NVlabs/CircuitOps)：13个公开芯片物理设计研究实现，覆盖Nangate45、ASAP7、SKY130HD；包含AES、Ariane、BlackParrot、Ibex、JPEG、SweRV、NVDLA等。没有将它们一概称为已流片量产芯片。
- CircuitOps全源690,475条net中10,463条缺上游IR未导出的I/O，按完整DEF连接整网拒绝；这些拒绝均不在当前目标degree集合。目标degree合计3341条，几何合法3341条。
- 所有数据按原始degree 10/20/30/40/50筛选，未截断、拆分、补点或抽样。ISPD08有18条目标net因重复XY导致degree改变而拒绝。GCD没有目标degree样本，保留零样本行，不伪造结果。
- 三组相同归一化几何来自不同源net（均在各自同一ISPD设计内部）；保留真实网络计数并明确记录。
- 与原有262,498条合成validation/test及ISPD18/19逐无序pin hash比较，几何交集为0。此项不等同于重新扫描全部历史训练流。

| 数据集 | 下载设计数 | 有效设计数 | 合法nets |
|---|---:|---:|---:|
| CircuitOps ASAP7 | 4 | 4 | 968 |
| CircuitOps Nangate45 | 8 | 7 | 2,330 |
| CircuitOps SKY130HD | 1 | 1 | 43 |
| ISPD 2008 GR | 16 | 16 | 131,209 |

完整逐design×degree计数：[input_coverage.csv](../tables/input_coverage.csv)。来源与许可：[竞赛说明](../notes/contests_sources.md)、[开源设计说明](../notes/real_design_sources.md)。ICCAD15/17官方失效链接已记录，未计入已测试数据。

## 全方法结果

下表数值均为relative regret ×100（%），越低越好。每个seed先独立计算指标，再对3个seed等权平均；P95不是先平均每net预测再取分位数。

| 数据集 | 方法 | mean (%) | P95 (%) | design-macro (%) | worst-design (%) |
|---|---|---:|---:|---:|---:|
| CircuitOps ASAP7 | ERM | 0.572189 | 2.638406 | 0.567795 | 1.055589 |
| CircuitOps ASAP7 | Mixed ERM | 0.573910 | 2.445582 | 0.617043 | 1.045720 |
| CircuitOps ASAP7 | Hard Mining | 0.580250 | 2.543154 | 0.587864 | 1.077049 |
| CircuitOps ASAP7 | GroupDRO | 0.565024 | 2.592557 | 0.588938 | 1.110049 |
| CircuitOps ASAP7 | WDRO-REST | 0.559662 | 2.534022 | 0.558340 | 1.003485 |
| CircuitOps Nangate45 | ERM | 0.411894 | 2.023615 | 0.454826 | 0.743384 |
| CircuitOps Nangate45 | Mixed ERM | 0.445916 | 2.223370 | 0.472154 | 0.708491 |
| CircuitOps Nangate45 | Hard Mining | 0.406180 | 2.020860 | 0.441207 | 0.673695 |
| CircuitOps Nangate45 | GroupDRO | 0.418803 | 2.095662 | 0.455179 | 0.715270 |
| CircuitOps Nangate45 | WDRO-REST | 0.415875 | 2.032954 | 0.445832 | 0.674449 |
| CircuitOps SKY130HD | ERM | 0.427106 | 1.835662 | 0.427106 | 0.427106 |
| CircuitOps SKY130HD | Mixed ERM | 0.533288 | 2.304956 | 0.533288 | 0.533288 |
| CircuitOps SKY130HD | Hard Mining | 0.417869 | 2.035837 | 0.417869 | 0.417869 |
| CircuitOps SKY130HD | GroupDRO | 0.394320 | 2.305473 | 0.394320 | 0.394320 |
| CircuitOps SKY130HD | WDRO-REST | 0.396196 | 1.479074 | 0.396196 | 0.396196 |
| ISPD 2008 GR | ERM | 0.837252 | 3.585434 | 0.823242 | 1.162458 |
| ISPD 2008 GR | Mixed ERM | 0.795928 | 3.368011 | 0.779342 | 1.059524 |
| ISPD 2008 GR | Hard Mining | 0.831530 | 3.575975 | 0.816725 | 1.169390 |
| ISPD 2008 GR | GroupDRO | 0.797753 | 3.369262 | 0.782520 | 1.072444 |
| ISPD 2008 GR | WDRO-REST | 0.828677 | 3.553875 | 0.813187 | 1.169081 |
| All eligible nets | ERM | 0.827848 | 3.554503 | 0.680498 | 1.162458 |
| All eligible nets | Mixed ERM | 0.788186 | 3.345608 | 0.670572 | 1.096006 |
| All eligible nets | Hard Mining | 0.822224 | 3.545271 | 0.675906 | 1.169390 |
| All eligible nets | GroupDRO | 0.789388 | 3.342178 | 0.659166 | 1.122450 |
| All eligible nets | WDRO-REST | 0.819455 | 3.521469 | 0.670049 | 1.169081 |

![各数据集mean和P95](figures/public_external_mean_p95.png)

整体汇总由ISPD08的大样本量主导，应优先查看分数据集/设计结果。不同方法和degree的不利结果全部保留。

## 配对差值与解释边界

WDRO减各基线的配对mean与design-macro差值、95%区间和各seed结果见[paired_design_cluster_ci.csv](../tables/paired_design_cluster_ci.csv)。2000次bootstrap以design-family为簇，同一家族的不同实现、所有nets和三个seed共同重采样；区间反映有限设计家族重采样，不能解释为对任意芯片的保证。SKY130HD只有一个设计家族，其区间退化，不提供有意义的跨设计不确定性估计。

![逐设计WDRO减ERM](figures/wdro_vs_erm_per_design.png)

测试度量是二维无障碍RSMT relative regret。ISPD输入的层、拥塞、容量以及真实设计的障碍、DRC、时序均不进入该指标，因此不能直接声称提高完整芯片布线评分或PPA。

## 执行与验收

- 双尺度reference：134,550条网络分别在2^27/2^28求解，共269,100次正式GeoSteiner calls，全量optimal状态、整数连通嵌入和长度相等核验通过。
- 75/75模型任务完成；固定batch256、T=1 greedy，每条预测独立Python拓扑与线长核验，所有输入/模型/reference/输出hash和配对身份通过。
- 正式GPU分配：{2: 18, 3: 18, 4: 20, 5: 19}；最大同时16个评价进程；24个CPU参考worker。GPU队列墙钟约404.3秒。
- 单任务torch最大reserved为294.0MiB。显存按实际推理需求分配；已满载共享GPU不被抢占或清理。各卡实测记录见[performance_summary.json](../logs/performance_summary.json)。
- CPU单元/统计测试、跨GPU断点恢复和完成幂等检查通过。正式结果全量独立统计复算通过；本轮GPU任务退出、文件锁可获取释放。
- 除连接完整性审计外，独立DEF+LEF复算AES/ibex/JPEG共348,507个pin，包含本轮649条net/7100个pin，最大误差0.875 DBU低于1 DBU阈值；见[独立坐标审计](../notes/selected_pin_coordinate_audit.json)。
- [独立交付审计](../delivery_verification.json)、[输入审计](../cache/input_audit.json)、[双尺度reference](../cache/reference_manifest.json)、[队列完成](../logs/queue/completion.json)、[统计入口](../summary.json)。

## 文件与恢复

统一入口是 `/remote-home/Guilin/DRO/Codex/benchmarks/public_designs_20260912/`。原始下载和大部分结果直接存于该目录；部分尚未启动任务的结果通过本目录内的逐任务符号链接接到项目已有 `Codex/shared/experiments/public_external_20260912/` 的NVMe存储，以加速原子写盘。逻辑路径与resolved path逐文件记录，详见[存储加速记录](../logs/storage_acceleration.json)。

完整入口：[实验计划](EXPERIMENT_PLAN.md)、[来源清单](../sources/contests/designs_manifest.json)、[真实设计清单](../sources/real_designs/design_manifest.json)、[冻结输入](../cache/input_manifest.json)、[逐模型结果](../results/)、[6张统计表](../tables/)、[图PNG/PDF](../figures/)。

任务已完成，不需要再次启动。若文件审计失败或中断需恢复，先检查`logs/pipeline/job.json`及任务PID、锁、心跳，再在独立tmux运行：

```bash
bash /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/benchmarks/public_designs_20260912/pipeline.py
bash /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/benchmarks/public_designs_20260912/verify_delivery.py
bash /remote-home/Guilin/DRO/Codex/wdro_precision_v2/wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python /remote-home/Guilin/DRO/Codex/benchmarks/public_designs_20260912/render_report.py
```

队列会观察仍在运行的原任务，参考/预测按已校验分片恢复；完成任务再次调用只核验现有产物。

# 真实 net 的 VAE 球壳有限候选 WDRO：首组完整对照阶段报告（seed101）


更新时间：2026-09-21T16:37:45.278586+00:00。开始时间：2026-09-21T16:07:59.025464+00:00。


代码已实现；已完成训练与完整自然测试 2/12 组。主比较为 degree30、last@256；性能结论由下列实际结果决定。


## 问题与固定实验口径

在相同真实训练数据、对应 seed 的同一 Actor/Critic 父 checkpoint、相同 256 updates × batch64 下，比较 RAW 坐标候选与 VAE 球壳候选的困难样本微调。优化器统一重新初始化，学习率 1e-5。所有方法采用同一逐步 net 索引序列、规范坐标排序、真实 REST/Lstar−1 评分；贪心 batch256，末块重复末行补齐后截取。

这是固定 K=8 候选池上的有限候选 WDRO；VAE 与候选池在三个 REST seeds 间共用。三个 seed 只度量 REST 微调随机性。VAE-only 指非 nominal 候选来源，slot0 始终保留真实原样本。

设备调度：VAE 在入场时空闲的 GPU 上拟合。若 REST 启动时所有 GPU 已被其他作业占用，则使用空闲 CPU；每个 seed 的全部方法固定同一设备类型，预算不变。设备选择记录在 artifacts/devices/，不停止其他任务。candidate_forward_rows 记录含padding的实际前向行数，cpu_rows/gpu_rows分别按实际设备计数。


## 数据与隔离

| 划分 | 保留 net 数 | 用途 |
| --- | --- | --- |
| A_train | 2810 | VAE 拟合、REST 微调与候选中心 |
| A_cal | 1202 | 半径与共同 rho 几何校准 |
| A_val | 1085 | VAE 和 REST checkpoint 选择 |
| B_test | 3991 | ISPD2008 三个保留设计 |
| C_external | 1944 | ISPD2018/2019 与 CircuitOps 外部测试 |

跨划分精确重复剔除：{}；测试内部原始重复 1 条，按方案保留。完整记录见 artifacts/splits.csv。使用全部 2810 个训练中心，没有启用 2048 截断。测试数据在历史项目中已被评估，此处实现设计隔离，不宣称全新盲测。

共同 rho=0.07090421，来自 A_cal 的 RAW 非 nominal 运输成本 q75，不读取 Actor 或测试表现。运输成本为平均 Hungarian-L1；lambda 初值0，每步按旧价选样，之后投影更新。


## 自然测试主结果：last@256

下表 mean/P95/design-macro 都以 regret 的百分数表示。Δ 为 SHELL−RAW 的百分点差，负值更好；相对 regret 下降率的分母是同 seed RAW 的 mean regret，不能解释成等比例线长缩短。

| 方法 | seed | update | B mean (%) | B P95 (%) | B design-macro (%) | 相对父模型 Δmean (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| RAW-WDRO | 101 | 256 | 2.9589 | 7.3140 | 2.8273 | 0.3969 |
| VAE-SHELL-WDRO | 101 | 256 | 2.9727 | 7.1522 | 2.8213 | 0.4107 |
| Parent | 101 | 0 | 2.5620 | 6.8304 | 2.2851 | 0.0000 |

| seed | SHELL−RAW mean (pp) | P95差 (pp) | design-macro差 (pp) | 相对 RAW regret 下降 (%) |
| --- | --- | --- | --- | --- |
| 101 | 0.0138 | -0.1618 | -0.0060 | -0.4679 |

跨 1 seeds 的配对 mean 差为 0.0138 pp，样本标准差 — pp。


## 外部来源与逐设计分解

C_external 不参与选模。CircuitOps 合计仅146 nets，且来源/设计较碎，属于探索性结果。各来源保留独立结果，未以合并平均隐藏退化。

| seed | 外部来源 | RAW mean (%) | SHELL mean (%) | SHELL−RAW (pp) |
| --- | --- | --- | --- | --- |
| 101 | circuitops_asap7 | 2.2789 | 2.1373 | -0.1416 |
| 101 | circuitops_nangate45 | 1.8742 | 1.9207 | 0.0465 |
| 101 | circuitops_sky130hd | 2.6510 | 2.7014 | 0.0505 |
| 101 | ispd2018_protocol_a | 2.0334 | 2.1861 | 0.1527 |
| 101 | ispd2019_protocol_a | 2.0721 | 2.0770 | 0.0049 |
| 101 | CircuitOps_pooled | 2.0650 | 2.0278 | -0.0372 |

| seed | B设计 | net数 | RAW mean (%) | SHELL mean (%) | SHELL−RAW (pp) |
| --- | --- | --- | --- | --- | --- |
| 101 | ispd2008_gr_adaptec5 | 320 | 2.5634 | 2.5308 | -0.0326 |
| 101 | ispd2008_gr_newblue5 | 1320 | 2.8378 | 2.8072 | -0.0305 |
| 101 | ispd2008_gr_newblue7 | 2351 | 3.0807 | 3.1258 | 0.0451 |

全部方法、全部设计的 mean/P95 见 tables/per_design.csv。逐 net 真线长、Lstar、regret 与设计信息见 artifacts/test/*.npz 和对应 *.csv；索引见 tables/test_files.csv。


## 验证最优 checkpoint（次结果）

best 由完整 A_val 的 design-macro mean regret 选择，候选包含 update0；同分保留更早 checkpoint。best=0 表示没有找到严格的验证改善，不能称为 VAE 训练收益。

| 方法 | seed | best update | B mean (%) | B P95 (%) | B design-macro (%) |
| --- | --- | --- | --- | --- | --- |
| RAW-WDRO | 101 | 0 | 2.5620 | 6.8304 | 2.2851 |
| VAE-SHELL-WDRO | 101 | 0 | 2.5620 | 6.8304 | 2.2851 |


## 生成器与选择机制诊断

选定 L=32，best epoch=280，A_val Hungarian-L1 均值 0.082672，中位数 0.081905；共同 rho=0.070904。是否触发预设 L32 回退：True；选定模型中位误差是否仍超过 rho：True。验证潜均值方差均值 0.754952，mean KL=1.170591。

一次性 VAE 拟合：L16: 300 epochs, 2.59 分钟；L32: 300 epochs, 1.49 分钟。

编码均值半径经验 q2.5/q97.5：r_min=3.020203，r_max=7.940220。这是经验边界，不是物理真实性或置信覆盖保证。

| 候选池 | 平均 unique K | 最小 K | 修复率 (%) | 最大修复位移 | 内界触发 (%) | 外界触发 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| RAW-WDRO | 8.0000 | 8 | 0.0000 | 0.00000000 | 0.0000 | 0.0000 |
| VAE-BALL-WDRO | 8.0000 | 8 | 0.0000 | 0.00000000 | 0.0000 | 14.0112 |
| VAE-SHELL-WDRO | 8.0000 | 8 | 0.0000 | 0.00000000 | 0.4220 | 14.0112 |

BALL/SHELL 非 nominal 候选逐坐标完全相同比例：99.578%。内界少触发时，无差异不能证明内界无用。

| 方法 | seed | nominal选择 (%) | 平均选中cost | 平均cost−rho | 最终lambda |
| --- | --- | --- | --- | --- | --- |
| RAW-WDRO | 101 | 11.6943 | 0.049065 | -0.021840 | 0.000000 |
| VAE-SHELL-WDRO | 101 | 35.0403 | 0.071305 | 0.000401 | 0.102686 |

有限步价格更新不保证平均运输严格满足 rho。搜索分数降低、候选变容易或遗漏困难样本均不是泛化改善的证据。修复后的候选也不再严格等于未经修复的解码输出。


## ERM 与内半径消融

| seed | 比较对象 | B SHELL−对照 mean (pp) |
| --- | --- | --- |


## 实际耗时与预算

截至本报告经过 0.496 小时墙钟时间。单组训练耗时包括共同验证和checkpoint写入；candidate/sample核心耗时另见 costs.csv。并行作业的耗时不能直接相加解释为墙钟时间。

| 方法 | seed | 设备 | updates | 训练秒 | 测试秒 | 候选真实请求 | 候选前向行数 | sampled请求 | 验证请求 | 测试请求 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW-WDRO | 101 | cpu | 256 | 1061.2 | 101.5 | 131072 | 131072 | 16384 | 5425 | 17805 |
| VAE-SHELL-WDRO | 101 | cpu | 256 | 1149.6 | 70.1 | 131072 | 131072 | 16384 | 5425 | 11870 |

每个 WDRO 完整组为131072个 greedy candidate 请求和16384个 sampled训练请求；ERM不做困难候选搜索。训练和数据预算相同，搜索与VAE预训练计算开销并不相同，全部单列。父模型自然测试跨同 seed 运行复用，不重复计入已复用的前向。

| 参考池 | 秒 | 新参考请求 | 缓存复用 distinct | 非nominal无效率 (%) |
| --- | --- | --- | --- | --- |
| RAW-WDRO | 15.5 | 19670 | 2808 | 0.0000 |
| VAE-BALL-WDRO | 0.5 | 83 | 22395 | 0.0000 |
| VAE-SHELL-WDRO | 26.8 | 19670 | 2808 | 0.0000 |

| 一次性阶段 | 实际秒 |
| --- | --- |
| data | 1.9 |
| raw_geometry | 5.6 |
| vae | 244.7 |
| latent_geometry | 8.1 |


## 当前结论与限制

已完成 1 个 seed 的主配对，仅作初步描述；不使用最好 seed 代替三 seed 结论。

本轮优先完成自然测试，未实施独立 RAW Random32/CEM 攻击；结论限于自然真实测试，不能宣称全面鲁棒性增强。B_test仅三个设计，未给出将几千个net当独立样本的显著性或过窄区间。原始负 regret 不裁剪，数值异常会停止对应任务。

未完成的组不会混入共同预算主表；进度与错误见 artifacts/status.json、artifacts/logs 和 artifacts/failures.json（如存在）。

基于正式前32步的动态预计（UTC）：seed101完整对照 2026-09-21T16:37:45.278025+00:00；三seed主对照 2026-09-21T17:04:28.493720+00:00；全部12组 2026-09-21T17:43:08.834909+00:00。含25%争用余量与每组240秒验证/测试余量，非保证；见 artifacts/runtime_estimate.json。


## 产物与复用

代码、固定原方案与配置位于本目录。大产物位于 /attached/remote-home1/Guilin/DRO/Codex/experiments/vae_wdro_real_20260921，本目录 artifacts 为链接。

- tables/main_last256.csv、main_valbest.csv：各 seed/来源主结果。
- tables/paired_shell_minus_raw.csv、paired_across_seeds.csv、paired_per_design.csv：配对效果与种子分散程度。
- tables/costs.csv：实际更新数、搜索/训练/验证/测试请求与padding GPU行数。
- artifacts/test/：逐net自然测试 NPZ/CSV；artifacts/runs/：latest/last/val_best和训练曲线。
- artifacts/pools/、reference_cache.sqlite：固定候选及共享参考。
- artifacts/vae/：best/last、曲线、校准和验证重建图。
- artifacts/figures/：主对照mean/P95、逐设计配对差（PNG/SVG）。


![B_test mean/P95](/attached/remote-home1/Guilin/DRO/Codex/experiments/vae_wdro_real_20260921/figures/main_mean_p95.png)

![逐设计配对差](/attached/remote-home1/Guilin/DRO/Codex/experiments/vae_wdro_real_20260921/figures/per_design_paired_difference.png)


下一步按实际结果决定：若重建误差/nominal选择暴露覆盖问题，先研究局部可用性；若自然泛化有一致收益，再用共享独立 RAW 攻击检验是否遗漏困难样本。不得把本轮有限候选结果解释成连续 Wasserstein 认证。

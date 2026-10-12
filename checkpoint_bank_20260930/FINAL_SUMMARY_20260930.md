# WDRO checkpoint bank and REST geometry: pilot results, 2026-09-30

## Checkpoint bank: independent-reference final update200
Degree10, seeds101/102, five arms, 200 updates x batch64 per arm (12800 sample-updates; total128000). Shared parent actor, critic, optimizers and scheduler. GPU5 FP32. 32 perturbations plus nominal are reduced by a fixed geometric shortlist to four scored candidates. Each comparison uses the same 1024 cached-reference test instances (256 per distribution). Teacher selection uses historical validation worst-OOD mean relative regret, subject to nominal feasibility: the minimum is this metric, not checkpoint size or training loss. Eligible banks use best checkpoints from prior runs, not the top epochs of one run. Test labels do not select bank members.

|Arm|Seed|Final mean regret %|Worst OOD %|Delta vs Geo, percentage points [95% CI]|Update seconds|Job seconds|Train Geo nets|Teacher nets|
|---|---:|---:|---:|---|---:|---:|---:|---:|
|geo|101|0.314698|0.392098|+0.000000 [+0.000000, +0.000000]|87.32|132.95|51200|0|
|k1|101|0.269017|0.363923|-0.045681 [-0.093431, -0.003917]|65.76|79.05|0|51200|
|k3|101|0.311640|0.332089|-0.003058 [-0.056209, +0.048769]|79.46|93.52|0|153600|
|k5|101|0.341686|0.447157|+0.026988 [-0.027730, +0.077253]|92.12|106.78|0|256000|
|refresh3|101|0.311640|0.332089|-0.003058 [-0.056209, +0.048769]|80.10|95.06|0|153600|
|geo|102|0.299828|0.318987|+0.000000 [+0.000000, +0.000000]|85.69|99.44|51200|0|
|k1|102|0.334768|0.406181|+0.034940 [-0.014521, +0.088047]|67.98|81.29|0|51200|
|k3|102|0.307131|0.354545|+0.007303 [-0.036443, +0.047342]|79.72|93.49|0|153600|
|k5|102|0.360401|0.412786|+0.060573 [+0.014057, +0.109928]|92.66|107.27|0|256000|
|refresh3|102|0.307131|0.354545|+0.007303 [-0.036443, +0.047342]|76.71|92.14|0|153600|

All ten validation-selected best snapshots are step0, the unchanged parent (test mean regret approximately0.234418%). Every final200 model is worse than that initial parent on mean test regret. Identical best results cannot establish equivalent learning. The refresh3 training bank has exactly one distinct membership tuple across all200 steps in each seed; it did not exercise an actual bank replacement and final results equal fixedK3.

Final mean delta averaged over the two seeds: K1 -0.005370 percentage points; K3 +0.002122; K5 +0.043780; refresh3 +0.002122. K1 changes sign between seeds. K5 seed102 is worse than Geo with conditional instance CI excluding zero. K3 intervals cross zero; absence of a detected difference is not equivalence. CIs use500 stratified paired instance bootstrap replicates, conditional on each trained model, not seed-level uncertainty. Two seeds,200 continuation steps, no equivalence margin and no improving validation snapshot cannot establish long-run replacement quality.

## Reference bias and cost
Actual regret is Ctheta/OPT-1. Feasible bank B=min_j Cj >= OPT makes Ctheta/B-1 a downward-biased proxy and can change adversarial rankings. Independent original references are retained for evaluation.

|Bank|Offline mean bank gap %|Proxy minus true regret, pp|Argmax agreement|
|---|---:|---:|---:|
|k1|0.234915|-0.234915|0.21875|
|k3|0.059461|-0.059557|0.87500|
|k5|0.052715|-0.052761|0.87500|

Offline diagnostic: best step0 actor on32 fixed uniform centers x9 candidates. Results repeat across seeds because actor and diagnostic inputs are identical, not independent replications. For K1 the evaluated actor is its own sole teacher, making proxy regret constant zero; its21.875% argmax agreement is tie-dependent, and within-center Spearman is undefined (NaN), not zero. This diagnostic does not evaluate the final200 actor. K3/K5 agreement87.5% still leaves ranking mismatches. Geo arm is diagnosed against a separateK3 bank, never the oracle against itself.

Queue wall time 981.02 seconds. Training GeoSteiner nets total 102400; all eight bank arms have zero training Geo solves. Teacher scored nets total 1228800 (padding excluded). Bank offline diagnostics use2880 fresh oracle nets. Geometry attack adds288, giving3168 offline and105568 training-plus-offline oracle nets; pre-existing cached labels are excluded. Update timing excludes periodic validation/test/diagnostics; full job timing includes those phases. Shared GPU timing is not a controlled hardware benchmark.

Zero training Geo solves does not mean an oracle-free history: historical teachers and cached validation labels are oracle-assisted. The shared original ReLU critic also limits representation of negative bank targets. Bank values remain frozen within inner search and refresh intervals.

## Conditional geometric uncertainty and hard-instance selection
REST factorizes p(i,j,o)=p(i)p(j,o|i). This study observes only p(j,o|decoded i): conditional diagnostics, not complete joint entropy. Degree30, temperature1, two frozen checkpoints (ERM M0900 and WDRO M1118), four families x512 common instances =2048 unique instances and4096 model evaluations. No n-fold endpoint enumeration.

With a,b denoting conditional second-endpoint/direction actions: Q=sum p_a p_b d(a,b); G=-0.5 sum p_a p_b d(a,b)(ln p_a+ln p_b); C=sum p_a d(a,a_decode). Natural logs and zero-probability limits are used. H/logK and max probability remain controls. Each L-route is represented at arc fractions0,1/2,1; d is the minimum forward/reverse average Manhattan matching distance divided by input bbox width+height. This distinguishes L directions without independent x/y scaling. Raw action entropy preserves state-distinct actions.

H Spearman correlations with true error span0.1129-0.3034; G spans0.1627-0.3296. These are modest associations, not a guarantee that high entropy implies high error. G is not superior in every group (WDRO mix: H0.3034, G0.2740). Uniform top-decile error precision is0.308 for G in both models, versus0.173/0.096 for H and prevalence0.102. Marginal correlation intervals do not establish a paired G-minus-H improvement.

Existing-pool hard selection uses disjoint candidate blocks with paired500-replicate block bootstrap (budgets8/16). For budget16, G-minus-random true error increases include ERM grid +0.586pp [0.260,0.922], ERM uniform +0.375 [0.016,0.734], WDRO edge +0.529 [0.043,0.992], WDRO mix +0.522 [0.109,1.001], WDRO uniform +0.382 [0.076,0.679]. Other intervals can cross zero; all tables are appended.

Separate finite perturbation search uses32 centers,8 candidates each plus nominal. Scores and choices are fixed before offline oracle evaluation of288 unique inputs shared across models. ERM G-selected regret1.635% vs random1.071%, paired difference+0.564pp [0.156,0.940]. WDRO G1.121% vs random0.996%, +0.125pp [-0.233,0.476]; WDRO C1.386%, +0.390pp [0.041,0.753]. No universal metric winner. Intervals use1000 paired center bootstraps; effective sampling units are32 centers, not256 candidates, with only8 centers per family.

Perturbations begin at coordinate noise +/-0.02 then undergo longest-side bbox normalization; final distances may exceed0.02. This is finite candidate search, not an epsilon-constrained WDRO inner optimum. No training using these geometry-selected samples was performed. Multiple comparisons are unadjusted; checkpoints have no seed replication here; historical test corpora are not a fresh confirmation set. Diagnostic runtime approximately322.02s; finite-search runtime approximately68.77s.

The first diagnostic REPORT describes its own phase as having no new perturbation attack; the separate ATTACK_REPORT below supplies that subsequent experiment. Neither phase demonstrates improved training.

## Source reports
Complete original comparisons follow. Source JSONs retain higher precision and per-family/step statistics.

# Checkpoint bank versus GeoSteiner reference training

All differences are arm minus Geo on the same held-out instance IDs; negative means lower true-reference regret. PRIMARY comparison is final update200; validation-selected best is secondary and may select the unchanged step0 parent. Identical best-parent results do not show equivalent learning. Both use the original cached reference, not the bank denominator.

| Arm | Seed | Final200 delta [95% instance CI] | Best selected step | Best delta [95% instance CI] | Training Geo nets | Teacher forward nets | Teacher seconds | Update seconds | Successful job seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo | 101 | +0.00000000 [+0.00000000, +0.00000000] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 51200 | 0 | 0.00 | 87.32 | 132.94873428344727 |
| k1 | 101 | -0.00045681 [-0.00093431, -0.00003917] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 51200 | 6.36 | 65.76 | 79.04838252067566 |
| k3 | 101 | -0.00003058 [-0.00056209, +0.00048769] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 153600 | 19.24 | 79.46 | 93.51573252677917 |
| k5 | 101 | +0.00026988 [-0.00027730, +0.00077253] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 256000 | 32.21 | 92.12 | 106.78029084205627 |
| refresh3 | 101 | -0.00003058 [-0.00056209, +0.00048769] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 153600 | 19.44 | 80.10 | 95.06356763839722 |
| geo | 102 | +0.00000000 [+0.00000000, +0.00000000] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 51200 | 0 | 0.00 | 85.69 | 99.43662595748901 |
| k1 | 102 | +0.00034940 [-0.00014521, +0.00088047] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 51200 | 6.53 | 67.98 | 81.28870868682861 |
| k3 | 102 | +0.00007303 [-0.00036443, +0.00047342] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 153600 | 19.51 | 79.72 | 93.49064493179321 |
| k5 | 102 | +0.00060573 [+0.00014057, +0.00109928] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 256000 | 32.48 | 92.66 | 107.27055788040161 |
| refresh3 | 102 | +0.00007303 [-0.00036443, +0.00047342] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 153600 | 17.92 | 76.71 | 92.13578271865845 |

| Arm | Observed seeds | Best delta mean [range] | Final delta mean [range] |
|---|---:|---:|---:|
| geo | 2 | +0.00000000 [0.0, 0.0] | +0.00000000 [0.0, 0.0] |
| k1 | 2 | +0.00000000 [0.0, 0.0] | -0.00005370 [-0.00045681183803155463, 0.0003494030890748578] |
| k3 | 2 | +0.00000000 [0.0, 0.0] | +0.00002122 [-3.0584026799954566e-05, 7.30275882898665e-05] |
| k5 | 2 | +0.00000000 [0.0, 0.0] | +0.00043780 [0.00026988172826160795, 0.0006057270677202878] |
| refresh3 | 2 | +0.00000000 [0.0, 0.0] | +0.00002122 [-3.0584026799954566e-05, 7.30275882898665e-05] |

Confidence intervals use 500 bootstrap replicates of instances, stratified by the four fixed groups (1024 observed instances per completed comparison). They describe instance uncertainty conditional on each trained model, not seed-to-seed uncertainty. Seed summaries report only actual available seeds; two seeds are too few for a strong training-variance claim. No equivalence margin was specified, so these results cannot establish statistical equivalence.

Offline Geo-arm diagnostic compares its actor against a separate K3 teacher bank; its agreement score is not a measure of the exact reference agreeing with itself. Other arms use their own banks, so different diagnostic banks must be acknowledged. Training Geo nets exclude the offline candidate diagnostic (288 oracle nets per completed arm) and cached validation/test labels. Teacher-forward nets are nominal scored nets; fixed-batch padding is not added to that counter. Teacher seconds include teacher inference and CPU REST length evaluation. Update seconds exclude periodic validation, test and offline diagnostic; successful queue seconds include the successful process run and can exclude earlier interrupted segments. Historical teachers and validation remain oracle-assisted.

Incomplete: none


# REST 条件几何不确定性：探索性实验结果

固定 d30、FP32 greedy、温度 1；每个模型/分布分别分析。指标观测已解码第一端点条件下的第二端点与方向概率，不是完整联合点对熵。真实误差采用 precision_v2 既有参考。所有区间是按实例或候选块重采样的 95% 百分位区间。

## M0900 / D0218，n=512

平均真实相对误差 0.842%，P95 2.316%。

| 指标 | Pearson | Spearman | 高分前10%中高误差比例 | 高误差基率 |
|---|---:|---:|---:|---:|
| Hnorm | 0.1094 | 0.1129 | 0.135 | 0.102 |
| maxp | -0.0866 | -0.0761 | 0.135 | 0.102 |
| Q | 0.1185 | 0.1298 | 0.096 | 0.102 |
| G | 0.1422 | 0.1627 | 0.135 | 0.102 |
| C | 0.1072 | 0.1170 | 0.096 | 0.102 |
| first_Hnorm | 0.1391 | 0.1235 | 0.135 | 0.102 |

maxp 表中相关系数使用原始最大概率，困难样本选择使用最低 maxp。各项置信区间及分步阶段结果在同目录 summary.json；相关系数未定义表示常量或样本不足。

| 候选预算 | 指标 | 所选平均真实误差% | 相对random绝对差/百分点 | 差值95%区间/百分点 |
|---:|---|---:|---:|---|
| 8 | random | 0.880 | 0.000 | [0.000, 0.000] |
| 8 | oracle_upper_bound | 2.222 | 1.342 | [1.121, 1.573] |
| 8 | Hnorm | 0.921 | 0.041 | [-0.286, 0.315] |
| 8 | maxp | 0.874 | -0.006 | [-0.342, 0.368] |
| 8 | Q | 0.949 | 0.069 | [-0.255, 0.355] |
| 8 | G | 1.082 | 0.203 | [-0.111, 0.496] |
| 8 | C | 0.979 | 0.100 | [-0.182, 0.382] |
| 8 | first_Hnorm | 1.023 | 0.144 | [-0.070, 0.385] |
| 16 | random | 0.796 | 0.000 | [0.000, 0.000] |
| 16 | oracle_upper_bound | 2.711 | 1.915 | [1.477, 2.316] |
| 16 | Hnorm | 1.052 | 0.256 | [-0.062, 0.584] |
| 16 | maxp | 0.940 | 0.144 | [-0.168, 0.506] |
| 16 | Q | 1.013 | 0.217 | [-0.118, 0.504] |
| 16 | G | 1.138 | 0.342 | [-0.045, 0.670] |
| 16 | C | 1.040 | 0.244 | [-0.106, 0.555] |
| 16 | first_Hnorm | 0.966 | 0.170 | [-0.236, 0.650] |

这些候选为冻结测试池的互斥区块；结果只衡量离线困难样本筛选，不能解释为生成扰动或提升WDRO训练效果。

## M0900 / D0223，n=512

平均真实相对误差 0.781%，P95 2.095%。

| 指标 | Pearson | Spearman | 高分前10%中高误差比例 | 高误差基率 |
|---|---:|---:|---:|---:|
| Hnorm | 0.1595 | 0.1596 | 0.173 | 0.102 |
| maxp | -0.1858 | -0.1886 | 0.192 | 0.102 |
| Q | 0.2266 | 0.2326 | 0.154 | 0.102 |
| G | 0.2734 | 0.2710 | 0.173 | 0.102 |
| C | 0.2254 | 0.2313 | 0.173 | 0.102 |
| first_Hnorm | 0.1852 | 0.1561 | 0.231 | 0.102 |

maxp 表中相关系数使用原始最大概率，困难样本选择使用最低 maxp。各项置信区间及分步阶段结果在同目录 summary.json；相关系数未定义表示常量或样本不足。

| 候选预算 | 指标 | 所选平均真实误差% | 相对random绝对差/百分点 | 差值95%区间/百分点 |
|---:|---|---:|---:|---|
| 8 | random | 0.840 | 0.000 | [0.000, 0.000] |
| 8 | oracle_upper_bound | 1.932 | 1.092 | [0.879, 1.309] |
| 8 | Hnorm | 0.930 | 0.090 | [-0.177, 0.364] |
| 8 | maxp | 0.980 | 0.140 | [-0.138, 0.460] |
| 8 | Q | 1.011 | 0.171 | [-0.057, 0.421] |
| 8 | G | 1.081 | 0.241 | [-0.051, 0.523] |
| 8 | C | 1.050 | 0.209 | [-0.050, 0.473] |
| 8 | first_Hnorm | 0.908 | 0.068 | [-0.140, 0.313] |
| 16 | random | 0.592 | 0.000 | [0.000, 0.000] |
| 16 | oracle_upper_bound | 2.368 | 1.776 | [1.458, 2.098] |
| 16 | Hnorm | 1.133 | 0.541 | [0.301, 0.817] |
| 16 | maxp | 1.132 | 0.540 | [0.235, 0.851] |
| 16 | Q | 1.074 | 0.482 | [0.207, 0.762] |
| 16 | G | 1.178 | 0.586 | [0.260, 0.922] |
| 16 | C | 1.177 | 0.586 | [0.293, 0.859] |
| 16 | first_Hnorm | 1.050 | 0.458 | [0.167, 0.745] |

这些候选为冻结测试池的互斥区块；结果只衡量离线困难样本筛选，不能解释为生成扰动或提升WDRO训练效果。

## M0900 / D0228，n=512

平均真实相对误差 1.214%，P95 3.011%。

| 指标 | Pearson | Spearman | 高分前10%中高误差比例 | 高误差基率 |
|---|---:|---:|---:|---:|
| Hnorm | 0.1667 | 0.1873 | 0.115 | 0.102 |
| maxp | -0.1480 | -0.1675 | 0.135 | 0.102 |
| Q | 0.1797 | 0.2299 | 0.212 | 0.102 |
| G | 0.2331 | 0.2538 | 0.250 | 0.102 |
| C | 0.1520 | 0.2143 | 0.173 | 0.102 |
| first_Hnorm | 0.2570 | 0.2724 | 0.250 | 0.102 |

maxp 表中相关系数使用原始最大概率，困难样本选择使用最低 maxp。各项置信区间及分步阶段结果在同目录 summary.json；相关系数未定义表示常量或样本不足。

| 候选预算 | 指标 | 所选平均真实误差% | 相对random绝对差/百分点 | 差值95%区间/百分点 |
|---:|---|---:|---:|---|
| 8 | random | 1.267 | 0.000 | [0.000, 0.000] |
| 8 | oracle_upper_bound | 2.892 | 1.625 | [1.320, 1.952] |
| 8 | Hnorm | 1.404 | 0.138 | [-0.150, 0.422] |
| 8 | maxp | 1.368 | 0.101 | [-0.141, 0.355] |
| 8 | Q | 1.356 | 0.089 | [-0.218, 0.440] |
| 8 | G | 1.626 | 0.360 | [0.056, 0.727] |
| 8 | C | 1.366 | 0.100 | [-0.235, 0.453] |
| 8 | first_Hnorm | 1.354 | 0.087 | [-0.256, 0.404] |
| 16 | random | 1.187 | 0.000 | [0.000, 0.000] |
| 16 | oracle_upper_bound | 3.542 | 2.355 | [1.924, 2.814] |
| 16 | Hnorm | 1.373 | 0.186 | [-0.246, 0.571] |
| 16 | maxp | 1.263 | 0.076 | [-0.334, 0.488] |
| 16 | Q | 1.308 | 0.121 | [-0.337, 0.607] |
| 16 | G | 1.663 | 0.476 | [-0.095, 1.085] |
| 16 | C | 1.271 | 0.085 | [-0.392, 0.572] |
| 16 | first_Hnorm | 1.545 | 0.358 | [-0.146, 0.874] |

这些候选为冻结测试池的互斥区块；结果只衡量离线困难样本筛选，不能解释为生成扰动或提升WDRO训练效果。

## M0900 / D0233，n=512

平均真实相对误差 0.776%，P95 2.116%。

| 指标 | Pearson | Spearman | 高分前10%中高误差比例 | 高误差基率 |
|---|---:|---:|---:|---:|
| Hnorm | 0.2330 | 0.2278 | 0.173 | 0.102 |
| maxp | -0.2371 | -0.2019 | 0.212 | 0.102 |
| Q | 0.3263 | 0.2899 | 0.308 | 0.102 |
| G | 0.3602 | 0.3296 | 0.308 | 0.102 |
| C | 0.3159 | 0.2834 | 0.308 | 0.102 |
| first_Hnorm | 0.2054 | 0.2050 | 0.231 | 0.102 |

maxp 表中相关系数使用原始最大概率，困难样本选择使用最低 maxp。各项置信区间及分步阶段结果在同目录 summary.json；相关系数未定义表示常量或样本不足。

| 候选预算 | 指标 | 所选平均真实误差% | 相对random绝对差/百分点 | 差值95%区间/百分点 |
|---:|---|---:|---:|---|
| 8 | random | 0.890 | 0.000 | [0.000, 0.000] |
| 8 | oracle_upper_bound | 1.900 | 1.010 | [0.833, 1.191] |
| 8 | Hnorm | 1.051 | 0.161 | [-0.065, 0.415] |
| 8 | maxp | 1.139 | 0.249 | [0.018, 0.459] |
| 8 | Q | 1.063 | 0.173 | [-0.035, 0.378] |
| 8 | G | 1.091 | 0.201 | [-0.026, 0.417] |
| 8 | C | 1.096 | 0.206 | [0.004, 0.428] |
| 8 | first_Hnorm | 0.954 | 0.064 | [-0.129, 0.256] |
| 16 | random | 0.926 | 0.000 | [0.000, 0.000] |
| 16 | oracle_upper_bound | 2.291 | 1.365 | [1.030, 1.677] |
| 16 | Hnorm | 1.176 | 0.250 | [-0.095, 0.548] |
| 16 | maxp | 1.348 | 0.422 | [0.023, 0.849] |
| 16 | Q | 1.236 | 0.310 | [-0.020, 0.652] |
| 16 | G | 1.301 | 0.375 | [0.016, 0.734] |
| 16 | C | 1.207 | 0.281 | [-0.057, 0.654] |
| 16 | first_Hnorm | 1.057 | 0.131 | [-0.251, 0.478] |

这些候选为冻结测试池的互斥区块；结果只衡量离线困难样本筛选，不能解释为生成扰动或提升WDRO训练效果。

## M1118 / D0218，n=512

平均真实相对误差 0.817%，P95 2.395%。

| 指标 | Pearson | Spearman | 高分前10%中高误差比例 | 高误差基率 |
|---|---:|---:|---:|---:|
| Hnorm | 0.0943 | 0.1345 | 0.115 | 0.102 |
| maxp | -0.0718 | -0.1085 | 0.058 | 0.102 |
| Q | 0.2202 | 0.2308 | 0.154 | 0.102 |
| G | 0.2811 | 0.2695 | 0.192 | 0.102 |
| C | 0.2103 | 0.2201 | 0.154 | 0.102 |
| first_Hnorm | 0.1409 | 0.1366 | 0.135 | 0.102 |

maxp 表中相关系数使用原始最大概率，困难样本选择使用最低 maxp。各项置信区间及分步阶段结果在同目录 summary.json；相关系数未定义表示常量或样本不足。

| 候选预算 | 指标 | 所选平均真实误差% | 相对random绝对差/百分点 | 差值95%区间/百分点 |
|---:|---|---:|---:|---|
| 8 | random | 0.701 | 0.000 | [0.000, 0.000] |
| 8 | oracle_upper_bound | 2.193 | 1.492 | [1.243, 1.752] |
| 8 | Hnorm | 0.842 | 0.140 | [-0.044, 0.341] |
| 8 | maxp | 0.782 | 0.080 | [-0.193, 0.328] |
| 8 | Q | 1.196 | 0.494 | [0.229, 0.781] |
| 8 | G | 1.330 | 0.629 | [0.344, 0.926] |
| 8 | C | 1.189 | 0.487 | [0.208, 0.800] |
| 8 | first_Hnorm | 0.951 | 0.250 | [0.038, 0.472] |
| 16 | random | 0.827 | 0.000 | [0.000, 0.000] |
| 16 | oracle_upper_bound | 2.698 | 1.871 | [1.429, 2.286] |
| 16 | Hnorm | 0.843 | 0.016 | [-0.368, 0.456] |
| 16 | maxp | 0.704 | -0.123 | [-0.400, 0.142] |
| 16 | Q | 1.196 | 0.369 | [-0.126, 0.834] |
| 16 | G | 1.357 | 0.529 | [0.043, 0.992] |
| 16 | C | 1.160 | 0.332 | [-0.087, 0.856] |
| 16 | first_Hnorm | 1.183 | 0.355 | [-0.109, 0.869] |

这些候选为冻结测试池的互斥区块；结果只衡量离线困难样本筛选，不能解释为生成扰动或提升WDRO训练效果。

## M1118 / D0223，n=512

平均真实相对误差 0.725%，P95 1.939%。

| 指标 | Pearson | Spearman | 高分前10%中高误差比例 | 高误差基率 |
|---|---:|---:|---:|---:|
| Hnorm | 0.1736 | 0.1784 | 0.173 | 0.102 |
| maxp | -0.1704 | -0.1712 | 0.192 | 0.102 |
| Q | 0.1993 | 0.2099 | 0.173 | 0.102 |
| G | 0.2445 | 0.2438 | 0.231 | 0.102 |
| C | 0.1922 | 0.2097 | 0.192 | 0.102 |
| first_Hnorm | 0.0740 | 0.0931 | 0.135 | 0.102 |

maxp 表中相关系数使用原始最大概率，困难样本选择使用最低 maxp。各项置信区间及分步阶段结果在同目录 summary.json；相关系数未定义表示常量或样本不足。

| 候选预算 | 指标 | 所选平均真实误差% | 相对random绝对差/百分点 | 差值95%区间/百分点 |
|---:|---|---:|---:|---|
| 8 | random | 0.836 | 0.000 | [0.000, 0.000] |
| 8 | oracle_upper_bound | 1.799 | 0.963 | [0.776, 1.143] |
| 8 | Hnorm | 0.905 | 0.069 | [-0.176, 0.275] |
| 8 | maxp | 0.819 | -0.017 | [-0.214, 0.177] |
| 8 | Q | 0.950 | 0.114 | [-0.121, 0.354] |
| 8 | G | 0.955 | 0.119 | [-0.109, 0.358] |
| 8 | C | 0.929 | 0.093 | [-0.149, 0.350] |
| 8 | first_Hnorm | 0.765 | -0.071 | [-0.296, 0.152] |
| 16 | random | 0.910 | 0.000 | [0.000, 0.000] |
| 16 | oracle_upper_bound | 2.189 | 1.279 | [1.041, 1.544] |
| 16 | Hnorm | 0.949 | 0.039 | [-0.291, 0.426] |
| 16 | maxp | 0.897 | -0.013 | [-0.313, 0.289] |
| 16 | Q | 1.040 | 0.130 | [-0.139, 0.437] |
| 16 | G | 1.122 | 0.212 | [-0.056, 0.491] |
| 16 | C | 1.023 | 0.113 | [-0.183, 0.423] |
| 16 | first_Hnorm | 0.790 | -0.120 | [-0.456, 0.184] |

这些候选为冻结测试池的互斥区块；结果只衡量离线困难样本筛选，不能解释为生成扰动或提升WDRO训练效果。

## M1118 / D0228，n=512

平均真实相对误差 1.138%，P95 2.864%。

| 指标 | Pearson | Spearman | 高分前10%中高误差比例 | 高误差基率 |
|---|---:|---:|---:|---:|
| Hnorm | 0.2761 | 0.3034 | 0.115 | 0.102 |
| maxp | -0.2537 | -0.2721 | 0.212 | 0.102 |
| Q | 0.2178 | 0.2308 | 0.154 | 0.102 |
| G | 0.2567 | 0.2740 | 0.192 | 0.102 |
| C | 0.2048 | 0.2217 | 0.154 | 0.102 |
| first_Hnorm | 0.2437 | 0.2434 | 0.173 | 0.102 |

maxp 表中相关系数使用原始最大概率，困难样本选择使用最低 maxp。各项置信区间及分步阶段结果在同目录 summary.json；相关系数未定义表示常量或样本不足。

| 候选预算 | 指标 | 所选平均真实误差% | 相对random绝对差/百分点 | 差值95%区间/百分点 |
|---:|---|---:|---:|---|
| 8 | random | 1.048 | 0.000 | [0.000, 0.000] |
| 8 | oracle_upper_bound | 2.643 | 1.594 | [1.335, 1.849] |
| 8 | Hnorm | 1.437 | 0.388 | [0.034, 0.707] |
| 8 | maxp | 1.350 | 0.302 | [-0.025, 0.623] |
| 8 | Q | 1.385 | 0.337 | [0.060, 0.578] |
| 8 | G | 1.486 | 0.438 | [0.131, 0.725] |
| 8 | C | 1.293 | 0.245 | [-0.044, 0.549] |
| 8 | first_Hnorm | 1.332 | 0.284 | [-0.026, 0.597] |
| 16 | random | 1.023 | 0.000 | [0.000, 0.000] |
| 16 | oracle_upper_bound | 3.208 | 2.185 | [1.765, 2.620] |
| 16 | Hnorm | 1.451 | 0.428 | [0.027, 0.814] |
| 16 | maxp | 1.392 | 0.369 | [-0.060, 0.782] |
| 16 | Q | 1.212 | 0.189 | [-0.102, 0.531] |
| 16 | G | 1.545 | 0.522 | [0.109, 1.001] |
| 16 | C | 1.200 | 0.177 | [-0.152, 0.557] |
| 16 | first_Hnorm | 1.545 | 0.522 | [0.166, 0.945] |

这些候选为冻结测试池的互斥区块；结果只衡量离线困难样本筛选，不能解释为生成扰动或提升WDRO训练效果。

## M1118 / D0233，n=512

平均真实相对误差 0.716%，P95 1.985%。

| 指标 | Pearson | Spearman | 高分前10%中高误差比例 | 高误差基率 |
|---|---:|---:|---:|---:|
| Hnorm | 0.1433 | 0.1542 | 0.096 | 0.102 |
| maxp | -0.1677 | -0.1600 | 0.212 | 0.102 |
| Q | 0.2668 | 0.2450 | 0.269 | 0.102 |
| G | 0.3155 | 0.2832 | 0.308 | 0.102 |
| C | 0.2574 | 0.2336 | 0.288 | 0.102 |
| first_Hnorm | 0.1103 | 0.1288 | 0.154 | 0.102 |

maxp 表中相关系数使用原始最大概率，困难样本选择使用最低 maxp。各项置信区间及分步阶段结果在同目录 summary.json；相关系数未定义表示常量或样本不足。

| 候选预算 | 指标 | 所选平均真实误差% | 相对random绝对差/百分点 | 差值95%区间/百分点 |
|---:|---|---:|---:|---|
| 8 | random | 0.842 | 0.000 | [0.000, 0.000] |
| 8 | oracle_upper_bound | 1.761 | 0.919 | [0.739, 1.091] |
| 8 | Hnorm | 0.890 | 0.049 | [-0.138, 0.250] |
| 8 | maxp | 0.886 | 0.045 | [-0.191, 0.244] |
| 8 | Q | 1.006 | 0.165 | [-0.092, 0.387] |
| 8 | G | 1.094 | 0.252 | [0.023, 0.469] |
| 8 | C | 1.020 | 0.178 | [-0.039, 0.396] |
| 8 | first_Hnorm | 0.722 | -0.119 | [-0.338, 0.130] |
| 16 | random | 0.742 | 0.000 | [0.000, 0.000] |
| 16 | oracle_upper_bound | 2.160 | 1.418 | [1.093, 1.761] |
| 16 | Hnorm | 0.752 | 0.011 | [-0.313, 0.427] |
| 16 | maxp | 0.889 | 0.147 | [-0.229, 0.592] |
| 16 | Q | 1.124 | 0.383 | [0.095, 0.675] |
| 16 | G | 1.124 | 0.382 | [0.076, 0.679] |
| 16 | C | 1.144 | 0.402 | [0.050, 0.772] |
| 16 | first_Hnorm | 0.741 | -0.001 | [-0.279, 0.272] |

这些候选为冻结测试池的互斥区块；结果只衡量离线困难样本筛选，不能解释为生成扰动或提升WDRO训练效果。

## 数据支持的范围

已生成 8 个模型×分布结果。未开展训练效果实验或新扰动攻击；这里不能推断这两类效果。高熵是否对应高误差应逐组以相关系数和筛选结果判断，不能由指标定义保证。多指标探索未做多重比较校正，历史测试集不是新独立确认集。


# 几何不确定性有限扰动搜索：探索性结果

32个中心（每分布8个），每中心8个固定独立扰动候选和1个nominal。所有指标在同候选池选择，之后才用GeoSteiner精评全部288实例；真值不反馈搜索。原始扰动每坐标±0.02，经bbox最长边统一缩放到0.8后最终距离可能超过0.02，实际indexed L1及最大L∞记录在summary.json。不是既定ε约束的WDRO内层优化。

| 模型 | 方法 | 所选平均真实误差% | 相对random差/百分点 | 95%区间/百分点 | 相对nominal差/百分点 |
|---|---|---:|---:|---|---:|
| M0900 | random | 1.071 | 0.000 | [0.000,0.000] | -0.155 |
| M0900 | Hnorm | 1.475 | 0.404 | [0.025,0.755] | 0.249 |
| M0900 | maxp | 1.491 | 0.421 | [0.080,0.743] | 0.266 |
| M0900 | Q | 1.566 | 0.495 | [0.106,0.902] | 0.340 |
| M0900 | G | 1.635 | 0.564 | [0.156,0.940] | 0.409 |
| M0900 | C | 1.509 | 0.438 | [0.066,0.841] | 0.283 |
| M0900 | first_Hnorm | 1.388 | 0.317 | [-0.092,0.703] | 0.162 |
| M0900 | oracle_upper_bound | 2.374 | 1.303 | [0.982,1.607] | 1.148 |
| M0900 | nominal | 1.226 | 0.155 | [-0.317,0.615] | 0.000 |
| M1118 | random | 0.996 | 0.000 | [0.000,0.000] | 0.064 |
| M1118 | Hnorm | 1.066 | 0.070 | [-0.349,0.489] | 0.134 |
| M1118 | maxp | 1.059 | 0.063 | [-0.285,0.412] | 0.127 |
| M1118 | Q | 1.229 | 0.233 | [-0.114,0.600] | 0.297 |
| M1118 | G | 1.121 | 0.125 | [-0.233,0.476] | 0.189 |
| M1118 | C | 1.386 | 0.390 | [0.041,0.753] | 0.454 |
| M1118 | first_Hnorm | 1.166 | 0.170 | [-0.301,0.723] | 0.234 |
| M1118 | oracle_upper_bound | 2.267 | 1.271 | [0.842,1.735] | 1.335 |
| M1118 | nominal | 0.932 | -0.064 | [-0.428,0.284] | 0.000 |

区间按中心配对重采样，不将同一中心的8个候选当独立样本。每分布仅8中心，分组结果在summary.json。oracle_upper_bound是仅供离线上限比较的真实最大误差选择。负的相对random差说明该指标在此样本中没有优于随机；不能预设高熵对应高误差。未运行基于这些候选的训练，因此不能推断训练收益。
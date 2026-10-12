# 阶段6 GeoSteiner参考精度阻塞与修复建议

2026-09-07。当前状态：`BLOCKED`，停在6.3。14个新增WDRO训练及完整审计、75模型冻结已经完成；合成test尚未完成，不能生成最终主表或标记阶段6 DONE。

## 已完成与保留的恢复点

- 75个模型（5方法×5degree×3seed）的150个latest/best及状态重验通过，冻结索引仍为`wdro_legacy_v1/wdro_rest_min/manifests/stage6_models_frozen_v1.json`，冻结时间早于test。新增14个WDRO共28000updates及14份完整审计616项检查通过。
- 已提交13/300完整test格；全部游标合计144336条预测，逐分片和完整预测hash均通过。主heartbeat中的8格是中途计数，当前以`stage6_reference_blocker_v1.json`逐格清单为准。
- 主控制器PID101825、d40 PID121808、d30 PID121824、d50 PID121792均已退出；d50旧job/heartbeat中的running/evaluating是失去进程后的历史状态。原作业文件保留，不伪造正常完成。
- d40及d30评价日志均因regret断言失败；本次逐样本复现和归因以d40反例为准。d50在控制器停止依赖任务时保留已提交分片。
- 6.4统计、paired CI、交付复核及阶段7均未启动。已有结果是旧reference下的中途产物，不能当作精确regret结论。

## 可复现的反例

失败样本为`test:edge_bias:d40:s303:n008760`，模型为冻结的ERM/d40/s101。输入、模型、缓存及失败batch的hash和双路径均在`shared/experiments/logs/stage6_negative_regret/diagnosis.json`。

| 检查 | 长度或结果 |
|---|---|
| REST C evaluator | 3.707879826426506 |
| 独立Python按线段区间求和 | 3.707879826426506 |
| REST拓扑连通分量 | 1，40个terminal全部连接 |
| 冻结GeoSteiner缓存 | 3.708010122179985 |
| 原坐标重新调用GeoSteiner | 3.708010122179985 |
| 相对regret | -0.000035138996169298866 |
| 直接API返回码/solution status | 0 / optimal |
| 精确乘2^10、2^20或2^27求解后缩回 | 三者均为3.707879826426506 |

因此存在严格短于冻结参考值的合法解。差值0.000130295753479不是读取错位、NaN、缓存损坏或REST C求和误差。`optimal`状态本身也不足以证明此输入的原坐标结果正确。

八种单net诊断保存在`solver_status.json`。倒序、固定随机置换均复现原误差；改变EPS_MULT_FACTOR没有消除误差；单独启用LP_SCALE返回了更大的3.7907765582203865，不能作为修复。

只读legacy源码`DRO/DRO/WDRO-REST/algorithms/geosteiner/lp_solve_2.3/README.custom`第118–132行指出内置lp_solve数值不稳定及绝对容差；`lpkit.h`第98行的reduced-cost容差为1e-6。观测结果与尺度相关的数值问题一致，但尚未证明具体内部触发路径。不能把单net缩放通过当作全局求解正确性证明。

## 影响超出当前test格

预先限定的坐标级诊断覆盖20个validation格各首64nets（1280），以及五degree的seed101/update1训练batch各前128nets（640），共1920输入。没有加载这些输入的Actor或比较方法性能。

- 原坐标重新调用逐项复现全部抽查validation缓存。
- 2^27与2^28均为精确二进制乘法；全部输入整数化且反变换逐位一致，两种尺度的1920个长度逐项相同。
- 11个validation输入和1个训练输入的原reference严格大于缩放后的可行长度。
- 12个差异样本进一步保存完整树见证，以整数L1边长求和及全图连通性独立验明：全部terminal连接，长度严格小于原reference。证据为`feasible_tree_witnesses.json`。
- 本次总计额外5781次GeoSteiner调用和256个Actor/REST评价，独立于正式轨迹记账。

这只能证明问题也存在于训练和validation，不能估计总体错误率、证明未抽查输入正确，或断言所有已选checkpoint都会变化。训练损失、Hard Mining/WDRO真实候选分数、lambda与regret的相对尺度、validation best/nominal约束和阶段5选择均依赖该reference，需要按依赖关系复核。旧审计证明确定性重放和缓存一致性，不能替代最优性证据。

## 建议的修复范围（待授权，尚未执行）

1. 在Codex内新增隔离的参考求解实现版本，采用有数学依据的精确尺度处理；候选方案是在GeoSteiner内部将契约规定的float32归一化坐标精确乘2^27为整数，返回长度再除回。继续验证求解状态、几何可行性、解析小例及独立求解/最优性证据。不得修改数据、归一化规则或REST输入，不按test收益选择参数。
2. 对冻结固定数据及受影响训练reference做版本化纠错审计。保留全部v1文件，新cache身份同时绑定数据、solver binary/参数/实现版本；不能按单个方法的REST结果修补reference，也不能仅替换出现负regret的test样本。
3. 依据纠错差异定位受影响的最早步骤。必要时从同一DAC21初始化重新执行原seed/batch256/2000updates轨迹，并复核原validation-only选模与阶段5冻结选择。已有checkpoint不能被直接当作纠错后训练的恢复点；是否需重放某个run由其完整依赖证据决定。保持原实验范围，不新增超参搜索、不用test结果取舍配置。
4. 纠错审计通过后，在新版本索引中重新冻结完整75模型，再对全部共享test重新评价、统计和验收；保留旧版本作为可追溯历史。阶段7继续关闭。

第2–3项涉及阶段1 reference及阶段2–5训练/validation，超出用户本次“只处理阶段6”的边界。原`protocol_v1`、`data_contract_v1`、阶段5选择配置、cache、模型和操作合约没有改动。当前不能直接运行旧`launch_stage6.py`、放宽断言或截断负regret来完成阶段6。

## 证据与复现

主索引：`wdro_legacy_v1/wdro_rest_min/manifests/stage6_reference_blocker_v1.json`，包含脚本/产物SHA-256、逻辑/resolved路径、原PID/状态、test恢复游标和调用数。

诊断均在前台小规模CPU/GPU smoke完成，没有新增持久批跑。所有临时目录、runtime cache、日志位于Codex；`shared/experiments`稳定解析到`/attached/remote-home1/Guilin/DRO/Codex/shared/experiments`。

最近预检`host_runtime_20260907T031133Z.json`：GPU4/5满足当前容量门槛，Codex根可用166.075GiB，产物盘可用2827.541GiB。GPU2/3在前一个快照可用、随后负载升高；不把任何编号当作永久分配。未来任何启动/恢复仍需新预检。

在`/remote-home/Guilin/DRO/Codex`执行：

```bash
# 生成新快照，保存输出的manifest路径
bash wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python wdro_rest_min/scripts/capture_host_runtime.py --purpose stage6_reference_diagnosis --measured-peak-mib 1319 --safety-margin-mib 2048

# 单net直接求解器状态/尺度复现；只读复用已保存的失败batch
bash wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python wdro_rest_min/scripts/diagnose_stage6_geosteiner_status.py

# 检查已保存的几何见证、模型审计和逐batch恢复点，刷新阻塞索引
bash wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python wdro_rest_min/scripts/record_stage6_reference_blocker.py

# 当前监控入口
cat wdro_rest_min/manifests/stage6_reference_blocker_v1.json
```

完整坐标级抽查入口为`diagnose_stage6_reference_scope.py --preflight 新生成的manifest路径`；逐组结果原子保存，可复用已完成组。当前正式测试恢复命令被阶段门禁阻塞，待修复范围授权和新实现验收后另行冻结可执行恢复入口。

# 如何解释与 GroupDRO 等方法的差异

本次统一的是**评价条件**：同一套坐标、相同 Actor 架构、相同 greedy 解码与固定前向批大小、相同自身线长计算。已训练模型的初始化、学习目标和数据经历保留原样。

| 项目 | precision_v2 主线模型 | RL-only 新模型 |
|---|---|---|
| 初始化 | DAC21 预训练 Actor | 相同 seed 对应的新随机 Actor |
| 后续更新 | 2,000，batch 256 | 2,048，batch 64 |
| 后续 sampled 训练样本/模型 | 512,000 | 131,072 |
| 基础优化器学习率 | 5e-5 | 1e-4 |
| Actor/基线反馈 | 相对 regret：自身线长除以最优参考再减一 | 自身动作线长 |
| 数据尺度 | 每 net bbox 等比归一化至规定窗口 | 固定窗口，不逐 net bbox 缩放 |
| 训练分布覆盖 | Mixed ERM/GroupDRO 每批含四分布各 64 个样本 | 训练中心均来自 uniform；WDRO 分支在其附近生成扰动 |
| 验证选模 | 原冻结 relative-regret 规则，部分基线有 nominal 退化约束 | 四组最差平均自身线长最低的最早 checkpoint |

主线后续 sampled 样本数约为新模型的 3.91 倍，另有 DAC21 预训练历史；该比例不等同总计算量比例。Hard Mining、WDRO、self-critical 和学习对手的额外前向、参考评分、优化步骤不同。

## 五种主线方法分别做什么

- **ERM**：在 nominal uniform 样本上训练。
- **Mixed ERM**：每批均衡取四种分布，等权优化损失。
- **GroupDRO**：使用同样的四组采样，但提高高损失组的权重。实现中的更新是 `q_next = softmax(log(q) + eta * group_loss)`，随后按组权重组合 Actor 与 Critic 损失。
- **Hard Mining**：在共同候选支持集上选相对 regret 较大的输入，不使用 WDRO 的运输惩罚。
- **WDRO-REST**：在候选选择中加入 λ×运输成本，并更新 λ。

当前项目中的这些主线模型以参考 regret 训练，所以训练历史仍依赖 GeoSteiner。此轮只调用它们的冻结 Actor，并用自身输出拓扑计算长度，评价不需要最优参考。

## GroupDRO 是否天然必须依赖 GeoSteiner

不必须。GroupDRO 的组权重更新只需要可观测的每组损失。数学上可以将组损失定义为当前 Actor 自身动作长度的样本平均，再以相同的组权重组合强化学习更新；这样不需要最优值、MST 或几何界。

但目标会变成“各组自身线长风险的最坏值”，与“各组相对 regret 风险的最坏值”不同。原始线长会混入各组固有几何难度，不能把两者视为无差别替换。这样的 **RL-only GroupDRO 是新的训练配置**；本次对照使用项目已有的 GroupDRO 权重，尚未训练和验证该新配置。

因此，本轮可以判断现有模型谁在相同输入上给出更短的解；若要判断“去掉 GeoSteiner 本身造成了多少误差或训练开销”，需要另做同起点、同数据暴露、同预算、同选模规则的训练消融。

实现依据：主线 `wdro_rest_min/src/baseline_data.py`、`baseline_trainer.py` 与冻结 trainer/protocol；新方案 `rl_only_20260915/train.py`、`protocol.json`。本文件为机制与比较范围说明，不改变任何冻结训练或评价配置。

# WDRO-REST 论文同步清单（protocol v1）

本清单对应冻结配置 `wdro_legacy_v1/wdro_rest_min/configs/protocol_v1.json`。阶段 9 修改论文时逐项核对，不能在没有冻结结果支撑时保留原草稿主张。

## 必须删除或改写

- 删除“优于 canonicalization / canonicalized ERM”的对比主张；归一化是五种方法共享的标准预处理。
- 删除 KL-DRO、CVaR 主基线及对应优越性表述。
- controlled shifts 从五类改为 mix_gauss、edge_bias、grid_cluster；删除 anisotropic 和 void 的必做表述。
- 主表固定为 ERM、Mixed ERM、Hard Mining、GroupDRO、WDRO-REST 五种方法。
- 删除 GW/FGW、T=8、FLUTE 主对比，以及未执行的多档 K、log-sum-exp/raw-length 消融承诺。
- CEM 仅保留为 degree=30 的 Random-64/CEM-64 等调用预算消融，以及所有冻结模型共享的 CEM-256 独立攻击。
- 将 design-level train/validation/blind 协议 B 改为协议 A：所有训练与调参只用合成数据，ISPD 只在配置冻结后作外部测试。
- 不使用“researcher-blind”表述；允许使用“模型对 ISPD 零样本迁移”，并说明仓库已有旧 merged REST 结果。
- 若最终投稿 ICLR，迁移模板并重新核对结构、篇幅、匿名性和截止日期；若仍投 ICCAD，保留 ICCAD 模板但同步实验范围。

## 结果主张门槛

- 只有相对 ERM、Mixed ERM、Hard Mining 的目标分布 P95 或 worst-group 在多个 seed 上方向一致且置信区间支持，才写稳定改善。
- nominal mean relative regret 相对 ERM 的绝对恶化必须不超过 0.002；否则不能作主要性能主张。
- 独立 CEM-256 攻击下优势仍存在，才可写抗强攻击结论。
- 预算活跃时，lambda、mean selected cost、rho 和 budget gap 必须支持运输预算解释。
- 若只改善尾部，只写尾部鲁棒性；若 Mixed ERM/GroupDRO 相当，只强调显式几何预算与诊断能力。
- ISPD 若与合成结果矛盾，明确限制适用范围；不得返回 validation 阶段以 ISPD 或 test 调参。
- 不删除不利 degree、distribution 或 seed，不把不显著结果写成显著。

## 表、图与追溯检查

- 主表：5 methods × nominal/3 OOD/ISPD；mean、P95、worst-group、95% CI。
- 附表：distribution × degree × seed 的完整长表与 paired per-net bootstrap CI。
- ISPD：net-weighted mean、design-macro mean、P95、worst-design mean、逐 design/degree 样本数与无效数。
- 诊断图：lambda、mean selected cost、rho、budget gap。
- 效率：evaluator/GeoSteiner calls、训练 wall-clock、推理时间、峰值显存。
- 正文、表格、图、CSV/JSON 的每个数字均追溯到 run manifest、模型 hash、数据 hash 与 `L_star` cache hash。
- 删除所有无法由最终产物追溯的 TBD 与手工数字。


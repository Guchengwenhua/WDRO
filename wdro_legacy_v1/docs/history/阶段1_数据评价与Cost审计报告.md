# 阶段 1：数据、评价与 Cost 审计报告

完成时间：2026-09-04 08:17 UTC。

## 结论

阶段 1 已完成，端到端审计状态为 `pass`。共享 normalization/inverse、合法性投影、稳定数据 ID/hash、四类合成数据、GeoSteiner `L_star` cache、relative-regret 指标、paired bootstrap、Hungarian-L1、symmetric Chamfer-L1、调用计数和 ISPD 协议 A 整理均已实现。

共冻结 40 个合成数据格、240,000 个 nets 和 40 个一一对应的 `L_star` cache。validation/test 的 net ID 交集为 0，无序 pin 内容交集为 0。ISPD 只整理输入、来源 ID 和无效计数，未计算或读取任何模型指标。

## 共享数据契约

- 数据契约：`wdro_legacy_v1/wdro_rest_min/configs/data_contract_v1.json`，权限 `0444`。
- SHA-256：`2121d1119bc20dcd493183fcab4ce8aaa986eee84523e487fb42c2cc29612055`。
- 计算使用 float64，NPZ pin 存储使用 float32；落盘后再次审计合法性。
- bbox 平移到原点，按最长边统一缩放并在两轴加 0.1；最长边跨度为 0.8，保持纵横比和 pin 顺序。
- inverse 保存每个 net 的 `offset_xy` 与 `scale`。
- 精确重复 pin 稳定去重；固定 degree 因此改变时判无效。双轴全零跨度无效，单轴零跨度有效。
- 归一化域最小 L1 pin 间距为 `1e-6`；NaN/Inf 无效；候选投影采用确定性裁剪与间距修复。
- pin-set hash 对输入顺序不变；dataset hash 同时覆盖数组顺序、dtype、shape 和 net IDs。

实现位于：

- `wdro_legacy_v1/wdro_rest_min/src/geometry.py`
- `wdro_legacy_v1/wdro_rest_min/src/datasets.py`
- `wdro_legacy_v1/wdro_rest_min/src/costs.py`
- `wdro_legacy_v1/wdro_rest_min/src/geosteiner.py`
- `wdro_legacy_v1/wdro_rest_min/src/metrics.py`
- `wdro_legacy_v1/wdro_rest_min/src/accounting.py`
- `wdro_legacy_v1/wdro_rest_min/src/ispd.py`

## 合成 validation/test 冻结结果

- validation：4 distributions × 5 degrees × 2,000 = 40,000 nets，base seed 202。
- test：4 distributions × 5 degrees × 10,000 = 200,000 nets，base seed 303。
- 总计：40 格、240,000 nets。
- 每个 split/distribution/degree 使用 SHA-256 namespace 派生独立 RNG seed。
- 生成时共拒绝并重采样 1 个样本：test edge_bias、degree 40、`minimum_spacing_violation`；最终冻结数据全部合法。
- 数据 manifest：`shared/experiments/cache/datasets/protocol_v1/datasets_manifest.json`。
- manifest SHA-256：`c4cc067a7c809c0e490d723d7bc1feee62f73e699001b81f798605741015502c`。
- 数据目录约 53 MiB，所有 NPZ 与 metadata 均为只读。

审计逐格验证了文件 SHA-256、数组 bundle SHA-256、shape、count、ID 唯一性、坐标范围、最长边跨度、最小间距和 inverse→normalize 抽查，并从冻结 seed/参数重新生成全部 40 格，内容哈希全部复现。validation/test 的 240,000 个 ID 全部不同；无序 pin fingerprint 也无交集。

## GeoSteiner L_star cache

- cache 数：40，与合成数据格一一对应。
- 精确 evaluator calls：240,000。
- cache key：dataset `array_bundle_sha256`，不依赖文件名。
- 每个 cache 同时保存 `net_ids`，读取时强制与 dataset 对齐。
- 所有 `L_star` 均为有限正数。
- 审计独立重算每格首尾两个 net，共 80 次，全部在 `atol=rtol=1e-12` 下相等。
- 8 个 CPU worker 的逐格 wall time 合计约 266.98 秒。
- index：`shared/experiments/cache/lstar/protocol_v1/lstar_index.json`。
- index SHA-256：`3b8618996e9de158eaaf9cfedf3c7a42fabeb4d2efaac1d2c36b4301207c5257`。

## 指标与 cost

指标模块实现逐 net `(L_REST-L_star)/L_star`、mean、P95、worst-group group mean，以及确定性 paired bootstrap mean-difference 95% CI。所有零/负 `L_star`、NaN/Inf 和 shape 不一致输入都会拒绝。

主 cost 是 equal-degree unordered pin set 上的 Hungarian matching mean L1；消融 cost 是两个方向最近邻 mean 的平均，即 symmetric Chamfer-L1。规定的七项 cost 审计全部通过：

1. `c(X,X)=0`；
2. pin permutation invariance；
3. 小平移有精确预期值；
4. 扰动幅度单调；
5. 不同结构不坍缩；
6. mean 定义在重复结构测试中跨 degree 尺度一致；
7. 重复 pin 可有限计算，NaN 和 degree mismatch 被拒绝。

调用账本统一按“被评价的 net 数”计数，而不是 Python batch 函数调用次数，字段覆盖 Actor、surrogate、精确 REST evaluator 和 GeoSteiner。

## ISPD 协议 A 整理

- 基础 design：20 个；design/net ID 全部保留。
- 指定 degrees 的有效输入：22,498 nets；无效 0。
- design×degree 条目：100，其中 95 个源 degree 文件存在并生成只读 NPZ，5 个源文件本身不存在并明确记录为 missing。
- 排除 `_metal5`：6 个。它们与基础版本在指定 degree 上的 net ID 序列完全一致，确认是同一 design identity；坐标文件不是逐字节一致，因此保留 canonical 基础版本而不是把两者混入统计。
- model metrics 始终标记为 locked，产物不含预测、regret 或选模信息。
- manifest：`shared/experiments/cache/ispd_protocol_a/ispd_protocol_a_manifest.json`。
- manifest SHA-256：`35a257e5e6fbaea45fbf42295d385109c11b8988119b23a1a1bd3aedf940c4e9`。

## 验证证据

- 单元测试：37/37 通过。
- 端到端 audit：`shared/experiments/logs/stage1_audit.json`，SHA-256 `8a8ac6cbc37558f248c6bac26c965d98cbee1bc88deaa8b23afa466b1d41f13f`。
- 阶段 1 代码/环境 manifest：`wdro_legacy_v1/wdro_rest_min/manifests/preflight_20260904T081746Z.json`，SHA-256 `e7f488eb107f22e70819f0992445df1621e6381c1346e0988ce036731bab2e6d`。
- manifest 中新代码 combined SHA-256：`81d7dbb8cfa7d745c42534c32e1878ca2a3560e955e45b8c37c0e30c85d9add5`。
- audit 明确记录：40 格全齐并可确定性重建、240,000 nets、0 split leakage、40 个 cache、240,000 calls、80 次精确复算、20 个 ISPD designs、95 个 ISPD 输入文件。

复现命令见 `wdro_legacy_v1/wdro_rest_min/README.md`。核心验证命令为：

```bash
/remote-home/Guilin/miniconda3/envs/py311/bin/python -m unittest discover -s Codex/wdro_legacy_v1/wdro_rest_min/tests -v
/remote-home/Guilin/miniconda3/envs/py311/bin/python Codex/wdro_legacy_v1/wdro_rest_min/scripts/verify_stage1.py
```

## 剩余阻塞与下一步

08:17 UTC preflight 显示工作盘可用空间降至 15.143 GiB，仍低于 100 GiB 批跑门槛；`development_ready=true`，`batch_ready=false`。本阶段新增实验文件约 60.7 MB，不解释文件系统其余空间变化。

下一步为阶段 2：实现统一训练器，先运行 degree 30、seed 101、200 updates 的 ERM 训练/恢复/评价闭环。该 smoke 可在当前空间下开发，但扩展到五个 degree × 三个 seed 的正式 checkpoint 批跑前必须解决可写空间门禁。

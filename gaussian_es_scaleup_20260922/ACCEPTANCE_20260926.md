# Gaussian-ES 扩规模实验验收（2026-09-26）

**工程与数值验收通过；预注册的 ES64 稳定训练收益未建立。** 实验已于 2026-09-25 14:12:59 UTC 完成，97/97 项任务结束，从首个任务启动至正式完成耗时 80.56 小时。本次验收重新核对原始产物、独立计算统计量，并在服务器 GPU 上复算 checkpoint 输出，不以原自动报告作为验收依据。

主比较 degree30 中，ES64 的 OOD P95 相对 Random64 下降 **0.706%**，5 个种子中 4 个改善，但未达到预设的 ≥1% 降幅，单侧精确种子符号翻转检验 **p=0.1875**，也未达到 0.05。结论是当前协议下证据不足，不能据此断言两种方法等效，或 ES 不可能有效。

## 验收范围与协议执行

本次验收对象为 [gaussian_es_scaleup_20260922](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922)。原 [protocol.json](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/protocol.json)、[理论与实验方案](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/docs/理论与实验方案.md)、训练代码、checkpoint、测试结果及历史交接文件均未修改。新文件仅写入本项目下的 `artifacts/review_20260926` 和本报告。

| 阶段 | 完成情况 | 实际规模 |
|---|---:|---|
| 扩展预训练 | 12/12 | degree30 的 5 个正式种子及独立 seed901：8192 updates、batch128；degree20/40 各 3 种子：4096 updates、batch128 |
| 外层调参 | 4/4 | 仅 seed901；1024 updates、batch64；学习率和 nominal fraction 的 2×2 组合 |
| 固定模型搜索与预算验证 | 15/15 | 9 个 ES 参数组合、Random/CEM 对照、独立预算集上的 ES/Random K64/K256 |
| 正式后训练 | 33/33 | ERM、Random64、ES64 配对；degree30 每种方法 5 种子×8192 updates，其余各 3 种子×4096；batch64 |
| 统一测试 | 33/33 | 每模型 4 分布×4096 常规测试点，另有 4 分布×128 独立 CEM64 压力中心 |

最终由 validation 选中 `mu=0.01`、`search_step=0.01`、`lr=1e-5`、`nominal_fraction=0.25`。选择过程已重新计算并与冻结清单一致。正式模型在统一测试开始前冻结；同 degree/seed 的三种后训练方法使用相同父 checkpoint、源中心序列、更新数和学习率。seed901 不进入正式五种子统计。

所有 49 个训练任务合计执行 **274,432 次更新**，对应 **22,282,240 次训练样本使用**；配对方法会重复使用同一源中心，这个数不能解释为独立、不重复的 net 数量。累计训练计时约 **559.60 worker-hours**，包含参考求解及共享资源等待，不能当作纯 GPU 计算时长。

## 独立证据核对

| 检查 | 结果及覆盖范围 |
|---|---|
| 来源与文件完整性 | 71 个冻结来源文件；5,137 项唯一文件绑定重新 SHA-256 核对，GPU 复算后再次确认全部未变 |
| 任务生命周期 | 97 个完成回执与任务、PID/start tick、exit、heartbeat 一致；旧进程身份均已退出；98 把任务/控制器锁均可释放；正式任务无 failure 回执 |
| checkpoint 与续训 | 49 组 latest/best 的步数、身份、数值和状态通过；历史连续 6 步与中断恢复至 6 步的 Actor、Critic、Adam、lambda、RNG 逐项相等 |
| 更新与调用预算 | 274,432 条更新记录无漏步/重步；逐步核对真实评分、采样更新、参考调用及 lambda 更新方程；请求/完成账本一致 |
| 配对与训练输入 | 所有后训练配对中心 hash 序列一致；另独立重建每训练任务首、中、末批，共 147 批 |
| 搜索合法性与选优 | 核对保存的 1,108 个训练审计中心及其完整候选 support；合法性、Hungarian-L1 成本及原始 penalized score argmax 通过 |
| 全部最终测试产物 | 4,752 个 NPZ 分片、557,568 条输出的完整性、模型绑定、坐标合法性、regret 恒等式与配对通过；其中普通测试 540,672 条、压力测试 16,896 条 |
| 汇总与隔离 | 从原始数组重算全部 264 行指标、801 条 validation 曲线；每 degree 的 6 个 held-out split 全量重建，45 组跨 split 比较无交集，split 内无重复 |
| GPU 重放 | 49 个 best checkpoint 的全部 validation：50,176 次预测；33 个正式模型另抽查 1,056 个测试输出，共 51,232 次预测，最大指标/线长误差均为 0 |
| 参考数值 | 重放参考值重新求解；1,056 个抽查测试点另用 GeoSteiner 2^27/2^28 双尺度复核，两尺度结果一致 |
| 统计与测试 | 独立重算全部 32 个种子符号组合、2000 次交叉 bootstrap，与原始统计一致；冻结的 5 项单元测试重跑通过 |

GPU 重放使用新鲜资源准入选出的 GPU7（RTX 6000 Ada），约 54 秒，峰值 reserved 显存 212 MiB；仅推理和参考复算，无新训练。验收程序复用冻结的数据生成、几何和模型基础实现，但没有调用原最终报告、模型选择或统计函数。独立性是验收逻辑和统计计算的独立实现，不代表另有独立人员审稿或完全独立的几何求解器。

## 预注册主指标

指标为 `relative_regret = L_REST/L_star - 1`，越低越好。先对每个种子、每个 OOD 分布计算 P95，再对三个 OOD 分布和种子平均；不是把所有测试点混合后的 pooled P95。

| degree30 主判据 | 实测 | 门槛 | 判定 |
|---|---:|---:|---|
| ES64 相对 Random64 的 OOD P95 降幅 | 0.706% | ≥1% | 未达到 |
| 单侧精确种子符号翻转 p | 0.1875 | ≤0.05 | 未达到 |
| 改善种子 | 4/5 | ≥4/5 | 通过 |
| uniform mean：ES−ERM | +0.0000391133 | ≤0.002 | 通过 |
| 独立 CEM64 四分布 mean：ES−Random | −0.0003458961 | ≤0 | 通过 |

Random64 与 ES64 的 OOD P95 分别为 **2.556474%**、**2.538421%**，差值为 **−0.018054 个百分点**。描述性交叉 bootstrap 95% 区间为 **[−0.072908, +0.033919] 个百分点**，跨过 0；显著性判定依据是预注册的种子检验，不将数万个 net 当作独立训练重复。

## 次要结果与搜索结果

普通 OOD 测试如下；正的相对降幅有利于 ES，负值表示变差。degree20/40 各只有 3 个种子，以下是描述性结果，不新增事后显著性结论。

| degree | ERM OOD P95 | Random64 OOD P95 | ES64 OOD P95 | ES 相对 Random 降幅 | 改善种子 |
|---|---:|---:|---:|---:|---:|
| 20 | 2.049627% | 2.045997% | 2.038842% | +0.350% | 2/3 |
| 30 | 2.579687% | 2.556474% | 2.538421% | +0.706% | 4/5 |
| 40 | 3.075852% | 3.039509% | 3.045210% | -0.188% | 1/3 |

独立 CEM64 压力测试的三个 OOD 分布平均 P95（与主判据的四分布 mean 不同）：

| degree | Random64 训练模型 | ES64 训练模型 | ES 相对 Random 降幅 |
|---|---:|---:|---:|
| 20 | 5.696385% | 5.920441% | -3.933% |
| 30 | 6.456393% | 6.350629% | +1.638% |
| 40 | 6.796423% | 6.478225% | +4.682% |

degree20 在压力测试中反而变差，degree40 普通 OOD P95 略差，因此不能将 degree30 的方向性改善推广成跨规模稳定优势。

固定模型、独立 budget-validation 集上的搜索结果：

| 每中心真实评分预算 | Random 平均搜索 gain | ES 平均搜索 gain | ES 的 gain 相对增加 | Random / ES 实测秒数 |
|---|---:|---:|---:|---:|
| 64 | 0.02279253 | 0.02941530 | 29.06% | 45.61 / 58.45 |
| 256 | 0.03279728 | 0.05185272 | 58.10% | 195.44 / 220.27 |

此处 gain 是 `max_Z[regret(Z)−lambda*c(X,Z)]−f(X)`。它度量攻击器找到更高原始目标值的能力，来自一个冻结 degree30 父模型，不能当作训练后 regret 降低 29% 或 58%，也不证明所有模型上都优于 CEM。

## 影响解释的训练诊断

**增加更新数不等于最终模型使用了训练末轮。** 12 个预训练任务只有 1 个选中最后一步，实际下游初始化步数如下（种子→best step）：

| degree | 完成更新数 | 选中的父模型 |
|---|---:|---|
| 20 | 4096 | 101→256；102→2816；103→2048 |
| 30 | 8192 | 101→1792；102→1792；103→8192；104→2560；105→7424；901→3840 |
| 40 | 4096 | 101→256；102→0；103→256 |

后训练也只有 5/33 个选中末步。这符合冻结的 validation 选模规则，但当前实验没有独立的预训练时长因果对照，不能宣称延长预训练本身带来收益。

**运输预算在多数更新中未形成持续的正对偶惩罚。** 下表平均 cost/rho 由每个种子的全程平均成本计算，lambda=0 比例给出种子间范围：

| degree | 方法 | 平均 cost/rho | 更新后 lambda=0 的步数占比 |
|---|---|---:|---:|
| 20 | random | 0.870 | 91.28%–92.26% |
| 20 | es | 0.953 | 85.82%–86.23% |
| 30 | random | 0.817 | 98.07%–98.30% |
| 30 | es | 0.938 | 94.62%–94.81% |
| 40 | random | 0.803 | 98.49%–98.88% |
| 40 | es | 0.958 | 83.25%–84.47% |

逐步更新方程已核对无误，lambda=0 本身不是实现错误。当 lambda 为 0 时，内层目标等于原始 regret，因此大量训练步骤更接近无运输惩罚的难例挖掘；这些记录不足以证明正对偶惩罚在训练中持续发挥作用。

**真实调用数配平不等于 GPU 工作量配平。** ES64 与 Random64 每个 batch64 都进行 4096 次真实候选评分，但当前 ES 分块补齐为 5632 个 actor forward rows，Random 为 4096。全部任务账本中真实评分 556,762,112 次、forward rows 664,331,264 次，padding 107,569,152 次。33 个正式后训练任务的累计计时：ERM 8.10、Random 242.01、ES 278.62 worker-hours；ES 比 Random 高约 15.1%。共享服务器上的计时含争用和 CPU 求解，不能作为隔离硬件基准。

## 结论边界与归档

可接受的结论是：扩规模训练、调参、选模和统一评价按冻结协议完成，数值与产物可复核；高斯平滑在本次固定模型验证中提升了搜索 gain，但 ES64 相对同真实调用预算 Random64 的预注册训练收益未建立。

未覆盖的内容必须保留：没有逐一重建全部训练 net 并做完整 train-test 内容交集检查；已验证独立命名空间、全部 held-out split 隔离及 147 批训练重建。保存的候选 support 只覆盖 1,108 个中心，并非全部约 5.6 亿次候选评分。全部测试数组已做代数与几何核查，GPU 上重跑的是全部 best validation 和抽样测试，未重新求解全部最终测试。有限候选、有限分布和有限种子不能给出连续 Wasserstein 最坏情形证书，也不能给出普遍优越性结论。

后续研究应先检查早期选模、validation 方差和 rho/惩罚的实际作用，再设计预训练时长与等计算成本对照。任何进一步调参需另立协议并使用新的最终测试集；本次验收不修改现有选模、不追调已有测试、不追加训练。

验收过程中初版脚本误将 step0 checkpoint 的空 Adam state 判为错误；已更正为只要求 step>0 时存在已初始化的 Adam state，完整重跑通过。原异常保留在 `audit_failure.json`。它与历史磁盘迁移时人工中断产生的 `artifacts/tests/gpu_failure.json` 均不属于正式训练任务失败。`audit.json` 保留 CPU 阶段的 `PENDING_GPU_REPLAY` 名称，最终联合判定以本次 `acceptance.json` 为准。

产物入口：

- [联合验收清单 acceptance.json](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/acceptance.json)：最终工程/科学状态与本次审计文件 SHA-256。
- [CPU 产物审计 audit.json](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/audit.json)、[GPU 重放 replay.json](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/replay.json)、[最终复核 final_recheck.json](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/final_recheck.json)。
- [独立统计 statistics_recomputed.json](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/statistics_recomputed.json)、[逐模型分布指标 metrics_recomputed.csv](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/metrics_recomputed.csv)、[配对比较 comparisons.csv](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/comparisons.csv)。
- [训练诊断 training_audit.csv](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/training_audit.csv)、[补充诊断 diagnostics.json](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/diagnostics.json)、[搜索记录 search_cells.csv](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/search_cells.csv)。
- [CPU 验收代码 audit.py](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/audit.py)、[GPU 重放代码 replay.py](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/replay.py)、[单元测试输出 unit.log](/remote-home/Guilin/DRO/Codex/gaussian_es_scaleup_20260922/artifacts/review_20260926/unit.log)。

逻辑入口全部位于 `DRO/Codex`。`artifacts` 沿用原实验的 NVMe 存储链接，实际位置为 `/attached/remote-home1/Guilin/DRO/Codex/experiments/gaussian_es_scaleup_20260922`。原任务与本次 GPU 复算均已退出，不需要继续挂 tmux 或等待 GPU 训练。

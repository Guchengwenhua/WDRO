# 2026-09-20 平级方法重构

## 当前结构

10 个方法/版本/计划目录已实际放在 `Codex` 根目录，名单见 [全目录索引](DIRECTORY_INDEX.md)。独立方法的理论、进展、结果和运行说明集中在各自 `docs/` 中；模型、配置和测试保持在所属方法内。3 个跨方法测评工程统一位于 `benchmarks/`，跨方法汇总位于 `summaries/`，原始依赖及附加盘入口位于 `shared/`。

## 路径与冻结证据

已同步修改当前维护的 Python/Shell 路径、祖先目录推导、文档输出位置及阅读链接。完整 [路径映射](../archive/restructure_20260920/path_map.json) 与 [迁移日志](../archive/restructure_20260920/moves.json) 可核对。

按用户选择，保留 `revisions`、`improvements`、`wdro_rest_min`、`experiments`、`external_benchmarks`、`docs`、`project_tools`、`mainline`、`external` 和 6 个直接绑定历史文件名的兼容入口。其余根目录历史文档已分流，原字节可从迁移归档读取。

兼容源码读取迁移前快照，当前代码读取平级目录；两者不能混为同一源码版本。1,148 份源码、配置和文档的原始 SHA-256 独立保存。冻结协议 JSON、既有模型、数据、统计表及历史验收清单没有批量改写。历史清单的 resolved path 随搬迁可能变化，由迁移映射解释；不伪造旧哈希或重新签署科学结论。严格续跑仍需通过原有源码/路径身份检查，本次没有绕过绑定检查。

查询路径：

```bash
python3 /remote-home/Guilin/DRO/Codex/tools/resolve_path.py revisions/reference_precision_v2/Codex/wdro_rest_min
```

## 重复内容处理

| 内容 | 核对结果 | 处理 |
|---|---|---|
| precision_v2 下的 `DRO` | 原本就是指向原始 `DRO/DRO` 的链接 | 无重复数据可删；归入共享依赖 |
| v3 与 v3.1 的 WDRO-REST 依赖副本 | 各 583 个非缓存文件，全部与原始依赖逐文件 SHA-256 一致 | 删除 1,166 个重复文件，共 24,598,182 字节，引用原始依赖 |
| 2026-09-04 本地实验缓存备份 | 356 个文件，全部与附加盘对应文件逐文件 SHA-256 一致 | 去重为共享文件链接，60,742,740 字节；不再是独立灾备副本 |

合计去掉重复文件内容 85,340,922 字节（约 **81.39 MiB**）。这不是重构后的净磁盘差：迁移清单、原始源码快照和测试记录也占用少量空间。原始 `DRO` 工程保持原位，未删除原始模型、benchmark、论文或不同方法的独有实现。当前目录中的多数大文件属于真实实验产物，不能因为名称相似而合并。

证据：[依赖副本逐文件清单](../archive/restructure_20260920/duplicates_removed.json)、[缓存备份去重清单](../archive/restructure_20260920/storage_backup_deduplication.json)。

## 验证范围

- 迁移前主线 87 项 CPU 测试通过；迁移后 13 组回归检查通过，合计 345 项 unittest、3 项搜索函数检查及 1 个最终发布流程检查。
- 回归检查覆盖 v1/precision_v2、v3.1、Critic、Oracle-free、RL-only、RL 六方向、论文迁移和两类外部比较。GPU 训练与完整科学验收没有重跑。
- 全部维护代码的语法、项目阅读链接、冻结源码原字节与迁移文件身份由 [验证脚本](../tools/verify_layout_20260920.py) 检查。最终结果及计数见 [verification.json](../archive/restructure_20260920/verification.json)。
- 检查发现原外部测评测试只隔离了 `reference.BASE`，未隔离 `common.BASE`，导致已有单样本测试夹具 `d10_0_1/job.json` 的瞬时任务标记更新。已修复测试隔离并单独复测。该 117 字节运行标记的原字节未留存；[变更记录](../archive/restructure_20260920/test_fixture_changes.json) 保留迁移前元数据及迁移后哈希。正式模型、参考数组和统计结果未受影响。

验证命令：

```bash
python3 /remote-home/Guilin/DRO/Codex/tools/verify_layout_20260920.py
```

原始源码和恢复材料见 [迁移归档说明](../archive/restructure_20260920/README.md)。

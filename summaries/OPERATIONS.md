# 环境、运行、恢复与资料维护

本文是 2026-09-15 新整理的运行说明。历史引用的 `summaries/history/02_新对话执行提示词.md` 原文件此前缺失；该旧入口现在是一份新写的根目录导航短文，其中提供本文链接，不代表恢复了已丢失原文。

## 环境和入口

主项目的 PyTorch 环境为 `/remote-home/Guilin/miniconda3/envs/py311/bin/python`。GPU 训练、推理和旧审计使用对应版本的 `wdro_legacy_v1/wdro_rest_min/scripts/stage3_env.sh` 包装环境；系统 `python3` 仅适合本次纯标准库资料整理脚本。

当前主线位于 [precision_v2](../wdro_precision_v2/)，三路线研究位于 [v3.1](../adversarial_v3_1_20260912/)，本次研究位于 [critic_search_20260915](../critic_search_20260915/)。执行前先确定版本，勿从旧 v1 的同名脚本误启动新工作。

## 长任务检查和恢复

1. 先读该分支的状态、完成凭证和具体失败日志。DONE 需要真实产物、验证和退出记录，不能由 smoke 或旧状态文字代替。
2. 刷新 `nvidia-smi`、CPU、磁盘和进程状态。历史文档中的 GPU 空闲、显存和耗时仅是当时快照。
3. 对照 tmux、PID、进程启动时间、命令、心跳、锁和 checkpoint。PID 存在本身不足以识别原任务。
4. 已完成且验收通过的任务不重复启动。恢复使用同一协议、配置、数据流和预算；完整保存失败、重试及不确定调用。
5. 输出先写 `.partial`，验证后原子发布；对应完成索引绑定数据、代码、模型和结果。

原始分支运行细节仍在 [v3.1 RUNBOOK](../adversarial_v3_1_20260912/docs/RUNBOOK.md) 和 [各阶段原文归档](../archive/documentation_20260915/manifest.json)。新研究的调度、资源配置、任务清单和恢复命令由新研究目录维护。

## 存储与分类入口

`Codex/shared/experiments` 指向既有附加盘存储；各实验使用自己的子目录。`mainline/`、`improvements/`、`external/` 与 `archive/` 主要提供分类导航，不复制已有大数据。

实验源码和模型留在既有位置，避免改变代码通过 `__file__` 寻找依赖的行为。已绑定最终报告和根历史进度保留原位。移动的历史根文档通过原路径链接读取，内容 hash 不变；记录中的 resolved path 若来自旧盘点，属于历史元数据，应按 [整理映射](../archive/documentation_20260915/manifest.json) 核对。

## 文档维护规则

- 当前总状态在 [STATUS.md](STATUS.md)，每个新分支维护自己的理论方案、配置、运行状态和最终结果。
- 每个阶段一个当前汇总；中间恢复细节写入日志或归档，不再不断增加根目录的阶段快照文件。
- 完成报告被 manifest 绑定后保存原字节。新解释写独立修订和来源链接，不直接改写历史证据。
- 研究结论保留分布、degree、seed、真实评分预算、统计口径与保证边界。

## 验证和恢复本次整理

文档完整性及新导航检查：

```bash
python3 /remote-home/Guilin/DRO/Codex/tools/organize_docs_20260915.py --verify
```

恢复单份归档原文时先使用 `tar -xOf` 查看，避免覆盖活动文件。例如：

```bash
tar -xOf /remote-home/Guilin/DRO/Codex/archive/documentation_20260915/original_documents.tar.gz '阶段6_最终交接复核_20260909.md'
```

若需要全部解压以比较，在 `DRO/Codex/archive/` 下新建空目录，向该空目录解压；压缩包内的路径是相对 Codex 的原路径。每份 SHA-256、原大小、原权限、动作和归档目标均有清单，足以恢复移动前的原文。整理验证不等同于重跑旧模型或完整科学数值审计。

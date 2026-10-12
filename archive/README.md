# 历史资料与可恢复归档

| 入口 | 内容 |
|---|---|
| [documentation_20260915](documentation_20260915/) | 57 份整理前项目文档的原文压缩包、逐份 hash 和实际移动记录 |
| [adversarial_v3_superseded](adversarial_v3_superseded/) | 早期 v3 失败分支，保留 greedy 批形状问题与原始证据 |
| [storage_migration_backup_20260904](storage_migration_backup_20260904/) | 既有存储迁移备份；本次没有删除或改写 |

压缩包 [original_documents.tar.gz](documentation_20260915/original_documents.tar.gz) 保留原始相对路径和文件内容。[manifest.json](documentation_20260915/manifest.json) 记录 SHA-256、大小、权限、分类和移动目标；[verification.json](documentation_20260915/verification.json) 记录对原路径和压缩包的逐份复核。

根目录中的历史阶段/提案文件已移入 `documentation_20260915/originals/`，旧路径以兼容符号链接提供原文。为保持历史审计绑定，根长进度和 revision 最终报告原位保留。原文中的运行状态、权限要求和时间估算是当时记录；当前状态见 [docs/STATUS.md](../docs/STATUS.md)。

当前阅读版已合并，详见 [整理说明](../docs/ORGANIZATION.md)。恢复示例见 [运行与维护](../docs/OPERATIONS.md)。原始数据、checkpoint、日志和失败证据没有被当作冗余删除。

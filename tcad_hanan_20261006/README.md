# TCAD Hanan：服务器实验准备

本专题承接 `/remote-home/Guilin/TCAD - Hanan` 的后续实验。遵循该项目 `SKILL.md`：代码、配置、数据入口、日志和报告统一从 `/remote-home/Guilin/DRO/Codex` 定位。

- [通俗说明与执行设计](docs/服务器实验方案_通俗说明与执行设计.md)
- [数据入口](data/)
- [复制来源与数量](metadata/data_copy_receipt.json)
- [本次数据复制脚本](scripts/stage_data.py)

原始数据实际存储位置为 `/attached/remote-home1/Guilin/DRO/Codex/tcad_hanan_20261006/data`。在论文项目的 `04_程序与证据/服务器实验_20261006` 也可进入本专题。

当前仅完成方案和输入副本准备。历史笔记本工作作为理论和原型材料；本专题正式科研实验尚未开始。数据副本不等于已具备驱动端完整的实验输入，合法网数量要在后续数据提取时得到。

`data/raw/` 保存原始材料；`data/legacy_coordinate_inputs/` 是参考坐标文本；`data/source_records/original/` 保留上游来源文件；`data/source_records/relocated_circuitops_descriptors/` 将 CircuitOps 路径转为本次副本入口。这些旧 parser 描述不包含本论文所需的完整源点处理，不能直接当作执行配置。

复制脚本是本次一次性准备操作，已有目标文件时拒绝覆盖；无需为阅读或使用数据再次运行。未执行逐文件哈希或附加工程验证。

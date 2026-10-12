# Critic / ICCAD checkpoint-bank 扩展训练

2026-10-11 接续[原21项短程实验](../critic_paper_bank_20261011/acceptance/REPORT.md)。用户授权“扩展训练规模，继续实验”。本目录是新的探索性扩展，原短程产物保留原义。

入口：[冻结方案](PROTOCOL.md)、[主线参数](main/protocol.json)、[修正Critic参数](repaired/protocol.json)、[实时结果](RESULTS.md)、[运行交接](HANDOFF.md)。

2026-10-12：[扩展实验验收报告](acceptance_20261012/REPORT.md)。30/30项完成（27项训练+3项校准），无失败或缺失。修正Critic的final参考预测全部低于HPWL，真实误差三个种子均差于Geo对照；23ICCAD K3的best成本平均降低1.91%、25ICCAD降低0.61%，但种子不一致、耗时分别1.99/2.64倍。报告含逐样本统计、短长预算对照和Critic轨迹图。

| 组别 | 任务数 | 起点与预算 |
|---|---:|---|
| main | 21 | 原完整latest断点，REST从128续到2048，Timing/Buffer从256续到2048，batch128、种子101/102/103 |
| calibration | 3 | 同种子旧Geo update128 Actor固定，每个种子4096训练+1024训练侧holdout，4次动作采样/网；拟合新Critic 256步 |
| repaired | 最多6 | 每种子相同Actor+校准Critic，Geo/critic_inverse配对，各新增2048步；未通过校准的种子明确gated |

原21项新增38784次更新、4964352个训练样本位置。修正组全部启动时额外12288次更新、1572864个样本位置。校准另需15360个GeoSteiner标签、61440次采样动作线长评估，以及98304个Critic拟合样本位置；这些不算Actor训练量。

原组完整保留模型、Adam、RNG、lambda、历史最佳和数据命名空间。修正组更换Critic和优化器起点，单独成组，不能与原组直接归因比较。批量保持128，以保留续训轨迹；通过最多8张GPU并行提高总吞吐。

启动与恢复：`bash launch.sh`。tmux会话`critic-paper-scale-20261011`。状态`artifacts/state.json`，每项结束自动生成RESULTS.md。原组保存512/1024/2048步里程碑，每8步保存可恢复latest。达到2048后再评估best/final，不用测试选择停止步数。

大数据实际路径为`/attached/remote-home1/Guilin/DRO/Codex/critic_paper_scale_20261011/artifacts`，通过本目录artifacts访问。代码运行、缓存和日志都从DRO/Codex定位。不新增工程测试、smoke或内容哈希。

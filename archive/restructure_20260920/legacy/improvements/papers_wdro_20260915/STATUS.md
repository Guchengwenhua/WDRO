# 实验与验收状态

更新：2026-09-19T12:43:16.297130+00:00。

**DONE：24 次正式训练和 4 次 GPU 预检已完成，2026-09-19 独立验收 PASS。**

- 正式实验于 2026-09-16 05:34:26 UTC 完成；24×2048=49,152 有效更新，原任务退出、66 把锁释放，tmux 已结束，无需继续监控。
- 288 冻结文件、初始化配对、选模、评分账本和 96 LP 重验通过；1,984 行指标和 1,072 行比较复算与原表逐字节一致。
- 新增 GPU 抽查 24 best / 12,288 次推理，792 个数组全部精确复现；未新增训练或更改科研参数。
- 科学结论：未建立跨训练 seed 稳定的 WDRO 优势。REST-WDRO 三个 best 全为初始化；Hub 多数 best 选择零 hub；Timing/Buffer 仅有局部改善信号。
- degree10 组件迁移结果不能解释为原纯线长主线改善、完整论文复现或连续 Wasserstein 证书。

以[2026-09-19 验收报告](artifacts/review_20260919/REPORT.md)及[总凭证](artifacts/review_20260919/acceptance.json)为准。原始自动报告见 [artifacts/report/REPORT.md](artifacts/report/REPORT.md)。旧交接快照保存在验收目录 before_closeout/。

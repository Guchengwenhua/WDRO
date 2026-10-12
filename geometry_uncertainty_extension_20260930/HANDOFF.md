# 扩展研究运行交接
启动日期：2026-09-30，Server03时间15:14:59。
任务：conditional geometry extension；目录 /remote-home/Guilin/DRO/Codex/geometry_uncertainty_extension_20260930。
启动命令 bash launch.sh 7；tmux geometry-extension-20260930；socket artifacts/extension.sock。
Python PID1336577；GPU7，batch256，FP32 greedy，GPU显存fraction0.10，CPU/BLAS线程1。
首批正式证据：artifacts/heartbeat.json显示18.765秒完成256条，d30_multiseed/dev/M0900/D0238，committed256/1024。
总目标61440推理实例，剩61184；首批吞吐13.642条/秒，推理剩约4485秒（75分钟），不含统计。首批估计可能波动，不是完成保证。
资源：初查GPU7 23%/47629MiB free；紧启动快照变100%/30601MiB free，属于共享繁忙卡。未杀或迁移任何运行进程，正式首批推进正常。
日志artifacts/run.log；heartbeat artifacts/heartbeat.json；冻结元信息artifacts/frozen_metadata.json；分片artifacts/d*/predictions/M*/D*/*.npz。
结果完成后为REPORT.md以及artifacts/d*/test_summary.json；当前不得声称已有扩展结论。
恢复：确认原session已结束后，仍用bash launch.sh 7；固定参数/源size和mtime不变，已提交分片自动跳过。不得同时重复启动同名session。
下一gate：本任务仅离线诊断/候选池困难检索，不执行WDRO优化。真实几何指导训练待根任务baseline忠实性审计和开发/测试结果。

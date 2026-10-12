# 熵预算、教师相对损失与尾部风险 WDRO

新探索分支，目标是训练后的REST Actor；现有主线只读。先读 [完整理论和116项实验方案](docs/理论与实验方案.md)。

7分支：ERM、hard-max WDRO、softmax+递推价格、熵正则+每批预算求解、温度延续、冻结教师相对损失、mean/CVaR混合。全程无GeoSteiner，不报告最优regret。

`artifacts` 指向附加盘 `/attached/remote-home1/Guilin/DRO/Codex/entropic_teacher_wdro_20260922/artifacts`，所有新产物通过本目录访问。

## 运行

```bash
cd /remote-home/Guilin/DRO/Codex/entropic_teacher_wdro_20260922
bash launch.sh
tmux -S /remote-home/Guilin/DRO/Codex/entropic_teacher_wdro_20260922/artifacts/operations/et.sock attach -t et-wdro
```

使用 `Ctrl-b d` 脱离，不影响训练。`artifacts/operations/queue.json` 是实时阶段、GPU/PID和进度，`eta.json` 是当前阶段估时，完整总估时见 `HANDOFF.md`。`artifacts/runs/<id>/` 包含每步日志、validation、latest/best和完成凭证。

暂停：创建 `artifacts/operations/PAUSE`，worker完成当前更新后存断点并退出；控制器保持等待。继续：删除该文件。不要直接重复启动worker。服务器重启后再运行launch；若旧worker仍在，控制器拒绝重复接管，等其完成后再启动。

调温完成后写 `selection.json`；全部训练完成后写 `model_freeze.json`，才会自动运行独立自然/Random64/ES64测试。完成后生成 `artifacts/report/REPORT.md`、`metrics.csv`、`audit.json` 和 `artifacts/operations/delivery.json`。`NEEDS_ATTENTION` 表示失败待检查，不能算完成。不得将工程PASS解读为方法有效。

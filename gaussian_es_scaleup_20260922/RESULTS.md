# Gaussian-ES 扩规模结果

工程与数值验收通过，33 个后训练模型全部完成统一测试。

预注册主收益判定：未建立。

- ES64−Random64 的 OOD P95 平均差：-0.00018053661（负数有利于ES）。
- 相对 P95 降幅：0.71%；改善种子：4/5。
- seed 符号翻转单侧 p=0.1875；描述性交叉bootstrap95%CI=[-0.0007290803873782624, 0.0003391904542629314]。
- uniform mean相对ERM差：3.9113323e-05；独立CEM64相对Random差：-0.00034589607。

逐degree/seed/分布完整数据在 artifacts/metrics.csv，全部任务与产物绑定在 artifacts/completion.json。负结果全部保留；不能推广为连续Wasserstein证书、普遍优越性或完整预训练因果结论。

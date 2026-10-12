# 共同微调与BN诊断（仅A_val）

更新：2026-09-22T14:49:51.541108+00:00。以下不是测试结果，全部选择只看A_val。

6种设置：BN正常更新/冻结运行统计×学习率1e-5、3e-6、1e-6；ERM/RAW各用seeds101/102，共24项，4个标准设置对照引用前轮、20项新训练。每项256updates，冻结BN仍训练其仿射参数。

| 设置 | 方法 | seed | 验证macro (%) | 相对父模型(pp) | 旧结果复用 |
| --- | --- | --- | --- | --- | --- |
| frozen_bn_1e5 | erm_a | 101 | 1.8616 | -0.0375 | False |
| frozen_bn_1e5 | erm_a | 102 | 1.9228 | -0.0505 | False |
| frozen_bn_1e5 | raw_wdro | 101 | 1.8609 | -0.0381 | False |
| frozen_bn_1e5 | raw_wdro | 102 | 1.8731 | -0.1002 | False |
| frozen_bn_1e6 | erm_a | 101 | 1.8867 | -0.0124 | False |
| frozen_bn_1e6 | erm_a | 102 | 1.9575 | -0.0158 | False |
| frozen_bn_1e6 | raw_wdro | 101 | 1.8842 | -0.0148 | False |
| frozen_bn_1e6 | raw_wdro | 102 | 1.9637 | -0.0096 | False |
| frozen_bn_3e6 | erm_a | 101 | 1.9009 | 0.0019 | False |
| frozen_bn_3e6 | erm_a | 102 | 1.9343 | -0.0390 | False |
| frozen_bn_3e6 | raw_wdro | 101 | 1.8920 | -0.0071 | False |
| frozen_bn_3e6 | raw_wdro | 102 | 1.9084 | -0.0649 | False |
| standard_1e5 | erm_a | 101 | 2.6779 | 0.7789 | True |
| standard_1e5 | erm_a | 102 | 2.8254 | 0.8521 | True |
| standard_1e5 | raw_wdro | 101 | 2.4950 | 0.5959 | True |
| standard_1e5 | raw_wdro | 102 | 2.6114 | 0.6381 | True |
| standard_1e6 | erm_a | 101 | 2.7207 | 0.8217 | False |
| standard_1e6 | erm_a | 102 | 2.8749 | 0.9016 | False |
| standard_1e6 | raw_wdro | 101 | 2.5817 | 0.6826 | False |
| standard_1e6 | raw_wdro | 102 | 2.7320 | 0.7587 | False |
| standard_3e6 | erm_a | 101 | 2.7030 | 0.8039 | False |
| standard_3e6 | erm_a | 102 | 2.8833 | 0.9100 | False |
| standard_3e6 | raw_wdro | 101 | 2.5403 | 0.6413 | False |
| standard_3e6 | raw_wdro | 102 | 2.6925 | 0.7192 | False |

共同设置选为 **frozen_bn_1e5**，四项update256验证macro平均=1.87958%；父模型相应平均=1.93617%。全部设置均退化：False。不为各方法单独选择超参。

| 设置 | 四项验证macro平均 (%) |
| --- | --- |
| standard_1e5 | 2.6524 |
| standard_3e6 | 2.7048 |
| standard_1e6 | 2.7273 |
| frozen_bn_1e5 | 1.8796 |
| frozen_bn_3e6 | 1.9089 |
| frozen_bn_1e6 | 1.9230 |

## BN统计交叉替换诊断

下列模型只在内存中替换统计，在A_val诊断；不作为主训练模型或测试选模候选。固定权重时的变化说明统计的贡献，不能代替重新训练对照。

| seed | 原方法 | 权重 | BN统计 | 验证mean (%) | 验证macro (%) |
| --- | --- | --- | --- | --- | --- |
| 101 | erm_a | parent | parent | 1.9191 | 1.8990 |
| 101 | erm_a | parent | last | 2.7534 | 2.7172 |
| 101 | erm_a | last | parent | 1.9701 | 1.9952 |
| 101 | erm_a | last | last | 2.6284 | 2.6779 |
| 101 | raw_wdro | parent | parent | 1.9191 | 1.8990 |
| 101 | raw_wdro | parent | last | 2.6340 | 2.6845 |
| 101 | raw_wdro | last | parent | 1.8647 | 1.8775 |
| 101 | raw_wdro | last | last | 2.4866 | 2.4950 |
| 102 | erm_a | parent | parent | 1.9487 | 1.9733 |
| 102 | erm_a | parent | last | 2.9425 | 2.9160 |
| 102 | erm_a | last | parent | 1.9440 | 1.9997 |
| 102 | erm_a | last | last | 2.7908 | 2.8254 |
| 102 | raw_wdro | parent | parent | 1.9487 | 1.9733 |
| 102 | raw_wdro | parent | last | 2.7781 | 2.7279 |
| 102 | raw_wdro | last | parent | 1.9132 | 1.9297 |
| 102 | raw_wdro | last | last | 2.6317 | 2.6114 |

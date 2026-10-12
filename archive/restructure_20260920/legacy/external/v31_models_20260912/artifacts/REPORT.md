# 新训练模型：全部现有真实数据评测

全部 18 个新模型及 15 个原有对照均完成。每个模型评测 11,032 条网络，共 364,056 次预测。

覆盖库中全部 49 个设计，43 个含合法 degree=30 网络；零样本设计仍保留在覆盖表中。

新增模型只训练了 degree=30，因此结果限于这一引脚数。所有模型使用原合成验证集选出的 best checkpoint，真实数据没有参与训练或选择。
固定 256 行 greedy 推理；尾批重复最后一行后丢弃补齐结果。所有 15 个对照按同一规则重测。GeoSteiner 参考值复用已核验的双尺度精确结果。

数值均为相对 regret，越小越好。先对每个训练种子计算，再取三个种子的均值；P95 是各个种子 P95 的均值。

| 方法 | 全部网络 mean | P95 | 设计宏平均 | 最差设计均值（种子均值） |
|---|---:|---:|---:|---:|
| random_top4 | 0.021108 | 0.057477 | 0.015625 | 0.028556 |
| critic_calibrated4 | 0.021357 | 0.058145 | 0.015728 | 0.027990 |
| rl_frozen4 | 0.021044 | 0.057009 | 0.015579 | 0.027843 |
| rl_joint4 | 0.021312 | 0.057856 | 0.015664 | 0.028182 |
| random64 | 0.021024 | 0.057676 | 0.015418 | 0.027695 |
| es64 | 0.021030 | 0.057306 | 0.015487 | 0.027822 |
| erm | 0.021349 | 0.057953 | 0.016049 | 0.029685 |
| mixed_erm | 0.020078 | 0.053185 | 0.015413 | 0.028423 |
| hard_mining | 0.021430 | 0.058626 | 0.016071 | 0.027907 |
| group_dro | 0.019833 | 0.052430 | 0.015190 | 0.025737 |
| wdro_rest | 0.021534 | 0.058737 | 0.016116 | 0.028650 |

## 分数据集

| 数据集 | 方法 | 网络数 | mean | P95 | 设计宏平均 |
|---|---|---:|---:|---:|---:|
| circuitops_asap7 | random_top4 | 65 | 0.013949 | 0.034265 | 0.011696 |
| circuitops_asap7 | critic_calibrated4 | 65 | 0.013150 | 0.031267 | 0.009820 |
| circuitops_asap7 | rl_frozen4 | 65 | 0.013912 | 0.032011 | 0.011316 |
| circuitops_asap7 | rl_joint4 | 65 | 0.013321 | 0.030474 | 0.011581 |
| circuitops_asap7 | random64 | 65 | 0.013234 | 0.036076 | 0.009264 |
| circuitops_asap7 | es64 | 65 | 0.013352 | 0.031843 | 0.010542 |
| circuitops_asap7 | erm | 65 | 0.013816 | 0.033670 | 0.012726 |
| circuitops_asap7 | mixed_erm | 65 | 0.014467 | 0.035113 | 0.012470 |
| circuitops_asap7 | hard_mining | 65 | 0.013716 | 0.038881 | 0.012369 |
| circuitops_asap7 | group_dro | 65 | 0.014308 | 0.036637 | 0.012834 |
| circuitops_asap7 | wdro_rest | 65 | 0.013633 | 0.029809 | 0.010667 |
| circuitops_nangate45 | random_top4 | 79 | 0.010890 | 0.027563 | 0.011019 |
| circuitops_nangate45 | critic_calibrated4 | 79 | 0.011713 | 0.027722 | 0.013087 |
| circuitops_nangate45 | rl_frozen4 | 79 | 0.011776 | 0.033045 | 0.012584 |
| circuitops_nangate45 | rl_joint4 | 79 | 0.011320 | 0.028517 | 0.012073 |
| circuitops_nangate45 | random64 | 79 | 0.011507 | 0.029094 | 0.013487 |
| circuitops_nangate45 | es64 | 79 | 0.010964 | 0.027212 | 0.012152 |
| circuitops_nangate45 | erm | 79 | 0.011399 | 0.027338 | 0.012436 |
| circuitops_nangate45 | mixed_erm | 79 | 0.011861 | 0.028682 | 0.012398 |
| circuitops_nangate45 | hard_mining | 79 | 0.011711 | 0.031109 | 0.013020 |
| circuitops_nangate45 | group_dro | 79 | 0.011753 | 0.029286 | 0.011058 |
| circuitops_nangate45 | wdro_rest | 79 | 0.011768 | 0.029062 | 0.012951 |
| circuitops_sky130hd | random_top4 | 2 | 0.016935 | 0.032176 | 0.016935 |
| circuitops_sky130hd | critic_calibrated4 | 2 | 0.011105 | 0.017409 | 0.011105 |
| circuitops_sky130hd | rl_frozen4 | 2 | 0.011510 | 0.017649 | 0.011510 |
| circuitops_sky130hd | rl_joint4 | 2 | 0.009664 | 0.015403 | 0.009664 |
| circuitops_sky130hd | random64 | 2 | 0.008996 | 0.013475 | 0.008996 |
| circuitops_sky130hd | es64 | 2 | 0.011510 | 0.017649 | 0.011510 |
| circuitops_sky130hd | erm | 2 | 0.017637 | 0.031135 | 0.017637 |
| circuitops_sky130hd | mixed_erm | 2 | 0.016518 | 0.021702 | 0.016518 |
| circuitops_sky130hd | hard_mining | 2 | 0.011219 | 0.020457 | 0.011219 |
| circuitops_sky130hd | group_dro | 2 | 0.009187 | 0.017456 | 0.009187 |
| circuitops_sky130hd | wdro_rest | 2 | 0.018609 | 0.024634 | 0.018609 |
| ispd2008_global_routing | random_top4 | 9088 | 0.022759 | 0.060769 | 0.020824 |
| ispd2008_global_routing | critic_calibrated4 | 9088 | 0.023037 | 0.061703 | 0.021207 |
| ispd2008_global_routing | rl_frozen4 | 9088 | 0.022693 | 0.060158 | 0.020631 |
| ispd2008_global_routing | rl_joint4 | 9088 | 0.022982 | 0.061283 | 0.020890 |
| ispd2008_global_routing | random64 | 9088 | 0.022667 | 0.061133 | 0.020767 |
| ispd2008_global_routing | es64 | 9088 | 0.022693 | 0.060786 | 0.020846 |
| ispd2008_global_routing | erm | 9088 | 0.023017 | 0.061143 | 0.021036 |
| ispd2008_global_routing | mixed_erm | 9088 | 0.021536 | 0.056219 | 0.019756 |
| ispd2008_global_routing | hard_mining | 9088 | 0.023082 | 0.062088 | 0.021373 |
| ispd2008_global_routing | group_dro | 9088 | 0.021251 | 0.055252 | 0.019622 |
| ispd2008_global_routing | wdro_rest | 9088 | 0.023216 | 0.062526 | 0.021368 |
| ispd2018_protocol_a | random_top4 | 468 | 0.013483 | 0.035297 | 0.012699 |
| ispd2018_protocol_a | critic_calibrated4 | 468 | 0.013456 | 0.033400 | 0.012663 |
| ispd2018_protocol_a | rl_frozen4 | 468 | 0.013544 | 0.035982 | 0.012825 |
| ispd2018_protocol_a | rl_joint4 | 468 | 0.013661 | 0.035355 | 0.012877 |
| ispd2018_protocol_a | random64 | 468 | 0.013559 | 0.035389 | 0.012782 |
| ispd2018_protocol_a | es64 | 468 | 0.013273 | 0.034521 | 0.012697 |
| ispd2018_protocol_a | erm | 468 | 0.013616 | 0.034891 | 0.012942 |
| ispd2018_protocol_a | mixed_erm | 468 | 0.013297 | 0.034556 | 0.012666 |
| ispd2018_protocol_a | hard_mining | 468 | 0.013658 | 0.034617 | 0.012932 |
| ispd2018_protocol_a | group_dro | 468 | 0.013175 | 0.033310 | 0.012770 |
| ispd2018_protocol_a | wdro_rest | 468 | 0.014100 | 0.035040 | 0.013374 |
| ispd2019_protocol_a | random_top4 | 1330 | 0.013474 | 0.035925 | 0.012843 |
| ispd2019_protocol_a | critic_calibrated4 | 1330 | 0.013642 | 0.035229 | 0.013050 |
| ispd2019_protocol_a | rl_frozen4 | 1330 | 0.013334 | 0.035009 | 0.012861 |
| ispd2019_protocol_a | rl_joint4 | 1330 | 0.013593 | 0.036421 | 0.013093 |
| ispd2019_protocol_a | random64 | 1330 | 0.013389 | 0.034767 | 0.012461 |
| ispd2019_protocol_a | es64 | 1330 | 0.013385 | 0.035383 | 0.012632 |
| ispd2019_protocol_a | erm | 1330 | 0.013642 | 0.035153 | 0.013172 |
| ispd2019_protocol_a | mixed_erm | 1330 | 0.013270 | 0.035088 | 0.012941 |
| ispd2019_protocol_a | hard_mining | 1330 | 0.013843 | 0.034831 | 0.013264 |
| ispd2019_protocol_a | group_dro | 1330 | 0.013257 | 0.034221 | 0.013287 |
| ispd2019_protocol_a | wdro_rest | 1330 | 0.013626 | 0.035271 | 0.012770 |

## 逐对差异与解释边界

下表为全体网络的配对 mean 差值（新方法减对照），负数有利于新方法。95% CI 按设计家族聚类自助法计算，已将共享完全相同归一化几何的设计连接到同一簇。

| 新方法 | 对照 | 差值 | 95% CI |
|---|---|---:|---|
| random_top4 | erm | -0.000241 | [-0.000350, -0.000117] |
| random_top4 | wdro_rest | -0.000426 | [-0.000637, -0.000193] |
| critic_calibrated4 | erm | 0.000008 | [-0.000095, 0.000135] |
| critic_calibrated4 | wdro_rest | -0.000177 | [-0.000437, 0.000106] |
| rl_frozen4 | erm | -0.000305 | [-0.000433, -0.000145] |
| rl_frozen4 | wdro_rest | -0.000490 | [-0.000774, -0.000174] |
| rl_joint4 | erm | -0.000038 | [-0.000151, 0.000060] |
| rl_joint4 | wdro_rest | -0.000223 | [-0.000380, -0.000058] |
| random64 | erm | -0.000325 | [-0.000411, -0.000217] |
| random64 | wdro_rest | -0.000510 | [-0.000743, -0.000232] |
| es64 | erm | -0.000319 | [-0.000407, -0.000210] |
| es64 | wdro_rest | -0.000504 | [-0.000743, -0.000240] |
| critic_calibrated4 | random_top4 | 0.000249 | [0.000125, 0.000403] |
| rl_frozen4 | random_top4 | -0.000063 | [-0.000167, 0.000066] |
| rl_joint4 | random_top4 | 0.000204 | [0.000046, 0.000329] |
| rl_joint4 | rl_frozen4 | 0.000267 | [0.000071, 0.000443] |
| es64 | random64 | 0.000006 | [-0.000066, 0.000088] |

完整逐设计、逐种子、分数据集 CI 见 tables/。未对多重比较作校正；这些区间是探索性配对比较。
数据量较大的 ISPD2008 主导网络加权总体值，因此需同时查看设计宏平均和 CircuitOps/各 ISPD 年份拆分。
保留原始协议的全部样本（重复归一化几何行数 3）；不能把这些行视作独立芯片。
CircuitOps 是开放芯片设计的物理实现数据，本评测不主张这些设计均已流片。
这是原始真实坐标上的泛化评测；没有新运行对抗攻击，不据此给出 Wasserstein 鲁棒性证书。

## 验收

全部结果逐网验证树连通性、独立 Python 线长、输入身份、参考值、regret、分片及发布文件一致性；模型和依赖哈希保持不变，33 个工作进程退出且锁已释放。

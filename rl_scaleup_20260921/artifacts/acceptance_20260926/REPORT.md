# RL 优化方向：独立验收与统计

**状态：PASS**。已通过 80/80 个协议任务。

## 统计口径

所有差值均为方法减对照。固定模型攻击的 gain 越大越好；共同训练后的自身动作线长越小越好。
先在每个训练 seed 内对相同中心配对，再对训练 seed 和各分布内共同中心做交叉 bootstrap（2000 次）：每次抽取一组 seed，所有 seed 共用同一组抽样中心；各 seed、各分布等权。没有把 seed×net 当作独立训练重复。
八个训练 seed；本表 95% 区间为描述性区间。预注册的八个主比较另由 confirmatory.py 做 Holm 校正。候选攻击不能证明连续 Wasserstein 鲁棒性或最优性差距。

metrics.csv 给出逐 seed、逐分布的 mean/P95，以及共同训练的 worst-group mean。comparisons.csv 包含逐 seed 配对区间和 seed 汇总区间；区间针对平均差，P95 与 worst-group 是描述性统计。

## 汇总比较

| 阶段 | 指标 | 方法 − 对照 | 均值差 | 95% CI | seeds |
|---|---|---|---:|---|---:|
| coev | es32_length | bestof_ppo − erm | -0.146046 | [-0.251697, -0.0422043] | 8 |
| coev | es32_length | bestof_ppo − es | -0.0598542 | [-0.308076, 0.109092] | 8 |
| coev | es32_length | bestof_ppo − random | 0.0116949 | [-0.104845, 0.145135] | 8 |
| coev | es32_length | bestof_ppo − reinforce | -0.177039 | [-0.411929, -0.00126366] | 8 |
| coev | es32_length | bestof_ppo − sequential_ppo | -0.138893 | [-0.379226, 0.0331747] | 8 |
| coev | es32_length | erm − es | 0.086192 | [-0.197195, 0.26417] | 8 |
| coev | es32_length | erm − random | 0.157741 | [0.0734739, 0.231918] | 8 |
| coev | es32_length | erm − reinforce | -0.0309925 | [-0.296839, 0.156923] | 8 |
| coev | es32_length | erm − sequential_ppo | 0.00715369 | [-0.287266, 0.218776] | 8 |
| coev | es32_length | es − random | 0.0715491 | [-0.0842905, 0.29236] | 8 |
| coev | es32_length | es − reinforce | -0.117185 | [-0.17661, -0.0635822] | 8 |
| coev | es32_length | es − sequential_ppo | -0.0790383 | [-0.176599, 0.00812833] | 8 |
| coev | es32_length | random − reinforce | -0.188734 | [-0.394702, -0.0192855] | 8 |
| coev | es32_length | random − sequential_ppo | -0.150587 | [-0.384769, 0.0338303] | 8 |
| coev | es32_length | reinforce − sequential_ppo | 0.0381462 | [-0.0790746, 0.161205] | 8 |
| coev | nominal_length | bestof_ppo − erm | -0.0948529 | [-0.167972, -0.0217594] | 8 |
| coev | nominal_length | bestof_ppo − es | -0.0506496 | [-0.226308, 0.0680933] | 8 |
| coev | nominal_length | bestof_ppo − random | 0.0188001 | [-0.0619135, 0.126603] | 8 |
| coev | nominal_length | bestof_ppo − reinforce | -0.117043 | [-0.268721, -0.00485045] | 8 |
| coev | nominal_length | bestof_ppo − sequential_ppo | -0.0868283 | [-0.24775, 0.0362278] | 8 |
| coev | nominal_length | erm − es | 0.0442033 | [-0.153812, 0.167302] | 8 |
| coev | nominal_length | erm − random | 0.113653 | [0.0454744, 0.173275] | 8 |
| coev | nominal_length | erm − reinforce | -0.0221899 | [-0.195147, 0.0906009] | 8 |
| coev | nominal_length | erm − sequential_ppo | 0.00802458 | [-0.180004, 0.150311] | 8 |
| coev | nominal_length | es − random | 0.0694497 | [-0.0381718, 0.220476] | 8 |
| coev | nominal_length | es − reinforce | -0.0663932 | [-0.10613, -0.0258225] | 8 |
| coev | nominal_length | es − sequential_ppo | -0.0361787 | [-0.0998824, 0.0187866] | 8 |
| coev | nominal_length | random − reinforce | -0.135843 | [-0.258884, -0.0365992] | 8 |
| coev | nominal_length | random − sequential_ppo | -0.105628 | [-0.241463, 0.0138462] | 8 |
| coev | nominal_length | reinforce − sequential_ppo | 0.0302145 | [-0.0367981, 0.105801] | 8 |
| coev | random32_length | bestof_ppo − erm | -0.148274 | [-0.254688, -0.0459417] | 8 |
| coev | random32_length | bestof_ppo − es | -0.0619734 | [-0.312457, 0.110296] | 8 |
| coev | random32_length | bestof_ppo − random | 0.00769358 | [-0.112115, 0.144216] | 8 |
| coev | random32_length | bestof_ppo − reinforce | -0.181171 | [-0.42016, -0.00187117] | 8 |
| coev | random32_length | bestof_ppo − sequential_ppo | -0.144064 | [-0.388594, 0.0308917] | 8 |
| coev | random32_length | erm − es | 0.0863006 | [-0.198873, 0.266522] | 8 |
| coev | random32_length | erm − random | 0.155968 | [0.0725874, 0.233319] | 8 |
| coev | random32_length | erm − reinforce | -0.0328966 | [-0.303143, 0.15734] | 8 |
| coev | random32_length | erm − sequential_ppo | 0.0042099 | [-0.290432, 0.217768] | 8 |
| coev | random32_length | es − random | 0.069667 | [-0.0891314, 0.293599] | 8 |
| coev | random32_length | es − reinforce | -0.119197 | [-0.175568, -0.0669091] | 8 |
| coev | random32_length | es − sequential_ppo | -0.0820907 | [-0.181308, 0.00718026] | 8 |
| coev | random32_length | random − reinforce | -0.188864 | [-0.402951, -0.0178409] | 8 |
| coev | random32_length | random − sequential_ppo | -0.151758 | [-0.386585, 0.0324307] | 8 |
| coev | random32_length | reinforce − sequential_ppo | 0.0371065 | [-0.079279, 0.160414] | 8 |
| fixed | lambda10__gain | bestof_ppo − es | 0.0663792 | [0.047129, 0.088378] | 8 |
| fixed | lambda10__gain | bestof_ppo − random | 0.0758342 | [0.0558781, 0.0995403] | 8 |
| fixed | lambda10__gain | bestof_ppo − reinforce | 0.00593615 | [-0.00681853, 0.0184476] | 8 |
| fixed | lambda10__gain | reinforce − es | 0.060443 | [0.0494105, 0.0740316] | 8 |
| fixed | lambda10__gain | reinforce − random | 0.0698981 | [0.0580727, 0.0843936] | 8 |
| fixed | lambda10__gain | sequential_ppo − es | 0.0679294 | [0.0544597, 0.0808612] | 8 |
| fixed | lambda10__gain | sequential_ppo − random | 0.0773845 | [0.063057, 0.0916633] | 8 |
| fixed | lambda10__gain | sequential_ppo − reinforce | 0.00748641 | [-0.00404277, 0.0192475] | 8 |
| fixed | lambda1__gain | bestof_ppo − es | 0.228956 | [0.2123, 0.246403] | 8 |
| fixed | lambda1__gain | bestof_ppo − random | 0.250766 | [0.236969, 0.265217] | 8 |
| fixed | lambda1__gain | bestof_ppo − reinforce | 0.0371113 | [0.0263119, 0.0477956] | 8 |
| fixed | lambda1__gain | reinforce − es | 0.191845 | [0.176734, 0.205565] | 8 |
| fixed | lambda1__gain | reinforce − random | 0.213655 | [0.202463, 0.225113] | 8 |
| fixed | lambda1__gain | sequential_ppo − es | 0.163855 | [0.146931, 0.18011] | 8 |
| fixed | lambda1__gain | sequential_ppo − random | 0.185665 | [0.17171, 0.198855] | 8 |
| fixed | lambda1__gain | sequential_ppo − reinforce | -0.0279899 | [-0.0361999, -0.019928] | 8 |
| fixed | lambda5__gain | bestof_ppo − es | 0.165459 | [0.151414, 0.182198] | 8 |
| fixed | lambda5__gain | bestof_ppo − random | 0.185816 | [0.169571, 0.204405] | 8 |
| fixed | lambda5__gain | bestof_ppo − reinforce | 0.0170006 | [0.00575199, 0.0288506] | 8 |
| fixed | lambda5__gain | reinforce − es | 0.148458 | [0.137451, 0.160144] | 8 |
| fixed | lambda5__gain | reinforce − random | 0.168815 | [0.157582, 0.18096] | 8 |
| fixed | lambda5__gain | sequential_ppo − es | 0.142524 | [0.127668, 0.156934] | 8 |
| fixed | lambda5__gain | sequential_ppo − random | 0.162881 | [0.147249, 0.177332] | 8 |
| fixed | lambda5__gain | sequential_ppo − reinforce | -0.00593359 | [-0.0167023, 0.00445487] | 8 |

## 独立检查范围

协议、源码、依赖及产物 SHA256；CUDA 执行；正式更新及调用预算；有限正线长；坐标窗口、pin 间距、位移限制；运输成本和攻击 gain 独立重算；nominal topology 重新计算 REST 线长。

actual_call_log_totals 同时保留正式预算、已完成重试额外调用、以及未配对的中断开始事件；未配对请求行数不能当成已完成评分。任一任务缺失、正式预算不符、配对失败或绑定失败均不能标记 DONE。此报告不会启动、恢复或修改任何训练任务。

# RL 优化方向：独立验收与统计

**状态：PASS**。已通过 10/10 个协议任务。

## 统计口径

所有差值均为方法减对照。固定模型攻击的 gain 越大越好；共同训练后的自身动作线长越小越好。
先在每个训练 seed 内对相同中心配对，再对训练 seed 和各分布内共同中心做交叉 bootstrap（2000 次）：每次抽取一组 seed，所有 seed 共用同一组抽样中心；各 seed、各分布等权。没有把 seed×net 当作独立训练重复。
八个训练 seed；本表 95% 区间为描述性区间。预注册的八个主比较另由 confirmatory.py 做 Holm 校正。候选攻击不能证明连续 Wasserstein 鲁棒性或最优性差距。

metrics.csv 给出逐 seed、逐分布的 mean/P95，以及共同训练的 worst-group mean。comparisons.csv 包含逐 seed 配对区间和 seed 汇总区间；区间针对平均差，P95 与 worst-group 是描述性统计。

## 汇总比较

| 阶段 | 指标 | 方法 − 对照 | 均值差 | 95% CI | seeds |
|---|---|---|---:|---|---:|
| coev | es32_length | bestof_ppo − erm | -0.0716044 | [-0.148726, 0.00269724] | 1 |
| coev | es32_length | bestof_ppo − es | 0.267596 | [0.204701, 0.333468] | 1 |
| coev | es32_length | bestof_ppo − random | 0.286549 | [0.218984, 0.349887] | 1 |
| coev | es32_length | bestof_ppo − reinforce | 0.232623 | [0.135652, 0.336866] | 1 |
| coev | es32_length | bestof_ppo − sequential_ppo | -0.0445741 | [-0.120397, 0.0255794] | 1 |
| coev | es32_length | erm − es | 0.339201 | [0.255227, 0.422774] | 1 |
| coev | es32_length | erm − random | 0.358153 | [0.272139, 0.445158] | 1 |
| coev | es32_length | erm − reinforce | 0.304228 | [0.213826, 0.404931] | 1 |
| coev | es32_length | erm − sequential_ppo | 0.0270304 | [-0.00793408, 0.0630332] | 1 |
| coev | es32_length | es − random | 0.0189524 | [-0.0474508, 0.0826681] | 1 |
| coev | es32_length | es − reinforce | -0.0349729 | [-0.146825, 0.0820649] | 1 |
| coev | es32_length | es − sequential_ppo | -0.31217 | [-0.397479, -0.228639] | 1 |
| coev | es32_length | random − reinforce | -0.0539253 | [-0.163651, 0.0664208] | 1 |
| coev | es32_length | random − sequential_ppo | -0.331123 | [-0.418348, -0.239305] | 1 |
| coev | es32_length | reinforce − sequential_ppo | -0.277197 | [-0.371142, -0.194888] | 1 |
| coev | nominal_length | bestof_ppo − erm | -0.0466287 | [-0.151301, 0.0673372] | 1 |
| coev | nominal_length | bestof_ppo − es | 0.222994 | [0.135447, 0.312445] | 1 |
| coev | nominal_length | bestof_ppo − random | 0.181606 | [0.11658, 0.248908] | 1 |
| coev | nominal_length | bestof_ppo − reinforce | 1.31998 | [1.05394, 1.59525] | 1 |
| coev | nominal_length | bestof_ppo − sequential_ppo | -0.0550487 | [-0.1603, 0.0622706] | 1 |
| coev | nominal_length | erm − es | 0.269623 | [0.144033, 0.387505] | 1 |
| coev | nominal_length | erm − random | 0.228234 | [0.105734, 0.337197] | 1 |
| coev | nominal_length | erm − reinforce | 1.36661 | [1.09237, 1.64713] | 1 |
| coev | nominal_length | erm − sequential_ppo | -0.00842002 | [-0.0416154, 0.0270104] | 1 |
| coev | nominal_length | es − random | -0.0413882 | [-0.116823, 0.0341861] | 1 |
| coev | nominal_length | es − reinforce | 1.09699 | [0.820284, 1.38372] | 1 |
| coev | nominal_length | es − sequential_ppo | -0.278043 | [-0.395575, -0.148561] | 1 |
| coev | nominal_length | random − reinforce | 1.13838 | [0.859458, 1.43417] | 1 |
| coev | nominal_length | random − sequential_ppo | -0.236654 | [-0.344249, -0.113596] | 1 |
| coev | nominal_length | reinforce − sequential_ppo | -1.37503 | [-1.65686, -1.09906] | 1 |
| coev | random32_length | bestof_ppo − erm | -0.141478 | [-0.227227, -0.0603809] | 1 |
| coev | random32_length | bestof_ppo − es | 0.288299 | [0.219487, 0.353763] | 1 |
| coev | random32_length | bestof_ppo − random | 0.289179 | [0.222887, 0.353882] | 1 |
| coev | random32_length | bestof_ppo − reinforce | 0.204327 | [0.0943172, 0.317267] | 1 |
| coev | random32_length | bestof_ppo − sequential_ppo | -0.0880671 | [-0.172853, -0.00691943] | 1 |
| coev | random32_length | erm − es | 0.429777 | [0.347836, 0.511949] | 1 |
| coev | random32_length | erm − random | 0.430657 | [0.344565, 0.521254] | 1 |
| coev | random32_length | erm − reinforce | 0.345805 | [0.244529, 0.464979] | 1 |
| coev | random32_length | erm − sequential_ppo | 0.0534109 | [0.00763642, 0.107925] | 1 |
| coev | random32_length | es − random | 0.000880312 | [-0.064154, 0.0656334] | 1 |
| coev | random32_length | es − reinforce | -0.0839721 | [-0.209622, 0.0485505] | 1 |
| coev | random32_length | es − sequential_ppo | -0.376366 | [-0.469664, -0.28399] | 1 |
| coev | random32_length | random − reinforce | -0.0848524 | [-0.204548, 0.0373209] | 1 |
| coev | random32_length | random − sequential_ppo | -0.377247 | [-0.469807, -0.289221] | 1 |
| coev | random32_length | reinforce − sequential_ppo | -0.292394 | [-0.408515, -0.18756] | 1 |
| fixed | lambda10__gain | bestof_ppo − es | 0.128265 | [0.0491353, 0.214181] | 1 |
| fixed | lambda10__gain | bestof_ppo − random | 0.196091 | [0.107059, 0.286402] | 1 |
| fixed | lambda10__gain | bestof_ppo − reinforce | 0.118405 | [0.0372084, 0.21044] | 1 |
| fixed | lambda10__gain | reinforce − es | 0.00986008 | [-0.0794356, 0.0944963] | 1 |
| fixed | lambda10__gain | reinforce − random | 0.0776866 | [-0.0207116, 0.167421] | 1 |
| fixed | lambda10__gain | sequential_ppo − es | -0.0302342 | [-0.1212, 0.0645151] | 1 |
| fixed | lambda10__gain | sequential_ppo − random | 0.0375923 | [-0.0690767, 0.142092] | 1 |
| fixed | lambda10__gain | sequential_ppo − reinforce | -0.0400942 | [-0.129927, 0.0614388] | 1 |
| fixed | lambda1__gain | bestof_ppo − es | 0.188475 | [0.0979436, 0.279714] | 1 |
| fixed | lambda1__gain | bestof_ppo − random | 0.268845 | [0.174612, 0.359551] | 1 |
| fixed | lambda1__gain | bestof_ppo − reinforce | 0.0768003 | [-0.00301248, 0.160375] | 1 |
| fixed | lambda1__gain | reinforce − es | 0.111674 | [0.0289388, 0.193971] | 1 |
| fixed | lambda1__gain | reinforce − random | 0.192045 | [0.100121, 0.277631] | 1 |
| fixed | lambda1__gain | sequential_ppo − es | 0.167001 | [0.0712548, 0.264238] | 1 |
| fixed | lambda1__gain | sequential_ppo − random | 0.247371 | [0.143081, 0.345558] | 1 |
| fixed | lambda1__gain | sequential_ppo − reinforce | 0.0553263 | [-0.0447983, 0.159288] | 1 |
| fixed | lambda5__gain | bestof_ppo − es | 0.20568 | [0.10082, 0.303962] | 1 |
| fixed | lambda5__gain | bestof_ppo − random | 0.188178 | [0.0938306, 0.27755] | 1 |
| fixed | lambda5__gain | bestof_ppo − reinforce | 0.105789 | [0.000893068, 0.19842] | 1 |
| fixed | lambda5__gain | reinforce − es | 0.0998908 | [0.0119013, 0.194582] | 1 |
| fixed | lambda5__gain | reinforce − random | 0.0823885 | [0.00714847, 0.163767] | 1 |
| fixed | lambda5__gain | sequential_ppo − es | 0.0461387 | [-0.0567426, 0.151668] | 1 |
| fixed | lambda5__gain | sequential_ppo − random | 0.0286364 | [-0.0674049, 0.121016] | 1 |
| fixed | lambda5__gain | sequential_ppo − reinforce | -0.0537521 | [-0.154989, 0.0422805] | 1 |

## 独立检查范围

协议、源码、依赖及产物 SHA256；CUDA 执行；正式更新及调用预算；有限正线长；坐标窗口、pin 间距、位移限制；运输成本和攻击 gain 独立重算；nominal topology 重新计算 REST 线长。

actual_call_log_totals 同时保留正式预算、已完成重试额外调用、以及未配对的中断开始事件；未配对请求行数不能当成已完成评分。任一任务缺失、正式预算不符、配对失败或绑定失败均不能标记 DONE。此报告不会启动、恢复或修改任何训练任务。

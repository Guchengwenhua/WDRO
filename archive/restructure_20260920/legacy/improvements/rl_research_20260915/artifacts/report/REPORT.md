# RL 优化方向：独立验收与统计

**状态：PASS**。已通过 51/51 个协议任务。

## 统计口径

所有差值均为方法减对照。固定模型攻击的 gain 越大越好；共同训练后的自身动作线长越小越好。
先在每个训练 seed 内对相同中心配对，再对训练 seed 和各分布内共同中心做交叉 bootstrap（2000 次）：每次抽取一组 seed，所有 seed 共用同一组抽样中心；各 seed、各分布等权。没有把 seed×net 当作独立训练重复。
仅三个训练 seed，95% 区间不稳定；多方法、多分布比较未做多重比较校正，只作探索性证据。候选攻击不能证明连续 Wasserstein 鲁棒性或最优性差距。

metrics.csv 给出逐 seed、逐分布的 mean/P95，以及共同训练的 worst-group mean。comparisons.csv 包含逐 seed 配对区间和 seed 汇总区间；区间针对平均差，P95 与 worst-group 是描述性统计。

## 汇总比较

| 阶段 | 指标 | 方法 − 对照 | 均值差 | 95% CI | seeds |
|---|---|---|---:|---|---:|
| coev | es32_length | bestof_ppo − erm | -0.141533 | [-0.384859, 0.0470693] | 3 |
| coev | es32_length | bestof_ppo − es | -0.21904 | [-0.661046, 0.150241] | 3 |
| coev | es32_length | bestof_ppo − maddpg_bandit | -0.208875 | [-0.447155, 0.203242] | 3 |
| coev | es32_length | bestof_ppo − mappo | -0.201521 | [-0.287201, -0.0714029] | 3 |
| coev | es32_length | bestof_ppo − population_ppo | -0.135798 | [-0.314839, 0.169618] | 3 |
| coev | es32_length | bestof_ppo − ppo | -0.14343 | [-0.478453, 0.1149] | 3 |
| coev | es32_length | bestof_ppo − random | -0.00941452 | [-0.245922, 0.147625] | 3 |
| coev | es32_length | bestof_ppo − reinforce | -0.21299 | [-0.572521, 0.14206] | 3 |
| coev | es32_length | bestof_ppo − sequential_ppo | -0.0260746 | [-0.321642, 0.219333] | 3 |
| coev | es32_length | erm − es | -0.0775065 | [-0.576894, 0.234355] | 3 |
| coev | es32_length | erm − maddpg_bandit | -0.0673421 | [-0.289763, 0.150946] | 3 |
| coev | es32_length | erm − mappo | -0.0599875 | [-0.192234, 0.131387] | 3 |
| coev | es32_length | erm − population_ppo | 0.00573497 | [-0.187064, 0.12644] | 3 |
| coev | es32_length | erm − ppo | -0.00189648 | [-0.0899542, 0.0728645] | 3 |
| coev | es32_length | erm − random | 0.132119 | [0.0386292, 0.226078] | 3 |
| coev | es32_length | erm − reinforce | -0.0714565 | [-0.189219, 0.0901932] | 3 |
| coev | es32_length | erm − sequential_ppo | 0.115459 | [-0.00856412, 0.301277] | 3 |
| coev | es32_length | es − maddpg_bandit | 0.0101644 | [-0.281175, 0.28295] | 3 |
| coev | es32_length | es − mappo | 0.0175191 | [-0.222653, 0.381917] | 3 |
| coev | es32_length | es − population_ppo | 0.0832415 | [-0.143261, 0.387417] | 3 |
| coev | es32_length | es − ppo | 0.0756101 | [-0.328659, 0.607839] | 3 |
| coev | es32_length | es − random | 0.209625 | [-0.114664, 0.807912] | 3 |
| coev | es32_length | es − reinforce | 0.00605 | [-0.419397, 0.468427] | 3 |
| coev | es32_length | es − sequential_ppo | 0.192965 | [-0.188649, 0.883194] | 3 |
| coev | es32_length | maddpg_bandit − mappo | 0.00735465 | [-0.27169, 0.187074] | 3 |
| coev | es32_length | maddpg_bandit − population_ppo | 0.0730771 | [-0.0318418, 0.147356] | 3 |
| coev | es32_length | maddpg_bandit − ppo | 0.0654456 | [-0.0892583, 0.320371] | 3 |
| coev | es32_length | maddpg_bandit − random | 0.199461 | [-0.116017, 0.521607] | 3 |
| coev | es32_length | maddpg_bandit − reinforce | -0.00411443 | [-0.132638, 0.183301] | 3 |
| coev | es32_length | maddpg_bandit − sequential_ppo | 0.182801 | [-0.162052, 0.59637] | 3 |
| coev | es32_length | mappo − population_ppo | 0.0657224 | [-0.0485657, 0.237433] | 3 |
| coev | es32_length | mappo − ppo | 0.058091 | [-0.226701, 0.228341] | 3 |
| coev | es32_length | mappo − random | 0.192106 | [0.00250875, 0.423152] | 3 |
| coev | es32_length | mappo − reinforce | -0.0114691 | [-0.318488, 0.209122] | 3 |
| coev | es32_length | mappo − sequential_ppo | 0.175446 | [-0.0760509, 0.495964] | 3 |
| coev | es32_length | population_ppo − ppo | -0.00763145 | [-0.17479, 0.216243] | 3 |
| coev | es32_length | population_ppo − random | 0.126384 | [-0.083742, 0.415286] | 3 |
| coev | es32_length | population_ppo − reinforce | -0.0771915 | [-0.265867, 0.0760774] | 3 |
| coev | es32_length | population_ppo − sequential_ppo | 0.109724 | [-0.126435, 0.490086] | 3 |
| coev | es32_length | ppo − random | 0.134015 | [-0.0281656, 0.239181] | 3 |
| coev | es32_length | ppo − reinforce | -0.0695601 | [-0.140362, 0.0236087] | 3 |
| coev | es32_length | ppo − sequential_ppo | 0.117355 | [-0.0737998, 0.269878] | 3 |
| coev | es32_length | random − reinforce | -0.203575 | [-0.352258, 0.0554081] | 3 |
| coev | es32_length | random − sequential_ppo | -0.01666 | [-0.0794554, 0.0703714] | 3 |
| coev | es32_length | reinforce − sequential_ppo | 0.186915 | [-0.0991228, 0.412669] | 3 |
| coev | nominal_length | bestof_ppo − erm | -0.104991 | [-0.246426, 0.101056] | 3 |
| coev | nominal_length | bestof_ppo − es | -0.0972263 | [-0.338948, 0.122862] | 3 |
| coev | nominal_length | bestof_ppo − maddpg_bandit | -0.101179 | [-0.252305, 0.163666] | 3 |
| coev | nominal_length | bestof_ppo − mappo | -0.0984712 | [-0.157036, -0.0409309] | 3 |
| coev | nominal_length | bestof_ppo − population_ppo | -0.0674678 | [-0.165755, 0.0931481] | 3 |
| coev | nominal_length | bestof_ppo − ppo | -0.0431278 | [-0.227309, 0.0983758] | 3 |
| coev | nominal_length | bestof_ppo − random | 0.00265828 | [-0.147051, 0.159575] | 3 |
| coev | nominal_length | bestof_ppo − reinforce | -0.0618656 | [-0.221084, 0.129681] | 3 |
| coev | nominal_length | bestof_ppo − sequential_ppo | 0.0626897 | [-0.10069, 0.26544] | 3 |
| coev | nominal_length | erm − es | 0.00776423 | [-0.169565, 0.169667] | 3 |
| coev | nominal_length | erm − maddpg_bandit | 0.00381141 | [-0.0513241, 0.0597341] | 3 |
| coev | nominal_length | erm − mappo | 0.00651939 | [-0.140371, 0.148371] | 3 |
| coev | nominal_length | erm − population_ppo | 0.0375227 | [-0.00870827, 0.0809456] | 3 |
| coev | nominal_length | erm − ppo | 0.0618628 | [-0.00366187, 0.165944] | 3 |
| coev | nominal_length | erm − random | 0.107649 | [-0.101864, 0.321299] | 3 |
| coev | nominal_length | erm − reinforce | 0.043125 | [0.0198821, 0.0808788] | 3 |
| coev | nominal_length | erm − sequential_ppo | 0.16768 | [-0.0752846, 0.426871] | 3 |
| coev | nominal_length | es − maddpg_bandit | -0.00395283 | [-0.175693, 0.128088] | 3 |
| coev | nominal_length | es − mappo | -0.00124485 | [-0.161485, 0.178889] | 3 |
| coev | nominal_length | es − population_ppo | 0.0297585 | [-0.0894535, 0.212137] | 3 |
| coev | nominal_length | es − ppo | 0.0540985 | [-0.1515, 0.343002] | 3 |
| coev | nominal_length | es − random | 0.0998846 | [-0.122314, 0.500395] | 3 |
| coev | nominal_length | es − reinforce | 0.0353607 | [-0.14507, 0.248626] | 3 |
| coev | nominal_length | es − sequential_ppo | 0.159916 | [-0.0958313, 0.6067] | 3 |
| coev | nominal_length | maddpg_bandit − mappo | 0.00270798 | [-0.202332, 0.153633] | 3 |
| coev | nominal_length | maddpg_bandit − population_ppo | 0.0337113 | [-0.0698837, 0.0939197] | 3 |
| coev | nominal_length | maddpg_bandit − ppo | 0.0580514 | [-0.0639254, 0.213875] | 3 |
| coev | nominal_length | maddpg_bandit − random | 0.103837 | [-0.163443, 0.37103] | 3 |
| coev | nominal_length | maddpg_bandit − reinforce | 0.0393136 | [-0.0323648, 0.119007] | 3 |
| coev | nominal_length | maddpg_bandit − sequential_ppo | 0.163869 | [-0.137626, 0.476046] | 3 |
| coev | nominal_length | mappo − population_ppo | 0.0310034 | [-0.0662216, 0.130798] | 3 |
| coev | nominal_length | mappo − ppo | 0.0553434 | [-0.128919, 0.164317] | 3 |
| coev | nominal_length | mappo − random | 0.101129 | [-0.0497585, 0.318885] | 3 |
| coev | nominal_length | mappo − reinforce | 0.0366056 | [-0.122801, 0.168861] | 3 |
| coev | nominal_length | mappo − sequential_ppo | 0.161161 | [-0.00205904, 0.425517] | 3 |
| coev | nominal_length | population_ppo − ppo | 0.02434 | [-0.0571789, 0.129189] | 3 |
| coev | nominal_length | population_ppo − random | 0.0701261 | [-0.091846, 0.286195] | 3 |
| coev | nominal_length | population_ppo − reinforce | 0.00560223 | [-0.0509615, 0.0432983] | 3 |
| coev | nominal_length | population_ppo − sequential_ppo | 0.130158 | [-0.0659243, 0.392506] | 3 |
| coev | nominal_length | ppo − random | 0.0457861 | [-0.097871, 0.155771] | 3 |
| coev | nominal_length | ppo − reinforce | -0.0187378 | [-0.0926592, 0.0312846] | 3 |
| coev | nominal_length | ppo − sequential_ppo | 0.105818 | [-0.0708955, 0.261494] | 3 |
| coev | nominal_length | random − reinforce | -0.0645239 | [-0.249874, 0.130079] | 3 |
| coev | nominal_length | random − sequential_ppo | 0.0600314 | [0.0274033, 0.103534] | 3 |
| coev | nominal_length | reinforce − sequential_ppo | 0.124555 | [-0.103587, 0.355503] | 3 |
| coev | random32_length | bestof_ppo − erm | -0.134491 | [-0.374637, 0.058085] | 3 |
| coev | random32_length | bestof_ppo − es | -0.226211 | [-0.682572, 0.133288] | 3 |
| coev | random32_length | bestof_ppo − maddpg_bandit | -0.208343 | [-0.441777, 0.187701] | 3 |
| coev | random32_length | bestof_ppo − mappo | -0.196926 | [-0.287394, -0.0681468] | 3 |
| coev | random32_length | bestof_ppo − population_ppo | -0.128443 | [-0.297247, 0.168438] | 3 |
| coev | random32_length | bestof_ppo − ppo | -0.147734 | [-0.48, 0.111175] | 3 |
| coev | random32_length | bestof_ppo − random | -0.010627 | [-0.232576, 0.136241] | 3 |
| coev | random32_length | bestof_ppo − reinforce | -0.209817 | [-0.534925, 0.132078] | 3 |
| coev | random32_length | bestof_ppo − sequential_ppo | -0.027833 | [-0.314752, 0.210058] | 3 |
| coev | random32_length | erm − es | -0.0917198 | [-0.599981, 0.253059] | 3 |
| coev | random32_length | erm − maddpg_bandit | -0.0738527 | [-0.285337, 0.123866] | 3 |
| coev | random32_length | erm − mappo | -0.0624353 | [-0.20118, 0.137174] | 3 |
| coev | random32_length | erm − population_ppo | 0.00604741 | [-0.187297, 0.125184] | 3 |
| coev | random32_length | erm − ppo | -0.013243 | [-0.0996223, 0.0550311] | 3 |
| coev | random32_length | erm − random | 0.123864 | [0.0190492, 0.217154] | 3 |
| coev | random32_length | erm − reinforce | -0.0753259 | [-0.174303, 0.0666877] | 3 |
| coev | random32_length | erm − sequential_ppo | 0.106658 | [-0.0228807, 0.292651] | 3 |
| coev | random32_length | es − maddpg_bandit | 0.0178671 | [-0.302301, 0.311619] | 3 |
| coev | random32_length | es − mappo | 0.0292845 | [-0.196435, 0.397451] | 3 |
| coev | random32_length | es − population_ppo | 0.0977672 | [-0.14317, 0.409846] | 3 |
| coev | random32_length | es − ppo | 0.0784768 | [-0.358522, 0.626255] | 3 |
| coev | random32_length | es − random | 0.215584 | [-0.117851, 0.822445] | 3 |
| coev | random32_length | es − reinforce | 0.0163939 | [-0.415245, 0.475507] | 3 |
| coev | random32_length | es − sequential_ppo | 0.198378 | [-0.201838, 0.899027] | 3 |
| coev | random32_length | maddpg_bandit − mappo | 0.0114173 | [-0.253536, 0.191563] | 3 |
| coev | random32_length | maddpg_bandit − population_ppo | 0.0799001 | [-0.019222, 0.158383] | 3 |
| coev | random32_length | maddpg_bandit − ppo | 0.0606097 | [-0.0856077, 0.308679] | 3 |
| coev | random32_length | maddpg_bandit − random | 0.197716 | [-0.10992, 0.507029] | 3 |
| coev | random32_length | maddpg_bandit − reinforce | -0.00147327 | [-0.111891, 0.160216] | 3 |
| coev | random32_length | maddpg_bandit − sequential_ppo | 0.18051 | [-0.149126, 0.58325] | 3 |
| coev | random32_length | mappo − population_ppo | 0.0684828 | [-0.0396457, 0.232249] | 3 |
| coev | random32_length | mappo − ppo | 0.0491924 | [-0.247705, 0.229788] | 3 |
| coev | random32_length | mappo − random | 0.186299 | [0.00286459, 0.419743] | 3 |
| coev | random32_length | mappo − reinforce | -0.0128906 | [-0.3031, 0.198169] | 3 |
| coev | random32_length | mappo − sequential_ppo | 0.169093 | [-0.0869757, 0.494861] | 3 |
| coev | random32_length | population_ppo − ppo | -0.0192904 | [-0.203405, 0.211868] | 3 |
| coev | random32_length | population_ppo − random | 0.117816 | [-0.0902089, 0.408943] | 3 |
| coev | random32_length | population_ppo − reinforce | -0.0813734 | [-0.256804, 0.060343] | 3 |
| coev | random32_length | population_ppo − sequential_ppo | 0.10061 | [-0.129455, 0.483772] | 3 |
| coev | random32_length | ppo − random | 0.137107 | [-0.0330463, 0.249163] | 3 |
| coev | random32_length | ppo − reinforce | -0.062083 | [-0.148488, 0.0198878] | 3 |
| coev | random32_length | ppo − sequential_ppo | 0.119901 | [-0.0697944, 0.269498] | 3 |
| coev | random32_length | random − reinforce | -0.19919 | [-0.351138, 0.0542973] | 3 |
| coev | random32_length | random − sequential_ppo | -0.017206 | [-0.0891961, 0.0712673] | 3 |
| coev | random32_length | reinforce − sequential_ppo | 0.181984 | [-0.0932983, 0.419141] | 3 |
| fixed | lambda10__gain | bestof_ppo − es | 0.293573 | [0.0382317, 0.474185] | 3 |
| fixed | lambda10__gain | bestof_ppo − random | 0.34259 | [0.0709309, 0.529721] | 3 |
| fixed | lambda10__gain | bestof_ppo − reinforce | -0.000806229 | [-0.0426277, 0.0446081] | 3 |
| fixed | lambda10__gain | maddpg_bandit − es | -0.306513 | [-0.388837, -0.209343] | 3 |
| fixed | lambda10__gain | maddpg_bandit − random | -0.257496 | [-0.329239, -0.173411] | 3 |
| fixed | lambda10__gain | maddpg_bandit − reinforce | -0.600892 | [-0.829289, -0.23156] | 3 |
| fixed | lambda10__gain | mappo − es | 0.232952 | [-0.106665, 0.45194] | 3 |
| fixed | lambda10__gain | mappo − random | 0.281969 | [-0.0720878, 0.506205] | 3 |
| fixed | lambda10__gain | mappo − reinforce | -0.0614265 | [-0.135663, 0.0211143] | 3 |
| fixed | lambda10__gain | ppo − es | 0.282503 | [0.0210413, 0.463377] | 3 |
| fixed | lambda10__gain | ppo − random | 0.33152 | [0.0553818, 0.517904] | 3 |
| fixed | lambda10__gain | ppo − reinforce | -0.0118755 | [-0.0534745, 0.0282252] | 3 |
| fixed | lambda10__gain | reinforce − es | 0.294379 | [0.0308287, 0.467462] | 3 |
| fixed | lambda10__gain | reinforce − random | 0.343396 | [0.0690379, 0.520541] | 3 |
| fixed | lambda10__gain | sequential_ppo − es | 0.269351 | [0.0171932, 0.449141] | 3 |
| fixed | lambda10__gain | sequential_ppo − random | 0.318368 | [0.0509163, 0.505261] | 3 |
| fixed | lambda10__gain | sequential_ppo − reinforce | -0.0250281 | [-0.0867062, 0.0313055] | 3 |
| fixed | lambda1__gain | bestof_ppo − es | 0.510882 | [0.169098, 0.775821] | 3 |
| fixed | lambda1__gain | bestof_ppo − random | 0.556016 | [0.200313, 0.851942] | 3 |
| fixed | lambda1__gain | bestof_ppo − reinforce | 0.0104523 | [-0.0778044, 0.084683] | 3 |
| fixed | lambda1__gain | maddpg_bandit − es | -0.271514 | [-0.419531, -0.156464] | 3 |
| fixed | lambda1__gain | maddpg_bandit − random | -0.226381 | [-0.392816, -0.112929] | 3 |
| fixed | lambda1__gain | maddpg_bandit − reinforce | -0.771944 | [-1.09799, -0.299406] | 3 |
| fixed | lambda1__gain | mappo − es | 0.487836 | [0.0817371, 0.743223] | 3 |
| fixed | lambda1__gain | mappo − random | 0.53297 | [0.110938, 0.803833] | 3 |
| fixed | lambda1__gain | mappo − reinforce | -0.0125938 | [-0.0682172, 0.0457626] | 3 |
| fixed | lambda1__gain | ppo − es | 0.495768 | [0.107865, 0.836759] | 3 |
| fixed | lambda1__gain | ppo − random | 0.540902 | [0.146228, 0.915024] | 3 |
| fixed | lambda1__gain | ppo − reinforce | -0.00466137 | [-0.13409, 0.143252] | 3 |
| fixed | lambda1__gain | reinforce − es | 0.50043 | [0.142517, 0.734359] | 3 |
| fixed | lambda1__gain | reinforce − random | 0.545563 | [0.172573, 0.792483] | 3 |
| fixed | lambda1__gain | sequential_ppo − es | 0.539729 | [0.118767, 0.8045] | 3 |
| fixed | lambda1__gain | sequential_ppo − random | 0.584862 | [0.148327, 0.87039] | 3 |
| fixed | lambda1__gain | sequential_ppo − reinforce | 0.0392988 | [-0.0242309, 0.107778] | 3 |
| fixed | lambda5__gain | bestof_ppo − es | 0.438315 | [0.110302, 0.676427] | 3 |
| fixed | lambda5__gain | bestof_ppo − random | 0.465325 | [0.155046, 0.702083] | 3 |
| fixed | lambda5__gain | bestof_ppo − reinforce | 0.0227186 | [-0.0205033, 0.0702533] | 3 |
| fixed | lambda5__gain | maddpg_bandit − es | -0.32434 | [-0.400783, -0.239803] | 3 |
| fixed | lambda5__gain | maddpg_bandit − random | -0.297329 | [-0.393917, -0.201747] | 3 |
| fixed | lambda5__gain | maddpg_bandit − reinforce | -0.739936 | [-0.983301, -0.35002] | 3 |
| fixed | lambda5__gain | mappo − es | 0.364696 | [-0.0344196, 0.625089] | 3 |
| fixed | lambda5__gain | mappo − random | 0.391706 | [0.00688407, 0.636106] | 3 |
| fixed | lambda5__gain | mappo − reinforce | -0.0509006 | [-0.156698, 0.0760482] | 3 |
| fixed | lambda5__gain | ppo − es | 0.41213 | [0.0928267, 0.657329] | 3 |
| fixed | lambda5__gain | ppo − random | 0.43914 | [0.137958, 0.683336] | 3 |
| fixed | lambda5__gain | ppo − reinforce | -0.00346653 | [-0.0480899, 0.049061] | 3 |
| fixed | lambda5__gain | reinforce − es | 0.415596 | [0.122041, 0.630501] | 3 |
| fixed | lambda5__gain | reinforce − random | 0.442607 | [0.160484, 0.651602] | 3 |
| fixed | lambda5__gain | sequential_ppo − es | 0.420839 | [0.0582344, 0.646219] | 3 |
| fixed | lambda5__gain | sequential_ppo − random | 0.44785 | [0.102622, 0.667995] | 3 |
| fixed | lambda5__gain | sequential_ppo − reinforce | 0.00524291 | [-0.0691132, 0.0912441] | 3 |

## 独立检查范围

协议、源码、依赖及产物 SHA256；CUDA 执行；正式更新及调用预算；有限正线长；坐标窗口、pin 间距、位移限制；运输成本和攻击 gain 独立重算；nominal topology 重新计算 REST 线长。

actual_call_log_totals 同时保留正式预算、已完成重试额外调用、以及未配对的中断开始事件；未配对请求行数不能当成已完成评分。任一任务缺失、正式预算不符、配对失败或绑定失败均不能标记 DONE。此报告不会启动、恢复或修改任何训练任务。

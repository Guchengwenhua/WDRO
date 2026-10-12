# Checkpoint bank versus GeoSteiner reference training

All differences are arm minus Geo on the same held-out instance IDs; negative means lower true-reference regret. PRIMARY comparison is final update200; validation-selected best is secondary and may select the unchanged step0 parent. Identical best-parent results do not show equivalent learning. Both use the original cached reference, not the bank denominator.

| Arm | Seed | Final200 delta [95% instance CI] | Best selected step | Best delta [95% instance CI] | Training Geo nets | Teacher forward nets | Teacher seconds | Update seconds | Successful job seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo | 101 | +0.00000000 [+0.00000000, +0.00000000] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 51200 | 0 | 0.00 | 87.32 | 132.94873428344727 |
| k1 | 101 | -0.00045681 [-0.00093431, -0.00003917] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 51200 | 6.36 | 65.76 | 79.04838252067566 |
| k3 | 101 | -0.00003058 [-0.00056209, +0.00048769] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 153600 | 19.24 | 79.46 | 93.51573252677917 |
| k5 | 101 | +0.00026988 [-0.00027730, +0.00077253] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 256000 | 32.21 | 92.12 | 106.78029084205627 |
| refresh3 | 101 | -0.00003058 [-0.00056209, +0.00048769] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 153600 | 19.44 | 80.10 | 95.06356763839722 |
| geo | 102 | +0.00000000 [+0.00000000, +0.00000000] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 51200 | 0 | 0.00 | 85.69 | 99.43662595748901 |
| k1 | 102 | +0.00034940 [-0.00014521, +0.00088047] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 51200 | 6.53 | 67.98 | 81.28870868682861 |
| k3 | 102 | +0.00007303 [-0.00036443, +0.00047342] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 153600 | 19.51 | 79.72 | 93.49064493179321 |
| k5 | 102 | +0.00060573 [+0.00014057, +0.00109928] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 256000 | 32.48 | 92.66 | 107.27055788040161 |
| refresh3 | 102 | +0.00007303 [-0.00036443, +0.00047342] | 0 | +0.00000000 [+0.00000000, +0.00000000] | 0 | 153600 | 17.92 | 76.71 | 92.13578271865845 |

| Arm | Observed seeds | Best delta mean [range] | Final delta mean [range] |
|---|---:|---:|---:|
| geo | 2 | +0.00000000 [0.0, 0.0] | +0.00000000 [0.0, 0.0] |
| k1 | 2 | +0.00000000 [0.0, 0.0] | -0.00005370 [-0.00045681183803155463, 0.0003494030890748578] |
| k3 | 2 | +0.00000000 [0.0, 0.0] | +0.00002122 [-3.0584026799954566e-05, 7.30275882898665e-05] |
| k5 | 2 | +0.00000000 [0.0, 0.0] | +0.00043780 [0.00026988172826160795, 0.0006057270677202878] |
| refresh3 | 2 | +0.00000000 [0.0, 0.0] | +0.00002122 [-3.0584026799954566e-05, 7.30275882898665e-05] |

Confidence intervals use 500 bootstrap replicates of instances, stratified by the four fixed groups (1024 observed instances per completed comparison). They describe instance uncertainty conditional on each trained model, not seed-to-seed uncertainty. Seed summaries report only actual available seeds; two seeds are too few for a strong training-variance claim. No equivalence margin was specified, so these results cannot establish statistical equivalence.

Offline Geo-arm diagnostic compares its actor against a separate K3 teacher bank; its agreement score is not a measure of the exact reference agreeing with itself. Other arms use their own banks, so different diagnostic banks must be acknowledged. Training Geo nets exclude the offline candidate diagnostic (288 oracle nets per completed arm) and cached validation/test labels. Teacher-forward nets are nominal scored nets; fixed-batch padding is not added to that counter. Teacher seconds include teacher inference and CPU REST length evaluation. Update seconds exclude periodic validation, test and offline diagnostic; successful queue seconds include the successful process run and can exclude earlier interrupted segments. Historical teachers and validation remain oracle-assisted.

Incomplete: none

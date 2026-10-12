# Checkpoint-bank exploratory results

Degree 10; 200 continued updates; batch 64. Original GeoSteiner cached test references, 256 fixed instances per group. All values below are measured means; incomplete arms remain explicitly marked.

| Arm | Seed | Status | Initial mean regret | Best mean regret | Change | Best worst OOD group | Bank gap | Proxy bias | f argmax agreement |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| geo | 101 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00059461 | -0.00059557 | 0.8750 |
| k1 | 101 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00234915 | -0.00234915 | 0.2188 |
| k3 | 101 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00059461 | -0.00059557 | 0.8750 |
| k5 | 101 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00052715 | -0.00052761 | 0.8750 |
| refresh3 | 101 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00059461 | -0.00059557 | 0.8750 |
| geo | 102 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00059461 | -0.00059557 | 0.8750 |
| k1 | 102 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00234915 | -0.00234915 | 0.2188 |
| k3 | 102 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00059461 | -0.00059557 | 0.8750 |
| k5 | 102 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00052715 | -0.00052761 | 0.8750 |
| refresh3 | 102 | complete | 0.00234418 | 0.00234418 | +0.00000000 | 0.00262800 | 0.00059461 | -0.00059557 | 0.8750 |

These are exploratory continuation results, not a replication of the 2000-update/batch-256 formal protocol. Historical teachers and cached validation labels are oracle-assisted. A negative change means lower independently referenced test error. Geo arm diagnostic uses K3 only for offline comparison. No claim of statistical significance is made from this table.

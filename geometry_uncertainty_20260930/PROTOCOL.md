# REST geometric conditional uncertainty, 2026-09-30

Frozen exploratory study, not a confirmatory test of generalization. No engineering tests or extra validation commands are authorized or run.

## Frozen inputs and estimands
Defaults: d30 seed102 ERM M0900 and WDRO M1118, catalog selection by method/seed rather than test ranking. Precision_v2/test datasets D0218, D0223, D0228, D0233; 512 instances per family, selected by seed20260930 independently of reference labels. Known reference is used only after scores and selection rules are fixed. These are historical test corpora, not a new independent holdout; no metric hyperparameter fitting is performed. Neither checkpoint is claimed to have oracle-free training history.

Actor has p(i|history) for the unvisited first endpoint, followed by p(j,o|i,history) over visited second endpoint and orientation. Hooks observe actual greedy first i and all second logits. Scores are CONDITIONAL diagnostics; they are not full joint entropy p(i)p(j,o|i). No n-fold first-endpoint enumeration. Start-node entropy is excluded. First-endpoint normalized raw entropy is an additional separate control.

For p over legal conditional actions, K=2(step+1), natural logarithm, temperature1:
Hnorm=-sum(p log p)/log K; maxp=max p.
Q=sum(pa pb d_ab).
G=-0.5 sum(pa pb d_ab(log pa+log pb)).
C=sum(pa d(a,a_decode)).
Zero-probability terms contribute zero, log implementation clips solely to avoid log0.
maxp is reported as confidence and its LOWEST values select uncertainty; all other scores select highest.

Ordered topology (u,v) draws vertical at u then horizontal at v, corner (x_u,y_v). For each candidate actual L path take arclength fractions0,1/2,1. d is minimum forward/reversed mean Manhattan distance between three points, divided by input bbox width+height. No independent x/y scaling. Reversed identical path therefore contributes zero, while distinct L orientations can differ. Raw action entropy keeps state-distinct actions; geometric distances give identical sampled paths zero. No geometric-deduplicated entropy claim.

Inference uses frozen FP32 eval greedy, batch256 with last-batch repetition, TF32 off, same implementation and length metric as frozen cross-dataset benchmark. No old source modifications; forward hooks merely observe logits. Exact conditional pairwise geometry is O((2n)^2) per step, restricted to n30; full joint O(n4) is not evaluated.

## Analysis and hard search
Per-instance mean scores and per-instance early/middle/late step averages. Pearson and Spearman versus true regret length/reference-1; within each model and geometry family, not pooled. 500 instance-bootstrap percentile95% intervals, never treat steps as independent samples. Top10% precision uses true-regret90th percentile; ties can make prevalence above10%, and actual prevalence is reported. Constant correlation returns null with explanation. This exploratory multi-metric study does not claim multiplicity-adjusted significance.

Fixed-budget hard-instance selection: disjoint seeded blocks of8 and16 from the same frozen pool; every metric sees identical candidates. Random selection is shared across metrics, oracle true-error maximum is offline upper bound. Selected true regret and paired delta versus random, with block-bootstrap intervals. THIS IS OFFLINE HARD SELECTION, NOT a perturbation attack or WDRO inner maximization. No newly perturbed samples or unsupported error inference from raw length.

## Execution and resume
All code/output lives under this directory. Artifacts are small (~tens of MB), no new attached-disk directory is needed. Model weights remain unchanged. Each batch is atomically committed to NPZ; restart skips committed shards, reconstructs statistics, and checks frozen catalog checkpoint size/mtime plus identical run config. Inference is deterministic and each subsampling/bootstrap stream derives from fixed seed; there is no optimizer state. No new dependencies, no hashing. Forward probabilities need not be saved in full; step summaries retain candidate-control resolution through step index. Failed or incomplete files are not counted as complete.

Run only after supervisor checks current disk/GPU and chooses shared GPU allocation. GPU fraction0.10 and one CPU thread. launch.sh uses tmux, saves command and log. Do not overwrite an existing tmux session. Long job handoff is separate from scientific report.

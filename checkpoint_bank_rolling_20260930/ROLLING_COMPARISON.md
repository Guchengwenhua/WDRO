# Rolling-bank continuation supplement

Fixed rule: historical top1 anchor plus two most recent distinct completed-window snapshots, updated after50/100/150 for the NEXT window. Initial empty slots filled by historical teachers. Not global validation top3. Frozen independent eval copies; no online new oracle labels. The first refresh3 policy legitimately rejected all candidates; this supplement does not reinterpret that result.

|Seed|Final200 mean regret %|Delta vs Geo pp [95% CI]|Delta vs fixedK3 pp [95% CI]|Actual changes|Best step|Refresh seconds|Update seconds|Mean negative-target fraction|
|---|---:|---|---|---:|---:|---:|---:|---:|
|101|0.302789|-0.011909 [-0.066753,+0.035951]|-0.008850 [-0.051472,+0.037269]|3|0|4.98|79.26|0.03188|
|102|0.295187|-0.004641 [-0.055051,+0.038564]|-0.011943 [-0.056928,+0.027145]|3|0|5.01|79.82|0.03109|

Seed101 actual membership transitions:

- After step50, effective step51: formal_hard_mining_d10_s102_u2000, window_s101_u50, formal_wdro_d10_s102_u2000_v1
- After step100, effective step101: formal_hard_mining_d10_s102_u2000, window_s101_u100, window_s101_u50
- After step150, effective step151: formal_hard_mining_d10_s102_u2000, window_s101_u150, window_s101_u100

Seed102 actual membership transitions:

- After step50, effective step51: formal_hard_mining_d10_s102_u2000, window_s102_u50, formal_wdro_d10_s102_u2000_v1
- After step100, effective step101: formal_hard_mining_d10_s102_u2000, window_s102_u100, window_s102_u50
- After step150, effective step151: formal_hard_mining_d10_s102_u2000, window_s102_u150, window_s102_u100

500 paired instance bootstrap replicates stratified by4 families;1024 existing independent-reference test instances per comparison, conditional on each model, not seed-level confidence. Primary is final200; best can return unchanged step0 and cannot establish training benefit. Negative targets are retained; original ReLU critic unchanged. Refresh walltime includes periodic validation and snapshot write/load, and is separate from update time. Detailed call accounting and exact membership in JSON. No long-run equivalence claim.

Incomplete seeds: []

Both final means remain above unchanged initial0.234418%; best-step0 equality cannot establish learning equivalence. Two successful runs took94.15s and94.85s, queue189.00s. Training Geo calls0 in both; teacher-forward nets153600 each; offline288 Geo calls each; cached validation/test labels reused. Mean negative-target fraction is an observed training quantity (3.188% and3.109%), not a claim that all K1 training targets vanish.

## Baseline fidelity audit
# Pilot-to-formal mapping
Original stage6/trainer.py train_wdro calls stage5/search.py search_batch with32+nominal candidates and exact_rescore_top_k4; the shortlist is not a newly introduced pilot speed shortcut. stage4/dual.py select_wdro and update_lambda implement regret-minus-lambda Hungarian meanL1 with q75 rho and rate1. Pilot imports those dual functions and evaluates reference at each candidate Z.
Pilot run_bank.py intentionally loads hard_s102 best actor+critic+Adam, batches64,200updates,validation50,new stateless sampling namespace. Original stage6 loads legacy actor and fresh critic/Adam, batch256,2000updates,validation100. Thus pilot Geo is a reimplementation of the scientific update in a continued-training setting, not original end-to-end training reproduction.
Actor stochastic REINFORCE target C/reference-1 and detached advantage match. Negative bank targets retained; original ReLU critic unchanged. No evidence of a denominator/sign/eval-mode bug explains all final degradation. Optimizer constructor1e-4 is overwritten by parent state5e-5.
Finite loss checks in pilot occurred after stepping; formal finite gradients/lambda guards were missing. No recorded nonfinite loss in completed pilot; no evidence of affected results. New rolling implementation places finite loss/gradient/lambda guards before optimizer update. This guard changes no finite valid update.
K1 proxy constant zero was confirmed only for identical initial greedy actor/teacher in the offline diagnostic, not all stochastic training.

## Separately authorized identity-reference small example
Original stage5 search and original stage6 loss expressions versus Geo adapter on same CPU d10/b8 input and RNG: support,f,reference,Z,loss,lambda,actor and critic one-update state all equal; loss=-0.0032737297005951405. This is a local adapter check, not a complete2000-step equivalence proof. Larger original-initialization study is separate and has no completed results yet.

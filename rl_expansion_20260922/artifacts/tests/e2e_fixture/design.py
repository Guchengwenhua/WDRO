"""Materialize the preregistered expansion; never runs training."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def config(**changes):
    return dict(dict(width=128, depth=3, inner=512, batch=64, lr=5e-5,
                     schedule='constant', optimizer='adam', weight_decay=0.,
                     freeze_bn=False, advantage_ema=False, augmentation=False,
                     objective='erm', eta=.01, uniform_mix=0., normalize_risk=False), **changes)

def main():
    arms = {
        'E50': config(),
        'E20': config(lr=2e-5),
        'E100': config(lr=1e-4),
        'COS': config(schedule='cosine'),
        'WD': config(optimizer='adamw', weight_decay=1e-4),
        'BNF': config(freeze_bn=True),
        'ADV': config(advantage_ema=True),
        'D4': config(augmentation=True),
        'GD': config(objective='group'),
        'GDF': config(objective='group', uniform_mix=.2),
        'GDS': config(objective='group', eta=.001),
        'GDN': config(objective='group', normalize_risk=True),
        'GDNF': config(objective='group', normalize_risk=True, uniform_mix=.2),
        'B256': config(batch=256),
    }
    architectures = {
        'BASE': config(schedule='original'),
        'WIDE': config(width=256, inner=1024, schedule='original'),
        'DEEP': config(depth=6, schedule='original'),
    }
    p = dict(experiment='rl_expansion_20260922_v1', degree=30,
             screening_seeds=[101,102,103], new_seeds=[104,105], architecture_seeds=[201,202,203],
             arms=arms, architectures=architectures, parent_step=12288, screen_target=20480,
             final_target=49152, architecture_final_target=32768, validation_every=1024,
             checkpoint_every=256, val_per_group=2048, calibration_per_group=256,
             test_per_group=8192, secondary_degrees=[20,40], secondary_test_per_group=2048,
             seed_namespace='budget-v1/train', data_namespace='rl-expansion-20260922-v1',
             gradient_clip=1., advantage_ema_beta=.99, advantage_scale_floor=.05,
             cosine_start=12288, cosine_end=49152, cosine_min_lr=5e-6,
             selection_mean_tolerance=.01, promotion_threshold_percent=.2,
             maximum_promoted_variants=2,
             promotion_rule='fixed endpoint: best mean with worst <= control*1.01; best worst with mean <= control*1.01; >=0.2% aggregate gain and >=2/3 seed gains; unique eligible variants; B256 excluded',
             architecture_rule='best WIDE/DEEP endpoint worst with mean <= BASE*1.01; >=0.2% and >=2/3 seeds; if eligible extend both winner and BASE to32768',
             best_rule='validation worst minimum; mean <= own stage-start mean*1.01; earliest exact tie',
             test_gate='all planned and promoted training complete; immutable global model-selection manifest before opening test arrays',
             primary_slots=['budget_mean','budget_worst','promoted_mean_winner','promoted_worst_winner','architecture_winner'],
             bootstrap_replicates=10000, bootstrap_seed=9222026, primary_ci=.99,
             primary_min_improvement_percent=.5, max_workers=8, cpu_threads_per_worker=3,
             min_disk_gib=20, max_branch_gib=28, reference_solver_calls=0,
             evaluation='single greedy forward; raw supplied-topology REST span length; no oracle',
             capacity_comparison='same samples and updates; larger models are not FLOP matched',
             batch_comparison='B256 step14336 vs E50 step20480 equal new samples; B256 step20480 vs E50 step20480 equal new updates',
             max_new_updates=1200128)
    path=ROOT/'protocol.json'
    if path.exists(): raise RuntimeError('Protocol already materialized')
    path.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__': main()

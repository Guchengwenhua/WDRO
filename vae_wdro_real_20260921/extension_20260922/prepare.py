"""Read-only reuse of the first experiment, with new shared 1024-step streams."""
import os
import shutil
import sqlite3
import time
from exp_common import *

PREV=Path(CFG['previous_artifacts'])

def link(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source)

def main():
    if (ART/'done'/'prepare.json').exists():return
    start=time.monotonic();ART.mkdir(parents=True,exist_ok=True)
    for name in ['data.npz','training_centers.npz','splits.csv','split_summary.json','rho.json','calibration_raw_costs.npz']:
        link(PREV/name,ART/name)
    data=load_data();centers=np.load(ART/'training_centers.npz')['indices']
    assert len(centers)==2810 and np.all(data['split'][centers]=='A_train')
    for family in np.unique(data['family_ids']):
        if len(set(data['split'][data['family_ids']==family]))!=1:raise ValueError('cross-family leakage')
    for seed in CFG['seeds']:
        seq=np.random.default_rng(seed).integers(0,len(centers),size=(CFG['training']['updates'],64))
        old=np.load(PREV/'indices'/f'seed{seed}.npz')['indices']
        if not np.array_equal(seq[:256],old):raise ValueError('first 256 center steps differ')
        atomic_npz(ART/'indices'/f'seed{seed}.npz',indices=seq)
    mapping={'raw_wdro':'raw_wdro','legacy_shell':'vae_shell_wdro'}
    for new,old in mapping.items():
        for suffix in ['.npz','.geometry.npz','.geometry.json','.reference.json']:
            link(PREV/'pools'/(old+suffix),ART/'pools'/(new+suffix))
        mark_done(new+'_reference',reused_from=str(PREV/'pools'/(old+'.npz')),new_reference_requests=0)
    # Copy only the small coordinate cache through SQLite's consistent backup.
    if not (ART/'reference_cache.sqlite').exists():
        source=sqlite3.connect(f'file:{PREV}/reference_cache.sqlite?mode=ro',uri=True)
        target=sqlite3.connect(ART/'reference_cache.sqlite');source.backup(target);source.close();target.close()
    for seed in CFG['recipe_selection']['seeds']:
        for method in CFG['recipe_selection']['methods']:
            old=PREV/'runs'/f'{method}_s{seed}';v=read_json(old/'validation.json')
            folder=ART/'diagnostics'/f'standard_1e5_{method}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
            atomic_json(folder/'summary.json',dict(recipe='standard_1e5',method=method,seed=seed,
                validations=v,update=256,new_training_seconds=0.,reused_from=str(old),device_type='cpu'))
            mark_done(f'diag_standard_1e5_{method}_s{seed}',reused_from=str(old),update=256)
    cfg=dict(CFG);cfg.update(started_at=utc(),status='running',train_centers=len(centers))
    atomic_json(ART/'config.executed.json',cfg)
    mark_done('prepare',seconds=time.monotonic()-start)

if __name__=='__main__':main()

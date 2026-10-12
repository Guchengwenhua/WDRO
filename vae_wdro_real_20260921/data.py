"""Family split, ID-aligned references and exact-byte cross-split deduplication."""
from collections import Counter
import time
import numpy as np
from exp_common import *

def prepare():
    if (ART / 'done' / 'data.json').exists(): return
    start = time.monotonic()
    with np.load(CFG['input_npz'], allow_pickle=False) as src:
        data = {k: src[k] for k in ['pins', 'net_ids', 'design_ids', 'dataset_ids', 'family_ids']}
    for k in ['net_ids', 'design_ids', 'dataset_ids', 'family_ids']: data[k] = data[k].astype(str)
    if len(set(data['net_ids'])) != len(data['net_ids']): raise ValueError('duplicate net IDs')
    with np.load(CFG['reference_npz'], allow_pickle=False) as ref:
        ids = ref['net_ids'].astype(str)
        if len(set(ids)) != len(ids): raise ValueError('duplicate reference IDs')
        lookup = dict(zip(ids, ref['lstar'], strict=True))
        data['lstar'] = np.array([lookup[n] for n in data['net_ids']], dtype=np.float64)
    if not np.isfinite(data['lstar']).all() or np.any(data['lstar'] <= 0): raise ValueError('bad references')
    data['pins'] = np.stack([canonical(x) for x in data['pins']])
    for begin in range(0, len(data['pins']), 256): check_pins(data['pins'][begin:begin+256])
    assignment = {}
    for split in ['A_train', 'A_cal', 'A_val', 'B_test']:
        for family in CFG['splits'][split]:
            if family in assignment: raise ValueError('family assigned twice')
            assignment[family] = split
    splits = []
    for family, dataset in zip(data['family_ids'], data['dataset_ids'], strict=True):
        if dataset in CFG['splits']['C_external_datasets']:
            if family in assignment: raise ValueError('external family overlaps ISPD2008')
            splits.append('C_external')
        elif dataset == 'ispd2008_global_routing': splits.append(assignment[family])
        else: raise ValueError('unexpected data source ' + dataset)
    splits = np.array(splits)
    counts = dict(Counter(splits))
    if counts != CFG['splits']['expected_rows_before_cross_split_dedup']: raise ValueError(counts)
    for family in np.unique(data['family_ids']):
        if len(set(splits[data['family_ids'] == family])) != 1: raise ValueError('family leakage')
    rank = dict(A_train=0, A_cal=1, A_val=2, B_test=3, C_external=3)
    best_rank = {}; split_members = {}; keys = []
    for x, split in zip(data['pins'], splits, strict=True):
        key = x.tobytes(); keys.append(key)
        best_rank[key] = max(best_rank.get(key, -1), rank[split])
        split_members.setdefault(key, set()).add(str(split))
    keep = np.array([rank[s] == best_rank[key] for key, s in zip(keys, splits, strict=True)])
    rows = []
    for i, split in enumerate(splits):
        rows.append(dict(net_id=data['net_ids'][i], design_id=data['design_ids'][i], family_id=data['family_ids'][i],
                         dataset_id=data['dataset_ids'][i], split=split, retained=bool(keep[i]),
                         reason='' if keep[i] else 'exact_geometry_in_higher_priority_split'))
    write_csv(ART / 'splits.csv', rows)
    removed = dict(Counter(splits[~keep])); retained = dict(Counter(splits[keep]))
    test_keys = [k for k, s in zip(keys, splits, strict=True) if rank[s] == 3]
    summary = dict(before=counts, removed=removed, retained=retained,
                   test_duplicate_rows=len(test_keys)-len(set(test_keys)),
                   cross_B_C_equal_geometries=sum({'B_test','C_external'}.issubset(v) for v in split_members.values()),
                   note='B/C均为测试级；测试内原始重复保留。仅完整float32坐标字节相等去重，无近似几何合并。')
    atomic_json(ART / 'split_summary.json', summary)
    data = {k: v[keep] for k,v in data.items()}; data['split'] = splits[keep]
    atomic_npz(ART / 'data.npz', **data)
    centers = np.flatnonzero(data['split'] == 'A_train')
    atomic_npz(ART / 'training_centers.npz', indices=centers)
    for seed in CFG['seeds']:
        seq = np.random.default_rng(seed).integers(0, len(centers), size=(CFG['training']['updates'],CFG['training']['batch_size']))
        atomic_npz(ART / 'indices' / f'seed{seed}.npz', indices=seq)
    actual = dict(CFG); actual.update(status='running', started_at=utc(), train_center_count=len(centers),
        input_order='canonical float32 lexicographic x,y', center_cap_applied=False,
        optional_attack_scheduled=False, optional_extension_scheduled=False)
    atomic_json(ART / 'config.executed.json', actual)
    mark_done('data', seconds=time.monotonic()-start, **summary)

if __name__ == '__main__': prepare()

"""Cross-role canonical geometry audit including every paid fixed-search point.

Reads existing NPZ coordinates only; performs no model or reference evaluation.
Duplicate nets within a role are permitted and counted rather than discarded
from the input-row accounting. Cross-role matches retain first-file provenance.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path('/remote-home/Guilin/DRO/Codex/improvements/critic_search_20260915')
EXP=ROOT/'artifacts'
OUT=Path(__file__).resolve().parent
SEEDS=(101,102,103)
VARIANTS=('refresh32_raw','refresh8_raw','refresh32_ema','refresh8_ema')


def main():
    started=time.monotonic()
    roles={name:{} for name in ('train','validation','test')}
    records=[]
    def add(role,path,key,category):
        with np.load(path,allow_pickle=False) as stored:
            values=np.asarray(stored[key],dtype=np.float32)
        if values.shape[-2:]!=(30,2):raise ValueError(f'bad geometry shape {path}:{key}')
        points=np.ascontiguousarray(values.reshape(-1,30,2))
        if not np.isfinite(points).all():raise ValueError(f'nonfinite geometry {path}:{key}')
        order=np.lexsort((points[:,:,1],points[:,:,0]),axis=1)
        canonical=np.ascontiguousarray(np.take_along_axis(points,order[:,:,None],axis=1))
        first_id=len(records)
        local=set()
        for i,p in enumerate(canonical):
            h=hashlib.sha256(p.tobytes()).digest()
            local.add(h)
            roles[role].setdefault(h,(first_id,i))
        records.append(dict(role=role,category=category,path=str(path),npz_key=key,
                            input_shape=list(values.shape),rows=len(points),unique_in_file=len(local)))

    # Main Actor training centers and all paid online surrogate labels are train.
    for seed in SEEDS:
        for variant in VARIANTS:
            task=EXP/'training'/f'{variant}_d30_s{seed}'
            for path in sorted((task/'inputs').glob('*.npz')):
                family=path.name.split('__',1)[0]
                role={'train':'train','validation':'validation','evaluation':'test'}[family]
                add(role,path,'pins','joint_'+family+'_inputs')
            for path in sorted((task/'search').glob('u*.npz')):
                add('train',path,'evaluated_pins','joint_all_paid_search_candidates')

    # Fixed-search validation includes every candidate used to choose configs.
    for name,role in [('validation_s101','validation')]+[(f'test_s{s}','test') for s in SEEDS]:
        task=EXP/'fixed'/name
        for path in sorted((task/'inputs').glob('*.npz')):
            add(role,path,'pins','fixed_'+role+'_centers')
        for path in sorted((task/'evaluation').glob('*.npz')):
            add(role,path,'support','fixed_'+role+'_all_paid_support')
        if role=='test':
            for pool,pool_role in [('uniform','train'),('search_region','train'),('shared_validation','validation')]:
                add(pool_role,task/'surrogates/data'/pool/'labels.npz','pins','surrogate_'+pool+'_labels')
        print(name,'processed',len(records),'arrays',flush=True)

    intersections=[]
    names=list(roles)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            common=roles[left].keys() & roles[right].keys()
            details=[]
            for h in sorted(common):
                a,ai=roles[left][h];b,bi=roles[right][h]
                details.append(dict(canonical_sha256=h.hex(),left_source=records[a],left_flat_index=ai,
                                    right_source=records[b],right_flat_index=bi))
            intersections.append(dict(left=left,right=right,count=len(common),matches=details))
    counts={}
    for row in records:
        counts.setdefault(row['category'],dict(files=0,rows=0))
        counts[row['category']]['files']+=1;counts[row['category']]['rows']+=row['rows']
    status='pass' if all(x['count']==0 for x in intersections) else 'fail'
    result=dict(status=status,at=datetime.now(timezone.utc).isoformat(),
        canonical_rule='cast float32; lexsort x then y; SHA256 contiguous sorted coordinate bytes',
        treatment='same-role duplicates allowed; every actual fixed-search support candidate included',
        role_unique_nets={k:len(v) for k,v in roles.items()},
        role_total_rows={k:sum(r['rows'] for r in records if r['role']==k) for k in roles},
        sources_by_category=counts,intersections=intersections,source_arrays=records,
        elapsed_seconds=time.monotonic()-started,review_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=OUT/'expanded_data_isolation.json';temp=path.with_suffix('.json.partial')
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_arrays',)},ensure_ascii=False,indent=2),flush=True)
    return 0 if status=='pass' else 1


if __name__=='__main__':raise SystemExit(main())

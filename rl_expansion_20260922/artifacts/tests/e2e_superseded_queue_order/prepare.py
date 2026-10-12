"""Copy accepted pure-RL parents and freeze entirely new evaluation coordinates."""
import shutil
import numpy as np
import geometry as G
from common import ROOT,GROUPS,read,now
from scheduler import resource_snapshot

def main():
    p=read(ROOT/'protocol.json'); out=ROOT/'artifacts'
    if (out/'data/manifest.json').exists(): raise RuntimeError('Already frozen')
    old=ROOT.parent/'rl_budget_scaling_20260921'; acceptance=old/'artifacts/acceptance_20260922/complete.json'
    a=read(acceptance)
    if a['status']!='PASS_FOR_FROZEN_THREE_SEED_EXPLORATORY_PROTOCOL': raise RuntimeError('Parent not accepted')
    lineage={}
    for seed in p['screening_seeds']:
        run=old/'artifacts/runs'/f'ME_s{seed}'; c=read(run/'complete.json'); name='step12288.pt'
        if c['step']!=12288 or G.sha256(run/name)!=c['artifacts'][name]: raise RuntimeError('Parent mismatch')
        dest=out/'inputs'/f'ME_s{seed}.pt'; shutil.copyfile(run/name,dest)
        lineage[str(seed)]=dict(path=str(dest.relative_to(ROOT)),sha256=G.sha256(dest),
            source=str(run/name),source_completion_sha256=G.sha256(run/'complete.json'),acceptance_sha256=G.sha256(acceptance))
    G.atomic_json(out/'inputs/lineage.json',lineage)
    seen=set()
    for name in ('validation','test'):
        with np.load(old/'artifacts/data'/f'{name}.npz') as z:
            for g in z.files:
                for row in z[g]: seen.add(row.tobytes())
    hashes={}; counts={}
    specs=[('calibration',p['calibration_per_group'],30),('validation',p['val_per_group'],30),('test',p['test_per_group'],30)]
    specs += [(f'test_degree{d}',p['secondary_test_per_group'],d) for d in p['secondary_degrees']]
    for split,count,degree in specs:
        arrays={g:G.make_data(count,degree,g,f"{p['data_namespace']}/{split}/{g}",0,True) for g in GROUPS}
        for group,rows in arrays.items():
            for row in rows:
                if row.tobytes() in seen: raise RuntimeError('Dataset overlap')
                seen.add(row.tobytes())
        G.atomic_npz(out/'data'/f'{split}.npz',**arrays); hashes[f'{split}.npz']=G.sha256(out/'data'/f'{split}.npz')
        counts[split]={g:len(x) for g,x in arrays.items()}
    G.atomic_json(out/'data/manifest.json',dict(artifacts=hashes,counts=counts,created=now(),
        protocol_sha256=G.sha256(ROOT/'protocol.json'),lineage_sha256=G.sha256(out/'inputs/lineage.json'),
        overlap_with_new_splits_or_previous_validation_test=0,test_inference_performed=False))
    G.atomic_json(out/'operations/initial_admission.json',resource_snapshot())
    print('New coordinate data and accepted RL parents frozen',flush=True)

if __name__=='__main__': main()

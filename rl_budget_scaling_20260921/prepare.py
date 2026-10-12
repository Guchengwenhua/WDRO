"""Freeze coordinate datasets and hash-verified pure-RL parents, without inference."""
import json
from pathlib import Path
import shutil
import numpy as np
import geometry as G
from common import ROOT,GROUPS,training_batch,now

def main():
    out=ROOT/'artifacts'
    p=json.loads((ROOT/'protocol.json').read_text())
    if (out/'data/manifest.json').exists():
        raise RuntimeError('Inputs already frozen; do not regenerate')
    lineage={}
    old=ROOT.parent/'rl_only_20260915'
    for seed in p['seeds']:
        src=old/'artifacts/runs'/f'erm_rollout_s{seed}'
        c=json.loads((src/'complete.json').read_text())
        if c['step']!=2048 or c['initial_actor_sha256']!=json.loads((src/'run.json').read_text())['initial_actor_sha256']:
            raise RuntimeError('Unexpected historical parent')
        for name in ('latest.pt','progress.jsonl','run.json'):
            if G.sha256(src/name)!=c['artifacts'][name]: raise RuntimeError('Historical artifact changed')
        dest=out/'inputs'/f's{seed}'
        dest.mkdir(parents=True,exist_ok=True)
        for name in ('latest.pt','complete.json','run.json','progress.jsonl'):
            shutil.copyfile(src/name,dest/name)
        lineage[str(seed)]=dict(source=str(src),source_completion_sha256=G.sha256(src/'complete.json'),
                               parent_sha256=G.sha256(dest/'latest.pt'),initial_actor_sha256=c['initial_actor_sha256'])
    G.atomic_json(out/'inputs/lineage.json',lineage)
    hashes={}; seen=set(); counts={}
    for split,count in [('validation',p['val_per_group']),('test',p['test_per_group'])]:
        arrays={g:G.make_data(count,30,g,f'budget-20260921-v1/{split}/{g}',0,True) for g in GROUPS}
        for g,a in arrays.items():
            for row in a:
                key=row.tobytes()
                if key in seen: raise RuntimeError('Duplicate validation/test net')
                seen.add(key)
        G.atomic_npz(out/'data'/f'{split}.npz',**arrays)
        hashes[f'{split}.npz']=G.sha256(out/'data'/f'{split}.npz')
        counts[split]={g:len(a) for g,a in arrays.items()}
    for seed in p['seeds']:
        for arm in ('U','M'):
            for step in (1,2048,2049,4096,8192,12288):
                a,_=training_batch(arm,seed,step)
                if any(row.tobytes() in seen for row in a): raise RuntimeError('Train probe leakage')
    G.atomic_json(out/'data/manifest.json',dict(created=now(),artifacts=hashes,counts=counts,
        lineage_sha256=G.sha256(out/'inputs/lineage.json'),independent_gaussian_layouts=True,
        exact_split_overlap=0,training_probe_overlap=0,protocol_sha256=G.sha256(ROOT/'protocol.json'),
        test_inference_performed=False))
    print(json.dumps(dict(status='inputs_frozen',hashes=hashes,parents=lineage),indent=2))

if __name__=='__main__': main()

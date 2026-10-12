"""Other discovered fixed datasets, appended before evaluation starts."""
from prepare_data import *

def main():
    if (ART/'data_supplement_complete.json').exists():return
    c=Catalog();c.rows=read(ART/'datasets.json')
    for branch in ('adversarial_v3_20260912','adversarial_v3_1_20260912'):
        seen=set()
        for p in sorted((PROJECT/branch/'experiments/fixed').glob('*/inputs/*.npz')):
            z=arrays(p)
            if p.stem not in DISTS or 'pins' not in z:continue
            key=(p.stem,z['pins'].shape[1])
            if key in seen:continue
            seen.add(key);c.add(f'{branch}/fixed/{p.stem}/d{key[1]}',z['pins'],p)
    for phase in ('test','validation'):
        for p in sorted((PROJECT/f'critic_search_20260915/artifacts/fixed/{phase}_s101/inputs').glob('*.npz')):
            z=arrays(p);c.add(f'critic_search/fixed/{phase}/{p.stem}',z['pins'],p,kind=phase)
    for p in sorted((PROJECT/'critic_search_20260915/artifacts/training/refresh8_raw_d30_s101/inputs').glob('evaluation__*.npz')):
        z=arrays(p);c.add('critic_search/training/'+p.stem,z['pins'],p)
    ps=sorted((PROJECT/'oracle_free_20260915/artifacts/runs').glob('*/initial.npz'))
    if ps:
        z=arrays(ps[0])
        for key,x in z.items():
            if key.endswith('__pins'):c.add('oracle_free/test/'+key,x,ps[0],kind='historical_test')
    seen=set()
    for p in sorted((PROJECT/'rl_prepost_scaleup_20260922/artifacts/runs').glob('*/evaluations/*.npz')):
        z=arrays(p)
        for key,x in z.items():
            if x.ndim==3 and x.shape[-1]==2 and (key.endswith('__nominal_pins') or key in [d+'__pins' for d in DISTS]):
                if key in seen:continue
                seen.add(key);c.add('rl_prepost/'+key,x,p)
        if len(seen)==8:break
    # Preserve the finite, persisted training sets as explicitly seen-data columns.
    for seed in (101,102,103):
        ps=sorted((PROJECT/'papers_wdro_20260915/artifacts/data').glob(f'train_s{seed}_*.npz'))
        if not ps:continue
        xs=[]
        for p in ps:
            z=arrays(p);x=z['pins'];xs.append(x.reshape(-1,x.shape[-2],2))
        c.add(f'papers/seen_training/s{seed}',np.concatenate(xs),ps[0].parent,kind='seen_training',files=[str(p) for p in ps])
    for p in sorted((PROJECT/'vae_wdro_real_20260921/extension_20260922/artifacts/pools').glob('*.geometry.npz')):
        z=arrays(p);x=z['pins']
        c.add('vae_extension/'+p.stem,x.reshape(-1,x.shape[-2],2),p,kind='fixed_attack_pool' if 'attack' in p.stem else 'seen_training_candidate_pool')
    atomic(ART/'datasets.json',c.rows);pack_data(c.rows)
    atomic(ART/'data_supplement_complete.json',dict(columns=len(c.rows),nets=sum(d['count'] for d in c.rows)))

if __name__=='__main__':main()

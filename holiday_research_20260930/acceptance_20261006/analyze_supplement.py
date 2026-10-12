"""Read saved pilot, reference and attack outputs; no new inference."""
from analyze_results import ROOT, OUT, GROUPS, read, save, csvsave
from pathlib import Path
from collections import Counter
import numpy as np

source_reference=[]
for g in GROUPS:
    path=ROOT/f'wdro_precision_v2/experiments/cache/datasets/protocol_v1/test/{g}_d10_s303_n10000.npz'
    meta=read(path.with_suffix('.meta.json'))
    refpath=ROOT/'wdro_precision_v2/experiments/cache/lstar/protocol_v1'/f"{meta['array_bundle_sha256']}.npz"
    with np.load(path) as d,np.load(refpath) as r,np.load(ROOT/f'checkpoint_bank_scale_20260930/artifacts/geo_s101/test_latest/{g}.npz') as z:
        source_reference.append(dict(group=g,n=len(z['regret']),dataset_ids_match=bool(np.array_equal(d['net_ids'],z['net_ids'])),
            reference_ids_match=bool(np.array_equal(r['net_ids'],z['net_ids'])),reference_max_abs_error=float(np.max(np.abs(r['lstar']-z['reference']))),path=str(refpath)))
pilot=[];diagnostic=[]
for folder,arms in [('checkpoint_bank_20260930',['geo','k1','k3','k5','refresh3']),('checkpoint_bank_rolling_20260930',['lagged3'])]:
    for seed in [101,102]:
        for arm in arms:
            p=ROOT/folder/'artifacts'/f'{arm}_s{seed}'
            with np.load(p/'final_test.npz') as z,np.load(ROOT/f'checkpoint_bank_20260930/artifacts/geo_s{seed}/final_test.npz') as ref,np.load(p/'initial_test.npz') as initial:
                vv=[z[g+'_regret'] for g in GROUPS];base=[ref[g+'_regret'] for g in GROUPS]
                pilot.append(dict(arm=arm,seed=seed,complete=read(p/'complete.json')['status'],steps=read(p/'complete.json')['steps'],
                    n=sum(map(len,vv)),mean_pct=100*np.concatenate(vv).mean(),
                    worst_ood_pct=100*max(v.mean() for v in vv[1:]),
                    delta_vs_geo_pp=100*np.mean([np.mean(a-b) for a,b in zip(vv,base)]),
                    initial_mean_pct=100*np.mean([initial[g+'_regret'].mean() for g in GROUPS]),
                    ids_match=all(np.array_equal(z[g+'_ids'],ref[g+'_ids']) for g in GROUPS)))
            with np.load(p/'offline_bank_diagnostic.npz') as z:
                actual=z['current']/z['oracle']-1;proxy=z['current']/z['bank']-1
                diagnostic.append(dict(arm=arm,seed=seed,bank_gap_pct=100*np.mean(z['bank']/z['oracle']-1),
                    proxy_bias_pp=100*np.mean(proxy-actual),argmax_agreement=float(np.mean(z['oracle_f'].argmax(1)==z['bank_f'].argmax(1)))))
csvsave('pilot_metrics.csv',pilot);csvsave('pilot_reference_bias.csv',diagnostic)
attack=ROOT/'geometry_uncertainty_20260930/artifacts/attack';summary=read(attack/'summary.json');arows=[]
refs=[]
for p in sorted(attack.glob('reference_*.npz')):
    with np.load(p) as z:refs.append(z['reference'])
ref=np.stack(refs)
for mid in summary:
    with np.load(attack/mid/'predictions.npz') as z:error=z['length']/ref-1
    sel=read(attack/mid/'selection.json')['selected'];sel['oracle_upper_bound']=(1+error[:,1:].argmax(1)).tolist();sel['nominal']=[0]*len(error)
    rnd=error[np.arange(len(error)),sel['random']]
    for method,idx in sel.items():
        value=error[np.arange(len(error)),idx];s=summary[mid]['all'][method]
        arows.append(dict(model=mid,method=method,n_centers=len(value),mean_pct=100*value.mean(),delta_random_pp=100*(value-rnd).mean(),
            ci_low_pp=100*s['delta_vs_random_ci95'][0],ci_high_pp=100*s['delta_vs_random_ci95'][1],
            delta_nominal_pp=100*(value-error[:,0]).mean(),max_postnorm_linf=s['max_postnorm_linf'],
            summary_mean_error=abs(value.mean()-s['mean_true_regret'])))
csvsave('attack_metrics.csv',arows)
cost={}
for folder in ['checkpoint_bank_20260930','checkpoint_bank_rolling_20260930','checkpoint_bank_scale_20260930','geometry_uncertainty_20260930','geometry_uncertainty_extension_20260930']:
    p=ROOT/folder/'artifacts';cost[folder]=dict(artifact_logical_bytes=sum(f.stat().st_size for f in p.rglob('*') if f.is_file()))
    if (p/'queue_status.json').exists():
        q=read(p/'queue_status.json');start=q.get('start',q.get('started'));end=q.get('end',q.get('finished_time'))
        cost[folder]['queue_elapsed_seconds']=end-start
    elif (p/'complete.json').exists():cost[folder]['elapsed_seconds']=read(p/'complete.json')['elapsed_seconds']
cost['geometry_attack']=dict(elapsed_seconds=read(attack/'complete.json')['elapsed_seconds'])
save('supplement.json',dict(source_reference=source_reference,pilot=pilot,reference_bias=diagnostic,attack=arows,cost=cost))
print('Saved pilot, attack, source reference and cost analyses.')

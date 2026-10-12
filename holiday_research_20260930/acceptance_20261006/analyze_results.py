"""CPU-only analysis of saved holiday results; does not modify experiments."""
import csv
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import numpy as np

ROOT = Path('/remote-home/Guilin/DRO/Codex')
OUT = Path(__file__).resolve().parent
GROUPS = ['uniform', 'mix_gauss', 'edge_bias', 'grid_cluster']
SCORES = ['Hnorm', 'maxp', 'Q', 'G', 'C', 'first_Hnorm']
read = lambda p: json.loads(Path(p).read_text())
def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')
def csvsave(name, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def interval(d, seed=20261006, reps=2000):
    rng=np.random.default_rng(seed); draws=np.zeros(reps)
    for v in d:
        for start in range(0,reps,50):
            ix=rng.integers(0,len(v),(min(50,reps-start),len(v)))
            draws[start:start+len(ix)]+=v[ix].mean(1)/len(d)
    return (100*np.quantile(draws,[.025,.975])).tolist()

def bank():
    root=ROOT/'checkpoint_bank_scale_20260930'; rows=[]; arrays={}; details={}
    for seed in [101,102,103]:
        for arm in ['geo','k3','lagged3']:
            job=f'{arm}_s{seed}';p=root/'artifacts'/job;r=read(p/'logs/result.json')
            evcount=Counter();steps=[];negative=[];lam=[];cost=[];last={};initial=None
            with (p/'logs/events.jsonl').open() as f:
                for line in f:
                    e=json.loads(line);evcount[e['event']]+=1
                    if e['event']=='train':
                        steps.append(e['update']);negative.append(e['negative_target_fraction'])
                        lam.append(e['method_detail']['lambda_after']);cost.append(e['method_detail']['mean_selected_cost'])
                        last=e['reference_accounting']
                    elif e['event']=='validation' and e['update']==0:initial=e['result']['selection']
            details[job]=dict(events=dict(evcount),updates_contiguous=steps==list(range(1,2001)),
                final_accounting=last,negative_target_mean=float(np.mean(negative)),
                lambda_zero_fraction=float(np.mean(np.array(lam)==0)),lambda_max=max(lam),
                mean_selected_cost=float(np.mean(cost)),final_lambda=lam[-1],rho=r['method_state']['rho'],
                initial_validation=initial,best_validation=r['best'],budget=r['method_state']['budget'],
                nan_detected=r['nan_detected'],policy_update_nets=r['policy_update_nets'],
                peak_allocated_mib=r['peak_memory_allocated_bytes']/2**20,
                segment_wall_seconds=r['segment_wall_seconds'],complete=read(p/'complete.json')['status'])
            for kind in ['latest','best']:
                summary=read(p/f'test_{kind}.json');vv=[];err=0.;ids_ok=True;refs_ok=True;group_means={}
                for g in GROUPS:
                    with np.load(p/f'test_{kind}'/f'{g}.npz') as z:
                        v=z['length']/z['reference']-1
                        err=max(err,float(np.max(np.abs(v-z['regret']))))
                        with np.load(root/'artifacts/geo_s101/test_latest'/f'{g}.npz') as ref:
                            ids_ok &= np.array_equal(z['net_ids'],ref['net_ids'])
                            refs_ok &= np.array_equal(z['reference'],ref['reference'])
                        vv.append(v);group_means[g]=float(v.mean())
                arrays[arm,seed,kind]=vv
                rows.append(dict(arm=arm,seed=seed,checkpoint=kind,update=summary['checkpoint_update'],n=sum(map(len,vv)),
                    mean_pct=100*np.mean([v.mean() for v in vv]),p95_pct=100*np.quantile(np.concatenate(vv),.95),
                    worst_ood_pct=100*max(group_means[g] for g in GROUPS[1:]),
                    uniform_pct=100*group_means['uniform'],mix_gauss_pct=100*group_means['mix_gauss'],
                    edge_bias_pct=100*group_means['edge_bias'],grid_cluster_pct=100*group_means['grid_cluster'],
                    regret_formula_max_abs_error=err,paired_ids_match=bool(ids_ok),paired_references_match=bool(refs_ok),
                    summary_mean_abs_error=abs(summary['overall_mean']-np.mean([v.mean() for v in vv]))))
            print('read',job,flush=True)
    contrasts=[]
    for kind in ['latest','best']:
        for arm in ['k3','lagged3']:
            ds=[]
            for seed in [101,102,103]:
                d=[a-b for a,b in zip(arrays[arm,seed,kind],arrays['geo',seed,kind])];ds.append(d)
                ci=interval(d,20261006+seed)
                contrasts.append(dict(arm=arm,checkpoint=kind,seed=str(seed),delta_pp=100*np.mean([v.mean() for v in d]),ci_low_pp=ci[0],ci_high_pp=ci[1]))
            d=[np.stack([x[g] for x in ds]).mean(0) for g in range(4)];ci=interval(d)
            means=[100*np.mean([v.mean() for v in x]) for x in ds]
            contrasts.append(dict(arm=arm,checkpoint=kind,seed='mean_of_3',delta_pp=float(np.mean(means)),ci_low_pp=ci[0],ci_high_pp=ci[1]))
    csvsave('bank_metrics.csv',rows);csvsave('bank_contrasts.csv',contrasts)
    queue=read(root/'artifacts/queue_status.json')
    save('bank_analysis.json',dict(metrics=rows,contrasts=contrasts,training=details,queue=queue,
         ci_scope='2000 paired bootstrap draws within four fixed families, conditional on trained models; mean_of_3 averages models before resampling same IDs; not training-seed confidence; no multiplicity adjustment'))

def geometry():
    root=ROOT/'geometry_uncertainty_extension_20260930/artifacts';meta=read(root/'frozen_metadata.json');cfg=read(root/'config.json')
    result={};cellrows=[]
    for stage in cfg['stages']:
        folder=root/stage['name'];manifest=read(folder/'split_manifest.json');summary=read(folder/'test_summary.json');gate=read(folder/'dev_gate.json')
        allvals={};counts=Counter();id_errors=0;test_ids=set();dev_ids=set();pilot_overlap=set();point_error=0.
        for did,m in manifest.items():
            (test_ids if m['split']=='test' else dev_ids).update(m['net_ids'])
            if m['split']=='test':
                for p in (ROOT/'geometry_uncertainty_20260930/artifacts').glob(f'M*/{did}/*.npz'):
                    with np.load(p) as z:
                        if 'net_ids' in z:pilot_overlap.update(set(z['net_ids'].astype(str)) & set(m['net_ids']))
        for split in ['dev','test']:
            for mid in stage['models']:
                for did in stage[split]:
                    chunks=[]
                    for p in sorted((folder/'predictions'/mid/did).glob('*.npz')):
                        with np.load(p) as z:chunks.append({k:z[k] for k in ['length','mean_scores','net_ids','indices']})
                    length=np.concatenate([z['length'] for z in chunks]);scores=np.concatenate([z['mean_scores'] for z in chunks])
                    ids=np.concatenate([z['net_ids'] for z in chunks]);idx=np.concatenate([z['indices'] for z in chunks]);m=manifest[did]
                    with np.load(meta['datasets'][did]['path']) as z:
                        ref=z['reference'][m['indices']];id_errors+=int(not np.array_equal(ids,z['net_ids'][m['indices']]))
                    id_errors+=int(not np.array_equal(idx,m['indices']))+int(not np.array_equal(ids,np.array(m['net_ids'])))
                    error=length/ref-1;counts[split]+=len(error)
                    rng=np.random.default_rng(cfg['seed']+int(did[1:]));blocks=rng.permutation(len(error)).reshape(-1,16)
                    rnd=rng.integers(0,16,len(blocks));rr=error[blocks];row=np.arange(len(blocks))
                    values={'random':rr[row,rnd],'oracle_upper_bound':rr.max(1)}
                    for j,name in enumerate(SCORES):
                        s=scores[blocks,j];s=-s if name=='maxp' else s;values[name]=rr[row,s.argmax(1)]
                    allvals[split,mid,did]=values
                    if split=='test':
                        stored=summary['cells'][mid+'/'+did]
                        point_error=max(point_error,abs(error.mean()-stored['mean_true_regret']))
                        for name,v in values.items():point_error=max(point_error,abs(v.mean()-stored['selected_mean_regret'][name]))
                        for base in ['Hnorm','maxp','random']:
                            d=values[gate['winner']]-values[base];c=stored['primary_contrasts'][base]
                            point_error=max(point_error,abs(d.mean()-c['delta_mean']))
                            cellrows.append(dict(stage=stage['name'],model=mid,dataset=did,family=m['family'],baseline=base,
                                blocks=len(d),delta_pp=100*d.mean(),ci_low_pp=100*c['ci95'][0],ci_high_pp=100*c['ci95'][1],holm_p=c['holm_p_within_degree']))
        dev_scores={name:float(np.mean([(v[name]-v['random']).mean() for (s,_,_),v in allvals.items() if s=='dev'])) for name in ['Q','G','C']}
        aggregate={}
        for base in ['Hnorm','maxp','random']:
            family=[]
            for did in stage['test']:
                family.append(np.stack([allvals['test',mid,did][gate['winner']]-allvals['test',mid,did][base] for mid in stage['models']]).mean(0))
            # Match the original alphabetical family order and RNG to reproduce the published aggregate CI.
            families={manifest[did]['family']:v for did,v in zip(stage['test'],family)}
            family=[families[g] for g in ['edge_bias','grid_cluster','mix_gauss','uniform']]
            rng=np.random.default_rng(cfg['seed']+len(base));draws=[]
            for _ in range(2000):draws.append(float(np.mean([v[rng.integers(0,len(v),len(v))].mean() for v in family])))
            ci=np.quantile(draws,[.025,.975]);stored=summary['aggregate'][base]
            rs=[r for r in cellrows if r['stage']==stage['name'] and r['baseline']==base]
            aggregate[base]=dict(delta_pp=100*np.mean([v.mean() for v in family]),ci_pp=(ci*100).tolist(),
                reproduced_ci_abs_error=float(np.max(np.abs(ci-stored['grouped_ci95']))),
                holm_significant_positive=sum(r['holm_p']<.05 and r['delta_pp']>0 for r in rs),positive=sum(r['delta_pp']>0 for r in rs),cells=len(rs))
        cs=[c for cell in summary['cells'].values() for c in cell['primary_contrasts'].values()]
        ps=np.array([c['p_two_sided'] for c in cs]);order=np.argsort(ps);adj=np.empty(len(ps));acc=0
        for pos,i in enumerate(order):acc=max(acc,(len(ps)-pos)*ps[i]);adj[i]=min(1.,acc)
        result[stage['name']]=dict(rows=dict(counts),unique_dev=len(dev_ids),unique_test=len(test_ids),
            dev_test_overlap=len(dev_ids & test_ids),pilot_test_overlap=len(pilot_overlap),id_errors=id_errors,
            selected_metric=gate['winner'],recomputed_dev_scores=dev_scores,
            max_point_error=point_error,holm_recompute_max_error=float(np.max(np.abs(adj-np.array([c['holm_p_within_degree'] for c in cs])))),
            aggregate=aggregate,models={mid:meta['models'][mid] for mid in stage['models']})
        print('read geometry',stage['name'],dict(counts),flush=True)
    csvsave('geometry_cells.csv',cellrows);save('geometry_analysis.json',result)

def inventory():
    holiday=ROOT/'holiday_research_20260930';old=read(holiday/'artifacts/acceptance_inventory.json');oldjobs={r['id']:r for r in old['jobs']};jobs=[]
    for branch in ['week_bank','week_geometry']:
        for j in read(ROOT/f'{branch}_20260930/jobs.json')['jobs']:
            jobs.append(dict(branch=branch,id=j['id'],kind=j['kind'],completion_marker=j['complete'],
                marker_exists=Path(j['complete']).exists(),historical_status=oldjobs[j['id']]['state']['status'],
                actual_status='not_started_controller_error',attempts=oldjobs[j['id']]['state'].get('attempts',0)))
    csvsave('unrun_jobs.csv',jobs)
    save('inventory.json',dict(as_of=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        controller=read(holiday/'artifacts/state.json'),old_inventory_generated=old['generated'],
        old_status_counts=dict(Counter(j['historical_status'] for j in jobs)),
        branch_counts=dict(Counter(j['branch'] for j in jobs)),jobs=jobs))

if __name__=='__main__':
    bank();geometry();inventory()

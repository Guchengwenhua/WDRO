"""Paired statistics with explicit net, degree, seed and design estimands."""
import numpy as np

from stage6.statistics import bootstrap_cell, ci
from stage8.common import *
from stage8.attack import task_id

METHODS=['erm','mixed_erm','hard_mining','group_dro','wdro_rest']
DEGREES=[10,20,30,40,50]
SEEDS=[101,102,103]
GROUPS=['uniform','mix_gauss','edge_bias','grid_cluster']


def metric_row(dataset,method,degree,distribution,metric,points,draws):
    points=np.asarray(points,dtype=np.float64)
    return dict(dataset=dataset,method=method,degree=degree,distribution=distribution,metric=metric,
        estimate=float(points.mean()),**ci(np.asarray(draws).mean(axis=1)),
        seed101=float(points[0]),seed102=float(points[1]),seed103=float(points[2]))


def paired_rows(dataset,degree,distribution,metric,points,draws):
    rows=[]
    for baseline in METHODS[:-1]:
        delta=points['wdro_rest']-points[baseline]
        b=draws['wdro_rest']-draws[baseline]
        for si,seed in enumerate(SEEDS+['mean3']):
            point=float(delta[si]) if si<3 else float(delta.mean())
            boot=b[:,si] if si<3 else b.mean(axis=1)
            rows.append(dict(dataset=dataset,candidate='wdro_rest',baseline=baseline,degree=degree,
                distribution=distribution,metric=metric,seed=seed,difference=point,**ci(boot),
                all_three_seed_improve=bool(np.all(delta<0))))
    return rows


def cached_bootstrap(path,binding,calculate):
    path=Path(path);meta=path.with_suffix('.json')
    if meta.exists():
        r=read(meta);assert r['binding']==binding and r['status']=='pass'
        return load_npz(checked(r['arrays'])),artifact(meta)
    arrays=calculate()
    atomic_npz(path,arrays)
    atomic_write_json(meta,dict(status='pass',binding=binding,arrays=artifact(path)),mode=0o444)
    return arrays,artifact(meta)


def attack_statistics(lease):
    plan=gate();models={(m['method'],m['degree'],m['seed']):m for m in plan['models']}
    point,boot,raw,identities={}, {}, {}, {}
    cells,per_seed,main,paired,records,differences=[],[],[],[],[],[]
    for degree in DEGREES:
        for gi,group in enumerate(GROUPS):
            for method in METHODS:
                vectors=[];sources=[];metrics=[]
                for seed in SEEDS:
                    path=LOG/'attacks'/task_id(models[(method,degree,seed)],group)/'result.json'
                    result=read(path);assert result['status']=='complete' and result['completed_nets']==2000
                    a=load_npz(checked(result['predictions']))
                    if (degree,group) in identities:
                        np.testing.assert_array_equal(identities[(degree,group)],a['net_ids'])
                    else:identities[(degree,group)]=a['net_ids']
                    vectors.append(a['regret']);sources.append(artifact(path))
                    per_seed.append(dict(method=method,degree=degree,distribution=group,seed=seed,count=2000,
                        mean=result['summary']['mean'],p95=result['summary']['p95'],
                        original_mean=result['original_summary']['mean'],original_p95=result['original_summary']['p95'],
                        mean_cost=result['cost']['mean'],p95_cost=result['cost']['p95'],max_cost=result['cost']['maximum'],
                        changed_fraction=result['changed_fraction'],legal_fraction=result['legal_fraction'],
                        actor_calls=512000,rest_calls=512000,search_geosteiner_calls=512000,
                        verification_geosteiner_calls=2000,wall_seconds=result['times']['wall_seconds'],
                        verification_seconds=result['verification_seconds'],peak_reserved_mib=result['peak_reserved_mib'],
                        result_sha256=sources[-1]['sha256']))
                values=np.stack(vectors);key=(method,degree,group);raw[key]=values
                points=dict(mean=values.mean(axis=1),p95=np.quantile(values,.95,axis=1));point[key]=points
                binding=dict(inputs=sources,plan=artifact(PLAN),code=artifact(Path(__file__)),seed=808+degree*10+gi)
                b,rec=cached_bootstrap(CACHE/'bootstrap_attack'/f'{method}_d{degree}_{group}.npz',binding,
                    lambda:bootstrap_cell(values,samples=10000,seed=binding['seed'],
                        heartbeat=lambda n:lease.update(len(cells),'attack_bootstrap',degree=degree,group=group,method=method,replicates=n)))
                records.append(rec);boot[key]=b
                for metric in ('mean','p95'):
                    cells.append(metric_row('attack',method,degree,group,metric,points[metric],b[metric]))
            for baseline in METHODS[:-1]:
                path=EXP/'tables/stage8/paired_nets'/f'wdro_minus_{baseline}_d{degree}_{group}.npz'
                atomic_npz(path,dict(net_ids=identities[(degree,group)],seeds=np.array(SEEDS),
                    relative_regret_difference=raw[('wdro_rest',degree,group)]-raw[(baseline,degree,group)]))
                differences.append(artifact(path))
            for metric in ('mean','p95'):
                paired+=paired_rows('attack',degree,group,metric,
                    {m:point[(m,degree,group)][metric] for m in METHODS},{m:boot[(m,degree,group)][metric] for m in METHODS})
    # Arithmetic degree macro retains the Stage6 definition; P95 is computed
    # independently per degree and trained seed before this macro average.
    for degree in DEGREES+['macro5']:
        group_points,group_boot={},{}
        for group in GROUPS:
            for method in METHODS:
                for metric in ('mean','p95'):
                    p=point[(method,degree,group)][metric] if degree!='macro5' else np.mean([point[(method,d,group)][metric] for d in DEGREES],axis=0)
                    b=boot[(method,degree,group)][metric] if degree!='macro5' else np.mean([boot[(method,d,group)][metric] for d in DEGREES],axis=0)
                    group_points[(method,group,metric)]=p;group_boot[(method,group,metric)]=b
                    if degree=='macro5':main.append(metric_row('attack',method,degree,group,metric,p,b))
            if degree=='macro5':
                for metric in ('mean','p95'):
                    paired+=paired_rows('attack',degree,group,metric,
                        {m:group_points[(m,group,metric)] for m in METHODS},{m:group_boot[(m,group,metric)] for m in METHODS})
        # Primary worst-group estimand matches Stage6: maximum of group means
        # after seed aggregation. Keep per-seed maxima for direction reporting.
        wp,wb={},{}
        for method in METHODS:
            seed_points=np.max([group_points[(method,g,'mean')] for g in GROUPS[1:]],axis=0)
            p=max(group_points[(method,g,'mean')].mean() for g in GROUPS[1:])
            b=np.max([group_boot[(method,g,'mean')].mean(axis=1) for g in GROUPS[1:]],axis=0)
            seed_boot=np.max([group_boot[(method,g,'mean')] for g in GROUPS[1:]],axis=0)
            wp[method]=(p,seed_points);wb[method]=(b,seed_boot)
            row=dict(dataset='attack',method=method,degree=degree,distribution='ood_worst_group',metric='mean',
                estimate=float(p),**ci(b),seed101=float(seed_points[0]),seed102=float(seed_points[1]),seed103=float(seed_points[2]))
            (main if degree=='macro5' else cells).append(row)
        for baseline in METHODS[:-1]:
            for si,seed in enumerate(SEEDS+['mean3']):
                delta=wp['wdro_rest'][1]-wp[baseline][1]
                paired.append(dict(dataset='attack',candidate='wdro_rest',baseline=baseline,degree=degree,
                    distribution='ood_worst_group',metric='mean',seed=seed,
                    difference=float(delta[si] if si<3 else wp['wdro_rest'][0]-wp[baseline][0]),
                    **ci(wb['wdro_rest'][1][:,si]-wb[baseline][1][:,si] if si<3 else wb['wdro_rest'][0]-wb[baseline][0]),
                    all_three_seed_improve=bool(np.all(delta<0))))
    return dict(per_seed=per_seed,per_degree=cells,main=main,paired=paired,bootstrap=records,paired_nets=differences)


def ispd_bootstrap(values,designs,strata,*,samples=10000,seed=808,heartbeat=None):
    """Fixed design-degree strata; pooled net P95 is recomputed for every draw."""
    a=np.asarray(values,dtype=np.float64);designs=np.asarray(designs);strata=np.asarray(strata)
    assert a.ndim==2 and a.shape[1]==len(designs)==len(strata) and np.isfinite(a).all()
    groups=[np.flatnonzero(strata==s) for s in sorted(np.unique(strata))]
    order=np.concatenate(groups)
    labels=designs[order]
    by_design=[np.flatnonzero(labels==d) for d in sorted(np.unique(designs))]
    rng=np.random.default_rng(seed)
    out={k:np.empty((samples,a.shape[0])) for k in ('net_weighted_mean','design_macro_mean','p95','worst_design_mean')}
    for start in range(0,samples,8):
        n=min(8,samples-start)
        indices=np.concatenate([g[rng.integers(0,len(g),size=(n,len(g)))] for g in groups],axis=1)
        for si in range(a.shape[0]):
            draws=a[si,indices]
            means=np.stack([draws[:,g].mean(axis=1) for g in by_design],axis=1)
            out['net_weighted_mean'][start:start+n,si]=draws.mean(axis=1)
            out['design_macro_mean'][start:start+n,si]=means.mean(axis=1)
            out['worst_design_mean'][start:start+n,si]=means.max(axis=1)
            out['p95'][start:start+n,si]=np.quantile(draws,.95,axis=1)
        if heartbeat and start%160==0:heartbeat(start)
    return out


def ispd_statistics(lease):
    from stage7.delivery import design_statistics
    plan=gate();models={(m['method'],m['degree'],m['seed']):m for m in plan['models']}
    points,boots,records={}, {}, []
    main,per_seed,paired=[],[],[]
    for method in METHODS:
        values=[];sources=[];ids0=None
        for seed in SEEDS:
            parts=[]
            for degree in DEGREES:
                path=EXP/'logs/stage7/evaluation'/f'{method}_d{degree}_s{seed}'/'result.json'
                r=read(path);assert r['binding']['model']==models[(method,degree,seed)]
                a=load_npz(checked(r['predictions']));a['degree']=np.full(len(a['net_ids']),degree)
                parts.append(a);sources.append(artifact(path))
            a={k:np.concatenate([p[k] for p in parts]) for k in ('net_ids','design_ids','regret','degree')}
            if ids0 is None:ids0=a['net_ids']
            else:np.testing.assert_array_equal(ids0,a['net_ids'])
            values.append(a['regret'])
        values=np.stack(values);designs=a['design_ids']
        strata=np.array([f'{int(d)}:{name.decode()}' for d,name in zip(a['degree'],designs,strict=True)])
        p=[design_statistics(v,designs) for v in values]
        points[method]={k:np.array([r[k] for r in p]) for k in ('net_weighted_mean','design_macro_mean','p95','worst_design_mean')}
        binding=dict(inputs=sources,plan=artifact(PLAN),code=artifact(Path(__file__)),seed=808,
            strata='fixed design-degree; same draws across methods and seeds')
        b,rec=cached_bootstrap(CACHE/'bootstrap_ispd'/f'{method}.npz',binding,
            lambda:ispd_bootstrap(values,designs,strata,heartbeat=lambda n:lease.update(n,'ispd_bootstrap',method=method)))
        records.append(rec);boots[method]=b
        for metric in points[method]:
            main.append(metric_row('ispd',method,'pooled','ispd',metric,points[method][metric],b[metric]))
            for si,seed in enumerate(SEEDS):
                per_seed.append(dict(method=method,seed=seed,metric=metric,estimate=float(points[method][metric][si]),**ci(b[metric][:,si])))
    for metric in points['erm']:
        paired+=paired_rows('ispd','pooled','ispd',metric,{m:points[m][metric] for m in METHODS},{m:boots[m][metric] for m in METHODS})
    return dict(main=main,per_seed=per_seed,paired=paired,bootstrap=records)

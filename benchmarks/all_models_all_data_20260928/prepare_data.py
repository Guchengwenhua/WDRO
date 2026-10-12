"""Materialize the project's fixed evaluation datasets and original generator settings.

No training or model-based selection. Existing coordinates keep their original
scaling; reference values are used only where their current provenance is known.
"""
import ast
import collections
import importlib.util
import re
import sys
import time
from pathlib import Path
import numpy as np
from common import *

class Catalog:
    def __init__(self):
        self.rows=[]; self.notes=[]
        self.out=ART/'data'; self.out.mkdir(exist_ok=True)

    def add(self,label,pins,source,kind='heldout',reference=None,ids=None,**extra):
        x=np.asarray(pins,dtype=np.float32)
        if x.ndim!=3 or x.shape[-1]!=2 or len(x)==0:
            self.notes.append(dict(source=str(source),reason='no nonempty [N,degree,2] coordinate array',shape=list(x.shape)));return
        if not np.isfinite(x).all():
            raise ValueError(f'nonfinite input: {label}')
        cid=f'D{len(self.rows):04d}';p=self.out/(cid+'.npz')
        ref=np.full(len(x),np.nan) if reference is None else np.asarray(reference,np.float64)
        if ref.shape!=(len(x),): raise ValueError(f'reference shape: {label}')
        values=dict(pins=x,reference=ref)
        if ids is not None:values['net_ids']=np.asarray(ids).astype(str)
        npz(p,**values)
        row=dict(id=cid,label=label,path=str(p),source=str(source),degree=x.shape[1],count=len(x),kind=kind,
                 reference_available=bool(np.isfinite(ref).all()),minimum=float(x.min()),maximum=float(x.max()),**extra)
        self.rows.append(row)
        print(json.dumps(dict(phase='prepare_data',id=cid,label=label,n=len(x)),ensure_ascii=False),flush=True)

def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def aligned_reference(z, path):
    if not path.exists():return None
    ref=arrays(path)
    if 'lstar' not in ref:return None
    if 'net_ids' in z and 'net_ids' in ref:
        idx={v:i for i,v in enumerate(ref['net_ids'].astype(str))}
        return np.array([ref['lstar'][idx[v]] if v in idx else np.nan for v in z['net_ids'].astype(str)])
    return ref['lstar'] if len(ref['lstar'])==len(z['pins']) else None

def real_datasets(c):
    # Published prepared inputs, not result shards. Split roles are metadata only.
    role_map={}
    for p in sorted((PROJECT/'net_structure_real_20260922/artifacts/multidegree').glob('*.npz')):
        z=arrays(p)
        if 'net_ids' in z:
            role=p.stem.split('_',1)[1]
            for n in z['net_ids'].astype(str):role_map[n.removeprefix('public::')]=role
    real=arrays(PROJECT/'net_structure_real_20260922/artifacts/data.npz')
    for n,role in zip(real['net_ids'].astype(str),real['split']):role_map[n.removeprefix('public::')]=str(role)
    for suite,folder,ref_folder in [
        ('ISPD2018-2019',PROJECT/'wdro_precision_v2/experiments/cache/stage7/inputs',None),
        ('Public-ISPD2008-CircuitOps',PROJECT/'benchmarks/public_designs_20260912/cache/inputs',PROJECT/'benchmarks/public_designs_20260912/cache/references')]:
        for p in sorted(folder.glob('degree_*.npz')):
            z=arrays(p);d=z['pins'].shape[1]
            if ref_folder:
                reference=aligned_reference(z,ref_folder/p.name)
            else:
                # Completed stage7 predictions bind references to net_ids.
                candidates=sorted((PROJECT/'wdro_precision_v2/experiments/logs/stage7/evaluation').glob(f'erm_d{d}_s101/predictions.npz'))
                reference=aligned_reference(z,candidates[0]) if candidates else None
            designs=z['design_ids'].astype(str)
            roles=np.array([role_map.get(n.removeprefix('public::'),'historical_test') for n in z['net_ids'].astype(str)])
            for design in sorted(set(designs)):
                for role in sorted(set(roles[designs==design])):
                    take=(designs==design)&(roles==role)
                    c.add(f'{suite}/{design}/d{d}/{role}',z['pins'][take],p,kind='real_'+role,
                          reference=None if reference is None else reference[take],ids=z['net_ids'][take],design=design,
                          real_training_role=role)

def mainline(c):
    refs={}
    for p in sorted((PROJECT/'wdro_precision_v2/experiments/cache/lstar/protocol_v1').glob('*.npz')):
        z=arrays(p)
        for n,v in zip(z['net_ids'].astype(str),z['lstar']):refs[n]=float(v)
    for p in sorted((PROJECT/'wdro_legacy_v1/experiments/cache/datasets/protocol_v1').rglob('*.npz')):
        z=arrays(p)
        if 'pins' not in z:continue
        role=p.parent.name
        ids=z.get('net_ids',np.arange(len(z['pins'])).astype(str)).astype(str)
        ref=np.array([refs.get(n,np.nan) for n in ids])
        c.add(f'precision_v2/{role}/{p.stem}',z['pins'],p,kind=role,reference=ref,ids=ids)

def standalone_sources(c):
    for p in sorted((PROJECT/'papers_wdro_20260915/sources/Timing-RST/data/test_data').glob('array*.npy')):
        x=np.load(p,allow_pickle=False)
        c.add('Timing-RST官方/'+p.stem,x[...,:2],p,coordinate_convention='official original coordinates; pin 0 treated as root by rooted methods')
    # Both vendor directories retain independently named original test files.
    # The common filename is the released set; one source with both locations recorded.
    for p in sorted((PROJECT/'shared/legacy_dro/REST-main/test_set').glob('test*.txt')):
        degree=int(re.search(r'test(\d+)',p.stem).group(1))
        x=np.loadtxt(p).reshape(-1,degree,2)
        c.add(f'REST-DAC21原始/test/d{degree}',x,p,aliases=[str(PROJECT/'shared/legacy_dro/WDRO-REST/test_set'/p.name)])

def small_files(c):
    # These are already frozen complete evaluation inputs, distinct random streams.
    for branch in ('rl_budget_scaling_20260921','rl_expansion_20260922'):
        for p in sorted((PROJECT/branch/'artifacts/data').glob('*.npz')):
            z=arrays(p)
            for name,x in z.items():
                if x.ndim==3 and x.shape[-1]==2:
                    c.add(f'{branch}/{p.stem}/{name}',x,p,kind='heldout' if 'test' in p.stem else 'validation_or_calibration')
    for p in sorted((PROJECT/'papers_wdro_20260915/artifacts/data').glob('*.npz')):
        if p.stem.startswith('train_'):continue
        z=arrays(p)
        c.add('papers/'+p.stem,z['pins'],p,kind='heldout' if p.stem.startswith('test') else 'validation_or_attack_center',ids=z.get('net_ids'))
        if 'candidates' in z:
            a=z['candidates'];c.add('papers/'+p.stem+'/fixed_K33_pool',a.reshape(-1,a.shape[-2],2),p,kind='fixed_candidate_pool')
    for p in sorted((PROJECT/'benchmarks/rl_only_vs_mainline_20260919/artifacts/data').glob('*.npz')):
        z=arrays(p);c.add('rl_vs_mainline/'+p.stem,z['pins'].reshape(-1,30,2),p,kind='historical_test')
    # Natural heldout streams are common across seeds and arms of each branch.
    for branch in ('oracle_free_20260915','rl_only_20260915','rl_research_20260915','rl_scaleup_20260921','rl_prepost_scaleup_20260922'):
        candidates=sorted((PROJECT/branch/'artifacts/runs').glob('*/results.npz'))
        natural={}
        for p in candidates:
            z=arrays(p)
            for k,x in z.items():
                if x.ndim==3 and x.shape[-1]==2 and ((k.startswith('final__') and k.endswith('__pins')) or k.endswith('__nominal_pins')):
                    natural.setdefault(k,(p,x))
            if natural:break
        for key,(p,x) in natural.items():c.add(f'{branch}/{key}',x,p)
    for branch in ('adversarial_v3_20260912','adversarial_v3_1_20260912'):
        for mode in ('fixed','joint'):
            seen=set()
            for p in sorted((PROJECT/branch/f'experiments/{mode}').glob('*/inputs/evaluation__*.npz')):
                z=arrays(p);d=z['pins'].shape[1];key=(d,p.stem)
                if key in seen:continue
                seen.add(key);c.add(f'{branch}/{mode}/{p.stem}/d{d}',z['pins'],p)
    # Heldout stress centers used by the original mainline attack benchmark.
    for p in sorted((PROJECT/'wdro_precision_v2/experiments/cache/stage8/inputs').glob('*.npz')):
        z=arrays(p)
        if 'pins' in z and z['pins'].ndim==3:c.add('precision_v2/stage8/'+p.stem,z['pins'],p,kind='attack_center',ids=z.get('net_ids'))

def load_function(path,name):
    # Extract the pure generator from the original source without importing its
    # plotting/solver/training entrypoints.
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
    return env[name]

def historical_sweeps(c):
    source=PROJECT/'shared/legacy_dro/REST-main/test_robust.py'
    generate=load_function(source,'generate_terminals')
    specs={}
    for vendor in ('REST-main','WDRO-REST'):
        for p in sorted((PROJECT/f'shared/legacy_dro/{vendor}').glob('results_sweep*/**/runs.jsonl')):
            for line in p.read_text().splitlines():
                try:args=json.loads(line)['args']
                except (ValueError,KeyError):continue
                keys=('degree','dimension','test_size','seed','point_dist','nonuniform','clusters','sigma')
                if all(k in args for k in keys):
                    vals=tuple(args[k] for k in keys)
                    if args.get('test_data'):continue
                    specs.setdefault(vals,str(p))
    for vals,record in sorted(specs.items()):
        degree,dim,n,seed,dist,nu,clusters,sigma=vals
        x=generate(n,degree,dim,seed,dist,nu,clusters,sigma)
        label=f'历史规则/{dist}/nu{nu}/k{clusters}/sigma{sigma}/d{degree}/s{seed}'
        c.add(label,x,source,kind='reconstructed_historical_generator',generator_parameters=dict(degree=degree,dimension=dim,count=n,seed=seed,distribution=dist,nonuniform=nu,clusters=clusters,sigma=sigma),definition_record=record)

def on_demand_protocols(c):
    # Same generators and namespace arguments as the completed experiments.
    sys.path.insert(0,str(PROJECT/'adversarial_v3_1_20260912'))
    from adv_v3.common import make_data
    protocol=read(PROJECT/'gaussian_es_scaleup_20260922/protocol.json')
    for d in protocol['degrees']:
        for dist in DISTS:
            x=make_data(d,protocol['test_count'],dist,protocol['data_namespace']+':test',0)
            c.add(f'gaussian_es_scaleup/test/{dist}/d{d}',x,PROJECT/'gaussian_es_scaleup_20260922/common.py')
    spec=importlib.util.spec_from_file_location('_cross_rl_geometry',PROJECT/'entropic_teacher_wdro_20260922/geometry.py')
    geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
    protocol=read(PROJECT/'entropic_teacher_wdro_20260922/protocol.json')
    for d in map(int,protocol['degrees']):
        for dist in DISTS:
            x=geo.make_data(protocol['test_n'],d,dist,protocol['namespace']+'/test',0)
            c.add(f'entropic_teacher/test/{dist}/d{d}',x,PROJECT/'entropic_teacher_wdro_20260922/engine.py')

def main():
    c=Catalog()
    for step in (real_datasets,mainline,standalone_sources,small_files,on_demand_protocols,historical_sweeps):
        step(c)
    atomic(ART/'datasets.json',c.rows)
    atomic(ART/'data_notes.json',c.notes)
    pack_data(c.rows)

def pack_data(dataset_rows):
    # One concatenated immutable-by-protocol array per degree reduces thousands
    # of tiny GPU launches on rare real-chip designs. Offsets retain every column.
    packs=[]
    for d in sorted({v['degree'] for v in dataset_rows}):
        rows=[v for v in dataset_rows if v['degree']==d];x=[];refs=[];offset=0;mapping=[]
        for row in rows:
            z=arrays(row['path']);x.append(z['pins']);refs.append(z['reference'])
            mapping.append(dict(id=row['id'],start=offset,end=offset+row['count']));offset+=row['count']
        pp=ART/'packed'/f'd{d}.npy';pp.parent.mkdir(exist_ok=True)
        temp=pp.with_suffix('.partial')
        with temp.open('wb') as f:np.save(f,np.concatenate(x));f.flush();os.fsync(f.fileno())
        os.replace(temp,pp)
        refp=pp.with_name(f'd{d}_reference.npy')
        with refp.open('wb') as f:np.save(f,np.concatenate(refs))
        packs.append(dict(degree=d,path=str(pp),reference=str(refp),count=offset,columns=mapping))
    atomic(ART/'packs.json',packs)
    print(json.dumps(dict(status='data_ready',columns=len(dataset_rows),nets=sum(d['count'] for d in dataset_rows),degrees=[p['degree'] for p in packs])),flush=True)

if __name__=='__main__':main()

"""Small fixed-budget perturbation search with strictly offline GeoSteiner reference."""
import argparse,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from research import ROOT,PROJECT,BENCH,Runner,SCORES,atomic,npz,read,ci
def normalized(x):
    x=np.asarray(x,dtype=np.float64)
    x=(x-x.min(0))*(.8/np.ptp(x,axis=0).max())+.1
    x=np.clip(x,.1,.9).astype(np.float32)
    d=np.abs(x[:,None,:].astype(float)-x[None,:,:]).sum(-1);np.fill_diagonal(d,np.inf)
    if d.min()<1e-6:raise ValueError('candidate violates minimum pin spacing')
    return x
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',default='4');args=ap.parse_args()
    out=ROOT/'artifacts/attack';out.mkdir(parents=True,exist_ok=True)
    cfg=dict(models=['M0900','M1118'],datasets=['D0372','D0377','D0382','D0387'],centers_per_family=8,candidates=8,noise_halfwidth=.02,seed=20260931,batch=256,gpu=args.gpu,temperature=1.)
    if (out/'config.json').exists() and read(out/'config.json')!=cfg:raise RuntimeError('config changed')
    atomic(out/'config.json',cfg);start=time.time()
    metadata_path=out/'inputs_metadata.json'
    if metadata_path.exists():
        metadata=read(metadata_path)
    else:
        model_catalog={m['id']:m for m in read(BENCH/'artifacts/models.json')}
        dataset_catalog={d['id']:d for d in read(BENCH/'artifacts/datasets.json')}
        frozen_models={mid:model_catalog[mid] for mid in cfg['models']}
        frozen_datasets={did:dataset_catalog[did] for did in cfg['datasets']}
        source_stats={}
        for entry in list(frozen_models.values())+list(frozen_datasets.values()):
            path=Path(entry['path']);st=path.stat()
            source_stats[str(path)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
        metadata=dict(models=frozen_models,datasets=frozen_datasets,source_stats=source_stats)
        atomic(metadata_path,metadata)
    for source,expected in metadata['source_stats'].items():
        stat=Path(source).stat()
        if stat.st_size!=expected['bytes'] or stat.st_mtime_ns!=expected['mtime_ns']:
            raise RuntimeError('frozen source changed: '+source)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.10)
    def beat(phase,**extra):
        atomic(out/'heartbeat.json',dict(pid=os.getpid(),status=phase,gpu=args.gpu,batch=256,elapsed_seconds=time.time()-start,**extra))
        print(json.dumps(dict(phase=phase,**extra)),flush=True)
    if not (out/'inputs.npz').exists():
        ds=metadata['datasets']
        rng=np.random.default_rng(cfg['seed']);pins=[];ids=[];families=[];indices=[]
        for did in cfg['datasets']:
            with np.load(ds[did]['path'],allow_pickle=False) as z:
                ix=rng.permutation(len(z['pins']))[:cfg['centers_per_family']]
                for index in ix:
                    center=z['pins'][index].astype(np.float32)
                    group=[center]
                    for _ in range(cfg['candidates']):
                        for attempt in range(100):
                            try:
                                candidate=normalized(center+rng.uniform(-cfg['noise_halfwidth'],cfg['noise_halfwidth'],center.shape))
                                break
                            except ValueError:
                                if attempt==99:raise
                        group.append(candidate)
                    pins.append(group);ids.append(str(z['net_ids'][index]));families.append(did);indices.append(index)
        x=np.array(pins,dtype=np.float32)
        cost=np.abs(x.astype(float)-x[:,:1].astype(float)).sum(-1).mean(-1)
        linf=np.abs(x.astype(float)-x[:,:1].astype(float)).max((2,3))
        npz(out/'inputs.npz',pins=x,center_ids=np.array(ids),family=np.array(families),source_indices=np.array(indices),indexed_mean_l1=cost,post_normalization_linf=linf)
    with np.load(out/'inputs.npz',allow_pickle=False) as z:
        x=z['pins'];families=z['family'];cost=z['indexed_mean_l1'];linf=z['post_normalization_linf']
    flat=x.reshape(-1,x.shape[-2],2)
    models=metadata['models']
    # Freeze all scores and metric-chosen indices before any oracle evaluation.
    for mid in cfg['models']:
        target=out/mid;target.mkdir(exist_ok=True)
        if (target/'selection.json').exists():continue
        runner=Runner(models[mid],cfg['batch']);paths=[]
        for a in range(0,len(flat),cfg['batch']):
            b=min(a+cfg['batch'],len(flat));path=target/f'prediction_{a:06d}_{b:06d}.npz';paths.append(path)
            if path.exists():continue
            length,steps=runner.predict(flat[a:b])
            npz(path,length=length,scores=steps.mean(1))
            beat('inference',model=mid,committed=b,total=len(flat))
        shards=[np.load(path,allow_pickle=False) for path in paths]
        length=np.concatenate([z['length'] for z in shards]).reshape(x.shape[:2])
        scores=np.concatenate([z['scores'] for z in shards]).reshape(x.shape[:2]+(len(SCORES),))
        for z in shards:z.close()
        npz(target/'predictions.npz',length=length,scores=scores)
        selection={'random':np.random.default_rng(cfg['seed']+1).integers(1,9,len(x)).tolist()}
        for j,name in enumerate(SCORES):
            s=scores[:,1:,j];s=-s if name=='maxp' else s
            selection[name]=(1+s.argmax(-1)).tolist()
        atomic(target/'selection.json',dict(scope='8 perturbed candidates; nominal excluded',selected=selection,oracle_used=False))
        for hook in runner.hooks:hook.remove()
        del runner;torch.cuda.empty_cache()
        beat('selections_frozen',model=mid,centers=len(x))
    spec=importlib.util.spec_from_file_location('_attack_geosteiner',PROJECT/'wdro_precision_v2/wdro_rest_min/src/geosteiner.py')
    geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
    evaluator=geo.GeoSteinerEvaluator(PROJECT/'shared/legacy_dro/WDRO-REST/algorithms/libeval.so')
    refs=[]
    for k,group in enumerate(x):
        path=out/f'reference_{k:03d}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:r=z['reference']
        else:
            # Reference is never sent back into candidate generation or selection.
            r=np.array([evaluator.solve(net,require_integral=True) for net in group])
            npz(path,reference=r)
        refs.append(r);beat('offline_reference',committed_centers=k+1,total_centers=len(x))
    ref=np.stack(refs);rows=np.arange(len(x));rng=np.random.default_rng(cfg['seed']+2)
    result={}
    for mid in cfg['models']:
        with np.load(out/mid/'predictions.npz',allow_pickle=False) as z:length=z['length']
        regret=length/ref-1
        if regret.min() < -1e-5:raise RuntimeError('negative true regret: incompatible reference')
        selected=read(out/mid/'selection.json')['selected']
        selected['oracle_upper_bound']=(1+regret[:,1:].argmax(-1)).tolist()
        selected['nominal']=[0]*len(x)
        cells={}
        for group in ['all']+cfg['datasets']:
            keep=np.arange(len(x)) if group=='all' else np.flatnonzero(families==group)
            random=regret[rows,np.array(selected['random'])]
            cell={}
            for name,which in selected.items():
                which=np.array(which);values=regret[rows,which];delta=values-random;nomdelta=values-regret[:,0]
                boot=[float(delta[keep][rng.integers(0,len(keep),len(keep))].mean()) for _ in range(1000)]
                cell[name]=dict(n_centers=len(keep),mean_true_regret=float(values[keep].mean()),
                                delta_vs_random=float(delta[keep].mean()),delta_vs_random_ci95=ci(boot),
                                delta_vs_nominal=float(nomdelta[keep].mean()),
                                fraction_worse_than_nominal=float((nomdelta[keep]>0).mean()),
                                mean_indexed_l1_transport=float(cost[rows,which][keep].mean()),
                                max_postnorm_linf=float(linf[rows,which][keep].max()))
            cells[group]=cell
        result[mid]=cells
    atomic(out/'summary.json',result)
    lines=['# 几何不确定性有限扰动搜索：探索性结果','',
           '32个中心（每分布8个），每中心8个固定独立扰动候选和1个nominal。所有指标在同候选池选择，之后才用GeoSteiner精评全部288实例；真值不反馈搜索。原始扰动每坐标±0.02，经bbox最长边统一缩放到0.8后最终距离可能超过0.02，实际indexed L1及最大L∞记录在summary.json。不是既定ε约束的WDRO内层优化。','',
           '| 模型 | 方法 | 所选平均真实误差% | 相对random差/百分点 | 95%区间/百分点 | 相对nominal差/百分点 |','|---|---|---:|---:|---|---:|']
    for mid,cells in result.items():
        for name,v in cells['all'].items():
            lo,hi=v['delta_vs_random_ci95']
            lines.append(f"| {mid} | {name} | {100*v['mean_true_regret']:.3f} | {100*v['delta_vs_random']:.3f} | [{100*lo:.3f},{100*hi:.3f}] | {100*v['delta_vs_nominal']:.3f} |")
    lines+=['','区间按中心配对重采样，不将同一中心的8个候选当独立样本。每分布仅8中心，分组结果在summary.json。oracle_upper_bound是仅供离线上限比较的真实最大误差选择。负的相对random差说明该指标在此样本中没有优于随机；不能预设高熵对应高误差。未运行基于这些候选的训练，因此不能推断训练收益。']
    (ROOT/'ATTACK_REPORT.md').write_text('\n'.join(lines))
    beat('complete',total_centers=len(x))
    atomic(out/'complete.json',dict(status='complete',elapsed_seconds=time.time()-start))
if __name__=='__main__':main()

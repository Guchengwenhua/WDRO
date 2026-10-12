"""Staged held-out conditional geometric uncertainty research; no optimizer."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
PILOT=PROJECT/'geometry_uncertainty_20260930'
sys.path.insert(0,str(PILOT))
from research import Runner,SCORES,corr,rank,entropy
sys.path.insert(0,str(PROJECT/'benchmarks/all_models_all_data_20260928'))
from common import read,atomic,npz
BENCH=PROJECT/'benchmarks/all_models_all_data_20260928/artifacts'
ART=ROOT/'artifacts'
GROUPS=['edge_bias','grid_cluster','mix_gauss','uniform']
STAGES=[
 dict(name='d30_multiseed',degree=30,models=['M0900','M1118','M0902','M1120'],
      dev=['D0238','D0243','D0248','D0253'],test=['D0218','D0223','D0228','D0233'],dev_n=1024,test_n=2048),
 dict(name='d10_degree_control',degree=10,models=['M0888','M1108'],
      dev=['D0236','D0241','D0246','D0251'],test=['D0216','D0221','D0226','D0231'],dev_n=512,test_n=1024)]
def bootstrap_ci(x,rng,samples=1000):
    x=np.asarray(x,dtype=float)
    return np.quantile(x[rng.integers(0,len(x),(samples,len(x)))].mean(-1),[.025,.975]).tolist()
def stat_corr(score,error):
    return dict(pearson=corr(score,error),spearman=corr(rank(score),rank(error)))
def paired_test(delta,rng):
    delta=np.asarray(delta)
    # Fixed-budget paired randomization; tests exchangeability under zero effect.
    signs=rng.choice([-1.,1.],size=(9999,len(delta)))
    null=(signs*delta).mean(-1);observed=float(delta.mean())
    p=float((1+(np.abs(null)>=abs(observed)).sum())/(len(null)+1))
    return dict(delta_mean=observed,ci95=bootstrap_ci(delta,rng),p_two_sided=p,n_blocks=len(delta))
def holm(values):
    order=np.argsort(values);out=np.zeros(len(values));running=0
    for position,index in enumerate(order):
        running=max(running,(len(values)-position)*values[index]);out[index]=min(1.,running)
    return out.tolist()
class Experiment:
    def __init__(self,args):
        self.args=args;self.start=time.time();self.processed=0
        ART.mkdir(exist_ok=True)
        self.config=dict(version=1,stages=STAGES,seed=20261001,bootstrap=1000,budget=16,
                         gpu=args.gpu,batch=256,temperature=1.,float='FP32',memory_fraction=.1,
                         dev_selection_candidates=['Q','G','C'],test_baselines=['Hnorm','maxp','random'],
                         scope='conditional p(second_endpoint,direction|greedy first endpoint,history)')
        if (ART/'config.json').exists() and read(ART/'config.json')!=self.config:raise RuntimeError('config changed')
        atomic(ART/'config.json',self.config)
        if (ART/'frozen_metadata.json').exists():self.meta=read(ART/'frozen_metadata.json')
        else:
            models={m['id']:m for m in read(BENCH/'models.json')};datasets={d['id']:d for d in read(BENCH/'datasets.json')}
            mids=sorted({mid for stage in STAGES for mid in stage['models']})
            dids=sorted({did for stage in STAGES for did in stage['dev']+stage['test']})
            frozen_models={mid:models[mid] for mid in mids};frozen_datasets={did:datasets[did] for did in dids}
            for mid,m in frozen_models.items():
                if m['family']!='rest':raise RuntimeError('not REST')
            for did,d in frozen_datasets.items():
                if not d.get('reference_available'):raise RuntimeError('missing trusted reference '+did)
            paths=[Path(m['path']) for m in frozen_models.values()]+[Path(d['path']) for d in frozen_datasets.values()]
            paths += [PILOT/'research.py',PROJECT/'benchmarks/all_models_all_data_20260928/evaluate.py',
                      PROJECT/'rl_expansion_20260922/vendor/models/actor_critic.py']
            sources={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
            self.meta=dict(models=frozen_models,datasets=frozen_datasets,sources=sources)
            atomic(ART/'frozen_metadata.json',self.meta)
        for path,expected in self.meta['sources'].items():
            st=Path(path).stat()
            if st.st_size!=expected['bytes'] or st.st_mtime_ns!=expected['mtime_ns']:raise RuntimeError('frozen source changed '+path)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.set_per_process_memory_fraction(.1)
        self.remaining=sum(len(s['models'])*4*(s['dev_n']+s['test_n']) for s in STAGES)
        for p in ART.glob('d*/predictions/*/*/*.npz'):
            with np.load(p,allow_pickle=False) as z:self.remaining-=len(z['length'])
        self.initial_remaining=self.remaining
    def beat(self,status,**extra):
        elapsed=time.time()-self.start;rate=self.processed/elapsed if elapsed else 0
        value=dict(pid=os.getpid(),status=status,gpu=self.args.gpu,batch=256,
                   elapsed_seconds=elapsed,rows_this_invocation=self.processed,
                   pending_inference_rows=max(0,self.initial_remaining-self.processed),
                   inference_rows_per_total_second=rate,
                   inference_only_eta_seconds=max(0,self.initial_remaining-self.processed)/rate if rate else None,
                   eta_note='excludes future statistical analysis; shared GPU load may change',**extra)
        atomic(ART/'heartbeat.json',value);print(json.dumps(value),flush=True)
    def split(self,stage):
        folder=ART/stage['name'];folder.mkdir(exist_ok=True)
        path=folder/'split_manifest.json'
        if path.exists():return read(path)
        manifest={}
        dev_ids=set();test_ids=set()
        for split in ['dev','test']:
            for family,did in zip(GROUPS,stage[split]):
                d=self.meta['datasets'][did]
                with np.load(d['path'],allow_pickle=False) as z:
                    ids=z['net_ids'].astype(str)
                    excluded=set()
                    if split=='test':
                        for p in (PILOT/'artifacts').glob('M*/'+did+'/*.npz'):
                            with np.load(p,allow_pickle=False) as old:excluded.update(old['net_ids'].astype(str).tolist())
                    eligible=np.array([i for i,name in enumerate(ids) if name not in excluded])
                    seed=self.config['seed']+int(did[1:])+(0 if split=='dev' else 10000)
                    selected=np.random.default_rng(seed).permutation(eligible)[:stage[split+'_n']]
                    if len(selected)<stage[split+'_n']:raise RuntimeError('not enough independent rows')
                    picked=ids[selected].tolist()
                    if len(set(picked))!=len(picked):raise RuntimeError('duplicate instance identity')
                    (dev_ids if split=='dev' else test_ids).update(picked)
                    manifest[did]=dict(split=split,family=family,indices=selected.tolist(),net_ids=picked,
                                       excluded_pilot_instances=len(excluded),count=len(picked),seed=seed)
        if dev_ids & test_ids:raise RuntimeError('dev/test instance overlap')
        atomic(path,manifest);return manifest
    def evaluate(self,stage,split,manifest):
        for mid in stage['models']:
            self.beat('loading_model',stage=stage['name'],split=split,model=mid)
            runner=Runner(self.meta['models'][mid],256)
            for did in stage[split]:
                folder=ART/stage['name']/'predictions'/mid/did;folder.mkdir(parents=True,exist_ok=True)
                row=manifest[did];indices=np.array(row['indices'])
                with np.load(self.meta['datasets'][did]['path'],allow_pickle=False) as z:x=z['pins'][indices]
                for a in range(0,len(x),256):
                    b=min(a+256,len(x));target=folder/f'{a:06d}_{b:06d}.npz'
                    if target.exists():continue
                    length,steps=runner.predict(x[a:b])
                    phases=np.stack([steps[:,ix].mean(1) for ix in np.array_split(np.arange(steps.shape[1]),3)],axis=1)
                    npz(target,length=length,mean_scores=steps.mean(1),phase_scores=phases,net_ids=np.array(row['net_ids'][a:b]),indices=indices[a:b])
                    self.processed+=b-a
                    atomic(folder/'cursor.json',dict(committed=b,total=len(x)))
                    self.beat('inference',stage=stage['name'],split=split,model=mid,dataset=did,committed=b,total=len(x))
            for hook in runner.hooks:hook.remove()
            del runner;torch.cuda.empty_cache()
    def cell(self,stage,mid,did,manifest):
        folder=ART/stage['name']/'predictions'/mid/did
        chunks=[np.load(p,allow_pickle=False) for p in sorted(folder.glob('*.npz'))]
        scores=np.concatenate([z['mean_scores'] for z in chunks])
        phases=np.concatenate([z['phase_scores'] for z in chunks])
        length=np.concatenate([z['length'] for z in chunks])
        for z in chunks:z.close()
        with np.load(self.meta['datasets'][did]['path'],allow_pickle=False) as z:reference=z['reference'][manifest[did]['indices']]
        if not np.isfinite(reference).all() or (reference<=0).any():raise RuntimeError('missing reference')
        regret=length/reference-1
        if regret.min() < -1e-5:raise RuntimeError('negative true regret')
        # Same candidate blocks/random choices across methods and model seeds.
        rng=np.random.default_rng(self.config['seed']+int(did[1:]))
        order=rng.permutation(len(regret));blocks=order.reshape(-1,self.config['budget'])
        random=rng.integers(0,self.config['budget'],len(blocks))
        rr=regret[blocks]
        values={'random':rr[np.arange(len(blocks)),random],'oracle_upper_bound':rr.max(-1)}
        for j,name in enumerate(SCORES):
            s=scores[blocks,j];s=-s if name=='maxp' else s
            values[name]=rr[np.arange(len(blocks)),s.argmax(-1)]
        return scores,phases,regret,values
    def choose(self,stage,manifest):
        target=ART/stage['name']/'dev_gate.json'
        if target.exists():return read(target)['winner']
        deltas={name:[] for name in ['Q','G','C']}
        for mid in stage['models']:
            for did in stage['dev']:
                _,_,_,values=self.cell(stage,mid,did,manifest)
                for name in deltas:deltas[name].append(float((values[name]-values['random']).mean()))
        means={name:float(np.mean(v)) for name,v in deltas.items()}
        winner=max(means,key=means.get)
        atomic(target,dict(winner=winner,dev_mean_delta_vs_random=means,
                          rule='maximum equally weighted model-family mean chosen regret minus shared random; Q,G,C tie order',
                          test_reference_accessed=False,training_gate='not authorized by this diagnostic outcome alone'))
        self.beat('dev_gate_frozen',stage=stage['name'],winner=winner)
        return winner
    def analyze(self,stage,manifest,winner):
        target=ART/stage['name']/'test_summary.json'
        if target.exists():return
        result={};primary=[];locations=[];paired={}
        for mid in stage['models']:
            for did in stage['test']:
                self.beat('test_statistics',stage=stage['name'],model=mid,dataset=did)
                scores,phases,error,values=self.cell(stage,mid,did,manifest)
                rng=np.random.default_rng(self.config['seed']+int(mid[1:])+int(did[1:]))
                threshold=np.quantile(error,.9);high=error>=threshold;count=int(np.ceil(.1*len(error)))
                metrics={}
                for j,name in enumerate(SCORES):
                    score=scores[:,j];u=-score if name=='maxp' else score
                    chosen=np.argsort(-u,kind='stable')[:count]
                    boot=[];bootp=[]
                    for _ in range(500):
                        ix=rng.integers(0,len(error),len(error))
                        boot.append(corr(rank(score[ix]),rank(error[ix])))
                        eu=-score[ix] if name=='maxp' else score[ix]
                        eh=error[ix]>=np.quantile(error[ix],.9)
                        bootp.append(float(eh[np.argsort(-eu,kind='stable')[:count]].mean()))
                    finite=[v for v in boot if v is not None]
                    metrics[name]=dict(**stat_corr(score,error),spearman_ci95=np.quantile(finite,[.025,.975]).tolist() if finite else None,
                                       precision_top10=float(high[chosen].mean()),precision_ci95=np.quantile(bootp,[.025,.975]).tolist(),
                                       high_error_prevalence=float(high.mean()),
                                       phase_correlations={label:stat_corr(phases[:,i,j],error) for i,label in enumerate(['early','middle','late'])})
                contrasts={}
                for base in ['Hnorm','maxp','random']:
                    delta=values[winner]-values[base]
                    contrasts[base]=paired_test(delta,rng)
                    primary.append(contrasts[base]['p_two_sided']);locations.append((mid,did,base))
                    family=manifest[did]['family']
                    paired.setdefault((family,base),[]).append(delta)
                result[mid+'/'+did]=dict(model=mid,dataset=did,seed=self.meta['models'][mid].get('seed'),n=len(error),
                    mean_true_regret=float(error.mean()),p95_true_regret=float(np.quantile(error,.95)),
                    metrics=metrics,selected_mean_regret={name:float(v.mean()) for name,v in values.items()},primary_contrasts=contrasts)
        corrected=holm(primary)
        for (mid,did,base),adjusted in zip(locations,corrected):
            result[mid+'/'+did]['primary_contrasts'][base]['holm_p_within_degree']=adjusted
        # Average model/seed outcomes on identical blocks, then bootstrap blocks by family.
        aggregate={}
        for base in ['Hnorm','maxp','random']:
            family_delta=[np.stack(paired[(family,base)]).mean(0) for family in GROUPS]
            rng=np.random.default_rng(self.config['seed']+len(base))
            draws=[]
            for _ in range(2000):
                draws.append(float(np.mean([v[rng.integers(0,len(v),len(v))].mean() for v in family_delta])))
            aggregate[base]=dict(mean_delta=float(np.mean([v.mean() for v in family_delta])),
                                grouped_ci95=np.quantile(draws,[.025,.975]).tolist(),
                                independent_units='candidate blocks stratified by family; models/seeds paired within each block',
                                model_count=len(stage['models']))
        atomic(target,dict(stage=stage['name'],winner=winner,cells=result,aggregate=aggregate,
                           multiple_comparison_family=len(primary),correction='Holm within each degree; d10 secondary',
                           inference_scope='offline candidate selection, not perturbation generation or WDRO training'))
        self.report()
    def report(self):
        lines=['# 扩展几何不确定性：实际结果','',
               '开发集选出Q/G/C之一后冻结；测试集中排除pilot实例。条件分布诊断、T=1、greedy FP32、batch256；所有误差依赖既有precision_v2可信参考。历史测试语料曾用于其他benchmark，本次未据其标签选指标，但不称全新外部数据。','']
        for path in sorted(ART.glob('d*/test_summary.json')):
            s=read(path);lines += [f"## {s['stage']}，开发集冻结指标：{s['winner']}",'',
                '| 对照 | 所选真实误差差/百分点 | 配对分层95%区间/百分点 |','|---|---:|---|']
            for name,v in s['aggregate'].items():
                lo,hi=v['grouped_ci95']
                lines.append(f"| {name} | {100*v['mean_delta']:.4f} | [{100*lo:.4f}, {100*hi:.4f}] |")
            lines+=['','正差表示在相同16候选预算中选到更高真实误差。区间以相同候选块配对，先对模型/种子平均再按分布重采样；种子不增加独立样本量。','',
                    f"组内主要对照共{s['multiple_comparison_family']}项，Holm校正在test_summary.json；d10为次级泛化诊断。其余逐指标相关/分步结果为探索性。",'']
        lines+=['未开展几何指导的WDRO训练，不能据本实验声称训练收益。候选池困难检索不等于连续对抗扰动生成。']
        (ROOT/'REPORT.md').write_text('\n'.join(lines))
    def run(self):
        for stage in STAGES:
            manifest=self.split(stage)
            self.evaluate(stage,'dev',manifest)
            winner=self.choose(stage,manifest)
            self.evaluate(stage,'test',manifest)
            self.analyze(stage,manifest,winner)
        self.beat('complete');atomic(ART/'complete.json',dict(status='complete',elapsed_seconds=time.time()-self.start))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',default='7');args=parser.parse_args()
    Experiment(args).run()

"""Conditional REST geometric uncertainty; frozen offline research, no training."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
BENCH=PROJECT/'benchmarks/all_models_all_data_20260928'
sys.path.insert(0,str(BENCH))
from evaluate import import_rest, lengths
from common import atomic,npz,extract_state,read
SCORES=['Hnorm','maxp','Q','G','C','first_Hnorm']
def entropy(p):
    return -(p*np.log(np.clip(p,1e-300,None))).sum(-1)
def summarize_steps(x,first,second):
    b,n,_=x.shape
    scale=np.ptp(x,axis=1).sum(-1)
    if np.any(scale<=0): raise ValueError('zero bbox scale')
    result=[]
    for step,(l1,l2) in enumerate(zip(first,second)):
        # Only p(j,o | decoded i, current history), not the full joint.
        i=l1.argmax(-1); chosen=l2.argmax(-1); mask=np.isfinite(l2)
        k=mask.sum(-1)
        p=np.exp(l2-np.max(l2,axis=-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        p1=np.exp(l1-np.max(l1,axis=-1,keepdims=True));p1/=p1.sum(-1,keepdims=True)
        h1=entropy(p1);k1=np.isfinite(l1).sum(-1)
        h1=np.divide(h1,np.log(np.maximum(k1,2)),out=np.zeros_like(h1),where=k1>1)
        # All 2n actions are retained, masked actions have zero probability.
        j=np.tile(np.arange(n),2); direction=np.repeat([0,1],n)
        u=np.where(direction[None,:]==1,i[:,None],j[None,:])
        v=np.where(direction[None,:]==1,j[None,:],i[:,None])
        rows=np.arange(b)[:,None]
        start=x[rows,u]; end=x[rows,v]
        bend=np.stack([start[:,:,0],end[:,:,1]],axis=-1)
        a=np.abs(bend-start).sum(-1);c=np.abs(end-bend).sum(-1);half=(a+c)/2
        frac1=np.divide(half,a,out=np.zeros_like(a),where=a>0)
        frac2=np.divide(half-a,c,out=np.zeros_like(c),where=c>0)
        middle=np.where((half<=a)[...,None],start+(bend-start)*frac1[...,None],
                        bend+(end-bend)*frac2[...,None])
        route=np.stack([start,middle,end],axis=2).astype(np.float64)
        direct=np.abs(route[:,:,None]-route[:,None,:]).sum(-1).mean(-1)
        reverse=np.abs(route[:,:,None]-route[:,None,:,::-1]).sum(-1).mean(-1)
        dist=np.minimum(direct,reverse)/scale[:,None,None]
        pp=p[:,:,None]*p[:,None,:]
        logp=np.log(np.clip(p,1e-300,None))
        q=(pp*dist).sum((1,2))
        g=(-.5*pp*dist*(logp[:,:,None]+logp[:,None,:])).sum((1,2))
        cc=(p*dist[np.arange(b),:,chosen]).sum(-1)
        result.append(np.stack([entropy(p)/np.log(k),p.max(-1),q,g,cc,h1],axis=-1))
    return np.stack(result,axis=1)
class Runner:
    def __init__(self,model,batch):
        self.batch=batch;self.device=torch.device('cuda:0')
        cls=import_rest();self.net=cls(model['train_degree'],self.device,width=model['width'],depth=model['depth'],inner=model['inner'])
        path=Path(model['path'])
        if path.stat().st_size!=model['bytes'] or path.stat().st_mtime_ns!=model['mtime_ns']:
            raise RuntimeError('Frozen catalog checkpoint changed')
        cp=torch.load(path,map_location='cpu',weights_only=False)
        self.net.load_state_dict(extract_state(cp,model),strict=True);self.net.eval()
        self.first=[];self.second=[]
        self.hooks=[self.net.ptr1.register_forward_hook(lambda m,a,o:self.first.append(o.detach().cpu().numpy().astype(np.float64))),
                    self.net.ptr2.register_forward_hook(lambda m,a,o:self.second.append(o.detach().cpu().numpy().astype(np.float64)))]
    def predict(self,x):
        keep=len(x);self.net.degree=x.shape[1]
        if keep<self.batch:x=np.concatenate([x,np.repeat(x[-1:],self.batch-keep,axis=0)])
        self.first.clear();self.second.clear()
        with torch.inference_mode():topology,_=self.net(x,True)
        score=summarize_steps(x,self.first,self.second)
        return lengths(x[:keep],topology[:keep].cpu().numpy()),score[:keep]
def rank(x):
    order=np.argsort(x,kind='stable');out=np.empty(len(x),float)
    sx=x[order]; starts=np.r_[0,np.flatnonzero(np.diff(sx))+1]; ends=np.r_[starts[1:],len(x)]
    for a,b in zip(starts,ends):out[order[a:b]]=(a+b-1)/2
    return out
def corr(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])
def ci(values):
    a=np.array([v for v in values if v is not None],float)
    return [float(v) for v in np.quantile(a,[.025,.975])] if len(a) else None
def statistics(scores,regret,seed):
    rng=np.random.default_rng(seed);n=len(regret);top=max(1,int(np.ceil(n*.1)))
    high=regret>=np.quantile(regret,.9)
    output={}
    for j,name in enumerate(SCORES):
        # maxp remains a confidence control; low confidence is used for selection.
        s=scores[:,j];u=-s if name=='maxp' else s
        selected=np.argsort(-u,kind='stable')[:top]
        bootr=[];boots=[];bootp=[]
        for _ in range(500):
            ix=rng.integers(0,n,n);xx=s[ix];yy=regret[ix]
            bootr.append(corr(xx,yy));boots.append(corr(rank(xx),rank(yy)))
            uu=-xx if name=='maxp' else xx
            bootp.append(float((yy>=np.quantile(yy,.9))[np.argsort(-uu,kind='stable')[:top]].mean()))
        output[name]=dict(pearson=corr(s,regret),spearman=corr(rank(s),rank(regret)),
                         pearson_ci95=ci(bootr),spearman_ci95=ci(boots),
                         precision_top10=float(high[selected].mean()),precision_ci95=ci(bootp),
                         high_error_prevalence=float(high.mean()),selected_count=top,
                         selection='lowest maxp' if name=='maxp' else 'highest score',
                         undefined_reason='constant score or error' if corr(s,regret) is None else None)
    return output
def hard_selection(scores,regret,seed):
    rng=np.random.default_rng(seed);out={}
    # Disjoint candidate blocks; same pools and random choices for all methods.
    order=rng.permutation(len(regret))
    for budget in (8,16):
        blocks=order[:len(order)//budget*budget].reshape(-1,budget)
        r=regret[blocks];rand=rng.integers(0,budget,len(blocks))
        random=r[np.arange(len(blocks)),rand]
        methods={'random':random,'oracle_upper_bound':r.max(-1)}
        for j,name in enumerate(SCORES):
            s=scores[blocks,j];s=-s if name=='maxp' else s
            methods[name]=r[np.arange(len(blocks)),s.argmax(-1)]
        rows={}
        for name,values in methods.items():
            delta=values-random
            bootstrap=[float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(500)]
            rows[name]=dict(mean_selected_true_regret=float(values.mean()),delta_vs_random=float(delta.mean()),delta_ci95=ci(bootstrap))
        out[str(budget)]=dict(blocks=len(blocks),methods=rows)
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',default='M0900,M1118')
    ap.add_argument('--datasets',default='D0218,D0223,D0228,D0233')
    ap.add_argument('--count',type=int,default=512);ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--seed',type=int,default=20260930);ap.add_argument('--gpu',default='4')
    args=ap.parse_args()
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.10)
    models={m['id']:m for m in read(BENCH/'artifacts/models.json')}
    datasets={d['id']:d for d in read(BENCH/'artifacts/datasets.json')}
    cfg=vars(args)|dict(scores=SCORES,temperature=1.,dtype='float32',scope='conditional second endpoint and direction on greedy trajectory',model_metadata=[models[k] for k in args.models.split(',')],dataset_metadata=[datasets[k] for k in args.datasets.split(',')])
    config=out/'run_config.json'
    if config.exists() and read(config)!=cfg:raise RuntimeError('config differs; use a new output directory')
    atomic(config,cfg);start=time.time();processed=0
    total=0
    for mid in args.models.split(','):
        for did in args.datasets.split(','):
            folder=out/mid/did
            if (folder/'summary.json').exists():continue
            count=min(args.count,datasets[did]['count'])
            for a in range(0,count,args.batch):
                b=min(a+args.batch,count)
                if not (folder/f'{a:06d}_{b:06d}.npz').exists():total+=b-a
    for mi,mid in enumerate(args.models.split(',')):
        runner=Runner(models[mid],args.batch)
        for di,did in enumerate(args.datasets.split(',')):
            folder=out/mid/did;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'summary.json').exists():continue
            d=datasets[did]
            if not d.get('reference_available'):raise RuntimeError('dataset has no trusted reference')
            with np.load(d['path'],allow_pickle=False) as z:
                # Subsample depends only on seed and instance identity, never labels.
                order=np.random.default_rng(args.seed+di).permutation(len(z['pins']))[:args.count]
                x=z['pins'][order];ref=z['reference'][order];ids=z['net_ids'][order]
            if not np.isfinite(ref).all() or (ref<=0).any():raise RuntimeError('missing true reference')
            shards=[]
            for a in range(0,len(x),args.batch):
                b=min(a+args.batch,len(x));path=folder/f'{a:06d}_{b:06d}.npz';shards.append(path)
                if path.exists():continue
                length,steps=runner.predict(x[a:b])
                npz(path,length=length,reference=ref[a:b],net_ids=ids[a:b],indices=order[a:b],step_scores=steps,mean_scores=steps.mean(1))
                processed+=b-a;elapsed=time.time()-start
                atomic(folder/'cursor.json',dict(committed=b,total=len(x),shards=[v.name for v in shards]))
                atomic(out/'heartbeat.json',dict(pid=os.getpid(),status='inference',model=mid,dataset=did,committed=b,total=len(x),gpu=args.gpu,batch=args.batch,rows_this_invocation=processed,elapsed_seconds=elapsed,rows_per_second=processed/elapsed,remaining_seconds_estimate=(total-processed)/(processed/elapsed)))
                print(json.dumps(dict(model=mid,dataset=did,committed=b,total=len(x),elapsed_seconds=elapsed)),flush=True)
            atomic(out/'heartbeat.json',dict(pid=os.getpid(),status='statistics',model=mid,dataset=did,gpu=args.gpu,batch=args.batch,committed=len(x)))
            zz=[np.load(p,allow_pickle=False) for p in shards]
            length=np.concatenate([z['length'] for z in zz]);reference=np.concatenate([z['reference'] for z in zz])
            scores=np.concatenate([z['mean_scores'] for z in zz]);steps=np.concatenate([z['step_scores'] for z in zz])
            for z in zz:z.close()
            if not np.isfinite(reference).all() or (reference<=0).any():raise RuntimeError('untrusted/missing true reference')
            regret=length/reference-1
            if regret.min() < -1e-5:raise RuntimeError('negative regret: incompatible reference convention')
            summary=dict(model=mid,dataset=did,n=len(regret),degree=x.shape[1],mean_regret=float(regret.mean()),p95_regret=float(np.quantile(regret,.95)),
                         metrics=statistics(scores,regret,args.seed),hard_selection=hard_selection(scores,regret,args.seed),
                         phase_metrics={name:statistics(steps[:,ix].mean(1),regret,args.seed) for name,ix in zip(['early','middle','late'],np.array_split(np.arange(steps.shape[1]),3))})
            atomic(folder/'summary.json',summary)
            atomic(out/'heartbeat.json',dict(pid=os.getpid(),status='dataset_complete',model=mid,dataset=did,gpu=args.gpu,batch=args.batch,committed=len(x)))
        for hook in runner.hooks:hook.remove()
        del runner;torch.cuda.empty_cache()
    atomic(out/'heartbeat.json',dict(pid=os.getpid(),status='complete',gpu=args.gpu,batch=args.batch,elapsed_seconds=time.time()-start))
    from report import render
    render()
    atomic(out/'complete.json',dict(status='complete',elapsed_seconds=time.time()-start))
if __name__=='__main__':main()

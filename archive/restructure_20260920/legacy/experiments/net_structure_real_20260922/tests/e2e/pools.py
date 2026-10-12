"""Fixed local candidate banks, validation/test geometry diagnostics, actor-independent."""
from exp_common import *
from candidates import legalize,save_geometry
from src.costs import hungarian_l1_mean
from structure import features,graph_noise,tile_project
import argparse

def gate_fit():
    p=ART/'gate.json'
    if p.exists():return read_json(p)
    data=load_data();train=np.stack([features(x) for x in data['pins'][data['split']=='A_train']]);cal=np.stack([features(x) for x in data['pins'][data['split']=='A_cal']])
    center=np.median(train,axis=0);scale=np.maximum(np.median(np.abs(train-center),axis=0)*1.4826,.02)
    score=np.max(np.abs((cal-center)/scale),axis=1);r=dict(center=center,scale=scale,threshold=float(np.quantile(score,.99)),calibration='A_cal 99% empirical threshold; no coverage theorem')
    atomic_json(p,r);return r

def make(method,generator=301):
    name=f'{method}_g{generator}' if method in ['set_dae','graph_dae','graph_tile'] else method
    if (ART/'done'/f'{name}_geometry.json').exists():return
    start=time.monotonic();data=load_data();physical=np.load(ART/'physical.npz');idx=np.load(ART/'training_centers.npz')['indices']
    model=None;seed_all(generator)
    if method in ['set_dae','graph_dae','graph_tile']:
        import torch
        from generator import load_model
        torch.cuda.set_per_process_memory_fraction(CFG['runtime']['gpu_memory_fraction'])
        model,_=load_model('set' if method=='set_dae' else 'graph',generator)
    gate=gate_fit()
    rng=np.random.default_rng(20260922);amplitudes=np.array([.01,.02,.04,.08,.04,.04,.04]);rows=[]
    def generate(ids,label):
        x=data['pins'][ids];noise=rng.normal(size=(len(x),7,30,2)).astype(np.float32)*amplitudes[None,:,None,None]
        # Same Gaussian draws and amplitudes for learned-vs-tile_raw comparisons, across generator seeds.
        proposals=x[:,None]+noise
        if method=='feature_gate':
            if label=='A_train':
                with np.load(ART/'pools/raw_wdro.npz') as f:proposals=f['pins'][:,1:].copy()
            else:
                from candidates import raw_bank
                proposals=raw_bank(x,20260921,name+'_raw_diagnostic')[0][:,1:]
        if model is not None:
            import torch
            flat=proposals.reshape(-1,30,2);sigma=np.tile(amplitudes,len(x));out=[]
            with torch.no_grad():
                for b in range(0,len(flat),256):out.append(model(torch.tensor(flat[b:b+256],device='cuda',dtype=torch.float32),torch.tensor(sigma[b:b+256],device='cuda',dtype=torch.float32)).cpu().numpy())
            decoded=np.concatenate(out).reshape(proposals.shape);proposals=.5*proposals+.5*decoded
        bank=np.repeat(x[:,None],8,axis=1);cost=np.zeros((len(x),8));repair=np.zeros_like(cost,dtype=bool);disp=np.zeros_like(cost);unique=[];fallback=0;backtracks=0;cross=0
        def score(z):return np.max(np.abs((features(z)-gate['center'])/gate['scale']))
        for j,(orig,i) in enumerate(zip(x,ids)):
            for k in range(7):
                proposed=proposals[j,k]
                if method=='laplacian':proposed=orig+graph_noise(orig,noise[j,k])
                if method in ['tile_raw','graph_tile']:
                    y,fraction=tile_project(orig,proposed,physical['lower'][i],physical['upper'][i]);backtracks+=int(fraction<1);fallback+=int(fraction==0)
                    cross+=int(np.any(y<physical['lower'][i]-6e-8) or np.any(y>physical['upper'][i]+6e-8));y=canonical(y);r=False;d=0.
                else:
                    y,r,d=legalize(proposed)
                    if method=='feature_gate' and score(y)>gate['threshold']:
                        accepted=False
                        for fraction in [.5,.25,.125]:
                            z,rr,dd=legalize(orig+fraction*(proposed-orig))
                            if score(z)<=gate['threshold']:y,r,d=z,rr,dd;accepted=True;break
                        if not accepted:y=orig;fallback+=1
                bank[j,k+1]=y;cost[j,k+1]=hungarian_l1_mean(orig,y);repair[j,k+1]=r;disp[j,k+1]=d
            unique.append(len({a.tobytes() for a in bank[j]}))
            if j%128==0:progress(name+'_geometry',subset=label,completed=j,total=len(x),seconds=time.monotonic()-start)
        if label!='A_train':
            fs=np.stack([features(z) for z in bank.reshape(-1,30,2)]).reshape(len(x),8,-1)
            atomic_npz(ART/'diagnostics'/f'{name}_{label}_features.npz',features=fs,gate_score=np.max(np.abs((fs-gate['center'])/gate['scale']),axis=-1),threshold=gate['threshold'])
        rows.append(dict(method=name,subset=label,centers=len(x),mean_unique_K=float(np.mean(unique)),cost_mean=float(cost[:,1:].mean()),cost_q50=float(np.median(cost[:,1:])),cost_q95=float(np.quantile(cost[:,1:],.95)),fallback=fallback,backtracks=backtracks,tile_violations=cross,layer_policy='original per-pin layers retained as metadata' if method in ['tile_raw','graph_tile'] else 'not constrained'))
        return bank,cost,repair,disp,np.array(unique)
    result=generate(idx,'A_train');save_geometry(name,*result,idx,generator=generator,seconds=time.monotonic()-start)
    # Geometry diagnostics on fixed 32 centers/design; no test scores used to select anything.
    for label in ['A_val','B_test']:
        ids=[]
        for design in np.unique(data['design_ids'][data['split']==label]):
            subset=np.flatnonzero((data['split']==label)&(data['design_ids']==design));ids.extend(subset[:32])
        result=generate(np.array(ids),label);atomic_npz(ART/'diagnostics'/f'{name}_{label}.npz',pins=result[0],cost=result[1],center_index=ids)
    write_csv(ART/'diagnostics'/f'{name}_geometry.csv',rows);mark_done(name+'_geometry',rows=rows,seconds=time.monotonic()-start)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--generator',type=int,default=301);a=p.parse_args();make(a.method,a.generator)

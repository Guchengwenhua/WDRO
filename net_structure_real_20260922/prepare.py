"""Read-only historical inputs; isolated multidegree and physical metadata caches."""
from exp_common import *
from concurrent.futures import ProcessPoolExecutor
import hashlib, shutil

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def physical_design(folder):
    folder=Path(folder);cache=np.load(folder/'degree_30.npz');ids=cache['net_ids'].astype(str)
    wanted={s.split(':n')[0] if False else int(i):j for j,i in enumerate(cache['source_index'])}
    manifest=read_json(folder/'manifest.json');source=Path(manifest['binding']['design']['path'])
    found={};layers_out={};raw_out={}
    with source.open() as f:
        grid=np.array(f.readline().split()[1:],int)
        header=[f.readline().split() for _ in range(5)]
        line=f.readline()
        while not line.strip():line=f.readline()
        tile=np.array(line.split(),float)
        for line in f:
            if line.startswith('num net'):nn=int(line.split()[2]);break
        for i in range(nn):
            line=f.readline()
            while not line.strip():line=f.readline()
            name,ni,degree,width=line.split();degree=int(degree)
            if i not in wanted:
                for _ in range(degree):f.readline()
                continue
            raw=np.array([list(map(float,f.readline().split())) for _ in range(degree)])
            j=wanted[i];assert ids[j].endswith(f':{name}::{ni}:d30')
            normalized=(raw[:,:2]-cache['offsets'][j])*cache['scales'][j]+.1
            order=np.lexsort((normalized[:,1],normalized[:,0]));x=normalized[order].astype(np.float32)
            np.testing.assert_array_equal(x,canonical(cache['pins'][j]))
            physical=raw[order,:2];cells=np.floor((physical-tile[:2])/tile[2:]).astype(int)
            if (cells<0).any() or (cells>=grid[:2]).any():raise ValueError('outside routing grid')
            lower=(tile[:2]+cells*tile[2:]-cache['offsets'][j])*cache['scales'][j]+.1
            upper=lower+tile[2:]*cache['scales'][j]
            found[ids[j]]=(x,lower,upper,raw[order,2].astype(int),physical,cache['scales'][j],cache['offsets'][j],cells)
    if len(found)!=len(ids):raise ValueError('missing physical metadata')
    return str(folder.name),found,dict(path=str(source),sha256=sha(source),grid=grid,tile=tile,header=header)

def main():
    if (ART/'done/data.json').exists():return
    ART.mkdir(parents=True,exist_ok=True)
    old=Path(CFG['old_artifacts']);bindings=[]
    for name in ['data.npz','training_centers.npz','rho.json','splits.csv','split_summary.json']:
        shutil.copy2(old/name,ART/name);bindings.append(dict(path=str(old/name),sha256=sha(old/name)))
    data=load_data();rows=[];parts={}
    for d in CFG['generator']['degrees']:
        for split in ['A_train','A_cal','A_val','B_test']:
            pp=[];dd=[];ii=[]
            for family in CFG['splits'][split]:
                p=Path(CFG['public_cache'])/f'ispd2008_gr_{family}'/f'degree_{d}.npz'
                if not p.exists():continue
                with np.load(p) as f:
                    pins=np.stack([canonical(x) for x in f['pins']]);pp.extend(pins);dd.extend([family]*len(pins));ii.extend(f['net_ids'].astype(str))
                bindings.append(dict(path=str(p),sha256=sha(p)))
            parts[d,split]=[np.stack(pp),np.array(dd),np.array(ii)]
        # Remove exact cross-split copies, prioritizing test over validation over calibration over train.
        seen=set()
        for split in ['B_test','A_val','A_cal','A_train']:
            x,ds,ids=parts[d,split];keys=[a.tobytes() for a in x];keep=np.array([k not in seen for k in keys]);seen.update(keys)
            x,ds,ids=x[keep],ds[keep],ids[keep]
            atomic_npz(ART/'multidegree'/f'd{d}_{split}.npz',pins=x,design=ds,net_ids=ids)
            for design in np.unique(ds):rows.append(dict(degree=d,split=split,design=design,nets=int((ds==design).sum())))
    write_csv(ART/'data_counts.csv',rows)
    counts={s:sum(r['nets'] for r in rows if r['split']==s) for s in ['A_train','A_cal','A_val','B_test']}
    families=sum([CFG['splits'][s] for s in ['A_train','A_cal','A_val','B_test']],[])
    records={};physical_sources=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name,found,binding in pool.map(physical_design,[str(Path(CFG['public_cache'])/f'ispd2008_gr_{f}') for f in families]):
            records.update(found);physical_sources.append(binding);print('PHYSICAL',name,len(found),flush=True)
    nn=len(data['pins']);lo=np.full((nn,30,2),np.nan);hi=lo.copy();layers=np.zeros((nn,30),int);scale=np.full(nn,np.nan);offset=np.full((nn,2),np.nan);cells=np.zeros((nn,30,2),int)
    for i,n in enumerate(data['net_ids']):
        key=n.removeprefix('public::')
        if key not in records:continue
        x,lo[i],hi[i],layers[i],raw,scale[i],offset[i],cells[i]=records[key];np.testing.assert_array_equal(x,data['pins'][i])
    for s in ['A_train','A_cal','A_val','B_test']:
        assert np.isfinite(lo[data['split']==s]).all()
    atomic_npz(ART/'physical.npz',lower=lo,upper=hi,layer=layers,scale=scale,offset=offset,cells=cells)
    for seed in CFG['seeds']:
        seq=np.random.default_rng(seed).integers(0,int((data['split']=='A_train').sum()),size=(CFG['training']['updates'],64))
        atomic_npz(ART/'indices'/f'seed{seed}.npz',indices=seq)
    for method in ['raw_wdro','vae_shell_wdro']:
        p=old/'pools'/f'{method}.npz';target=ART/'pools'/f'{method}.npz';target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
        bindings.append(dict(path=str(p),sha256=sha(p)))
        mark_done(method+'_reference',reused=str(p),sha256=sha(p))
    atomic_json(ART/'input_manifest.json',dict(time=utc(),sources=bindings,physical_sources=physical_sources,multidegree_counts=counts))
    mark_done('data',multidegree_counts=counts,degree30_splits={s:int((data['split']==s).sum()) for s in np.unique(data['split'])})
if __name__=='__main__':main()

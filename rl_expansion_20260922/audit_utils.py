import csv
import fcntl
import json
import numpy as np

def independent_span(pins,topology):
    """Independent masked interval reduction, separate from scatter-based scorer."""
    result=[]
    for start in range(0,len(pins),128):
        x=np.asarray(pins[start:start+128],dtype=np.float64)
        edges=topology[start:start+128].reshape(len(x),-1,2)
        row=np.arange(len(x))[:,None]; total=np.zeros(len(x))
        for anchor in range(x.shape[1]):
            vertical=np.where(edges[:,:,0]==anchor,x[row,edges[:,:,1],1],x[:,anchor,1,None])
            horizontal=np.where(edges[:,:,1]==anchor,x[row,edges[:,:,0],0],x[:,anchor,0,None])
            total+=np.maximum(vertical.max(1),x[:,anchor,1])-np.minimum(vertical.min(1),x[:,anchor,1])
            total+=np.maximum(horizontal.max(1),x[:,anchor,0])-np.minimum(horizontal.min(1),x[:,anchor,0])
        result.append(total)
    return np.concatenate(result)

def metric(v,name):
    # v shape: seed, group, net. Worst-group is per trained model before seed mean.
    means=v.mean(-1)
    return means.mean(1) if name=='mean' else means.max(1)

def compare(base,candidate,name,replicates,rng):
    b=metric(base,name); c=metric(candidate,name)
    point=float(b.mean()-c.mean()); deltas=[]
    for _ in range(replicates):
        seed_idx=rng.integers(len(base),size=len(base))
        net_idx=rng.integers(base.shape[-1],size=(base.shape[1],base.shape[-1]))
        # Each bootstrap replicate uses paired nets for all selected training seeds.
        bb=base[seed_idx[:,None,None],np.arange(base.shape[1])[None,:,None],net_idx[None,:,:]]
        cc=candidate[seed_idx[:,None,None],np.arange(base.shape[1])[None,:,None],net_idx[None,:,:]]
        deltas.append(float((metric(bb,name)-metric(cc,name)).mean()))
    return dict(baseline_value=float(b.mean()),candidate_value=float(c.mean()),improvement=point,
        improvement_percent=100*point/float(b.mean()),improved_seeds=int(np.sum(c<b)),
        seed_improvement=(b-c).tolist(),ci95=np.quantile(deltas,[.025,.975]).tolist(),
        simultaneous_ci99=np.quantile(deltas,[.005,.995]).tolist())

def csvwrite(path,rows):
    if not rows: return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def lock_free(path):
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(f,fcntl.LOCK_UN)

def calls(path):
    starts={}; ends=set()
    for line in path.read_text().splitlines():
        r=json.loads(line)
        if r['event']=='start':
            if r['id'] in starts: raise AssertionError('duplicate call id')
            starts[r['id']]=r
        else:
            if r['id'] not in starts or r['id'] in ends: raise AssertionError('invalid completed call')
            ends.add(r['id'])
    return {phase:dict(completed=sum(x['rows'] for k,x in starts.items() if k in ends and x['phase']==phase),
                        attempted_upper=sum(x['rows'] for x in starts.values() if x['phase']==phase))
            for phase in ('sampled','rollout','validation','test')}


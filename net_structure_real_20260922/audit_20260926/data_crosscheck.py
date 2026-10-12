from common import *
if __name__=='__main__':
 rows=[]
 for d in CFG['generator']['degrees']:
  seen=set()
  for split in ['B_test','A_val','A_cal','A_train']:
   xs=[];ids=[];families=[]
   for family in CFG['splits'][split]:
    p=Path(CFG['public_cache'])/f'ispd2008_gr_{family}'/f'degree_{d}.npz'
    if not p.exists():continue
    f=npz(p)
    for net,x in zip(f['net_ids'].astype(str),f['pins']):ids.append(net);xs.append(x[np.lexsort((x[:,1],x[:,0]))]);families.append(family)
   keys=[x.tobytes() for x in xs];keep=[i for i,k in enumerate(keys) if k not in seen];seen.update(keys);actual=npz(ART/'multidegree'/f'd{d}_{split}.npz')
   assert np.array_equal(actual['pins'],np.stack([xs[i] for i in keep]));assert np.array_equal(actual['net_ids'],np.array([ids[i] for i in keep]));assert np.array_equal(actual['design'],np.array([families[i] for i in keep]))
   rows.append(dict(degree=d,split=split,nets=len(keep),source_nets=len(xs),dropped=len(xs)-len(keep)))
 write(OUT/'multidegree_source.json',dict(status='PASS',nets=sum(r['nets'] for r in rows),rows=rows,time=now()));print('SOURCE_RECONSTRUCTION PASS',sum(r['nets'] for r in rows),flush=True)

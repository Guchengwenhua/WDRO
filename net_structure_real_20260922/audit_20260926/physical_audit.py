from common import *
from concurrent.futures import ProcessPoolExecutor

def one(source):
 data=npz(ART/'data.npz');phys=npz(ART/'physical.npz');path=Path(source['path']);family=path.name.split('.')[0];ids=np.flatnonzero(data['family_ids']==family);byid={str(data['net_ids'][i]):int(i) for i in ids};seen=set();maxcoord=0.;maxbounds=0.;layers=[]
 with path.open() as f:
  grid=[int(x) for x in next(f).split()[1:]]
  for _ in range(5):next(f)
  for line in f:
   if line.strip():origin=np.array(list(map(float,line.split())));break
  for line in f:
   if line.split()[:2]==['num','net']:n=int(line.split()[2]);break
  for _ in range(n):
   line=next(f)
   while not line.strip():line=next(f)
   name,index,degree,width=line.split();degree=int(degree);key=f'public::public:ispd2008_gr_{family}:{name}::{index}:d30'
   if key not in byid:
    for k in range(degree):next(f)
    continue
   raw=np.array([list(map(float,next(f).split())) for k in range(degree)]);i=byid[key];seen.add(i);xy=raw[:,:2];offset=xy.min(0);scale=.8/(xy.max(0)-offset).max();order=np.lexsort((xy[:,1],xy[:,0]));xy=xy[order];lay=raw[order,2].astype(int);normalized=.1+(xy-offset)*scale;cells=np.floor((xy-origin[:2])/origin[2:]).astype(int);lo=.1+(origin[:2]+cells*origin[2:]-offset)*scale;hi=lo+origin[2:]*scale
   maxcoord=max(maxcoord,float(abs(normalized-data['pins'][i]).max()));maxbounds=max(maxbounds,float(abs(lo-phys['lower'][i]).max()),float(abs(hi-phys['upper'][i]).max()))
   assert np.array_equal(lay,phys['layer'][i]);assert np.array_equal(cells,phys['cells'][i]);np.testing.assert_allclose(offset,phys['offset'][i],atol=0,rtol=0);assert scale==phys['scale'][i];layers.extend(lay.tolist())
 assert seen==set(ids);assert maxcoord<3.1e-8 and maxbounds==0
 return dict(family=family,nets=len(seen),pin_coordinate_max_roundoff=maxcoord,bounds_max_error=maxbounds,layers=sorted(set(layers)))
if __name__=='__main__':
 sources=read(ART/'input_manifest.json')['physical_sources']
 with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,sources))
 csvout('physical_source_reparse.csv',rows);write(OUT/'physical_source.json',dict(status='PASS',nets=sum(r['nets'] for r in rows),designs=len(rows),rows=rows,time=now()));print('PHYSICAL PASS',sum(r['nets'] for r in rows),flush=True)

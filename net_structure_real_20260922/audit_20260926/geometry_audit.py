from common import *
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor
import time

def audit_pool(name):
 start=time.monotonic();p=npz(ART/'pools'/f'{name}.npz');data=npz(ART/'data.npz');center=p['center_index'];x=data['pins'][center];bank=p['pins'];recomputed=np.zeros(bank.shape[:2]);minspace=10.;canonical=True
 for b in range(0,len(bank),128):
  flat=bank[b:b+128].reshape(-1,30,2);delta=np.abs(flat[:,:,None]-flat[:,None,:]).sum(-1);delta[:,np.arange(30),np.arange(30)]=np.inf;minspace=min(minspace,float(delta.min()))
 for i,xx in enumerate(x):
  for k,z in enumerate(bank[i]):
   d=np.abs(xx[:,None,:].astype(float)-z[None,:,:]).sum(-1);rr,cc=linear_sum_assignment(d);recomputed[i,k]=d[rr,cc].mean()
   canonical=canonical and np.array_equal(z,z[np.lexsort((z[:,1],z[:,0]))])
 unique=np.array([len({z.tobytes() for z in row}) for row in bank]);result=dict(pool=name,centers=len(x),candidates=int(np.prod(bank.shape[:2])),sha256=sha(ART/'pools'/f'{name}.npz'),cost_max_error=float(abs(recomputed-p['cost']).max()),min_spacing=minspace,canonical=canonical,mean_unique_K=float(unique.mean()),min_unique_K=int(unique.min()),mean_non_nominal_cost=float(recomputed[:,1:].mean()),median_non_nominal_cost=float(np.median(recomputed[:,1:])),p95_non_nominal_cost=float(np.quantile(recomputed[:,1:],.95)),nominal_copy_fraction=float(np.mean(np.all(bank[:,1:]==bank[:,0,None],axis=(2,3)))),reference_invalid_fraction=float((~p['valid']).mean()))
 assert result['cost_max_error']<1e-12 and minspace>=1e-6-1e-9 and canonical
 if name=='tile_raw' or name.startswith('graph_tile'):
  w=npz(ART/'acceptance'/f'{name}_physical_witness.npz');phys=npz(ART/'physical.npz');perm=w['original_to_candidate'].astype(int);assert np.all(np.sort(perm,axis=-1)==np.arange(30));assert np.array_equal(w['original_layers'],phys['layer'][center]);ordered=bank[np.arange(len(bank))[:,None,None],np.arange(8)[None,:,None],perm]
  violation=np.maximum(phys['lower'][center,None]-ordered,ordered-phys['upper'][center,None]);result['tile_max_normalized_boundary_roundoff']=float(max(0,violation.max()));assert violation.max()<=6e-8
  assert np.array_equal(bank.min(2),np.repeat(x.min(1)[:,None],8,axis=1));assert np.array_equal(bank.max(2),np.repeat(x.max(1)[:,None],8,axis=1))
  result['physical_witness_candidates']=int(np.prod(bank.shape[:2]));result['original_pin_layers']=sorted(np.unique(w['original_layers']).tolist())
 # Reference dual-scale spot checks are independent of the existing sqlite cache.
 import src.geosteiner as geo
 oracle=geo.GeoSteinerEvaluator(CFG['library']);rng=np.random.default_rng(771);sample=rng.choice(bank.shape[0]*bank.shape[1],min(32,bank.shape[0]*bank.shape[1]),replace=False);maxref=0.;scaledev=0.
 for j in sample:
  z=bank.reshape(-1,30,2)[j];geo.SCALE=2**27;a=oracle.solve(z,require_integral=True);geo.SCALE=2**28;b=oracle.solve(z,require_integral=True);maxref=max(maxref,abs(a-p['lstar'].reshape(-1)[j]));scaledev=max(scaledev,abs(a-b))
 result.update(reference_sampled=len(sample),dual_scale_solves=len(sample)*2,reference_max_error=maxref,dual_scale_max_difference=scaledev,seconds=time.monotonic()-start)
 assert maxref<1e-8 and scaledev<1e-8
 return result

def main():
 names=['raw_wdro','vae_shell_wdro','feature_gate','laplacian','tile_raw','set_dae_g301','graph_dae_g301','graph_tile_g301','graph_tile_g302','graph_tile_g303','attack32']
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(audit_pool,names):rows.append(r);print(r,flush=True)
 csvout('candidate_geometry.csv',rows);write(OUT/'geometry.json',dict(status='PASS',time=now(),pools=rows,total_candidates=sum(r['candidates'] for r in rows),total_dual_scale_solves=sum(r['dual_scale_solves'] for r in rows)))
if __name__=='__main__':main()

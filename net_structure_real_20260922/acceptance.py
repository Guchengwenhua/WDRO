"""Independent saved-artifact checks, including a pin-to-original-tile witness."""
from exp_common import *
from scipy.optimize import linear_sum_assignment

def validate():
    data=load_data();ids=np.load(ART/'training_centers.npz')['indices'];f=np.load(ART/'physical.npz');physical={k:f[k] for k in f.files};f.close();results=[]
    names=['raw_wdro','vae_shell_wdro','feature_gate','laplacian','tile_raw','set_dae_g301','graph_dae_g301','graph_tile_g301','graph_tile_g302','graph_tile_g303']
    for name in names:
        with np.load(ART/'pools'/f'{name}.npz') as f:p={k:f[k] for k in f.files}
        assert np.array_equal(p['center_index'],ids);assert np.array_equal(p['pins'][:,0],data['pins'][ids]);assert np.array_equal(p['lstar'][:,0],data['lstar'][ids]);assert p['valid'][:,0].all()
        assert p['pins'].shape==(len(ids),8,30,2);assert np.isfinite(p['pins']).all();assert ((~p['valid'][:,1:]).mean()<=.01)
        tile=name=='tile_raw' or name.startswith('graph_tile');witness=None
        if tile:
            witness=np.empty((len(ids),8,30),np.int16)
            for j,i in enumerate(ids):
                x=data['pins'][i];lo=physical['lower'][i];hi=physical['upper'][i]
                for k,z in enumerate(p['pins'][j]):
                    # Find and persist a bijection from original labeled pins to canonical output pins.
                    allowed=((z[None]>=lo[:,None]-6e-8)&(z[None]<=hi[:,None]+6e-8)).all(-1)
                    cost=np.abs(x[:,None]-z[None]).sum(-1)+1e6*(~allowed)
                    rr,cc=linear_sum_assignment(cost);assert allowed[rr,cc].all(),(name,j,k,'tile mismatch')
                    np.testing.assert_array_equal(x.min(0),z.min(0));np.testing.assert_array_equal(x.max(0),z.max(0));witness[j,k]=cc
            # canonical_candidate[witness[i,k,pin]] belongs to original pin p's tile and keeps its layer.
            atomic_npz(ART/'acceptance'/f'{name}_physical_witness.npz',original_to_candidate=witness,original_layers=physical['layer'][ids],center_index=ids)
        results.append(dict(method=name,centers=len(ids),K=8,invalid_fraction=float((~p['valid'][:,1:]).mean()),tile_witness_checked=tile))
    run_files=list((ART/'done').glob('main_*_u2048.json'));assert len(run_files)==33
    for p in run_files:
        r=read_json(p);assert r['update']==2048 and r['ledger']['sampled_requested']==2048*64
        for t in r['tests']:
            f=np.load(t['path']);assert len(f['regret'])==5935 and np.isfinite(f['regret']).all() and f['regret'].min()>=-1e-6
    attacks=list((ART/'done').glob('attack_*_g*_s*.json'));assert len(attacks)==33
    atomic_json(ART/'acceptance/final.json',dict(status='PASS_ENGINEERING_ONLY',time=utc(),pools=results,rest_runs=len(run_files),independent_attacks=len(attacks),note='Independent invariant audit, not independent human review or scientific superiority'))
if __name__=='__main__':validate()

"""Freeze geometry-only settings before training; never read a model or an optimum."""
import json
import numpy as np
import geometry as G

def main():
    p=G.ROOT/'protocol.json'
    if p.exists():
        raise FileExistsError('Protocol already frozen')
    x=G.make_data(2048,30,'uniform','rho-calibration',711)
    d=np.random.default_rng(714).normal(0,.04,(2048,7,30,2))
    z=G.legalize(x,np.clip(d,-.08,.08))
    c=G.costs(x,z)
    G.atomic_npz(G.ROOT/'artifacts/calibration.npz',pins=x,costs=c)
    settings=dict(schema_version=1,experiment='oracle_free_v1',degree=30,seeds=[101,102,103],
        methods=['erm_mst','erm_lb','wdro_random','wdro_es','wdro_critic','wdro_rl'],
        updates=2048,batch_size=64,actor_lr=1e-4,baseline_lr=1e-4,surrogate_lr=1e-4,proposal_lr=1e-4,
        candidates=8,epsilon=.08,search_sigma=.04,es_step_size=.04,critic_steps=4,critic_step_size=.01,
        lambda_init=1.,lambda_lr=1.,rho=float(np.quantile(c,.75)),
        validation_every=128,checkpoint_every=32,val_per_distribution=256,test_per_distribution=1024,
        attack_per_distribution=128,attack_candidates=32,greedy_microbatch=64,
        distributions=list(G.PARAMETERS),parameters=G.PARAMETERS,
        initialization='fresh random Actor, matched seed; no external weights',
        loss='L/max(HPWL,Manhattan_MST/2)-1; erm_mst uses L/Manhattan_MST-1',
        model_selection='minimum validation worst-group mean q_lb; earliest ties',
        references='none, no optimal labels, no GeoSteiner library or executable',
        acceptance_scope='degree30 proof of concept, not reproduction or universal robustness proof',
        calibration=dict(count=2048,candidates_per_center=7,quantile=.75,seed=711,
            sha256=G.sha256(G.ROOT/'artifacts/calibration.npz')),
        statistics=dict(bootstrap_replicates=2000,paired=True,training_seeds=3,
            primary='raw length, q_mst, q_lb mean/P95 and worst group; before/after; cross method'))
    G.atomic_json(p,settings)
    print(json.dumps(dict(rho=settings['rho'],protocol=str(p),sha256=G.sha256(p))))

if __name__=='__main__': main()

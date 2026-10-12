"""Per-center search with separately auditable true evaluator budgets."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from src.candidates import CandidateSupport, generate_support, exact_score_support
from src.datasets import array_bundle_sha256, normalize_fixed_degree_batch
from src.geometry import pin_set_sha256, project_normalized_net, normalize_net, has_spacing_violation
from src.costs import batch_cost
from src.metrics import relative_regret


def transport_costs(centers, candidates, kind='hungarian_l1_mean'):
    b,k,n,_=candidates.shape
    return batch_cost(np.repeat(centers,k,axis=0),candidates.reshape(-1,n,2),kind=kind).reshape(b,k)


def exact_all(pins, *, actor, rest, geosteiner, cache, ledger, search):
    b,k,n,_=pins.shape;flat=pins.reshape(-1,n,2);outputs=[];was_training=actor.training
    actor.eval()
    with torch.no_grad():
        for start in range(0,len(flat),search['score_microbatch']):
            out,_=actor(flat[start:start+search['score_microbatch']],True);outputs.append(out.cpu().numpy())
    actor.train(was_training)
    lengths=rest.lengths(flat,np.concatenate(outputs));lstar,misses=cache.lengths(flat,geosteiner)
    ledger.add('actor_forward_nets',len(flat));ledger.add('exact_rest_evaluator_nets',len(flat));ledger.add('geosteiner_nets',misses)
    return relative_regret(lengths,lstar).reshape(b,k),lstar.reshape(b,k),misses


def cem_round(batch, *, update, training_seed, round_index, mean, std, seen, config):
    """Stateless seeds and canonical-coordinate motion preserve pin-order symmetry."""
    centers=batch.pins;b,n,_=centers.shape;count=16;raw=np.empty((b,count,n,2),dtype=np.float64)
    block=(update-1)//config['refresh_updates'];repairs=0
    for i,(center,net_id) in enumerate(zip(centers,batch.net_ids,strict=True)):
        order=np.lexsort((center[:,1],center[:,0]));canonical=center[order].astype(np.float64)
        digest=hashlib.sha256(b'cem-v1:'+bytes(net_id)+f':{training_seed}:{block}:{round_index}'.encode()).digest()
        rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
        motion=rng.normal(mean[i],std[i],size=(count,n,2))
        motion=np.clip(motion,-config['cem']['maximum_motion'],config['cem']['maximum_motion'])
        raw[i,:,order,:]=np.clip(canonical+motion,0.1,0.9).transpose(1,0,2)
    flat=raw.reshape(-1,n,2);normalized,_,_,valid,_=normalize_fixed_degree_batch(flat)
    for i in np.flatnonzero(~valid):
        normalized[i]=normalize_net(project_normalized_net(flat[i],minimum_l1_spacing=2e-6),expected_degree=n)[0];repairs+=1
    pins=normalized.astype(np.float32).reshape(b,count,n,2)
    if round_index==0:pins[:,0]=centers
    for i in range(b):
        for j in range(count):
            p=pins[i,j];key=pin_set_sha256(p)
            for attempt in range(32):
                if key not in seen[i] and not has_spacing_violation(p):break
                p=p.astype(np.float64);order=np.lexsort((p[:,1],p[:,0]))
                p[order[(j+round_index*16+attempt)%n],attempt%2]+=(attempt+1)*1e-4
                p=normalize_net(project_normalized_net(p,minimum_l1_spacing=2e-6),expected_degree=n)[0].astype(np.float32)
                key=pin_set_sha256(p);repairs+=1
            if key in seen[i] or has_spacing_violation(p):raise RuntimeError('CEM support legality/deduplication failure')
            pins[i,j]=p;seen[i].add(key)
    return pins,repairs


def adapt_cem(batch,pins,regrets,price,rho,config):
    costs=transport_costs(batch.pins,pins,config['ground_cost'])
    # Only the global selection applies the exact rho=0 feasible limit. The
    # proposal may explore positive costs; no additional evaluation is charged.
    scores=regrets-price*costs
    elite=np.argsort(-scores,axis=1,kind='stable')[:,:4]
    chosen=pins[np.arange(len(pins))[:,None],elite]
    offsets=np.empty_like(chosen,dtype=np.float64)
    for i,center in enumerate(batch.pins):
        order=np.lexsort((center[:,1],center[:,0]));offsets[i]=chosen[i][:,order]-center[order]
    return offsets.mean(axis=1),np.maximum(offsets.std(axis=1),config['cem']['minimum_std']),elite


def search_batch(batch, *, update, training_seed, config, price, rho, actor, rest, geosteiner, cache, ledger):
    common=dict(actor=actor,rest=rest,geosteiner=geosteiner,cache=cache,ledger=ledger,search=config)
    b,n,_=batch.pins.shape
    if config['searcher']=='random_sobol':
        support=generate_support(batch,training_seed=training_seed,update=update,search=config)
        ids,regrets,refs,detail=exact_score_support(support,**common)
        return support,ids,regrets,refs,detail
    rounds=[];misses=0;repaired=0
    if config['searcher']=='random_64':
        full=dict(config,candidate_count=63)
        support=generate_support(batch,training_seed=training_seed,update=update,search=full)
        regrets,refs,misses=exact_all(support.pins,**common);repaired=support.repaired
    elif config['searcher']=='cem_64':
        mean=np.zeros((b,n,2));std=np.full_like(mean,config['cem']['initial_std']);seen=[set() for _ in range(b)]
        all_pins=[];all_regrets=[];all_refs=[]
        for t in range(4):
            pins,repair=cem_round(batch,update=update,training_seed=training_seed,round_index=t,mean=mean,std=std,seen=seen,config=config)
            scores,reference,miss=exact_all(pins,**common)
            mean,std,elite=adapt_cem(batch,pins,scores,price,rho,config)
            rounds.append({'round':t,'elite_indices':elite.tolist(),'pins_sha256':array_bundle_sha256({'pins':pins}),'exact_calls_per_center':16})
            all_pins.append(pins);all_regrets.append(scores);all_refs.append(reference);misses+=miss;repaired+=repair
        pins=np.concatenate(all_pins,axis=1);regrets=np.concatenate(all_regrets,axis=1);refs=np.concatenate(all_refs,axis=1)
        support=CandidateSupport(pins,array_bundle_sha256({'pins':pins,'center_ids':batch.net_ids}),b*63,repaired,(update-1)//config['refresh_updates'])
    else:raise ValueError('unknown searcher')
    ids=np.tile(np.arange(64),(b,1))
    detail={'centers':b,'perturbations':b*63,'surrogate_nets':0,'exact_rescore_nets':b*64,
        'reference_requests':b*64,'reference_cache_misses':misses,'reference_cache_hits':b*64-misses,
        'repaired_candidates':repaired,'support_sha256':support.sha256,'refresh_block':support.refresh_block,
        'shortlist_sha256':array_bundle_sha256({'indices':ids}),'exact_shortlist_indices':ids.tolist(),
        'exact_relative_regrets':regrets.tolist(),'cem_rounds':rounds,'exact_calls_per_center':64}
    return support,ids,regrets,refs,detail

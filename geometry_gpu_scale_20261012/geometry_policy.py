"""GPU-only entropy ranking with fixed original+top3 exact shortlist."""
import time
import numpy as np
import torch
from adapters import SETTINGS
import src.candidates as candidates
import stage5.search as original_search
from gpu_scores import DecoderScore
ORIGINAL_SHORTLIST=candidates.shortlist_support
ORIGINAL_SEARCH=original_search.search_batch

def search_batch(batch,**kwargs):
    arm=SETTINGS['geometry_arm'];started=time.monotonic();context={}
    def shortlist(support,top_k=4):
        begin=time.monotonic();b,k,n,_=support.pins.shape
        if arm=='original':
            ids,scores=ORIGINAL_SHORTLIST(support,top_k)
        elif arm=='random':
            rng=np.random.default_rng(kwargs['training_seed']*1000003+kwargs['update'])
            scores=rng.random((b,k))
            ids=np.concatenate((np.zeros((b,1),int),np.argsort(-scores[:,1:],axis=1,kind='stable')[:,:top_k-1]+1),axis=1)
        else:
            actor=kwargs['actor'];mode=actor.training;actor.eval();scorer=DecoderScore(actor,arm)
            flat=support.pins.reshape(-1,n,2);values=[];microbatch=SETTINGS['ranking_microbatch']
            try:
                with torch.no_grad():
                    for start in range(0,len(flat),microbatch):
                        pins=flat[start:start+microbatch];scorer.start(pins)
                        actor(pins,True);values.append(scorer.result())
                scores_gpu=torch.cat(values).reshape(b,k)
                if not torch.isfinite(scores_gpu).all().item():raise RuntimeError('Nonfinite GPU entropy score')
                chosen=torch.argsort(scores_gpu[:,1:],dim=1,descending=True,stable=True)[:,:top_k-1]+1
                ids=torch.cat((torch.zeros((b,1),device=chosen.device,dtype=torch.long),chosen),dim=1).cpu().numpy()
                scores=scores_gpu.cpu().numpy()
            finally:
                scorer.close();actor.train(mode)
            kwargs['ledger'].add('actor_forward_nets',len(flat))
        context.update(shortlist_strategy=arm,geometry_metric='G',shortlist_wall_seconds=time.monotonic()-begin,
                       extra_shortlist_actor_nets=b*k if arm in ['geometry','entropy'] else 0,
                       ranking_microbatch=SETTINGS['ranking_microbatch'],scoring_device='cuda' if arm in ['geometry','entropy'] else 'cpu',
                       scoring_version='gpu_upper_triangle_fp64_v1',candidate_count=k,exact_shortlist=top_k,temperature=1.)
        return ids,scores
    candidates.shortlist_support=shortlist
    try:
        support,ids,regrets,refs,detail=ORIGINAL_SEARCH(batch,**kwargs)
        detail.update(context,search_wall_seconds=time.monotonic()-started)
        return support,ids,regrets,refs,detail
    finally:candidates.shortlist_support=ORIGINAL_SHORTLIST

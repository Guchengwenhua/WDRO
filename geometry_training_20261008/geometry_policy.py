"""Only change shortlist ranking; original support, exact oracle, dual and loss stay intact."""
import sys,time
import numpy as np
import torch
from adapters import SETTINGS,ROOT
import src.candidates as candidates
import stage5.search as original_search
sys.path.insert(0,str(ROOT/'geometry_uncertainty_20260930'))
from research import summarize_steps,SCORES
ORIGINAL_SHORTLIST=candidates.shortlist_support
ORIGINAL_SEARCH=original_search.search_batch
def search_batch(batch,**kwargs):
 arm=SETTINGS['geometry_arm']
 if arm=='original':return ORIGINAL_SEARCH(batch,**kwargs)
 actor=kwargs['actor'];ledger=kwargs['ledger'];context={}
 def shortlist(support,top_k=4):
  b,k,n,_=support.pins.shape
  if arm=='random':
   seed=kwargs['training_seed']*1000003+kwargs['update']
   scores=np.random.default_rng(seed).random((b,k));extra=0
  else:
   flat=support.pins.reshape(-1,n,2);values=[];first=[];second=[]
   hooks=[actor.ptr1.register_forward_hook(lambda m,a,o:first.append(o.detach().cpu().numpy().astype(float))),
          actor.ptr2.register_forward_hook(lambda m,a,o:second.append(o.detach().cpu().numpy().astype(float)))]
   mode=actor.training;actor.eval()
   try:
    with torch.no_grad():
     for a in range(0,len(flat),kwargs['config']['score_microbatch']):
      first.clear();second.clear();pins=flat[a:a+kwargs['config']['score_microbatch']]
      actor(pins,True)
      if arm=='entropy':
       hs=[]
       for logits in second:
        probability=np.exp(logits-np.max(logits,axis=-1,keepdims=True));probability/=probability.sum(-1,keepdims=True)
        kvalid=np.isfinite(logits).sum(-1);raw=-(probability*np.log(np.clip(probability,1e-300,None))).sum(-1)
        h=np.divide(raw,np.log(np.maximum(kvalid,2)),out=np.zeros_like(raw),where=kvalid>1)
        hs.append(h)
       score=np.zeros((len(pins),len(SCORES)));score[:,SCORES.index('Hnorm')]=np.stack(hs,axis=1).mean(1);values.append(score)
      else:values.append(summarize_steps(pins,first,second).mean(1))
   finally:
    for hook in hooks:hook.remove()
    actor.train(mode)
   column=SCORES.index('Hnorm' if arm=='entropy' else SETTINGS['geometry_metric'])
   scores=np.concatenate(values)[:,column].reshape(b,k);extra=len(flat);ledger.add('actor_forward_nets',extra)
  if not np.isfinite(scores).all():raise RuntimeError('nonfinite shortlist scores')
  indices=np.concatenate([np.zeros((b,1),int),np.argsort(-scores[:,1:],axis=1,kind='stable')[:,:top_k-1]+1],axis=1)
  context.update(shortlist_strategy=arm,geometry_metric=SETTINGS['geometry_metric'],extra_shortlist_actor_nets=extra,candidate_count=k,transport_scope='original final-coordinate Hungarian L1 soft dual; no hard-radius filter',temperature=1.)
  return indices,scores
 candidates.shortlist_support=shortlist
 try:
  support,ids,regrets,refs,detail=ORIGINAL_SEARCH(batch,**kwargs);detail.update(context)
  return support,ids,regrets,refs,detail
 finally:candidates.shortlist_support=ORIGINAL_SHORTLIST

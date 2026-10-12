"""User-authorized small correctness check, limited to reference substitution seam."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
from adapters import *
from src.trainer import derived_seed,set_all_seeds
from src.rest import build_actor,build_critic,RestTreeEvaluator
from stage5.search import search_batch,transport_costs
from stage4.dual import select_wdro,update_lambda
from src.accounting import CallLedger
from src.metrics import relative_regret

def main():
 torch.set_num_threads(1);device=torch.device('cpu');protocol=json.loads((BASE/'wdro_rest_min/configs/protocol_v1.json').read_text());search=json.loads((BASE/'wdro_rest_min/configs/stage3_baselines_v1.json').read_text())['search'];search.update(searcher='random_sobol',ground_cost='hungarian_l1_mean')
 set_all_seeds(derived_seed(protocol['protocol_id'],10,101,'actor-construction'));a=build_actor(10,device);load_trusted_legacy_actor(a,'ignored',map_location=device)
 set_all_seeds(derived_seed(protocol['protocol_id'],10,101,'critic-initialization'));c=build_critic(10,device)
 original_actor=copy.deepcopy(a.state_dict());original_critic=copy.deepcopy(c.state_dict());rest_eval=RestTreeEvaluator(LIB)
 SETTINGS.update(arm='geo',device=device,pool=[],rest=rest_eval);solver=make_geosteiner()
 batch=materialize_training_batch(protocol=protocol,degree=10,training_seed=101,update=1,batch_size=8)
 class DirectOriginalCache:
  def lengths(self,pins,evaluator):return evaluator.lengths(pins),len(pins)
 def one(cache):
  a.load_state_dict(original_actor);c.load_state_dict(original_critic);a.train();c.train();opt=torch.optim.Adam(list(a.parameters())+list(c.parameters()),lr=5e-5,eps=1e-5,weight_decay=0.);set_all_seeds(derived_seed(protocol['protocol_id'],10,101,'policy-training'))
  support,indices,regrets,refs,detail=search_batch(batch,training_seed=101,update=1,config=search,price=0.,rho=.029607180878520014,actor=a,rest=rest_eval,geosteiner=solver,cache=cache,ledger=CallLedger())
  z=support.pins[np.arange(8)[:,None],indices];cost=transport_costs(batch.pins,z,'hungarian_l1_mean');selected,columns=select_wdro(indices,regrets,cost,0.,.029607180878520014);chosen=support.pins[np.arange(8),selected]
  outputs,lp=a(chosen,False);length=rest_eval.lengths(chosen,outputs.detach().numpy());target=torch.as_tensor(relative_regret(length,refs[np.arange(8),columns]),dtype=torch.float32);pred=c(chosen);loss=((target-pred).detach()*lp).mean()+((pred-target)**2).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),1.);torch.nn.utils.clip_grad_norm_(c.parameters(),1.);opt.step()
  return {'support':support.pins.copy(),'f':regrets.copy(),'reference':refs.copy(),'selected_Z':chosen.copy(),'loss':float(loss.detach()),'lambda_after':update_lambda(0.,float(cost[np.arange(8),columns].mean()),.029607180878520014,1.),'actor':copy.deepcopy(a.state_dict()),'critic':copy.deepcopy(c.state_dict())}
 original=one(DirectOriginalCache());adapted=one(ResearchReference(HERE/'artifacts/equivalence/reference'))
 checks={k:bool(np.array_equal(original[k],adapted[k])) for k in ['support','f','reference','selected_Z']};checks.update(loss=original['loss']==adapted['loss'],lambda_after=original['lambda_after']==adapted['lambda_after'],actor_one_update=all(torch.equal(v,adapted['actor'][k]) for k,v in original['actor'].items()),critic_one_update=all(torch.equal(v,adapted['critic'][k]) for k,v in original['critic'].items()))
 result={'status':'pass' if all(checks.values()) else 'fail','scope':'CPU batch8/d10 same input/RNG; actual original stage5 search plus original stage6 outer loss expressions versus Geo reference adapter; not full 2000-step trainer equivalence proof','checks':checks,'loss_original':original['loss'],'loss_adapted':adapted['loss'],'candidate_count':33,'shortlist':4,'seed':101,'content_hashes':'not computed; original identifier-seed hash retained'}
 (HERE/'equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()

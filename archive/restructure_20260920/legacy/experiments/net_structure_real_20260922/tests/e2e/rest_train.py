"""Formal validation-only tuning and matched-budget main continuation."""
import argparse
import time
import numpy as np
import torch
from exp_common import *
from src.rest import build_actor,build_critic,RestTreeEvaluator
from evaluation import greedy,regret,validation,evaluate_budget

def pool_name(method,generator):
    return f'{method}_g{generator}' if method in ['set_dae','graph_dae','graph_tile'] else method

def train(method,seed,generator,target,phase,recipe,device_name):
    base=f'{method}_g{generator}_s{seed}'
    if phase=='diag':
        name=f'diag_{recipe}_{method}_s{seed}';folder=ART/'diagnostics'/f'{recipe}_{method}_s{seed}'
    else:
        recipe=read_json(ART/'recipe_selected.json')['recipe'];name=f'main_{base}_u{target}';folder=ART/'runs'/base
    if (ART/'done'/f'{name}.json').exists():return
    folder.mkdir(parents=True,exist_ok=True);setting=CFG['recipes'][recipe]
    started=time.monotonic();seed_all(seed);device=torch.device(device_name);data=load_data()
    actor=build_actor(30,device);critic=build_critic(30,device);rest=RestTreeEvaluator(CFG['library'])
    parent=CFG['parent_template'].format(seed=seed);cp=torch.load(parent,map_location='cpu',weights_only=False)
    actor.load_state_dict(cp['actor_state_dict'],strict=True);critic.load_state_dict(cp['critic_state_dict'],strict=True)
    opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=setting['lr'])
    center=np.load(ART/'training_centers.npz')['indices'];sequence=np.load(ART/'indices'/f'seed{seed}.npz')['indices']
    pool=None
    if method!='erm_a':
        with np.load(ART/'pools'/f'{pool_name(method,generator)}.npz') as f:pool={k:f[k] for k in f.files}
        if not np.array_equal(pool['center_index'],center):raise RuntimeError('pool center mismatch')
        if not np.array_equal(pool['pins'][:,0],data['pins'][center]):raise RuntimeError('nominal mismatch')
        if not np.array_equal(pool['lstar'][:,0],data['lstar'][center]):raise RuntimeError('nominal reference mismatch')
        if pool['pins'].shape!=(len(center),8,30,2) or not pool['valid'][:,0].all():raise RuntimeError('pool shape or validity mismatch')
    seed_all(seed+1000000);update=0;price=0.;history=[];vals=[];best=None;ledger={};elapsed=0.
    rho=read_json(ART/'rho.json')['rho'];latest=folder/'latest.pt'
    if latest.exists():
        cp=torch.load(latest,map_location='cpu',weights_only=False)
        if (cp['method'],cp['seed'],cp['recipe'],cp['device_type'])!=(method,seed,recipe,device_name):raise RuntimeError('resume identity mismatch')
        actor.load_state_dict(cp['actor_state_dict'],strict=True);critic.load_state_dict(cp['critic_state_dict'],strict=True)
        opt.load_state_dict(cp['optimizer_state_dict']);set_rng(cp['rng'])
        update=cp['update'];price=cp['lambda'];history=cp['history'];vals=cp['validations'];best=cp['best'];ledger=cp['ledger'];elapsed=cp['training_seconds']
    wall=time.monotonic();ledger_at_start=dict(ledger);initial_update=update
    def save(path):
        atomic_torch(path,dict(method=method,seed=seed,generator=generator,recipe=recipe,device_type=device_name,parent=parent,
            update=update,next_update_index=update,actor_state_dict=actor.state_dict(),critic_state_dict=critic.state_dict(),
            optimizer_state_dict=opt.state_dict(),rng=rng_state(),**{'lambda':price},rho=rho,history=history,
            validations=vals,best=best,ledger=ledger,training_seconds=elapsed+time.monotonic()-wall))
    if best is None:
        v=dict(update=0,**validation(actor,rest,data,ledger));vals.append(v);best=v
        save(folder/'val_best.pt');save(latest)
    (folder/'training.jsonl').write_text(''.join(json.dumps(h)+'\n' for h in history))
    while update<target:
        tick=time.monotonic();idx=sequence[update];old=price
        if method=='erm_a':
            pins=data['pins'][center[idx]];refs=data['lstar'][center[idx]];slots=np.zeros(64,int);cost=np.zeros(64)
            selgreedy=nomgreedy=maxgreedy=None
        else:
            bank=pool['pins'][idx];length=greedy(actor,rest,bank.reshape(-1,30,2),ledger,'candidate').reshape(64,8)
            valid=pool['valid'][idx];r=np.full((64,8),-np.inf);r[valid]=regret(length[valid],pool['lstar'][idx][valid])
            slots=(r-price*pool['cost'][idx]).argmax(1);rows=np.arange(64)
            pins=bank[rows,slots];refs=pool['lstar'][idx][rows,slots];cost=pool['cost'][idx][rows,slots]
            selgreedy=float(r[rows,slots].mean());nomgreedy=float(r[:,0].mean());maxgreedy=float(r.max(1).mean())
        actor.train();critic.train()
        if setting['freeze_bn']:
            for model in [actor,critic]:
                for m in model.modules():
                    if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
        topology,logp=actor(pins,False);length=rest.lengths(pins,topology.detach().cpu().numpy());sample=regret(length,refs)
        y=torch.as_tensor(sample,dtype=torch.float32,device=device);baseline=critic(pins)
        if y.shape!=baseline.shape or y.shape!=logp.shape:raise RuntimeError('Actor/Critic shape')
        aloss=((y-baseline).detach()*logp).mean();closs=(baseline-y).square().mean();loss=aloss+closs
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        an=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);cn=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.)
        if not torch.isfinite(an) or not torch.isfinite(cn):raise RuntimeError('nonfinite gradient')
        opt.step()
        if not all(torch.isfinite(p).all() for m in [actor,critic] for p in m.parameters()):raise RuntimeError('nonfinite parameters')
        for key in ['sampled_requested','sampled_forward_rows','sampled_gpu_rows' if device_name=='cuda' else 'sampled_cpu_rows']:ledger[key]=ledger.get(key,0)+64
        if method!='erm_a':price=max(0.,price+float(cost.mean())-rho)
        update+=1
        h=dict(update=update,seconds=time.monotonic()-tick,actor_loss=float(aloss.detach()),critic_loss=float(closs.detach()),
            sampled_mean_regret=float(sample.mean()),nominal_fraction=float((slots==0).mean()),selected_cost=float(cost.mean()),
            budget_gap=float(cost.mean()-rho),lambda_before=old,lambda_after=price,slot_histogram=np.bincount(slots,minlength=8).tolist(),
            selected_greedy_regret=selgreedy,nominal_greedy_regret=nomgreedy,candidate_max_regret=maxgreedy)
        history.append(h);append_json(folder/'training.jsonl',h)
        if update%CFG['training']['validation_interval_updates']==0:
            v=dict(update=update,**validation(actor,rest,data,ledger));vals.append(v)
            if v['macro']<best['macro']:best=v;save(folder/'val_best.pt')
            atomic_json(folder/'validation.json',vals)
        if update%32==0:
            save(latest);progress(name,phase=phase,method=method,seed=seed,generator=generator,recipe=recipe,update=update,
                target=target,device=device_name,seconds=elapsed+time.monotonic()-wall,best_update=best['update'],
                first32_seconds=float(np.mean([x['seconds'] for x in history[:32]])))
            print('TRAIN',name,update,target,'best',best['update'],'seconds',round(elapsed+time.monotonic()-wall,1),flush=True)
    save(latest);save(folder/f'last_{target}.pt')
    bestcp=torch.load(folder/'val_best.pt',map_location='cpu',weights_only=False);atomic_torch(folder/f'val_best_{target}.pt',bestcp)
    trainseconds=elapsed+time.monotonic()-wall;tests=[];testseconds=0.
    if phase=='main':
        tick=time.monotonic();tests=evaluate_budget(actor,rest,data,ledger,folder,method,seed,generator,target);testseconds=time.monotonic()-tick
        # Test changes the in-memory actor. Update only accounting in the saved training state.
        state=torch.load(latest,map_location='cpu',weights_only=False);state['ledger']=ledger;atomic_torch(latest,state)
    summary=dict(phase=phase,method=method,seed=seed,generator=generator,recipe=recipe,setting=setting,device_type=device_name,
        update=target,best_update=best['update'],validations=vals,last_validation=vals[-1],best_validation=best,
        training_seconds=trainseconds,new_training_seconds=trainseconds-elapsed,test_seconds=testseconds,
        process_seconds=time.monotonic()-started,initial_update=initial_update,new_updates=target-initial_update,
        ledger=ledger,ledger_delta={k:ledger[k]-ledger_at_start.get(k,0) for k in ledger},tests=tests,rho=rho,lambda_final=price,
        nominal_fraction=float(np.mean([h['nominal_fraction'] for h in history])),
        mean_selected_cost=float(np.mean([h['selected_cost'] for h in history])),mean_budget_gap=float(np.mean([h['budget_gap'] for h in history])))
    atomic_json(folder/(f'summary_{target}.json' if phase=='main' else 'summary.json'),summary);mark_done(name,**summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--generator',type=int,default=0);p.add_argument('--target',type=int,default=256)
    p.add_argument('--phase',choices=['diag','main'],default='main');p.add_argument('--recipe',default='standard_1e5')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    train(a.method,a.seed,a.generator,a.target,a.phase,a.recipe,a.device)

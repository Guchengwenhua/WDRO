"""256-update finite-candidate true-regret WDRO, with shared center schedules."""
import argparse
import time
import numpy as np
import torch
from exp_common import *
from src.rest import build_actor,build_critic,RestTreeEvaluator
from evaluate import greedy_lengths,relative_regret,validation,finish_tests

def train(method,seed,device_name='cuda'):
    name=f'train_{method}_s{seed}'
    if (ART/'done'/f'{name}.json').exists():return
    session_start=time.monotonic();seed_all(seed);data=load_data()
    folder=ART/'runs'/f'{method}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    device=torch.device(device_name);actor=build_actor(30,device);critic=build_critic(30,device)
    rest=RestTreeEvaluator(CFG['library'])
    parent=CFG['parent_template'].format(seed=seed)
    state=torch.load(parent,map_location='cpu',weights_only=False)
    actor.load_state_dict(state['actor_state_dict'],strict=True);critic.load_state_dict(state['critic_state_dict'],strict=True)
    opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=CFG['training']['learning_rate'])
    indices=np.load(ART/'indices'/f'seed{seed}.npz')['indices'];center_ids=np.load(ART/'training_centers.npz')['indices']
    pool=None
    if method!='erm_a':
        with np.load(ART/'pools'/f'{method}.npz') as f:pool={k:f[k] for k in f.files}
        if not np.array_equal(pool['center_index'],center_ids):raise RuntimeError('candidate centers differ across methods')
        if not np.array_equal(pool['pins'][:,0],data['pins'][center_ids]):raise RuntimeError('nominal slot mismatch')
        if not np.array_equal(pool['lstar'][:,0],data['lstar'][center_ids]):raise RuntimeError('nominal reference mismatch')
    rho=read_json(ART/'rho.json')['rho'];price=0.;update=0;best=None;history=[];validations=[];ledger={};elapsed=0.
    seed_all(seed+1000000)  # Action RNG is independent of pool creation and center schedules.
    if (folder/'latest.pt').exists():
        cp=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        if cp['method']!=method or cp['seed']!=seed:raise RuntimeError('resume identity mismatch')
        if cp['device_type']!=device.type:raise RuntimeError('resume must use the same device type')
        actor.load_state_dict(cp['actor_state_dict'],strict=True);critic.load_state_dict(cp['critic_state_dict'],strict=True)
        opt.load_state_dict(cp['optimizer_state_dict']);set_rng(cp['rng'])
        update=cp['update'];price=cp['lambda'];best=cp['best'];history=cp['history'];validations=cp['validations'];ledger=cp['ledger'];elapsed=cp['elapsed_seconds']
    wall=time.monotonic()
    def checkpoint(path):
        atomic_torch(path,dict(method=method,seed=seed,device_type=device.type,parent=parent,update=update,next_update_index=update,
            actor_state_dict=actor.state_dict(),critic_state_dict=critic.state_dict(),optimizer_state_dict=opt.state_dict(),
            rng=rng_state(),lambda_value=price,**{'lambda':price},rho=rho,best=best,history=history,
            validations=validations,ledger=ledger,elapsed_seconds=elapsed+time.monotonic()-wall))
    if best is None:
        val=validation(actor,rest,data,ledger);val['update']=0;validations.append(val);best=val
        checkpoint(folder/'val_best.pt');checkpoint(folder/'latest.pt')
    # Remove any uncommitted rows after a crash; they are replayed from the checkpoint.
    (folder/'training.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))
    atomic_json(folder/'validation.json',validations)
    while update<CFG['training']['updates']:
        tick=time.monotonic();chosen=indices[update];old_price=price
        if method=='erm_a':
            pins=data['pins'][center_ids[chosen]];refs=data['lstar'][center_ids[chosen]]
            slot=np.zeros(len(chosen),int);selected_cost=np.zeros(len(chosen));selected_greedy=None;nominal_greedy=None;candidate_max=None
        else:
            batch=pool['pins'][chosen];lengths=greedy_lengths(actor,rest,batch.reshape(-1,30,2),ledger,'candidate').reshape(len(chosen),8)
            valid=pool['valid'][chosen];references=pool['lstar'][chosen]
            regrets=np.full((len(chosen),8),-np.inf)
            regrets[valid]=relative_regret(lengths[valid],references[valid])
            objective=regrets-price*pool['cost'][chosen]
            # np.argmax returns the first maximum: nominal, then ascending slot.
            slot=np.argmax(objective,axis=1);row=np.arange(len(chosen))
            pins=batch[row,slot];refs=references[row,slot];selected_cost=pool['cost'][chosen][row,slot]
            selected_greedy=float(regrets[row,slot].mean());nominal_greedy=float(regrets[:,0].mean());candidate_max=float(regrets.max(1).mean())
        actor.train();critic.train();topology,log_probs=actor(pins,False)
        sampled_length=rest.lengths(pins,topology.detach().cpu().numpy());regret=relative_regret(sampled_length,refs)
        target=torch.as_tensor(regret,dtype=torch.float32,device=device);prediction=critic(pins)
        if log_probs.shape!=target.shape or prediction.shape!=target.shape:raise RuntimeError('Actor/Critic shape mismatch')
        actor_loss=((target-prediction).detach()*log_probs).mean();critic_loss=(prediction-target).square().mean()
        loss=actor_loss+critic_loss
        if not torch.isfinite(loss):raise RuntimeError('nonfinite REST loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        an=torch.nn.utils.clip_grad_norm_(actor.parameters(),CFG['training']['gradient_clip'])
        cn=torch.nn.utils.clip_grad_norm_(critic.parameters(),CFG['training']['gradient_clip'])
        if not torch.isfinite(an) or not torch.isfinite(cn):raise RuntimeError('nonfinite REST gradients')
        opt.step()
        if not all(torch.isfinite(p).all() for m in (actor,critic) for p in m.parameters()):raise RuntimeError('nonfinite REST parameters')
        ledger['sampled_requested']=ledger.get('sampled_requested',0)+len(pins)
        ledger['sampled_forward_rows']=ledger.get('sampled_forward_rows',0)+len(pins)
        backend_key='sampled_gpu_rows' if device.type=='cuda' else 'sampled_cpu_rows'
        ledger[backend_key]=ledger.get(backend_key,0)+len(pins)
        if method!='erm_a':price=max(0.,price+CFG['training']['lambda_learning_rate']*(float(selected_cost.mean())-rho))
        update+=1
        record=dict(update=update,actor_loss=float(actor_loss.detach()),critic_loss=float(critic_loss.detach()),
            sampled_mean_regret=float(regret.mean()),lambda_before=old_price,lambda_after=price,
            selected_cost=float(selected_cost.mean()),budget_gap=float(selected_cost.mean()-rho),
            nominal_fraction=float((slot==0).mean()),slot_histogram=np.bincount(slot,minlength=8).tolist(),
            selected_greedy_regret=selected_greedy,nominal_greedy_regret=nominal_greedy,candidate_max_regret=candidate_max,
            actor_grad_norm=float(an),critic_grad_norm=float(cn),seconds=time.monotonic()-tick)
        history.append(record);append_json(folder/'training.jsonl',record)
        if update%CFG['training']['validation_interval_updates']==0:
            val=validation(actor,rest,data,ledger);val['update']=update;validations.append(val)
            if val['design_macro_mean_regret']<best['design_macro_mean_regret']:
                best=val;checkpoint(folder/'val_best.pt')
            atomic_json(folder/'validation.json',validations)
        if update%CFG['training']['checkpoint_interval_updates']==0:
            checkpoint(folder/'latest.pt')
            elapsed_now=elapsed+time.monotonic()-wall
            progress(name,method=method,seed=seed,device_type=device.type,update=update,total_updates=CFG['training']['updates'],
                best_update=best['update'],lambda_value=price,elapsed_seconds=elapsed_now,
                mean_update_seconds=float(np.mean([r['seconds'] for r in history])),
                estimated_training_remaining_seconds=(CFG['training']['updates']-update)*elapsed_now/update)
            print('TRAIN',method,seed,update,'seconds',round(elapsed_now,1),'best',best['update'],flush=True)
    checkpoint(folder/'last.pt');checkpoint(folder/'latest.pt')
    training_seconds=elapsed+time.monotonic()-wall
    progress(name,method=method,seed=seed,update=update,stage='complete_natural_test',training_seconds=training_seconds)
    test_start=time.monotonic();tests=finish_tests(actor,rest,data,ledger,folder,method,seed)
    summary=dict(method=method,seed=seed,device_type=device.type,updates=update,batch_size=64,train_centers=len(center_ids),
        parent=parent,rho=rho,lambda_final=price,best_update=best['update'],best_validation=best,
        last_validation=validations[-1],training_seconds=training_seconds,
        sampled_and_candidate_update_seconds=sum(r['seconds'] for r in history),test_seconds=time.monotonic()-test_start,
        process_seconds=time.monotonic()-session_start,ledger=ledger,
        nominal_fraction=float(np.mean([r['nominal_fraction'] for r in history])),
        mean_selected_cost=float(np.mean([r['selected_cost'] for r in history])),
        mean_budget_gap=float(np.mean([r['budget_gap'] for r in history])),
        tests=tests,checkpoint_last=str(folder/'last.pt'),checkpoint_val_best=str(folder/'val_best.pt'))
    atomic_json(folder/'summary.json',summary);mark_done(name,**summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['raw_wdro','vae_shell_wdro','vae_ball_wdro','erm_a']);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args();train(args.method,args.seed,args.device)

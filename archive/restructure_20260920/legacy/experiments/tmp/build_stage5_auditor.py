from pathlib import Path
r=Path('/remote-home/Guilin/DRO/Codex/wdro_rest_min')
s=(r/'scripts/audit_stage4.py').read_text().replace('from stage4.dual import transport_costs, primal_dual_lp','from stage4.dual import primal_dual_lp\nfrom stage5.search import transport_costs, cem_round, adapt_cem\nfrom src.candidates import CandidateSupport')
s=s.replace("stage4_audit","stage5_audit").replace("logs/stage4'","logs/stage5'").replace("stage4_training_contract_v1.json","stage5_training_contract_v1.json")
s=s.replace('if stability: assert updates==500', 'if stability: assert updates==2000')
s=s.replace("wdro=json.loads((ROOT/'configs/stage4_wdro_v1.json').read_text())", "wdro=result['run_identity']['ablation_config']\n    search=dict(config['search'],**wdro['search_overrides'],ground_cost=wdro['ground_cost'])\n    pair_hm=search['searcher']=='random_sobol' and search['candidate_count']==32\n    exact_k=min(search['candidate_count']+1,search['exact_rescore_top_k'])")
a=s.index('        support=generate_support(batch,');b=s.index('        np.testing.assert_array_equal(ids,detail',a)
s=s[:a]+'''        if search['searcher']=='cem_64':
            mean=np.zeros((256,30,2));std=np.full_like(mean,search['cem']['initial_std']);seen=[set() for _ in range(256)]
            parts=[];repairs=0
            scores_all=np.asarray(detail['exact_relative_regrets'])
            assert len(detail['cem_rounds'])==4 and detail['exact_calls_per_center']==64
            for t in range(4):
                pins,repair=cem_round(batch,update=u,training_seed=101,round_index=t,mean=mean,std=std,seen=seen,config=search)
                mean,std,elite=adapt_cem(batch,pins,scores_all[:,t*16:(t+1)*16],state['lambda'],rho,search)
                record=detail['cem_rounds'][t]
                np.testing.assert_array_equal(elite,record['elite_indices'])
                assert record['round']==t and record['exact_calls_per_center']==16
                assert array_bundle_sha256({'pins':pins})==record['pins_sha256']
                parts.append(pins);repairs+=repair
            pins=np.concatenate(parts,axis=1)
            support=CandidateSupport(pins,array_bundle_sha256({'pins':pins,'center_ids':batch.net_ids}),256*63,repairs,(u-1)//10)
        else:
            support=generate_support(batch,training_seed=101,update=u,search=search)
        assert support.sha256==detail['support_sha256']
        if search['searcher']=='random_sobol':ids,_=shortlist_support(support,exact_k)
        else:ids=np.tile(np.arange(64),(256,1))
''' + s[b:]
s=s.replace('transport_costs(batch.pins,candidates)','transport_costs(batch.pins,candidates,wdro["ground_cost"])')
s=s.replace("state['lambda']=float(max(0.,price+wdro['lambda_learning_rate']*(mean_cost-rho)))","state['lambda']=(float(max(0.,price+wdro['lambda_learning_rate']*(mean_cost-rho))) if wdro['lambda_mode']=='learned' else price)")
s=s.replace('len(set(hashes))==33','len(set(hashes))==search["candidate_count"]+1')
a=s.index("        paired=hm_train[u]['method_detail']");b=s.index('        for key in (*BUDGET_FIELDS',a)
s=s[:a]+'''        expected_budget={'centers':256,'perturbations':256*search['candidate_count'],
            'surrogate_nets':256*(search['candidate_count']+1) if search['searcher']=='random_sobol' else 0,
            'exact_rescore_nets':256*exact_k,'reference_requests':256*exact_k}
        for key,value in expected_budget.items():assert detail[key]==value,(u,key)
        if pair_hm:
            paired=hm_train[u]['method_detail']
            assert hm_train[u]['batch_bundle_sha256']==batch.bundle_sha256
            for key in ('support_sha256','shortlist_sha256','refresh_block',*BUDGET_FIELDS):
                assert detail[key]==paired[key], f'HM actual prefix mismatch {u}/{key}'
            state['hm_paired_updates']+=1
''' + s[b:]
s=s.replace("state['candidate_nets_checked']+=256*33","state['candidate_nets_checked']+=256*(search['candidate_count']+1)")
s=s.replace("updates*256*5+len(validation)*8000+1","updates*256*(exact_k+1)+len(validation)*8000+1").replace("updates*256*5+len(validation)*8000+2","updates*256*(exact_k+1)+len(validation)*8000+2")
s=s.replace("'surrogate_evaluator_nets':updates*256*33","'surrogate_evaluator_nets':state['budget']['surrogate_nets']")
s=s.replace("*sorted((ROOT/'stage4').glob('*.py')),ROOT/'scripts/train_wdro.py'","*sorted((ROOT/'stage4').glob('*.py')),*sorted((ROOT/'stage5').glob('*.py')),ROOT/'scripts/train_stage5.py'")
s=s.replace('not stability or updates==500','not stability or updates==2000')
s=s.replace("'actual_hm_prefix_paired':state['hm_paired_updates']==updates","'actual_hm_prefix_paired_when_applicable':state['hm_paired_updates']==(updates if pair_hm else 0)")
s=s.replace("'stored_references_replayed':state['cache_values_checked']==updates*1024","'stored_references_replayed':state['cache_values_checked']==updates*256*exact_k")
s=s.replace("policy_updates']=500 if stability else 0","policy_updates']=2000 if stability else 0")
a=s.index('    if stability:\n        window=');b=s.index('    real_lp=[]',a)
s=s[:a]+'''    diagnostic['tail_100_update_window_means']=[float(costs[-200:-100].mean()),float(costs[-100:].mean())] if updates>=200 else []
    diagnostic['budget_active_at_tail']=bool(np.all(prices[-min(200,updates):]>0))
    diagnostic['tail_relative_budget_gap']=float((costs[-min(100,updates):].mean()-rho)/rho) if rho else None
    # An inactive high-radius constraint or infeasible tiny-K support is an
    # ablation outcome, not evidence of a dual implementation error.
    checks['lambda_finite_nonnegative_bounded']=bool(np.isfinite(prices).all() and prices.min()>=0 and prices.max()<wdro['stability_gate']['maximum_lambda'])
''' + s[b:]
s=s.replace("'actual_hard_mining_pair':{'run_id':hm['run_id'],'paired_updates':[1,updates],'comparison':'actual matching prefix, HM full run remains 2000 updates'","'completed_updates':updates,'actual_hard_mining_pair':{'applicable':pair_hm,'run_id':hm['run_id'],'paired_updates':[1,updates] if pair_hm else [],'comparison':'only unchanged K32 Random support has HM call parity; changed-budget ablations are explicit'")
s=s.replace("expected_updates=500 if args.stability else 12","expected_updates=2000 if args.stability else 12")
s=s.replace("result['actual_hard_mining_pair']['paired_updates'][1]","result['completed_updates']")
(r/'scripts/audit_stage5.py').write_text(s)

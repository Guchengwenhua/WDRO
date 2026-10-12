"""Read-only raw-prediction statistical reacceptance, 2026-09-19.

Only this review/statistics directory receives outputs. Training, checkpoints,
frozen source/protocol and the original report are never modified.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,pickle,sys,time
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
spec=importlib.util.spec_from_file_location('frozen_paper_report',ROOT/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def load(path):
    return json.loads(path.read_text())


def atomic(name,obj):
    report.atomic_json(OUT/name,obj)


def csv_compare(name):
    old,new=ROOT/'artifacts/report'/name,OUT/name
    with old.open(newline='') as f:a=list(csv.DictReader(f))
    with new.open(newline='') as f:b=list(csv.DictReader(f))
    old_fields=list(a[0]) if a else [];new_fields=list(b[0]) if b else []
    differences=[]
    for i,(left,right) in enumerate(zip(a,b),1):
        for key in sorted(set(left)|set(right)):
            if left.get(key)!=right.get(key):
                entry={'row':i,'field':key,'original':left.get(key),'recomputed':right.get(key)}
                try:entry['numeric_abs_difference']=abs(float(left[key])-float(right[key]))
                except (ValueError,TypeError,KeyError):pass
                differences.append(entry)
    return {'original_sha256':report.sha(old),'recomputed_sha256':report.sha(new),'byte_identical':old.read_bytes()==new.read_bytes(),
            'same_header':old_fields==new_fields,'original_rows':len(a),'recomputed_rows':len(b),
            'same_row_count':len(a)==len(b),'field_differences_count':len(differences),'field_differences':differences}


def extract():
    p=load(ROOT/'protocol.json');frozen=load(ROOT/'artifacts/frozen.json');start=time.time()
    required={str((ROOT/name).resolve()):report.sha(ROOT/name) for name in ['report.py','protocol.json']}
    resolved={str(Path(k).resolve()):v for k,v in frozen['files'].items()}
    for path,digest in required.items():assert resolved[path]==digest,(path,'source changed')
    results={};metric_rows=[];input_hashes={};run_notes=[];lp_rows=[];checks=[];training_descriptive=[]
    for family in p['families']:
        for method in p['methods']:
            for seed in p['seeds']:
                runid=f'{family}_{method}_s{seed}';run=ROOT/'artifacts/runs'/runid
                complete=load(run/'complete.json');stored=load(run/'metrics.json')
                assert complete['status']=='complete' and complete['scope']=='degree10_pilot'
                assert stored['identity']==complete['identity'] and complete['completed_updates']==p['updates']
                values={};rows=[]
                for stage in ['test_best','test_initial']:
                    path=run/(stage+'.npz');input_hashes[str(path)]=report.sha(path)
                    assert input_hashes[str(path)]==complete['artifacts'][path.name]
                    arrays=report.npz(path);values[stage]=arrays
                    for key,v in arrays.items():
                        dist,metric=key.split('__',1)
                        assert dist in p['distributions'] and v.shape==(p['test_count'],) and np.isfinite(v).all()
                        v=v.astype(np.float64);mean=float(np.mean(v));p95=float(np.quantile(v,.95))
                        report.near(stored[stage][dist][metric]['mean'],mean,runid+'/'+stage+'/'+key+' mean mismatch',atol=1e-30)
                        report.near(stored[stage][dist][metric]['p95'],p95,runid+'/'+stage+'/'+key+' P95 mismatch',atol=1e-30)
                        rows.append(dict(family=family,method=method,seed=seed,checkpoint=stage,distribution=dist,metric=metric,n=len(v),mean=mean,p95=p95))
                assert set(values['test_best'])==set(values['test_initial'])
                attack_path=run/'attack.npz';input_hashes[str(attack_path)]=report.sha(attack_path)
                assert input_hashes[str(attack_path)]==complete['artifacts']['attack.npz']
                attack=report.npz(attack_path);risks={}
                for dist in p['distributions']:
                    loss,cost,mass=(attack[dist+'__'+s] for s in ['loss','cost','mass'])
                    assert loss.shape==cost.shape==mass.shape==(p['attack_count'],p['search']['candidate_count']+1)
                    frozen_attack=report.npz(ROOT/f'artifacts/data/attack_{dist}.npz')
                    assert np.array_equal(cost,frozen_attack['cost_unrooted' if family=='rest' else 'cost_rooted'])
                    risks[dist]=report.independent_lp(loss,cost,stored['identity']['rho'],mass)
                    for key in ['risk','nominal','unconstrained_max','average_cost','row_error']:
                        report.near(stored['attack'][dist][key],risks[dist][key],runid+'/'+dist+'/'+key+' LP mismatch')
                    rows.append(dict(family=family,method=method,seed=seed,checkpoint='best_attack',distribution=dist,metric='finite_support_risk',n=loss.shape[0],mean=risks[dist]['risk'],p95=''))
                    lp_rows.append({'run_id':runid,'distribution':dist,'rho':stored['identity']['rho'],**risks[dist]})
                note={'run_id':runid,'family':family,'method':method,'seed':seed,'selected_update':stored['best']['update'],
                      'selected_initialization':stored['best']['update']==0,
                      'best_initial_cost_identical':{d:np.array_equal(values['test_best'][d+'__cost'],values['test_initial'][d+'__cost']) for d in p['distributions']},
                      'seconds':stored['seconds'],'effective_calls':stored['calls_effective_checkpoint_trajectory']}
                evidence={'identity':stored['identity'],'selected_update':stored['best']['update']}
                training={'family':family,'method':method,'seed':seed,'selected_update':stored['best']['update']}
                for dist in p['distributions']:
                    before=values['test_initial'][dist+'__cost'].astype(np.float64)
                    after=values['test_best'][dist+'__cost'].astype(np.float64)
                    training[dist]={'mean_initial':float(before.mean()),'mean_best':float(after.mean()),
                                    'delta':float((after-before).mean()),'changed_nets':int(np.count_nonzero(after!=before)),
                                    'LP':risks[dist]['risk']}
                    if family=='hub':
                        for stage,label in [('test_initial','initial'),('test_best','best')]:
                            hubs=values[stage][dist+'__hub_count']
                            training[dist]['hub_'+label]=float(hubs.mean())
                            training[dist]['nonzero_hub_nets_'+label]=int(np.count_nonzero(hubs))
                training_descriptive.append(training)
                results[(family,method,seed)]=(evidence,values,risks,rows)
                metric_rows.extend(rows);run_notes.append(note)
                print(json.dumps({'extracted':runid,'LP_checked':len(lp_rows)}),flush=True)
    for family in p['families']:
        for seed in p['seeds']:
            a,b=(results[(family,m,seed)][1]['test_initial'] for m in ['erm','wdro'])
            for key in a:assert np.array_equal(a[key],b[key]),(family,seed,key,'initial paired predictions differ')
            checks.append({'family':family,'seed':seed,'all_initial_metrics_identical':True})
    data={'results':results,'metric_rows':metric_rows,'run_notes':run_notes,'lp_rows':lp_rows,'input_hashes':input_hashes,'pair_checks':checks}
    with (OUT/'raw_recomputed.pkl').open('wb') as f:pickle.dump(data,f,protocol=5)
    atomic('raw_recompute_receipt.json',{'status':'PASS','scope':'raw NPZ means/P95 and fresh finite LP; bootstrap pending',
           'seconds':time.time()-start,'runs':len(results),'LP_count':len(lp_rows),'NPZ_artifacts':input_hashes,
           'source_hashes':required,'paired_initial_predictions':checks,'raw_recomputed_sha256':report.sha(OUT/'raw_recomputed.pkl')})
    report.write_csv(OUT/'lp_recomputed.csv',lp_rows)
    atomic('selected_checkpoints.json',run_notes)
    atomic('training_descriptive.json',training_descriptive)


def finish(audited_path):
    start=time.time();p=load(ROOT/'protocol.json')
    with (OUT/'raw_recomputed.pkl').open('rb') as f:data=pickle.load(f)
    with audited_path.open('rb') as f:audited=pickle.load(f)
    results=data['results'];assert set(audited)==set(results)
    for key,record in results.items():
        independently_audited=audited[key]
        record[0]['initial_actor_digest']=independently_audited[0]['initial_actor_digest']
        for stage in ['test_best','test_initial']:
            for name,arr in record[1][stage].items():assert np.array_equal(arr,independently_audited[1][stage][name])
        for d,risks in record[2].items():
            for name,val in risks.items():report.near(val,independently_audited[2][d][name],'independent LP rechecks disagree')
    metric_rows=report.aggregate_metrics(data['metric_rows'],len(p['seeds']))
    report.write_csv(OUT/'metrics.csv',metric_rows)
    print('Recomputing paired bootstrap: 2000 repetitions for every comparison',flush=True)
    comparison_rows=report.comparisons(p,results)
    report.write_csv(OUT/'comparisons.csv',comparison_rows)
    original_complete=load(ROOT/'artifacts/report/complete.json')
    receipt={name:csv_compare(name) for name in ['metrics.csv','comparisons.csv']}
    for name,value in receipt.items():
        assert value['original_sha256']==original_complete['artifacts'][name],name+' old report binding changed'
    assert all(x['byte_identical'] and x['same_header'] and x['same_row_count'] and x['field_differences_count']==0 for x in receipt.values()),'CSV reproduction discrepancy'
    aggregate=[x for x in comparison_rows if x['seed']=='aggregate3' and x['metric'] in ['cost','finite_support_risk']]
    report.write_csv(OUT/'aggregate_cost_effects.csv',aggregate)
    interpretation_summary=compact_summary(p,comparison_rows,metric_rows)
    atomic('compact_summary.json',interpretation_summary)
    explanation=interpret(p,comparison_rows,data['run_notes'],metric_rows)
    (OUT/'STATISTICAL_REVIEW.md').write_text(explanation)
    final={'status':'PASS','scope':'independent raw-data statistical reacceptance; no training/GPU inference',
           'runs':len(results),'LP_reoptimizations':len(data['lp_rows']),'bootstrap_repetitions':p['statistics']['bootstrap_replicates'],
           'metric_rows':len(metric_rows),'comparison_rows':len(comparison_rows),'seconds':time.time()-start,
           'audited_structure_sha256':report.sha(audited_path),'frozen_report_py_sha256':report.sha(ROOT/'report.py'),
           'protocol_sha256':report.sha(ROOT/'protocol.json'),'csv_comparison':receipt,
           'selected_initializations':[r['run_id'] for r in data['run_notes'] if r['selected_initialization']],
           'original_files_preserved':True,'output_directory':str(OUT),
           'limits':['bootstrap conditional on three trained seeds; no multiplicity correction','P95 aggregate is mean of per-seed P95s',
                     'LP is finite support, not a continuous certificate','LP bootstrap is seed-only, not a net reoptimized-LP interval',
                     'prediction replay was not run; analysis reads frozen NPZ']}
    atomic('comparison_receipt.json',final)
    files=['metrics.csv','comparisons.csv','aggregate_cost_effects.csv','lp_recomputed.csv','selected_checkpoints.json','training_descriptive.json','compact_summary.json','STATISTICAL_REVIEW.md','raw_recompute_receipt.json','comparison_receipt.json','recompute.py']
    atomic('complete.json',{'status':'complete','scope':'statistics_review_20260919','artifacts':{name:report.sha(OUT/name) for name in files}})
    print(json.dumps({'status':'PASS','metric_rows':len(metric_rows),'comparison_rows':len(comparison_rows),'byte_identical':True,'seconds':time.time()-start}),flush=True)


def compact_summary(p,comparisons,metrics):
    cost=[r for r in comparisons if r['comparison']=='wdro_minus_erm' and r['seed']=='aggregate3' and r['metric']=='cost']
    lp=[r for r in comparisons if r['comparison']=='wdro_minus_erm' and r['seed']=='aggregate3' and r['metric']=='finite_support_risk']
    excluded=lambda r,k,u:r[k+'_'+u+'_ci_low']>0 or r[k+'_'+u+'_ci_high']<0
    families=[]
    for family in p['families']:
        effects=[]
        for d in p['distributions']:
            a=next(r for r in metrics if r['family']==family and r['method']=='erm' and r['seed']=='aggregate3' and r['checkpoint']=='test_best' and r['distribution']==d and r['metric']=='cost')
            b=next(r for r in metrics if r['family']==family and r['method']=='wdro' and r['seed']=='aggregate3' and r['checkpoint']=='test_best' and r['distribution']==d and r['metric']=='cost')
            effects.append({'distribution':d,'erm_mean':a['mean'],'wdro_mean':b['mean'],'relative_percent':100*(b['mean']-a['mean'])/a['mean']})
        cc=[r for r in cost if r['family']==family]
        families.append({'family':family,'distributions':effects,'relative_percent_min':min(r['relative_percent'] for r in effects),
                         'relative_percent_max':max(r['relative_percent'] for r in effects),
                         'cost_mean_net_CI_excludes_zero':sum(excluded(r,'mean','net') for r in cc),
                         'cost_mean_seed_CI_excludes_zero':sum(excluded(r,'mean','seed') for r in cc),
                         'cost_P95_net_CI_excludes_zero':sum(excluded(r,'p95','net') for r in cc),
                         'cost_P95_seed_CI_excludes_zero':sum(excluded(r,'p95','seed') for r in cc)})
    nruns=len(p['families'])*len(p['methods'])*len(p['seeds']);ndist=len(p['distributions'])
    return {'status':'descriptive_reexpression_of_frozen_primary_cost_not_a_new_primary_metric','families':families,
            'aggregate_cells':len(cost),'cost_mean_net_CI_excludes_zero':sum(excluded(r,'mean','net') for r in cost),
            'cost_mean_seed_CI_excludes_zero':sum(excluded(r,'mean','seed') for r in cost),
            'cost_P95_net_CI_excludes_zero':sum(excluded(r,'p95','net') for r in cost),
            'cost_P95_seed_CI_excludes_zero':sum(excluded(r,'p95','seed') for r in cost),
            'finite_LP_seed_CI_excludes_zero':sum(excluded(r,'mean','seed') for r in lp),
            'test_best_cost_predictions':nruns*ndist*p['test_count'],'test_initial_cost_predictions':nruns*ndist*p['test_count'],
            'attack_cost_predictions':nruns*ndist*p['attack_count']*(p['search']['candidate_count']+1),
            'LP_reoptimizations':nruns*ndist,'test_unique_geometries':ndist*p['test_count'],'attack_unique_centers':ndist*p['attack_count'],
            'metric_rows':len(metrics),'comparison_rows':len(comparisons),
            'net_bootstrap_comparison_rows':sum('bootstrap_repetitions' in r for r in comparisons),
            'net_bootstrap_resampling_iterations':sum(r.get('bootstrap_repetitions',0) for r in comparisons),
            'cost_primary_comparison_rows':sum(r['metric']=='cost' for r in comparisons),
            'LP_comparison_rows':sum(r['metric']=='finite_support_risk' for r in comparisons),
            'CI_rule':'strictly excludes zero; touching zero does not count; no multiplicity adjustment; three training seeds'}


def interpret(p,rows,notes,metric_rows):
    def c(f,d,comparison='wdro_minus_erm',metric='cost'):
        return next(r for r in rows if r['family']==f and r['distribution']==d and r['comparison']==comparison and r['metric']==metric and r['seed']=='aggregate3')
    def ci(row,kind,unit):return f"[{row[kind+'_'+unit+'_ci_low']:.8g}, {row[kind+'_'+unit+'_ci_high']:.8g}]"
    lines=['# 2026-09-19 独立统计复核','',
           '**统计复现通过**：直接读取24个正式运行原始NPZ，重算描述统计、2000次paired bootstrap和96个有限候选LP。重生成的metrics.csv与comparisons.csv和原表逐字段及逐字节一致。','',
           '下文差值为WDRO−ERM，负数表示相应成本下降；不跨家族比较成本。净样本区间只反映固定这些模型后的测试net不确定性，seed区间只使用3次训练，不能把3072条预测当成3072次独立训练。P95汇总是每seed P95的均值。','',
           '## 对实验效益的结论','',
           '四个家族均未在本轮建立跨3个训练seed稳定的WDRO收益。Timing/Buffer存在某些分布的平均改善信号，但训练seed区间跨0或包含0，且P95/有限候选风险并不在所有分布同向。','']
    for family in p['families']:
        lines += [f'## {family}','',
            '| 分布 | 均值差 | net 95% CI | seed 95% CI | 改善seed | P95差 | P95 net CI | P95 seed CI | LP差 | LP seed CI |','|---|---:|---|---|---:|---:|---|---|---:|---|']
        for d in p['distributions']:
            r=c(family,d);lp=c(family,d,metric='finite_support_risk')
            lines.append(f"| {d} | {r['mean_delta']:.8g} | {ci(r,'mean','net')} | {ci(r,'mean','seed')} | {r['mean_negative_seeds']}/3 | {r['p95_delta']:.8g} | {ci(r,'p95','net')} | {ci(r,'p95','seed')} | {lp['mean_delta']:.8g} | {ci(lp,'mean','seed')} |")
        n=[r for r in notes if r['family']==family]
        lines += ['', '最佳checkpoint更新数（seed101/102/103）：'+ '；'.join(m+'='+','.join(str(r['selected_update']) for r in n if r['method']==m) for m in p['methods'])+'。','',
                  '| 方法 | 分布 | best−initial 均值差 | net 95% CI | seed 95% CI | 改善seed | best−initial P95差 |','|---|---|---:|---|---|---:|---:|']
        for method in p['methods']:
            for d in p['distributions']:
                r=c(family,d,method+'_best_minus_initial')
                lines.append(f"| {method} | {d} | {r['mean_delta']:.8g} | {ci(r,'mean','net')} | {ci(r,'mean','seed')} | {r['mean_negative_seeds']}/3 | {r['p95_delta']:.8g} |")
        lines.append('')
    lines += ['## 关键解释','',
              '- REST：WDRO的3个seed均选择初始化（update0），其best−initial各分布均为0。因此相对ERM的负差不能解释为WDRO训练使Actor改善；其来源是其中一个ERM选中的checkpoint在测试集上变差。两个方法共有的DAC21初始化本已较强。',
              '- Timing：四分布均值差均为负且net条件区间排除0，但每个分布仅1/3训练seed下降，训练seed区间跨0；mix_gauss的P95上升，uniform/grid_cluster的LP风险上升。说明平均收益由一个训练重复带动，不足以支持稳定改善。',
              '- Hub：uniform/grid_cluster方向略好、mix_gauss/edge_bias方向略差。全部3个ERM最佳模型、WDRO的seed102/103在四分布测试集上hub_count均为0；只有WDRO seed101少量选hub（各分布平均0.049–0.110）。大幅best−initial下降主要对应从多hub策略转为立即停止并调用无hub的MSPD；不能解释为学到了稳定优质的非平凡hub集合。两方法之间的微小差异主要来自单个seed，训练seed区间包含0。',
              '- Buffer：mix_gauss/grid_cluster的均值net条件区间排除0，但训练seed区间跨0；uniform/grid_cluster的P95上升、edge_bias的LP风险上升。可以报告局部改善信号，不能报告全面优势。','',
              '## 成本与边界','',
              '每个ERM运行有效真实评分173568，每个WDRO运行697856，约4.02倍；其中训练内层额外524288次真实候选评分。等2048次更新并非等计算量。Hub每真实评分还含11次PD参数构造。','',
              '此次degree10、每次2048updates/batch64、三个训练seed、小规模固定合成分布验证了组合可运行及其有限预算表现。不能推出论文完整训练效果、其他degree/真实芯片收益、Pareto前沿优势或连续Wasserstein鲁棒性证书；多重比较未校正。LP只对每中心33个固定候选及平均rho预算成立，且没有初始化LP产物，因此未声称LP的训练前后改善。','']
    return '\n'.join(lines)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--extract',action='store_true');ap.add_argument('--integrity-results',type=Path)
    args=ap.parse_args()
    if args.extract:extract()
    if args.integrity_results:finish(args.integrity_results)

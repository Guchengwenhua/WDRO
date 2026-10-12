#!/usr/bin/env python3
"""Independent final point-estimate, receipt and execution-time reconciliation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage8.common import *
from stage8.delivery import read_csv,write_csv
from stage8.attack import task_id


def main():
    plan=gate()
    completion=ROOT/'manifests/stage8_completion_v1.json'
    delivery=ROOT/'manifests/stage8_delivery_verification_v1.json'
    done=read(completion);proof=read(delivery)
    assert done['status']==proof['status']=='pass' and not done['stage9_started']
    checked(proof['completion']);checked(proof['report'])
    final=read(checked(read(LOG/'statistics/result.json')['raw_json']))
    published={(r['method'],r['degree'],r['seed'],r['distribution']):r for r in final['attack_per_seed']}
    points={};receipts=[];totals={key:dict(actor=0,rest=0,reference=0,verification=0) for key in (
        'committed','recorded_completed_attempts','completed_replay_extra','unfinished_possible_extra_upper_bound',
        'actual_calls_lower_bound','actual_calls_upper_bound')}
    for model in plan['models']:
        for group in ('uniform','mix_gauss','edge_bias','grid_cluster'):
            directory=LOG/'attacks'/task_id(model,group)
            audit=read(directory/'audit.json');assert audit['status']=='pass'
            r=read(checked(audit['binding']['result']));a=load_npz(checked(r['predictions']))
            assert len(a['net_ids'])==2000 and np.all(a['regret']>=a['original_regret'])
            key=(model['method'],model['degree'],model['seed'],group)
            p=dict(mean=float(a['regret'].mean()),p95=float(np.quantile(a['regret'],.95)),
                original_mean=float(a['original_regret'].mean()),original_p95=float(np.quantile(a['original_regret'],.95)),
                mean_cost=float(a['cost'].mean()),p95_cost=float(np.quantile(a['cost'],.95)),max_cost=float(a['cost'].max()),
                changed_fraction=float(a['changed'].mean()))
            for k,v in p.items():np.testing.assert_allclose(v,published[key][k],atol=2e-15,rtol=0)
            points[key]=p
            for bucket in totals:
                for kind in totals[bucket]:totals[bucket][kind]+=audit['accounting'][bucket][kind]
            receipts.append(artifact(directory/'audit.json'))
    for row in final['attack_main']:
        if row['distribution']=='ood_worst_group':
            expected=max(np.mean([p['mean'] for key,p in points.items() if key[0]==row['method'] and key[3]==g])
                for g in ('mix_gauss','edge_bias','grid_cluster'))
        else:
            expected=np.mean([p[row['metric']] for key,p in points.items() if key[0]==row['method'] and key[3]==row['distribution']])
        np.testing.assert_allclose(row['estimate'],expected,atol=2e-15,rtol=0)
    assert totals['committed']==dict(actor=153600000,rest=153600000,reference=153600000,verification=600000)
    claims=final['claim_support']
    for row in claims:
        assert row['supported_in_this_test']==bool(row['ci_high']<0 and row['all_three_seed_improve'] and row['nominal_guard_pass'])
        if row['supported_in_this_test']:assert row['difference']<0
    nominal_path=EXP/'tables/stage6_per_seed.csv'
    nominal={(r['method'],r['degree'],r['seed']):float(r['mean']) for r in read_csv(nominal_path) if r['distribution']=='uniform'}
    nominal_deltas=[nominal[('wdro_rest',d,s)]-nominal[('erm',d,s)] for m,d,s in nominal if m=='wdro_rest']
    assert len(nominal_deltas)==15 and max(nominal_deltas)<=.002
    execution=read(LOG/'recovery_v1/attack_execution_summary.json')
    checked(execution['table'])
    for name,kind in [('actor_calls','actor'),('rest_calls','rest'),('search_reference_calls','reference'),('selected_verification_calls','verification')]:
        assert execution['sums'][name]==totals['committed'][kind]
    ledger_rows=[dict(bucket=bucket,**values) for bucket,values in totals.items()]
    ledger_csv=write_csv(EXP/'tables/stage8/actual_call_reconciliation.csv',ledger_rows)
    rows=[]
    for method in ('erm','mixed_erm','hard_mining','group_dro','wdro_rest'):
        means={g:next(r for r in final['attack_main'] if r['method']==method and r['distribution']==g and r['metric']=='mean')['estimate']
               for g in ('uniform','mix_gauss','edge_bias','grid_cluster','ood_worst_group')}
        rows.append(dict(method=method,**means))
    summary=write_csv(EXP/'tables/stage8/attack_mean_summary.csv',rows)
    report=dict(status='pass',stage=8,at_utc=utc_now(),code=artifact(Path(__file__)),
        completion=artifact(completion),delivery=artifact(delivery),
        all_300_per_seed_metrics_recomputed=True,all_45_attack_macro_points_recomputed=True,
        receipts=totals,receipt_sources=receipts,actual_calls_table=ledger_csv,
        execution=artifact(LOG/'recovery_v1/attack_execution_summary.json'),execution_table=execution['table'],
        attack_mean_summary=summary,metric_specific_claim_conditions_exact=True,
        nominal_check=dict(source=artifact(nominal_path),all_15_degree_seed_deltas_within_guard=True,
            maximum_delta=max(nominal_deltas),macro_delta=float(np.mean(nominal_deltas))),stage9_started=False)
    atomic_write_json(ROOT/'manifests/stage8_delivery_review_v1.json',report,mode=0o444)
    print({k:v for k,v in report.items() if k not in ('receipt_sources','code','completion','delivery','execution','execution_table','actual_calls_table','attack_mean_summary')},flush=True)


if __name__=='__main__':main()

"""Explicit fixture test of the complete 300-cell aggregation and CSV/figure schema.

These repeated smoke vectors and constant method offsets are test fixtures only.
They are never candidates for formal Stage 8 statistics or scientific claims.
"""
import copy
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage8.common import *
from stage8 import statistics as stats, delivery
from stage8.attack import task_id
from src.metrics import summarize_regret


def main():
    plan=gate()
    directory=LOG/'smoke/final_aggregation_fixture'
    assert not (LOG/'smoke/final_aggregation.json').exists()
    lease=type('FixtureLease',(),{'update':lambda *args,**kwargs:None})()
    for model in plan['models']:
        source=LOG/'smoke'/('resumed' if model['degree']==50 else f"peak_d{model['degree']}")/'result.json'
        original=read(source);values=load_npz(checked(original['predictions']))
        for group in stats.GROUPS:
            out=directory/'logs/attacks'/task_id(model,group)
            vector=np.resize(values['regret'],2000)+.0001*(4-stats.METHODS.index(model['method']))
            arrays=dict(regret=vector,net_ids=np.array([f'FIXTURE-d{model["degree"]}-{group}-{i}' for i in range(2000)],dtype='S64'))
            atomic_npz(out/'predictions.npz',arrays)
            result=copy.deepcopy(original)
            result.update(synthetic_test_fixture=True,completed_nets=2000,predictions=artifact(out/'predictions.npz'),
                          summary=summarize_regret(vector))
            atomic_write_json(out/'result.json',result)
    actual_bootstrap=stats.bootstrap_cell
    with patch.object(stats,'LOG',directory/'logs'),patch.object(stats,'EXP',directory),patch.object(stats,'CACHE',directory/'cache'),\
         patch.object(stats,'bootstrap_cell',lambda values,**kwargs:actual_bootstrap(values,**{**kwargs,'samples':64})):
        result=stats.attack_statistics(lease)
    assert {k:len(result[k]) for k in result}==dict(per_seed=300,per_degree=225,main=45,paired=864,bootstrap=100,paired_nets=80)
    for row in result['paired']:
        expected=-.0001*(4-stats.METHODS.index(row['baseline']))
        for field in ('difference','ci_low','ci_high'):
            np.testing.assert_allclose(row[field],expected,atol=2e-14,rtol=0)
        assert row['all_three_seed_improve']
    support=read(LOG/'supporting_statistics/result.json')
    rows=dict(attack_per_seed=result['per_seed'],attack_per_degree=result['per_degree'],attack_main=result['main'],
        attack_paired_ci=result['paired'],final_main_long=support['synthetic_main']+support['ispd']['main']+result['main'])
    assert len(rows['final_main_long'])==110
    rows['claim_support']=delivery.claim_rows(rows['final_main_long'],result['paired'],support['ispd']['paired'])
    tables=[]
    for name,data in rows.items():
        record=delivery.write_csv(directory/'tables'/(name+'.csv'),data);tables.append(record)
        actual=delivery.read_csv(checked(record))
        assert len(actual)==len(data)
        for a,b in zip(actual,data,strict=True):
            assert a=={k:('' if v is None else str(v)) for k,v in b.items()}
    (directory/'figures').mkdir(parents=True,exist_ok=True)
    with patch.object(delivery,'EXP',directory),patch.object(delivery,'budget_figures',lambda:[]):
        figs=delivery.figures(rows['final_main_long'])
    atomic_write_json(directory/'typed_tables_fixture.json',rows)
    report=dict(status='pass',synthetic_test_fixture=True,never_scientific_results=True,at_utc=utc_now(),
        code=artifact(Path(__file__)),bootstrap_replicates_for_fixture=64,
        row_counts={k:len(v) for k,v in rows.items()},constant_paired_effects_exact=True,
        all_csv_json_values_exact=True,typed_fixture=artifact(directory/'typed_tables_fixture.json'),tables=tables,figures=figs)
    atomic_write_json(LOG/'smoke/final_aggregation.json',report,mode=0o444)
    print({k:v for k,v in report.items() if k not in ('tables','figures')},flush=True)


if __name__=='__main__':main()

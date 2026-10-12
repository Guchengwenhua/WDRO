"""Independently audit every prediction, model identity, table and paired CI."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CODEX=ROOT.parent;EXP=CODEX/'experiments';sys.path.insert(0,str(ROOT))
from src.io_utils import atomic_write_json
from src.job_control import JobLease,utc_now
from src.rest import file_sha256,RestTreeEvaluator
from src.metrics import relative_regret,summarize_regret
from stage6.evaluation import artifact,load_group
from stage6.statistics import ci
from stage6_pipeline import gate

def main():
    gate();index=ROOT/'manifests/stage6_models_frozen_v1.json';frozen=json.loads(index.read_text());assert frozen['status']=='frozen'
    stats_path=EXP/'logs/stage6_statistics/result.json';stats=json.loads(stats_path.read_text());assert stats['status']=='pass'
    job=EXP/'logs/stage6_delivery';command=f'bash {ROOT}/scripts/stage3_env.sh {sys.executable} {Path(__file__)}'
    with JobLease(job,run_id='stage6_delivery',exact_command=command,resume_command=command,monitor_command=f'cat {job}/heartbeat.json',expected_updates=300,physical_gpu_index=None,host_runtime_manifest=None) as lease:
        checks={};records=[];values={};boot={};evaluated=0;hardware=set();rest=RestTreeEvaluator()
        for degree in [10,20,30,40,50]:
            models=[m for m in frozen['models'] if m['degree']==degree];assert len(models)==15
            for dist in ['uniform','mix_gauss','edge_bias','grid_cluster']:
                data,refs,sources=load_group(degree,dist,'test')
                for m in models:
                    method=m['method'];seed=m['seed'];key=(method,degree,seed,dist)
                    path=EXP/'logs/stage6_test'/f'{method}_d{degree}_s{seed}'/dist/'result.json';r=json.loads(path.read_text())
                    assert r['status']=='complete' and r['binding']['model']==m and r['binding']['sources']==sources
                    assert r['binding']['split']=='test' and r['binding']['batch_size']==256 and r['binding']['T']==1 and r['binding']['mode']=='deterministic_greedy'
                    assert r['binding']['evaluator_sha256']==file_sha256(ROOT/'stage6/evaluation.py') and not r['test_used_for_selection']
                    pred=Path(r['predictions']['logical_path']);assert file_sha256(pred)==r['predictions']['sha256']
                    receipt=job/'cells'/f'{method}_d{degree}_s{seed}_{dist}.json'
                    binding=dict(result_sha256=file_sha256(path),prediction_sha256=file_sha256(pred),sources=sources,auditor_sha256=file_sha256(Path(__file__)))
                    already=receipt.exists() and json.loads(receipt.read_text()).get('binding')==binding
                    with np.load(pred,allow_pickle=False) as arrays:
                        np.testing.assert_array_equal(arrays['net_ids'],data['net_ids']);np.testing.assert_array_equal(arrays['lstar'],refs['lstar'])
                        lengths=arrays['rest_lengths'] if already else rest.lengths(data['pins'],arrays['topology']);np.testing.assert_array_equal(lengths,arrays['rest_lengths'])
                        regret=relative_regret(lengths,refs['lstar']);np.testing.assert_array_equal(regret,arrays['regret'])
                    assert r['summary']==summarize_regret(regret) and r['completed_nets']==10000
                    assert r['call_ledger']==dict(actor_forward_nets=10000,exact_rest_evaluator_nets=10000,geosteiner_nets=0)
                    if not already:atomic_write_json(receipt,dict(status='pass',binding=binding,rest_nets_recomputed=10000))
                    values[key]=regret;hardware.add(json.dumps(r['binding']['hardware'],sort_keys=True));records.extend([artifact(path),artifact(pred)])
                    checks['_'.join(map(str,key))]=True;evaluated+=10000
                for method in ['erm','mixed_erm','hard_mining','group_dro','wdro_rest']:
                    bp=EXP/'cache/stage6_bootstrap'/f'{method}_d{degree}_{dist}.npz';meta=json.loads(bp.with_suffix('.json').read_text())
                    assert meta['artifact']['sha256']==file_sha256(bp)
                    with np.load(bp) as f:boot[(method,degree,dist)]={k:f[k] for k in f.files}
                lease.update(update=len(checks),status='checking_all_predictions')
        assert len(checks)==300 and evaluated==3000000 and len(hardware)==1
        checks['all_300_cells_and_shared_hardware']=True
        for m in frozen['models']:
            for k in ['best','latest','result']:assert file_sha256(Path(m[k]['logical_path']))==m[k]['sha256']
        checks['75_models_unchanged_since_pretest_freeze']=True
        names=['uniform','mix_gauss','edge_bias','grid_cluster'];degrees=[10,20,30,40,50]
        def point(method,degree,seed,dist,stat):
            ds=degrees if degree=='macro5' else [int(degree)];ss=[101,102,103] if seed=='mean3' else [int(seed)]
            if dist=='ood_worst_group':return max(point(method,degree,seed,g,'mean') for g in names[1:])
            return float(np.mean([values[(method,d,s,dist)].mean() if stat=='mean' else np.quantile(values[(method,d,s,dist)],.95) for d in ds for s in ss]))
        def draws(method,degree,seed,dist,stat):
            if dist=='ood_worst_group':return np.max([draws(method,degree,seed,g,'mean') for g in names[1:]],axis=0)
            ds=degrees if degree=='macro5' else [int(degree)];ss=[0,1,2] if seed=='mean3' else [[101,102,103].index(int(seed))]
            return np.mean([boot[(method,d,dist)][stat][:,ss].mean(axis=1) for d in ds],axis=0)
        def close(x,y):assert np.isclose(float(x),float(y),rtol=1e-12,atol=1e-14),(x,y)
        expected_counts={'per_seed':300,'per_degree':100,'main':20,'paired_ci':676,'worst_group':30}
        for table in stats['tables']:
            path=Path(table['logical_path']);assert file_sha256(path)==table['sha256']
            name=path.stem.removeprefix('stage6_');rows=list(csv.DictReader(path.open()));assert len(rows)==expected_counts[name]
            keys=set()
            for row in rows:
                if name=='paired_ci':
                    degree=row['degree'];seed=row['seed'];dist=row['distribution'];stat=row['metric'];base=row['baseline']
                    key=(base,degree,seed,dist,stat);assert key not in keys;keys.add(key)
                    close(row['difference'],point('wdro_rest',degree,seed,dist,stat)-point(base,degree,seed,dist,stat))
                    expected=ci(draws('wdro_rest',degree,seed,dist,stat)-draws(base,degree,seed,dist,stat))
                    for k,v in expected.items():close(row[k],v)
                elif name=='worst_group':
                    method=row['method'];degree=row['degree'];key=(method,degree);assert key not in keys;keys.add(key)
                    close(row['value'],point(method,degree,'mean3','ood_worst_group','mean'))
                    for k,v in ci(draws(method,degree,'mean3','ood_worst_group','mean')).items():close(row[k],v)
                else:
                    method=row['method'];degree=row.get('degree','macro5');seed=row.get('seed','mean3');dist=row['distribution']
                    key=(method,degree,seed,dist);assert key not in keys;keys.add(key)
                    for stat in ['mean','p95']:
                        close(row[stat],point(method,degree,seed,dist,stat))
                        for k,v in ci(draws(method,degree,seed,dist,stat)).items():close(row[stat+'_'+k],v)
            checks['table_'+name]=True
        paired_artifacts=[]
        for base in ['erm','mixed_erm','hard_mining','group_dro']:
            for degree in degrees:
                for dist in names:
                    p=EXP/'tables/stage6_paired_nets'/f'wdro_minus_{base}_d{degree}_{dist}.npz'
                    with np.load(p) as f:
                        np.testing.assert_array_equal(f['seeds'],[101,102,103])
                        np.testing.assert_array_equal(f['relative_regret_difference'],np.stack([values[('wdro_rest',degree,s,dist)]-values[(base,degree,s,dist)] for s in [101,102,103]]))
                    paired_artifacts.append(artifact(p))
        checks['all_paired_net_differences']=True
        checks['frozen_input_gate']=True
        assert not list((EXP/'logs/stage6_test').glob('**/*.partial'))
        checks['no_partial_test_outputs']=True
        completion=dict(status='pass',stage=6,at_utc=utc_now(),model_count=75,test_cells=300,test_net_evaluations=3000000,
            model_index=artifact(index),statistics=artifact(stats_path),tables=stats['tables'],predictions=records,paired_net_differences=paired_artifacts,
            checks=checks,audit_call_ledger=dict(actor_forward_nets=0,exact_rest_evaluator_nets=3000000,geosteiner_nets=0),
            test_used_for_selection=False,stage7_started=False,definitions=stats['definitions'])
        dest=ROOT/'manifests/stage6_completion_v1.json';atomic_write_json(dest,completion,mode=0o444)
        atomic_write_json(ROOT/'manifests/stage6_delivery_verification_v1.json',dict(status='pass',at_utc=utc_now(),completion=artifact(dest),checks_passed=len(checks),
            model_count=75,test_cells=300,stage7_started=False),mode=0o444)
        lease.finish(update=300,status='complete')
if __name__=='__main__':main()

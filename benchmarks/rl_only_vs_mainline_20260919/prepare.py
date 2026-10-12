"""Freeze existing Actors and both complete degree-30 synthetic test suites."""
from pathlib import Path
import json
import time
import numpy as np
from common import ROOT, PROJECT, RL, MAIN, G, GROUPS, SEEDS, read, sha, checked, atomic, data_path

def main():
    assert not (ROOT/'manifest.json').exists(), 'already frozen'
    models = []; bindings = {}
    main_manifest = MAIN/'wdro_rest_min/manifests/stage6_models_frozen_v1.json'
    main_data = MAIN/'experiments/cache/datasets/protocol_v1/datasets_manifest.json'
    for path in (main_manifest, main_data, RL/'protocol.json', RL/'geometry.py',
                 MAIN/'wdro_rest_min/configs/protocol_v1.json', MAIN/'wdro_rest_min/configs/stage2_trainer_v1.json',
                 MAIN/'wdro_rest_min/configs/stage3_baselines_v1.json'):
        bindings[str(path)] = sha(path)
    dependencies = G.dependencies()
    old_legacy = Path('/remote-home/Guilin/DRO/Codex/shared/legacy_dro/WDRO-REST/models')
    for path, digest in dependencies.items():
        other = old_legacy/Path(path).name
        checked(other, digest)
        bindings[str(other)] = digest
    bindings.update(dependencies)
    for row in read(main_manifest)['models']:
        if row['degree'] != 30: continue
        checkpoint = checked(row['best']['logical_path'], row['best']['sha256'])
        checked(row['result']['logical_path'], row['result']['sha256'])
        bindings[str(checkpoint)] = row['best']['sha256']
        bindings[row['result']['logical_path']] = row['result']['sha256']
        models.append(dict(id=f'mainline_{row["method"]}_s{row["seed"]}', family='mainline',
                           method=row['method'], seed=row['seed'], checkpoint=str(checkpoint),
                           checkpoint_sha256=row['best']['sha256'], state_key='actor_state_dict'))
    assert len(models) == 15
    source_result = None
    for method in read(RL/'protocol.json')['methods']:
        for seed in SEEDS:
            folder = RL/'artifacts/runs'/f'{method}_s{seed}'
            c = read(folder/'complete.json')
            assert c['status'] == 'complete' and c['acceptance'] == 'accepted_gpu'
            for name in ('best.pt', 'results.npz'):
                checked(folder/name, c['artifacts'][name])
                bindings[str(folder/name)] = c['artifacts'][name]
            bindings[str(folder/'complete.json')] = sha(folder/'complete.json')
            for name, digest in c['binding']['code_sha256'].items():
                checked(name, digest); bindings[name] = digest
            for name, digest in c['binding']['dependencies'].items():
                checked(name, digest); bindings[name] = digest
            models.append(dict(id=f'rl_only_{method}_s{seed}', family='rl_only', method=method, seed=seed,
                               checkpoint=str(folder/'best.pt'), checkpoint_sha256=c['artifacts']['best.pt'],
                               state_key='actor', saved_results=str(folder/'results.npz'),
                               saved_results_sha256=c['artifacts']['results.npz'],
                               selected_update=c['best_step'], actor_sha256=c['final_actor_sha256']))
            if source_result is None: source_result = folder/'results.npz'
    assert len(models) == 33
    (ROOT/'artifacts/data').mkdir(parents=True, exist_ok=True)
    data = []; original = np.load(source_result, allow_pickle=False)
    for group in GROUPS:
        x = original[f'final__{group}__pins']
        np.testing.assert_array_equal(x, G.make_data(1024, 30, group, 'test/'+group, seed=0))
        path = data_path('rl_test', group)
        G.atomic_npz(path, pins=x)
        data.append(dict(suite='rl_test', group=group, path=str(path), sha256=sha(path), count=len(x), source=str(source_result)))
        x = original[f'attack__{group}__pins']
        assert x.shape == (128, 32, 30, 2)
        path = data_path('rl_attack32', group)
        G.atomic_npz(path, pins=x, cost=original[f'attack__{group}__cost'])
        data.append(dict(suite='rl_attack32', group=group, path=str(path), sha256=sha(path), count=128*32, source=str(source_result)))
    original.close()
    for row in read(main_data)['records']:
        if row['degree'] != 30 or row['split'] != 'test': continue
        # The inherited data manifest predates relocation; resolve within precision_v2.
        source = MAIN / row['path'].removeprefix('Codex/')
        checked(source, row['npz_sha256']); bindings[str(source)] = row['npz_sha256']
        with np.load(source, allow_pickle=False) as z:
            pins = z['pins']; ids = z['net_ids']
        assert pins.shape == (10000, 30, 2) and pins.dtype == np.float32
        path = data_path('mainline_test', row['distribution'])
        G.atomic_npz(path, pins=pins, net_ids=ids)
        data.append(dict(suite='mainline_test', group=row['distribution'], path=str(path), sha256=sha(path), count=10000, source=str(source)))
    assert len(data) == 12
    # Identical coordinates across all reused model predictions, independently bound.
    for model in models:
        if model['family'] != 'rl_only': continue
        with np.load(model['saved_results'], allow_pickle=False) as z:
            for group in GROUPS:
                for suite, prefix in [('rl_test', 'final'), ('rl_attack32', 'attack')]:
                    with np.load(data_path(suite, group), allow_pickle=False) as d:
                        np.testing.assert_array_equal(d['pins'], z[f'{prefix}__{group}__pins'])
    protocol = dict(experiment='rl_only_vs_precision_v2_frozen_actors', degree=30, training_seeds=SEEDS,
        mainline_methods=['erm','mixed_erm','hard_mining','group_dro','wdro_rest'],
        rl_only_methods=read(RL/'protocol.json')['methods'], groups=GROUPS,
        inference=dict(T=1, greedy=True, forward_batch=64, tail_padding='repeat last then discard',
                       coordinate_preprocessing='none: identical stored coordinates for every Actor within a suite'),
        suites={'rl_test':'all 4096 original fixed-window test nets',
                'mainline_test':'all 40000 original bbox-normalized degree30 mainline test nets',
                'rl_attack32':'same 512 heldout centers and 32 candidates per center; maximize own length, lambda=0'},
        output_metric='own REST span-accounted length; no optimum/reference labels',
        model_selection='retain every previously frozen validation-best Actor; no tuning or training',
        reuse='RL-only original rl_test and rl_attack32 predictions; 64-row replay for each of 8 cells verifies compatibility',
        statistics=dict(replicates=2000, seed=20260919, grouping='equal distribution and training seed weights',
                        within_family='paired seed and net resampling',
                        cross_family='independent training-seed resampling; paired net resampling',
                        baselines=['mainline:group_dro','mainline:mixed_erm','mainline:wdro_rest'],
                        multiple_testing_correction=False),
        limitations=['frozen-model comparison, not matched initialization/training budget ablation',
            'mainline has DAC21 pretraining, 2000x256 fine tuning and relative-regret reference feedback',
            'RL-only starts randomly, 2048x64 updates and own-length feedback',
            'same named distributions have different generators/normalization; report suites separately',
            'Random32 window maximum is not a rho-constrained worst-case certificate',
            'No reference solver calls or new training; old model training lineage still used reference algorithms'])
    atomic(ROOT/'protocol.json', protocol)
    bindings[str(ROOT/'protocol.json')] = sha(ROOT/'protocol.json')
    for row in data: bindings[row['path']] = row['sha256']
    manifest = dict(created_at=time.time(), protocol_sha256=sha(ROOT/'protocol.json'), models=models, data=data, bindings=bindings)
    atomic(ROOT/'manifest.json', manifest)
    tasks = [dict(id=m['id'], kind='evaluation', args=['evaluate.py','--model-id',m['id']],
                  complete=f'artifacts/runs/{m["id"]}/complete.json', depends_on=[], required_free_mib=4096,
                  cpu_threads=3, completion_checks={'status':'complete','acceptance':'accepted_gpu'},
                  binding_checks={'model_id':m['id'],'manifest_sha256':sha(ROOT/'manifest.json')}) for m in models]
    # Interleave families, methods and seeds without changing any statistical choices.
    tasks.sort(key=lambda t: (int(t['id'].rsplit('s',1)[1]), 'group_dro' not in t['id'], t['id']))
    atomic(ROOT/'queue.json', dict(tasks=tasks, source_paths=['protocol.json','manifest.json'],
         postprocess=dict(args=['report.py'], complete='artifacts/report/complete.json',
                          completion_checks={'status':'complete','acceptance':'accepted_gpu'})))
    print(json.dumps({'status':'frozen','models':len(models),'data_files':len(data),'new_inference_rows':1627200,'replay_check_rows':9216}), flush=True)

if __name__ == '__main__': main()

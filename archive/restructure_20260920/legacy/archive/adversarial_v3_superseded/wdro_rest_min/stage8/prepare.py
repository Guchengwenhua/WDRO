"""Freeze independent inputs and bind the accepted precision_v2 model set."""
import itertools

from src.datasets import build_synthetic_dataset, array_bundle_sha256, load_dataset
from src.geometry import pin_set_sha256
from src.protocol import load_protocol
from src.geosteiner import reference_identity
from stage8.common import *


def prepare():
    if PLAN.exists():
        return gate()
    host_path, host = capture('Stage8.1 independent input preparation (CPU)')
    with Lease(LOG / 'prepare', 'stage8_prepare', str(host_path)) as lease:
        protocol = load_protocol()
        accepted6 = TOP / 'wdro_rest_min/manifests/precision_v2_replay_completion.json'
        accepted7 = ROOT / 'manifests/stage7_delivery_verification_v1.json'
        assert read(accepted6)['status'] == 'pass' and read(accepted6)['models'] == 75
        assert read(accepted7)['status'] == 'pass'
        checked(read(accepted7)['completion'])
        models_path = ROOT / 'manifests/stage6_models_frozen_v1.json'
        frozen = read(models_path)
        assert frozen['status'] == 'frozen' and frozen['selection'] == 'synthetic_validation_only'
        assert not frozen['test_used_for_selection'] and not frozen['test_evaluated_at_freeze']
        models = frozen['models']
        assert {(m['method'], m['degree'], m['seed']) for m in models} == set(itertools.product(
            protocol['scope']['methods'], protocol['scope']['degrees'], protocol['data']['seeds']['training']))
        for m in models:
            for key in ('best', 'latest', 'result'):
                checked(m[key])
        for rel, digest in frozen['code_files'].items():
            assert file_sha256(CODEX / rel) == digest, rel
        selected = ROOT / 'configs/stage5_selected_wdro_v1.json'
        assert not read(selected)['test_or_ispd_used'] and read(selected)['no_post_test_retuning']
        dependencies = [artifact(p) for p in [accepted6, accepted7, models_path, selected,
            TOP / 'WDRO-REST_最简实验执行计划.md', ROOT / 'configs/stage5/cem_64.json']]
        for name in ('protocol_v1.json', 'data_contract_v1.json', 'stage2_trainer_v1.json'):
            p = ROOT / 'configs' / name
            assert file_sha256(p) == file_sha256(TOP / 'wdro_rest_min/configs' / name)
            dependencies.append(artifact(p))
        # Model-free sampling. The attack namespace is disjoint from train_u*,
        # validation and test while retaining the protocol's evaluation base seed.
        groups, checks, all_ids = [], [], set()
        for degree in protocol['scope']['degrees']:
            heldout = set()
            for split in ('validation', 'test'):
                count = protocol['data']['sizes_per_degree_distribution'][split]
                seed = protocol['data']['seeds'][split]
                for distribution in protocol['data']['distributions']:
                    path = EXP / 'cache/datasets/protocol_v1' / split / f'{distribution}_d{degree}_s{seed}_n{count}.npz'
                    data, _ = load_dataset(path)
                    heldout.update(pin_set_sha256(x) for x in data['pins'])
            attack_hashes = set()
            for distribution, parameters in protocol['data']['distributions'].items():
                kwargs = dict(protocol_id=protocol['protocol_id'], split='attack', distribution=distribution,
                    degree=degree, base_seed=303, count=2000, parameters=parameters)
                arrays, meta = build_synthetic_dataset(**kwargs)
                replay, replay_meta = build_synthetic_dataset(**kwargs)
                assert meta == replay_meta
                for k in arrays:
                    np.testing.assert_array_equal(arrays[k], replay[k])
                ids = set(arrays['net_ids'].tolist())
                hashes = {pin_set_sha256(x) for x in arrays['pins']}
                assert len(ids) == len(hashes) == 2000 and not ids & all_ids
                assert not hashes & (heldout | attack_hashes), 'attack input intersects another fixed split'
                all_ids.update(ids)
                attack_hashes.update(hashes)
                path = CACHE / 'inputs' / f'{distribution}_d{degree}.npz'
                if path.exists():
                    assert array_bundle_sha256(load_npz(path)) == meta['array_bundle_sha256']
                else:
                    atomic_npz(path, arrays)
                    path.chmod(0o444)
                group = dict(degree=degree, distribution=distribution, count=2000,
                    input=artifact(path), metadata=meta)
                groups.append(group)
                checks.append(dict(degree=degree, distribution=distribution, unique_nets=2000,
                    deterministic_rebuild=True, fixed_split_content_intersections=0,
                    training_namespace_disjoint=True))
                lease.update(len(groups), 'preparing_independent_inputs')
        # Ordinary-user access check, through logical paths only.
        writes = []
        for directory in (CODEX, EXP / 'tmp', EXP / 'cache/runtime', LOG, CACHE):
            directory.mkdir(parents=True, exist_ok=True)
            p = directory / ('stage8_writecheck_' + stamp())
            payload = os.urandom(32)
            p.write_bytes(payload)
            assert p.read_bytes() == payload
            p.unlink()
            writes.append(dict(logical_path=str(directory), resolved_path=str(directory.resolve()),
                               ordinary_user_write_read_cleanup='pass'))
        check_path = LOG / 'prepare/input_audit.json'
        atomic_write_json(check_path, dict(status='pass', at_utc=utc_now(), groups=checks,
            models=75, unique_attack_nets=40000, actor_calls=0, model_based_sampling=False,
            train_overlap_evidence='Distinct immutable RNG/ID namespaces; no full training-pool content replay claimed.',
            preflight=artifact(host_path), storage_access=writes), mode=0o444)
        dependencies.append(artifact(check_path))
        cem = dict(read(ROOT / 'configs/stage5/cem_64.json')['search_overrides']['cem'], rounds=16)
        plan = dict(schema_version=1, stage=8, status='frozen', at_utc=utc_now(), revision='reference_precision_v2',
            dependencies=dependencies, models=models, groups=groups, input_base_seed=303, input_split='attack',
            expected_cells=300, expected_unique_centers=40000, expected_model_center_attacks=600000,
            expected_search_calls=153600000, reference_identity=reference_identity(),
            attack=dict(searcher='cem_256', rounds=16, population=16, elite=4,
                calls_per_center=256, original_included_once=True, objective='maximum_true_relative_regret',
                transport_penalty=0, ground_cost='hungarian_l1_mean', cem=cem,
                refresh_updates=10, seed=303, center_batch=16, score_microbatch=256,
                tie_break='earliest candidate', source='stage5.cem_round with independent attack net IDs; 16 rounds',
                cost_interpretation='Unconstrained local search stress test; measure actual cost. Not a certified Wasserstein ball.'),
            inference=dict(T=1, mode='deterministic_greedy', hardware_class='NVIDIA RTX 6000 Ada Generation',
                batch_size=256, shared_net_order=True, same_attack_rng_for_all_models=True),
            statistics=dict(bootstrap_samples=10000, confidence_level=.95, seed=808,
                uncertainty='paired net bootstrap conditional on three fixed trained seeds; not seed-population uncertainty',
                seed_aggregation='mean of separately computed seed metrics',
                p95='linear quantile per seed; do not average per-net seed values before taking a tail',
                comparisons='WDRO minus each of four baselines; retain every degree/distribution/seed',
                claim_gate='observed direction, 95% paired CI, all three seed directions, nominal delta <= 0.002'),
            training=False, model_selection=False, stage9_started=False)
        atomic_write_json(PLAN, plan, mode=0o444)
        progress('8.1 75冻结模型及40000独立attack中心核验通过', 'DONE', str(check_path),
                 '前台验证CEM-256真实预算、恢复、几何及五degree显存')
        return plan

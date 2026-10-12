"""Bind the accepted precision_v2 models and unchanged external input files."""
import itertools

import numpy as np

from src.datasets import load_dataset, array_bundle_sha256
from src.ispd import normalize_ispd_degree_file, parse_degree_file
from src.protocol import load_protocol, protocol_sha256
from src.geosteiner import reference_identity
from stage7.common import *


def prepare():
    if PLAN.exists():
        return gate()
    preflight, _ = capture('Stage7.1 frozen-model and ISPD input preparation (CPU)')
    with Lease(LOG / 'prepare', 'stage7_prepare', str(preflight)) as lease:
        protocol = load_protocol()
        completion = TOP / 'wdro_rest_min/manifests/precision_v2_replay_completion.json'
        c = read(completion)
        assert c['status'] == 'pass' and c['stage'] == 6 and c['models'] == 75
        assert c['reference_correction'] == 'precision_v2'
        for row in c['files']:
            checked(row)
        model_path = ROOT / 'manifests/stage6_models_frozen_v1.json'
        frozen = read(model_path)
        assert frozen['status'] == 'frozen' and frozen['selection'] == 'synthetic_validation_only'
        assert not frozen['test_used_for_selection'] and not frozen['test_evaluated_at_freeze']
        selected = ROOT / 'configs/stage5_selected_wdro_v1.json'
        choice = read(selected)
        assert choice['status'] == 'frozen' and choice['test_or_ispd_used'] is False
        assert choice['no_post_test_retuning']
        models = frozen['models']
        assert {(m['method'], m['degree'], m['seed']) for m in models} == set(itertools.product(
            protocol['scope']['methods'], protocol['scope']['degrees'], protocol['data']['seeds']['training']))
        for rel, digest in frozen['code_files'].items():
            assert file_sha256(CODEX / rel) == digest, rel
        for m in models:
            for key in ('best', 'latest', 'result'):
                checked(m[key])
        dependencies = [artifact(completion), artifact(model_path), artifact(selected)]
        for name in ('protocol_v1.json', 'data_contract_v1.json', 'stage2_trainer_v1.json'):
            p = ROOT / 'configs' / name
            assert file_sha256(p) == file_sha256(TOP / 'wdro_rest_min/configs' / name)
            dependencies.append(artifact(p))
        input_path = EXP / 'cache/ispd_protocol_a/ispd_protocol_a_manifest.json'
        manifest = read(input_path)
        assert manifest['protocol'] == 'A' and manifest['model_metrics_locked']
        assert not manifest['contains_model_metrics']
        assert manifest['base_design_count'] == 20 and manifest['totals']['valid_count'] == 22498
        assert manifest['protocol_canonical_sha256'] == protocol_sha256(protocol)
        assert all(x['duplicate_design_identity_confirmed'] for x in manifest['excluded_variants'])
        dependencies.append(artifact(input_path))
        groups, all_ids, counts, source_audits = [], set(), [], []
        for degree in protocol['scope']['degrees']:
            chunks = []
            for rec in manifest['records']:
                if rec['degree'] != degree:
                    continue
                count = {k: rec[k] for k in ('design_id', 'degree', 'parsed_count', 'valid_count', 'invalid_count')}
                count['source_missing'] = 'dataset_path' not in rec
                count['invalid_reasons'] = rec.get('invalid_reasons', {})
                counts.append(count)
                if count['source_missing']:
                    source = CODEX.parent / 'DRO/benchmark' / (rec['design_id'] + '.input') / f'degree_{degree}.txt'
                    assert not source.exists(), 'frozen missing input has appeared; do not expand scope'
                    continue
                path = CODEX.parent / rec['dataset_path']
                data, meta = load_dataset(path, verify=True)
                assert meta['array_bundle_sha256'] == rec['array_bundle_sha256']
                assert meta['design_id'] == rec['design_id'] and meta['degree'] == degree
                assert len(data['pins']) == rec['valid_count']
                source = CODEX.parent / meta['source_path']
                assert file_sha256(source) == rec['source_sha256'] == meta['source_sha256']
                # Read-only source replay uses the one shared normalization implementation.
                rebuilt, replay_meta = normalize_ispd_degree_file(source, rec['design_id'], degree)
                for k in data:
                    np.testing.assert_array_equal(data[k], rebuilt[k])
                assert replay_meta['invalid_count'] == rec['invalid_count']
                assert replay_meta['invalid_reasons'] == rec['invalid_reasons']
                expected_ids = [f"ispd:{rec['design_id']}:{net.net_name}:d{degree}" for net in parse_degree_file(source)]
                assert all(len(x.encode()) <= 128 for x in expected_ids), 'truncated source ID'
                ids = set(data['net_ids'].tolist())
                assert len(ids) == len(data['pins']) and not (ids & all_ids), 'duplicate external net ID'
                assert all(x.decode().startswith(f"ispd:{rec['design_id']}:") for x in ids)
                all_ids.update(ids)
                chunk = dict(data, design_ids=np.full(len(data['pins']), rec['design_id'].encode(), dtype='S32'),
                             source_index=np.arange(len(data['pins']), dtype=np.int64))
                chunks.append(chunk)
                dependencies.extend([artifact(path), artifact(path.with_suffix('.meta.json'))])
                source_audits.append(dict(design_id=rec['design_id'], degree=degree,
                    source_logical_path=str(source), source_resolved_path=str(source.resolve()),
                    source_sha256=rec['source_sha256'], rebuilt_nets=len(data['pins'])))
                lease.update(len(source_audits), 'rebuilding_input_identity')
            packed = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
            out = CACHE / 'inputs' / f'degree_{degree}.npz'
            atomic_npz(out, packed)
            out.chmod(0o444)
            groups.append(dict(degree=degree, count=len(packed['pins']), input=artifact(out),
                               array_bundle_sha256=array_bundle_sha256(packed)))
        assert len(all_ids) == 22498 and len(source_audits) == 95 and len(counts) == 100
        check_path = LOG / 'prepare/input_audit.json'
        atomic_write_json(check_path, dict(status='pass', checked_source_files=source_audits,
            source_replay_nets=22498, duplicate_net_ids=0, missing_degree_files=5,
            counts=counts, models_verified=75, frozen_source_hashes_verified=len(frozen['code_files']),
            model_calls=0, preflight=artifact(preflight)), mode=0o444)
        dependencies.append(artifact(check_path))
        p = dict(schema_version=1, stage=7, status='frozen', at_utc=utc_now(),
            protocol='A', revision='reference_precision_v2', dependencies=dependencies,
            models=models, groups=groups, counts=counts, reference_identity=reference_identity(),
            inference=dict(batch_size=256, T=1, mode='deterministic_greedy',
                           hardware_class='NVIDIA RTX 6000 Ada Generation', batch_order='degree then frozen design order then source net order'),
            reference=dict(batch_size=128, primary_scale=2**27, verification_scale=2**28,
                           integral_geometry_required=True),
            statistics=dict(seed_aggregation='arithmetic mean of the three independently computed seed metrics',
                degree_aggregation='pool all degree nets before computing each seed metric',
                design_macro='equal mean over the 20 design means',
                p95='numpy linear quantile across all nets, computed separately for each seed',
                worst_design='maximum design mean per seed; report mean of three seed maxima',
                missing='retain all 100 design-degree rows; missing is not invalid',
                confidence_intervals='final cross-experiment inference belongs to Stage8; Stage7 reports all seeds and paired net differences'),
            expected_models=75, expected_design_degree_cells=1425, expected_predictions=337470,
            model_metrics_unlocked_after_freeze=True, train_or_retune=False, researcher_blind_claim=False)
        atomic_write_json(PLAN, p, mode=0o444)
        lease.update(95, 'pass')
        progress('7.1 冻结模型及ISPD输入核验通过', 'DONE', str(check_path),
                 '生成双尺度参考缓存并验证评价恢复；通过后执行75模型外部测试')
        return p

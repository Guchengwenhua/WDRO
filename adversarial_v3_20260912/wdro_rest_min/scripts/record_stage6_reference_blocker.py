"""Verify geometric counterexamples and publish a Stage 6 recovery handoff."""
import ctypes as C
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT.parent
EXP = CODEX / 'experiments'
OUT = EXP / 'logs/stage6_negative_regret'
sys.path.insert(0, str(ROOT))
import numpy as np
from src.datasets import load_dataset
from src.geosteiner import GeoSteinerEvaluator, DEFAULT_LIBRARY
from src.io_utils import atomic_write_json
from src.job_control import utc_now
from src.training_data import materialize_training_batch
from src.protocol import load_protocol
from src.rest import file_sha256
from stage6.evaluation import artifact

def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] not in ('Z', 'X')
    except FileNotFoundError:
        return False

def main():
    scope = json.loads((OUT / 'reference_scope.json').read_text())
    assert scope['status'] == 'complete' and len(scope['groups']) == 25 and scope['total_nets'] == 1920
    solver = json.loads((OUT / 'solver_status.json').read_text())
    assert solver['complete'] and len(solver['records']) == 8
    assert json.loads((OUT / 'frozen_models_recheck.log').read_text()) == dict(status='pass', models=75, reused_freeze=True)
    index = json.loads((ROOT / 'manifests/stage6_models_frozen_v1.json').read_text())
    assert len(index['models']) == 75 and not index['test_evaluated_at_freeze']
    lib = GeoSteinerEvaluator()._library
    dp, ip = C.POINTER(C.c_double), C.POINTER(C.c_int)
    lib.gst_rsmt.argtypes = [C.c_int, dp, dp, ip, dp, ip, ip, ip, C.c_void_p]
    lib.gst_rsmt.restype = C.c_int
    witnesses = []
    for group in scope['groups']:
        if not group['changed']:
            continue
        if group['name'].startswith('validation:'):
            data, _ = load_dataset(group['sources']['dataset']['logical_path'], verify=True)
            pins = data['pins']
        else:
            degree = int(group['name'].split(':')[2][1:])
            batch = materialize_training_batch(protocol=load_protocol(), degree=degree, training_seed=101, update=1, batch_size=256)
            assert batch.bundle_sha256 == group['sources']['training_bundle']
            pins = batch.pins
        for item in group['changed']:
            n = pins.shape[1]
            points = np.ascontiguousarray(pins[item['index']].astype(np.float64) * 2**27)
            assert np.array_equal(points, np.rint(points))
            ns, ne, status = C.c_int(), C.c_int(), C.c_int(-99)
            steiners = np.zeros((n*4, 2), dtype=np.float64)
            edges = np.zeros((n*6, 2), dtype=np.int32)
            length = C.c_double()
            rc = lib.gst_rsmt(n, points.ctypes.data_as(dp), C.byref(length), C.byref(ns),
                             steiners.ctypes.data_as(dp), C.byref(ne), edges.ctypes.data_as(ip), C.byref(status), None)
            assert rc == 0 and status.value == 0
            nodes = np.concatenate((points, steiners[:ns.value]))
            assert np.array_equal(nodes, np.rint(nodes))
            nodes = nodes.astype(np.int64)
            topology = edges[:ne.value]
            assert np.all(topology >= 0) and np.all(topology < len(nodes))
            parent = list(range(len(nodes)))
            def root(k):
                while parent[k] != k:
                    k = parent[k]
                return k
            for a, b in topology:
                parent[root(int(a))] = root(int(b))
            assert len({root(i) for i in range(len(nodes))}) == 1
            integer_length = int(np.abs(nodes[topology[:,0]] - nodes[topology[:,1]]).sum())
            assert integer_length == length.value
            assert integer_length / 2**27 == item['scaled_27'] < item['default']
            witnesses.append(dict(net_id=item['net_id'], degree=n, default_length=item['default'],
                                  feasible_integer_length=integer_length, divisor=2**27,
                                  feasible_length=integer_length/2**27, connected_components=1,
                                  nodes_integer=nodes.tolist(), edges=topology.tolist()))
    atomic_write_json(OUT / 'feasible_tree_witnesses.json', dict(status='pass', at_utc=utc_now(), witnesses=witnesses,
                      proof='integer Manhattan edge sums and connected graph through every terminal; feasible upper bounds only'))
    cells = []
    for path in sorted((EXP / 'logs/stage6_test').glob('*/*/cursor.json')):
        cursor = json.loads(path.read_text())
        for shard in cursor['shards']:
            item = shard['artifact']
            assert file_sha256(Path(item['logical_path'])) == item['sha256']
        result = path.parent / 'result.json'
        if result.exists():
            value = json.loads(result.read_text())
            assert value['status'] == 'complete' and value['completed_nets'] == 10000
            assert value['binding'] == cursor['binding']
            assert file_sha256(Path(value['predictions']['logical_path'])) == value['predictions']['sha256']
        cells.append(dict(cell=str(path.parent.relative_to(EXP / 'logs/stage6_test')), cursor=artifact(path),
                          complete_nets=cursor['complete_nets'], complete=result.exists(), shards=len(cursor['shards'])))
    jobs = []
    for path in [EXP / 'logs/stage6_pipeline/job.json', *sorted((EXP/'logs/stage6_evaluation_jobs').glob('*/job.json'))]:
        job = json.loads(path.read_text())
        jobs.append(dict(artifact=artifact(path), pid=job['pid'], alive=alive(job['pid']), recorded_status=job['status']))
    assert not any(job['alive'] for job in jobs)
    full_checks = 0
    for item in index['full_wdro_audits']:
        assert file_sha256(Path(item['logical_path'])) == item['sha256']
        audit = json.loads(Path(item['logical_path']).read_text())
        assert audit['status'] == 'pass' and all(audit['checks'].values()) and audit['completed_updates'] == 2000
        full_checks += len(audit['checks'])
    commands = [
        'bash wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python wdro_rest_min/scripts/capture_host_runtime.py --purpose stage6_reference_diagnosis --measured-peak-mib 1319 --safety-margin-mib 2048',
        'bash wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python wdro_rest_min/scripts/diagnose_stage6_geosteiner_status.py',
        'bash wdro_rest_min/scripts/stage3_env.sh /remote-home/Guilin/miniconda3/envs/py311/bin/python wdro_rest_min/scripts/record_stage6_reference_blocker.py',
    ]
    report = dict(status='blocked', stage=6, substep='6.3', at_utc=utc_now(),
                  reason='A connected feasible tree is shorter than the frozen GeoSteiner reference; the issue also occurs in validation and a training input.',
                  formal_training=dict(models=75, new_wdro_runs=14, new_wdro_updates=28000,
                                       full_new_run_audits=14, full_new_run_checks=full_checks,
                                       frozen_model_hashes_and_states_recheck='pass'),
                  test=dict(expected_cells=300, complete_cells=sum(c['complete'] for c in cells),
                            committed_prediction_nets=sum(c['complete_nets'] for c in cells), cells=cells), jobs=jobs,
                  diagnosis=dict(single_test_net=solver['net_id'], scope_nets=1920,
                                 validation_changed=sum(g['changed_count'] for g in scope['groups'] if g['name'].startswith('validation:')),
                                 training_changed=sum(g['changed_count'] for g in scope['groups'] if g['name'].startswith('training:')),
                                 independently_verified_feasible_witnesses=len(witnesses),
                                 all_scope_integer_scales_agree=all(g['two_integer_scales_identical'] for g in scope['groups']),
                                 geosteiner_calls=1+8+scope['geosteiner_calls']+len(witnesses), actor_rest_calls=256,
                                 limitation='Bounded diagnostic sample, not a population error-rate estimate or a proof of global solver exactness'),
                  evidence=[artifact(OUT/name) for name in ['diagnosis.json','failed_batch.npz','solver_status.json','reference_scope.json',
                            'feasible_tree_witnesses.json','frozen_models_recheck.log']],
                  code=[artifact(ROOT/'scripts'/name) for name in ['diagnose_stage6_negative_regret.py','diagnose_stage6_geosteiner_status.py',
                            'diagnose_stage6_reference_scope.py','record_stage6_reference_blocker.py']],
                  legacy_library=artifact(DEFAULT_LIBRARY), frozen_models=artifact(ROOT/'manifests/stage6_models_frozen_v1.json'),
                  current_host_runtime=artifact(ROOT/'manifests/host_runtime_20260907T031133Z.json'),
                  frozen_files_modified=False, launch_authorized=False,
                  reproduction_commands=commands,
                  monitor_command='cat wdro_rest_min/manifests/stage6_reference_blocker_v1.json',
                  resume_gate='Do not run launch_stage6.py until the reference correction scope is approved and separately validated. Existing training/test cursors remain preserved.',
                  proposed_scope='Versioned GeoSteiner reference precision repair, audit fixed references and affected training/validation; replay affected stages under the same frozen research settings before repeating Stage 6 test. No test-driven tuning.')
    atomic_write_json(ROOT/'manifests/stage6_reference_blocker_v1.json', report)
    print(json.dumps({k:report[k] for k in ('status','formal_training','diagnosis')}))
    print(json.dumps({k:v for k,v in report['test'].items() if k != 'cells'}))

if __name__ == '__main__':
    main()

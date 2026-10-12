"""Resumable exact references and deterministic external inference."""
from __future__ import annotations

import signal
import time

import numpy as np
import torch

from src import geosteiner as gst
from src.datasets import array_bundle_sha256
from src.metrics import relative_regret, summarize_regret
from src.rest import build_actor, RestTreeEvaluator
from src.trainer import configure_determinism, load_trainer_config
from stage7.common import *

STOP = False


def request_stop(*unused):
    global STOP
    STOP = True


def read_shards(state):
    parts = []
    end = 0
    for shard in state['shards']:
        assert shard['start'] == end and shard['end'] > end
        part = load_npz(checked(shard['artifact']))
        assert all(len(v) == shard['end'] - end for v in part.values())
        parts.append(part)
        end = shard['end']
    assert end == state['complete_nets']
    return {k: np.concatenate([v[k] for v in parts]) for k in parts[0]} if parts else {}


def load_group(degree):
    plan = gate()
    group = next(g for g in plan['groups'] if g['degree'] == degree)
    data = load_npz(checked(group['input']))
    assert len(data['pins']) == group['count']
    assert array_bundle_sha256(data) == group['array_bundle_sha256']
    return plan, group, data


def reference_task(degree, preflight, max_new_batches=None):
    global STOP
    STOP = False
    signal.signal(signal.SIGTERM, request_stop)
    plan, group, data = load_group(degree)
    directory = CACHE / 'references' / f'degree_{degree}'
    binding = dict(input=group, reference_identity=gst.reference_identity(),
                   scales=[2**27, 2**28], source_sha256=file_sha256(Path(__file__)))
    n = group['count']
    with Lease(LOG / 'references' / f'degree_{degree}', f'stage7_reference_d{degree}', preflight) as lease:
        cursor = directory / 'cursor.json'
        state = read(cursor) if cursor.exists() else dict(binding=binding, shards=[], complete_nets=0)
        assert state['binding'] == binding, 'reference input changed'
        read_shards(state)
        result_path = directory / 'result.json'
        if result_path.exists():
            r = read(result_path)
            assert r['status'] == 'complete' and r['binding'] == binding
            checked(r['reference'])
            lease.update(n, 'complete')
            return r
        evaluator = gst.GeoSteinerEvaluator()
        added = 0
        events = LOG / 'references' / f'degree_{degree}' / 'calls.jsonl'
        for start in range(state['complete_nets'], n, 128):
            end = min(n, start + 128)
            attempt = stamp()
            append_jsonl(events, dict(id=attempt, phase='started', start=start, end=end,
                planned_primary_calls=end-start, planned_verification_calls=end-start, at_utc=utc_now()))
            began = time.monotonic()
            assert gst.SCALE == 2**27
            primary = np.array([evaluator.solve(x, require_integral=True) for x in data['pins'][start:end]])
            try:
                gst.SCALE = 2**28
                secondary = np.array([evaluator.solve(x, require_integral=True) for x in data['pins'][start:end]])
            finally:
                gst.SCALE = 2**27
            np.testing.assert_array_equal(primary, secondary)
            arrays = dict(net_ids=data['net_ids'][start:end], lstar=primary, verification_lstar=secondary)
            path = directory / 'shards' / f'{start:06d}_{end:06d}.npz'
            atomic_npz(path, arrays)
            append_jsonl(events, dict(id=attempt, phase='completed', primary_calls=end-start,
                verification_calls=end-start, wall_seconds=time.monotonic()-began, at_utc=utc_now()))
            state['shards'].append(dict(start=start, end=end, artifact=artifact(path), attempt=attempt))
            state.update(complete_nets=end, updated_at_utc=utc_now())
            atomic_write_json(cursor, state)
            lease.update(end, 'references')
            added += 1
            if end < n and (STOP or (max_new_batches is not None and added >= max_new_batches)):
                lease.update(end, 'paused')
                return dict(status='paused', completed_nets=end, cursor=artifact(cursor))
        arrays = read_shards(state)
        np.testing.assert_array_equal(arrays['net_ids'], data['net_ids'])
        np.testing.assert_array_equal(arrays['lstar'], arrays['verification_lstar'])
        assert np.all(np.isfinite(arrays['lstar'])) and np.all(arrays['lstar'] > 0)
        out = directory / f"{group['array_bundle_sha256']}.npz"
        atomic_npz(out, arrays)
        out.chmod(0o444)
        r = dict(status='complete', completed_nets=n, at_utc=utc_now(), binding=binding,
                 reference=artifact(out), committed_primary_calls=n, committed_verification_calls=n,
                 interrupted_calls_accounting='started/completed events retained; unfinished calls are bounded by that batch',
                 checks=dict(two_scales_equal=True, optimal_status_all=True,
                             connected_integer_embedding_all=True, original_float32_input_unchanged=True))
        atomic_write_json(result_path, r, mode=0o444)
        lease.update(n, 'complete')
        return r


def reference_gate():
    plan = gate()
    rows = []
    for g in plan['groups']:
        path = CACHE / 'references' / f"degree_{g['degree']}" / 'result.json'
        r = read(path)
        assert r['status'] == 'complete' and r['completed_nets'] == g['count']
        assert r['binding']['input'] == g and r['binding']['reference_identity'] == gst.reference_identity()
        assert r['binding']['source_sha256'] == file_sha256(Path(__file__))
        ref = load_npz(checked(r['reference']))
        data = load_npz(checked(g['input']))
        np.testing.assert_array_equal(ref['net_ids'], data['net_ids'])
        np.testing.assert_array_equal(ref['lstar'], ref['verification_lstar'])
        assert np.all(ref['lstar'] > 0) and np.isfinite(ref['lstar']).all()
        rows.append(artifact(path))
    return rows


def model_id(model):
    return f"{model['method']}_d{model['degree']}_s{model['seed']}"


def evaluate(model, preflight, gpu, *, output=None, limit=None, max_new_batches=None):
    global STOP
    STOP = False
    signal.signal(signal.SIGTERM, request_stop)
    plan, group, data = load_group(model['degree'])
    assert model in plan['models']
    references = read(CACHE / 'references' / f"degree_{model['degree']}" / 'result.json')
    assert references['binding']['input'] == group and references['status'] == 'complete'
    assert references['binding']['reference_identity'] == gst.reference_identity()
    ref = load_npz(checked(references['reference']))
    np.testing.assert_array_equal(ref['net_ids'], data['net_ids'])
    n = group['count'] if limit is None else min(limit, group['count'])
    directory = output or LOG / 'evaluation' / model_id(model)
    directory = Path(directory)
    cursor, result_path = directory / 'cursor.json', directory / 'result.json'
    checkpoint = checked(model['best'])
    assert torch.cuda.get_device_name(0) == plan['inference']['hardware_class']
    hardware = dict(name=torch.cuda.get_device_name(0), torch_version=torch.__version__, cuda_version=torch.version.cuda)
    binding = dict(model=model, input=group, reference=references['reference'], count=n,
        batch_size=256, T=1, mode='deterministic_greedy', hardware=hardware,
        evaluator_sha256=file_sha256(Path(__file__)), plan=artifact(PLAN), role='external_test')
    with Lease(directory, 'stage7_' + model_id(model), preflight) as lease:
        state = read(cursor) if cursor.exists() else dict(binding=binding, complete_nets=0, shards=[])
        assert state['binding'] == binding, 'evaluation input changed; refusing resume'
        read_shards(state)
        if result_path.exists():
            result = read(result_path)
            assert result['status'] == 'complete' and result['binding'] == binding
            checked(result['predictions'])
            lease.update(n, 'complete')
            return result
        configure_determinism(load_trainer_config())
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert ck['run_identity']['degree'] == model['degree'] and ck['run_identity']['training_seed'] == model['seed']
        actor = build_actor(model['degree'], torch.device('cuda:0'))
        actor.load_state_dict(ck['actor_state_dict'], strict=True)
        actor.eval()
        rest = RestTreeEvaluator()
        torch.cuda.reset_peak_memory_stats()
        added = 0
        for start in range(state['complete_nets'], n, 256):
            end = min(n, start + 256)
            attempt = stamp()
            append_jsonl(directory / 'calls.jsonl', dict(id=attempt, phase='started', start=start, end=end,
                actor_planned=end-start, rest_planned=end-start, physical_gpu=gpu, at_utc=utc_now()))
            pins = data['pins'][start:end]
            torch.cuda.synchronize()
            began = time.monotonic()
            with torch.no_grad():
                topology, _ = actor(pins, True)
            topology = topology.detach().cpu().numpy()
            torch.cuda.synchronize()
            actor_seconds = time.monotonic() - began
            began = time.monotonic()
            lengths = rest.lengths(pins, topology)
            rest_seconds = time.monotonic() - began
            lstar = ref['lstar'][start:end]
            regret = relative_regret(lengths, lstar)
            assert np.isfinite(regret).all() and regret.min() >= -1e-12, 'negative reference regret; stop Stage7'
            arrays = dict(net_ids=data['net_ids'][start:end], design_ids=data['design_ids'][start:end],
                source_index=data['source_index'][start:end], rest_lengths=lengths, lstar=lstar,
                regret=regret, topology=topology, rest_lengths_inverse=lengths/data['scales'][start:end],
                lstar_inverse=lstar/data['scales'][start:end])
            path = directory / 'shards' / f'{start:06d}_{end:06d}.npz'
            atomic_npz(path, arrays)
            append_jsonl(directory / 'calls.jsonl', dict(id=attempt, phase='completed',
                actor_calls=end-start, rest_calls=end-start, actor_seconds=actor_seconds,
                rest_seconds=rest_seconds, at_utc=utc_now()))
            state['shards'].append(dict(start=start, end=end, artifact=artifact(path), attempt=attempt,
                actor_seconds=actor_seconds, rest_seconds=rest_seconds, physical_gpu=gpu))
            state.update(complete_nets=end, updated_at_utc=utc_now(),
                peak_reserved_mib=max(state.get('peak_reserved_mib', 0), torch.cuda.max_memory_reserved()/2**20),
                rng_policy='deterministic eval/greedy; no random draw; reinitialize identical torch configuration on resume')
            atomic_write_json(cursor, state)
            lease.update(end, 'evaluating', physical_gpu=gpu, peak_reserved_mib=state['peak_reserved_mib'])
            added += 1
            if end < n and (STOP or (max_new_batches is not None and added >= max_new_batches)):
                lease.update(end, 'paused')
                return dict(status='paused', completed_nets=end, cursor=artifact(cursor))
        arrays = read_shards(state)
        np.testing.assert_array_equal(arrays['net_ids'], data['net_ids'][:n])
        np.testing.assert_array_equal(arrays['lstar'], ref['lstar'][:n])
        np.testing.assert_array_equal(arrays['regret'], relative_regret(arrays['rest_lengths'], arrays['lstar']))
        path = directory / 'predictions.npz'
        atomic_npz(path, arrays)
        path.chmod(0o444)
        result = dict(status='complete', at_utc=utc_now(), binding=binding, completed_nets=n,
            selected_update=ck['update'], predictions=artifact(path), summary=summarize_regret(arrays['regret']),
            call_ledger=dict(actor_forward_nets=n, exact_rest_evaluator_nets=n, geosteiner_nets=0),
            peak_reserved_mib=state['peak_reserved_mib'],
            actor_seconds=sum(s['actor_seconds'] for s in state['shards']),
            rest_seconds=sum(s['rest_seconds'] for s in state['shards']),
            model_selection=False, accounting='committed batches; all attempts separately retained')
        atomic_write_json(result_path, result, mode=0o444)
        lease.update(n, 'complete')
        return result


def python_tree_lengths(pins, topology):
    """Independent rectilinear segment-union length and connectivity check."""
    nets = np.asarray(pins, dtype=np.float64)
    t = np.asarray(topology)
    count, degree, _ = nets.shape
    assert t.shape == (count, 2*(degree-1)) and np.issubdtype(t.dtype, np.integer)
    assert np.all(t >= 0) and np.all(t < degree)
    result = np.empty(count)
    for i, (xy, pairs) in enumerate(zip(nets, t.reshape(count, degree-1, 2), strict=True)):
        xlo, xhi = xy[:, 0].copy(), xy[:, 0].copy()
        ylo, yhi = xy[:, 1].copy(), xy[:, 1].copy()
        parent = list(range(degree))
        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for x, y in pairs:
            ylo[x], yhi[x] = min(ylo[x], xy[y, 1]), max(yhi[x], xy[y, 1])
            xlo[y], xhi[y] = min(xlo[y], xy[x, 0]), max(xhi[y], xy[x, 0])
            parent[root(int(x))] = root(int(y))
        assert len({root(a) for a in range(degree)}) == 1, 'REST tree disconnected'
        result[i] = np.sum(xhi-xlo) + np.sum(yhi-ylo)
    return result

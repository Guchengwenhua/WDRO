"""CEM-256 with immutable per-round receipts and exact round-boundary recovery."""
from __future__ import annotations

import signal
import time
from types import SimpleNamespace

import numpy as np
import torch

from src import geosteiner as gst
from src.geometry import pin_set_sha256
from src.metrics import relative_regret, summarize_regret
from src.rest import build_actor, RestTreeEvaluator
from src.trainer import configure_determinism, load_trainer_config
from stage5.search import cem_round, transport_costs
from stage8.common import *

STOP = False


def request_stop(*unused):
    global STOP
    STOP = True


def model_id(model):
    return f"{model['method']}_d{model['degree']}_s{model['seed']}"


def task_id(model, distribution):
    return model_id(model) + '_' + distribution


def adapt(centers, pins, regret, minimum_std):
    """The attack maximizes regret, with no learned model-dependent price."""
    elite = np.argsort(-regret, axis=1, kind='stable')[:, :4]
    chosen = pins[np.arange(len(pins))[:, None], elite]
    offsets = np.empty_like(chosen, dtype=np.float64)
    for i, center in enumerate(centers):
        order = np.lexsort((center[:, 1], center[:, 0]))
        offsets[i] = chosen[i][:, order] - center[order]
    return offsets.mean(axis=1), np.maximum(offsets.std(axis=1), minimum_std), elite


def independent_tree_lengths(pins, topology):
    """Independent integer segment union plus disjoint-set connectivity.

    REST's topology (x,y) connects terminal x's vertical segment to terminal
    y's horizontal segment. Reconstruct those unions without the C evaluator.
    """
    xy = np.asarray(pins, dtype=np.float64) * 2**27
    assert np.array_equal(xy, np.rint(xy))
    xy = xy.astype(np.int64)
    count, degree, _ = xy.shape
    t = np.asarray(topology)
    assert t.shape == (count, 2*(degree-1)) and np.issubdtype(t.dtype, np.integer)
    assert np.all(t >= 0) and np.all(t < degree)
    pairs = t.reshape(count, degree-1, 2)
    rows = np.arange(count)
    parent = np.broadcast_to(np.arange(degree), (count, degree)).copy()
    xlo, xhi = xy[:, :, 0].copy(), xy[:, :, 0].copy()
    ylo, yhi = xy[:, :, 1].copy(), xy[:, :, 1].copy()
    def roots(nodes):
        for _ in range(degree):
            nxt = parent[rows, nodes]
            if np.array_equal(nxt, nodes):
                return nodes
            nodes = nxt
        raise AssertionError('cyclic disjoint-set state')
    for j in range(degree-1):
        a, b = pairs[:, j, 0], pairs[:, j, 1]
        ylo[rows, a] = np.minimum(ylo[rows, a], xy[rows, b, 1])
        yhi[rows, a] = np.maximum(yhi[rows, a], xy[rows, b, 1])
        xlo[rows, b] = np.minimum(xlo[rows, b], xy[rows, a, 0])
        xhi[rows, b] = np.maximum(xhi[rows, b], xy[rows, a, 0])
        ra, rb = roots(a), roots(b)
        parent[rows, rb] = ra
    components = parent.copy()
    for _ in range(degree):
        nxt = np.take_along_axis(parent, components, axis=1)
        if np.array_equal(nxt, components):
            break
        components = nxt
    assert np.all(components == components[:, :1]), 'REST tree disconnected'
    return ((xhi-xlo).sum(axis=1) + (yhi-ylo).sum(axis=1)).astype(np.float64) / 2**27


def restore_rounds(centers, rows, config, ids, *, reconstruct=False):
    b, n, _ = centers.shape
    mean = np.zeros((b, n, 2))
    std = np.full_like(mean, config['cem']['initial_std'])
    seen = [set() for _ in range(b)]
    records = []
    batch = SimpleNamespace(pins=centers, net_ids=ids)
    for index, row in enumerate(rows):
        assert row['round'] == index
        a = load_npz(checked(row['artifact']))
        if reconstruct:
            proposal, repaired = cem_round(batch, update=1, training_seed=config['seed'],
                round_index=index, mean=mean, std=std, seen=seen, config=config)
            np.testing.assert_array_equal(proposal, a['pins'])
            assert repaired == row['repaired']
        else:
            for i in range(b):
                for p in a['pins'][i]:
                    key = pin_set_sha256(p)
                    assert key not in seen[i], 'duplicate CEM candidate'
                    seen[i].add(key)
        mean, std, elite = adapt(centers, a['pins'], a['regret'], config['cem']['minimum_std'])
        np.testing.assert_array_equal(mean, a['next_mean'])
        np.testing.assert_array_equal(std, a['next_std'])
        np.testing.assert_array_equal(elite, a['elite'])
        records.append(a)
    return mean, std, seen, records


def reduce_candidates(centers, ids, records):
    assert len(records) == 16
    pins = np.concatenate([a['pins'] for a in records], axis=1)
    regret = np.concatenate([a['regret'] for a in records], axis=1)
    lstar = np.concatenate([a['lstar'] for a in records], axis=1)
    length = np.concatenate([a['rest_lengths'] for a in records], axis=1)
    topology = np.concatenate([a['topology'] for a in records], axis=1)
    np.testing.assert_array_equal(pins[:, 0], centers)
    best = regret.argmax(axis=1)
    row = np.arange(len(centers))
    selected = pins[row, best]
    cost = transport_costs(centers, selected[:, None])[:, 0]
    assert np.all(regret[row, best] >= regret[:, 0]) and np.all(cost >= 0)
    return dict(net_ids=ids, pins=selected, selected_index=best, cost=cost,
        regret=regret[row, best], rest_lengths=length[row, best], lstar=lstar[row, best],
        topology=topology[row, best], original_regret=regret[:, 0], original_lstar=lstar[:, 0],
        changed=(best != 0), candidates_per_center=np.full(len(centers), 256, dtype=np.int16))


def attempt_event(directory, attempt, phase, **detail):
    append_jsonl(directory / 'calls.jsonl', dict(id=attempt, phase=phase, at_utc=utc_now(), **detail))


def attack(model, distribution, gpu, preflight, *, output=None, limit=None, max_new_rounds=None, workers=3):
    global STOP
    STOP = False
    signal.signal(signal.SIGTERM, request_stop)
    plan = gate()
    config = plan['attack']
    group = next(g for g in plan['groups'] if g['degree'] == model['degree'] and g['distribution'] == distribution)
    data = load_npz(checked(group['input']))
    n = group['count'] if limit is None else min(limit, group['count'])
    directory = Path(output) if output else LOG / 'attacks' / task_id(model, distribution)
    assert directory.is_absolute() and directory.is_relative_to(CODEX)
    checkpoint = checked(model['best'])
    assert torch.cuda.get_device_name(0) == plan['inference']['hardware_class']
    binding = dict(plan=artifact(PLAN), model=model, input=group, count=n, attack=config,
        source_sha256=file_sha256(Path(__file__)), reference_identity=gst.reference_identity(),
        hardware=dict(name=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda),
        role='independent_attack', model_selection=False)
    with Lease(directory, 'stage8_' + task_id(model, distribution), str(preflight)) as lease:
        cursor = directory / 'cursor.json'
        state = read(cursor) if cursor.exists() else dict(binding=binding, complete_nets=0, batches=[], current=None,
            rng_policy='stateless per attack net ID and round; identical across models; CEM state restored from committed rounds')
        assert state['binding'] == binding, 'attack input/code changed; refusing resume'
        for batch in state['batches']:
            checked(batch['predictions'])
            for row in batch['rounds']:
                checked(row['artifact'])
        result_path = directory / 'result.json'
        if result_path.exists():
            r = read(result_path)
            assert r['status'] == 'complete' and r['binding'] == binding and r['completed_nets'] == n
            checked(r['predictions'])
            lease.update(n, 'complete')
            return r
        configure_determinism(load_trainer_config())
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert ck['run_identity']['degree'] == model['degree'] and ck['run_identity']['training_seed'] == model['seed']
        actor = build_actor(model['degree'], torch.device('cuda:0'))
        actor.load_state_dict(ck['actor_state_dict'], strict=True)
        actor.eval()
        rest = RestTreeEvaluator()
        torch.cuda.reset_peak_memory_stats()
        added = 0
        with gst.ParallelGeoSteinerEvaluator(workers) as reference:
            for start in range(state['complete_nets'], n, config['center_batch']):
                end = min(n, start + config['center_batch'])
                centers, ids = data['pins'][start:end], data['net_ids'][start:end]
                current = state['current'] or dict(start=start, end=end, rounds=[])
                assert (current['start'], current['end']) == (start, end)
                mean, std, seen, records = restore_rounds(centers, current['rounds'], config, ids)
                batch = SimpleNamespace(pins=centers, net_ids=ids)
                for t in range(len(records), 16):
                    began = time.monotonic()
                    pins, repaired = cem_round(batch, update=1, training_seed=config['seed'], round_index=t,
                        mean=mean, std=std, seen=seen, config=config)
                    flat = pins.reshape(-1, model['degree'], 2)
                    calls, attempt = len(flat), stamp()
                    assert calls <= 256 and pins.dtype == np.float32
                    attempt_event(directory, attempt, 'started', start=start, end=end, round=t,
                        actor_planned=calls, rest_planned=calls, reference_planned=calls, gpu=gpu)
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    with torch.no_grad():
                        topology, _ = actor(flat, True)
                    topology = topology.cpu().numpy().astype(np.int16)
                    torch.cuda.synchronize()
                    actor_seconds = time.monotonic() - tick
                    attempt_event(directory, attempt, 'actor_completed', calls=calls, seconds=actor_seconds)
                    tick = time.monotonic()
                    lengths = rest.lengths(flat, topology.astype(np.int32))
                    rest_seconds = time.monotonic() - tick
                    attempt_event(directory, attempt, 'rest_completed', calls=calls, seconds=rest_seconds)
                    np.testing.assert_array_equal(independent_tree_lengths(flat, topology), lengths)
                    tick = time.monotonic()
                    lstar = reference.lengths(flat)
                    reference_seconds = time.monotonic() - tick
                    attempt_event(directory, attempt, 'reference_completed', calls=calls, seconds=reference_seconds)
                    regret = relative_regret(lengths, lstar).reshape(len(centers), 16)
                    assert np.isfinite(regret).all() and regret.min() >= -1e-12, 'negative reference regret; stop Stage8'
                    mean, std, elite = adapt(centers, pins, regret, config['cem']['minimum_std'])
                    arrays = dict(pins=pins, topology=topology.reshape(len(centers), 16, -1),
                        rest_lengths=lengths.reshape(len(centers), 16), lstar=lstar.reshape(len(centers), 16),
                        regret=regret, next_mean=mean, next_std=std, elite=elite)
                    path = directory / 'rounds' / f'{start:06d}_r{t:02d}.npz'
                    atomic_npz(path, arrays)
                    row = dict(round=t, artifact=artifact(path), attempt=attempt, calls=calls, repaired=repaired,
                        actor_seconds=actor_seconds, rest_seconds=rest_seconds, reference_seconds=reference_seconds,
                        wall_seconds=time.monotonic()-began, physical_gpu=gpu)
                    current['rounds'].append(row)
                    records.append(arrays)
                    state.update(current=current, updated_at_utc=utc_now(), peak_reserved_mib=max(
                        state.get('peak_reserved_mib', 0), torch.cuda.max_memory_reserved()/2**20))
                    atomic_write_json(cursor, state)
                    lease.update(start, 'attacking', round=t+1, current_end=end, physical_gpu=gpu,
                                 committed_search_calls=start*256+(t+1)*calls,
                                 peak_reserved_mib=state['peak_reserved_mib'])
                    added += 1
                    if STOP or (max_new_rounds is not None and added >= max_new_rounds):
                        lease.update(start, 'paused', round=t+1, cursor=str(cursor))
                        return dict(status='paused', completed_nets=start)
                chosen = reduce_candidates(centers, ids, records)
                attempt = stamp()
                attempt_event(directory, attempt, 'verification_started', planned=end-start, start=start, end=end)
                began = time.monotonic()
                try:
                    assert gst.SCALE == 2**27
                    gst.SCALE = 2**28
                    chosen['verification_lstar'] = np.array([reference.local.solve(x, require_integral=True) for x in chosen['pins']])
                finally:
                    gst.SCALE = 2**27
                np.testing.assert_array_equal(chosen['verification_lstar'], chosen['lstar'])
                verify_seconds = time.monotonic()-began
                attempt_event(directory, attempt, 'verification_completed', calls=end-start, seconds=verify_seconds)
                dest = directory / 'selected' / f'{start:06d}_{end:06d}.npz'
                atomic_npz(dest, chosen)
                current.update(predictions=artifact(dest), verification_attempt=attempt, verification_seconds=verify_seconds)
                state['batches'].append(current)
                state.update(complete_nets=end, current=None, updated_at_utc=utc_now())
                atomic_write_json(cursor, state)
                lease.update(end, 'attacking', committed_search_calls=end*256, physical_gpu=gpu)
        parts = [load_npz(checked(b['predictions'])) for b in state['batches']]
        arrays = {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}
        np.testing.assert_array_equal(arrays['net_ids'], data['net_ids'][:n])
        assert len(arrays['net_ids']) == n and len(set(arrays['net_ids'])) == n
        dest = directory / 'predictions.npz'
        atomic_npz(dest, arrays)
        dest.chmod(0o444)
        rounds = [r for b in state['batches'] for r in b['rounds']]
        r = dict(status='complete', at_utc=utc_now(), binding=binding, completed_nets=n,
            predictions=artifact(dest), selected_update=ck['update'], summary=summarize_regret(arrays['regret']),
            original_summary=summarize_regret(arrays['original_regret']), cost=summarize_regret(arrays['cost']),
            changed_fraction=float(arrays['changed'].mean()), legal_fraction=1.0,
            repaired_candidates=sum(x['repaired'] for x in rounds),
            call_ledger=dict(actor_forward_nets=n*256, exact_rest_evaluator_nets=n*256,
                geosteiner_search_nets=n*256, geosteiner_selected_verification_nets=n),
            times={k:sum(r[k] for r in rounds) for k in ['actor_seconds','rest_seconds','reference_seconds','wall_seconds']},
            verification_seconds=sum(b['verification_seconds'] for b in state['batches']),
            peak_reserved_mib=state['peak_reserved_mib'], model_selection=False,
            accounting='committed rounds; completed replay attempts and unfinished-call bounds kept in calls.jsonl')
        atomic_write_json(result_path, r, mode=0o444)
        lease.update(n, 'complete')
        return r

#!/usr/bin/env python3
"""Evaluate one frozen v3.1 joint model or precision_v2 control on all real degree-30 nets.

No training or model selection takes place. Input arrays are pins float32[N,d,2],
net_ids, design_ids, source_index, offsets and scales. Reference arrays are
net_ids, lstar and verification_lstar. Optional per-net arrays, including
dataset_ids/family_ids, are preserved in predictions. The returned topology is
the Actor's ordered REST action encoding (x-index, y-index pairs).

CLI outputs an immutable predictions.npz/result.json, resumable cursor/shards,
attempt ledger and process/heartbeat records under --output. SIGTERM pauses at
the next committed batch; flock rejects concurrent writers. Completed outputs
are validated, including all shard hashes and independent tree-length audits,
before an idempotent return. Run through the existing stage3_env.sh.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np

TOP = Path('/remote-home/Guilin/DRO/Codex')
EXTENSION = Path('/remote-home/Guilin/DRO/Codex/benchmarks/v31_models_20260912')
REVISION = Path('/remote-home/Guilin/DRO/Codex/wdro_precision_v2')
PACKAGE = REVISION / 'wdro_rest_min'
PARENT_FROZEN = PACKAGE / 'manifests/stage6_models_frozen_v1.json'
FROZEN = EXTENSION / 'artifacts/models_frozen.json'
STOP = False


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def project_path(path):
    result = Path(os.path.abspath(path))
    require(result.is_relative_to(TOP), f'path must stay under {TOP}: {result}')
    return result


def artifact(path):
    source = project_path(path)
    return dict(logical_path=str(source), resolved_path=str(source.resolve()), sha256=sha256(source))


def checked(record):
    source = project_path(record['logical_path'])
    require(str(source.resolve()) == record['resolved_path'], f'resolved path changed: {source}')
    require(sha256(source) == record['sha256'], f'artifact hash changed: {source}')
    return source


@contextlib.contextmanager
def atomic_target(path, mode=None):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.partial')
    try:
        with partial.open('wb') as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            partial.chmod(mode)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def atomic_json(path, value, mode=None):
    with atomic_target(path, mode) as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def atomic_npz(path, arrays, mode=None):
    with atomic_target(path, mode) as stream:
        np.savez_compressed(stream, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def append_event(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def pid_identity():
    tail = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=os.getpid(), state=tail[0], start_ticks=tail[19], host=socket.gethostname())


class Lease:
    def __init__(self, directory, gpu):
        self.directory, self.gpu = Path(directory), gpu
        self.count = 0

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / 'run.lock').open('a+')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError(f'duplicate evaluation writer rejected: {self.directory}') from None
        self.job = dict(status='running', process=pid_identity(), started_at_utc=now(),
            physical_gpu=self.gpu, exact_command=shlex.join([sys.executable, *sys.argv]),
            recovery_command=shlex.join(['bash', str(PACKAGE / 'scripts/stage3_env.sh'), sys.executable, *sys.argv]),
            tmux_session=os.environ.get('WDRO_TMUX_SESSION'),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        atomic_json(self.directory / 'job.json', self.job)
        append_event(self.directory / 'attempts.jsonl', self.job)
        self.update(0, 'running')
        return self

    def update(self, count, status, **details):
        self.count, self.status = count, status
        atomic_json(self.directory / 'heartbeat.json', dict(updated_at_utc=now(),
            process=pid_identity(), completed_nets=count, status=status,
            physical_gpu=self.gpu, **details))

    def __exit__(self, kind, error, traceback):
        self.job.update(status='failed' if error else 'exited', finished_at_utc=now(),
            completed_nets=self.count, outcome='failed' if error else self.status,
            error=str(error) if error else None, lock_released=True)
        try:
            atomic_json(self.directory / 'job.json', self.job)
            append_event(self.directory / 'attempts.jsonl', self.job)
            if error:
                self.update(self.count, 'failed', error=str(error))
        finally:
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()


def validate_inputs(data, degree):
    needed = {'pins', 'net_ids', 'design_ids', 'source_index', 'scales', 'offsets'}
    require(needed.issubset(data), f'missing normalized fields: {needed - data.keys()}')
    pins = data['pins']
    require(pins.dtype == np.float32, 'pins must be unchanged float32 normalized coordinates')
    require(pins.ndim == 3 and pins.shape[1:] == (degree, 2), 'pins shape/degree mismatch')
    n = len(pins)
    require(n > 0, 'empty degree group')
    require(all(v.ndim > 0 and len(v) == n for v in data.values()), 'all input arrays must be per-net')
    for key in ('net_ids', 'design_ids'):
        require(data[key].shape == (n,) and data[key].dtype.kind in 'SU', f'{key} must be string vector')
        require(bool(np.all(data[key] != data[key].dtype.type(''))), f'empty {key}')
    require(len(np.unique(data['net_ids'])) == n, 'duplicate net identities')
    require(data['source_index'].shape == (n,) and data['source_index'].dtype.kind in 'iu', 'invalid source_index')
    require(bool(np.all(data['source_index'] >= 0)), 'negative source_index')
    require(data['offsets'].shape == (n, 2) and np.isfinite(data['offsets']).all(), 'invalid offsets')
    require(data['scales'].shape == (n,) and np.isfinite(data['scales']).all() and np.all(data['scales'] > 0), 'invalid inverse scales')
    require(np.isfinite(pins).all(), 'nonfinite pins')
    # Float32 endpoints differ slightly from the decimal float64 constants.
    require(pins.min() >= np.float32(0.1) and pins.max() <= np.float32(0.9), 'pins outside frozen normalized bounds')
    scaled = pins.astype(np.float64) * (2**27)
    require(np.array_equal(scaled, np.rint(scaled)), 'pins violate precision_v2 exact integer scaling')
    # Vectorized pairwise checks in bounded chunks; no repair or changed pin order.
    for start in range(0, n, 512):
        xy = pins[start:start + 512].astype(np.float64)
        distance = np.abs(xy[:, :, None, :] - xy[:, None, :, :]).sum(axis=-1)
        distance[:, np.arange(degree), np.arange(degree)] = np.inf
        require(distance.min() >= 1e-6, 'normalized pins violate minimum L1 spacing')
    reserved = {'topology', 'actions', 'rest_lengths', 'regret', 'rest_lengths_inverse', 'lstar_inverse', 'actor_log_probability'}
    require(not (reserved & data.keys()), f'input collides with prediction fields: {reserved & data.keys()}')


def validate_reference(reference, data):
    require({'net_ids', 'lstar', 'verification_lstar'}.issubset(reference), 'incomplete dual-scale reference')
    require(np.array_equal(reference['net_ids'], data['net_ids']), 'reference net identity/order mismatch')
    n = len(data['pins'])
    for key in ('lstar', 'verification_lstar'):
        require(reference[key].shape == (n,), f'{key} shape mismatch')
        require(np.isfinite(reference[key]).all() and np.all(reference[key] > 0), f'invalid {key}')
    require(np.array_equal(reference['lstar'], reference['verification_lstar']), 'reference dual scales disagree')


def python_tree_lengths(pins, topology):
    """Independent ordered REST segment-union length and connectivity audit.

    Every (x,y) action extends the vertical segment at pin x to pin y's Y,
    and the horizontal segment at pin y to pin x's X. Their shared bend is
    implicit. Components of the induced pin graph must form one tree.
    """
    nets, encoded = np.asarray(pins, dtype=np.float64), np.asarray(topology)
    require(nets.ndim == 3 and nets.shape[-1] == 2 and np.isfinite(nets).all(), 'invalid tree pins')
    n, degree, _ = nets.shape
    require(encoded.shape == (n, 2 * (degree - 1)) and encoded.dtype.kind in 'iu', 'invalid topology shape/dtype')
    require(np.all(encoded >= 0) and np.all(encoded < degree), 'topology index out of range')
    result = np.empty(n, dtype=np.float64)
    for i, (xy, actions) in enumerate(zip(nets, encoded.reshape(n, degree - 1, 2), strict=True)):
        xlo, xhi, ylo, yhi = xy[:, 0].copy(), xy[:, 0].copy(), xy[:, 1].copy(), xy[:, 1].copy()
        parent = list(range(degree))
        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for x, y in actions:
            x, y = int(x), int(y)
            require(root(x) != root(y), 'REST topology contains a cycle/self-loop')
            parent[root(x)] = root(y)
            ylo[x], yhi[x] = min(ylo[x], xy[y, 1]), max(yhi[x], xy[y, 1])
            xlo[y], xhi[y] = min(xlo[y], xy[x, 0]), max(xhi[y], xy[x, 0])
        require(len({root(a) for a in range(degree)}) == 1, 'REST tree disconnected')
        result[i] = np.sum(xhi - xlo) + np.sum(yhi - ylo)
    return result


def audit_predictions(arrays, data, reference, start=0, end=None):
    end = len(data['pins']) if end is None else end
    require(all(v.ndim > 0 and len(v) == end - start for v in arrays.values()), 'prediction row counts differ')
    for key, value in data.items():
        if key not in ('lstar', 'verification_lstar'):
            require(key in arrays and np.array_equal(arrays[key], value[start:end]), f'prediction input identity changed: {key}')
    require(np.array_equal(arrays['lstar'], reference['lstar'][start:end]), 'prediction reference changed')
    require(np.array_equal(arrays['verification_lstar'], reference['verification_lstar'][start:end]), 'prediction verification changed')
    require(np.array_equal(arrays['actions'], arrays['topology']), 'action encoding differs from topology')
    require(np.isfinite(arrays['actor_log_probability']).all(), 'nonfinite Actor action log probability')
    independent = python_tree_lengths(data['pins'][start:end], arrays['topology'])
    require(np.array_equal(independent, arrays['rest_lengths']), 'independent REST length mismatch')
    expected = (independent - arrays['lstar']) / arrays['lstar']
    require(np.isfinite(expected).all() and np.all(expected >= -1e-12), 'negative/nonfinite regret: reference or model audit failed')
    require(np.array_equal(expected, arrays['regret']), 'regret formula mismatch')
    require(np.array_equal(arrays['rest_lengths_inverse'], independent / data['scales'][start:end]), 'inverse REST length mismatch')
    require(np.array_equal(arrays['lstar_inverse'], arrays['lstar'] / data['scales'][start:end]), 'inverse reference length mismatch')


def read_shards(state, data, reference):
    chunks, end = [], 0
    for shard in state['shards']:
        require(shard['start'] == end and shard['end'] > end, 'noncontiguous evaluation shards')
        require(shard['end'] <= len(data['pins']), 'shard exceeds input')
        arrays = load_npz(checked(shard['artifact']))
        audit_predictions(arrays, data, reference, end, shard['end'])
        if chunks:
            require(arrays.keys() == chunks[0].keys(), 'shard schemas differ')
        chunks.append(arrays)
        end = shard['end']
    require(end == state['complete_nets'], 'cursor does not match committed shards')
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]} if chunks else {}


def load_model(index, method, degree, seed):
    frozen = read_json(FROZEN)
    require(index == FROZEN, 'model index must be this extension frozen index')
    require(frozen['status'] == 'frozen' and frozen['selection'] == 'synthetic_validation_only', 'invalid selection')
    require(not frozen['test_used_for_selection'], 'external test selection prohibited')
    chosen = [m for m in frozen['models'] if (m['method'], m['degree'], m['seed']) == (method, degree, seed)]
    require(len(chosen) == 1, 'model identity must resolve exactly once')
    model = chosen[0]
    checked(model['best'])
    if model['kind'] == 'v31_joint':
        receipt = read_json(checked(model['training_completion']))
        require(receipt['status'] == 'pass' and not receipt['smoke'], 'training not accepted')
        require((receipt['method'], receipt['degree'], receipt['seed'], receipt['best_update']) ==
                (method, degree, seed, model['selected_update']), 'joint checkpoint identity mismatch')
    else:
        parent = read_json(PARENT_FROZEN)
        require(any(m['best'] == model['best'] and (m['method'],m['degree'],m['seed']) ==
                    (method,degree,seed) for m in parent['models']), 'parent checkpoint differs')
    return model


def source_binding():
    legacy = REVISION.parent / 'DRO/WDRO-REST'
    sources = [Path(__file__), EXTENSION / 'protocol.json', PACKAGE / 'src/rest.py', PACKAGE / 'src/geosteiner.py',
        PACKAGE / 'src/trainer.py', PACKAGE / 'src/metrics.py',
        PACKAGE / 'configs/stage2_trainer_v1.json', legacy / 'algorithms/libeval.so']
    sources += sorted((legacy / 'models').rglob('*.py'))
    return [artifact(path) for path in sources]


def stop_requested(*unused):
    global STOP
    STOP = True


def evaluate(args):
    global STOP
    STOP = False
    input_path, reference_path = project_path(args.input), project_path(args.reference or args.input)
    index, output = project_path(args.model_index), project_path(args.output)
    require(output.is_relative_to(EXTENSION), 'new outputs must stay under this external benchmark extension')
    require(args.batch_size == 256, 'fixed inference batch size is 256')
    require(args.gpu >= 0, 'physical GPU index must be nonnegative')
    require(args.max_new_batches is None or args.max_new_batches > 0, 'max-new-batches must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    sys.path.insert(0, str(PACKAGE))
    import torch
    from src import geosteiner as gst
    from src.metrics import summarize_regret
    from src.rest import build_actor, RestTreeEvaluator
    from src.trainer import configure_determinism, load_trainer_config
    model = load_model(index, args.method, args.degree, args.seed)
    data, reference = load_npz(input_path), load_npz(reference_path)
    validate_inputs(data, args.degree)
    validate_reference(reference, data)
    n = len(data['pins'])
    reference_manifest = project_path(args.reference_manifest) if args.reference_manifest else reference_path.with_suffix('.json')
    provenance = None
    if reference_manifest.exists():
        receipt = read_json(reference_manifest)
        require(receipt['status'] == 'complete' and receipt['completed_nets'] == n, 'incomplete reference receipt')
        require(checked(receipt['input']) == input_path and checked(receipt['reference']) == reference_path, 'reference receipt file binding differs')
        require(receipt['reference_identity'] == gst.reference_identity(), 'reference solver identity mismatch')
        require(receipt['scales'] == [2**27, 2**28], 'reference exact scales mismatch')
        provenance = artifact(reference_manifest)
    elif args.reference_manifest:
        raise FileNotFoundError(reference_manifest)
    hardware = dict(name=torch.cuda.get_device_name(0), torch_version=torch.__version__, cuda_version=torch.version.cuda)
    binding = dict(schema_version=1, role='all_real_v31_joint_external_test', revision='adversarial_v3_1_20260912',
        model=model, canonical_frozen_manifest=artifact(FROZEN), model_index=artifact(index),
        input=artifact(input_path), reference=artifact(reference_path), reference_manifest=provenance,
        reference_identity=gst.reference_identity() if provenance else None,
        count=n, degree=args.degree, inference=dict(batch_size=256, T=1, mode='deterministic_greedy', tail_padding='repeat_last_requested_point_then_discard', hardware=hardware),
        sources=source_binding(), train_or_retune=False, model_selection=False)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    with Lease(output, args.gpu) as lease:
        cursor, result_path = output / 'cursor.json', output / 'result.json'
        state = read_json(cursor) if cursor.exists() else dict(binding=binding, complete_nets=0, shards=[])
        require(state['binding'] == binding, 'evaluation binding changed; refusing resume')
        arrays = read_shards(state, data, reference)
        if result_path.exists():
            result = read_json(result_path)
            require(result['status'] == 'complete' and result['binding'] == binding and state['complete_nets'] == n, 'completed result binding/count changed')
            published = load_npz(checked(result['predictions']))
            audit_predictions(published, data, reference)
            require(arrays.keys() == published.keys() and all(np.array_equal(arrays[k], published[k]) for k in arrays), 'published output differs from committed shards')
            require(result['completed_nets'] == n and result['summary'] == summarize_regret(published['regret']), 'completed result summary/count changed')
            require(result['call_ledger'] == dict(actor_forward_nets=n, actor_forward_rows=((n+255)//256)*256, actor_padding_rows=((n+255)//256)*256-n, exact_rest_evaluator_nets=n,
                geosteiner_nets=0, independent_python_tree_nets=n), 'completed result call ledger changed')
            lease.update(n, 'complete', idempotent=True)
            return result
        configure_determinism(load_trainer_config())
        checkpoint = torch.load(checked(model['best']), map_location='cpu', weights_only=False)
        require(int(checkpoint['update']) == model['selected_update'], 'selected update mismatch')
        if model['kind'] == 'precision_v2_control':
            identity = checkpoint['run_identity']
            require((identity['method'], identity['degree'], identity['training_seed']) == (args.method, args.degree, args.seed), 'checkpoint run identity mismatch')
            require(identity['reference_solver'] == gst.reference_identity(), 'checkpoint reference mismatch')
        actor = build_actor(args.degree, torch.device('cuda:0'))
        actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
        actor.eval()
        rest = RestTreeEvaluator()
        torch.cuda.reset_peak_memory_stats()
        added, invocation_start = 0, time.monotonic()
        for start in range(state['complete_nets'], n, 256):
            end = min(start + 256, n)
            attempt = f'{os.getpid()}_{time.time_ns()}'
            append_event(output / 'calls.jsonl', dict(id=attempt, phase='started', at_utc=now(), start=start, end=end, physical_gpu=args.gpu))
            pins = data['pins'][start:end]
            torch.cuda.synchronize()
            began = time.monotonic()
            with torch.no_grad():
                forward_pins = pins if len(pins) == 256 else np.concatenate((pins, np.repeat(pins[-1:], 256-len(pins), axis=0)))
                encoded, log_probability = actor(forward_pins, True)
                encoded, log_probability = encoded[:len(pins)], log_probability[:len(pins)]
            topology = encoded.detach().cpu().numpy()
            log_probability = log_probability.detach().cpu().numpy()
            torch.cuda.synchronize()
            actor_seconds = time.monotonic() - began
            began = time.monotonic()
            lengths = rest.lengths(pins, topology)
            rest_seconds = time.monotonic() - began
            lstar = reference['lstar'][start:end]
            batch = {key: value[start:end] for key, value in data.items()}
            batch.update(topology=topology, actions=topology, actor_log_probability=log_probability,
                rest_lengths=lengths, lstar=lstar, verification_lstar=reference['verification_lstar'][start:end],
                regret=(lengths-lstar)/lstar, rest_lengths_inverse=lengths/data['scales'][start:end],
                lstar_inverse=lstar/data['scales'][start:end])
            began = time.monotonic()
            audit_predictions(batch, data, reference, start, end)
            audit_seconds = time.monotonic() - began
            path = output / 'shards' / f'{start:09d}_{end:09d}.npz'
            atomic_npz(path, batch, mode=0o444)
            event = dict(id=attempt, phase='completed', at_utc=now(), start=start, end=end,
                actor_calls=end-start, actor_forward_rows=256, actor_padding_rows=256-(end-start), rest_calls=end-start, python_tree_audit_calls=end-start,
                actor_seconds=actor_seconds, rest_seconds=rest_seconds, audit_seconds=audit_seconds, physical_gpu=args.gpu)
            append_event(output / 'calls.jsonl', event)
            state['shards'].append(dict(start=start, end=end, artifact=artifact(path), attempt=attempt,
                actor_seconds=actor_seconds, rest_seconds=rest_seconds, audit_seconds=audit_seconds, physical_gpu=args.gpu))
            state.update(complete_nets=end, updated_at_utc=now(),
                peak_reserved_mib=max(state.get('peak_reserved_mib', 0), torch.cuda.max_memory_reserved()/2**20),
                peak_allocated_mib=max(state.get('peak_allocated_mib', 0), torch.cuda.max_memory_allocated()/2**20))
            atomic_json(cursor, state)
            lease.update(end, 'evaluating', peak_reserved_mib=state['peak_reserved_mib'])
            added += 1
            if end < n and (STOP or (args.max_new_batches is not None and added >= args.max_new_batches)):
                lease.update(end, 'paused')
                return dict(status='paused', completed_nets=end, cursor=artifact(cursor))
        arrays = read_shards(state, data, reference)
        require(state['complete_nets'] == n, 'evaluation incomplete')
        path = output / 'predictions.npz'
        atomic_npz(path, arrays, mode=0o444)
        # Validate the serialized artifact, rather than only its in-memory source.
        audit_predictions(load_npz(path), data, reference)
        result = dict(status='complete', at_utc=now(), binding=binding, completed_nets=n,
            selected_update=int(checkpoint['update']), predictions=artifact(path), summary=summarize_regret(arrays['regret']),
            peak_reserved_mib=state['peak_reserved_mib'], peak_allocated_mib=state['peak_allocated_mib'],
            actor_seconds=sum(s['actor_seconds'] for s in state['shards']),
            rest_seconds=sum(s['rest_seconds'] for s in state['shards']),
            python_tree_audit_seconds=sum(s['audit_seconds'] for s in state['shards']),
            last_invocation_seconds=time.monotonic()-invocation_start,
            call_ledger=dict(actor_forward_nets=n, actor_forward_rows=((n+255)//256)*256, actor_padding_rows=((n+255)//256)*256-n, exact_rest_evaluator_nets=n, geosteiner_nets=0,
                independent_python_tree_nets=n),
            checks=dict(all_tree_lengths_independently_equal=True, all_trees_connected=True,
                dual_scale_references_equal=True, references_provenance_verified=provenance is not None,
                all_regrets_ge_minus_1e_12=True, all_input_rows_preserved=True),
            topology_semantics='ordered REST (x-index,y-index) action pairs; actions aliases topology',
            accounting='committed batches; interrupted/replayed attempts retained in calls.jsonl', model_selection=False)
        atomic_json(result_path, result, mode=0o444)
        lease.update(n, 'complete')
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--reference', help='NPZ with aligned net_ids/lstar/verification_lstar; default: input NPZ')
    parser.add_argument('--reference-manifest')
    parser.add_argument('--model-index', default=str(FROZEN))
    parser.add_argument('--method', required=True, choices=['erm', 'mixed_erm', 'hard_mining', 'group_dro', 'wdro_rest', 'random_top4', 'critic_calibrated4', 'rl_frozen4', 'rl_joint4', 'random64', 'es64'])
    parser.add_argument('--degree', required=True, type=int, choices=[30])
    parser.add_argument('--seed', required=True, type=int, choices=[101, 102, 103])
    parser.add_argument('--gpu', required=True, type=int, help='physical GPU index; mapped to local cuda:0 before importing torch')
    parser.add_argument('--output', required=True, help='evaluation directory under the extension root')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-new-batches', type=int)
    args = parser.parse_args(argv)
    result = evaluate(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

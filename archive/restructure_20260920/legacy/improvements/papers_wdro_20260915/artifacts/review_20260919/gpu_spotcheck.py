"""Read-only checkpoint forward audit. No training, retuning, or source edits."""
from pathlib import Path
import gc
import json
import os
import subprocess
import time
import traceback

import numpy as np
import torch
from core import ROOT, atomic_json, make_adapter, read_protocol, seed_all, set_active_device, sha

OUT = Path(__file__).resolve().parent / 'gpu_spotcheck'
OUT.mkdir(parents=True, exist_ok=True)
receipt = {'status': 'running', 'started_at_epoch': time.time(), 'pid': os.getpid(),
           'scope': 'First original batch (64) in each test distribution and flattened attack support; all 24 selected checkpoints; exact comparison of costs and every test metric.',
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'script_sha256': sha(__file__), 'preflight': [], 'runs': []}
try:
    # Three fresh admission samples; no attempt to claim already-used capacity.
    for i in range(3):
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        receipt['preflight'].append({'time': time.time(), 'gpus': raw})
        own = [line.split(',') for line in raw.splitlines() if line.split(',')[0].strip() == os.environ['CUDA_VISIBLE_DEVICES']][0]
        if int(own[2]) < 8192 or int(own[3]) > 15:
            raise RuntimeError('Selected GPU no longer meets idle admission threshold')
        if i < 2:
            time.sleep(2)
    device = torch.device('cuda:0')
    set_active_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(0.10, device)
    p = read_protocol()
    receipt['protocol_sha256'] = sha(ROOT / 'protocol.json')
    receipt['frozen_sha256'] = sha(ROOT / 'artifacts/frozen.json')
    receipt['device_name'] = torch.cuda.get_device_name(device)
    receipt['torch_version'] = torch.__version__
    inputs = {}
    for dist in p['distributions']:
        with np.load(ROOT / f'artifacts/data/test_{dist}.npz', allow_pickle=False) as z:
            inputs[dist, 'test'] = z['pins'][:64].copy()
        with np.load(ROOT / f'artifacts/data/attack_{dist}.npz', allow_pickle=False) as z:
            inputs[dist, 'attack'] = z['candidates'].reshape(-1, p['degree'], 2)[:64].copy()
    for family in p['families']:
        for method in p['methods']:
            for seed in p['seeds']:
                seed_all(seed)
                run_id = f'{family}_{method}_s{seed}'
                folder = ROOT / 'artifacts/runs' / run_id
                state = torch.load(folder / 'best.pt', map_location=device, weights_only=False)
                params = dict(p['parameters'][family], cache_path=str(OUT / 'references.sqlite'))
                adapter = make_adapter(family, p['degree'], device, params)
                adapter.load_state_dict(state['adapter'], strict=True)
                adapter.eval()
                item = {'run_id': run_id, 'best_update': state['best']['update'],
                        'input_sha256': {name: sha(folder / name) for name in ['best.pt', 'test_best.npz', 'attack.npz']},
                        'comparisons': []}
                saved = np.load(folder / 'test_best.npz', allow_pickle=False)
                attack = np.load(folder / 'attack.npz', allow_pickle=False)
                actual = {}
                with torch.no_grad():
                    for dist in p['distributions']:
                        cost, _, metrics = adapter.evaluate(inputs[dist, 'test'], True)
                        values = {'cost': cost.cpu().numpy(), **{k: np.asarray(v, dtype=np.float64) for k, v in metrics.items()}}
                        expected_names = {k.split('__', 1)[1] for k in saved.files if k.startswith(dist + '__')}
                        if set(values) != expected_names:
                            raise ValueError(f'{run_id}/{dist}: metric names mismatch')
                        for name, value in values.items():
                            expected = saved[dist + '__' + name][:64]
                            actual[dist + '__' + name] = value
                            item['comparisons'].append({'kind': 'test', 'distribution': dist, 'metric': name,
                                'n': len(value), 'exact_equal': bool(np.array_equal(value, expected)),
                                'max_abs_difference': float(np.max(np.abs(value.astype(float) - expected.astype(float))))})
                        loss, _, _ = adapter.evaluate(inputs[dist, 'attack'], True)
                        value = loss.cpu().numpy()
                        expected = attack[dist + '__loss'].reshape(-1)[:64]
                        actual[dist + '__attack_loss'] = value
                        item['comparisons'].append({'kind': 'attack', 'distribution': dist, 'metric': 'loss',
                            'n': len(value), 'exact_equal': bool(np.array_equal(value, expected)),
                            'max_abs_difference': float(np.max(np.abs(value.astype(float) - expected.astype(float))))})
                saved.close()
                attack.close()
                if hasattr(adapter, 'close'):
                    adapter.close()
                np.savez_compressed(OUT / f'{run_id}.npz', **actual)
                item['output_sha256'] = sha(OUT / f'{run_id}.npz')
                item['status'] = 'PASS' if all(x['exact_equal'] for x in item['comparisons']) else 'MISMATCH'
                receipt['runs'].append(item)
                atomic_json(OUT / 'receipt.json', receipt)
                print(run_id, item['status'], flush=True)
                del adapter, state, cost, loss
                gc.collect()
                torch.cuda.empty_cache()
    receipt['max_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    receipt['max_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    receipt['forward_predictions'] = 24 * 4 * 64 * 2
    receipt['status'] = 'PASS' if all(x['status'] == 'PASS' for x in receipt['runs']) else 'MISMATCH'
except Exception:
    receipt['status'] = 'ERROR'
    receipt['traceback'] = traceback.format_exc()
    raise
finally:
    receipt['finished_at_epoch'] = time.time()
    receipt['elapsed_seconds'] = receipt['finished_at_epoch'] - receipt['started_at_epoch']
    atomic_json(OUT / 'receipt.json', receipt)

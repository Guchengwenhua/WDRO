"""2048-update continuation and a separate calibrated-Critic paired cohort."""
import argparse
import copy
import fcntl
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import upstream_runner as base
import repair_data

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'critic_paper_bank_20261011'

class SetCritic(nn.Module):
    """Unrooted set model; output > -1 with no zero-gradient output ReLU."""
    def __init__(self, degree, device):
        super().__init__()
        self.device = torch.device(device)
        self.embed = nn.Sequential(nn.Linear(2, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, math.log(math.expm1(1.01)))
        self.to(device)

    def forward(self, pins):
        x = torch.as_tensor(pins, dtype=torch.float32, device=self.device)
        h = self.embed(x)
        raw = self.head(torch.cat([h.mean(1), h.max(1).values], dim=-1)).squeeze(-1)
        return torch.nn.functional.softplus(raw) - 1.0

def bootstrap_main(family, arm, seed):
    run = f'{family}_{arm}_s{seed}'
    out = HERE / 'artifacts/main' / run
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'bootstrap.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'latest.pt').exists():
        lock.close(); return
    src = OLD / 'artifacts' / run
    done = base.read(src / 'complete.json')
    if done['status'] != 'complete': raise ValueError('parent run is unfinished')
    state = torch.load(src / 'latest.pt', map_location='cpu', weights_only=False)
    original_config = copy.deepcopy(state['config'])
    state['config']['protocol'] = base.read(HERE / 'main/protocol.json')
    for name in ['train.jsonl', 'validation.jsonl']:
        shutil.copyfile(src / name, out / name)
    base.save_pt(out / 'best.pt', state['best_weights'])
    base.atomic_json(out / 'resume_origin.json', {'source': str(src), 'step': state['step'], 'training_seconds': state['train_seconds'],
                                                'ledger': state['ledger'], 'original_config': original_config,
                                                'continuation': 'actor critic optimizer RNG lambda best and update preserved; budget extended only', 'time': time.time()})
    # latest is the commit point; an interrupted bootstrap can be repeated.
    base.save_pt(out / 'latest.pt', state)
    lock.close()

class ScaleExperiment(base.Experiment):
    def __init__(self, family, arm, seed):
        self.segment_started = time.monotonic()
        self.segment_step = None
        self.segment_training_seconds = None
        super().__init__(family, arm, seed)
        self.segment_step = self.step
        self.segment_training_seconds = self.train_seconds

    def heartbeat(self, phase):
        super().heartbeat(phase)
        p = self.out / 'heartbeat.json'; row = base.read(p)
        row['segment_start_step'] = self.segment_step
        row['segment_wall_seconds'] = time.monotonic() - self.segment_started
        if self.segment_step is not None and self.step > self.segment_step:
            rate = (self.train_seconds - self.segment_training_seconds) / (self.step - self.segment_step)
            row['segment_seconds_per_update'] = rate
            row['remaining_training_seconds'] = rate * (self.p['updates'][self.family] - self.step)
        base.atomic_json(p, row)

    def validation(self):
        super().validation()
        if self.step in [512, 1024, 2048]:
            base.save_pt(self.out / f'milestone_u{self.step:04d}.pt', self.state())

def calibration(seed):
    out = HERE / f'artifacts/calibration/s{seed}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'complete.json').exists(): return 0
    p = base.read(HERE / 'repaired/protocol.json'); cfg = p['calibration']
    base.atomic_json(out / 'config.json', {'seed': seed, 'protocol': p})
    device = torch.device('cuda:0'); base.set_active_device(device)
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True); base.seed_all(seed)
    actor = base.build_actor(10, device)
    parent_path = OLD / f'artifacts/rest_geo_s{seed}/latest.pt'
    parent = torch.load(parent_path, map_location=device, weights_only=False)
    actor.load_state_dict(parent['actor']); actor.eval()
    critic = SetCritic(10, device)
    opt = torch.optim.Adam(critic.parameters(), lr=cfg['learning_rate'])
    rest = base.RestTreeEvaluator(base.LIB); geo = base.GeoSteinerEvaluator(base.LIB)
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    started = time.monotonic(); all_data = {}; label_seconds = 0.
    def heartbeat(phase, step, **more):
        base.atomic_json(out / 'heartbeat.json', {'phase': phase, 'step': step, 'steps': cfg['updates'], 'time': time.time(),
                         'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'identity': base.proc_identity(), 'segment_seconds': time.monotonic()-started, **more})
    for split, count, offset in [('train', cfg['train_nets'], 1000000), ('holdout', cfg['holdout_nets'], 2000000)]:
        xs, ys = [], []
        for start in range(0, count, 128):
            part_path = out / f'{split}_labels_{start:04d}.pt'
            heartbeat('oracle_labeling', 0, split=split, label_start=start, label_count=count)
            if part_path.exists():
                part = torch.load(part_path, map_location='cpu', weights_only=False)
                base.restore_rng(part['rng'])
            else:
                tick = time.monotonic()
                x = repair_data.materialize_training_batch(protocol=p, degree=10, training_seed=seed, update=offset+start//128, batch_size=128).pins
                reference = geo.lengths(x); sampled = []
                with torch.no_grad():
                    for _ in range(cfg['samples_per_net']):
                        topology, _ = actor(x, False)
                        sampled.append(rest.lengths(x, topology.cpu().numpy()))
                target = np.mean(sampled, axis=0) / reference - 1
                part = {'pins': x, 'reference': reference, 'sampled_lengths': np.stack(sampled), 'target': target,
                        'rng': base.rng_state(), 'seconds': time.monotonic()-tick}
                base.save_pt(part_path, part)
            xs.append(part['pins']); ys.append(part['target']); label_seconds += part['seconds']
            if stopped[0]: heartbeat('paused_after_label_chunk', 0); return 75
        all_data[split] = (np.concatenate(xs), np.concatenate(ys))
    current = 0; fit_seconds = 0.
    if (out / 'latest.pt').exists():
        state = torch.load(out / 'latest.pt', map_location=device, weights_only=False)
        critic.load_state_dict(state['critic']); opt.load_state_dict(state['optimizer']); base.restore_rng(state['rng'])
        current = state['step']; fit_seconds = state['fit_seconds']
        if (out / 'train.jsonl').exists():
            rows = []
            for line in (out / 'train.jsonl').read_text().splitlines():
                try: row = json.loads(line)
                except json.JSONDecodeError: continue
                if row['step'] <= current: rows.append(row)
            (out / 'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    x, y = all_data['train']
    for step in range(current+1, cfg['updates']+1):
        tick = time.monotonic(); ids = np.random.randint(len(x), size=cfg['batch_size'])
        critic.train(); pred = critic(x[ids]); target = torch.as_tensor(y[ids], dtype=torch.float32, device=device)
        loss = (pred-target).square().mean(); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True); opt.step()
        fit_seconds += time.monotonic()-tick
        base.append(out / 'train.jsonl', {'step': step, 'mse': float(loss.detach()), 'prediction_mean': float(pred.mean().detach())})
        if step % cfg['checkpoint_every'] == 0 or step == cfg['updates'] or stopped[0]:
            base.save_pt(out / 'latest.pt', {'critic': critic.state_dict(), 'optimizer': opt.state_dict(), 'rng': base.rng_state(), 'step': step, 'fit_seconds': fit_seconds})
            heartbeat('fitting', step)
        if stopped[0]: return 75
    critic.eval(); hx, hy = all_data['holdout']
    with torch.no_grad(): prediction = np.concatenate([critic(hx[i:i+128]).cpu().numpy() for i in range(0,len(hx),128)])
    mse = float(np.mean((prediction-hy)**2)); zero_mse = float(np.mean(hy**2)); std = float(prediction.std())
    finite = bool(np.isfinite(prediction).all() and np.all(1+prediction > 0))
    qualified = bool(finite and mse < zero_mse and std > 1e-5)
    corr = float(np.corrcoef(prediction, hy)[0,1]) if std > 0 and hy.std() > 0 else None
    result = {'seed': seed, 'qualified': qualified, 'gate': cfg['gate'], 'heldout_mse': mse, 'zero_predictor_mse': zero_mse,
              'training_mean_predictor_mse': float(np.mean((y.mean()-hy)**2)), 'prediction_mean': float(prediction.mean()), 'prediction_std': std,
              'prediction_min': float(prediction.min()), 'prediction_max': float(prediction.max()), 'holdout_target_mean': float(hy.mean()), 'correlation': corr,
              'label_oracle_nets': cfg['train_nets']+cfg['holdout_nets'], 'sampled_actor_nets': (cfg['train_nets']+cfg['holdout_nets'])*cfg['samples_per_net'],
              'label_seconds': label_seconds, 'fit_seconds': fit_seconds, 'parent_actor': str(parent_path),
              'scope': 'initialization gate only, not a certificate of generalization or useful adversarial ranking'}
    base.atomic_npz(out / 'holdout_predictions.npz', pins=hx, target=hy, prediction=prediction)
    base.atomic_json(out / 'results.json', result)
    if qualified:
        base.save_pt(out / 'parent.pt', {'actor_state_dict': actor.state_dict(), 'critic_state_dict': critic.state_dict(), 'calibration': result})
    base.atomic_json(out / 'complete.json', {'status': 'complete', 'qualified': qualified, 'time': time.time()})
    heartbeat('complete' if qualified else 'gated', cfg['updates'])
    print(result, flush=True)
    return 0

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--cohort', required=True, choices=['main','calibration','repaired'])
    parser.add_argument('--family'); parser.add_argument('--arm'); parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    if args.cohort == 'calibration': return calibration(args.seed)
    base.HERE = HERE / args.cohort
    if args.cohort == 'main': bootstrap_main(args.family, args.arm, args.seed)
    else:
        marker = base.read(HERE / f'artifacts/calibration/s{args.seed}/complete.json')
        if not marker.get('qualified'): raise RuntimeError('Calibrated critic did not meet its frozen initialization gate')
        base.build_critic = SetCritic
        base.materialize_training_batch = repair_data.materialize_training_batch
        base.generate_support = repair_data.generate_support
    experiment = ScaleExperiment(args.family, args.arm, args.seed)
    try:
        return experiment.run()
    except torch.cuda.OutOfMemoryError as error:
        base.atomic_json(experiment.out / f'capacity_oom_{time.time_ns()}.json', {'status': 'capacity_pause', 'message': str(error), 'resume': 'last atomic latest.pt'})
        return 75
    finally: experiment.lock.close()

if __name__ == '__main__': sys.exit(main())

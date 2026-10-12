"""Small paired scientific experiment; historical sources are imported read-only."""
import argparse, copy, fcntl, json, os, signal, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPERS = ROOT / 'papers_wdro_20260915'
BASE = ROOT / 'wdro_precision_v2'
sys.path.insert(0, str(PAPERS))
sys.path.insert(0, str(BASE / 'wdro_rest_min'))
from core import atomic_json, atomic_npz, append, Baseline, make_adapter, costs, dataset, set_active_device, seed_all, rng_state, restore_rng, proc_identity
import src.rest as restmod
restmod.LEGACY_ROOT = Path('/remote-home/Guilin/DRO/DRO/WDRO-REST')
from src.rest import build_actor, build_critic, RestTreeEvaluator
from src.geosteiner import GeoSteinerEvaluator
from src.datasets import load_dataset
from src.candidates import shortlist_support
from research_data import materialize_training_batch, generate_support
LIB = restmod.LEGACY_ROOT / 'algorithms/libeval.so'

def read(p): return json.loads(Path(p).read_text())
def save_pt(p, obj):
    p = Path(p); tmp = p.with_name(p.name + '.partial')
    with tmp.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)

def prepare_inputs():
    """Select only on pre-existing validation records. Never open test metrics."""
    result = {'parents': {}, 'banks': {}, 'selection': 'historical validation only'}
    old = read(ROOT / 'checkpoint_bank_20260930/inputs.json')
    result['banks']['rest'] = old['eligible'][:3]
    for family in ['rest', 'timing', 'buffer']:
        result['parents'][family] = {}
        for seed in [101, 102, 103]:
            if family == 'rest':
                path = BASE / f'experiments/checkpoints/stage2/formal_erm_d10_s{seed}_u2000/best.pt'
            else:
                path = PAPERS / f'artifacts/runs/{family}_erm_s{seed}/latest.pt'
                done = read(path.parent / 'complete.json')
                if done['status'] != 'complete': raise ValueError('unfinished parent')
            result['parents'][family][str(seed)] = {'path': str(path)}
        if family != 'rest':
            rows = []
            for method in ['erm', 'wdro']:
                for seed in [101, 102, 103]:
                    path = PAPERS / f'artifacts/runs/{family}_{method}_s{seed}/best.pt'
                    done = read(path.parent / 'complete.json')
                    if done['status'] != 'complete': raise ValueError('unfinished bank')
                    ck = torch.load(path, map_location='cpu', weights_only=False)
                    rows.append({'path': str(path), 'run': path.parent.name, 'score': ck['best']['objective'], 'update': ck['best']['update']})
            result['banks'][family] = sorted(rows, key=lambda r: (r['score'], r['run']))[:3]
    for collection in [*result['banks'].values(), *[list(x.values()) for x in result['parents'].values()]]:
        for row in collection:
            st = Path(row['path']).stat(); row.update(size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    atomic_json(HERE / 'inputs.json', result)

class Experiment:
    def __init__(self, family, arm, seed):
        self.p = read(HERE / 'protocol.json'); self.inputs = read(HERE / 'inputs.json')
        self.family, self.arm, self.seed = family, arm, seed
        self.out = HERE / 'artifacts' / f'{family}_{arm}_s{seed}'
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / 'run.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.device = torch.device('cuda:0'); set_active_device(self.device)
        torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.20)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        seed_all(seed)
        self.actor = self.model(); self.critic = build_critic(10, self.device) if family == 'rest' else Baseline(self.device)
        parent = self.inputs['parents'][family][str(seed)]
        ck = torch.load(parent['path'], map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck['actor_state_dict'] if family == 'rest' else ck['adapter'])
        self.critic.load_state_dict(ck['critic_state_dict'] if family == 'rest' else ck['critic'])
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.p['learning_rate'])
        self.rest = RestTreeEvaluator(LIB) if family == 'rest' else None
        self.geo = GeoSteinerEvaluator(LIB) if family == 'rest' and arm == 'geo' else None
        self.rho = (read(BASE / 'experiments/cache/stage6/calibration_d10_s101.json')['hungarian_l1_mean']['q75'] if family == 'rest'
                    else read(PAPERS / 'artifacts/data/calibration.json')['rooted'])
        self.teachers = []
        saved = rng_state()
        if arm == 'k3':
            for row in self.inputs['banks'][family]:
                m = self.model(); t = torch.load(row['path'], map_location=self.device, weights_only=False)
                m.load_state_dict(t['actor_state_dict'] if family == 'rest' else t['adapter']); m.eval()
                for parameter in m.parameters(): parameter.requires_grad_(False)
                self.teachers.append(m)
        restore_rng(saved)
        self.step = 0; self.price = 0.; self.best = {'score': 1e100, 'step': -1}
        self.train_seconds = 0.; self.ledger = {'oracle_train_nets': 0, 'teacher_train_nets': 0, 'critic_reference_nets': 0}
        self.config = {'family': family, 'arm': arm, 'seed': seed, 'protocol': self.p, 'parent': parent, 'bank': self.inputs['banks'][family], 'rho': self.rho}
        if (self.out / 'config.json').exists() and read(self.out / 'config.json') != self.config: raise ValueError('config drift')
        atomic_json(self.out / 'config.json', self.config)
        self.stopped = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stopped', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stopped', True))
        if (self.out / 'latest.pt').exists():
            c = torch.load(self.out / 'latest.pt', map_location=self.device, weights_only=False)
            if c['config'] != self.config: raise ValueError('checkpoint config drift')
            self.actor.load_state_dict(c['actor']); self.critic.load_state_dict(c['critic']); self.opt.load_state_dict(c['optimizer'])
            for key in ['step', 'price', 'best', 'train_seconds', 'ledger']: setattr(self, key, c[key])
            restore_rng(c['rng'])
            # Logs beyond the checkpoint belong to an interrupted segment.
            for name in ['train.jsonl', 'validation.jsonl']:
                path = self.out / name
                if path.exists():
                    rows = []
                    for line in path.read_text().splitlines():
                        try: row = json.loads(line)
                        except json.JSONDecodeError: continue
                        if row['step'] <= self.step: rows.append(row)
                    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            save_pt(self.out / 'best.pt', c['best_weights'])
        self.heartbeat('initialized')

    def model(self):
        return build_actor(10, self.device) if self.family == 'rest' else make_adapter(self.family, 10, self.device, self.p['parameters'][self.family])

    def heartbeat(self, phase):
        atomic_json(self.out / 'heartbeat.json', {'phase': phase, 'step': self.step, 'steps': self.p['updates'][self.family], 'time': time.time(), 'identity': proc_identity(), 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'lambda': self.price,
                    'training_seconds': self.train_seconds, 'seconds_per_update': self.train_seconds / self.step if self.step else None,
                    'remaining_training_seconds': self.train_seconds / self.step * (self.p['updates'][self.family] - self.step) if self.step else None})

    def state(self):
        return {'config': self.config, 'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'optimizer': self.opt.state_dict(), 'rng': rng_state(),
                'step': self.step, 'price': self.price, 'best': self.best, 'train_seconds': self.train_seconds, 'ledger': self.ledger,
                'best_weights': torch.load(self.out / 'best.pt', map_location='cpu', weights_only=False)}

    def score(self, model, x, greedy):
        if self.family == 'rest':
            topology, lp = model(x, greedy)
            value = self.rest.lengths(x, topology.detach().cpu().numpy())
            return value, lp, {'length': value}
        value, lp, metrics = model.evaluate(x, greedy)
        return value.detach().cpu().numpy().astype(np.float64), lp, metrics

    def greedy(self, model, x):
        model.eval(); values = []; metrics = {}; batch = self.p['greedy_microbatch']
        with torch.no_grad():
            for i in range(0, len(x), batch):
                part = x[i:i+batch]; n = len(part)
                if n < batch: part = np.concatenate([part, np.repeat(part[-1:], batch-n, axis=0)])
                value, _, more = self.score(model, part, True); values.append(value[:n])
                for k, v in more.items(): metrics.setdefault(k, []).append(np.asarray(v)[:n])
        return np.concatenate(values), {k: np.concatenate(v) for k, v in metrics.items()}

    def prediction(self, x):
        self.critic.eval(); batch = self.p['greedy_microbatch']; parts = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                chunk = x[i:i+batch]; n = len(chunk)
                if n < batch: chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch-n, axis=0)])
                parts.append(self.critic(chunk).cpu().numpy()[:n])
        return np.concatenate(parts).astype(np.float64)

    def eval_data(self, split, group):
        count = self.p[split + '_count']
        if self.family == 'rest':
            n, seed = (2000, 202) if split == 'validation' else (10000, 303)
            a, meta = load_dataset(BASE / f'experiments/cache/datasets/protocol_v1/{split}/{group}_d10_s{seed}_n{n}.npz', verify=False)
            b, _ = load_dataset(BASE / 'experiments/cache/lstar/protocol_v1' / (meta['array_bundle_sha256'] + '.npz'), verify=False)
            if not np.array_equal(a['net_ids'], b['net_ids']): raise ValueError('reference IDs')
            return a['pins'][:count], b['lstar'][:count], a['net_ids'][:count]
        batch = dataset(self.p, split, group, 261011, count)
        return batch.pins, None, batch.net_ids

    def evaluate(self, split, label):
        arrays = {}; summary = {}
        for group in self.p['distributions']:
            x, truth, ids = self.eval_data(split, group)
            raw, metrics = self.greedy(self.actor, x)
            value = raw / truth - 1 if self.family == 'rest' else raw
            metrics = {k: np.asarray(v, dtype=np.float64) for k, v in {'cost': value, **metrics}.items()}
            if self.family == 'rest':
                pred = self.prediction(x); denom = 1 + pred
                if (denom <= 0).any() or not np.isfinite(denom).all(): raise ValueError('invalid critic conversion')
                predicted_ref = raw / denom
                metrics.update(critic=pred, reference=truth, predicted_reference=predicted_ref,
                               predicted_reference_relative_error=predicted_ref / truth - 1,
                               predicted_current_length=truth * denom,
                               predicted_current_length_relative_error=truth * denom / raw - 1,
                               hpwl_violation=(predicted_ref < np.ptp(x, axis=1).sum(1)).astype(float))
            arrays[group + '__ids'] = ids
            for key, v in metrics.items(): arrays[group + '__' + key] = np.asarray(v)
            summary[group] = {k: {'mean': float(np.mean(v)), 'p95': float(np.quantile(v, .95)), 'mae': float(np.mean(np.abs(v)))} for k, v in metrics.items()}
        if label: atomic_npz(self.out / (label + '.npz'), **arrays)
        return summary

    def validation(self):
        saved = rng_state(); summary = self.evaluate('validation', None)
        score = max(summary[g]['cost']['mean'] for g in summary if g != 'uniform')
        if score < self.best['score']:
            self.best = {'score': score, 'step': self.step}
            save_pt(self.out / 'best.pt', {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'best': self.best, 'price': self.price})
        append(self.out / 'validation.jsonl', {'step': self.step, 'summary': summary, 'best': self.best})
        restore_rng(saved)

    def run(self):
        if (self.out / 'complete.json').exists(): return 0
        if self.best['step'] < 0:
            self.validation(); save_pt(self.out / 'latest.pt', self.state())
        total = self.p['updates'][self.family]; batchsize = self.p['batch_size']
        for step in range(self.step + 1, total + 1):
            started = time.monotonic(); self.heartbeat('training')
            b = materialize_training_batch(protocol=self.p, degree=10, training_seed=self.seed, update=step, batch_size=batchsize)
            support = generate_support(b, training_seed=self.seed, update=step, search=self.p['search'])
            indices, _ = shortlist_support(support, 4)
            z = support.pins[np.arange(batchsize)[:, None], indices]; flat = z.reshape(-1, 10, 2)
            current, _ = self.greedy(self.actor, flat)
            reference_started = time.monotonic()
            if self.arm == 'geo':
                reference = self.geo.lengths(flat); self.ledger['oracle_train_nets'] += len(flat)
            elif self.arm == 'k3':
                reference = np.min([self.greedy(m, flat)[0] for m in self.teachers], axis=0)
                self.ledger['teacher_train_nets'] += 3 * len(flat)
            elif self.arm == 'critic_inverse':
                pred = self.prediction(flat); denom = 1 + pred
                if (denom <= 0).any() or not np.isfinite(denom).all(): raise ValueError('invalid critic denominator')
                reference = current / denom; self.ledger['critic_reference_nets'] += len(flat)
            else: reference = np.ones(len(flat))
            reference_seconds = time.monotonic() - reference_started
            if (reference <= 0).any() or not np.isfinite(reference).all(): raise ValueError('invalid reference')
            offset = 0 if self.arm == 'original' else 1
            surrogate = (current / reference - offset).reshape(batchsize, 4)
            transport = costs(b.pins, z, self.family != 'rest')
            col = np.argmax(surrogate - self.price * transport, axis=1); rows = np.arange(batchsize)
            x = z[rows, col]; ref = reference.reshape(batchsize, 4)[rows, col]
            mean_transport = float(transport[rows, col].mean()); self.price = max(0., self.price + mean_transport - self.rho)
            self.actor.train(); self.critic.train()
            sampled, logp, _ = self.score(self.actor, x, False)
            target = torch.as_tensor(sampled / ref - offset, dtype=torch.float32, device=self.device)
            baseline = self.critic(x)
            loss = ((target - baseline).detach() * logp).mean() + (baseline - target).square().mean()
            self.opt.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.p['gradient_clip'], error_if_nonfinite=True)
            self.opt.step(); self.step = step
            seconds = time.monotonic() - started; self.train_seconds += seconds
            append(self.out / 'train.jsonl', {'step': step, 'seconds': seconds, 'reference_seconds': reference_seconds, 'lambda': self.price, 'rho': self.rho,
                    'transport_mean': mean_transport, 'target_mean': float(target.mean()), 'critic_mean': float(baseline.mean().detach()), 'negative_target_fraction': float((target < 0).float().mean()),
                    'loss': float(loss.detach()), 'grad_norm': float(norm), 'selected_columns': col.tolist(), 'ledger': self.ledger.copy()})
            if step % self.p['validation_every'] == 0 or step == total: self.validation()
            if step % self.p['checkpoint_every'] == 0 or step == total or self.stopped: save_pt(self.out / 'latest.pt', self.state())
            if step % 8 == 0: self.heartbeat('training'); print(json.dumps({'family': self.family, 'arm': self.arm, 'seed': self.seed, 'step': step, 'seconds': seconds}), flush=True)
            if self.stopped: self.heartbeat('paused'); return 75
        self.heartbeat('evaluating')
        final = self.evaluate('test', 'test_final')
        best = torch.load(self.out / 'best.pt', map_location=self.device, weights_only=False)
        self.actor.load_state_dict(best['actor']); self.critic.load_state_dict(best['critic'])
        selected = self.evaluate('test', 'test_best')
        atomic_json(self.out / 'results.json', {'family': self.family, 'arm': self.arm, 'seed': self.seed, 'best': self.best, 'test_best': selected, 'test_final': final,
                    'training_seconds': self.train_seconds, 'ledger': self.ledger, 'lambda_final': self.price, 'max_gpu_allocated_mib': torch.cuda.max_memory_allocated() / 2**20})
        atomic_json(self.out / 'complete.json', {'status': 'complete', 'updates': self.step, 'time': time.time(), 'scope': self.p['scope']})
        self.heartbeat('complete'); return 0

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--family'); parser.add_argument('--arm'); parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.prepare: prepare_inputs(); return 0
    experiment = Experiment(args.family, args.arm, args.seed)
    try: return experiment.run()
    finally: experiment.lock.close()

if __name__ == '__main__': sys.exit(main())

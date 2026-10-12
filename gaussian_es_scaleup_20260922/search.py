"""Gaussian smoothing of the exact legal objective, with exact request budgets."""
import numpy as np
from common import legalize, derived_seed, pin_set_sha256
from adv_v3.search import _History, antithetic_gradient, BUDGETS, attack as legacy_attack
from src.candidates import generate_support
from adv_v3.search import _batch, _SEARCH

def attack(method, centers, oracle, price, seed, *, mu=.02, step=.01, budget=64):
    if method == 'cem':
        if budget != 64: raise ValueError('independent CEM budget must be 64')
        return legacy_attack('cem64', centers, oracle, price, seed)
    if method not in ('es', 'random'): raise ValueError(method)
    if budget not in (64, 256): raise ValueError('budget must be 64 or 256')
    b, n, _ = centers.shape
    history = _History(centers, oracle, price)
    key = f'scaleup_{method}{budget}'; BUDGETS[key] = budget
    if method == 'random':
        support = generate_support(_batch(centers), training_seed=seed, update=1,
                                   search=dict(_SEARCH, candidate_count=budget-1))
        history.evaluate(support.pins)
        return history.finish(key, {'budget': budget})
    pairs, rounds = (4, 7) if budget == 64 else (8, 15)
    delta = np.zeros_like(centers, dtype=np.float64)
    orders = [np.lexsort((x[:,1], x[:,0])) for x in centers]
    identities = [pin_set_sha256(x) for x in centers]
    history.evaluate(centers[:,None])
    rms_values = []
    for r in range(rounds):
        eps = np.empty((b, pairs, n, 2))
        for i, order in enumerate(orders):
            rng = np.random.default_rng(derived_seed('gaussian-scaleup-es', seed, identities[i], r))
            eps[i][:,order,:] = rng.normal(size=(pairs,n,2))
        pos = np.clip(delta[:,None] + mu*eps, -.08, .08)
        neg = np.clip(delta[:,None] - mu*eps, -.08, .08)
        rec = history.evaluate(legalize(centers, np.concatenate([pos,neg],axis=1)))
        grad = antithetic_gradient(rec['score'][:,:pairs], rec['score'][:,pairs:], eps, mu)
        rms = np.sqrt(np.mean(grad*grad, axis=(1,2)))
        direction = np.divide(grad, rms[:,None,None], out=np.zeros_like(grad), where=rms[:,None,None]>0)
        delta = np.clip(delta+step*direction, -.08, .08)
        history.evaluate(legalize(centers,delta)[:,None]); rms_values.append(float(rms.mean()))
    return history.finish(key, dict(mu=mu, step=step, pairs=pairs, rounds=rounds,
                                   gradient_rms=rms_values, budget=budget))

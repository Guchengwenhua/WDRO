"""Finite-support WDRO with uniform nominal mass and mean L1 transport."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from src.costs import hungarian_l1_mean


def transport_costs(centers, candidates):
    x, z = np.asarray(centers), np.asarray(candidates)
    if x.ndim != 3 or z.ndim != 4 or z.shape[0] != len(x) or z.shape[2:] != x.shape[1:]:
        raise ValueError("expected centers [B,n,2], candidates [B,K,n,2]")
    return np.asarray([[hungarian_l1_mean(a, b) for b in row]
                       for a, row in zip(x, z, strict=True)])


def validate_scores(regrets, costs, rho):
    loss, cost = np.asarray(regrets, dtype=np.float64), np.asarray(costs, dtype=np.float64)
    if loss.ndim != 2 or not all(loss.shape) or loss.shape != cost.shape:
        raise ValueError("expected matching nonempty per-center score matrices")
    if not np.isfinite(loss).all() or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("invalid regret or nonnegative transport cost")
    if not np.isfinite(rho) or rho < 0 or not np.all(np.any(cost == 0, axis=1)):
        raise ValueError("nonnegative finite rho and a zero-cost candidate per center required")
    return loss, cost


def select_wdro(indices, regrets, costs, price, rho):
    loss, cost = validate_scores(regrets, costs, rho)
    ids = np.asarray(indices)
    if ids.shape != loss.shape or ids.dtype.kind not in "iu" or not np.isfinite(price) or price < 0:
        raise ValueError("invalid candidate indices or nonnegative lambda")
    # rho=0 is the exact lambda -> infinity limit: only zero transport is feasible.
    scores = np.where(cost == 0, loss, -np.inf) if rho == 0 else loss - price * cost
    columns = scores.argmax(axis=1)
    return ids[np.arange(len(ids)), columns], columns


def update_lambda(price, mean_cost, rho, eta):
    if not np.isfinite([price, mean_cost, rho, eta]).all() or min(price, mean_cost, rho) < 0 or eta <= 0:
        raise ValueError("invalid projected dual update")
    return float(max(0.0, price + eta * (mean_cost - rho)))


def primal_dual_lp(regrets, costs, rho):
    """Independent primal max and dual min LPs; conditional row masses sum to 1."""
    loss, cost = validate_scores(regrets, costs, rho)
    b, k = loss.shape
    equalities = np.zeros((b, b*k))
    for i in range(b):
        equalities[i, i*k:(i+1)*k] = 1
    primal = linprog(-loss.ravel()/b, A_ub=cost.reshape(1,-1)/b, b_ub=[rho],
                     A_eq=equalities, b_eq=np.ones(b), bounds=(0, None), method="highs")
    # t_i + lambda*c_ik >= loss_ik. t_i is unrestricted.
    constraints = np.zeros((b*k, b+1))
    constraints[:, 0] = -cost.ravel()
    for i in range(b):
        constraints[i*k:(i+1)*k, i+1] = -1
    dual = linprog(np.r_[rho, np.ones(b)/b], A_ub=constraints, b_ub=-loss.ravel(),
                   bounds=[(0,None)]+[(None,None)]*b, method="highs")
    if not primal.success or not dual.success:
        raise RuntimeError(f"LP failed: {primal.message}; {dual.message}")
    mass = primal.x.reshape(b,k)
    price = float(dual.x[0])
    primal_value, dual_value = float(-primal.fun), float(dual.fun)
    cost_used = float((mass*cost).sum()/b)
    explicit_dual = float(price*rho + (loss-price*cost).max(axis=1).mean())
    checks = {"mass_nonnegative": bool(mass.min() >= -1e-9),
              "row_marginals": bool(np.allclose(mass.sum(axis=1),1,atol=1e-9)),
              "budget_feasible": cost_used <= rho+1e-9,
              "dual_price_nonnegative": price >= 0,
              "weak_and_strong_duality": abs(dual_value-primal_value) <= 1e-8,
              "explicit_dual_matches_lp": abs(dual_value-explicit_dual) <= 1e-8,
              "complementarity": abs(price*(cost_used-rho)) <= 1e-8}
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
            "rho": float(rho), "primal_value": primal_value, "dual_value": dual_value,
            "gap": dual_value-primal_value, "lambda": price, "mean_cost": cost_used,
            "conditional_mass": mass.tolist(), "explicit_dual_value": explicit_dual}

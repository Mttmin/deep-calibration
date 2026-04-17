"""
Phase 0B: Andersen (2008) QE Heston Monte Carlo Pricer
======================================================

Pure-numpy implementation of the Quadratic-Exponential discretisation scheme
from Andersen, L. (2008), "Simple and Efficient Simulation of the Heston
Stochastic Volatility Model", Journal of Computational Finance 11(3).

Serves as ground-truth reference pricer for the multi-method COS/asymptotic
pricer built in Phase 1 and validated in Phase 2 of SCENARIO_PLAN.md.

Not optimised for speed; vectorised over paths with a Python loop over time
steps. At 50k paths, 252 steps, one parameter set: ~1-3s. Suitable for Phase
0B validation. Phase 2 may add a faster backend behind the same signature.

All formulas are cited against Andersen (2008) eq. numbers in the docstrings.
See /home/mttmin/.claude/plans/scalable-scribbling-wave.md for derivations
and hallucination-check table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

PSI_C: float = 1.5
EPS_STAB: float = 1e-14
DEFAULT_SEED: int = 20260417


@dataclass
class MCMeta:
    dt: float
    n_total: int
    mat_indices: np.ndarray
    antithetic: bool
    n_paths: int
    seed: int


def _qe_variance_step(
    V: np.ndarray,
    dt: float,
    kappa: float,
    theta: float,
    sigma: float,
    Zv: np.ndarray,
    Uv: np.ndarray,
):
    """One QE step for the CIR variance process.

    Returns (V_new, quad_mask, a, b2, p, beta) where a/b2 are valid on
    quad_mask and p/beta are valid on ~quad_mask; entries elsewhere
    are safe placeholders (never NaN/inf) to keep vectorised martingale
    correction simple downstream.

    Andersen (2008) eqs. (17), (18), (23)-(30).
    """
    E = math.exp(-kappa * dt)
    one_minus_E = 1.0 - E

    # Conditional mean and variance of V(t+dt) | V(t)  [Andersen (17), (18)]
    m = theta + (V - theta) * E
    s2 = (
        V * (sigma * sigma) * E * one_minus_E / kappa
        + theta * (sigma * sigma) * (one_minus_E * one_minus_E) / (2.0 * kappa)
    )

    m_safe2 = np.maximum(m * m, EPS_STAB)
    psi = s2 / m_safe2
    quad = psi <= PSI_C

    # Quadratic branch  [Andersen (27), (28), (23)-(24)]
    # b2 = 2/psi - 1 + sqrt((2/psi)(2/psi - 1))
    inv_psi = 1.0 / np.maximum(psi, EPS_STAB)
    two_inv = 2.0 * inv_psi
    radicand = np.maximum(two_inv * (two_inv - 1.0), 0.0)  # >=0 iff psi<=2
    b2 = two_inv - 1.0 + np.sqrt(radicand)
    b2 = np.where(quad, np.maximum(b2, 0.0), 0.0)
    a = np.where(quad, m / (1.0 + b2), 0.0)
    # V_q defined only for quad paths; mask to 0 elsewhere for safety
    V_q = a * (np.sqrt(b2) + Zv) ** 2

    # Exponential branch  [Andersen (29), (30), (25)-(26)]
    # Protect psi->1 from below; where psi<=1 the exp branch is unreachable
    # (since PSI_C=1.5>1 forces quad=True there) but guards are cheap.
    psi_clip = np.where(quad, 2.0, psi)  # any value > 1 works when quad
    p = (psi_clip - 1.0) / (psi_clip + 1.0)
    beta = (1.0 - p) / np.maximum(m, EPS_STAB)
    # V_e: 0 if U<=p else ln((1-p)/(1-U))/beta
    one_minus_U = np.maximum(1.0 - Uv, EPS_STAB)
    one_minus_p = np.maximum(1.0 - p, EPS_STAB)
    V_e = np.where(Uv <= p, 0.0, np.log(one_minus_p / one_minus_U) / np.maximum(beta, EPS_STAB))

    V_new = np.where(quad, V_q, V_e)
    return V_new, quad, a, b2, p, beta


def _log_stock_step(
    logS: np.ndarray,
    V: np.ndarray,
    V_new: np.ndarray,
    dt: float,
    r: float,
    q: float,
    rho: float,
    sigma: float,
    kappa: float,
    quad: np.ndarray,
    a: np.ndarray,
    b2: np.ndarray,
    p: np.ndarray,
    beta: np.ndarray,
    Zs: np.ndarray,
):
    """Log-stock update with Andersen's martingale correction K0*.

    Andersen (2008) eqs. (32)-(41). gamma_1 = gamma_2 = 0.5 (central
    discretisation). The scalar K coefficients are constant per step; K0*
    is path-dependent via branch and (a, b2) or (p, beta).
    """
    kappa_rho_over_sigma = kappa * rho / sigma
    half_dt = 0.5 * dt
    K1_const = half_dt * (kappa_rho_over_sigma - 0.5) - rho / sigma
    K2_const = half_dt * (kappa_rho_over_sigma - 0.5) + rho / sigma
    K3_const = half_dt * (1.0 - rho * rho)
    K4_const = K3_const
    A = K2_const + 0.5 * K4_const

    # Stability guards. For equity regime (rho<0), A is strongly negative
    # dominated by rho/sigma term, so both guards hold comfortably.
    denom_q = 1.0 - 2.0 * A * a
    if quad.any() and not np.all(denom_q[quad] > EPS_STAB):
        bad = float(denom_q[quad].min())
        raise ValueError(
            f"QE martingale correction unstable (quadratic branch): "
            f"min(1-2Aa) = {bad:.3e}; A={A:.3e}, dt={dt:.3e}, "
            f"kappa={kappa}, sigma={sigma}, rho={rho}"
        )
    if (~quad).any() and not np.all((beta[~quad] - A) > EPS_STAB):
        bad = float((beta[~quad] - A).min())
        raise ValueError(
            f"QE martingale correction unstable (exponential branch): "
            f"min(beta - A) = {bad:.3e}; A={A:.3e}, dt={dt:.3e}, "
            f"kappa={kappa}, sigma={sigma}, rho={rho}"
        )

    # K0* piecewise  [Andersen (40), (41)]
    K0s_q = 0.5 * np.log(np.maximum(denom_q, EPS_STAB)) - A * a * b2 / np.maximum(denom_q, EPS_STAB) \
        - (K1_const + 0.5 * K3_const) * V
    # exponential branch: p + beta*(1-p)/(beta - A) must be > 0
    exp_term = p + beta * (1.0 - p) / np.maximum(beta - A, EPS_STAB)
    K0s_e = -np.log(np.maximum(exp_term, EPS_STAB)) - (K1_const + 0.5 * K3_const) * V
    K0s = np.where(quad, K0s_q, K0s_e)

    diffusion_var = np.maximum(K3_const * V + K4_const * V_new, 0.0)
    logS_new = (
        logS
        + (r - q) * dt
        + K0s
        + K1_const * V
        + K2_const * V_new
        + np.sqrt(diffusion_var) * Zs
    )
    return logS_new


def _build_time_grid(T_arr_sorted: np.ndarray, steps_per_year: int):
    """Uniform dt grid; every maturity T snaps to the nearest step index.

    Returns (dt, mat_indices, n_total) where step k corresponds to time
    (k+1)*dt for k in [0, n_total-1].  mat_indices[j] is the 0-based step
    whose end-time equals T_arr_sorted[j] (within dt/2 tolerance).
    """
    T_max = float(T_arr_sorted[-1])
    n_total = max(1, int(math.ceil(T_max * steps_per_year)))
    dt = T_max / n_total
    # end-of-step times are dt*(1..n_total)
    mat_indices = np.rint(T_arr_sorted / dt).astype(np.int64) - 1
    mat_indices = np.clip(mat_indices, 0, n_total - 1)
    snap_err = np.abs((mat_indices + 1) * dt - T_arr_sorted)
    if np.any(snap_err > 0.5 * dt + 1e-12):
        raise ValueError(
            f"Maturity grid snap failed: max err {snap_err.max():.3e} > dt/2. "
            f"Raise steps_per_year."
        )
    # check uniqueness
    if len(np.unique(mat_indices)) != len(mat_indices):
        raise ValueError("Two maturities collided on same step index; raise steps_per_year.")
    return dt, mat_indices, n_total


def price_european_grid(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    r: float,
    q: float,
    K_arr,
    T_arr,
    spot: float,
    n_paths: int,
    steps_per_year: int = 252,
    seed: Optional[int] = None,
    antithetic: bool = True,
    option_type: str = "call",
):
    """Price European options on an NK x NT grid for a single Heston parameter set.

    Returns (prices, se, meta) with prices and se of shape (NT, NK) matching
    the order of T_arr and K_arr as supplied by the caller.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    K_arr = np.asarray(K_arr, dtype=np.float64)
    T_arr_in = np.asarray(T_arr, dtype=np.float64)
    if K_arr.ndim != 1 or T_arr_in.ndim != 1:
        raise ValueError("K_arr and T_arr must be 1-D")
    if np.any(T_arr_in <= 0):
        raise ValueError("All T must be > 0")

    NT = T_arr_in.size
    NK = K_arr.size

    # Sort maturities ascending, build inverse permutation to restore caller order
    perm = np.argsort(T_arr_in, kind="stable")
    inv_perm = np.argsort(perm)
    T_sorted = T_arr_in[perm]

    dt, mat_indices, n_total = _build_time_grid(T_sorted, steps_per_year)

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("antithetic=True requires even n_paths")
        n_primary = n_paths // 2
    else:
        n_primary = n_paths

    actual_seed = int(seed) if seed is not None else DEFAULT_SEED
    rng = np.random.default_rng(actual_seed)

    V = np.full(n_paths, v0, dtype=np.float64)
    logS = np.full(n_paths, math.log(spot), dtype=np.float64)

    sum_p = np.zeros((NT, NK), dtype=np.float64)
    sum_p2 = np.zeros((NT, NK), dtype=np.float64)

    # Map step index -> row index in output grid
    step_to_row = {int(idx): j for j, idx in enumerate(mat_indices)}

    call_sign = 1.0 if option_type == "call" else -1.0

    for step in range(n_total):
        Zv_p = rng.standard_normal(n_primary)
        Uv_p = rng.random(n_primary)  # [0,1)
        Zs_p = rng.standard_normal(n_primary)
        if antithetic:
            Zv = np.concatenate([Zv_p, -Zv_p])
            Uv = np.concatenate([Uv_p, 1.0 - Uv_p])
            Zs = np.concatenate([Zs_p, -Zs_p])
        else:
            Zv, Uv, Zs = Zv_p, Uv_p, Zs_p

        V_new, quad, a, b2, p, beta = _qe_variance_step(
            V, dt, kappa, theta, sigma, Zv, Uv
        )
        logS = _log_stock_step(
            logS, V, V_new, dt, r, q, rho, sigma, kappa,
            quad, a, b2, p, beta, Zs,
        )
        V = np.maximum(V_new, 0.0)

        if step in step_to_row:
            row = step_to_row[step]
            T_here = (step + 1) * dt
            S_T = np.exp(logS)
            disc = math.exp(-r * T_here)
            diff = call_sign * (S_T[:, None] - K_arr[None, :])
            payoff = disc * np.maximum(diff, 0.0)
            if antithetic:
                # Pair-mean statistics: the iid unit is (primary + mirror)/2.
                pair_mean = 0.5 * (payoff[:n_primary] + payoff[n_primary:])
                sum_p[row] = pair_mean.sum(axis=0)
                sum_p2[row] = (pair_mean * pair_mean).sum(axis=0)
            else:
                sum_p[row] = payoff.sum(axis=0)
                sum_p2[row] = (payoff * payoff).sum(axis=0)

    n_iid = n_primary if antithetic else n_paths
    mean = sum_p / n_iid
    var = np.maximum(sum_p2 / n_iid - mean * mean, 0.0)
    se = np.sqrt(var / n_iid)

    # Restore caller's T ordering
    prices_out = mean[inv_perm]
    se_out = se[inv_perm]

    meta = MCMeta(
        dt=dt,
        n_total=n_total,
        mat_indices=mat_indices[inv_perm].copy() if False else mat_indices.copy(),
        antithetic=antithetic,
        n_paths=n_paths,
        seed=actual_seed,
    )
    return prices_out, se_out, meta


def price_european_with_se(
    kappa, theta, sigma, rho, v0, r, q, K, T, spot,
    n_paths, n_steps, seed=None, antithetic=True, option_type="call",
):
    """Scalar wrapper that returns (price, standard_error)."""
    steps_per_year = max(1, int(round(n_steps / max(T, 1e-12))))
    prices, se, _ = price_european_grid(
        kappa, theta, sigma, rho, v0, r, q,
        K_arr=np.array([K], dtype=np.float64),
        T_arr=np.array([T], dtype=np.float64),
        spot=spot, n_paths=n_paths, steps_per_year=steps_per_year,
        seed=seed, antithetic=antithetic, option_type=option_type,
    )
    return float(prices[0, 0]), float(se[0, 0])


def price_european(
    kappa, theta, sigma, rho, v0, r, q, K, T, spot, n_paths, n_steps,
):
    """Mandatory scalar signature per SCENARIO_PLAN.md Phase 0B.

    Returns the MC price only; use price_european_with_se to get the SE.
    """
    price, _ = price_european_with_se(
        kappa, theta, sigma, rho, v0, r, q, K, T, spot, n_paths, n_steps,
    )
    return price

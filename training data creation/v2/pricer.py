"""
Phase 1: Multi-method Heston pricer (COS + FJL asymptotic fallback).
====================================================================

Dispatches each (param, K, T) cell across:

  1. COS with HALF_ABS=5.0, N_COS=512 in float64  (method: cos_standard)
  2. COS with HALF_ABS=10.0, N_COS=2048            (method: cos_extended)
  3. Forde-Jacquier-Lee small-time asymptotic      (method: asymptotic_fjl)
                                                   only when |log(K/S)| > 0.3
  4. Unpricable marker (below precision)           (method: unpricable)

Per SCENARIO_PLAN.md Phase 1A. Characteristic function (Little Trap form,
Albrecher et al. 2007) and call-payoff COS coefficients match the legacy
`training data creation/heston_datagen.py` pricer for bitwise continuity
of cells that both pricers can handle.

FJL small-time asymptotic: Forde & Jacquier (2009) Theorem 2.4, with
the ATM expansion of Theorem 2.5 used as a |x|→0 Taylor fallback.
Rate-function formulas derived directly from the pole structure of
Lambda(p) in Section 1.2 of v2/docs/asymptotic_methods.md (the doc's
bounds table has a factor-4 typo; we use u* = atan2(bar_rho, rho)
which is the correct pole of Lambda).

All math in pure numpy / math; intentionally scalar to keep the
dispatcher simple and debuggable. Batched/vectorised variants can be
layered on in Phase 3 once the scalar version is validated end-to-end.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np

HALF_ABS_STD: float = 5.0
N_COS_STD: int = 512
HALF_ABS_EXT: float = 10.0
N_COS_EXT: int = 2048
DEEP_OTM_THRESHOLD: float = 0.3
RATIO_ACCEPT: float = 1e-4
RATIO_PRECISION: float = 1e-6


# -------------------- Heston CF (Little Trap) --------------------

def _heston_cf(
    u: np.ndarray, T: float,
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    r: float, q: float,
) -> np.ndarray:
    """Characteristic function of log(S_T/S_0) under Heston (1993).

    Little-Trap branch-cut-safe form (Albrecher et al. 2007).
    """
    uc = u.astype(np.complex128)
    xi = kappa - 1j * rho * sigma * uc
    d = np.sqrt(xi * xi + sigma * sigma * uc * (uc + 1j))
    g = (xi - d) / (xi + d)
    e_dT = np.exp(-d * T)
    one_m_g = 1.0 - g
    one_m_g_eT = 1.0 - g * e_dT
    log_q = np.log(one_m_g_eT / one_m_g)
    A = 1j * (r - q) * uc * T
    B = (kappa * theta / (sigma * sigma)) * ((xi - d) * T - 2.0 * log_q)
    C = (v0 / (sigma * sigma)) * (xi - d) * (1.0 - e_dT) / one_m_g_eT
    return np.exp(A + B + C)


# -------------------- COS pricer (scalar) --------------------

def _cos_call_price_scalar(
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    r: float, q: float, K: float, T: float, spot: float,
    half_abs: float, n_cos: int,
) -> float:
    """Fang-Oosterlee COS call price, single (K, T). Returns float64.

    Returns NaN if the CF evaluation or aggregation produces non-finite
    values; returns a possibly-tiny but finite number otherwise (caller
    decides whether it is above precision).
    """
    a, b = -half_abs, half_abs
    ba = b - a
    k_idx = np.arange(n_cos, dtype=np.float64)
    u_k = k_idx * math.pi / ba
    ang0 = -u_k * a
    cos0 = np.cos(ang0)
    sin0 = np.sin(ang0)
    cos_kpi = np.cos(k_idx * math.pi)
    denom = 1.0 + u_k * u_k
    denom[0] = 1.0
    chi = (math.exp(b) * cos_kpi - cos0 - u_k * sin0) / denom
    chi[0] = math.exp(b) - 1.0
    psi = np.zeros(n_cos, dtype=np.float64)
    psi[0] = b
    psi[1:] = -sin0[1:] / u_k[1:]
    Vk = (2.0 / ba) * (chi - psi)
    Vk[0] *= 0.5

    phi = _heston_cf(u_k, T, kappa, theta, sigma, rho, v0, r, q)
    c_base = phi * np.exp(-1j * u_k * a)
    x_l = math.log(spot / K)
    phase = u_k * x_l
    integrand = c_base.real * np.cos(phase) - c_base.imag * np.sin(phase)
    s = float(np.sum(integrand * Vk))
    if not math.isfinite(s):
        return float("nan")
    return math.exp(-r * T) * K * s


def _cos_price(
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    r: float, q: float, K: float, T: float, spot: float,
    option_type: str, half_abs: float, n_cos: int,
) -> float:
    """COS price for call or put. Put via put-call parity."""
    call = _cos_call_price_scalar(
        kappa, theta, sigma, rho, v0, r, q, K, T, spot, half_abs, n_cos
    )
    if not math.isfinite(call):
        return float("nan")
    call = max(call, 0.0)
    if option_type == "call":
        return call
    put = call + K * math.exp(-r * T) - spot * math.exp(-q * T)
    return max(put, 0.0)


# -------------------- FJL small-time asymptotic --------------------

def _fjl_pole_bounds(sigma: float, rho: float) -> Tuple[float, float]:
    """Lambda(p) pole boundaries (p_-, p_+).

    Derived from cot(u*) = rho / sqrt(1-rho^2), i.e. the smallest u* in
    (0, pi) with tan(u*) = sqrt(1-rho^2)/rho is u* = atan2(bar_rho, rho).
    Then p_+ = 2 u* / (sigma bar_rho) and p_- = p_+ - 2 pi / (sigma bar_rho).
    """
    bar_rho = math.sqrt(1.0 - rho * rho)
    u_plus = math.atan2(bar_rho, rho)
    p_plus = 2.0 * u_plus / (sigma * bar_rho)
    p_minus = p_plus - 2.0 * math.pi / (sigma * bar_rho)
    return p_minus, p_plus


def _fjl_Lambda(p: float, v0: float, sigma: float, rho: float) -> float:
    """Cumulant generating function Lambda(p); asymptotic_methods.md §1.2.

    Taylor fallback for |u| < 1e-6 covers p->0 where cot(u) diverges but
    Lambda(0) = 0 analytically: Lambda(p) = v0/2 p^2 + v0 sigma rho/4 p^3 + O(p^4).
    """
    bar_rho = math.sqrt(1.0 - rho * rho)
    u = 0.5 * sigma * p * bar_rho
    if abs(u) < 1e-6:
        return 0.5 * v0 * p * p + 0.25 * v0 * sigma * rho * p ** 3
    g = bar_rho / math.tan(u) - rho
    return v0 * p / (sigma * g)


def _fjl_Lambda_prime(p: float, v0: float, sigma: float, rho: float) -> float:
    """Lambda'(p) = (v0/sigma) (g - p g') / g^2, with g' = -sigma bar_rho^2 / (2 sin^2 u)."""
    bar_rho = math.sqrt(1.0 - rho * rho)
    u = 0.5 * sigma * p * bar_rho
    if abs(u) < 1e-6:
        return v0 * p + 0.75 * v0 * sigma * rho * p * p
    sin_u = math.sin(u)
    cot_u = math.cos(u) / sin_u
    g = bar_rho * cot_u - rho
    g_prime = -0.5 * sigma * bar_rho * bar_rho / (sin_u * sin_u)
    return (v0 / sigma) * (g - p * g_prime) / (g * g)


def _fjl_find_pstar(
    x: float, v0: float, sigma: float, rho: float,
    max_iter: int = 200, tol: float = 1e-12,
) -> float:
    """Solve Lambda'(p) = x for p* in (p_-, p_+) via bisection."""
    p_minus, p_plus = _fjl_pole_bounds(sigma, rho)
    eps_frac = 1e-6
    if x > 0.0:
        lo = 1e-12
        hi = p_plus * (1.0 - eps_frac)
    else:
        lo = p_minus * (1.0 - eps_frac)
        hi = -1e-12
    f_lo = _fjl_Lambda_prime(lo, v0, sigma, rho) - x
    f_hi = _fjl_Lambda_prime(hi, v0, sigma, rho) - x
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return float("nan")
    if f_lo * f_hi > 0.0:
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _fjl_Lambda_prime(mid, v0, sigma, rho) - x
        if not math.isfinite(f_mid):
            return float("nan")
        if abs(f_mid) < tol or 0.5 * (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _fjl_iv(x: float, v0: float, sigma: float, rho: float) -> float:
    """Leading-order FJL implied BS vol (Forde-Jacquier 2009 Thm 2.4).

    For |x| small, uses the ATM expansion of Thm 2.5.
    """
    if abs(x) < 1e-6:
        z = sigma * x / v0
        return math.sqrt(v0) * (
            1.0 + 0.25 * rho * z
            + (1.0 / 24.0 - 5.0 * rho * rho / 48.0) * z * z
        )
    p_star = _fjl_find_pstar(x, v0, sigma, rho)
    if not math.isfinite(p_star):
        return float("nan")
    Lambda_at_p = _fjl_Lambda(p_star, v0, sigma, rho)
    if not math.isfinite(Lambda_at_p):
        return float("nan")
    Lambda_star = p_star * x - Lambda_at_p
    if Lambda_star <= 0.0:
        return float("nan")
    return x / math.sqrt(2.0 * Lambda_star)


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _bs_price(
    spot: float, K: float, T: float, r: float, q: float,
    sigma_bs: float, option_type: str,
) -> float:
    """Black-Scholes price from an implied BS vol."""
    if not math.isfinite(sigma_bs) or sigma_bs <= 0.0 or T <= 0.0:
        return float("nan")
    F = spot * math.exp((r - q) * T)
    disc = math.exp(-r * T)
    sqT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma_bs * sigma_bs * T) / (sigma_bs * sqT)
    d2 = d1 - sigma_bs * sqT
    N = lambda z: 0.5 * (1.0 + math.erf(z * _INV_SQRT2))
    if option_type == "call":
        return max(disc * (F * N(d1) - K * N(d2)), 0.0)
    return max(disc * (K * N(-d2) - F * N(-d1)), 0.0)


def _fjl_price(
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    r: float, q: float, K: float, T: float, spot: float, option_type: str,
) -> float:
    """Small-time-asymptotic Heston price: FJL IV → BS price."""
    x = math.log(K / spot)
    sigma_bs = _fjl_iv(x, v0, sigma, rho)
    if not math.isfinite(sigma_bs) or sigma_bs <= 0.0:
        return float("nan")
    return _bs_price(spot, K, T, r, q, sigma_bs, option_type)


# -------------------- Main dispatcher --------------------

def price_cell(
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    r: float, q: float, K: float, T: float, spot: float,
    option_type: Literal["call", "put"] = "call",
) -> Tuple[float, str, float]:
    """Multi-method price for a single (param, K, T) cell.

    Returns (price, method_flag, confidence) per SCENARIO_PLAN.md Phase 1A.

    method_flag ∈ {cos_standard, cos_small_price, cos_extended,
                   asymptotic_fjl, unpricable}
    """
    x = math.log(K / spot)

    # Step 1-3: standard COS.
    p_std = _cos_price(kappa, theta, sigma, rho, v0, r, q,
                       K, T, spot, option_type,
                       HALF_ABS_STD, N_COS_STD)
    if math.isfinite(p_std):
        ratio = p_std / spot
        if ratio > RATIO_ACCEPT:
            return p_std, "cos_standard", 1.0
        if ratio > RATIO_PRECISION:
            return p_std, "cos_small_price", 0.7
        # Finite but below 1e-6 precision. Fall through to FJL (step 5).
    else:
        # Step 4: extended COS (only fires on NaN/inf from standard).
        p_ext = _cos_price(kappa, theta, sigma, rho, v0, r, q,
                           K, T, spot, option_type,
                           HALF_ABS_EXT, N_COS_EXT)
        if math.isfinite(p_ext):
            ratio = p_ext / spot
            if ratio > RATIO_ACCEPT:
                return p_ext, "cos_extended", 1.0
            if ratio > RATIO_PRECISION:
                return p_ext, "cos_extended", 0.7

    # Step 5: FJL asymptotic, deep-OTM only.
    if abs(x) > DEEP_OTM_THRESHOLD:
        p_fjl = _fjl_price(kappa, theta, sigma, rho, v0, r, q,
                           K, T, spot, option_type)
        if math.isfinite(p_fjl) and p_fjl > 0.0 and (p_fjl / spot) > RATIO_PRECISION:
            return p_fjl, "asymptotic_fjl", 0.5

    # Step 6: no method produced a price above precision.
    return float("nan"), "unpricable", 0.0

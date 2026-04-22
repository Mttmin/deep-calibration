"""
Phase 1: Multi-method Heston pricer (COS standard + extended fallback).
=======================================================================

Dispatches each (param, K, T) cell across:

  1. COS with adaptive half_abs (L=10), N_COS=512   (method: cos_standard)
  2. COS with adaptive half_abs (L=15), N_COS=2048  (method: cos_extended)
     Invoked when cos_standard returns NaN or price/spot < 1e-4.
  3. cos_small_price flag if cos_extended yields price in [1e-6, 1e-4) of spot.
  4. Unpricable marker (below 1e-6 precision)       (method: unpricable)

Adaptive truncation (Phase 2A-v2 fix, 2026-04-18) sizes the COS window per
cell as max(0.5, |c1+x_l| + L*sqrt(c2)) using the simplified Heston
cumulants. Puts use direct payoff coefficients (not put-call parity) to
avoid catastrophic cancellation when the twin call is deep ITM.

Per SCENARIO_PLAN.md Phase 1A + updated plan 2026-04-18 (FJL dropped).

Characteristic function (Little Trap form, Albrecher et al. 2007) and
call-payoff COS coefficients match the legacy
`training data creation/heston_datagen.py` pricer for bitwise continuity
of cells that both pricers can handle.

The Forde-Jacquier-Lee small-time asymptotic was previously dispatched in
this module but Phase 2A MC validation (2026-04-18) showed leading-order
FJL is structurally inadequate at the maturities present in the training
grid (mean |rel err| 8908% on FJL-flagged cells). The FJL implementation
is retained below in a DEPRECATED block for reference but is not in the
dispatch path. Cells that previously routed to FJL now route to
cos_extended → cos_small_price → unpricable.

All math in pure numpy / math; intentionally scalar to keep the
dispatcher simple and debuggable. Batched/vectorised variants can be
layered on in Phase 3 once the scalar version is validated end-to-end.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np

N_COS_STD: int = 512
N_COS_EXT: int = 2048
RATIO_ACCEPT: float = 1e-4
RATIO_PRECISION: float = 1e-6
# Adaptive truncation: half_abs = max(HALF_FLOOR, |c1+x_l| + L_FACTOR*sqrt(c2)).
# Phase 2A-v2 (2026-04-18) showed static HALF_ABS=5 under-resolved short-T cells
# (node spacing >= density sigma). Adaptive scales window with Heston cumulants.
L_FACTOR: float = 10.0
HALF_FLOOR: float = 0.5
L_FACTOR_EXT: float = 15.0  # wider margin on extended tier


# -------------------- Heston cumulants (simplified) --------------------

def _heston_cumulants_simplified(
    T: float, kappa: float, theta: float, v0: float, r: float, q: float,
) -> Tuple[float, float]:
    """Return (c1, c2) for log(S_T/S_0). c2 uses integrated variance only.

    Exact c1. For c2 we use integ_var = E[int_0^T v_s ds] = theta*T +
    (v0-theta)*(1-e^(-kT))/k, which is a lower bound on Var(log S_T) but
    dominates at short-to-moderate T. The missing sigma_v^2 and rho*sigma_v
    contributions are absorbed by the L_FACTOR=10 margin in
    `_adaptive_half_abs`.
    """
    if kappa * T < 1e-10:
        em1 = kappa * T
    else:
        em1 = 1.0 - math.exp(-kappa * T)
    integ_var = theta * T + (v0 - theta) * em1 / kappa
    c1 = (r - q) * T - 0.5 * integ_var
    return c1, integ_var


def _adaptive_half_abs(
    T: float, kappa: float, theta: float, v0: float,
    r: float, q: float, x_l: float, L_factor: float,
) -> float:
    """Adaptive COS truncation half-width covering |c1+x_l| + L*sqrt(c2)."""
    c1, c2 = _heston_cumulants_simplified(T, kappa, theta, v0, r, q)
    return max(HALF_FLOOR, abs(c1 + x_l) + L_factor * math.sqrt(max(c2, 0.0)))


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

def _cos_price_scalar(
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    r: float, q: float, K: float, T: float, spot: float,
    half_abs: float, n_cos: int, option_type: str,
) -> float:
    """Fang-Oosterlee COS price, single (K, T), direct coeffs for both sides.

    Call integrates max(e^y - 1, 0) over y in [0, b]; put integrates
    max(1 - e^y, 0) over y in [a, 0]. Using the side-of-money payoff avoids
    the catastrophic cancellation that put-via-call-parity exhibits when
    the call is deep ITM (Phase 2A-v2 BLOCKER, 2026-04-18).

    Returns NaN if CF evaluation or aggregation produces non-finite values.
    """
    a, b = -half_abs, half_abs
    ba = b - a
    k_idx = np.arange(n_cos, dtype=np.float64)
    u_k = k_idx * math.pi / ba
    cos_ub = np.cos(u_k * b)
    sin_ub = np.sin(u_k * b)
    cos_kpi = np.cos(k_idx * math.pi)  # (-1)^k
    denom = 1.0 + u_k * u_k
    denom[0] = 1.0

    if option_type == "call":
        # chi_k = int_0^b e^y cos(k*pi*(y-a)/(b-a)) dy
        # psi_k = int_0^b cos(k*pi*(y-a)/(b-a)) dy
        chi = (math.exp(b) * cos_kpi - cos_ub - u_k * sin_ub) / denom
        chi[0] = math.exp(b) - 1.0
        psi = np.zeros(n_cos, dtype=np.float64)
        psi[0] = b
        psi[1:] = -sin_ub[1:] / u_k[1:]
        Vk = (2.0 / ba) * (chi - psi)
    else:  # put
        exp_a = math.exp(a)
        # chi_k = int_a^0 e^y cos(k*pi*(y-a)/(b-a)) dy
        # psi_k = int_a^0 cos(k*pi*(y-a)/(b-a)) dy
        # Upper bound y=0 gives cos(u_k*(-a))=cos(u_k*b); lower bound
        # y=a gives cos(0)=1 (no (-1)^k factor — contrast with call case).
        chi = (cos_ub + u_k * sin_ub - exp_a) / denom
        chi[0] = 1.0 - exp_a
        psi = np.zeros(n_cos, dtype=np.float64)
        psi[0] = -a
        psi[1:] = sin_ub[1:] / u_k[1:]
        Vk = (2.0 / ba) * (psi - chi)
    Vk[0] *= 0.5

    phi = _heston_cf(u_k, T, kappa, theta, sigma, rho, v0, r, q)
    c_base = phi * np.exp(-1j * u_k * a)
    x_l = math.log(spot / K)
    phase = u_k * x_l
    integrand = c_base.real * np.cos(phase) - c_base.imag * np.sin(phase)
    s = float(np.sum(integrand * Vk))
    if not math.isfinite(s):
        return float("nan")
    price = math.exp(-r * T) * K * s
    return max(price, 0.0)


def _cos_price(
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    r: float, q: float, K: float, T: float, spot: float,
    option_type: str, n_cos: int, L_factor: float,
) -> float:
    """COS price with adaptive truncation sized from Heston cumulants."""
    x_l = math.log(spot / K)
    half_abs = _adaptive_half_abs(T, kappa, theta, v0, r, q, x_l, L_factor)
    return _cos_price_scalar(
        kappa, theta, sigma, rho, v0, r, q, K, T, spot,
        half_abs, n_cos, option_type,
    )


# -------------------- Main dispatcher --------------------

def price_cell(
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    r: float, q: float, K: float, T: float, spot: float,
    option_type: Literal["call", "put"] = "call",
) -> Tuple[float, str, float]:
    """Multi-method price for a single (param, K, T) cell.

    Returns (price, method_flag, confidence) per SCENARIO_PLAN.md Phase 1A
    as revised on 2026-04-18 (FJL dropped from dispatch).

    Dispatch:
      1. cos_standard  (HALF_ABS=5,  N=512). Accept if price/spot > 1e-4.
      2. cos_extended  (HALF_ABS=10, N=2048). Invoked when cos_standard
         is NaN OR returns price/spot < 1e-4. Accept if price/spot > 1e-4.
      3. cos_small_price flag if best COS attempt yields price in
         [1e-6, 1e-4) of spot. Bulk accuracy not guaranteed; downstream
         training masks these cells.
      4. unpricable if all methods fail or yield price < 1e-6 of spot.

    method_flag ∈ {cos_standard, cos_extended, cos_small_price, unpricable}
    """
    # Step 1: standard COS (adaptive half_abs, N=512).
    p_std = _cos_price(kappa, theta, sigma, rho, v0, r, q,
                       K, T, spot, option_type,
                       N_COS_STD, L_FACTOR)
    if math.isfinite(p_std) and (p_std / spot) > RATIO_ACCEPT:
        return p_std, "cos_standard", 1.0

    # Step 2: extended COS — wider window + N=2048; fires on NaN or small p_std.
    p_ext = _cos_price(kappa, theta, sigma, rho, v0, r, q,
                       K, T, spot, option_type,
                       N_COS_EXT, L_FACTOR_EXT)
    if math.isfinite(p_ext) and (p_ext / spot) > RATIO_ACCEPT:
        return p_ext, "cos_extended", 1.0

    # Step 3: small-price flag if either COS produced a finite price in
    # [1e-6, 1e-4) of spot. Prefer the extended result when available.
    candidate = p_ext if math.isfinite(p_ext) else p_std
    if math.isfinite(candidate) and (candidate / spot) > RATIO_PRECISION:
        return candidate, "cos_small_price", 0.7

    # Step 4: below precision — unpricable.
    return float("nan"), "unpricable", 0.0


# ===========================================================================
# DEPRECATED — kept for reference, not in dispatch.
# ===========================================================================
#
# Forde-Jacquier-Lee small-time asymptotic. Removed from `price_cell`
# dispatch on 2026-04-18 after Phase 2A MC validation showed leading-order
# FJL is structurally inadequate at the training-grid maturities (T up to
# 3y), with mean |rel err| 8908% on FJL-flagged cells (BLOCKER report:
# docs/BLOCKER_phase2a_mc_validation.md). The next-to-leading correction
# is O(sqrt(T)); a higher-order implementation would be needed to make FJL
# usable as a wing fallback even on the short end of the grid. Until that
# work is done the dispatcher routes deep-OTM cells to cos_extended →
# cos_small_price → unpricable instead. A future researcher revisiting
# wing accuracy may want to start here: the helper functions below
# (_fjl_Lambda, _fjl_Lambda_prime, _fjl_find_pstar, _fjl_iv, _fjl_price)
# implement Forde-Jacquier 2009 Theorem 2.4 + 2.5 (ATM expansion), with
# rate-function pole bounds derived in docs/asymptotic_methods.md §1.2.

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

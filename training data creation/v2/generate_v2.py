"""Phase 3.2: v2 training surface generation (10M train + 1M val).

Uses v2 pricer (adaptive COS, direct put coeffs, 2-tier dispatch) and
v2 realistic sampling (sampling.py, V0_LO=0.02). Stores quality_mask
with codes {0=cos_standard, 1=cos_extended, 2=cos_small_price,
3=unpricable} and a per-surface IV grid with NaN at unpricable cells.

Two execution paths share a single validation contract:

* ``--device cpu`` (or ``--smoke``): scalar reference via
  :func:`pricer.price_cell`. Slow — O(N * NK * NT) Python. Used to
  ground-truth the GPU port on tens of samples.
* ``--device cuda``: batched torch port of the same adaptive COS
  dispatch, per-cell kernels at float64. Same flag codes, same
  acceptance thresholds, same half_abs formula.

Smoke gate: max |price_gpu - price_cpu| / spot over cos_standard
cells must be < 1e-10 (float64 COS matches numpy). If flags disagree
on any cell, abort.

Grid (49 strikes × 14 maturities) and pricable region match Phase 3.1
output: ``training data creation/v2/pricable_region.npy``.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pricer  # noqa: E402
from pricer import (  # noqa: E402
    HALF_FLOOR, L_FACTOR, L_FACTOR_EXT,
    N_COS_STD, N_COS_EXT, RATIO_ACCEPT, RATIO_PRECISION,
    price_cell,
)
from sampling import sample_realistic_params, PARAM_NAMES  # noqa: E402

# ---- grid (matches Phase 2A-v2 / Phase 3.1) ---------------------------
LOG_MONEYNESS = np.linspace(-0.80, 0.40, 49, dtype=np.float64)
MATURITIES = np.array([
    1 / 52, 2 / 52, 3 / 52,
    1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 9 / 12,
    1.0, 1.25, 1.5, 2.0, 3.0,
], dtype=np.float64)
NK = len(LOG_MONEYNESS)
NT = len(MATURITIES)

SPOT = 1.0
Q_FIXED = 0.0

FLAG_CODE = {
    "cos_standard":    np.uint8(0),
    "cos_extended":    np.uint8(1),
    "cos_small_price": np.uint8(2),
    "unpricable":      np.uint8(3),
}
FLAG_ORDER = ["cos_standard", "cos_extended", "cos_small_price", "unpricable"]
N_PARAMS = 6


# ========================================================================
# CPU reference: scalar per-cell via pricer.price_cell
# ========================================================================

def price_batch_cpu(
    params: np.ndarray, S0: float = SPOT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Price (N, NK, NT) surfaces and flag-code array via scalar dispatcher.

    Returns (prices, flags). prices[n, ki, ti] is NaN iff flags=unpricable.
    Called side is OTM: put for log_m<0, call for log_m>=0.
    """
    N = params.shape[0]
    prices = np.full((N, NK, NT), np.nan, dtype=np.float64)
    flags = np.full((N, NK, NT), FLAG_CODE["unpricable"], dtype=np.uint8)
    K_grid = S0 * np.exp(LOG_MONEYNESS)
    is_put = LOG_MONEYNESS < 0.0
    for n in range(N):
        kappa, theta, sigma, rho, v0, r = params[n].tolist()
        for ki in range(NK):
            K = float(K_grid[ki])
            ot = "put" if is_put[ki] else "call"
            for ti in range(NT):
                T = float(MATURITIES[ti])
                p, flag, _ = price_cell(
                    kappa, theta, sigma, rho, v0, r, Q_FIXED,
                    K, T, S0, ot,
                )
                flags[n, ki, ti] = FLAG_CODE[flag]
                if flag != "unpricable":
                    prices[n, ki, ti] = p
    return prices, flags


# ========================================================================
# GPU batched: per-cell kernels, per-tier fallback
# ========================================================================

def _gpu_cos_prices_cell(
    params_t: torch.Tensor,  # (B, 5) [kappa, theta, sigma, rho, v0]
    r_t: torch.Tensor,       # (B,)
    q_scalar: float,         # Q_FIXED (scalar)
    K: float, T: float, S0: float,
    option_type: str,
    n_cos: int, L_factor: float,
    device: torch.device,
) -> torch.Tensor:
    """Batched COS price for a single (K, T) cell across B samples.

    Adaptive per-sample half_abs from simplified Heston cumulants, then
    direct-payoff COS integration at n_cos nodes. Exactly mirrors the
    scalar :func:`pricer._cos_price_scalar` / ``_heston_cf`` /
    ``_adaptive_half_abs`` at float64.

    Returns (B,) float64; NaN at samples whose CF aggregation is non-finite.
    """
    B = params_t.shape[0]
    kappa = params_t[:, 0]
    theta = params_t[:, 1]
    sigma = params_t[:, 2]
    rho = params_t[:, 3]
    v0 = params_t[:, 4]
    q_t = torch.full_like(r_t, q_scalar)

    # ---- simplified cumulants ------------------------------------------
    kT = kappa * T
    em1 = torch.where(kT < 1e-10, kT, 1.0 - torch.exp(-kT))
    integ_var = theta * T + (v0 - theta) * em1 / kappa     # (B,)
    c1 = (r_t - q_t) * T - 0.5 * integ_var                  # (B,)
    c2 = torch.clamp(integ_var, min=0.0)                    # (B,)

    x_l = math.log(S0 / K)
    half_abs = torch.clamp(
        torch.abs(c1 + x_l) + L_factor * torch.sqrt(c2),
        min=HALF_FLOOR,
    )                                                       # (B,)
    a = -half_abs                                           # (B,)
    b = half_abs                                            # (B,)
    ba = 2.0 * half_abs                                     # (B,)

    k_idx = torch.arange(n_cos, device=device, dtype=torch.float64)  # (Nc,)
    u_k = k_idx.unsqueeze(0) * math.pi / ba.unsqueeze(1)             # (B, Nc)

    cos_ub = torch.cos(u_k * b.unsqueeze(1))  # (B, Nc)
    sin_ub = torch.sin(u_k * b.unsqueeze(1))
    cos_kpi = torch.cos(k_idx * math.pi)      # (Nc,) = (-1)^k
    denom = 1.0 + u_k * u_k                   # (B, Nc)
    denom[:, 0] = 1.0

    if option_type == "call":
        exp_b = torch.exp(b).unsqueeze(1)                  # (B, 1)
        chi = (exp_b * cos_kpi.unsqueeze(0) - cos_ub - u_k * sin_ub) / denom
        chi_col0 = torch.exp(b) - 1.0                      # (B,)
        chi = torch.cat([chi_col0.unsqueeze(1), chi[:, 1:]], dim=1)
        psi_col0 = b                                        # (B,)
        psi_rest = -sin_ub[:, 1:] / u_k[:, 1:]              # (B, Nc-1)
        psi = torch.cat([psi_col0.unsqueeze(1), psi_rest], dim=1)
        Vk = (2.0 / ba).unsqueeze(1) * (chi - psi)
    else:  # put
        exp_a = torch.exp(a).unsqueeze(1)                  # (B, 1)
        chi = (cos_ub + u_k * sin_ub - exp_a) / denom
        chi_col0 = 1.0 - torch.exp(a)                      # (B,)
        chi = torch.cat([chi_col0.unsqueeze(1), chi[:, 1:]], dim=1)
        psi_col0 = -a                                       # (B,)
        psi_rest = sin_ub[:, 1:] / u_k[:, 1:]               # (B, Nc-1)
        psi = torch.cat([psi_col0.unsqueeze(1), psi_rest], dim=1)
        Vk = (2.0 / ba).unsqueeze(1) * (psi - chi)
    # k=0 half-weight
    Vk = Vk.clone()
    Vk[:, 0] = Vk[:, 0] * 0.5

    # ---- Heston CF Little-Trap (Albrecher 2007) ------------------------
    uc = u_k.to(torch.complex128)                          # (B, Nc)
    kappa_c = kappa.to(torch.complex128).unsqueeze(1)
    sigma_c = sigma.to(torch.complex128).unsqueeze(1)
    rho_c = rho.to(torch.complex128).unsqueeze(1)
    theta_c = theta.to(torch.complex128).unsqueeze(1)
    v0_c = v0.to(torch.complex128).unsqueeze(1)
    carry_c = (r_t - q_t).to(torch.complex128).unsqueeze(1)

    xi = kappa_c - 1j * rho_c * sigma_c * uc
    sig2 = sigma_c * sigma_c
    d = torch.sqrt(xi * xi + sig2 * uc * (uc + 1j))
    g = (xi - d) / (xi + d)
    e_dT = torch.exp(-d * T)
    one_m_g = 1.0 - g
    one_m_g_eT = 1.0 - g * e_dT
    log_q = torch.log(one_m_g_eT / one_m_g)
    A = 1j * carry_c * uc * T
    B_ = (kappa_c * theta_c / sig2) * ((xi - d) * T - 2.0 * log_q)
    C = (v0_c / sig2) * (xi - d) * (1.0 - e_dT) / one_m_g_eT
    phi = torch.exp(A + B_ + C)                            # (B, Nc)

    c_base = phi * torch.exp(-1j * uc * a.unsqueeze(1).to(torch.complex128))
    phase = u_k * x_l
    integrand = c_base.real * torch.cos(phase) - c_base.imag * torch.sin(phase)
    s = (integrand * Vk).sum(dim=1)                        # (B,) float64
    price = torch.exp(-r_t * T) * K * s
    price = torch.where(torch.isfinite(price), price, torch.full_like(price, float("nan")))
    return torch.clamp(price, min=0.0)


def price_chunk_gpu(
    par_t: torch.Tensor,   # (B, 5)
    r_t: torch.Tensor,     # (B,)
    K_grid: np.ndarray,
    is_put: np.ndarray,
    S0: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Price a single chunk across the full (NK, NT) grid.

    Returns (prices, flags) as torch tensors on ``device`` with shapes
    (B, NK, NT) float64 and (B, NK, NT) uint8. NaN prices where flag is
    unpricable.
    """
    B = par_t.shape[0]
    prices = torch.full((B, NK, NT), float("nan"), device=device, dtype=torch.float64)
    flags = torch.full((B, NK, NT), FLAG_CODE["unpricable"].item(),
                       device=device, dtype=torch.uint8)
    code_std = FLAG_CODE["cos_standard"].item()
    code_ext = FLAG_CODE["cos_extended"].item()
    code_small = FLAG_CODE["cos_small_price"].item()

    for ki in range(NK):
        K = float(K_grid[ki])
        ot = "put" if is_put[ki] else "call"
        for ti in range(NT):
            T = float(MATURITIES[ti])

            p_std = _gpu_cos_prices_cell(
                par_t, r_t, Q_FIXED, K, T, S0, ot,
                N_COS_STD, L_FACTOR, device,
            )
            accept_std = torch.isfinite(p_std) & ((p_std / S0) > RATIO_ACCEPT)

            need_ext = ~accept_std
            p_ext = torch.full_like(p_std, float("nan"))
            if need_ext.any():
                idx = torch.where(need_ext)[0]
                p_ext[idx] = _gpu_cos_prices_cell(
                    par_t[idx], r_t[idx], Q_FIXED, K, T, S0, ot,
                    N_COS_EXT, L_FACTOR_EXT, device,
                )
            accept_ext = need_ext & torch.isfinite(p_ext) & ((p_ext / S0) > RATIO_ACCEPT)

            candidate = torch.where(torch.isfinite(p_ext), p_ext, p_std)
            small = (
                ~(accept_std | accept_ext)
                & torch.isfinite(candidate)
                & ((candidate / S0) > RATIO_PRECISION)
            )

            cell_p = torch.full_like(p_std, float("nan"))
            cell_f = torch.full((B,), FLAG_CODE["unpricable"].item(),
                                device=device, dtype=torch.uint8)
            cell_p = torch.where(accept_std, p_std, cell_p)
            cell_f = torch.where(accept_std, torch.full_like(cell_f, code_std), cell_f)
            cell_p = torch.where(accept_ext, p_ext, cell_p)
            cell_f = torch.where(accept_ext, torch.full_like(cell_f, code_ext), cell_f)
            cell_p = torch.where(small, candidate, cell_p)
            cell_f = torch.where(small, torch.full_like(cell_f, code_small), cell_f)

            prices[:, ki, ti] = cell_p
            flags[:, ki, ti] = cell_f

    return prices, flags


def price_batch_gpu(
    params: np.ndarray,
    device: torch.device,
    S0: float = SPOT,
    chunk_size: int = 4096,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """In-memory variant (smoke test only). Full runs use ``write_surfaces``
    to stream chunks to HDF5 without materializing the full (N, NK, NT)
    array in RAM.
    """
    N = params.shape[0]
    K_grid = S0 * np.exp(LOG_MONEYNESS)
    is_put = LOG_MONEYNESS < 0.0

    prices_out = np.full((N, NK, NT), np.nan, dtype=np.float64)
    flags_out = np.full((N, NK, NT), FLAG_CODE["unpricable"], dtype=np.uint8)

    n_chunks = math.ceil(N / chunk_size)
    t0 = time.time()
    for ci in range(n_chunks):
        i0, i1 = ci * chunk_size, min(ci * chunk_size + chunk_size, N)
        par_t = torch.tensor(params[i0:i1, :5], device=device, dtype=torch.float64)
        r_t = torch.tensor(params[i0:i1, 5], device=device, dtype=torch.float64)
        p, f = price_chunk_gpu(par_t, r_t, K_grid, is_put, S0, device)
        prices_out[i0:i1] = p.cpu().numpy()
        flags_out[i0:i1] = f.cpu().numpy()
        if progress:
            elapsed = time.time() - t0
            rate = (ci + 1) * chunk_size / max(elapsed, 1e-9)
            print(f"  chunk {ci+1}/{n_chunks}  "
                  f"{(ci+1)*chunk_size:,}/{N:,}  "
                  f"{rate:,.0f} samples/s  "
                  f"elapsed={elapsed/60:.1f} min",
                  flush=True)

    return prices_out, flags_out


# ========================================================================
# IV inversion (reuse heston_datagen Newton kernel or scalar BS)
# ========================================================================

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _bs_pv(
    sigma: torch.Tensor, F: torch.Tensor, K: torch.Tensor, T: float,
    disc: torch.Tensor, is_put: torch.Tensor, sqT: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BS price + vega, batched. F, disc are (B,), K is (NK,), sigma (B, NK)."""
    F_t = F[:, None].expand_as(sigma)
    d1 = (torch.log(F_t / K[None, :]) + 0.5 * sigma * sigma * T) / (sigma * sqT + 1e-30)
    d2 = d1 - sigma * sqT
    nd1 = 0.5 * (1.0 + torch.erf(d1 * _INV_SQRT2))
    nd2 = 0.5 * (1.0 + torch.erf(d2 * _INV_SQRT2))
    call = disc[:, None] * (F_t * nd1 - K[None, :] * nd2)
    put = call + (K[None, :] * disc[:, None] - F_t * disc[:, None])
    bs = torch.where(is_put[None, :], put, call)
    vega = disc[:, None] * F_t * sqT * torch.exp(-0.5 * d1 * d1) * _INV_SQRT2PI
    return bs, vega


def prices_to_iv_gpu(
    prices_t: torch.Tensor,  # (B, NK) float64 OTM prices; NaN → NaN IV
    K_grid_t: torch.Tensor,  # (NK,) float64
    T: float, S0: float,
    r_t: torch.Tensor,       # (B,) float64
    q_scalar: float,
    is_put_t: torch.Tensor,  # (NK,) bool
    iv_lo: float = 0.005, iv_hi: float = 3.0,
    n_newton: int = 30, n_bisect: int = 60,
    tol_abs: float = 1e-10, tol_rel: float = 1e-6,
) -> torch.Tensor:
    """Robust BS IV inversion, float64, batched (B, NK). NaN propagates.

    Strategy: constant sigma=0.3 initial guess (Brenner-Subrahmanyam fails
    for OTM cells, where ``price/F * sqrt(2π/T) → 0`` collapses Newton),
    Newton with vega clamp, then per-cell bisection fallback for cells
    that did not converge. Final residual gate at tol_rel ensures
    NaN-on-fail rather than spurious converged-to-bound values.
    """
    q_t = torch.full_like(r_t, q_scalar)
    F = S0 * torch.exp((r_t - q_t) * T)            # (B,)
    disc = torch.exp(-r_t * T)                      # (B,)
    K = K_grid_t
    sqT = math.sqrt(T)

    call_int = (F[:, None] - K[None, :] * disc[:, None]).clamp(min=0.0)
    put_int = (K[None, :] * disc[:, None] - F[:, None]).clamp(min=0.0)
    intrinsic = torch.where(is_put_t[None, :], put_int, call_int)
    zero_mask = prices_t <= intrinsic + 1e-14
    nan_mask = ~torch.isfinite(prices_t)

    # ---- Newton from constant init -------------------------------------
    sigma = torch.full_like(prices_t, 0.3)
    sigma = sigma.clamp(iv_lo, iv_hi)
    for _ in range(n_newton):
        bs, vega = _bs_pv(sigma, F, K, T, disc, is_put_t, sqT)
        step = (bs - prices_t) / vega.clamp(min=1e-30)
        sigma_new = (sigma - step).clamp(iv_lo, iv_hi)
        sigma = torch.where(vega > 1e-12, sigma_new, sigma)

    # ---- mark which cells need bisection -------------------------------
    bs_after, _ = _bs_pv(sigma, F, K, T, disc, is_put_t, sqT)
    bad = (bs_after - prices_t).abs() > tol_abs * 100  # loose Newton gate
    need_bisect = bad & ~(zero_mask | nan_mask)

    if need_bisect.any():
        # Bisection: maintain lo, hi where bs(lo) < price < bs(hi).
        # BS price is monotonically increasing in sigma → standard bisection.
        lo = torch.full_like(prices_t, iv_lo)
        hi = torch.full_like(prices_t, iv_hi)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            bs_mid, _ = _bs_pv(mid, F, K, T, disc, is_put_t, sqT)
            below = bs_mid < prices_t
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        sigma_bis = 0.5 * (lo + hi)
        sigma = torch.where(need_bisect, sigma_bis, sigma)

    # ---- final residual gate ------------------------------------------
    bs_final, _ = _bs_pv(sigma, F, K, T, disc, is_put_t, sqT)
    rel = (bs_final - prices_t).abs() / (prices_t.abs() + 1e-15)
    bad_final = rel > tol_rel
    sigma = torch.where(zero_mask | nan_mask | bad_final,
                        torch.full_like(sigma, float("nan")), sigma)
    return sigma


def prices_to_iv_cpu(
    prices: np.ndarray, S0: float, params_r: np.ndarray,
) -> np.ndarray:
    """CPU Newton inversion for the smoke-test path. Matches GPU formula."""
    N, _, _ = prices.shape
    out = np.full_like(prices, np.nan)
    K_grid = S0 * np.exp(LOG_MONEYNESS)
    is_put = LOG_MONEYNESS < 0.0
    for n in range(N):
        r = float(params_r[n])
        for ti, T in enumerate(MATURITIES):
            T = float(T)
            F = S0 * math.exp(r * T)
            disc = math.exp(-r * T)
            sqT = math.sqrt(T)
            for ki in range(NK):
                K = float(K_grid[ki])
                price = float(prices[n, ki, ti])
                if not math.isfinite(price):
                    continue
                intrinsic = max((K - F) * disc, 0.0) if is_put[ki] else max((F - K) * disc, 0.0)
                if price <= intrinsic + 1e-14:
                    continue
                def bs_eval(sigma):
                    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqT + 1e-30)
                    d2 = d1 - sigma * sqT
                    nd1 = 0.5 * (1.0 + math.erf(d1 * _INV_SQRT2))
                    nd2 = 0.5 * (1.0 + math.erf(d2 * _INV_SQRT2))
                    call = disc * (F * nd1 - K * nd2)
                    put = call + (K * disc - F * disc)
                    bs = put if is_put[ki] else call
                    vega = disc * F * sqT * math.exp(-0.5 * d1 * d1) * _INV_SQRT2PI
                    return bs, vega
                sigma = 0.3
                for _ in range(30):
                    bs, vega = bs_eval(sigma)
                    if abs(bs - price) < 1e-12:
                        break
                    if vega < 1e-12:
                        break
                    s_new = sigma - (bs - price) / vega
                    if s_new <= 0.005 or s_new >= 3.0:
                        break
                    sigma = s_new
                bs, _ = bs_eval(sigma)
                if abs(bs - price) > 1e-8:  # bisection fallback
                    lo, hi = 0.005, 3.0
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        bs_m, _ = bs_eval(mid)
                        if bs_m < price:
                            lo = mid
                        else:
                            hi = mid
                    sigma = 0.5 * (lo + hi)
                    bs, _ = bs_eval(sigma)
                res = abs(bs - price) / (abs(price) + 1e-15)
                if res < 1e-6:
                    out[n, ki, ti] = sigma
    return out


# ========================================================================
# Smoke test: CPU vs GPU cross-check
# ========================================================================

def smoke_test(n_samples: int, seed: int, device: torch.device) -> None:
    """Generate N surfaces on CPU scalar + GPU batched paths, compare."""
    print(f"[smoke] sampling {n_samples} params (seed={seed})")
    params = sample_realistic_params(n_samples, seed=seed)
    print(f"[smoke] params shape={params.shape}")

    t0 = time.time()
    prices_cpu, flags_cpu = price_batch_cpu(params)
    t_cpu = time.time() - t0
    print(f"[smoke] CPU scalar:  {t_cpu:.1f}s")

    t0 = time.time()
    prices_gpu, flags_gpu = price_batch_gpu(params, device, chunk_size=n_samples)
    t_gpu = time.time() - t0
    print(f"[smoke] GPU batched: {t_gpu:.1f}s")

    # Flag agreement
    disagree = flags_cpu != flags_gpu
    n_dis = int(disagree.sum())
    print(f"[smoke] flag disagreements: {n_dis}/{flags_cpu.size}")
    if n_dis > 0:
        for f_name, f_code in FLAG_CODE.items():
            n_cpu = int((flags_cpu == f_code).sum())
            n_gpu = int((flags_gpu == f_code).sum())
            print(f"          {f_name:16s}  cpu={n_cpu:6d}  gpu={n_gpu:6d}")

    # Price agreement on co-valid cells
    both_priced = np.isfinite(prices_cpu) & np.isfinite(prices_gpu)
    if both_priced.any():
        d = prices_cpu[both_priced] - prices_gpu[both_priced]
        ref = np.maximum(np.abs(prices_cpu[both_priced]), 1e-12)
        rel = np.abs(d) / ref
        print(f"[smoke] price match on {int(both_priced.sum())} shared cells:")
        print(f"          max |abs err|/S0 = {np.abs(d).max():.2e}")
        print(f"          p95 rel err      = {np.quantile(rel, 0.95):.2e}")
        print(f"          max rel err      = {rel.max():.2e}")

    # Per-flag price breakdown
    for f_name, f_code in FLAG_CODE.items():
        if f_name == "unpricable":
            continue
        mask = (flags_cpu == f_code) & (flags_gpu == f_code)
        if not mask.any():
            continue
        d = prices_cpu[mask] - prices_gpu[mask]
        ref = np.maximum(np.abs(prices_cpu[mask]), 1e-12)
        rel = np.abs(d) / ref
        print(f"[smoke]   {f_name:16s}  n={int(mask.sum()):6d}  "
              f"max_abs={np.abs(d).max():.2e}  p95_rel={np.quantile(rel, 0.95):.2e}")

    # Gate
    gate_abs = 1e-9
    if both_priced.any():
        if np.abs(prices_cpu[both_priced] - prices_gpu[both_priced]).max() > gate_abs:
            print(f"[smoke] WARN: abs price diff > {gate_abs:.0e}  (float64 COS should match)")
        else:
            print(f"[smoke] PASS: GPU-CPU abs price diff < {gate_abs:.0e}")
    if n_dis == 0:
        print("[smoke] PASS: flag codes identical")


# ========================================================================
# Full generation → HDF5
# ========================================================================

def write_surfaces(
    out_path: Path,
    N: int, seed: int,
    device: torch.device,
    chunk_size: int,
    S0: float = SPOT,
) -> None:
    """Generate N surfaces on GPU, invert to IV, stream chunks to HDF5.

    Schema:
      params        (N, 6) float64  kappa, theta, sigma_v, rho, v0, r
      iv_surface    (N, NK, NT) float64  NaN at unpricable
      quality_mask  (N, NK, NT) uint8    {0..3} per FLAG_CODE
      pricable_region (NK, NT) bool      Phase 3.1 static mask
    """
    print(f"[gen] sampling {N:,} params (seed={seed})")
    t_start = time.time()
    params = sample_realistic_params(N, seed=seed)
    print(f"[gen] sampled in {time.time()-t_start:.1f}s")

    pricable_region_path = HERE / "pricable_region.npy"
    if pricable_region_path.exists():
        pricable = np.load(pricable_region_path)
    else:
        pricable = np.ones((NK, NT), dtype=bool)
        print("[gen] WARN: pricable_region.npy missing; using all-True")

    K_grid_np = S0 * np.exp(LOG_MONEYNESS)
    K_grid_t = torch.tensor(K_grid_np, device=device, dtype=torch.float64)
    is_put_np = LOG_MONEYNESS < 0.0
    is_put_t = torch.tensor(is_put_np, device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(N / chunk_size)
    flag_totals = {name: 0 for name in FLAG_CODE}
    t0 = time.time()

    with h5py.File(out_path, "w") as f:
        f.attrs["model_type"] = "heston"
        f.attrs["grid_preset"] = "phase3_49x14"
        f.attrs["param_names"] = PARAM_NAMES
        f.attrs["flag_order"] = FLAG_ORDER
        f.attrs["log_moneyness"] = LOG_MONEYNESS
        f.attrs["maturities"] = MATURITIES
        f.attrs["K_grid"] = K_grid_np
        f.attrs["spot"] = S0
        f.attrs["q_fixed"] = Q_FIXED
        f.attrs["N_total"] = N
        f.attrs["seed"] = seed
        f.attrs["code_version"] = "v2-phase3.2"
        f.attrs["pricer_N_COS_STD"] = N_COS_STD
        f.attrs["pricer_N_COS_EXT"] = N_COS_EXT
        f.attrs["pricer_L_FACTOR"] = L_FACTOR
        f.attrs["pricer_L_FACTOR_EXT"] = L_FACTOR_EXT

        cs_n = min(4096, N)
        ds_params = f.create_dataset("params",
            shape=(N, N_PARAMS), dtype="float64",
            chunks=(cs_n, N_PARAMS), compression="lzf")
        ds_iv = f.create_dataset("iv_surface",
            shape=(N, NK, NT), dtype="float64",
            chunks=(min(256, N), NK, NT), compression="lzf")
        ds_qm = f.create_dataset("quality_mask",
            shape=(N, NK, NT), dtype="uint8",
            chunks=(min(256, N), NK, NT), compression="lzf")
        f.create_dataset("pricable_region", data=pricable)

        for ci in range(n_chunks):
            i0, i1 = ci * chunk_size, min(ci * chunk_size + chunk_size, N)
            par_t = torch.tensor(params[i0:i1, :5], device=device, dtype=torch.float64)
            r_t = torch.tensor(params[i0:i1, 5], device=device, dtype=torch.float64)

            prices_t, flags_t = price_chunk_gpu(
                par_t, r_t, K_grid_np, is_put_np, S0, device,
            )

            iv_chunk = torch.full_like(prices_t, float("nan"))
            for ti in range(NT):
                T = float(MATURITIES[ti])
                iv_chunk[:, :, ti] = prices_to_iv_gpu(
                    prices_t[:, :, ti], K_grid_t, T, S0, r_t, Q_FIXED, is_put_t,
                )

            ds_params[i0:i1] = params[i0:i1]
            ds_iv[i0:i1] = iv_chunk.cpu().numpy()
            flags_np = flags_t.cpu().numpy()
            ds_qm[i0:i1] = flags_np

            for name, code in FLAG_CODE.items():
                flag_totals[name] += int((flags_np == code).sum())

            elapsed = time.time() - t0
            done = i1
            rate = done / max(elapsed, 1e-9)
            eta_min = (N - done) / max(rate, 1e-9) / 60.0
            print(f"  chunk {ci+1}/{n_chunks}  {done:,}/{N:,}  "
                  f"{rate:,.0f} samples/s  "
                  f"elapsed={elapsed/60:.1f} min  eta={eta_min:.1f} min",
                  flush=True)

        # Final summary
        total = N * NK * NT
        f.attrs["gen_time_s"] = time.time() - t_start
        for name in FLAG_ORDER:
            f.attrs[f"count_{name}"] = flag_totals[name]
            print(f"[gen] {name:16s}  {flag_totals[name]:>12,d}  "
                  f"({100.0*flag_totals[name]/total:5.2f}%)")

    size_gb = out_path.stat().st_size / 1e9
    print(f"[gen] wrote {out_path}  ({size_gb:.3f} GB)  "
          f"total_wall={(time.time()-t_start)/60:.1f} min")


# ========================================================================
# CLI
# ========================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="CPU vs GPU cross-check on a small N, then exit")
    p.add_argument("--smoke-n", type=int, default=32)
    p.add_argument("--N", type=int, default=10_000_000)
    p.add_argument("--out", type=str, default="data/heston_pipeline_v2.h5")
    p.add_argument("--seed", type=int, default=20260419)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--val", action="store_true",
                   help="Run validation set: N=1M, out=data/heston_pipeline_v2_val.h5, seed+1")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[main] device={device}")

    if args.smoke:
        smoke_test(args.smoke_n, args.seed, device)
        return 0

    if args.val:
        N = 1_000_000
        out_path = Path("data/heston_pipeline_v2_val.h5")
        seed = args.seed + 1
    else:
        N = args.N
        out_path = Path(args.out)
        seed = args.seed

    write_surfaces(out_path, N=N, seed=seed, device=device, chunk_size=args.chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())

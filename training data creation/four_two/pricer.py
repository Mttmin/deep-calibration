"""COS pricer for the 4/2 model.

The Kummer series dominates the cost and does not depend on strike, but the
truncation window does. Each sample uses the widest window across the strike
grid at that maturity, so one characteristic function covers every strike.
``N_COS`` is sized for that wider window.

Quality flags match the Heston v2 generator:
    0 cos_standard, 1 cos_extended, 2 cos_small_price, 3 unpricable.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from cf import four_two_cf

N_COS_STD = 768
N_COS_EXT = 2048
RATIO_ACCEPT = 1e-4
RATIO_PRECISION = 1e-6
L_FACTOR = 10.0
L_FACTOR_EXT = 15.0
HALF_FLOOR = 0.5
Q_FIXED = 0.0
SPOT = 1.0

LOG_MONEYNESS = np.linspace(-0.80, 0.40, 49, dtype=np.float64)
MATURITIES = np.array(
    [
        1 / 52, 2 / 52, 3 / 52,
        1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 9 / 12,
        1.0, 1.25, 1.5, 2.0, 3.0,
    ],
    dtype=np.float64,
)
NK = len(LOG_MONEYNESS)
NT = len(MATURITIES)

FLAG_CODE = {
    "cos_standard": np.uint8(0),
    "cos_extended": np.uint8(1),
    "cos_small_price": np.uint8(2),
    "unpricable": np.uint8(3),
}


def integrated_variance(
    T: float, kappa: float, theta: float, sigma: float, v0: float, a: float, b: float,
) -> Tuple[float, float]:
    """Return ``(driftless_mean, c2)`` for ``log(S_T/S_0)`` excluding ``(r-q)T``.

    ``c2`` is ``E[int (a sqrt(v) + b/sqrt(v))^2 dt]``. Vol-of-vol and
    correlation fatten the true variance; the COS window multiplies ``sqrt(c2)``
    by ``L``, the same margin the Heston generator used.
    """
    kT = kappa * T
    em1 = kT if kT < 1e-10 else 1.0 - math.exp(-kT)
    integ_v = theta * T + (v0 - theta) * em1 / kappa
    integ_inv = _integ_inv_v(T, kappa, theta, sigma, v0)
    c2 = a * a * integ_v + 2.0 * a * b * T + b * b * integ_inv
    return -0.5 * c2, max(c2, 0.0)


def _integ_inv_v(T: float, kappa: float, theta: float, sigma: float, v0: float) -> float:
    if T <= 0.0:
        return 0.0
    ts = np.linspace(0.0, T, 16)
    vals = np.array([_cir_inv_mean(float(t), kappa, theta, sigma, v0) for t in ts])
    return float(np.trapezoid(vals, ts))


def _cir_inv_mean(t: float, kappa: float, theta: float, sigma: float, v0: float) -> float:
    """``E[1/v_t]`` for a Feller-satisfying CIR process."""
    if t < 1e-6:
        return 1.0 / v0
    sig2 = sigma * sigma
    decay = math.exp(-kappa * t)
    c = sig2 * (1.0 - decay) / (4.0 * kappa)
    if c < 1e-18:
        return 1.0 / v0
    lam = v0 * decay / c
    delta = 4.0 * kappa * theta / sig2
    if delta <= 2.05:
        return 1.0 / max(min(v0, theta), 1e-6)
    half_lam = 0.5 * lam
    term = math.exp(-half_lam)
    acc = term / (delta - 2.0)
    for j in range(1, 80):
        term *= half_lam / j
        acc += term / (delta - 2.0 + 2.0 * j)
        if term < 1e-14 * (abs(acc) + 1e-30):
            break
    return acc / c


def half_abs_for_strikes(c1: float, c2: float, x_l: np.ndarray, L_factor: float) -> float:
    width = np.abs(c1 + x_l) + L_factor * math.sqrt(max(c2, 0.0))
    return float(max(HALF_FLOOR, float(np.max(width))))


def cos_prices_numpy(
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    a: float, b: float, r: float, q: float,
    K: np.ndarray, T: float, spot: float,
    is_put: np.ndarray, n_cos: int, L_factor: float,
) -> np.ndarray:
    """OTM prices for one parameter row and one maturity, all strikes."""
    x_l = np.log(spot / K)
    c1_var, c2 = integrated_variance(T, kappa, theta, sigma, v0, a, b)
    c1 = (r - q) * T + c1_var
    half = half_abs_for_strikes(c1, c2, x_l, L_factor)
    a_tr, b_tr = -half, half
    ba = b_tr - a_tr
    u = np.arange(n_cos, dtype=np.float64) * math.pi / ba
    phi = four_two_cf(u, T, kappa, theta, sigma, rho, v0, a, b, r, q)
    if not np.isfinite(phi).all():
        return np.full(K.shape, np.nan)
    c_base = phi * np.exp(-1j * u * a_tr)
    Vk = _payoff_coeffs(u, a_tr, b_tr, is_put)
    phase = u[None, :] * x_l[:, None]
    shifted = c_base[None, :] * (np.cos(phase) + 1j * np.sin(phase))
    total = (shifted.real * Vk).sum(axis=-1)
    price = math.exp(-r * T) * K * total
    return np.where(np.isfinite(price), np.maximum(price, 0.0), np.nan)


def _payoff_coeffs(u: np.ndarray, a_tr: float, b_tr: float, is_put: np.ndarray) -> np.ndarray:
    """Cosine coefficients of the unit-strike payoff. Shape ``(NK, Nc)``."""
    ba = b_tr - a_tr
    k_idx = np.arange(u.size, dtype=np.float64)
    cos_ub = np.cos(u * b_tr)
    sin_ub = np.sin(u * b_tr)
    cos_kpi = np.cos(k_idx * math.pi)
    denom = 1.0 + u * u
    denom = denom.copy()
    denom[0] = 1.0

    exp_b = math.exp(b_tr)
    chi_call = (exp_b * cos_kpi - cos_ub - u * sin_ub) / denom
    chi_call[0] = exp_b - 1.0
    psi_call = np.zeros_like(u)
    psi_call[0] = b_tr
    psi_call[1:] = -sin_ub[1:] / u[1:]
    vk_call = (2.0 / ba) * (chi_call - psi_call)

    exp_a = math.exp(a_tr)
    chi_put = (cos_ub + u * sin_ub - exp_a) / denom
    chi_put[0] = 1.0 - exp_a
    psi_put = np.zeros_like(u)
    psi_put[0] = -a_tr
    psi_put[1:] = sin_ub[1:] / u[1:]
    vk_put = (2.0 / ba) * (psi_put - chi_put)

    vk_call[0] *= 0.5
    vk_put[0] *= 0.5
    return np.where(is_put[:, None], vk_put[None, :], vk_call[None, :])


# ---------------------------------------------------------------------------
# Torch. Parameters (B,), frequencies (B, Nc) because each row has its own
# truncation. One CF per maturity, reused across strikes.
# ---------------------------------------------------------------------------

_STIRLING_SHIFT = 12


def _loggamma_complex(z: torch.Tensor) -> torch.Tensor:
    """Complex log-gamma. Shifts ``Re(z)`` to ``>= 8``, then 4-term Stirling."""
    acc = torch.zeros_like(z)
    for _ in range(_STIRLING_SHIFT):
        need = z.real < 8.0
        acc = acc + torch.where(need, torch.log(z), torch.zeros_like(z))
        z = torch.where(need, z + 1.0, z)
    log_z = torch.log(z)
    inv = 1.0 / z
    inv3 = inv * inv * inv
    inv5 = inv3 * inv * inv
    inv7 = inv5 * inv * inv
    series = (z - 0.5) * log_z - z + 0.5 * math.log(2.0 * math.pi)
    series = series + inv / 12.0 - inv3 / 360.0 + inv5 / 1260.0 - inv7 / 1680.0
    return series - acc


def _sqrt_pos_real(z: torch.Tensor) -> torch.Tensor:
    s = torch.sqrt(z)
    return torch.where(s.real < 0, -s, s)


def cf_torch(
    u: torch.Tensor,
    T: float,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
    v0: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    r: torch.Tensor,
    q: float,
    n_terms: int | None = None,
) -> torch.Tensor:
    """CF of ``log(S_T/S_0)``. ``u`` is ``(B, Nc)`` float64. Returns complex128."""
    uc = u.to(torch.complex128)
    iu = 1j * uc

    def col(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.complex128).unsqueeze(1)

    sig, kap, th = col(sigma), col(kappa), col(theta)
    rh, v, aa, bb = col(rho), col(v0), col(a), col(b)
    kth = kap * th - 0.5 * sig * sig
    psi = 0.5 * (iu + uc * uc * (1.0 - rh * rh))
    mu = psi * aa * aa - iu * rh * aa * kap / sig
    eta = psi * bb * bb + iu * rh * bb * kth / sig
    lam = -iu * rh * aa / sig
    gam = iu * rh * bb / sig
    const = (
        -2.0 * aa * bb * psi * T
        - iu * rh * (aa / sig) * (v + kap * th * T)
        + iu * rh * (bb / sig) * (kap * T - torch.log(v))
    )
    kt = _sqrt_pos_real(kap * kap + 2.0 * sig * sig * mu)
    k1 = (kt - kap) / (sig * sig)
    d_dim = _sqrt_pos_real(kth * kth + 2.0 * eta * sig * sig)
    b1 = (d_dim - kth) / (sig * sig)
    log_c = torch.log(0.25 * sig * sig) + torch.log(-torch.expm1(-kt * T)) - torch.log(kt)
    half_d = 1.0 + 2.0 * d_dim / (sig * sig)
    zeta = v * torch.exp(-kt * T - log_c)
    one_2s = 1.0 + 2.0 * (lam - k1) * torch.exp(log_c)
    p_pow = gam - b1
    series_a = half_d + p_pow
    z = zeta / (2.0 * one_2s)

    if n_terms is None:
        zmax = float(z.detach().abs().max().item()) if z.numel() else 0.0
        n_terms = int(math.ceil(zmax + 12.0 * math.sqrt(max(zmax, 0.0)) + 40.0))
        n_terms = min(max(n_terms, 8), 320)

    log_z = torch.log(z)
    log_term = -0.5 * zeta
    scale = log_term.real.clone()
    series = torch.exp(log_term - scale)
    for j in range(n_terms):
        log_term = (
            log_term
            + torch.log(series_a + j)
            - torch.log(half_d + j)
            + log_z
            - math.log(j + 1.0)
        )
        new_scale = torch.maximum(scale, log_term.real)
        series = series * torch.exp(scale - new_scale) + torch.exp(log_term - new_scale)
        scale = new_scale

    log_moment = (
        p_pow * (log_c + math.log(2.0))
        - series_a * torch.log(one_2s)
        + _loggamma_complex(series_a)
        - _loggamma_complex(half_d)
        + scale
        + torch.log(series)
    )
    phi_fwd = torch.exp(
        const - k1 * (v + kap * th * T) - b1 * (kt * T - torch.log(v)) + log_moment
    )
    carry = r.to(torch.complex128).unsqueeze(1) - q
    return torch.exp(1j * uc * carry * T) * phi_fwd


def _cir_moments_torch(
    T: float, kappa, theta, sigma, v0, a, b, r, q: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kT = kappa * T
    em1 = torch.where(kT < 1e-10, kT, 1.0 - torch.exp(-kT))
    integ_v = theta * T + (v0 - theta) * em1 / kappa
    n = 8
    ts = torch.linspace(0.0, T, n, device=kappa.device, dtype=kappa.dtype)
    inv = torch.stack(
        [_cir_inv_mean_torch(float(t), kappa, theta, sigma, v0) for t in ts], dim=0,
    )
    integ_inv = torch.trapezoid(inv, dx=T / (n - 1), dim=0)
    c2 = (a * a * integ_v + 2.0 * a * b * T + b * b * integ_inv).clamp(min=0.0)
    c1 = (r - q) * T - 0.5 * c2
    return c1, c2


def _cir_inv_mean_torch(t: float, kappa, theta, sigma, v0) -> torch.Tensor:
    if t < 1e-6:
        return 1.0 / v0
    sig2 = sigma * sigma
    decay = torch.exp(-kappa * t)
    c = sig2 * (1.0 - decay) / (4.0 * kappa)
    lam = v0 * decay / c.clamp(min=1e-18)
    delta = 4.0 * kappa * theta / sig2
    half_lam = 0.5 * lam
    term = torch.exp(-half_lam)
    acc = term / (delta - 2.0).clamp(min=1e-3)
    for j in range(1, 40):
        term = term * half_lam / j
        acc = acc + term / (delta - 2.0 + 2.0 * j)
    out = acc / c.clamp(min=1e-18)
    return torch.where(delta <= 2.05, 1.0 / v0, out)


def _payoff_coeffs_torch(
    u: torch.Tensor, a_tr: torch.Tensor, b_tr: torch.Tensor, is_put: torch.Tensor,
) -> torch.Tensor:
    """``(B, NK, Nc)`` payoff coefficients. ``u`` is ``(B, Nc)``."""
    ba = (b_tr - a_tr).unsqueeze(1)  # (B, 1)
    # k index from the first row's spacing is not shared. Rebuild (-1)^k from
    # the frequency: u * ba / pi = k, so cos(k pi) = cos(u * ba).
    cos_kpi = torch.cos(u * ba)
    cos_ub = torch.cos(u * b_tr.unsqueeze(1))
    sin_ub = torch.sin(u * b_tr.unsqueeze(1))
    denom = 1.0 + u * u
    denom = denom.clone()
    denom[:, 0] = 1.0

    exp_b = torch.exp(b_tr).unsqueeze(1)
    chi_call = (exp_b * cos_kpi - cos_ub - u * sin_ub) / denom
    chi_call[:, 0] = torch.exp(b_tr) - 1.0
    psi_call = torch.zeros_like(u)
    psi_call[:, 0] = b_tr
    psi_call[:, 1:] = -sin_ub[:, 1:] / u[:, 1:]
    vk_call = (2.0 / ba) * (chi_call - psi_call)

    exp_a = torch.exp(a_tr).unsqueeze(1)
    chi_put = (cos_ub + u * sin_ub - exp_a) / denom
    chi_put[:, 0] = 1.0 - torch.exp(a_tr)
    psi_put = torch.zeros_like(u)
    psi_put[:, 0] = -a_tr
    psi_put[:, 1:] = sin_ub[:, 1:] / u[:, 1:]
    vk_put = (2.0 / ba) * (psi_put - chi_put)

    vk_call[:, 0] *= 0.5
    vk_put[:, 0] *= 0.5
    vk = torch.where(is_put.view(1, -1, 1), vk_put.unsqueeze(1), vk_call.unsqueeze(1))
    return vk


def cos_prices_torch(
    kappa, theta, sigma, rho, v0, a, b, r,
    K: torch.Tensor,
    is_put: torch.Tensor,
    T: float,
    spot: float,
    n_cos: int,
    L_factor: float,
    q: float = Q_FIXED,
) -> torch.Tensor:
    """Prices ``(B, NK)`` for one maturity. NaN where the sum is non-finite."""
    device = kappa.device
    x_l = torch.log(torch.tensor(spot, device=device, dtype=torch.float64) / K)
    c1, c2 = _cir_moments_torch(T, kappa, theta, sigma, v0, a, b, r, q)
    width = (c1.unsqueeze(1) + x_l.unsqueeze(0)).abs() + L_factor * torch.sqrt(c2).unsqueeze(1)
    half = torch.clamp(width.max(dim=1).values, min=HALF_FLOOR)
    a_tr, b_tr = -half, half
    ba = 2.0 * half
    k_idx = torch.arange(n_cos, device=device, dtype=torch.float64)
    u = k_idx.unsqueeze(0) * math.pi / ba.unsqueeze(1)
    phi = cf_torch(u, T, kappa, theta, sigma, rho, v0, a, b, r, q)
    c_base = phi * torch.exp(-1j * u * a_tr.unsqueeze(1))
    Vk = _payoff_coeffs_torch(u, a_tr, b_tr, is_put)
    phase = u.unsqueeze(1) * x_l.view(1, -1, 1)
    integrand = (
        c_base.real.unsqueeze(1) * torch.cos(phase)
        - c_base.imag.unsqueeze(1) * torch.sin(phase)
    )
    total = (integrand * Vk).sum(dim=-1)
    price = torch.exp(-r * T).unsqueeze(1) * K.unsqueeze(0) * total
    return torch.where(torch.isfinite(price), price.clamp(min=0.0), torch.full_like(price, float("nan")))

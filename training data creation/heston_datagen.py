"""
Bates (SVJ) IV surface generator — high-resolution grid.

Usage
-----
  python heston_datagen.py                         # 2M train set
  python heston_datagen.py --N 20000000            # 20M train set
  python heston_datagen.py --val                   # 200k val set
  python heston_datagen.py --check                 # sanity only
  python heston_datagen.py --chunk 2048            # lower VRAM (f64 is ~2x memory)
"""
from __future__ import annotations
import math, argparse, time
from pathlib import Path

import numpy as np
import torch
import h5py
from scipy.stats import qmc
from tqdm import tqdm


#  Grid presets — can switch from CLI via --grid

GRID_PRESETS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "base25x10": (
        np.array([
            -0.50, -0.40, -0.30, -0.25, -0.20, -0.15, -0.12, -0.10, -0.08, -0.06, -0.04,
            -0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
            0.40, 0.50,
        ], dtype=np.float64),
        np.array([1 / 52, 2 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0],
                 dtype=np.float64),
    ),
    # Higher resolution for richer smile/skew structure.
    "high41x14": (
        np.linspace(-0.80, 0.40, 49, dtype=np.float64),
        np.array([
            1 / 52, 2 / 52, 3 / 52,
            1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 9 / 12,
            1.0, 1.25, 1.5, 2.0, 3.0,
        ], dtype=np.float64),
    ),
}
DEFAULT_GRID_PRESET = "high41x14"


def _grid_from_preset(name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        log_m, mats = GRID_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown grid preset: {name}") from exc
    return log_m.copy(), mats.copy()


LOG_MONEYNESS, MATURITIES = _grid_from_preset(DEFAULT_GRID_PRESET)
NK: int = len(LOG_MONEYNESS)
NT: int = len(MATURITIES)
IS_PUT_SIDE: np.ndarray = LOG_MONEYNESS < 0.0
GRID_PRESET_NAME = DEFAULT_GRID_PRESET


def set_grid_preset(name: str) -> None:
    global LOG_MONEYNESS, MATURITIES, NK, NT, IS_PUT_SIDE, GRID_PRESET_NAME
    LOG_MONEYNESS, MATURITIES = _grid_from_preset(name)
    NK = len(LOG_MONEYNESS)
    NT = len(MATURITIES)
    IS_PUT_SIDE = LOG_MONEYNESS < 0.0
    GRID_PRESET_NAME = name


#  Parameter space — Bates (SVJ) = Heston + Merton log-normal jumps

PARAM_BOUNDS: np.ndarray = np.array([
    [0.10, 10.0],    # κ    mean-reversion speed
    [0.02,  0.25],   # θ    long-run variance  (σ_LR ∈ [14%, 50%])
    [0.05,  2.00],   # σ_v  vol-of-vol (wider to permit stronger short-tenor convexity)
    [-0.98,  0.10],  # ρ    equity skew almost always negative
    [0.02,  0.25],   # v₀   initial variance  (σ₀  ∈ [14%, 50%])
    [0.00,  4.00],   # λ_J  jump intensity (0 recovers pure Heston)
    [-0.30,  0.00],  # μ_J  mean log-jump size (negative = crash bias)
    [0.01,  0.45],   # σ_J  jump size volatility
], dtype=np.float64)
PARAM_NAMES = ["kappa", "theta", "sigma_v", "rho", "v0",
               "lambda_j", "mu_j", "sigma_j"]
N_PARAMS: int = len(PARAM_NAMES)  # 8


#  COS / IV hyperparameters

N_COS:    int   = 256     # Cosine terms; 256 gives < 0.0001% error in float64
HALF_ABS: float = 3.0     # ABSOLUTE truncation [a,b]=[-3,+3] for ALL maturities.
                          # L*sqrt(T) with L=12 gives b=17 at T=2Y → e^17=2.4e7
                          # → catastrophic cancellation in chi. Fixed width avoids this.
N_NEWTON: int   = 20      # Newton-Raphson IV steps; converges in ~6 at f64 precision
IV_LO:    float = 1e-6
IV_HI:    float = 10.0
CHUNK:    int   = 4_096   # GPU batch size; f64 uses 2x memory vs f32


def fill_nan_market_consistent(iv_chunk: torch.Tensor) -> torch.Tensor:
    """
    Fill NaNs in IV surfaces while preserving key no-arbitrage structure.

    Steps:
      1) For each maturity, fill missing strikes via linear interpolation between
         nearest valid left/right strikes; edge gaps use nearest valid value.
      2) Enforce non-decreasing total variance over maturity for each strike:
           w(k, T) = sigma(k, T)^2 * T,  w(k, T_{i+1}) >= w(k, T_i)
         via cumulative maximum in T.
    """
    iv = iv_chunk.clone()  # (B, NK, NT) float64
    B, n_k, n_t = iv.shape
    idx = torch.arange(n_k, device=iv.device, dtype=torch.int64)[None, :].expand(B, n_k)

    for ti in range(n_t):
        col = iv[:, :, ti]                 # (B, NK)
        valid = ~torch.isnan(col)          # (B, NK)

        # Nearest valid index on the left for each cell.
        left_idx = torch.where(valid, idx, torch.zeros_like(idx))
        left_idx = torch.cummax(left_idx, dim=1).values
        has_left = torch.cumsum(valid.to(torch.int32), dim=1) > 0

        # Nearest valid index on the right for each cell.
        rev_valid = torch.flip(valid, dims=[1])
        rev_idx = torch.flip(idx, dims=[1])
        right_idx_rev = torch.where(rev_valid, rev_idx, torch.full_like(rev_idx, n_k - 1))
        right_idx_rev = torch.cummin(right_idx_rev, dim=1).values
        right_idx = torch.flip(right_idx_rev, dims=[1])
        has_right = torch.flip(torch.cumsum(rev_valid.to(torch.int32), dim=1) > 0, dims=[1])

        left_val = torch.gather(col, 1, left_idx)
        right_val = torch.gather(col, 1, right_idx)

        left_dist = (idx - left_idx).to(torch.float64)
        right_dist = (right_idx - idx).to(torch.float64)
        denom = (left_dist + right_dist).clamp(min=1e-12)
        interp = (right_val * left_dist + left_val * right_dist) / denom

        nearest = torch.where(has_left, left_val, right_val)
        fill_val = torch.where(has_left & has_right, interp, nearest)
        fill_val = torch.where(
            has_left | has_right,
            fill_val,
            torch.full_like(fill_val, IV_LO),
        )

        iv[:, :, ti] = torch.where(valid, col, fill_val)

    iv = iv.clamp(min=IV_LO, max=IV_HI)

    # Calendar consistency in total variance.
    T = torch.tensor(MATURITIES, device=iv.device, dtype=torch.float64)[None, None, :]
    total_var = iv**2 * T
    total_var = torch.cummax(total_var, dim=2).values
    iv = torch.sqrt(total_var / T)

    return iv.clamp(min=IV_LO, max=IV_HI)


#  Parameter sampling — scrambled Sobol (no Feller rejection)

def sample_params(N: int, seed: int = 42) -> np.ndarray:
    """
    Returns (N, 8) float64 Bates parameters via scrambled Sobol sequence.

    Feller condition is NOT enforced: real markets frequently violate it,
    and the COS method still produces valid prices. Degenerate samples
    (excessive NaN) are naturally handled by the cell mask in training.

    Sobol oversampled 1.5× to ensure N samples after rounding to power of 2.
    Then a stress-regime mixture boosts short-tenor skew by forcing a fraction
    of samples into high-jump / negative-correlation regions.
    """
    lo, hi  = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    sampler = qmc.Sobol(d=N_PARAMS, scramble=True, seed=seed) # type: ignore
    n_raw   = int(2 ** math.ceil(math.log2(int(N * 1.5))))
    raw     = sampler.random(n_raw)
    pts     = qmc.scale(raw, lo, hi)   # float64
    pts     = pts[:N]

    # Mixture prior: dedicate a chunk of samples to stressed skew regimes.
    rng = np.random.default_rng(seed + 2026)
    stress_prob = 0.35
    stress = rng.random(N) < stress_prob
    n_stress = int(stress.sum())
    if n_stress > 0:
        pts[stress, 2] = rng.uniform(0.20, 2.00, size=n_stress)    # sigma_v
        pts[stress, 3] = rng.uniform(-0.98, -0.55, size=n_stress)   # rho
        pts[stress, 5] = rng.uniform(1.00, 4.00, size=n_stress)     # lambda_j
        pts[stress, 6] = rng.uniform(-0.30, -0.06, size=n_stress)   # mu_j
        pts[stress, 7] = rng.uniform(0.10, 0.45, size=n_stress)     # sigma_j

    κ, θ, σ = pts[:, 0], pts[:, 1], pts[:, 2]
    feller_pct = ((2.0 * κ * θ) >= σ**2).mean() * 100

    print(
        f"[sample] Sobol draws: {n_raw:,}  |  "
        f"Feller pass: {feller_pct:.1f}% (not filtered)  |  "
        f"stress regime: {100.0 * n_stress / N:.1f}%  |  keeping: {N:,}"
    )
    return pts   # (N, 8) float64


#  Bates CF — Heston (Little Trap) + Merton log-normal jumps, float64 / complex128

def bates_cf(
    u:      torch.Tensor,   # (Nc,)  real frequencies, float64
    T:      float,
    params: torch.Tensor,   # (B, 8) float64  [κ, θ, σ_v, ρ, v₀, λ_J, μ_J, σ_J]
    r:      float,
    q:      float,
) -> torch.Tensor:           # (B, Nc) complex128
    """
    φ(u) = E_Q[e^{iu·ln(S_T/S_0)}] under Bates (1996) SVJ model.

    Heston diffusive component (Little Trap, Albrecher et al. 2007):
        ξ  = κ − iuρσ_v
        d  = √(ξ² + σ_v²·u(u+i))
        g  = (ξ−d) / (ξ+d)
        logQ = ln((1−g·e^{−dT}) / (1−g))   ← branch-stable form

    Merton log-normal jump component (additive in log-CF):
        jump = λ·T·(exp(iu·μ_J − ½σ_J²·u²) − 1 − iu·(exp(μ_J + ½σ_J²) − 1))

    When λ_J = 0 this reduces to pure Heston.
    """
    κ    = params[:, 0:1]   # (B, 1)
    θ    = params[:, 1:2]
    σv   = params[:, 2:3]
    ρ    = params[:, 3:4]
    v0   = params[:, 4:5]
    lam  = params[:, 5:6]   # jump intensity
    mu_j = params[:, 6:7]   # mean log-jump
    sig_j = params[:, 7:8]  # jump vol

    uc   = u.to(dtype=torch.complex128)          # (Nc,)

    # ── Heston diffusive component ──
    ξ    = κ - 1j * ρ * σv * uc                 # (B, Nc)
    d    = torch.sqrt(ξ**2 + σv**2 * uc * (uc + 1j))
    g    = (ξ - d) / (ξ + d)
    e_dT = torch.exp(-d * T)
    logQ = torch.log((1.0 - g * e_dT) / (1.0 - g))

    A = 1j * uc * (r - q) * T
    B = (κ * θ / σv**2) * ((ξ - d) * T - 2.0 * logQ)
    C = (v0  / σv**2)   * (ξ - d) * (1.0 - e_dT) / (1.0 - g * e_dT)

    # ── Merton log-normal jump component ──
    compensator = torch.exp(mu_j + 0.5 * sig_j**2) - 1.0
    jump = lam * T * (
        torch.exp(1j * uc * mu_j - 0.5 * sig_j**2 * uc**2)
        - 1.0
        - 1j * uc * compensator
    )

    return torch.exp(A + B + C + jump)   # (B, Nc) complex128


#  COS call payoff coefficients V_k / K  (precomputed once, float64)

def cos_call_Vk(dev: torch.device) -> torch.Tensor:   # (Nc,) float64
    """
    V_k / K = (2/(b−a)) · [χ_k(0,b) − ψ_k(0,b)]
    K factors out; integral runs from y=0 to y=b (call payoff support).
    Half-weight on k=0 included here.
    """
    a, b  = -HALF_ABS, HALF_ABS
    ba    = b - a
    k     = torch.arange(N_COS, device=dev, dtype=torch.float64)
    alpha = k * math.pi / ba
    ang0  = -alpha * a

    cos0   = torch.cos(ang0)
    sin0   = torch.sin(ang0)
    coskpi = torch.cos(k * math.pi)

    denom    = (1.0 + alpha**2).clone(); denom[0] = 1.0
    chi      = (math.exp(b) * coskpi - cos0 - alpha * sin0) / denom
    chi[0]   = math.exp(b) - 1.0

    psi      = torch.zeros(N_COS, device=dev, dtype=torch.float64)
    psi[0]   = b
    psi[1:]  = -sin0[1:] / alpha[1:]

    Vk       = (2.0 / ba) * (chi - psi)
    Vk[0]   *= 0.5
    return Vk


#  Vectorised COS call pricer — one maturity, B params × NK strikes

def cos_call_prices(
    params:  torch.Tensor,   # (B, 8) float64
    Vk:      torch.Tensor,   # (Nc,)  float64
    T:       float,
    S0:      float,
    K_grid:  torch.Tensor,   # (NK,)  float64
    r:       float,
    q:       float,
    k_idx:   torch.Tensor,   # (Nc,)  int
) -> torch.Tensor:            # (B, NK) float64
    """
    GEMM decomposition: no (B, NK, Nc) tensor ever materialised.
        Re[φ·e^{−iu_k a}·e^{iu_k x_l}]·Vk = c.real @ W_cos − c.imag @ W_sin
    where c_base = φ·e^{−iu_k a} is (B,Nc) complex, W_{cos,sin} are (Nc,NK).
    Memory: O(B·Nc + Nc·NK) not O(B·Nc·NK).
    """
    a, b  = -HALF_ABS, HALF_ABS
    ba    = b - a
    u_k   = k_idx.double() * math.pi / ba           # (Nc,) float64

    phi     = bates_cf(u_k, T, params, r, q)        # (B, Nc) complex128
    phase_a = torch.exp(-1j * u_k.to(torch.complex128) * a)
    c_base  = phi * phase_a[None, :]                 # (B, Nc)

    x_l       = torch.log(torch.tensor(S0, device=params.device, dtype=torch.float64) / K_grid)
    phase_mat = u_k[None, :] * x_l[:, None]         # (NK, Nc)

    W_cos = (torch.cos(phase_mat) * Vk[None, :]).T  # (Nc, NK)
    W_sin = (torch.sin(phase_mat) * Vk[None, :]).T  # (Nc, NK)

    price_sum = c_base.real @ W_cos - c_base.imag @ W_sin   # (B, NK) float64
    prices    = math.exp(-r * T) * K_grid[None, :] * price_sum
    return prices.clamp(min=0.0)


#  OTM conversion via put-call parity

def call_to_otm(
    call_prices: torch.Tensor,   # (B, NK) float64
    K_grid:      torch.Tensor,   # (NK,)   float64
    T:           float,
    S0:          float,
    r:           float,
    q:           float,
    is_put:      torch.Tensor,   # (NK,)   bool
) -> torch.Tensor:                # (B, NK) float64
    disc  = math.exp(-r * T)
    fwd_d = S0 * math.exp(-q * T)
    put   = (call_prices + (K_grid * disc - fwd_d)[None, :]).clamp(min=0.0)
    return torch.where(is_put[None, :], put, call_prices)


#  BS price + vega, float64

_INV_SQRT2   = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _ncdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT2))


def _bs_price_vega(
    sigma:  torch.Tensor,   # (B, NK) float64
    F:      float,
    K:      torch.Tensor,   # (NK,)   float64
    T:      float,
    r:      float,
    is_put: torch.Tensor,   # (NK,)   bool
) -> tuple[torch.Tensor, torch.Tensor]:
    sqT  = math.sqrt(T); disc = math.exp(-r * T)
    F_t  = torch.full_like(sigma, F)
    d1   = (torch.log(F_t / K) + 0.5 * sigma**2 * T) / (sigma * sqT + 1e-30)
    d2   = d1 - sigma * sqT
    call = disc * (F_t * _ncdf(d1) - K * _ncdf(d2))
    put  = call + (K * disc - F_t * disc)
    vega = disc * F_t * sqT * torch.exp(-0.5 * d1**2) * _INV_SQRT2PI
    return torch.where(is_put[None, :], put, call), vega


#  IV inversion — Newton-Raphson, GPU-batched, float64

def prices_to_iv(
    otm:    torch.Tensor,   # (B, NK) float64 OTM prices
    K_grid: torch.Tensor,   # (NK,)   float64
    T:      float,
    S0:     float,
    r:      float,
    q:      float,
    is_put: torch.Tensor,   # (NK,)   bool
) -> torch.Tensor:           # (B, NK) float64, NaN where non-invertible
    """
    Newton-Raphson in float64 with Brenner-Subrahmanyam initial guess.
    20 iterations: converges to machine precision (~1e-14) in 6-8 steps.
    NaN flagged when price <= intrinsic + 1e-12 or residual > 1e-6.
    """
    F    = S0 * math.exp((r - q) * T)
    disc = math.exp(-r * T)

    call_int  = (F - K_grid * disc).clamp(min=0.0)
    put_int   = (K_grid * disc - F).clamp(min=0.0)
    intrinsic = torch.where(is_put, put_int, call_int)

    zero_mask = otm <= intrinsic[None, :] + 1e-12

    sigma = (otm / F) * math.sqrt(2.0 * math.pi / T)
    sigma = sigma.clamp(IV_LO, IV_HI)

    for _ in range(N_NEWTON):
        bs_p, v = _bs_price_vega(sigma, F, K_grid, T, r, is_put)
        sigma   = (sigma - (bs_p - otm) / v.clamp(min=1e-30)).clamp(IV_LO, IV_HI)

    bs_final, _ = _bs_price_vega(sigma, F, K_grid, T, r, is_put)
    bad = (bs_final - otm).abs() / (otm.abs() + 1e-15) > 1e-6

    sigma[zero_mask | bad] = float("nan")
    return sigma


#  Sanity check — compare GPU f64 COS vs numpy f64 reference

def sanity_check(dev: torch.device) -> None:
    """
    Two-part sanity check:
      1. Bates with λ=0 vs pure Heston numpy reference (regression test)
      2. Bates with jumps vs numpy Bates reference

    Expects < 0.001% price error (f64 truncation vs analytical).
    """
    print("\n[sanity] Part 1: Bates(λ=0) vs pure Heston numpy reference ...")

    def heston_cf_np(u, T, kappa, theta, sv, rho, v0, r=0., q=0.):
        xi=kappa-1j*rho*sv*u; d=np.sqrt(xi**2+sv**2*u*(u+1j))
        g=(xi-d)/(xi+d); e_dT=np.exp(-d*T); logQ=np.log((1-g*e_dT)/(1-g))
        return np.exp(1j*u*(r-q)*T + (kappa*theta/sv**2)*((xi-d)*T-2*logQ)
                      + (v0/sv**2)*(xi-d)*(1-e_dT)/(1-g*e_dT))

    def cos_f64_np(S0, K, T, kappa, theta, sv, rho, v0, r=0., q=0.,
                   lam=0., mu_j=0., sig_j=0.1, N=256, H=3.0):
        a,b=-H,H; ba=b-a; k=np.arange(N,dtype=float); u_k=k*math.pi/ba
        # Heston CF
        phi=heston_cf_np(u_k,T,kappa,theta,sv,rho,v0,r,q)
        # Merton jump CF (multiplicative in phi)
        if lam > 0:
            compensator = np.exp(mu_j + 0.5*sig_j**2) - 1.0
            jump = lam*T*(np.exp(1j*u_k*mu_j - 0.5*sig_j**2*u_k**2) - 1.0
                          - 1j*u_k*compensator)
            phi = phi * np.exp(jump)
        c=phi*np.exp(-1j*u_k*a); x=math.log(S0/K); alpha=k*math.pi/ba; ang0=-alpha*a
        denom=1+alpha**2; denom[0]=1.
        chi=(math.exp(b)*np.cos(k*np.pi)-np.cos(ang0)-alpha*np.sin(ang0))/denom; chi[0]=math.exp(b)-1
        psi=np.zeros(N); psi[0]=b; psi[1:]=-np.sin(ang0[1:])/alpha[1:]
        Vk=(2/ba)*(chi-psi); Vk_w=Vk.copy(); Vk_w[0]*=0.5
        s=np.sum((c.real*np.cos(u_k*x)-c.imag*np.sin(u_k*x))*Vk_w)
        return max(math.exp(-r*T)*K*s, max(S0*math.exp((r-q)*T)-K*math.exp(-r*T),0.))

    kappa, theta, sv, rho, v0 = 2.0, 0.04, 0.30, -0.70, 0.04
    S0, r, q = 1.0, 0.0, 0.0
    K_t    = torch.tensor([S0], device=dev, dtype=torch.float64)
    k_idx  = torch.arange(N_COS, device=dev)
    Vk     = cos_call_Vk(dev)
    # Bates params with λ=0 (pure Heston)
    par    = torch.tensor([[kappa, theta, sv, rho, v0, 0.0, 0.0, 0.1]],
                          device=dev, dtype=torch.float64)
    is_put = torch.tensor([False], device=dev)

    # Test a subset of maturities for speed (1M, 6M, 2Y)
    test_maturities = [MATURITIES[0], MATURITIES[5], MATURITIES[-1]]

    all_ok = True
    for T in test_maturities:
        ref  = cos_f64_np(S0, S0, float(T), kappa, theta, sv, rho, v0, r, q)
        gpu  = cos_call_prices(par, Vk, float(T), S0, K_t, r, q, k_idx)[0, 0].item()
        pe   = abs(gpu - ref) / (ref + 1e-15) * 100
        otm  = torch.tensor([[gpu]], device=dev, dtype=torch.float64)
        iv   = prices_to_iv(otm, K_t, float(T), S0, r, q, is_put)[0, 0].item()
        ok   = pe < 0.001 and not math.isnan(iv)
        all_ok &= ok
        print(
            f"  T={T:.4f}  ref={ref:.8f}  gpu={gpu:.8f}  "
            f"err={pe:.5f}%  iv={iv:.5f}  {'OK' if ok else 'FAIL'}"
        )

    print("\n[sanity] Part 2: Bates with jumps vs numpy Bates reference ...")
    lam, mu_j, sig_j = 0.5, -0.08, 0.15
    par_j = torch.tensor([[kappa, theta, sv, rho, v0, lam, mu_j, sig_j]],
                         device=dev, dtype=torch.float64)

    for T in test_maturities:
        ref  = cos_f64_np(S0, S0, float(T), kappa, theta, sv, rho, v0, r, q,
                          lam=lam, mu_j=mu_j, sig_j=sig_j)
        gpu  = cos_call_prices(par_j, Vk, float(T), S0, K_t, r, q, k_idx)[0, 0].item()
        pe   = abs(gpu - ref) / (ref + 1e-15) * 100
        otm  = torch.tensor([[gpu]], device=dev, dtype=torch.float64)
        iv   = prices_to_iv(otm, K_t, float(T), S0, r, q, is_put)[0, 0].item()
        ok   = pe < 0.001 and not math.isnan(iv)
        all_ok &= ok
        print(
            f"  T={T:.4f}  ref={ref:.8f}  gpu={gpu:.8f}  "
            f"err={pe:.5f}%  iv={iv:.5f}  {'OK' if ok else 'FAIL'}"
        )

    if not all_ok:
        raise RuntimeError("Sanity check FAILED")
    print("[sanity] PASSED\n")


#  Market environment bounds (sampled per-chunk to preserve GEMM decomposition)

R_BOUNDS = (0.00, 0.06)   # risk-free rate
Q_BOUNDS = (0.00, 0.04)   # dividend yield


#  Main generation loop

def generate(
    N:           int   = 2_000_000,
    output_path: str   = "heston_train.h5",
    S0:          float = 1.0,
    chunk_size:  int   = CHUNK,
    seed:        int   = 42,
    nan_policy:  str   = "market_consistent",
) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}" + (
        f"  ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""
    ))

    sanity_check(dev)

    params_np = sample_params(N, seed=seed)                      # (N, 8) float64
    K_grid_np = S0 * np.exp(LOG_MONEYNESS)                       # (NK,)  float64
    K_grid    = torch.tensor(K_grid_np, device=dev, dtype=torch.float64)
    k_idx     = torch.arange(N_COS, device=dev)
    Vk        = cos_call_Vk(dev)                                  # (Nc,)  float64
    is_put    = torch.tensor(IS_PUT_SIDE, device=dev)             # (NK,)  bool

    # RNG for per-chunk r, q sampling
    rng = np.random.default_rng(seed + 7777)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(N / chunk_size)

    with h5py.File(out, "w") as f:
        total_cells = NK * NT
        vc_dtype = "uint8" if total_cells <= 255 else "uint16"

        f.attrs["model_type"]    = "bates"
        f.attrs["nan_policy"]    = nan_policy
        f.attrs["grid_preset"]   = GRID_PRESET_NAME
        f.attrs["param_names"]   = PARAM_NAMES
        f.attrs["log_moneyness"] = LOG_MONEYNESS.tolist()
        f.attrs["maturities"]    = MATURITIES.tolist()
        f.attrs["K_grid"]        = K_grid_np.tolist()
        f.attrs["S0"]            = S0
        f.attrs["r_bounds"]      = list(R_BOUNDS)
        f.attrs["q_bounds"]      = list(Q_BOUNDS)
        f.attrs["N_COS"]         = N_COS
        f.attrs["HALF_ABS"]      = HALF_ABS
        f.attrs["N_total"]       = N
        f.attrs["param_lo"]      = PARAM_BOUNDS[:, 0].tolist()
        f.attrs["param_hi"]      = PARAM_BOUNDS[:, 1].tolist()
        f.attrs["NK"]            = NK
        f.attrs["NT"]            = NT

        cs_n = min(4096, N)
        f.create_dataset("params",
            shape=(N, N_PARAMS), dtype="float64",
            chunks=(cs_n, N_PARAMS), compression="lzf")
        f.create_dataset("market_params",
            shape=(N, 2), dtype="float64",
            chunks=(cs_n, 2), compression="lzf")
        f.create_dataset("iv_surface",
            shape=(N, NK, NT), dtype="float64",
            chunks=(min(256, N), NK, NT), compression="lzf")
        f.create_dataset("cell_mask",
            shape=(N, NK, NT), dtype="bool",
            chunks=(min(256, N), NK, NT), compression="lzf")
        f.create_dataset("raw_cell_mask",
            shape=(N, NK, NT), dtype="bool",
            chunks=(min(256, N), NK, NT), compression="lzf")
        f.create_dataset("valid_count",
            shape=(N,), dtype=vc_dtype,
            chunks=(cs_n,))

        ds_p  = f["params"]
        ds_mp = f["market_params"]
        ds_iv = f["iv_surface"]
        ds_cm = f["cell_mask"]
        ds_rcm = f["raw_cell_mask"]
        ds_vc = f["valid_count"]

        total_valid_cells = 0
        t0 = time.time()

        for ci in tqdm(range(n_chunks), desc="Generating"):
            i0 = ci * chunk_size
            i1 = min(i0 + chunk_size, N)
            B  = i1 - i0

            # Sample r, q per-chunk (all B samples share same r, q)
            r = float(rng.uniform(*R_BOUNDS))
            q = float(rng.uniform(*Q_BOUNDS))

            params_t = torch.tensor(params_np[i0:i1], device=dev, dtype=torch.float64)
            iv_chunk = torch.full((B, NK, NT), float("nan"),
                                  dtype=torch.float64, device=dev)

            for ti, T in enumerate(MATURITIES.tolist()):
                call_p = cos_call_prices(params_t, Vk, T, S0, K_grid, r, q, k_idx) # type: ignore
                otm_p  = call_to_otm(call_p, K_grid, T, S0, r, q, is_put) # type: ignore
                iv_chunk[:, :, ti] = prices_to_iv(otm_p, K_grid, T, S0, r, q, is_put) # type: ignore

            raw_mask_chunk = ~torch.isnan(iv_chunk)       # (B, NK, NT) bool

            if nan_policy == "mask":
                iv_out = iv_chunk
                mask_chunk = raw_mask_chunk
            elif nan_policy == "floor":
                iv_out = torch.nan_to_num(iv_chunk, nan=IV_LO).clamp(min=IV_LO, max=IV_HI)
                mask_chunk = torch.ones_like(raw_mask_chunk)
            elif nan_policy == "market_consistent":
                iv_out = fill_nan_market_consistent(iv_chunk)
                mask_chunk = torch.ones_like(raw_mask_chunk)
            else:
                raise ValueError(f"Unknown nan_policy: {nan_policy}")

            vcount     = mask_chunk.sum(dim=(1, 2))       # (B,) int

            total_valid_cells += int(mask_chunk.sum().item())

            # Store r, q per sample (same value within chunk)
            rq_chunk = np.full((B, 2), [r, q], dtype=np.float64)

            ds_p[i0:i1]  = params_np[i0:i1] # type: ignore
            ds_mp[i0:i1] = rq_chunk # type: ignore
            ds_iv[i0:i1] = iv_out.cpu().numpy() # type: ignore
            ds_cm[i0:i1] = mask_chunk.cpu().numpy() # type: ignore
            ds_rcm[i0:i1] = raw_mask_chunk.cpu().numpy() # type: ignore
            ds_vc[i0:i1] = vcount.cpu().numpy().astype(ds_vc.dtype) # type: ignore

        elapsed = time.time() - t0
        max_cells = N * NK * NT
        f.attrs["gen_time_s"]         = elapsed
        f.attrs["total_valid_cells"]  = total_valid_cells
        f.attrs["cell_fill_rate_pct"] = 100.0 * total_valid_cells / max_cells

    fill_pct = 100.0 * total_valid_cells / max_cells
    size     = out.stat().st_size / 1e9
    print(f"\n[done]  cell fill rate: {fill_pct:.2f}%  ({total_valid_cells:,} / {max_cells:,})")
    print(f"[file]  {out}  ({size:.3f} GB)")
    print(f"[perf]  {elapsed/60:.1f} min  |  {N/elapsed:,.0f} samples/s")
    if dev.type == "cuda":
        print(f"[vram]  peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


#  PyTorch Dataset — lazy HDF5, fork-safe

class BatesDataset(torch.utils.data.Dataset):
    """
    Returns (params_norm, iv_flat, mask_flat):
      params_norm : (N_PARAMS+2,) float32  normalised to [0,1] (8 Bates params + r, q)
      iv_flat     : (NK*NT,)      float32  IVs; NaN replaced with 0
      mask_flat   : (NK*NT,)      bool     True where IV is valid

    In the training loop, mask_flat is used to zero-out NaN cells in the loss:
        loss = (mask * (nn_out - iv_flat)**2).sum() / mask.sum().clamp(min=1)

    Lazy open: HDF5 file opened inside __getitem__ after DataLoader fork.
    """

    def __init__(self, h5_path: str, min_valid_cells: int | None = None):
        """
        min_valid_cells: discard samples with fewer than this many valid cells.
        Default is 50% of total grid cells (dynamic from NK*NT).
        Set to 0 to keep all samples.
        """
        self.h5_path = h5_path
        self._h5     = None

        with h5py.File(h5_path, "r") as f:
            nk = int(f.attrs.get("NK", f["iv_surface"].shape[1])) # type: ignore
            nt = int(f.attrs.get("NT", f["iv_surface"].shape[2])) # type: ignore
            default_min_valid = int(math.ceil(0.5 * nk * nt))
            self.min_valid_cells = default_min_valid if min_valid_cells is None else min_valid_cells

            vc            = f["valid_count"][:] # type: ignore
            self.idx      = np.where(vc >= self.min_valid_cells)[0] # type: ignore
            self.param_lo = np.array(f.attrs["param_lo"], dtype=np.float32)
            self.param_hi = np.array(f.attrs["param_hi"], dtype=np.float32)
            # r, q bounds for normalization
            r_bounds = np.array(f.attrs.get("r_bounds", [0.0, 0.06]), dtype=np.float32)
            q_bounds = np.array(f.attrs.get("q_bounds", [0.0, 0.04]), dtype=np.float32)
            self.market_lo = np.array([r_bounds[0], q_bounds[0]], dtype=np.float32)
            self.market_hi = np.array([r_bounds[1], q_bounds[1]], dtype=np.float32)

        print(
            f"[dataset] {h5_path}: {len(self.idx):,} samples "
            f"(min_valid_cells={self.min_valid_cells})"
        )

    def _open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        self._open()
        j = int(self.idx[i])

        params = self._h5["params"][j].astype(np.float32)        # type: ignore  (N_PARAMS,)
        rq     = self._h5["market_params"][j].astype(np.float32) # type: ignore  (2,)
        iv     = self._h5["iv_surface"][j].astype(np.float32)    # type: ignore  (NK, NT)
        mask   = self._h5["cell_mask"][j]                         # type: ignore  (NK, NT) bool

        # Normalize params and market params to [0, 1]
        params = (params - self.param_lo) / (self.param_hi - self.param_lo + 1e-12)
        rq     = (rq - self.market_lo) / (self.market_hi - self.market_lo + 1e-12)
        all_params = np.concatenate([params, rq])  # (N_PARAMS+2,)

        # Replace NaN with 0 in iv (loss will mask these out anyway)
        iv[~mask] = 0.0 # pyright: ignore[reportIndexIssue, reportOperatorIssue]

        return (
            torch.from_numpy(all_params),
            torch.from_numpy(iv.flatten()),         # (NK*NT,) # type: ignore
            torch.from_numpy(mask.flatten()),        # (NK*NT,) bool # type: ignore
        )

    def __del__(self) -> None:
        if self._h5 is not None:
            try: self._h5.close()
            except Exception: pass

# Backward-compatible alias
HestonDataset = BatesDataset


#  CLI

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--N",     type=int,   default=2_000_000)
    ap.add_argument("--out",   type=str,   default="heston_train.h5")
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--chunk", type=int,   default=CHUNK,
                    help=f"GPU batch size (default {CHUNK}; f64 uses 2x VRAM vs f32)")
    ap.add_argument("--grid",  type=str, default=DEFAULT_GRID_PRESET,
                    choices=sorted(GRID_PRESETS.keys()),
                    help="Grid preset: base25x10 or high41x14")
    ap.add_argument("--S0",    type=float, default=1.0)
    ap.add_argument("--nan-policy", type=str, default="market_consistent",
                    choices=["market_consistent", "mask", "floor"],
                    help="How to handle non-invertible IV cells")
    ap.add_argument("--val",   action="store_true",
                    help="200k val set (seed+9999, appends _val.h5)")
    ap.add_argument("--check", action="store_true",
                    help="Sanity check only, no generation")
    args = ap.parse_args()

    set_grid_preset(args.grid)
    print(
        f"[grid] preset={GRID_PRESET_NAME}  NK={NK}  NT={NT}  "
        f"cells/surface={NK*NT}"
    )

    if args.check:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sanity_check(dev)
    elif args.val:
        out = args.out.replace(".h5", "") + "_val.h5"
        generate(N=200_000, output_path=out, seed=args.seed + 9999,
                 chunk_size=args.chunk, S0=args.S0, nan_policy=args.nan_policy)
    else:
        generate(N=args.N, output_path=args.out, seed=args.seed,
                 chunk_size=args.chunk, S0=args.S0, nan_policy=args.nan_policy)
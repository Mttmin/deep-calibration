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


def load_guided_param_bank(path: str) -> np.ndarray:
    """
    Load flattened guided parameter candidates from calibration output HDF5.

    When top_weighted_rmse is present, filters to candidates with RMSE ≤ median
    to exclude low-quality fits that would corrupt guided warm-start sampling.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Guided param bank not found: {path}")

    with h5py.File(p, "r") as f:
        if "top_params" in f:
            arr = np.array(f["top_params"], dtype=np.float64)  # (M, K, 8)
            flat = arr.reshape(-1, arr.shape[-1])
            # Filter to best half by RMSE when scores are available
            if "top_weighted_rmse" in f:
                rmse_flat = np.array(f["top_weighted_rmse"]).reshape(-1)
                median_rmse = float(np.median(rmse_flat))
                keep = rmse_flat <= median_rmse
                flat = flat[keep]
                print(f"[guided_bank] RMSE filter: kept {keep.sum():,}/{len(keep):,} "
                      f"candidates (threshold={median_rmse:.4f})")
        elif "guided_params" in f:
            flat = np.array(f["guided_params"], dtype=np.float64)
        else:
            raise KeyError("Guided bank must contain 'top_params' or 'guided_params'")

    if flat.shape[1] != N_PARAMS:
        raise ValueError(
            f"Guided params width mismatch: expected {N_PARAMS}, got {flat.shape[1]}"
        )
    return flat


#  Parameter space — pure Heston (5 params)
#
#  Bounds tightened to market-realistic equity regimes:
#    θ, v₀ capped at 0.12  (σ_LR/σ₀ ≤ 35%) — 50% long-run vol is crisis-only
#             and pulls the training mean ATM IV far above typical market levels.
#    σ_v  capped at 1.20   (>1.20 produces unrealistic wings and numerical issues)
#  Historical SPX Heston calibrations (normal regimes) cluster well within these.

PARAM_BOUNDS: np.ndarray = np.array([
    [0.30, 8.00],    # κ    mean-reversion speed  (very low κ + high θ is degenerate)
    [0.02, 0.12],    # θ    long-run variance  (σ_LR ∈ [14%, 35%])
    [0.05, 1.20],    # σ_v  vol-of-vol
    [-0.98, 0.10],   # ρ    equity skew almost always negative
    [0.02, 0.12],    # v₀   initial variance  (σ₀  ∈ [14%, 35%])
], dtype=np.float64)
PARAM_NAMES = ["kappa", "theta", "sigma_v", "rho", "v0"]
N_PARAMS: int = len(PARAM_NAMES)  # 5


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

def sample_params(
    N: int,
    seed: int = 42,
    guided_bank: np.ndarray | None = None,
    guided_weight: float = 0.0,
    guided_jitter: float = 0.04,
) -> np.ndarray:
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
    guided_weight = float(np.clip(guided_weight, 0.0, 1.0))
    n_guided = int(round(N * guided_weight)) if guided_bank is not None else 0
    n_base = N - n_guided

    sampler = qmc.Sobol(d=N_PARAMS, scramble=True, seed=seed) # type: ignore
    n_raw   = int(2 ** math.ceil(math.log2(max(2, int(n_base * 1.5)))))
    raw     = sampler.random(n_raw)
    pts_base = qmc.scale(raw, lo, hi)[:n_base]   # float64

    # Mixture prior: dedicate a small fraction to stressed skew regimes.
    # Only σ_v and ρ are stressed — these create realistic short-tenor skew
    # without inflating the average IV level. Jump boosting is removed since
    # this is now a pure Heston surrogate.
    rng = np.random.default_rng(seed + 2026)
    stress_prob = 0.12
    stress = rng.random(n_base) < stress_prob
    n_stress = int(stress.sum())
    if n_stress > 0:
        pts_base[stress, 2] = rng.uniform(0.40, 1.20, size=n_stress)    # sigma_v: stressed but realistic
        pts_base[stress, 3] = rng.uniform(-0.98, -0.60, size=n_stress)   # rho: steep skew

    if n_guided > 0:
        pick = rng.integers(0, len(guided_bank), size=n_guided) # type: ignore
        pts_guided = guided_bank[pick].copy() # type: ignore
        widths = (hi - lo)[None, :]
        pts_guided += rng.normal(0.0, guided_jitter, size=pts_guided.shape) * widths
        pts_guided = np.clip(pts_guided, lo, hi)
        pts = np.concatenate([pts_base, pts_guided], axis=0)
    else:
        pts = pts_base

    κ, θ, σ = pts[:, 0], pts[:, 1], pts[:, 2]
    feller_pct = ((2.0 * κ * θ) >= σ**2).mean() * 100

    print(
        f"[sample] Sobol draws: {n_raw:,}  |  "
        f"Feller pass: {feller_pct:.1f}% (not filtered)  |  "
        f"stress regime: {100.0 * n_stress / max(1, n_base):.1f}% (base)  |  "
        f"guided mix: {100.0 * n_guided / max(1, N):.1f}%  |  keeping: {N:,}"
    )
    return pts   # (N, 8) float64


#  Bates CF — Heston (Little Trap) + Merton log-normal jumps, float64 / complex128

def bates_cf(
    u:      torch.Tensor,   # (Nc,)  real frequencies, float64
    T:      float,
    params: torch.Tensor,   # (B, 5) float64  [κ, θ, σ_v, ρ, v₀]  — pure Heston
    r:      float,
    q:      float,
) -> torch.Tensor:           # (B, Nc) complex128
    """
    φ(u) = E_Q[e^{iu·ln(S_T/S_0)}] under Heston (1993) model.

    Little Trap formulation (Albrecher et al. 2007) for branch-cut stability:
        ξ  = κ − iuρσ_v
        d  = √(ξ² + σ_v²·u(u+i))
        g  = (ξ−d) / (ξ+d)
        logQ = ln((1−g·e^{−dT}) / (1−g))
    """
    κ  = params[:, 0:1]   # (B, 1)
    θ  = params[:, 1:2]
    σv = params[:, 2:3]
    ρ  = params[:, 3:4]
    v0 = params[:, 4:5]

    uc   = u.to(dtype=torch.complex128)          # (Nc,)

    ξ    = κ - 1j * ρ * σv * uc                 # (B, Nc)
    d    = torch.sqrt(ξ**2 + σv**2 * uc * (uc + 1j))
    g    = (ξ - d) / (ξ + d)
    e_dT = torch.exp(-d * T)
    logQ = torch.log((1.0 - g * e_dT) / (1.0 - g))

    A = 1j * uc * (r - q) * T
    B = (κ * θ / σv**2) * ((ξ - d) * T - 2.0 * logQ)
    C = (v0  / σv**2)   * (ξ - d) * (1.0 - e_dT) / (1.0 - g * e_dT)

    return torch.exp(A + B + C)   # (B, Nc) complex128


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
    params:  torch.Tensor,   # (B, 5) float64  [κ, θ, σ_v, ρ, v₀]
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
    Heston COS pricer vs numpy f64 reference.
    Expects < 0.001% price error (f64 truncation vs analytical).
    """
    print("\n[sanity] Heston GPU-COS vs numpy reference ...")

    def heston_cf_np(u, T, kappa, theta, sv, rho, v0, r=0., q=0.):
        xi=kappa-1j*rho*sv*u; d=np.sqrt(xi**2+sv**2*u*(u+1j))
        g=(xi-d)/(xi+d); e_dT=np.exp(-d*T); logQ=np.log((1-g*e_dT)/(1-g))
        return np.exp(1j*u*(r-q)*T + (kappa*theta/sv**2)*((xi-d)*T-2*logQ)
                      + (v0/sv**2)*(xi-d)*(1-e_dT)/(1-g*e_dT))

    def cos_f64_np(S0, K, T, kappa, theta, sv, rho, v0, r=0., q=0., N=256, H=3.0):
        a,b=-H,H; ba=b-a; k=np.arange(N,dtype=float); u_k=k*math.pi/ba
        phi=heston_cf_np(u_k,T,kappa,theta,sv,rho,v0,r,q)
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
    par    = torch.tensor([[kappa, theta, sv, rho, v0]], device=dev, dtype=torch.float64)
    is_put = torch.tensor([False], device=dev)

    # Test 1M, 6M, 2Y
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
    guided_bank_path: str | None = None,
    guided_weight: float = 0.0,
    guided_jitter: float = 0.04,
) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}" + (
        f"  ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""
    ))

    sanity_check(dev)

    guided_bank = load_guided_param_bank(guided_bank_path) if guided_bank_path else None
    if guided_bank is not None:
        print(
            f"[guided] bank={guided_bank_path}  rows={len(guided_bank):,}  "
            f"weight={guided_weight:.2f}  jitter={guided_jitter:.3f}"
        )

    params_np = sample_params(
        N,
        seed=seed,
        guided_bank=guided_bank,
        guided_weight=guided_weight,
        guided_jitter=guided_jitter,
    )
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

        f.attrs["model_type"]    = "heston"
        f.attrs["nan_policy"]    = nan_policy
        f.attrs["guided_bank"]   = guided_bank_path if guided_bank_path else ""
        f.attrs["guided_weight"] = guided_weight
        f.attrs["guided_jitter"] = guided_jitter
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
    Returns (params_norm, iv_flat, mask_flat) or, when return_confidence=True,
    (params_norm, iv_flat, mask_flat, conf_flat):
      params_norm : (N_PARAMS+2,) float32  normalised to [0,1] (5 Heston params + r, q)
      iv_flat     : (NK*NT,)      float32  IVs; NaN replaced with 0
      mask_flat   : (NK*NT,)      bool     True where IV is valid
      conf_flat   : (NK*NT,)      float32  per-cell confidence proxy (when requested)
                    cells in raw_cell_mask get 1.0; filled-only cells get 0.3

    In the training loop, mask_flat is used to zero-out NaN cells in the loss:
        loss = (mask * (nn_out - iv_flat)**2).sum() / mask.sum().clamp(min=1)

    Can either lazily read from HDF5 per sample or preload the selected split
    into host RAM once during initialisation.
    """

    def __init__(
        self,
        h5_path: str,
        min_valid_cells: int | None = None,
        indices: np.ndarray | range | None = None,
        preload: bool = False,
        preload_device: str = "cpu",
        return_confidence: bool = False,
    ):
        """
        min_valid_cells: discard samples with fewer than this many valid cells.
        Default is 50% of total grid cells (dynamic from NK*NT).
        Set to 0 to keep all samples.

        indices: optional positions within the filtered valid-sample array.
        preload: if True, materialise the selected split into RAM once.
        """
        self.h5_path = h5_path
        self._h5     = None
        self.preload = preload
        self.preload_device = preload_device
        self.return_confidence = return_confidence
        self.params_tensor: torch.Tensor | None = None
        self.iv_tensor: torch.Tensor | None = None
        self.mask_tensor: torch.Tensor | None = None
        self.conf_tensor: torch.Tensor | None = None

        if self.preload_device not in {"cpu", "cuda"}:
            raise ValueError(f"Unknown preload device: {self.preload_device}")
        if self.preload and self.preload_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA preload requested but CUDA is unavailable")

        with h5py.File(h5_path, "r") as f:
            nk = int(f.attrs.get("NK", f["iv_surface"].shape[1])) # type: ignore
            nt = int(f.attrs.get("NT", f["iv_surface"].shape[2])) # type: ignore
            self.nk = nk
            self.nt = nt
            default_min_valid = int(math.ceil(0.5 * nk * nt))
            self.min_valid_cells = default_min_valid if min_valid_cells is None else min_valid_cells

            vc            = f["valid_count"][:] # type: ignore
            valid_idx     = np.where(vc >= self.min_valid_cells)[0] # type: ignore
            if indices is not None:
                split_idx = np.asarray(indices, dtype=np.int64)
                self.idx = valid_idx[split_idx]
            else:
                self.idx = valid_idx
            self.param_lo = np.array(f.attrs["param_lo"], dtype=np.float32)
            self.param_hi = np.array(f.attrs["param_hi"], dtype=np.float32)
            # r, q bounds for normalization
            r_bounds = np.array(f.attrs.get("r_bounds", [0.0, 0.06]), dtype=np.float32)
            q_bounds = np.array(f.attrs.get("q_bounds", [0.0, 0.04]), dtype=np.float32)
            self.market_lo = np.array([r_bounds[0], q_bounds[0]], dtype=np.float32)
            self.market_hi = np.array([r_bounds[1], q_bounds[1]], dtype=np.float32)

        print(
            f"[dataset] {h5_path}: {len(self.idx):,} samples "
            f"(min_valid_cells={self.min_valid_cells}, "
            f"preload={'yes' if self.preload else 'no'}:{self.preload_device})"
        )

        if self.preload:
            self._preload_into_ram()

    def _estimate_preload_bytes(self) -> int:
        n = len(self.idx)
        n_params = len(self.param_lo) + len(self.market_lo)
        n_flat = self.nk * self.nt
        total = n * (n_params * 4 + n_flat * 4 + n_flat)
        if self.return_confidence:
            total += n * n_flat * 4
        return total

    def _preload_into_ram(self, io_threads: int = 8) -> None:
        import concurrent.futures
        import time as _time

        est_gb = self._estimate_preload_bytes() / 1e9
        n = len(self.idx)
        print(
            f"[dataset] preloading {n:,} samples (~{est_gb:.2f} GB) "
            f"to {self.preload_device.upper()} using {io_threads} parallel reader threads …"
        )

        # Use a slice selector when indices are contiguous — orders of magnitude
        # faster than fancy indexing inside h5py because it avoids per-element
        # chunk lookups and lets the library do a single sequential read.
        idx = self.idx
        i0, i1 = 0, n   # defaults; only used when is_contiguous is True
        if n > 0 and int(idx[-1]) - int(idx[0]) + 1 == n and (
            n == 1 or bool(np.all(np.diff(idx) == 1))
        ):
            is_contiguous = True
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
        else:
            is_contiguous = False

        h5_path = self.h5_path

        def _read_slice(dsname: str, dtype: type, start: int, stop: int) -> np.ndarray:
            with h5py.File(h5_path, "r", swmr=True) as f:
                return np.array(f[dsname][start:stop], dtype=dtype)  # type: ignore

        def _read_fancy(dsname: str, dtype: type) -> np.ndarray:
            with h5py.File(h5_path, "r", swmr=True) as f:
                return np.array(f[dsname][idx], dtype=dtype)  # type: ignore

        t0 = _time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=io_threads) as ex:
            if is_contiguous:
                # Split the big iv_surface read across multiple threads;
                # params/market_params/cell_mask are small — one thread each.
                chunk = max(1, (i1 - i0 + io_threads - 4) // (io_threads - 3))
                iv_futures = [
                    ex.submit(_read_slice, "iv_surface", np.float32,
                              s, min(s + chunk, i1))
                    for s in range(i0, i1, chunk)
                ]
                f_params = ex.submit(_read_slice, "params",        np.float32, i0, i1)
                f_rq     = ex.submit(_read_slice, "market_params", np.float32, i0, i1)
                f_mask   = ex.submit(_read_slice, "cell_mask",     bool,       i0, i1)
                f_raw    = ex.submit(_read_slice, "raw_cell_mask", bool,       i0, i1) if self.return_confidence else None
                params = f_params.result()
                rq     = f_rq.result()
                mask   = f_mask.result()
                raw_mask = f_raw.result() if f_raw is not None else None
                iv     = np.concatenate([ft.result() for ft in iv_futures], axis=0)
            else:
                f_params = ex.submit(_read_fancy, "params",        np.float32)
                f_rq     = ex.submit(_read_fancy, "market_params", np.float32)
                f_iv     = ex.submit(_read_fancy, "iv_surface",    np.float32)
                f_mask   = ex.submit(_read_fancy, "cell_mask",     bool)
                f_raw    = ex.submit(_read_fancy, "raw_cell_mask", bool) if self.return_confidence else None
                params = f_params.result()
                rq     = f_rq.result()
                iv     = f_iv.result()
                mask   = f_mask.result()
                raw_mask = f_raw.result() if f_raw is not None else None
        print(f"[dataset] read done in {_time.perf_counter() - t0:.1f}s — normalising …")

        params = (params - self.param_lo) / (self.param_hi - self.param_lo + 1e-12)
        rq     = (rq - self.market_lo) / (self.market_hi - self.market_lo + 1e-12)
        all_params = np.concatenate([params, rq], axis=1)
        iv[~mask] = 0.0

        self.params_tensor = torch.from_numpy(all_params)
        self.iv_tensor = torch.from_numpy(iv.reshape(len(self.idx), -1))
        self.mask_tensor = torch.from_numpy(mask.reshape(len(self.idx), -1))
        if self.return_confidence:
            if raw_mask is None:
                raise RuntimeError("Confidence preload requested but raw_cell_mask was not loaded")
            conf = np.where(np.asarray(raw_mask, dtype=bool), 1.0, 0.3).astype(np.float32)
            self.conf_tensor = torch.from_numpy(conf.reshape(len(self.idx), -1))

        if self.preload_device == "cuda":
            self.params_tensor = self.params_tensor.to("cuda", non_blocking=False)
            self.iv_tensor = self.iv_tensor.to("cuda", non_blocking=False)
            self.mask_tensor = self.mask_tensor.to("cuda", non_blocking=False)
            if self.conf_tensor is not None:
                self.conf_tensor = self.conf_tensor.to("cuda", non_blocking=False)

        print(f"[dataset] preload total: {_time.perf_counter() - t0:.1f}s")

    def _open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        if self.params_tensor is not None and self.iv_tensor is not None and self.mask_tensor is not None:
            if self.return_confidence:
                if self.conf_tensor is None:
                    raise RuntimeError("return_confidence=True but confidence tensor is missing")
                return (
                    self.params_tensor[i],
                    self.iv_tensor[i],
                    self.mask_tensor[i],
                    self.conf_tensor[i],
                )
            return (
                self.params_tensor[i],
                self.iv_tensor[i],
                self.mask_tensor[i],
            )

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

        if self.return_confidence:
            # Confidence proxy: raw quotes get 1.0, filled-only cells get 0.3
            raw_mask = self._h5["raw_cell_mask"][j]               # type: ignore  (NK, NT) bool
            conf = np.where(np.asarray(raw_mask, dtype=bool), 1.0, 0.3).astype(np.float32)
            return (
                torch.from_numpy(all_params),
                torch.from_numpy(iv.flatten()),         # (NK*NT,) # type: ignore
                torch.from_numpy(mask.flatten()),        # (NK*NT,) bool # type: ignore
                torch.from_numpy(conf.flatten()),        # (NK*NT,) float32
            )

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
    ap.add_argument("--out",   type=str,   default="../data/heston_train.h5")
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
    ap.add_argument("--guided-bank", type=str, default=None,
                    help="HDF5 file with calibrated guided params (top_params/guided_params)")
    ap.add_argument("--guided-weight", type=float, default=0.0,
                    help="Fraction of samples drawn from guided param bank")
    ap.add_argument("--guided-jitter", type=float, default=0.04,
                    help="Relative jitter scale applied to guided params")
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
                 chunk_size=args.chunk, S0=args.S0, nan_policy=args.nan_policy,
                 guided_bank_path=args.guided_bank,
                 guided_weight=args.guided_weight,
                 guided_jitter=args.guided_jitter)
    else:
        generate(N=args.N, output_path=args.out, seed=args.seed,
                 chunk_size=args.chunk, S0=args.S0, nan_policy=args.nan_policy,
                 guided_bank_path=args.guided_bank,
                 guided_weight=args.guided_weight,
                 guided_jitter=args.guided_jitter)
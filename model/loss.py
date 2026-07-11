"""
model/loss.py
=============
Loss functions for deep calibration of the Bates/Heston surrogate.

Three components:
  1. vega_weighted_mse  — Black-Scholes vega-weighted MSE on implied vols
                          (upweights ATM/near-money, downweights deep OTM)
  2. calendar_spread_penalty  — physics-informed no-arb: ∂(σ²T)/∂T ≥ 0
  3. durrleman_butterfly_penalty — physics-informed no-arb: g(k,T) ≥ 0
                                   (Durrleman 2002 condition)

All functions work under BF16 autocast: inputs are cast to float32 internally
before any squared-error or division, to avoid underflow/overflow.

References
----------
Vega formula:  deep_calibration_research.tex §1.4.2 (lines 350-354)
Calendar arb:  Durrleman (2002) §2; equivalent to ∂w/∂T ≥ 0
Butterfly arb: Durrleman (2002), Theorem 2.1
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .network import GridConstants

_SQRT_2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Vega weight computation
# ---------------------------------------------------------------------------

def compute_vega_weights(
    iv_target: torch.Tensor,   # (B, N_FLAT) float32  — ground-truth IVs
    grid: GridConstants,
    r: torch.Tensor,            # (B,) float32  — risk-free rate (physical scale)
    q: torch.Tensor,            # (B,) float32  — dividend yield (physical scale)
    S: float = 1.0,
    eps: float = 1e-6,
    weight_floor: float = 0.0,
) -> torch.Tensor:
    """
    Compute Black-Scholes vega weights for each grid cell.

    Formula (research doc §1.4.2):
        d₁(k, T) = [-k + (r - q + σ²/2) · T] / (σ · √T + ε)
        φ(d₁)   = exp(-d₁² / 2) / √(2π)
        w_j     = √T · φ(d₁) · exp(-q · T)        (S = 1 normalised)

    After computation, weights are normalised so the mean over valid cells ≈ 1.

    Args:
        iv_target: (B, NK*NT) ground-truth IV surface (NaN cells already 0).
        grid:      GridConstants with log_moneyness (NK,) and maturities (NT,).
        r:         (B,) per-sample risk-free rate in physical units.
        q:         (B,) per-sample dividend yield in physical units.
        S:         Spot price (normalised to 1.0 in the dataset).
        eps:       Numerical floor.

    Returns:
        weights: (B, NK*NT) float32, mean-normalised vega weights.
    """
    # Work in float32 throughout (BF16 underflows in exp(-d1²/2))
    iv_target = iv_target.float()
    r = r.float()
    q = q.float()

    B  = iv_target.shape[0]
    NK = grid.NK
    NT = grid.NT

    # Reshape to (B, NK, NT)
    sigma = iv_target.view(B, NK, NT)  # (B, NK, NT)

    # Broadcast grid arrays onto (1, NK, 1) and (1, 1, NT)
    k = grid.log_moneyness.to(iv_target.device).view(1, NK, 1).float()  # log(K/S)
    T = grid.maturities.to(iv_target.device).view(1, 1, NT).float()

    # Per-sample r, q: (B, 1, 1)
    r3 = r.view(B, 1, 1)
    q3 = q.view(B, 1, 1)

    # d₁ = [-k + (r - q + σ²/2)·T] / (σ·√T + ε)
    sqrtT = T.sqrt()
    d1 = (-k + (r3 - q3 + 0.5 * sigma**2) * T) / (sigma * sqrtT + eps)

    # φ(d₁) = exp(-d₁²/2) / √(2π)
    phi = torch.exp(-0.5 * d1**2) / _SQRT_2PI

    # w = S · √T · φ(d₁) · exp(-q·T)   with S=1
    weights = sqrtT * phi * torch.exp(-q3 * T)   # (B, NK, NT)

    # Flatten and normalise: mean weight over all cells ≈ 1
    weights = weights.view(B, NK * NT)
    mean_w  = weights.mean(dim=1, keepdim=True).clamp(min=eps)
    weights = weights / mean_w

    # Weight floor: prevents zero-gradient wings. Floor is in units of post-norm
    # mean weight (e.g. 0.05 = 5% of ATM-normalised mean). Renormalise so mean ≈ 1.
    if weight_floor > 0.0:
        weights = weights.clamp(min=weight_floor)
        mean_w  = weights.mean(dim=1, keepdim=True).clamp(min=eps)
        weights = weights / mean_w

    return weights  # (B, NK*NT) float32


# ---------------------------------------------------------------------------
# Data loss: vega-weighted MSE
# ---------------------------------------------------------------------------

def vega_weighted_mse(
    iv_pred:    torch.Tensor,            # (B, N_FLAT)  float32 or bf16
    iv_target:  torch.Tensor,            # (B, N_FLAT)  float32
    mask:       torch.Tensor,            # (B, N_FLAT)  bool
    weights:    torch.Tensor,            # (B, N_FLAT)  float32
    confidence: torch.Tensor | None = None,  # (B, N_FLAT)  float32 ∈ [0,1]  optional
) -> torch.Tensor:
    """
    Vega-weighted mean squared error on implied volatilities.

        L_vega = Σ(mask · conf · w · (σ_NN - σ_target)²) / Σ(mask · conf · w).clamp(1)

    Both numerator and denominator are summed over the 686 cells then
    averaged over the batch dimension.

    Args:
        iv_pred:    Network output (cast to float32 internally).
        iv_target:  Ground-truth IV (already float32 from dataset).
        mask:       True where IV is valid (non-NaN, non-zero).
        weights:    Per-cell vega weights from compute_vega_weights().
        confidence: Optional per-cell confidence in [0,1].  When provided (e.g.
                    from real surface confidence grids), cells are down-weighted
                    proportionally.  Default: uniform 1.0 (no effect).

    Returns:
        Scalar loss tensor (float32).
    """
    pred   = iv_pred.float()
    target = iv_target.float()
    m      = mask.float()
    conf   = confidence.float() if confidence is not None else torch.ones_like(m)
    w      = weights.float() * m * conf

    sq_err = (pred - target) ** 2
    loss   = (w * sq_err).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    return loss.mean()


def masked_mse(
    iv_pred:    torch.Tensor,
    iv_target:  torch.Tensor,
    mask:       torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE on IVs with optional confidence weights."""
    pred = iv_pred.float()
    target = iv_target.float()
    m = mask.float()
    conf = confidence.float() if confidence is not None else torch.ones_like(m)
    w = m * conf
    sq_err = (pred - target) ** 2
    loss = (w * sq_err).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    return loss.mean()


def masked_log_mse(
    iv_pred:    torch.Tensor,
    iv_target:  torch.Tensor,
    mask:       torch.Tensor,
    confidence: torch.Tensor | None = None,
    floor:      float = 1e-4,
) -> torch.Tensor:
    """Masked MSE on log(IV) — uniform relative-error weighting across strikes."""
    pred   = iv_pred.float().clamp(min=floor)
    target = iv_target.float().clamp(min=floor)
    m      = mask.float()
    conf   = confidence.float() if confidence is not None else torch.ones_like(m)
    w      = m * conf
    sq_err = (torch.log(pred) - torch.log(target)) ** 2
    loss   = (w * sq_err).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    return loss.mean()


# ---------------------------------------------------------------------------
# PINN penalty 1: calendar-spread no-arbitrage
# ---------------------------------------------------------------------------

def calendar_spread_penalty(
    iv_pred:   torch.Tensor,   # (B, N_FLAT)  float32 or bf16
    grid:      GridConstants,
    mask:      torch.Tensor,   # (B, N_FLAT)  bool
) -> torch.Tensor:
    """
    Penalises violations of: ∂(σ²·T)/∂T ≥ 0  (calendar no-arb).

    Total implied variance w(k, T) = σ(k, T)² · T must be non-decreasing
    in T for each fixed k.  We penalise the squared hinge loss on negative
    forward differences along the T dimension.

        L_cal = mean( relu( w[:,:,t] − w[:,:,t+1] )² )   [over valid pairs]

    Args:
        iv_pred: Network-predicted IV surface (B, NK*NT).
        grid:    GridConstants.
        mask:    Valid-cell mask (B, NK*NT).

    Returns:
        Scalar penalty (float32).  Returns 0 if NT < 2.
    """
    if grid.NT < 2:
        return iv_pred.new_zeros(())

    B  = iv_pred.shape[0]
    NK = grid.NK
    NT = grid.NT

    sigma = iv_pred.float().view(B, NK, NT)
    T     = grid.maturities.to(iv_pred.device).view(1, 1, NT).float()
    m     = mask.float().view(B, NK, NT)

    # Total variance (B, NK, NT)
    tv = sigma**2 * T

    # Adjacent forward differences along T (B, NK, NT-1)
    dtv        = tv[:, :, 1:] - tv[:, :, :-1]
    pair_valid = m[:, :, :-1] * m[:, :, 1:]   # valid only if both cells valid

    violations = F.relu(-dtv) ** 2             # hinge²  (B, NK, NT-1)
    n_valid    = pair_valid.sum().clamp(min=1.0)
    adj_penalty = (violations * pair_valid).sum() / n_valid

    # Long-range calendar check at key index pairs (1W→1M, 1M→6M, 6M→1.5Y).
    # These correspond to high41x14 grid indices (0,3), (3,7), (7,11)
    # (maturity index 11 is 1.5y; 2.0y is index 12).
    # Prevents the network from fitting adjacent differences while violating
    # larger-scale total-variance monotonicity.
    lr_pairs = [(0, 3), (3, 7), (7, 11)]
    lr_pen = tv.new_zeros(())
    for lo_t, hi_t in lr_pairs:
        if hi_t < NT:
            diff     = tv[:, :, hi_t] - tv[:, :, lo_t]
            lr_valid = m[:, :, lo_t] * m[:, :, hi_t]
            n_lr     = lr_valid.sum().clamp(min=1.0)
            lr_pen   = lr_pen + (F.relu(-diff) ** 2 * lr_valid).sum() / n_lr

    return adj_penalty + 0.5 * lr_pen


# ---------------------------------------------------------------------------
# PINN penalty 2: Durrleman butterfly no-arbitrage
# ---------------------------------------------------------------------------

def durrleman_butterfly_penalty(
    iv_pred:   torch.Tensor,   # (B, N_FLAT)  float32 or bf16
    grid:      GridConstants,
    mask:      torch.Tensor,   # (B, N_FLAT)  bool
) -> torch.Tensor:
    """
    Penalises violations of the Durrleman (2002) butterfly no-arb condition.

    In total implied variance space w(k, T) = σ(k, T)² · T, the Durrleman
    g-function must be non-negative:

        dw   = (w[:, 2:, :] − w[:, :-2, :]) / (2·dk)        (central diff)
        d²w  = (w[:, 2:, :] − 2·w[:, 1:-1, :] + w[:, :-2, :]) / dk²

        g(k, T) = (1 − k·dw / (2w))²  −  dw²/4 · (1/w + 1/4)  +  d²w/2

        L_bfly = mean( relu(−g)² )   [interior strikes only, valid triples]

    Computed only on interior strikes (indices 1..NK-2) using central differences
    on the uniform log-moneyness grid (dk = 0.025 for high41x14).

    Args:
        iv_pred: Network-predicted IV surface (B, NK*NT).
        grid:    GridConstants (must have uniform log_moneyness spacing).
        mask:    Valid-cell mask (B, NK*NT).

    Returns:
        Scalar penalty (float32).  Returns 0 if NK < 3.
    """
    if grid.NK < 3:
        return iv_pred.new_zeros(())

    B  = iv_pred.shape[0]
    NK = grid.NK
    NT = grid.NT

    sigma = iv_pred.float().view(B, NK, NT)
    T     = grid.maturities.to(iv_pred.device).view(1, 1, NT).float()
    k_vec = grid.log_moneyness.to(iv_pred.device).float()       # (NK,)
    m     = mask.float().view(B, NK, NT)

    # dk — assume uniform spacing
    dk = float(k_vec[1] - k_vec[0])

    # Total variance  (B, NK, NT)
    w = (sigma**2 * T).clamp(min=1e-8)   # floor prevents division by zero

    # Interior slice: indices 1 .. NK-2  →  size (B, NK-2, NT)
    w_im1 = w[:, :-2, :]   # w(k-1)
    w_i   = w[:, 1:-1, :]  # w(k)
    w_ip1 = w[:, 2:, :]    # w(k+1)

    k_int = k_vec[1:-1].view(1, NK - 2, 1)   # interior k values (B, NK-2, NT)

    # Finite-difference derivatives
    dw  = (w_ip1 - w_im1) / (2.0 * dk)             # (B, NK-2, NT)
    d2w = (w_ip1 - 2.0 * w_i + w_im1) / (dk**2)   # (B, NK-2, NT)

    # Durrleman g-function
    g = (1.0 - k_int * dw / (2.0 * w_i))**2 \
        - dw**2 / 4.0 * (1.0 / w_i + 0.25) \
        + 0.5 * d2w                                 # (B, NK-2, NT)

    # Valid only where all three participating strike cells are valid
    triple_valid = m[:, :-2, :] * m[:, 1:-1, :] * m[:, 2:, :]

    violations = F.relu(-g) ** 2
    n_valid    = triple_valid.sum().clamp(min=1.0)
    return (violations * triple_valid).sum() / n_valid


# ---------------------------------------------------------------------------
# PINN penalty 3: ATM term-structure slope loss (κ identifiability)
# ---------------------------------------------------------------------------

def atm_term_structure_penalty(
    iv_pred:   torch.Tensor,   # (B, N_FLAT)  float32 or bf16
    iv_target: torch.Tensor,   # (B, N_FLAT)  float32
    grid:      GridConstants,
    mask:      torch.Tensor,   # (B, N_FLAT)  bool
) -> torch.Tensor:
    """
    Penalises mismatch in the ATM implied-vol term structure slope.

    Motivation: κ (mean-reversion speed) controls how fast the IV surface
    transitions from v₀-dominated (short maturities) to θ-dominated (long
    maturities).  With a 124 bps baseline model error, κ has only ~84 bps
    of sensitivity signal — it becomes essentially unidentifiable.  Explicitly
    supervising the ATM term structure slope provides a direct gradient for κ.

    Definition:
        ATM_slope = ATM_IV(T_last) − ATM_IV(T_first)   [in IV units]

    We penalise the squared difference between predicted and target slopes,
    averaged over the batch.  Only included when both endpoint cells are valid.

        L_ts = mean( mask_0 * mask_last * (slope_pred - slope_target)^2 )

    ATM is defined as the strike index nearest to log-moneyness = 0.

    Args:
        iv_pred:   (B, N_FLAT) predicted IV surface.
        iv_target: (B, N_FLAT) ground-truth IV surface.
        grid:      GridConstants.
        mask:      (B, N_FLAT) valid-cell boolean mask.

    Returns:
        Scalar penalty (float32).
    """
    B  = iv_pred.shape[0]
    NK = grid.NK
    NT = grid.NT

    pred   = iv_pred.float().view(B, NK, NT)
    target = iv_target.float().view(B, NK, NT)
    m      = mask.float().view(B, NK, NT)

    # ATM index: strike with log-moneyness closest to 0
    lm = grid.log_moneyness.to(iv_pred.device).float()
    atm_ik = int(lm.abs().argmin().item())

    # ATM IV at first and last maturity
    pred_short  = pred[:, atm_ik, 0]   # (B,)
    pred_long   = pred[:, atm_ik, -1]  # (B,)
    target_short = target[:, atm_ik, 0]
    target_long  = target[:, atm_ik, -1]

    # Valid only when both endpoints are valid
    valid = m[:, atm_ik, 0] * m[:, atm_ik, -1]  # (B,)

    slope_pred   = pred_long   - pred_short    # (B,)
    slope_target = target_long - target_short  # (B,)

    sq_err = (slope_pred - slope_target) ** 2  # (B,)
    n_valid = valid.sum().clamp(min=1.0)
    return (sq_err * valid).sum() / n_valid


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class LossBreakdown(NamedTuple):
    total:          torch.Tensor
    vega:           torch.Tensor
    calendar:       torch.Tensor
    butterfly:      torch.Tensor
    term_structure: torch.Tensor


def total_loss(
    iv_pred:    torch.Tensor,   # (B, N_FLAT)  float32 or bf16
    iv_target:  torch.Tensor,   # (B, N_FLAT)  float32
    mask:       torch.Tensor,   # (B, N_FLAT)  bool
    weights:    torch.Tensor,   # (B, N_FLAT)  float32  vega weights
    grid:       GridConstants,
    lambda_cal:  float = 0.1,
    lambda_bfly: float = 0.05,
    lambda_ts:   float = 0.10,
    confidence:  torch.Tensor | None = None,  # (B, N_FLAT) float32 ∈ [0,1]  optional
    data_loss:   str = "vega",
) -> LossBreakdown:
    """
    Combined training loss:

        L = L_vega  +  λ_cal · L_calendar  +  λ_bfly · L_butterfly  +  λ_ts · L_ts

    where λ_cal, λ_bfly, and λ_ts should be warmed up from 0 over epochs 10-30
    to prevent the PINN terms from dominating before the data loss converges.
    λ_ts targets ATM term-structure slope matching to help κ identifiability.

    Args:
        iv_pred:     Network output.
        iv_target:   Ground-truth IVs.
        mask:        Valid-cell boolean mask.
        weights:     Precomputed vega weights.
        grid:        GridConstants.
        lambda_cal:  Weight for calendar-spread PINN penalty.
        lambda_bfly: Weight for butterfly PINN penalty.
        lambda_ts:   Weight for ATM term-structure slope penalty (κ identifiability).
        confidence:  Optional per-cell confidence weights (see vega_weighted_mse).

    Returns:
        LossBreakdown(total, vega, calendar, butterfly, term_structure).
        All fields are scalar tensors; call .item() for logging.
    """
    if data_loss == "vega":
        l_vega = vega_weighted_mse(iv_pred, iv_target, mask, weights, confidence=confidence)
    elif data_loss == "ivrmse":
        l_vega = masked_mse(iv_pred, iv_target, mask, confidence=confidence)
    elif data_loss == "log_ivrmse":
        l_vega = masked_log_mse(iv_pred, iv_target, mask, confidence=confidence)
    else:
        raise ValueError(f"Unknown data_loss: {data_loss}")
    l_cal   = calendar_spread_penalty(iv_pred, grid, mask)
    l_bfly  = durrleman_butterfly_penalty(iv_pred, grid, mask)
    l_ts    = atm_term_structure_penalty(iv_pred, iv_target, grid, mask)

    l_total = l_vega + lambda_cal * l_cal + lambda_bfly * l_bfly + lambda_ts * l_ts

    return LossBreakdown(
        total=l_total,
        vega=l_vega,
        calendar=l_cal,
        butterfly=l_bfly,
        term_structure=l_ts,
    )


# ---------------------------------------------------------------------------
# Dual-head loss: IV + parameter prediction (for identifiability)
# ---------------------------------------------------------------------------

def dual_head_loss(
    iv_pred:       torch.Tensor,   # (B, N_FLAT)  float32 or bf16
    param_pred:    torch.Tensor,   # (B, 5)       float32 or bf16 — 5 Heston params
    iv_target:     torch.Tensor,   # (B, N_FLAT)  float32
    param_target:  torch.Tensor,   # (B, 5)       float32 — 5 Heston params in [0,1]
    mask:          torch.Tensor,   # (B, N_FLAT)  bool
    weights:       torch.Tensor,   # (B, N_FLAT)  float32  vega weights
    grid:          GridConstants,
    lambda_param:  float = 0.1,
    lambda_cal:    float = 0.1,
    lambda_bfly:   float = 0.05,
    lambda_ts:     float = 0.10,
    confidence:    torch.Tensor | None = None,
    data_loss:     str = "vega",
) -> LossBreakdown:
    """
    Combined dual-head training loss:

        L = L_vega  +  λ_param · L_param  +  λ_cal · L_calendar
              +  λ_bfly · L_butterfly  +  λ_ts · L_ts

    The auxiliary parameter prediction head helps disambiguate the parameter space
    when calibrating from IV surfaces, improving robustness to local minima.

    Training with parameter supervision ensures the network learns the inverse
    mapping: IV surface → 5 Heston parameters. This guides calibration toward
    physically meaningful solutions. λ_ts targets ATM term-structure slope
    matching to directly supervise κ's effect on the term-structure shape.

    Args:
        iv_pred:       IV surface predictions (B, N_FLAT).
        param_pred:    Heston parameter predictions (B, 5) from auxiliary head.
                       Expected: [kappa, theta, sigma, rho, v0], all in [0,1].
        iv_target:     Ground-truth IV surface (B, N_FLAT).
        param_target:  Ground-truth Heston parameters (B, 5) normalised to [0,1].
        mask:          Valid-cell boolean mask (B, N_FLAT).
        weights:       Precomputed vega weights (B, N_FLAT).
        grid:          GridConstants.
        lambda_param:  Weight for parameter MSE term. Default 0.1.
        lambda_cal:    Weight for calendar-spread PINN. Default 0.1.
        lambda_bfly:   Weight for butterfly PINN. Default 0.05.
        lambda_ts:     Weight for ATM term-structure slope penalty (κ identifiability). Default 0.10.
        confidence:    Optional per-cell confidence weights.

    Returns:
        LossBreakdown(total, vega, calendar, butterfly, term_structure).
        Note: The 'vega' field includes IV MSE; parameter MSE is implicit in total.
    """
    # IV data loss (vega-weighted MSE or plain masked MSE)
    if data_loss == "vega":
        l_vega = vega_weighted_mse(iv_pred, iv_target, mask, weights, confidence=confidence)
    elif data_loss == "ivrmse":
        l_vega = masked_mse(iv_pred, iv_target, mask, confidence=confidence)
    elif data_loss == "log_ivrmse":
        l_vega = masked_log_mse(iv_pred, iv_target, mask, confidence=confidence)
    else:
        raise ValueError(f"Unknown data_loss: {data_loss}")

    # Parameter MSE loss: weighted by inverse IV sensitivity.
    # κ has the weakest IV fingerprint (~84 bps range); ρ has the strongest (~400 bps).
    # Upweighting high-sensitivity params improves calibration of the slow-learning κ.
    # Order must match training data: [kappa, theta, sigma_v, rho, v0].
    param_pred_f = param_pred.float()
    param_target_f = param_target.float()
    _PARAM_WEIGHTS = torch.tensor([1.0, 3.0, 2.5, 5.0, 3.0], device=param_pred_f.device)
    sq_err = (param_pred_f - param_target_f) ** 2
    l_param = (sq_err * _PARAM_WEIGHTS).mean()

    # PINN penalties (as before)
    l_cal  = calendar_spread_penalty(iv_pred, grid, mask)
    l_bfly = durrleman_butterfly_penalty(iv_pred, grid, mask)
    l_ts   = atm_term_structure_penalty(iv_pred, iv_target, grid, mask)

    # Combined loss with parameter regularization and term-structure penalty
    l_total = (l_vega + lambda_param * l_param + lambda_cal * l_cal
               + lambda_bfly * l_bfly + lambda_ts * l_ts)

    return LossBreakdown(
        total=l_total,
        vega=l_vega,  # Contains IV MSE; param MSE is implicit in l_total
        calendar=l_cal,
        butterfly=l_bfly,
        term_structure=l_ts,
    )


# ---------------------------------------------------------------------------
# IV RMSE in basis points  (reporting metric)
# ---------------------------------------------------------------------------

def ivrmse_bps(
    iv_pred:   torch.Tensor,   # (B, N_FLAT)
    iv_target: torch.Tensor,   # (B, N_FLAT)
    mask:      torch.Tensor,   # (B, N_FLAT)  bool
) -> float:
    """
    IV root-mean-square error in basis points (1 bp = 0.0001 = 0.01%).

        IVRMSE_bps = sqrt( mean_valid( (σ_NN − σ_target)² ) ) × 10_000

    Target for academic deep-calibration work: < 10 bps.
    """
    with torch.no_grad():
        pred   = iv_pred.float()
        target = iv_target.float()
        m      = mask.float()
        mse = ((pred - target) ** 2 * m).sum() / m.sum().clamp(min=1.0)
        return float(mse.sqrt().item()) * 10_000.0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .network import BatesSurrogate, GridConstants

    torch.manual_seed(0)
    grid  = GridConstants.default()
    B     = 8
    N     = grid.N_FLAT

    # Random "prediction" and "target"
    pred   = torch.rand(B, N) * 0.3 + 0.1   # plausible IVs
    target = torch.rand(B, N) * 0.3 + 0.1
    mask   = torch.ones(B, N, dtype=torch.bool)
    r      = torch.full((B,), 0.03)
    q      = torch.full((B,), 0.01)

    # Vega weights
    w = compute_vega_weights(target, grid, r, q)
    print(f"Vega weights  shape={w.shape}  min={w.min():.4f}  max={w.max():.4f}  mean={w.mean():.4f}")
    assert abs(w.mean().item() - 1.0) < 0.1, "weights should be ~1 on average"

    # Vega weights with floor
    w_floor = compute_vega_weights(target, grid, r, q, weight_floor=0.05)
    print(f"Floored weights  min={w_floor.min():.4f}  max={w_floor.max():.4f}  mean={w_floor.mean():.4f}")
    assert w_floor.min().item() >= 0.05 * 0.9, "floored weights should respect floor after renorm"
    assert abs(w_floor.mean().item() - 1.0) < 0.1, "floored weights should still be ~1 mean"

    # Losses
    lv  = vega_weighted_mse(pred, target, mask, w)
    lc  = calendar_spread_penalty(pred, grid, mask)
    lb  = durrleman_butterfly_penalty(pred, grid, mask)
    print(f"L_vega={lv:.6f}  L_cal={lc:.6f}  L_bfly={lb:.6f}")

    # Zero loss when pred == target
    lv0 = vega_weighted_mse(target, target, mask, w)
    assert lv0.item() < 1e-10, "L_vega should be 0 when pred == target"

    # IVRMSE
    rmse = ivrmse_bps(pred, target, mask)
    print(f"IVRMSE = {rmse:.2f} bps")

    print("loss.py: all checks passed.")

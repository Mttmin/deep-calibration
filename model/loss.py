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

    # Forward differences along T (B, NK, NT-1)
    dtv        = tv[:, :, 1:] - tv[:, :, :-1]
    pair_valid = m[:, :, :-1] * m[:, :, 1:]   # valid only if both cells valid

    violations = F.relu(-dtv) ** 2             # hinge²  (B, NK, NT-1)
    n_valid    = pair_valid.sum().clamp(min=1.0)
    return (violations * pair_valid).sum() / n_valid


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
# Combined loss
# ---------------------------------------------------------------------------

class LossBreakdown(NamedTuple):
    total:    torch.Tensor
    vega:     torch.Tensor
    calendar: torch.Tensor
    butterfly: torch.Tensor


def total_loss(
    iv_pred:    torch.Tensor,   # (B, N_FLAT)  float32 or bf16
    iv_target:  torch.Tensor,   # (B, N_FLAT)  float32
    mask:       torch.Tensor,   # (B, N_FLAT)  bool
    weights:    torch.Tensor,   # (B, N_FLAT)  float32  vega weights
    grid:       GridConstants,
    lambda_cal:  float = 0.1,
    lambda_bfly: float = 0.05,
    confidence:  torch.Tensor | None = None,  # (B, N_FLAT) float32 ∈ [0,1]  optional
) -> LossBreakdown:
    """
    Combined training loss:

        L = L_vega  +  λ_cal · L_calendar  +  λ_bfly · L_butterfly

    where λ_cal and λ_bfly should be warmed up from 0 over epochs 10-30
    to prevent the PINN terms from dominating before the data loss converges.

    Args:
        iv_pred:     Network output.
        iv_target:   Ground-truth IVs.
        mask:        Valid-cell boolean mask.
        weights:     Precomputed vega weights.
        grid:        GridConstants.
        lambda_cal:  Weight for calendar-spread PINN penalty.
        lambda_bfly: Weight for butterfly PINN penalty.
        confidence:  Optional per-cell confidence weights (see vega_weighted_mse).

    Returns:
        LossBreakdown(total, vega, calendar, butterfly).
        All fields are scalar tensors; call .item() for logging.
    """
    l_vega  = vega_weighted_mse(iv_pred, iv_target, mask, weights, confidence=confidence)
    l_cal   = calendar_spread_penalty(iv_pred, grid, mask)
    l_bfly  = durrleman_butterfly_penalty(iv_pred, grid, mask)

    l_total = l_vega + lambda_cal * l_cal + lambda_bfly * l_bfly

    return LossBreakdown(
        total=l_total,
        vega=l_vega,
        calendar=l_cal,
        butterfly=l_bfly,
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

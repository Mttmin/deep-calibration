"""
model/calibrate.py
==================
Online calibration: recover Bates/Heston parameters from a market IV surface
by running L-BFGS *through* the frozen surrogate network.

This is the "two-step" approach of Horvath et al. (2021):
  1. BatesSurrogate trained offline  θ → Σ(θ)
  2. At inference: solve  θ* = argmin_θ ‖N_w(θ) − Σ_market‖²_W  via L-BFGS.

Key design decisions
--------------------
* Box projection (clamp to [0,1]) is applied **after** optimizer.step,
  not inside the closure.  Doing it inside the closure interferes with the
  strong-Wolfe line search.
* Multi-start with Sobol quasi-random initialisations + centroid [0.5,…].
* All optimisation is in normalised parameter space [0,1]^10.
* Denormalisation uses the exact physical bounds from heston_datagen.py.

References
----------
  deep_calibration_research.tex §6.3 (lines 1154-1179)
  Horvath, Muguruza & Tomas (2021) arXiv:1901.09647  §4
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import qmc

from .network import BatesSurrogate, GridConstants
from .loss import compute_vega_weights, ivrmse_bps


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

# Physical bounds match PARAM_BOUNDS in heston_datagen.py
# Row order: kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j, r, q
PARAM_BOUNDS_PHYSICAL = torch.tensor(
    [
        [0.10, 10.0],    # kappa
        [0.02,  0.25],   # theta
        [0.05,  2.00],   # sigma_v
        [-0.98,  0.10],  # rho
        [0.02,  0.25],   # v0
        [0.00,  4.00],   # lambda_j
        [-0.30,  0.00],  # mu_j
        [0.01,  0.45],   # sigma_j
        [0.00,  0.06],   # r
        [0.00,  0.04],   # q
    ],
    dtype=torch.float32,
)

PARAM_NAMES = [
    "kappa", "theta", "sigma_v", "rho", "v0",
    "lambda_j", "mu_j", "sigma_j", "r", "q",
]

# Wing clipping bounds for calibration active zone (Change 4)
# Cells with log-moneyness outside [WING_LO, WING_HI] are zeroed from eff_w.
# These extreme strikes are almost always outside the real market quote range.
WING_LO: float = -0.50
WING_HI: float =  0.30


def denormalize(theta_norm: torch.Tensor) -> torch.Tensor:
    """
    Maps normalised parameters from [0,1]^10 to physical units.

    Args:
        theta_norm: (..., 10) float32 in [0, 1].
    Returns:
        theta_raw:  (..., 10) float32 in physical units.
    """
    bounds = PARAM_BOUNDS_PHYSICAL.to(theta_norm.device)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return theta_norm * (hi - lo) + lo


def normalize(theta_raw: torch.Tensor) -> torch.Tensor:
    """
    Maps physical parameters to normalised [0,1]^10 space.

    Args:
        theta_raw: (..., 10) float32 in physical units.
    Returns:
        theta_norm: (..., 10) float32 in [0, 1].
    """
    bounds = PARAM_BOUNDS_PHYSICAL.to(theta_raw.device)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return (theta_raw - lo) / (hi - lo + 1e-12)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibrateResult:
    """Result of one L-BFGS calibration."""
    theta_norm:  torch.Tensor   # (10,) normalised ∈ [0, 1]
    theta_raw:   torch.Tensor   # (10,) physical units
    iv_fitted:   torch.Tensor   # (N_FLAT,) surrogate IV at calibrated params
    loss_final:  float          # final vega-weighted MSE
    ivrmse_bps:  float          # IV RMSE in basis points
    n_iters:     int            # L-BFGS iterations used
    converged:   bool           # True if loss < convergence_tol


# ---------------------------------------------------------------------------
# Multi-start initialisation
# ---------------------------------------------------------------------------

def _make_theta_inits(
    n_restarts: int,
    device:     torch.device,
    seed:       int = 0,
    guided_bank_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns (n_restarts, 10) initialisations in [0.1, 0.9].
    The first row is always the parameter centroid [0.5, …, 0.5].
    Remaining rows are drawn from guided_bank_norm (if supplied) then Sobol.
    """
    inits = []

    # Always include the centroid
    inits.append(torch.full((10,), 0.5, device=device))

    remaining = n_restarts - 1
    if guided_bank_norm is not None and remaining > 0 and len(guided_bank_norm) > 0:
        rng = np.random.default_rng(seed)
        n_bank = min(remaining, len(guided_bank_norm))
        idx = rng.integers(0, len(guided_bank_norm), size=n_bank)
        inits.append(guided_bank_norm[idx].to(device).float())
        remaining -= n_bank

    if remaining > 0:
        sampler = qmc.Sobol(d=10, scramble=True, seed=seed)
        raw = sampler.random(remaining).astype("float32")   # (remaining, 10)
        # Scale to [0.1, 0.9] to avoid boundary effects
        raw = 0.1 + 0.8 * raw
        inits.append(torch.from_numpy(raw).to(device))

    return torch.cat([t.unsqueeze(0) if t.dim() == 1 else t for t in inits], dim=0)


# ---------------------------------------------------------------------------
# Single-surface calibration
# ---------------------------------------------------------------------------

def calibrate_single(
    model:        BatesSurrogate,
    iv_market:    torch.Tensor,          # (N_FLAT,) float32  observed IV surface
    mask_market:  torch.Tensor,          # (N_FLAT,) bool     valid market cells
    grid:         GridConstants,
    theta_init:   torch.Tensor | None = None,  # (10,) or None → centroid
    r_norm:       float = 0.5,           # normalised r for vega weights
    q_norm:       float = 0.5,           # normalised q for vega weights
    n_restarts:   int  = 3,
    max_iter:     int  = 100,
    convergence_tol: float = 1e-7,
    device:       torch.device | None = None,
    raw_mask_market:    torch.Tensor | None = None,  # (N_FLAT,) bool  raw interp mask
    confidence_market:  torch.Tensor | None = None,  # (N_FLAT,) float32  [0,1]
    guided_bank_norm:   torch.Tensor | None = None,  # (M, 10) float32  pre-normalised
) -> CalibrateResult:
    """
    Calibrate Bates/Heston parameters to a single market IV surface.

    Solves:
        θ* = argmin_{θ ∈ [0,1]^10} ‖N_w(θ) − Σ_market‖²_W

    via multi-start L-BFGS (strong Wolfe line search) through the frozen
    surrogate.  Box constraints are enforced by projecting back to [0,1]^10
    after each L-BFGS step.

    Args:
        model:            Trained BatesSurrogate (frozen; no gradients on weights).
        iv_market:        (N_FLAT,) observed implied-volatility surface.
        mask_market:      (N_FLAT,) boolean mask of valid (non-NaN) market cells.
        grid:             GridConstants.
        theta_init:       (10,) optional warm-start in normalised space [0,1].
        r_norm:           Normalised risk-free rate for vega weight computation.
        q_norm:           Normalised dividend yield.
        n_restarts:       Number of L-BFGS restarts (including warm-start / centroid).
        max_iter:         Maximum L-BFGS iterations per restart.
        convergence_tol:  Stop early if loss < this threshold.
        device:           Computation device.  Defaults to iv_market.device.
        raw_mask_market:  (N_FLAT,) boolean mask of cells with direct market quotes
                          (not extrapolated). When provided, only these cells enter
                          the calibration loss.  Default: use mask_market.
        confidence_market:(N_FLAT,) per-cell confidence in [0,1].  Multiplied into
                          effective weights so high-spread / illiquid cells are
                          down-weighted.  Default: uniform 1.0.
        guided_bank_norm: (M, 10) float32 pre-normalised parameter bank.  Used to
                          seed multi-start restarts with market-realistic inits
                          instead of pure Sobol.  Default: Sobol only.

    Returns:
        CalibrateResult with the best (lowest-loss) result across all restarts.
    """
    if device is None:
        device = iv_market.device if iv_market.is_cuda else torch.device("cpu")

    model = model.to(device).eval()
    iv_market   = iv_market.float().to(device)
    mask_market = mask_market.to(device)

    # Vega weights for the calibration loss
    r = torch.tensor([r_norm * 0.06], device=device)
    q = torch.tensor([q_norm * 0.04], device=device)
    iv_m_2d = iv_market.unsqueeze(0)            # (1, N_FLAT)
    vega_w   = compute_vega_weights(iv_m_2d, grid, r, q).squeeze(0)   # (N_FLAT,)

    # Active mask: prefer raw market quotes over extrapolated fills (Change 2)
    active_mask = raw_mask_market.to(device) if raw_mask_market is not None else mask_market
    conf_w = confidence_market.float().to(device).clamp(0.0, 1.0) \
             if confidence_market is not None \
             else torch.ones(grid.N_FLAT, device=device)
    eff_w = vega_w * active_mask.float() * conf_w

    # Wing mask: zero extreme strikes not covered by real market quotes (Change 4)
    log_m_flat = grid.log_moneyness.to(device).unsqueeze(1).expand(grid.NK, grid.NT).reshape(-1)
    wing_mask = (log_m_flat >= WING_LO) & (log_m_flat <= WING_HI)
    eff_w = eff_w * wing_mask.float()

    # Initialisation points (Change 3: seed from guided bank when available)
    if theta_init is not None:
        extra = max(0, n_restarts - 1)
        extra_inits = _make_theta_inits(extra, device, guided_bank_norm=guided_bank_norm)[1:]
        all_inits = torch.cat(
            [theta_init.unsqueeze(0).to(device), extra_inits], dim=0
        ) if extra > 0 else theta_init.unsqueeze(0).to(device)
    else:
        all_inits = _make_theta_inits(n_restarts, device, guided_bank_norm=guided_bank_norm)

    best: CalibrateResult | None = None

    for i_restart in range(len(all_inits)):
        theta = all_inits[i_restart].clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [theta], lr=1.0, max_iter=max_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )
        iter_count = [0]

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            iv_pred = model(theta.unsqueeze(0)).squeeze(0)  # (N_FLAT,)
            err     = iv_pred.float() - iv_market
            loss    = (eff_w * err**2).sum() / eff_w.sum().clamp(min=1.0)
            loss.backward()
            iter_count[0] += 1
            return loss

        optimizer.step(closure)

        # Box projection: clamp to [0,1] after L-BFGS step
        with torch.no_grad():
            theta.clamp_(0.0, 1.0)

        # Final evaluation at projected point
        with torch.no_grad():
            iv_fitted = model(theta.unsqueeze(0)).squeeze(0).float()
            err    = iv_fitted - iv_market
            loss_f = float((eff_w * err**2).sum().item() / eff_w.sum().clamp(min=1.0).item())
            rmse   = float(ivrmse_bps(
                iv_fitted.unsqueeze(0), iv_market.unsqueeze(0), mask_market.unsqueeze(0)
            ))

        result = CalibrateResult(
            theta_norm = theta.detach().clone(),
            theta_raw  = denormalize(theta.detach()),
            iv_fitted  = iv_fitted.detach(),
            loss_final = loss_f,
            ivrmse_bps = rmse,
            n_iters    = iter_count[0],
            converged  = loss_f < convergence_tol,
        )

        if best is None or result.loss_final < best.loss_final:
            best = result

        if best.converged:
            break   # early exit if tolerances met

    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Batch calibration
# ---------------------------------------------------------------------------

def calibrate_batch(
    model:        BatesSurrogate,
    iv_market_batch: torch.Tensor,       # (N, N_FLAT) float32
    mask_batch:      torch.Tensor,       # (N, N_FLAT) bool
    grid:         GridConstants,
    theta_init_batch: torch.Tensor | None = None,  # (N, 10) float32
    r_norm_batch:  Sequence[float] | None = None,
    q_norm_batch:  Sequence[float] | None = None,
    n_restarts:    int = 3,
    max_iter:      int = 100,
    device:        torch.device | None = None,
    verbose:       bool = False,
    raw_mask_batch:     torch.Tensor | None = None,  # (N, N_FLAT) bool
    confidence_batch:   torch.Tensor | None = None,  # (N, N_FLAT) float32
    guided_bank_norm:   torch.Tensor | None = None,  # (M, 10) float32
) -> list[CalibrateResult]:
    """
    Calibrate a batch of N market IV surfaces.

    L-BFGS does not support true vectorised multi-objective optimisation,
    so this function iterates over each surface sequentially.

    Args:
        model:            Trained surrogate (frozen).
        iv_market_batch:  (N, N_FLAT) market IV surfaces.
        mask_batch:       (N, N_FLAT) valid-cell masks.
        grid:             GridConstants.
        theta_init_batch: (N, 10) optional warm-start parameters.
        r_norm_batch:     List of N normalised r values (default 0.5 each).
        q_norm_batch:     List of N normalised q values (default 0.5 each).
        n_restarts:       L-BFGS multi-start restarts per surface.
        max_iter:         Max L-BFGS iterations per restart.
        device:           Computation device.
        verbose:          Print progress per surface.
        raw_mask_batch:   (N, N_FLAT) raw market quote masks (see calibrate_single).
        confidence_batch: (N, N_FLAT) per-cell confidence weights.
        guided_bank_norm: (M, 10) normalised guided parameter bank.

    Returns:
        List of N CalibrateResult objects.
    """
    N = iv_market_batch.shape[0]
    results = []

    r_list = r_norm_batch if r_norm_batch is not None else [0.5] * N
    q_list = q_norm_batch if q_norm_batch is not None else [0.5] * N

    for i in range(N):
        init_i = theta_init_batch[i] if theta_init_batch is not None else None
        raw_mask_i   = raw_mask_batch[i]   if raw_mask_batch   is not None else None
        confidence_i = confidence_batch[i] if confidence_batch is not None else None
        res = calibrate_single(
            model               = model,
            iv_market           = iv_market_batch[i],
            mask_market         = mask_batch[i],
            grid                = grid,
            theta_init          = init_i,
            r_norm              = float(r_list[i]),
            q_norm              = float(q_list[i]),
            n_restarts          = n_restarts,
            max_iter            = max_iter,
            device              = device,
            raw_mask_market     = raw_mask_i,
            confidence_market   = confidence_i,
            guided_bank_norm    = guided_bank_norm,
        )
        results.append(res)
        if verbose:
            print(f"  surface {i+1}/{N}  ivrmse={res.ivrmse_bps:.2f} bps  "
                  f"converged={res.converged}")

    return results


# ---------------------------------------------------------------------------
# CLI  (quick calibration demo against a val-set surface)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training data creation"))
    from heston_datagen import BatesDataset   # type: ignore[import]

    ap = argparse.ArgumentParser(description="Calibrate surrogate to a val-set surface.")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained model checkpoint (best.pt)")
    ap.add_argument("--h5",         type=str, required=True,
                    help="HDF5 dataset (for val surface + grid)")
    ap.add_argument("--n-surfaces", type=int, default=5,
                    help="Number of val surfaces to calibrate")
    ap.add_argument("--n-restarts", type=int, default=3)
    ap.add_argument("--max-iter",   type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BatesSurrogate.from_checkpoint(args.checkpoint).to(device).eval()
    grid  = GridConstants.from_h5(args.h5).to(device)

    ds = BatesDataset(args.h5, min_valid_cells=0)
    N  = len(ds)
    val_start = int(N * 0.9)

    print(f"\nCalibrating {args.n_surfaces} surfaces from val set (idx {val_start}+)…\n")
    for i in range(args.n_surfaces):
        params_norm, iv_flat, mask_flat = ds[val_start + i]

        # Ground-truth normalised parameters
        r_norm = float(params_norm[8].item())
        q_norm = float(params_norm[9].item())

        res = calibrate_single(
            model       = model,
            iv_market   = iv_flat.to(device),
            mask_market = mask_flat.to(device),
            grid        = grid,
            r_norm      = r_norm,
            q_norm      = q_norm,
            n_restarts  = args.n_restarts,
            max_iter    = args.max_iter,
            device      = device,
        )

        # Ground-truth physical params
        gt_raw = denormalize(params_norm.to(device))
        print(f"Surface {i+1}/{args.n_surfaces}")
        print(f"  IVRMSE = {res.ivrmse_bps:.2f} bps  |  "
              f"loss = {res.loss_final:.6f}  |  "
              f"iters = {res.n_iters}  |  converged = {res.converged}")
        print(f"  Calibrated params (physical):")
        for j, name in enumerate(PARAM_NAMES):
            print(f"    {name:12s}  gt={gt_raw[j]:.4f}  calib={res.theta_raw[j]:.4f}")
        print()

"""Realistic 4/2 parameter sampling.

Same equity prior as the Heston v2 sampler, plus the two volatility loadings.
``b = 0`` is included so the training set contains the Heston embedding.
``b > 0`` is kept small: the vol floor is ``2 sqrt(a b)``, and a large floor
flattens the smile the CTMC pricer is asked to match.

Feller is enforced strictly (``2 kappa theta >= 1.05 sigma^2``). The 4/2
transform needs ``int 1/v`` finite, which fails on the Feller boundary.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import truncnorm

KAPPA_LO, KAPPA_HI = 0.30, 10.0
THETA_LO, THETA_HI = 0.01, 0.16
SIGMA_LO, SIGMA_HI = 0.10, 1.50
RHO_MU, RHO_SD = -0.60, 0.30
RHO_LO, RHO_HI = -0.90, -0.30
V0_LO, V0_HI = 0.02, 0.20
A_LO, A_HI = 0.40, 1.60
B_LO, B_HI = 0.00, 0.04
B_POS_LO = 1.0e-4
R_LO, R_HI = 0.00, 0.05

# Spot vol a*sqrt(v0) + b/sqrt(v0) kept inside an equity band so IV inversion
# does not spend the dataset on 5% or 150% names.
SPOT_VOL_LO, SPOT_VOL_HI = 0.08, 0.65
# 2*sqrt(a*b) above this eats the whole smile.
FLOOR_VOL_MAX = 0.25
FELLER_MARGIN = 1.05

PARAM_NAMES = ["kappa", "theta", "sigma_v", "rho", "v0", "a", "b", "r"]
N_MODEL = 7  # calibrated: kappa .. b
D = len(PARAM_NAMES)

PARAM_LO = np.array(
    [KAPPA_LO, THETA_LO, SIGMA_LO, RHO_LO, V0_LO, A_LO, B_LO], dtype=np.float64
)
PARAM_HI = np.array(
    [KAPPA_HI, THETA_HI, SIGMA_HI, RHO_HI, V0_HI, A_HI, B_HI], dtype=np.float64
)


def normalise_params(params: np.ndarray) -> np.ndarray:
    """Map physical (..., 7) or (..., 8) rows onto [0, 1].

    Eight-column input keeps ``r`` in the last slot, normalised on
    ``[R_LO, R_HI]``. Seven-column input normalises the calibrated block only.
    """
    params = np.asarray(params, dtype=np.float64)
    out = np.empty_like(params)
    out[..., :7] = (params[..., :7] - PARAM_LO) / (PARAM_HI - PARAM_LO)
    if params.shape[-1] == 8:
        out[..., 7] = (params[..., 7] - R_LO) / (R_HI - R_LO)
    return out


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size=n))


def sample_realistic_params(n: int, seed: int) -> np.ndarray:
    """Return ``(n, 8)`` float64 rows: kappa, theta, sigma_v, rho, v0, a, b, r."""
    rng = np.random.default_rng(seed)
    rho_a, rho_b = (RHO_LO - RHO_MU) / RHO_SD, (RHO_HI - RHO_MU) / RHO_SD
    out = np.empty((n, D), dtype=np.float64)
    filled = 0
    # Oversample. Feller plus the spot-vol gate reject roughly half to two thirds.
    batch = max(n * 4, 4096)
    while filled < n:
        kappa = _log_uniform(rng, KAPPA_LO, KAPPA_HI, batch)
        theta = _log_uniform(rng, THETA_LO, THETA_HI, batch)
        sigma = rng.uniform(SIGMA_LO, SIGMA_HI, size=batch)
        rho = truncnorm.rvs(rho_a, rho_b, loc=RHO_MU, scale=RHO_SD, size=batch, random_state=rng)
        v0 = _log_uniform(rng, V0_LO, V0_HI, batch)
        # Dense near a=1 so the Heston scale is not a rare corner.
        a = np.empty(batch)
        near = rng.random(batch) < 0.35
        a[near] = rng.uniform(0.85, 1.15, size=int(near.sum()))
        a[~near] = _log_uniform(rng, A_LO, A_HI, int((~near).sum()))
        a = np.clip(a, A_LO, A_HI)
        # 20% exact Heston embedding. The rest is a small 3/2 loading.
        b = np.zeros(batch)
        pos = rng.random(batch) >= 0.20
        b[pos] = _log_uniform(rng, B_POS_LO, B_HI, int(pos.sum()))
        r = rng.uniform(R_LO, R_HI, size=batch)

        feller = 2.0 * kappa * theta >= FELLER_MARGIN * sigma * sigma
        sqrt_v0 = np.sqrt(v0)
        spot_vol = a * sqrt_v0 + b / sqrt_v0
        floor_vol = 2.0 * np.sqrt(a * b)
        ok = (
            feller
            & (spot_vol >= SPOT_VOL_LO)
            & (spot_vol <= SPOT_VOL_HI)
            & (floor_vol <= FLOOR_VOL_MAX)
            & np.isfinite(spot_vol)
        )
        take = np.flatnonzero(ok)
        if take.size == 0:
            batch *= 2
            continue
        n_take = min(take.size, n - filled)
        rows = np.column_stack(
            (kappa[take[:n_take]], theta[take[:n_take]], sigma[take[:n_take]],
             rho[take[:n_take]], v0[take[:n_take]], a[take[:n_take]],
             b[take[:n_take]], r[take[:n_take]])
        )
        out[filled:filled + n_take] = rows
        filled += n_take
    return out

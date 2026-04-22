"""
Phase 3A: Realistic Heston parameter sampling (Feller-satisfying, equity-like).
==============================================================================

Supersedes the uniform / Latin-hypercube sampling in `heston_datagen.py` that
produced 41.4% Feller-violating sets (Phase 0C report) and, combined with
the out-of-regime FJL invocation, drove the Phase 2A failure on 2026-04-18.

Sampling policy (SCENARIO_PLAN.md L275-284, rho tightened per user direction):

    kappa    ~ log-uniform [0.3, 10.0]
    theta    ~ log-uniform [0.01, 0.16]          (long-run vol 10%..40%)
    sigma_v  ~ uniform     [0.10, 1.50]  with rejection on 2 kappa theta >= sigma^2
    rho      ~ truncated-normal(mu=-0.6, sd=0.3), clipped to [-0.90, -0.30]
    v0       ~ log-uniform [0.02, 0.20]          (short-run vol ~14%..45%)
    r        ~ uniform     [0.00, 0.05]          (unchanged from heston_datagen)

Rejection is IID, not Sobol: once Feller culls ~50% of candidates, Sobol low-
discrepancy is destroyed anyway, so we use a plain PRNG and oversample. Phase
3B (training data regeneration) can layer stratified rejection on top if QMC
properties are needed there; for Phase 2A (N <= 100) IID is sufficient.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import truncnorm

KAPPA_LO, KAPPA_HI = 0.30, 10.0
THETA_LO, THETA_HI = 0.01, 0.16
SIGMA_LO, SIGMA_HI = 0.10, 1.50
RHO_MU, RHO_SD     = -0.60, 0.30
RHO_LO, RHO_HI     = -0.90, -0.30
# Phase 2A-v1 (2026-04-18) MC validation revealed a heavy tail in cos_standard
# rel-err driven entirely by samples with v0 < 0.008: at sqrt(v0) ~ 7%, COS
# truncation half_abs=5 is too small for short-T deep-OTM cells. Tightening
# V0_LO from 0.005 to 0.02 (sqrt(v0) >= 14% short-run vol floor) collapses
# that tail without restricting long-run vol coverage (theta range unchanged).
V0_LO, V0_HI       = 0.02, 0.20
R_LO, R_HI         = 0.00, 0.05

PARAM_NAMES = ["kappa", "theta", "sigma_v", "rho", "v0", "r"]
D: int = len(PARAM_NAMES)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size=n))


def sample_realistic_params(
    n: int,
    seed: int,
    max_rejection_batches: int = 64,
    batch_mult: float = 4.0,
) -> np.ndarray:
    """Return (n, 6) float64 of Heston params satisfying the realistic policy.

    Columns: kappa, theta, sigma_v, rho, v0, r (PARAM_NAMES order).

    Raises RuntimeError if Feller rejection fails to yield n samples after
    `max_rejection_batches` attempts (should never trigger in practice —
    acceptance rate is ~40-60% across the box).
    """
    if n <= 0:
        return np.empty((0, D), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    a_std = (RHO_LO - RHO_MU) / RHO_SD
    b_std = (RHO_HI - RHO_MU) / RHO_SD
    batch_size = max(256, int(math.ceil(n * batch_mult)))

    out = np.empty((0, D), dtype=np.float64)
    for _ in range(max_rejection_batches):
        kappa = _log_uniform(rng, KAPPA_LO, KAPPA_HI, batch_size)
        theta = _log_uniform(rng, THETA_LO, THETA_HI, batch_size)
        sigma = rng.uniform(SIGMA_LO, SIGMA_HI, size=batch_size)
        rho = truncnorm.rvs(
            a_std, b_std, loc=RHO_MU, scale=RHO_SD,
            size=batch_size, random_state=rng,
        )
        v0 = _log_uniform(rng, V0_LO, V0_HI, batch_size)
        r = rng.uniform(R_LO, R_HI, size=batch_size)

        # Feller: 2 kappa theta >= sigma^2
        mask = 2.0 * kappa * theta >= sigma * sigma
        if not mask.any():
            continue
        batch = np.column_stack(
            [kappa[mask], theta[mask], sigma[mask], rho[mask], v0[mask], r[mask]]
        )
        out = np.vstack([out, batch]) if out.size else batch
        if out.shape[0] >= n:
            return out[:n].astype(np.float64, copy=True)

    raise RuntimeError(
        f"Realistic sampling failed to collect {n} Feller-satisfying samples "
        f"after {max_rejection_batches} batches of {batch_size}."
    )


def acceptance_rate_probe(n_probe: int = 100_000, seed: int = 20260418) -> dict:
    """Diagnostic: Feller acceptance rate on a large pre-rejection batch."""
    rng = np.random.default_rng(seed)
    kappa = _log_uniform(rng, KAPPA_LO, KAPPA_HI, n_probe)
    theta = _log_uniform(rng, THETA_LO, THETA_HI, n_probe)
    sigma = rng.uniform(SIGMA_LO, SIGMA_HI, size=n_probe)
    feller_ok = (2.0 * kappa * theta >= sigma * sigma)
    return dict(
        n_probe=n_probe,
        acceptance_rate=float(feller_ok.mean()),
        feller_ratio_median=float(np.median(2 * kappa * theta / (sigma * sigma))),
    )


if __name__ == "__main__":
    print("Feller acceptance probe:", acceptance_rate_probe())
    x = sample_realistic_params(50, seed=20260418)
    print("shape:", x.shape)
    for j, name in enumerate(PARAM_NAMES):
        print(f"  {name:8s}  min={x[:, j].min():.4f}  "
              f"med={np.median(x[:, j]):.4f}  max={x[:, j].max():.4f}")
    feller = 2.0 * x[:, 0] * x[:, 1] / (x[:, 2] ** 2)
    print(f"  Feller ratio median = {np.median(feller):.3f}, "
          f"min = {feller.min():.3f}")
    assert (feller >= 1.0).all(), "Feller rejection broken"

"""Grasselli (2017) 4/2 characteristic function.

The variance driver is CIR. Instantaneous volatility is ``a sqrt(v) + b / sqrt(v)``.
``b = 0`` is Heston with variance factor ``a**2 * v`` (``a = 1`` is plain Heston).
``a = 0`` is the 3/2 model. ``b > 0`` requires the Feller condition
``2 kappa theta >= sigma**2`` so that ``int 1/v`` stays finite.

``four_two_cf`` returns the characteristic function of ``log(S_T / S_0)``, the
same object the Heston COS pricer consumes. The Grasselli transform is of
``log(S_T / F_0)``; the forward drift ``exp(i u (r - q) T)`` is put back on.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import loggamma


def four_two_log_return_cf(
    u: np.ndarray,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    a: float,
    b: float,
) -> np.ndarray:
    """CF of ``log(S_T / F_0)`` under the 4/2 model. ``u`` is real or complex."""
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u
    kth = kappa * theta - 0.5 * sigma * sigma

    psi = 0.5 * (iu + u * u * (1.0 - rho * rho))
    mu = psi * a * a - iu * rho * a * kappa / sigma
    eta = psi * b * b + iu * rho * b * kth / sigma
    lam = -iu * rho * a / sigma
    gam = iu * rho * b / sigma
    const = (
        -2.0 * a * b * psi * T
        - iu * rho * (a / sigma) * (v0 + kappa * theta * T)
        + iu * rho * (b / sigma) * (kappa * T - np.log(v0))
    )

    kt = _sqrt_pos_real(kappa * kappa + 2.0 * sigma * sigma * mu)
    k1 = (kt - kappa) / (sigma * sigma)
    if b == 0.0:
        d_dim = np.full(u.shape, kth, dtype=np.complex128)
    else:
        d_dim = _sqrt_pos_real(kth * kth + 2.0 * eta * sigma * sigma)
    b1 = (d_dim - kth) / (sigma * sigma)

    log_c = np.log(0.25 * sigma * sigma) + np.log(-np.expm1(-kt * T)) - np.log(kt)
    half_d = 1.0 + 2.0 * d_dim / (sigma * sigma)
    zeta = v0 * np.exp(-kt * T - log_c)
    one_2s = 1.0 + 2.0 * (lam - k1) * np.exp(log_c)
    p_pow = gam - b1
    series_a = half_d + p_pow
    z = zeta / (2.0 * one_2s)

    log_moment = _log_kummer_moment(series_a, half_d, z, zeta, p_pow, log_c, one_2s)
    return np.exp(
        const
        - k1 * (v0 + kappa * theta * T)
        - b1 * (kt * T - np.log(v0))
        + log_moment
    )


def four_two_cf(
    u: np.ndarray,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    a: float,
    b: float,
    r: float,
    q: float,
) -> np.ndarray:
    """CF of ``log(S_T / S_0)``. Matches ``pricer._heston_cf`` at ``a=1, b=0``."""
    u = np.asarray(u, dtype=np.complex128)
    phi_fwd = four_two_log_return_cf(u, T, kappa, theta, sigma, rho, v0, a, b)
    return np.exp(1j * u * (r - q) * T) * phi_fwd


def _sqrt_pos_real(z: np.ndarray) -> np.ndarray:
    """Principal square root, then flip any sample whose real part is negative."""
    s = np.sqrt(np.asarray(z, dtype=np.complex128))
    return np.where(s.real < 0.0, -s, s)


def _log_kummer_moment(
    series_a: np.ndarray,
    half_d: np.ndarray,
    z: np.ndarray,
    zeta: np.ndarray,
    p_pow: np.ndarray,
    log_c: np.ndarray,
    one_2s: np.ndarray,
) -> np.ndarray:
    """Log of the noncentral-chi-square moment, via a log-space Kummer series.

    Sums ``exp(-zeta/2) * 1F1(A; half_d; z)`` and attaches the closed-form
    prefactors. Term count tracks the batch-max ``|z|``.
    """
    z = np.asarray(z, dtype=np.complex128)
    zmax = float(np.max(np.abs(z))) if z.size else 0.0
    n_terms = int(math.ceil(zmax + 12.0 * math.sqrt(max(zmax, 0.0)) + 40.0))
    n_terms = min(max(n_terms, 8), 400)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_z = np.log(z)
        log_term = -0.5 * zeta
        scale = np.real(log_term).copy()
        series = np.exp(log_term - scale)
        for j in range(n_terms):
            log_term = (
                log_term
                + np.log(series_a + j)
                - np.log(half_d + j)
                + log_z
                - np.log(j + 1.0)
            )
            new_scale = np.maximum(scale, np.real(log_term))
            series = series * np.exp(scale - new_scale) + np.exp(log_term - new_scale)
            scale = new_scale

    return (
        p_pow * (log_c + np.log(2.0))
        - series_a * np.log(one_2s)
        + loggamma(series_a)
        - loggamma(half_d)
        + scale
        + np.log(series)
    )

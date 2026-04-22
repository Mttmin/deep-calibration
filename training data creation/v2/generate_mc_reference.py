"""Phase 3.3: MC ground-truth reference surfaces.

For N parameter sets sampled from the realistic prior, generate the
IV surface via two methods:
    1. v2 multi-method COS pricer (same as data/heston_pipeline_v2.h5)
    2. Adaptive-sy Andersen QE Monte Carlo (per-maturity uniform sub-grid)

The MC reference uses sy_per_cell = max(252, ceil(64/T)) so every cell
is computed with at least 64 QE sub-steps. This eliminates the O(dt)
QE bias documented in the 2026-04-19 root-cause report.

Per-maturity option (b) per the plan: one MC call per distinct T_i with
its own uniform steps_per_year; stitch (NK,) outputs into (NT, NK).

Output: data/heston_mc_reference.h5
    params(N, 6) float64
    pricer_iv(N, NK, NT) float64 (NaN at unpricable)
    pricer_prices(N, NK, NT) float64 (NaN at unpricable)
    pricer_flags(N, NK, NT) uint8 (0..3)
    mc_iv(N, NK, NT) float64 (NaN where MC SE too large or invertible fails)
    mc_prices(N, NK, NT) float64
    mc_se(N, NK, NT) float64
    mc_steps_per_year(NT,) int64
    log_moneyness(NK,), maturities(NT,) float64
    pricable_region(NK, NT) bool
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import heston_mc  # noqa: E402
import pricer  # noqa: E402
from pricer import price_cell  # noqa: E402
from sampling import sample_realistic_params, PARAM_NAMES  # noqa: E402
from generate_v2 import (  # noqa: E402
    LOG_MONEYNESS, MATURITIES, NK, NT, SPOT, Q_FIXED,
    FLAG_CODE, FLAG_ORDER, N_PARAMS,
    prices_to_iv_cpu,
)

REPO = HERE.parent.parent
PRICABLE_NPY = HERE / "pricable_region.npy"


def adaptive_sy(T: float, floor_substeps: int = 64, sy_floor: int = 252) -> int:
    """sy_per_cell = max(sy_floor, ceil(floor_substeps / T))."""
    return max(sy_floor, int(math.ceil(floor_substeps / T)))


def mc_surface_for_set(
    par: np.ndarray, n_paths: int, seed_base: int, set_idx: int,
    maturities: np.ndarray, K_vec: np.ndarray, spot: float = SPOT,
    q: float = Q_FIXED, antithetic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mc_prices, mc_se, sy_used) shape (NK, NT), (NK, NT), (NT,)."""
    kappa, theta, sigma_v, rho, v0, r = (float(par[i]) for i in range(6))
    NT_loc = maturities.size
    NK_loc = K_vec.size
    mc_prices = np.full((NK_loc, NT_loc), np.nan, dtype=np.float64)
    mc_se = np.full((NK_loc, NT_loc), np.nan, dtype=np.float64)
    sy_used = np.zeros(NT_loc, dtype=np.int64)
    # ATM = log_m=0 -> moneyness 1; flip option type per cell to keep OTM.
    # Simpler: per-maturity, batch all K together; pick option type per K.
    is_put = K_vec < spot  # OTM puts for K<spot, OTM calls for K>=spot
    K_call = K_vec[~is_put]
    K_put = K_vec[is_put]
    for ti, T in enumerate(maturities):
        sy = adaptive_sy(float(T))
        sy_used[ti] = sy
        seed = seed_base + 1009 * set_idx + 7 * ti
        if K_call.size:
            pr_c, se_c, _ = heston_mc.price_european_grid(
                kappa=kappa, theta=theta, sigma=sigma_v, rho=rho, v0=v0,
                r=r, q=q, spot=spot,
                K_arr=K_call, T_arr=np.array([T], dtype=np.float64),
                n_paths=n_paths, steps_per_year=sy, seed=seed,
                antithetic=antithetic, option_type="call",
            )
            mc_prices[~is_put, ti] = pr_c[0]
            mc_se[~is_put, ti] = se_c[0]
        if K_put.size:
            pr_p, se_p, _ = heston_mc.price_european_grid(
                kappa=kappa, theta=theta, sigma=sigma_v, rho=rho, v0=v0,
                r=r, q=q, spot=spot,
                K_arr=K_put, T_arr=np.array([T], dtype=np.float64),
                n_paths=n_paths, steps_per_year=sy, seed=seed + 1,
                antithetic=antithetic, option_type="put",
            )
            mc_prices[is_put, ti] = pr_p[0]
            mc_se[is_put, ti] = se_p[0]
    return mc_prices, mc_se, sy_used


def pricer_surface_for_set(
    par: np.ndarray, K_vec: np.ndarray, maturities: np.ndarray,
    spot: float = SPOT, q: float = Q_FIXED,
) -> tuple[np.ndarray, np.ndarray]:
    """Run v2 multi-method COS pricer, return (prices, flag_codes)."""
    kappa, theta, sigma_v, rho, v0, r = (float(par[i]) for i in range(6))
    NT_loc = maturities.size
    NK_loc = K_vec.size
    prices = np.full((NK_loc, NT_loc), np.nan, dtype=np.float64)
    flags = np.full((NK_loc, NT_loc), FLAG_CODE["unpricable"], dtype=np.uint8)
    for ki, K in enumerate(K_vec):
        opt = "put" if K < spot else "call"
        for ti, T in enumerate(maturities):
            price, flag, _conf = price_cell(
                kappa=kappa, theta=theta, sigma=sigma_v, rho=rho, v0=v0,
                r=r, q=q, spot=spot, K=float(K), T=float(T), option_type=opt,
            )
            flags[ki, ti] = FLAG_CODE[flag]
            if flag != "unpricable":
                prices[ki, ti] = price
    return prices, flags


def mc_se_threshold_mask(mc_prices, mc_se, abs_tol: float = 5e-4,
                          rel_tol: float = 0.05) -> np.ndarray:
    """Return bool mask True where MC is reliable enough for IV inversion."""
    abs_p = np.abs(mc_prices)
    rel = np.where(abs_p > 1e-12, mc_se / np.maximum(abs_p, 1e-12), np.inf)
    return np.isfinite(mc_prices) & np.isfinite(mc_se) & ((mc_se < abs_tol) | (rel < rel_tol))


def _worker_one_set(args_tuple):
    (i, par, n_paths, seed_base, K_vec, maturities) = args_tuple
    pp, pf = pricer_surface_for_set(par, K_vec, maturities)
    mp, ms, _sy = mc_surface_for_set(par, n_paths, seed_base, i,
                                      maturities, K_vec)
    r_arr = np.array([float(par[5])], dtype=np.float64)
    pi = prices_to_iv_cpu(pp[None, ...], SPOT, r_arr)[0]
    mc_ok = mc_se_threshold_mask(mp, ms)
    mp_for_iv = np.where(mc_ok, mp, np.nan)
    mi = prices_to_iv_cpu(mp_for_iv[None, ...], SPOT, r_arr)[0]
    return i, pp, pf, pi, mp, ms, mi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200, help="num parameter sets")
    ap.add_argument("--paths", type=int, default=2_000_000, help="MC paths (must be even for antithetic)")
    ap.add_argument("--seed", type=int, default=20260420)
    ap.add_argument("--out", type=str, default=str(REPO / "data" / "heston_mc_reference.h5"))
    ap.add_argument("--workers", type=int, default=1,
                    help="multiprocessing workers (each takes ~1 core); 0/1 = serial")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke: --N 1 --paths 100k, write to /tmp")
    args = ap.parse_args()

    if args.smoke:
        args.N = 1
        args.paths = 100_000
        args.out = "/tmp/heston_mc_reference_smoke.h5"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pricable = np.load(PRICABLE_NPY) if PRICABLE_NPY.exists() else np.ones((NK, NT), bool)

    K_vec = SPOT * np.exp(LOG_MONEYNESS)
    print(f"[mc-ref] N_sets={args.N}  paths={args.paths:,}  out={out_path}")
    print(f"[mc-ref] adaptive sy per maturity:")
    for ti, T in enumerate(MATURITIES):
        sy = adaptive_sy(float(T))
        print(f"  T={float(T):.4f} ({float(T)*365:6.1f}d)  sy={sy:5d}  steps={int(round(float(T)*sy)):4d}")

    params = sample_realistic_params(args.N, seed=args.seed)
    assert params.shape == (args.N, N_PARAMS)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("params", data=params)
        f.create_dataset("log_moneyness", data=LOG_MONEYNESS)
        f.create_dataset("maturities", data=MATURITIES)
        f.create_dataset("pricable_region", data=pricable)
        sy_arr = np.array([adaptive_sy(float(T)) for T in MATURITIES], dtype=np.int64)
        f.create_dataset("mc_steps_per_year", data=sy_arr)
        ds_pp = f.create_dataset("pricer_prices", shape=(args.N, NK, NT), dtype=np.float64,
                                 fillvalue=np.nan)
        ds_pi = f.create_dataset("pricer_iv", shape=(args.N, NK, NT), dtype=np.float64,
                                 fillvalue=np.nan)
        ds_pf = f.create_dataset("pricer_flags", shape=(args.N, NK, NT), dtype=np.uint8,
                                 fillvalue=FLAG_CODE["unpricable"])
        ds_mp = f.create_dataset("mc_prices", shape=(args.N, NK, NT), dtype=np.float64,
                                 fillvalue=np.nan)
        ds_ms = f.create_dataset("mc_se", shape=(args.N, NK, NT), dtype=np.float64,
                                 fillvalue=np.nan)
        ds_mi = f.create_dataset("mc_iv", shape=(args.N, NK, NT), dtype=np.float64,
                                 fillvalue=np.nan)
        f.attrs["spot"] = SPOT
        f.attrs["q"] = Q_FIXED
        f.attrs["seed"] = args.seed
        f.attrs["n_paths"] = args.paths
        f.attrs["antithetic"] = True
        f.attrs["sy_floor"] = 252
        f.attrs["floor_substeps"] = 64
        f.attrs["flag_order"] = ",".join(FLAG_ORDER)

        t0 = time.time()
        if args.workers and args.workers > 1:
            tasks = [(i, params[i], args.paths, args.seed, K_vec, MATURITIES)
                     for i in range(args.N)]
            done = 0
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for fut in as_completed(ex.submit(_worker_one_set, t) for t in tasks):
                    i, pp, pf, pi, mp, ms, mi = fut.result()
                    ds_pp[i] = pp; ds_pi[i] = pi; ds_pf[i] = pf
                    ds_mp[i] = mp; ds_ms[i] = ms; ds_mi[i] = mi
                    done += 1
                    elapsed = time.time() - t0
                    eta = elapsed / done * (args.N - done)
                    print(f"[mc-ref] {done:3d}/{args.N}  set_idx={i:3d}  "
                          f"elapsed={elapsed/60:5.1f}min  eta={eta/60:5.1f}min", flush=True)
        else:
            for i in range(args.N):
                _, pp, pf, pi, mp, ms, mi = _worker_one_set(
                    (i, params[i], args.paths, args.seed, K_vec, MATURITIES))
                ds_pp[i] = pp; ds_pi[i] = pi; ds_pf[i] = pf
                ds_mp[i] = mp; ds_ms[i] = ms; ds_mi[i] = mi
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (args.N - i - 1)
                print(f"[mc-ref] set {i+1:3d}/{args.N}  "
                      f"elapsed={elapsed/60:5.1f}min  eta={eta/60:5.1f}min", flush=True)

    print(f"[mc-ref] done. wrote {out_path}  total_wall={(time.time()-t0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

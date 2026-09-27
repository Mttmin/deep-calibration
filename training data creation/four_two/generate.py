"""Generate 4/2 IV surfaces for surrogate training.

One characteristic function per maturity, shared across the 49-strike grid.
Prices are inverted to Black-Scholes IV on the OTM side (put for k<0, call
otherwise), matching the Heston v2 layout.

    params        (N, 8) float64   kappa, theta, sigma_v, rho, v0, a, b, r
    iv_surface    (N, 49, 14) float32   NaN where unpricable
    quality_mask  (N, 49, 14) uint8     0 standard, 1 extended, 2 small, 3 unpricable
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pricer import (  # noqa: E402
    FLAG_CODE, HALF_FLOOR, L_FACTOR, L_FACTOR_EXT,
    LOG_MONEYNESS, MATURITIES, N_COS_EXT, N_COS_STD, NK, NT,
    Q_FIXED, RATIO_ACCEPT, RATIO_PRECISION, SPOT,
    cos_prices_torch,
)
from sampling import PARAM_HI, PARAM_LO, PARAM_NAMES, R_HI, R_LO, sample_realistic_params  # noqa: E402

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _bs_pv(sigma, F, K, T, disc, is_put, sqT):
    F_t = F[:, None]
    d1 = (torch.log(F_t / K[None, :]) + 0.5 * sigma * sigma * T) / (sigma * sqT + 1e-30)
    d2 = d1 - sigma * sqT
    nd1 = 0.5 * (1.0 + torch.erf(d1 * _INV_SQRT2))
    nd2 = 0.5 * (1.0 + torch.erf(d2 * _INV_SQRT2))
    call = disc[:, None] * (F_t * nd1 - K[None, :] * nd2)
    put = call + disc[:, None] * (K[None, :] - F_t)
    bs = torch.where(is_put[None, :], put, call)
    vega = disc[:, None] * F_t * sqT * torch.exp(-0.5 * d1 * d1) * _INV_SQRT2PI
    return bs, vega


def prices_to_iv(prices, K, T, r, is_put, iv_lo=0.005, iv_hi=3.0):
    """Bisection IV inversion. NaN on non-finite prices and on failed inversions.

    Intrinsic is the discounted forward intrinsic, ``e^{-rT} max(F-K, 0)``.
    The undiscounted form ``F - K e^{-rT}`` sits above low-vol prices and
    was rejecting valid ATM quotes.
    """
    F = torch.exp(r * T)
    disc = torch.exp(-r * T)
    sqT = math.sqrt(T)
    call_int = (disc[:, None] * (F[:, None] - K[None, :])).clamp(min=0.0)
    put_int = (disc[:, None] * (K[None, :] - F[:, None])).clamp(min=0.0)
    intrinsic = torch.where(is_put[None, :], put_int, call_int)
    zero = prices <= intrinsic + 1e-12
    bad = ~torch.isfinite(prices)

    lo = torch.full_like(prices, iv_lo)
    hi = torch.full_like(prices, iv_hi)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        bs_mid, _ = _bs_pv(mid, F, K, T, disc, is_put, sqT)
        lo = torch.where(bs_mid < prices, mid, lo)
        hi = torch.where(bs_mid < prices, hi, mid)
    sigma = 0.5 * (lo + hi)
    bs_final, _ = _bs_pv(sigma, F, K, T, disc, is_put, sqT)
    rel = (bs_final - prices).abs() / (prices.abs() + 1e-12)
    fail = zero | bad | (rel > 1e-4) | (sigma <= iv_lo + 1e-8) | (sigma >= iv_hi - 1e-6)
    return torch.where(fail, torch.full_like(sigma, float("nan")), sigma)


def price_chunk(params: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return IV ``(B, NK, NT)`` and flags ``(B, NK, NT)``.

    A row whose standard COS leaves any cell non-finite is repriced at the
    extended truncation. The characteristic function is per maturity, so
    repricing the strike grid costs the same as repricing one cell.
    """
    B = params.shape[0]
    kappa, theta, sigma, rho, v0, a, b, r = (params[:, i] for i in range(8))
    K = torch.tensor(SPOT * np.exp(LOG_MONEYNESS), device=device, dtype=torch.float64)
    is_put = torch.tensor(LOG_MONEYNESS < 0.0, device=device)
    iv = torch.full((B, NK, NT), float("nan"), device=device, dtype=torch.float64)
    flags = torch.full((B, NK, NT), int(FLAG_CODE["unpricable"]), device=device, dtype=torch.uint8)
    code_std = int(FLAG_CODE["cos_standard"])
    code_ext = int(FLAG_CODE["cos_extended"])
    code_small = int(FLAG_CODE["cos_small_price"])

    for ti, T in enumerate(MATURITIES.tolist()):
        T = float(T)
        px = cos_prices_torch(
            kappa, theta, sigma, rho, v0, a, b, r, K, is_put, T, SPOT, N_COS_STD, L_FACTOR,
        )
        cell_iv = prices_to_iv(px, K, T, r, is_put)
        accept = torch.isfinite(cell_iv) & ((px / SPOT) > RATIO_ACCEPT)
        small = torch.isfinite(cell_iv) & ~accept & ((px / SPOT) > RATIO_PRECISION)
        cell_flag = torch.full((B, NK), int(FLAG_CODE["unpricable"]), device=device, dtype=torch.uint8)
        cell_flag = torch.where(accept, torch.full_like(cell_flag, code_std), cell_flag)
        cell_flag = torch.where(small, torch.full_like(cell_flag, code_small), cell_flag)

        need = ~torch.isfinite(cell_iv)
        if need.any():
            idx = torch.where(need.any(dim=1))[0]
            px_e = cos_prices_torch(
                kappa[idx], theta[idx], sigma[idx], rho[idx], v0[idx], a[idx], b[idx], r[idx],
                K, is_put, T, SPOT, N_COS_EXT, L_FACTOR_EXT,
            )
            iv_e = prices_to_iv(px_e, K, T, r[idx], is_put)
            take = need[idx] & torch.isfinite(iv_e)
            cell_iv[idx] = torch.where(take, iv_e, cell_iv[idx])
            ext_accept = take & ((px_e / SPOT) > RATIO_ACCEPT)
            ext_small = take & ~ext_accept
            cell_flag[idx] = torch.where(ext_accept, torch.full_like(cell_flag[idx], code_ext), cell_flag[idx])
            cell_flag[idx] = torch.where(ext_small, torch.full_like(cell_flag[idx], code_small), cell_flag[idx])

        iv[:, :, ti] = cell_iv
        flags[:, :, ti] = cell_flag
    return iv, flags


def _coherence(params: np.ndarray, iv: np.ndarray) -> None:
    """Abort if the first chunk does not look like an equity smile."""
    atm = 32  # log-moneyness grid hits 0 at index 32
    finite = np.isfinite(iv)
    frac = float(finite.mean())
    if frac < 0.80:
        raise SystemExit(f"coherence: only {frac:.1%} of cells priced")
    atm_iv = iv[:, atm, :]
    med = float(np.nanmedian(atm_iv))
    if not (0.06 <= med <= 0.70):
        raise SystemExit(f"coherence: median ATM IV {med:.3f} outside [0.06, 0.70]")
    steep = params[:, 3] < -0.55
    if int(steep.sum()) >= 8:
        skew = np.nanmedian(iv[steep, 24, 7] - iv[steep, 36, 7])
        if not np.isfinite(skew) or skew <= 0.0:
            raise SystemExit(f"coherence: expected negative skew, got {skew}")
    print(f"[coherence] finite={frac:.1%}  median ATM IV={med:.3f}", flush=True)


def generate(n: int, out: Path, seed: int, chunk: int, device: torch.device) -> None:
    print(f"[gen] sampling {n:,} params", flush=True)
    params = sample_realistic_params(n, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    K_grid = SPOT * np.exp(LOG_MONEYNESS)
    with h5py.File(out, "w") as f:
        f.create_dataset("params", data=params, compression="lzf")
        ds_iv = f.create_dataset(
            "iv_surface", shape=(n, NK, NT), dtype="float32",
            chunks=(min(256, n), NK, NT), compression="lzf",
        )
        ds_qm = f.create_dataset(
            "quality_mask", shape=(n, NK, NT), dtype="uint8",
            chunks=(min(256, n), NK, NT), compression="lzf",
        )
        f.attrs["NK"] = NK
        f.attrs["NT"] = NT
        f.attrs["q_fixed"] = Q_FIXED
        f.attrs["spot"] = SPOT
        f.attrs["param_names"] = PARAM_NAMES
        f.create_dataset("log_moneyness", data=LOG_MONEYNESS)
        f.create_dataset("maturities", data=MATURITIES)
        f.create_dataset("strikes", data=K_grid)
        f.create_dataset("param_lo", data=PARAM_LO)
        f.create_dataset("param_hi", data=PARAM_HI)
        f.attrs["r_lo"] = R_LO
        f.attrs["r_hi"] = R_HI
        f.attrs["half_floor"] = HALF_FLOOR
        f.attrs["n_cos_std"] = N_COS_STD

        t0 = time.time()
        n_chunks = math.ceil(n / chunk)
        finite_count = np.zeros((NK, NT), dtype=np.int64)
        for ci in range(n_chunks):
            i0, i1 = ci * chunk, min(ci * chunk + chunk, n)
            batch = torch.tensor(params[i0:i1], device=device, dtype=torch.float64)
            iv, flags = price_chunk(batch, device)
            iv_np = iv.detach().cpu().numpy()
            fl_np = flags.detach().cpu().numpy()
            if ci == 0:
                _coherence(params[i0:i1], iv_np)
            ds_iv[i0:i1] = iv_np.astype(np.float32)
            ds_qm[i0:i1] = fl_np
            finite_count += np.isfinite(iv_np).sum(axis=0)
            elapsed = time.time() - t0
            done = i1
            rate = done / max(elapsed, 1e-9)
            print(
                f"  {done:,}/{n:,}  {rate:,.0f} samples/s  "
                f"elapsed={elapsed/60:.1f} min  eta={(n-done)/rate/60:.1f} min",
                flush=True,
            )
        region = finite_count >= int(0.95 * n)
        f.create_dataset("pricable_region", data=region)
        f.attrs["pricable_fraction"] = float(region.mean())
        f.attrs["finite_fraction"] = float(finite_count.sum() / (n * NK * NT))
    print(f"[gen] wrote {out}  pricable={float(region.mean()):.1%}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    generate(args.n, args.out, args.seed, args.chunk, device)


if __name__ == "__main__":
    main()

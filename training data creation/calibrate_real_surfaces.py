"""
Phase 2: calibrate real IV surfaces to Bates parameter candidates (top-K per surface).

This is a pragmatic calibration engine for dataset guidance:
- samples many candidate Bates params + (r, q)
- prices full surfaces via the same COS/Bates pricer used by heston_datagen.py
- scores with weighted RMSE on valid real cells
- stores top-K candidates per real surface

Example
-------
python calibrate_real_surfaces.py \
  --real-h5 ../data/real_surfaces_q1_2025.h5 \
  --out ../data/guided_param_bank_q1_2025.h5 \
  --max-surfaces 200 --candidates 2048 --top-k 8
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import qmc
from tqdm import tqdm


def load_heston_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("heston_datagen_mod", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weighted_cell_weights(log_m: np.ndarray, mats: np.ndarray) -> np.ndarray:
    grid_m, grid_t = np.meshgrid(log_m, mats, indexing="ij")
    short_w = 1.0 / np.sqrt(np.maximum(grid_t, 1.0 / 365.25))
    put_w = np.where(grid_m < 0.0, 1.8, 1.0)
    ultra_short_boost = np.where(grid_t <= (2.0 / 52.0), 1.5, 1.0)
    return (short_w * put_w * ultra_short_boost).astype(np.float64)


def sample_candidates(
    n: int,
    param_bounds: np.ndarray,
    seed: int,
) -> np.ndarray:
    n_param = param_bounds.shape[0]
    sampler = qmc.Sobol(d=n_param, scramble=True, seed=seed)
    n_raw = int(2 ** math.ceil(math.log2(max(2, n))))
    raw = sampler.random(n_raw)[:n]

    lo = param_bounds[:, 0]
    hi = param_bounds[:, 1]
    scaled = qmc.scale(raw, lo, hi)
    return scaled


def simulate_candidates(
    mod,
    params_np: np.ndarray,
    r: float,
    q: float,
    chunk_size: int,
    dev: torch.device,
) -> np.ndarray:
    """Return simulated IV surfaces with shape (C, NK, NT)."""
    c = len(params_np)
    nk = mod.NK
    nt = mod.NT

    params_t_all = torch.tensor(params_np, device=dev, dtype=torch.float64)
    K_grid = torch.tensor(np.exp(mod.LOG_MONEYNESS), device=dev, dtype=torch.float64)
    is_put = torch.tensor(mod.IS_PUT_SIDE, device=dev)
    k_idx = torch.arange(mod.N_COS, device=dev)
    Vk = mod.cos_call_Vk(dev)

    out = np.empty((c, nk, nt), dtype=np.float64)

    for i0 in range(0, c, chunk_size):
        i1 = min(i0 + chunk_size, c)
        b = i1 - i0
        par = params_t_all[i0:i1]
        iv = torch.full((b, nk, nt), float("nan"), device=dev, dtype=torch.float64)

        for ti, T in enumerate(mod.MATURITIES.tolist()):
            call_p = mod.cos_call_prices(par, Vk, T, 1.0, K_grid, r, q, k_idx)
            otm_p = mod.call_to_otm(call_p, K_grid, T, 1.0, r, q, is_put)
            iv[:, :, ti] = mod.prices_to_iv(otm_p, K_grid, T, 1.0, r, q, is_put)

        # Use same market-consistent fill as generator before scoring.
        iv = mod.fill_nan_market_consistent(iv)
        out[i0:i1] = iv.detach().cpu().numpy()

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-h5", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-surfaces", type=int, default=200)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=str, default="high41x14", choices=["base25x10", "high41x14"])
    ap.add_argument("--use-raw-mask", action="store_true", help="Score only on raw invertible cells")
    ap.add_argument("--use-confidence", action="store_true", help="Multiply weights by cell_confidence if present")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    heston_path = here / "heston_datagen.py"
    mod = load_heston_module(heston_path)
    mod.set_grid_preset(args.grid)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}")

    with h5py.File(args.real_h5, "r") as f:
        real_iv = np.array(f["iv_surface"], dtype=np.float64)
        cell_mask = np.array(f["cell_mask"], dtype=bool)
        raw_mask = np.array(f["raw_cell_mask"], dtype=bool) if "raw_cell_mask" in f else None
        conf = np.array(f["cell_confidence"], dtype=np.float64) if "cell_confidence" in f else None
        tickers = np.array(f["ticker"], dtype=object)
        snap_dates = np.array(f["snapshot_date"], dtype=object)

        real_log_m = np.array(f.attrs["log_moneyness"], dtype=np.float64)
        real_mats = np.array(f.attrs["maturities"], dtype=np.float64)

    if real_iv.shape[1] != mod.NK or real_iv.shape[2] != mod.NT:
        raise ValueError(
            f"Grid mismatch: real={real_iv.shape[1:]} vs generator={(mod.NK, mod.NT)}. "
            "Use matching --grid."
        )
    if not np.allclose(real_log_m, mod.LOG_MONEYNESS):
        raise ValueError("log_moneyness grid mismatch between real_h5 and selected --grid")
    if not np.allclose(real_mats, mod.MATURITIES):
        raise ValueError("maturities grid mismatch between real_h5 and selected --grid")

    n_total = len(real_iv)
    start = max(0, args.start_index)
    end = min(n_total, start + args.max_surfaces)
    picked = np.arange(start, end)

    base_w = weighted_cell_weights(mod.LOG_MONEYNESS, mod.MATURITIES)

    top_params_all: list[np.ndarray] = []
    top_market_all: list[np.ndarray] = []
    top_scores_all: list[np.ndarray] = []
    src_idx_all: list[int] = []
    src_ticker: list[str] = []
    src_date: list[str] = []

    p_bounds = np.array(mod.PARAM_BOUNDS, dtype=np.float64)
    r_bounds = tuple(float(x) for x in mod.R_BOUNDS)
    q_bounds = tuple(float(x) for x in mod.Q_BOUNDS)

    for j in tqdm(picked, desc="Calibrating"):
        target = real_iv[j]
        if args.use_raw_mask and raw_mask is not None:
            valid = raw_mask[j] & np.isfinite(target)
        else:
            valid = cell_mask[j] & np.isfinite(target)

        if valid.sum() < 20:
            continue

        w = base_w.copy()
        if args.use_confidence and conf is not None:
            w *= np.clip(conf[j], 0.0, 1.0)
        w = np.where(valid, w, 0.0)
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue

        cand_params = sample_candidates(
            n=args.candidates,
            param_bounds=p_bounds,
            seed=args.seed + 17 * int(j),
        )

        # Current pricer uses scalar r,q per batch; calibrate one market pair per surface.
        rng = np.random.default_rng(args.seed + 101 * int(j))
        r_s = float(rng.uniform(*r_bounds))
        q_s = float(rng.uniform(*q_bounds))

        pred = simulate_candidates(
            mod=mod,
            params_np=cand_params,
            r=r_s,
            q=q_s,
            chunk_size=args.batch,
            dev=dev,
        )

        diff = pred - target[None, :, :]
        sq = diff * diff
        mse = (sq * w[None, :, :]).sum(axis=(1, 2)) / w_sum
        rmse = np.sqrt(mse)

        k = min(args.top_k, len(rmse))
        best = np.argpartition(rmse, k - 1)[:k]
        best = best[np.argsort(rmse[best])]

        top_params_all.append(cand_params[best])
        top_market_all.append(np.full((len(best), 2), [r_s, q_s], dtype=np.float64))
        top_scores_all.append(rmse[best])
        src_idx_all.append(int(j))

        t = tickers[j]
        d = snap_dates[j]
        src_ticker.append(t.decode("utf-8") if isinstance(t, (bytes, np.bytes_)) else str(t))
        src_date.append(d.decode("utf-8") if isinstance(d, (bytes, np.bytes_)) else str(d))

    if not top_params_all:
        raise RuntimeError("No calibrated surfaces produced. Try lowering --min valid coverage constraints.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ticker_dtype = h5py.string_dtype(encoding="utf-8")
    date_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out, "w") as f:
        f.attrs["source_real_h5"] = args.real_h5
        f.attrs["grid"] = args.grid
        f.attrs["N_surfaces"] = len(top_params_all)
        f.attrs["top_k"] = int(args.top_k)
        f.attrs["candidates_per_surface"] = int(args.candidates)

        f.create_dataset("source_index", data=np.array(src_idx_all, dtype=np.int32))
        f.create_dataset("source_ticker", data=np.array(src_ticker, dtype=object), dtype=ticker_dtype)
        f.create_dataset("source_snapshot_date", data=np.array(src_date, dtype=object), dtype=date_dtype)
        f.create_dataset("top_params", data=np.stack(top_params_all).astype(np.float64), compression="lzf")
        f.create_dataset("top_market_params", data=np.stack(top_market_all).astype(np.float64), compression="lzf")
        f.create_dataset("top_weighted_rmse", data=np.stack(top_scores_all).astype(np.float64), compression="lzf")

        # Flattened convenience view for generator guided sampling.
        flat = np.stack(top_params_all).reshape(-1, np.stack(top_params_all).shape[-1])
        f.create_dataset("guided_params", data=flat.astype(np.float64), compression="lzf")

    print(f"[done] calibrated {len(top_params_all)} surfaces -> {out}")


if __name__ == "__main__":
    main()

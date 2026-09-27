"""Coherence checks for a trained 4/2 surrogate.

Reports validation IVRMSE, skew sign, the Heston embedding (b=0), and a
short Levenberg-Marquardt recovery through the torch model. A coherent
surrogate has negative-rho put wings above the call wing, and a recovery
whose repriced surface stays within a few vol points of the target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training data creation" / "four_two"))

from model.loss import ivrmse_bps  # noqa: E402
from model.network import BatesSurrogate  # noqa: E402
from sampling import PARAM_HI, PARAM_LO, R_HI, R_LO  # noqa: E402

N_AUX = 7


def load(path: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        idx = np.sort(rng.choice(f["params"].shape[0], size=n, replace=False))
        params = f["params"][idx].astype(np.float64)
        iv = f["iv_surface"][idx].astype(np.float32)
        qm = f["quality_mask"][idx]
        region = f["pricable_region"][:].astype(bool)
    theta = np.zeros((n, 9), dtype=np.float32)
    span = (PARAM_HI - PARAM_LO).astype(np.float32)
    theta[:, :7] = ((params[:, :7] - PARAM_LO) / (PARAM_HI - PARAM_LO)).astype(np.float32)
    theta[:, 7] = ((params[:, 7] - R_LO) / (R_HI - R_LO)).astype(np.float32)
    flat = iv.reshape(n, -1)
    finite = np.isfinite(flat)
    mask = finite & (qm.reshape(n, -1) < 3) & region.reshape(1, -1)
    flat = np.where(finite, flat, 0.0)
    return params, theta, flat, mask, span


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/four_two/best.pt")
    ap.add_argument("--h5", default="data/four_two_val.h5")
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BatesSurrogate.from_checkpoint(args.checkpoint).to(device).eval()
    params, theta, iv, mask, span = load(args.h5, args.n, 1)
    th = torch.from_numpy(theta).to(device)
    iv_t = torch.from_numpy(iv).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    with torch.no_grad():
        pred = model(th)
    bps = ivrmse_bps(pred, iv_t, mask_t)
    print(f"[ivrmse] {bps:.2f} bps on {args.n} val surfaces")

    pred_np = pred.detach().cpu().numpy().reshape(args.n, 49, 14)
    iv_np = iv.reshape(args.n, 49, 14)
    steep = params[:, 3] < -0.55
    skew = np.nanmedian(pred_np[steep, 24, 7] - pred_np[steep, 36, 7])
    target_skew = np.nanmedian(iv_np[steep, 24, 7] - iv_np[steep, 36, 7])
    print(f"[skew] pred={skew:.4f}  target={target_skew:.4f}  (put wing minus call wing, rho<-0.55)")

    heston = (params[:, 6] < 1e-8) & (np.abs(params[:, 5] - 1.0) < 0.15)
    if heston.sum() >= 8:
        err = np.sqrt(np.nanmean((pred_np[heston] - iv_np[heston]) ** 2)) * 1e4
        print(f"[heston-embed] n={int(heston.sum())}  IVRMSE={err:.2f} bps")

    # One recovery: freeze r, fit the 7 normalised params to a masked surface.
    i = int(np.argmax(mask.mean(axis=1)))
    target = iv_t[i : i + 1]
    w = mask_t[i : i + 1].float()
    r_slot = th[i : i + 1, 7:].detach()
    x = th[i : i + 1, :N_AUX].detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=0.5, max_iter=40, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        full = torch.cat([x.clamp(0, 1), r_slot], dim=1)
        pred_i = model(full)
        resid = (pred_i - target) * w
        loss = (resid ** 2).sum() / w.sum().clamp(min=1)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        full = torch.cat([x.clamp(0, 1), r_slot], dim=1)
        rec = model(full)
        rec_bps = ivrmse_bps(rec, target, mask_t[i : i + 1])
    phys = x.detach().cpu().numpy()[0] * span + PARAM_LO
    true = params[i, :7]
    print(f"[recover] surface IVRMSE={rec_bps:.2f} bps")
    names = ["kappa", "theta", "sigma", "rho", "v0", "a", "b"]
    for name, hat, ref in zip(names, phys, true):
        print(f"  {name:6s}  hat={hat:.4f}  true={ref:.4f}")
    if skew <= 0 or bps > 40 or rec_bps > 30:
        raise SystemExit("coherence check failed")
    print("[coherence] ok")


if __name__ == "__main__":
    main()

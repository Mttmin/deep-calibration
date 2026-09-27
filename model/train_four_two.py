"""Train the 4/2 IV-surface surrogate.

Input layout, all in [0, 1]:
    kappa, theta, sigma_v, rho, v0, a, b, r_norm, q_norm
``q_norm`` is pinned at 0. Physical ``q`` is folded into carry the same way
the Heston v2 surrogate did. The auxiliary head predicts the seven
calibrated parameters so the latent state stays identifiable.

Usage
-----
    python -m model.train_four_two \\
        --train data/four_two_train.h5 --val data/four_two_val.h5

    python -m model.train_four_two --resume runs/four_two/best.pt --finetune
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.amp as amp
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training data creation" / "four_two"))

from model.loss import compute_vega_weights, ivrmse_bps, total_loss  # noqa: E402
from model.network import BatesSurrogate, GridConstants  # noqa: E402
from sampling import PARAM_HI, PARAM_LO, R_HI, R_LO  # noqa: E402

N_AUX = 7
N_INPUT = 9  # 7 calibrated + r + q
# kappa is flat; a and b trade off against the variance scale. Upweight them.
PARAM_LOSS_W = torch.tensor([2.0, 3.0, 2.5, 5.0, 3.0, 2.0, 3.0])


def load_split(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return theta (N, 9), iv (N, 686), mask (N, 686), pricable (686,)."""
    with h5py.File(path, "r") as f:
        params = f["params"][:].astype(np.float32)
        iv = f["iv_surface"][:].astype(np.float32)
        qm = f["quality_mask"][:]
        region = f["pricable_region"][:].astype(bool)
    theta = np.zeros((params.shape[0], N_INPUT), dtype=np.float32)
    theta[:, :7] = (params[:, :7] - PARAM_LO.astype(np.float32)) / (
        PARAM_HI.astype(np.float32) - PARAM_LO.astype(np.float32)
    )
    theta[:, 7] = (params[:, 7] - np.float32(R_LO)) / np.float32(R_HI - R_LO)
    iv_flat = iv.reshape(iv.shape[0], -1)
    finite = np.isfinite(iv_flat)
    iv_flat = np.where(finite, iv_flat, 0.0).astype(np.float32)
    mask = finite & (qm.reshape(qm.shape[0], -1) < 3) & region.reshape(1, -1)
    # Drop surfaces that lost the ATM column. Those rows teach the wrong shape.
    atm = 32 * iv.shape[2] + np.arange(iv.shape[2])
    keep = mask[:, atm].mean(axis=1) > 0.5
    return (
        torch.from_numpy(theta[keep]),
        torch.from_numpy(iv_flat[keep]),
        torch.from_numpy(mask[keep]),
        torch.from_numpy(region.reshape(-1)),
    )


def _warmup(epoch: int, start: int, end: int) -> float:
    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    return (epoch - start) / (end - start)


def _run_epoch(
    model: BatesSurrogate,
    loader: DataLoader,
    grid: GridConstants,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lambdas: tuple[float, float, float, float],
) -> float:
    train = optimizer is not None
    model.train(train)
    lam_cal, lam_bfly, lam_ts, lam_param = lambdas
    total_sq = 0.0
    total_n = 0
    w_param = PARAM_LOSS_W.to(device)
    for theta, iv, mask in loader:
        theta = theta.to(device, non_blocking=True)
        iv = iv.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        r = theta[:, 7] * (R_HI - R_LO) + R_LO
        q = torch.zeros_like(r)
        with torch.set_grad_enabled(train):
            with amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                iv_pred, param_pred = model.forward_dual(theta)
            weights = compute_vega_weights(iv, grid, r, q)
            breakdown = total_loss(
                iv_pred, iv, mask, weights, grid,
                lambda_cal=lam_cal, lambda_bfly=lam_bfly, lambda_ts=lam_ts,
            )
            param_mse = (((param_pred.float() - theta[:, :N_AUX]) ** 2) * w_param).mean()
            loss = breakdown.total + lam_param * param_mse
            # A butterfly spike can point the clipped step away from the IV fit
            # and wipe a good checkpoint in one epoch. Drop that batch.
            if not torch.isfinite(loss) or float(loss.detach()) > 5.0:
                if train:
                    optimizer.zero_grad(set_to_none=True)
                continue
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            # ivrmse_bps wants a mask; empty batches are skipped by the loader.
            if mask.any():
                bps = ivrmse_bps(iv_pred.float(), iv, mask)
                n = int(mask.sum().item())
                total_sq += (bps / 10_000.0) ** 2 * n
                total_n += n
    if total_n == 0:
        return float("nan")
    return (total_sq / total_n) ** 0.5 * 10_000.0


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    theta, iv, mask, region = load_split(args.train)
    v_theta, v_iv, v_mask, _ = load_split(args.val)
    print(f"[data] train={theta.shape[0]:,}  val={v_theta.shape[0]:,}  "
          f"pricable={float(region.float().mean()):.1%}", flush=True)

    grid = GridConstants.default()
    model = BatesSurrogate(
        n_params=N_INPUT, n_aux=N_AUX, n_outputs=iv.shape[1],
        width=args.width, n_blocks=args.n_blocks, nk=49, nt=14,
        rank=args.rank, dropout=args.dropout, pricable_region=region,
    )
    start_epoch = 0
    best = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_compatible_state_dict(ckpt["model_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("val_ivrmse_bps", float("inf")))
        print(f"[resume] {args.resume}  epoch={start_epoch}  val={best:.2f} bps", flush=True)
    model = model.to(device)

    lr = args.lr if not args.finetune else args.finetune_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6,
    )
    train_loader = DataLoader(
        TensorDataset(theta, iv, mask), batch_size=args.batch_size,
        shuffle=True, drop_last=True, pin_memory=device.type == "cuda", num_workers=2,
    )
    val_loader = DataLoader(
        TensorDataset(v_theta, v_iv, v_mask), batch_size=args.batch_size,
        shuffle=False, pin_memory=device.type == "cuda", num_workers=2,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stall = 0
    epochs = args.finetune_epochs if args.finetune else args.epochs
    for epoch in range(epochs):
        # Butterfly second differences are O(1/dk^2). They are safe only once
        # the surface is already close; before that they dominate the clipped step.
        if args.finetune or best <= args.pinn_after_bps:
            warm = 1.0 if args.finetune else min(1.0, (args.pinn_after_bps / max(best, 1.0)))
            lambdas = (0.02 * warm, 0.001 * warm, 0.05 * warm, 0.08 if args.finetune else 0.04)
        else:
            lambdas = (0.0, 0.0, 0.0, 0.02)
        t0 = time.time()
        tr = _run_epoch(model, train_loader, grid, device, optimizer, lambdas)
        va = _run_epoch(model, val_loader, grid, device, None, lambdas)
        scheduler.step(va)
        improved = va < best - 0.15
        if improved:
            best = va
            stall = 0
            _save(model, out / "best.pt", epoch + start_epoch, va, args, region)
        else:
            stall += 1
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch+1:3d}/{epochs}  train={tr:.2f} bps  val={va:.2f} bps  "
            f"best={best:.2f}  lr={lr_now:.2e}  {time.time()-t0:.1f}s",
            flush=True,
        )
        if stall >= args.patience:
            print(f"[stop] no improvement for {args.patience} epochs", flush=True)
            break
    _save(model, out / "last.pt", epoch + start_epoch, va, args, region)
    print(f"[done] best val IVRMSE {best:.2f} bps  -> {out / 'best.pt'}", flush=True)


def _save(model, path, epoch, val_bps, args, region) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_ivrmse_bps": val_bps,
            "config": {
                "n_params": N_INPUT,
                "n_aux": N_AUX,
                "n_outputs": 686,
                "width": args.width,
                "n_blocks": args.n_blocks,
                "nk": 49,
                "nt": 14,
                "rank": args.rank,
                "dropout": args.dropout,
                "param_lo": PARAM_LO.tolist(),
                "param_hi": PARAM_HI.tolist(),
                "r_lo": R_LO,
                "r_hi": R_HI,
            },
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn-after-bps", type=float, default=20.0)
    ap.add_argument("--train", type=Path, default=Path("data/four_two_train.h5"))
    ap.add_argument("--val", type=Path, default=Path("data/four_two_val.h5"))
    ap.add_argument("--out", type=Path, default=Path("runs/four_two"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--finetune-lr", type=float, default=3e-5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=18)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--finetune", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""
model/train.py
==============
Training loop for the BatesSurrogate feedforward IV-surface surrogate.

Usage
-----
  # Standard run (10M dataset, 200 epochs)
  python -m model.train --h5 data/heston_guided_pilot.h5

  # Ablation: no PINN penalties
  python -m model.train --h5 data/heston_guided_pilot.h5 --no-pinn

  # Resume from checkpoint
  python -m model.train --h5 data/heston_guided_pilot.h5 --resume model/runs/best.pt

    # Auto preload tiering (VRAM -> RAM -> disk)
    python -m model.train --h5 data/heston_guided_pilot.h5 --preload auto

  # Custom hyperparameters
  python -m model.train --h5 data/heston_guided_pilot.h5 \\
      --width 512 --n-blocks 8 --batch-size 2048 --epochs 300

Training details
----------------
  Precision:    BF16 autocast (forward), FP32 (loss)
  Optimizer:    Adam  lr=1e-3, weight_decay=1e-5
  Scheduler:    ReduceLROnPlateau  factor=0.5, patience=10
  Grad clip:    max_norm=1.0
  PINN warmup:  λ_cal and λ_bfly ramped from 0 over epochs 10-30
  Target:       val IVRMSE < 10 bps

References
----------
  Horvath, Muguruza & Tomas (2021) arXiv:1901.09647
  deep_calibration_research.tex §6 (training loop template, lines 1120-1146)
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import h5py
import torch
import torch.amp as amp
from torch.utils.data import DataLoader

# Allow running as  python -m model.train  from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.network import BatesSurrogate, GridConstants
from model.loss import (
    LossBreakdown,
    compute_vega_weights,
    ivrmse_bps,
    total_loss,
)

# BatesDataset lives in the data-generation script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training data creation"))
from heston_datagen import BatesDataset   # type: ignore[import]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

VRAM_RESERVE_FRACTION = 0.10
RAM_RESERVE_FRACTION = 0.30


def _format_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GiB"


def _available_vram_bytes(device: torch.device) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device=device)
        return int(free_bytes)
    except Exception:
        return 0


def _available_ram_bytes() -> int:
    # Linux: MemAvailable reflects reclaimable page cache plus free memory.
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        return 0
    return 0


def _h5_shape(h5_path: str) -> tuple[int, int, int]:
    with h5py.File(h5_path, "r") as f:
        iv = f["iv_surface"]
        n_total = int(iv.shape[0])
        nk = int(f.attrs.get("NK", iv.shape[1]))
        nt = int(f.attrs.get("NT", iv.shape[2]))
    return n_total, nk, nt


def _bytes_per_sample(nk: int, nt: int, include_confidence: bool) -> int:
    # params_norm(10 float32), iv(float32), mask(bool), optional confidence(float32)
    n_flat = nk * nt
    total = (10 * 4) + (n_flat * 4) + n_flat
    if include_confidence:
        total += n_flat * 4
    return total


def _estimate_split_bytes(
    h5_path: str,
    val_frac: float,
    val_h5: str | None,
    train_confidence: bool,
) -> tuple[int, int]:
    n_total, nk, nt = _h5_shape(h5_path)
    train_bps = _bytes_per_sample(nk, nt, include_confidence=train_confidence)

    if val_h5 is None:
        n_val = max(1, int(math.ceil(n_total * val_frac)))
        n_train = max(0, n_total - n_val)
        val_bps = _bytes_per_sample(nk, nt, include_confidence=False)
        return n_train * train_bps, n_val * val_bps

    n_val_total, nk_val, nt_val = _h5_shape(val_h5)
    val_bps = _bytes_per_sample(nk_val, nt_val, include_confidence=False)
    return n_total * train_bps, n_val_total * val_bps


def resolve_auto_preload(
    h5_path: str,
    val_frac: float,
    val_h5: str | None,
    train_confidence: bool,
    device: torch.device,
) -> tuple[str, str, str]:
    train_bytes, val_bytes = _estimate_split_bytes(h5_path, val_frac, val_h5, train_confidence)
    total_bytes = train_bytes + val_bytes

    vram_free = _available_vram_bytes(device)
    ram_free = _available_ram_bytes()
    vram_budget = int(vram_free * (1.0 - VRAM_RESERVE_FRACTION))
    ram_budget = int(ram_free * (1.0 - RAM_RESERVE_FRACTION))

    print(
        "[preload-auto] "
        f"need train={_format_gb(train_bytes)} val={_format_gb(val_bytes)} total={_format_gb(total_bytes)} | "
        f"free vram={_format_gb(vram_free)} (budget={_format_gb(vram_budget)}) | "
        f"free ram={_format_gb(ram_free)} (budget={_format_gb(ram_budget)})"
    )
    # Prefer RAM preload when the entire dataset fits in RAM, even if VRAM
    # could also hold it. Preloading to CPU RAM allows multiple DataLoader
    # workers (parallel batch preparation + pinning) while still using GPU
    # for compute, which often yields better host-device overlap.
    if total_bytes <= ram_budget:
        return "all", "cpu", "cpu"
    if device.type == "cuda" and total_bytes <= vram_budget:
        return "all", "cuda", "cuda"
    if train_bytes <= ram_budget:
        return "train", "cpu", "cpu"
    if device.type == "cuda" and train_bytes <= vram_budget:
        return "train", "cuda", "cpu"
    return "none", "cpu", "cpu"

def build_dataloaders(
    h5_path: str,
    batch_size: int = 8196,
    num_workers: int = 8,
    val_frac: float = 0.10,
    val_h5: str | None = None,
    seed: int = 42,
    preload: str = "none",
    train_preload_device: str = "cpu",
    val_preload_device: str = "cpu",
    return_confidence: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Builds train / val DataLoaders from a BatesDataset HDF5 file.

    Split strategy: last ``val_frac`` fraction of dataset indices as val.
    Sobol sequences are space-filling, so the last 10% covers parameter
    space as well as any random subset.

    If ``val_h5`` is provided, it is used as a separate validation dataset
    (ignores ``val_frac``).
    """
    if preload not in {"none", "train", "all"}:
        raise ValueError(f"Unknown preload mode: {preload}")

    if train_preload_device not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown train preload device: {train_preload_device}")
    if val_preload_device not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown val preload device: {val_preload_device}")

    train_preload = preload in {"train", "all"}
    val_preload = preload == "all"

    meta_ds = BatesDataset(h5_path, min_valid_cells=0, preload=False)
    N       = len(meta_ds)

    def _build_with_fallback(base_kwargs: dict, split_name: str, preload_now: bool, preload_device_now: str):
        kwargs = dict(base_kwargs)
        while True:
            kwargs["preload"] = preload_now
            kwargs["preload_device"] = preload_device_now
            try:
                ds = BatesDataset(**kwargs)
                return ds, preload_now, preload_device_now
            except (RuntimeError, MemoryError) as exc:
                if preload_now and preload_device_now == "cuda":
                    print(f"[data] {split_name}: CUDA preload failed ({exc}); falling back to CPU preload")
                    preload_device_now = "cpu"
                    continue
                if preload_now:
                    print(f"[data] {split_name}: CPU preload failed ({exc}); falling back to lazy disk loading")
                    preload_now = False
                    preload_device_now = "cpu"
                    continue
                raise

    if val_h5 is not None:
        train_ds, train_preload, train_preload_device = _build_with_fallback({
            "h5_path": h5_path,
            "min_valid_cells": 0,
            "return_confidence": return_confidence,
        }, "train", train_preload, train_preload_device)
        val_ds, val_preload, val_preload_device = _build_with_fallback({
            "h5_path": val_h5,
            "min_valid_cells": 0,
            "return_confidence": False,
        }, "val", val_preload, val_preload_device)
    else:
        n_val   = max(1, int(math.ceil(N * val_frac)))
        n_train = N - n_val

        train_ds, train_preload, train_preload_device = _build_with_fallback({
            "h5_path": h5_path,
            "min_valid_cells": 0,
            "indices": range(n_train),
            "return_confidence": return_confidence,
        }, "train", train_preload, train_preload_device)
        val_ds, val_preload, val_preload_device = _build_with_fallback({
            "h5_path": h5_path,
            "min_valid_cells": 0,
            "indices": range(n_train, N),
            "return_confidence": False,
        }, "val", val_preload, val_preload_device)

    # If preloading to CUDA, we must force single-process loading because
    # CUDA tensors cannot be easily shared across worker processes. If
    # preloading to CPU RAM, keep the requested number of workers so the
    # DataLoader can parallelise batching / pinning to speed host-side work.
    if train_preload and num_workers > 0 and train_preload_device == "cuda":
        print("[data] train preload active to CUDA; forcing train num_workers=0")
        train_workers = 0
    else:
        train_workers = num_workers

    if val_preload and num_workers > 0 and val_preload_device == "cuda":
        print("[data] val preload active to CUDA; forcing val num_workers=0")
        val_workers = 0
    else:
        val_workers = num_workers

    def _loader_kwargs(workers: int, pin_memory: bool) -> dict:
        return dict(
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=(workers > 0),
            prefetch_factor=2 if workers > 0 else None,
            drop_last=True,
        )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **_loader_kwargs(train_workers, pin_memory=(train_preload_device != "cuda")),
    )  # type: ignore[arg-type]
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **_loader_kwargs(val_workers, pin_memory=(val_preload_device != "cuda")),
    )  # type: ignore[arg-type]

    print(
        f"[data] train={len(train_ds):,}  val={len(val_ds):,}  batch={batch_size} | "
        f"train_preload={'yes' if train_preload else 'no'}:{train_preload_device} "
        f"val_preload={'yes' if val_preload else 'no'}:{val_preload_device} | "
        f"workers train={train_workers} val={val_workers}"
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# PINN lambda schedule
# ---------------------------------------------------------------------------

def _pinn_lambdas(
    epoch: int,
    target_cal:  float,
    target_bfly: float,
    warmup_start: int = 10,
    warmup_end:   int = 30,
) -> tuple[float, float]:
    """
    Linear warmup of PINN penalty weights:
      epochs  0 – warmup_start-1  : λ = 0
      epochs  warmup_start – warmup_end : linear ramp 0 → target
      epochs  warmup_end+ : λ = target
    """
    if epoch < warmup_start:
        return 0.0, 0.0
    if epoch >= warmup_end:
        return target_cal, target_bfly
    frac = (epoch - warmup_start) / max(1, warmup_end - warmup_start)
    return target_cal * frac, target_bfly * frac


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:      BatesSurrogate,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scaler:     amp.GradScaler,
    grid:       GridConstants,
    lambda_cal:  float,
    lambda_bfly: float,
    grad_clip:   float,
    device:      torch.device,
    time_batches: bool = False,
) -> dict[str, float]:
    model.train()

    sums  = dict(total=0.0, vega=0.0, calendar=0.0, butterfly=0.0)
    steps = 0
    batch_wait_s = 0.0
    h2d_s = 0.0
    compute_s = 0.0

    iterator = iter(loader)

    while True:
        wait_t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        wait_t1 = time.perf_counter()
        batch_wait_s += wait_t1 - wait_t0

        h2d_t0 = time.perf_counter()
        params_norm = batch[0].to(device, non_blocking=True)
        iv_flat     = batch[1].to(device, non_blocking=True)
        mask_flat   = batch[2].to(device, non_blocking=True)
        conf_flat   = batch[3].to(device, non_blocking=True) if len(batch) == 4 else None
        if time_batches and device.type == "cuda":
            torch.cuda.synchronize(device)
        h2d_t1 = time.perf_counter()
        h2d_s += h2d_t1 - h2d_t0

        # Denormalise r and q for vega weight computation
        compute_t0 = time.perf_counter()
        r = params_norm[:, 5] * 0.06   # r at index 5 (5 Heston params then r, q)
        q = params_norm[:, 6] * 0.04   # q at index 6

        # Vega weights: computed in float32 on the ground-truth IV
        vega_w = compute_vega_weights(iv_flat, grid, r, q)

        # Forward pass in BF16
        with amp.autocast("cuda", dtype=torch.bfloat16):
            iv_pred = model(params_norm)

        # Loss in FP32 (cast iv_pred here; grid passed directly)
        bd = total_loss(
            iv_pred.float(), iv_flat, mask_flat, vega_w, grid,
            lambda_cal=lambda_cal, lambda_bfly=lambda_bfly,
            confidence=conf_flat,
        )

        scaler.scale(bd.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if time_batches and device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_t1 = time.perf_counter()
        compute_s += compute_t1 - compute_t0

        sums["total"]     += bd.total.item()
        sums["vega"]      += bd.vega.item()
        sums["calendar"]  += bd.calendar.item()
        sums["butterfly"] += bd.butterfly.item()
        steps += 1

    metrics = {k: v / max(steps, 1) for k, v in sums.items()}
    metrics["batch_wait_ms"] = 1e3 * batch_wait_s / max(steps, 1)
    metrics["h2d_ms"] = 1e3 * h2d_s / max(steps, 1)
    metrics["compute_ms"] = 1e3 * compute_s / max(steps, 1)
    return metrics


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:      BatesSurrogate,
    loader:     DataLoader,
    grid:       GridConstants,
    lambda_cal:  float,
    lambda_bfly: float,
    device:      torch.device,
) -> dict[str, float]:
    model.eval()

    sums    = dict(total=0.0, vega=0.0, calendar=0.0, butterfly=0.0)
    sq_err_sum = 0.0
    valid_sum  = 0.0
    steps      = 0

    for batch in loader:
        params_norm = batch[0].to(device, non_blocking=True)
        iv_flat     = batch[1].to(device, non_blocking=True)
        mask_flat   = batch[2].to(device, non_blocking=True)

        r = params_norm[:, 5] * 0.06
        q = params_norm[:, 6] * 0.04

        vega_w  = compute_vega_weights(iv_flat, grid, r, q)
        iv_pred = model(params_norm)

        bd = total_loss(
            iv_pred.float(), iv_flat, mask_flat, vega_w, grid,
            lambda_cal=lambda_cal, lambda_bfly=lambda_bfly,
        )

        sums["total"]     += bd.total.item()
        sums["vega"]      += bd.vega.item()
        sums["calendar"]  += bd.calendar.item()
        sums["butterfly"] += bd.butterfly.item()

        m  = mask_flat.float()
        sq_err_sum += ((iv_pred.float() - iv_flat) ** 2 * m).sum().item()
        valid_sum  += m.sum().item()
        steps += 1

    metrics = {k: v / max(steps, 1) for k, v in sums.items()}
    mse     = sq_err_sum / max(valid_sum, 1.0)
    metrics["ivrmse_bps"] = math.sqrt(mse) * 10_000.0
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    out_dir:  Path,
    tag:      str,
    model:    BatesSurrogate,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch:    int,
    best_val_ivrmse_bps: float,
    best_val_monitor: float,
    config:   dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_ivrmse_bps": best_val_ivrmse_bps,
        "best_val_monitor": best_val_monitor,
        "config":              config,
    }
    torch.save(ckpt, out_dir / f"{tag}.pt")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train BatesSurrogate IV-surface surrogate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--h5",          type=str,   default="data/heston_guided_pilot.h5")
    ap.add_argument("--val-h5",      type=str,   default=None,
                    help="Separate validation HDF5 (overrides --val-frac split)")
    ap.add_argument("--out-dir",     type=str,   default="model/runs",
                    help="Directory for checkpoints and training log")
    ap.add_argument("--width",       type=int,   default=512)
    ap.add_argument("--n-blocks",    type=int,   default=6)
    ap.add_argument("--batch-size",  type=int,   default=8196)
    ap.add_argument("--epochs",      type=int,   default=200)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight-decay",type=float, default=1e-5)
    ap.add_argument("--lambda-cal",  type=float, default=0.10,
                    help="Calendar-spread PINN weight (at full warmup)")
    ap.add_argument("--lambda-bfly", type=float, default=0.05,
                    help="Butterfly PINN weight (at full warmup)")
    ap.add_argument("--grad-clip",   type=float, default=1.0)
    ap.add_argument("--num-workers", type=int,   default=8)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--resume",      type=str,   default=None,
                    help="Checkpoint path to resume training from")
    ap.add_argument("--no-pinn",     action="store_true",
                    help="Disable PINN penalties (ablation study)")
    ap.add_argument("--val-frac",    type=float, default=0.10)
    ap.add_argument("--early-stop",  type=int,   default=30,
                    help="Early stopping patience (epochs)")
    ap.add_argument("--preload",      type=str,   default="none",
                    choices=["none", "train", "all", "auto"],
                    help="Preload mode: none/train/all or auto (VRAM -> RAM -> disk)")
    ap.add_argument("--confidence",   action="store_true",
                    help="Use per-cell confidence weights in training loss (fine-tune mode)")
    ap.add_argument("--time-batches", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Measure DataLoader wait, host-to-device copy, and compute time per batch")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Grid constants
    grid = GridConstants.from_h5(args.h5).to(device)
    print(f"[grid] NK={grid.NK}  NT={grid.NT}  N_FLAT={grid.N_FLAT}")

    resolved_preload = args.preload
    train_preload_device = "cpu"
    val_preload_device = "cpu"
    if args.preload == "auto":
        resolved_preload, train_preload_device, val_preload_device = resolve_auto_preload(
            h5_path=args.h5,
            val_frac=args.val_frac,
            val_h5=args.val_h5,
            train_confidence=args.confidence,
            device=device,
        )
        print(
            f"[preload-auto] selected preload={resolved_preload} "
            f"train_device={train_preload_device} val_device={val_preload_device}"
        )

    # Data
    train_loader, val_loader = build_dataloaders(
        h5_path           = args.h5,
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
        val_frac          = args.val_frac,
        val_h5            = args.val_h5,
        seed              = args.seed,
        preload           = resolved_preload,
        train_preload_device = train_preload_device,
        val_preload_device   = val_preload_device,
        return_confidence = args.confidence,
    )

    # Model
    model_config = dict(
        n_params  = 7,   # 5 Heston params + r + q
        n_outputs = grid.N_FLAT,
        width     = args.width,
        n_blocks  = args.n_blocks,
    )
    model = BatesSurrogate(**model_config).to(device)
    print(f"[model] {model.n_parameters():,} parameters")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )
    scaler = amp.GradScaler("cuda")

    start_epoch = 0
    best_val_ivrmse = float("inf")
    best_val_monitor = float("inf")
    no_improve_count = 0
    warmup_end = 30

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_ivrmse = ckpt.get("best_val_ivrmse_bps", float("inf"))
        best_val_monitor = ckpt.get("best_val_monitor", best_val_ivrmse)
        print(f"[resume] epoch {start_epoch}  best_ivrmse={best_val_ivrmse:.2f} bps")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"

    # Write CSV header
    fieldnames = [
        "epoch", "lr",
        "train_total", "train_vega", "train_cal", "train_bfly",
        "train_batch_wait_ms", "train_h2d_ms", "train_compute_ms",
        "val_total", "val_vega", "val_cal", "val_bfly",
        "val_ivrmse_bps",
    ]
    write_header = not log_path.exists()
    log_fh = open(log_path, "a", newline="")
    writer = csv.DictWriter(log_fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    lambda_cal_target  = 0.0 if args.no_pinn else args.lambda_cal
    lambda_bfly_target = 0.0 if args.no_pinn else args.lambda_bfly

    end_epoch_display = start_epoch + args.epochs if args.resume else args.epochs
    print(f"\n[training] epochs={start_epoch}→{end_epoch_display}  PINN={'off' if args.no_pinn else 'on'}")
    print(f"[training] λ_cal={lambda_cal_target}  λ_bfly={lambda_bfly_target}")
    print(f"[output]   {out_dir}\n")

    end_epoch = start_epoch + args.epochs if args.resume else args.epochs
    for epoch in range(start_epoch, end_epoch):
        t0 = time.time()

        # Current PINN lambdas (warmed up over epochs 10-30)
        lam_cal, lam_bfly = _pinn_lambdas(
            epoch, lambda_cal_target, lambda_bfly_target
        )

        # Auto-reduce PINN lambdas if they dominate (>10× data loss)
        # (checked after validation, applied next epoch)

        train_m = train_one_epoch(
            model, train_loader, optimizer, scaler,
            grid, lam_cal, lam_bfly, args.grad_clip, device,
            time_batches=args.time_batches,
        )
        val_m = validate(
            model, val_loader, grid, lam_cal, lam_bfly, device,
        )

        # Adaptive PINN cap: if PINN/vega > 10, halve lambda targets
        if not args.no_pinn and epoch >= 30:
            pinn_total = (
                lambda_cal_target  * val_m["calendar"] +
                lambda_bfly_target * val_m["butterfly"]
            )
            if val_m["vega"] > 1e-12 and pinn_total / val_m["vega"] > 10.0:
                lambda_cal_target  *= 0.5
                lambda_bfly_target *= 0.5
                print(f"  [pinn cap]  λ_cal→{lambda_cal_target:.4f}  "
                      f"λ_bfly→{lambda_bfly_target:.4f}")

        lr_now = optimizer.param_groups[0]["lr"]
        val_monitor = val_m["total"]
        scheduler.step(val_monitor)

        ivrmse = val_m["ivrmse_bps"]
        elapsed = time.time() - t0

        print(
            f"epoch {epoch:4d}  "
            f"lr={lr_now:.2e}  "
            f"train_vega={train_m['vega']:.6f}  "
            f"val_vega={val_m['vega']:.6f}  "
            f"wait={train_m['batch_wait_ms']:.1f}ms  "
            f"h2d={train_m['h2d_ms']:.1f}ms  "
            f"compute={train_m['compute_ms']:.1f}ms  "
            f"val_ivrmse={ivrmse:.2f} bps  "
            f"[{elapsed:.1f}s]"
        )

        # Log to CSV
        row = {
            "epoch":          epoch,
            "lr":             lr_now,
            "train_total":    train_m["total"],
            "train_vega":     train_m["vega"],
            "train_cal":      train_m["calendar"],
            "train_bfly":     train_m["butterfly"],
            "train_batch_wait_ms": train_m["batch_wait_ms"],
            "train_h2d_ms":        train_m["h2d_ms"],
            "train_compute_ms":    train_m["compute_ms"],
            "val_total":      val_m["total"],
            "val_vega":       val_m["vega"],
            "val_cal":        val_m["calendar"],
            "val_bfly":       val_m["butterfly"],
            "val_ivrmse_bps": ivrmse,
        }
        writer.writerow(row)
        log_fh.flush()

        # Save best checkpoint only after PINN warmup is complete.
        if epoch >= warmup_end:
            best_val_ivrmse = min(best_val_ivrmse, ivrmse)
            if val_monitor < best_val_monitor:
                best_val_monitor = val_monitor
                no_improve_count = 0
                _save_checkpoint(
                    out_dir, "best", model, optimizer, scheduler,
                    epoch, best_val_ivrmse, best_val_monitor, model_config,
                )
                print(
                    f"  [checkpoint] best  val_total={best_val_monitor:.6f}  "
                    f"best_ivrmse={best_val_ivrmse:.2f} bps"
                )
            else:
                no_improve_count += 1
        else:
            print(f"  [warmup] best-checkpoint tracking starts at epoch {warmup_end}")

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            _save_checkpoint(
                out_dir, f"epoch_{epoch:04d}", model, optimizer, scheduler,
                epoch, best_val_ivrmse, best_val_monitor, model_config,
            )

        # Early stopping
        if no_improve_count >= args.early_stop:
            print(
                f"\n[early stop] no val_total improvement for {args.early_stop} "
                f"epochs (patience starts after epoch {warmup_end})."
            )
            break

    log_fh.close()
    print(f"\n[done] best val_total = {best_val_monitor:.6f}")
    print(f"[done] best val IVRMSE = {best_val_ivrmse:.2f} bps")
    print(f"[done] checkpoint → {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

"""NVIDIA Model Optimizer recipes on the trained 4/2 surrogate.

Same measurement as the Heston probe: post-training quantization, weight
compression, and 2:4 magnitude sparsity. The production ONNX stays the FP32
graph unless a recipe stays within 2 bps of the FP32 validation IVRMSE and
exports. Fake-quant does not speed the batch-1 calibration loop; this script
records that, it does not silently swap in a worse pricer.

    python -m model.optimize_four_two --checkpoint runs/four_two/best.pt
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
from model.optimize_modelopt import (  # noqa: E402
    bench,
    n_params,
    quant_configs,
    try_compress,
    try_magnitude_sparsity,
    try_quantize,
    weight_bytes,
)
from sampling import PARAM_HI, PARAM_LO, R_HI, R_LO  # noqa: E402


def load_batch(path: str, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        n_total = f["params"].shape[0]
        idx = np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))
        params = f["params"][idx].astype(np.float32)
        iv = f["iv_surface"][idx].astype(np.float32).reshape(len(idx), -1)
        qm = f["quality_mask"][idx].reshape(len(idx), -1)
        region = f["pricable_region"][:].astype(bool).reshape(-1)
    theta = np.zeros((params.shape[0], 9), dtype=np.float32)
    theta[:, :7] = (params[:, :7] - PARAM_LO.astype(np.float32)) / (
        PARAM_HI.astype(np.float32) - PARAM_LO.astype(np.float32)
    )
    theta[:, 7] = (params[:, 7] - np.float32(R_LO)) / np.float32(R_HI - R_LO)
    finite = np.isfinite(iv)
    mask = finite & (qm < 3) & region
    iv = np.where(finite, iv, 0.0)
    return (
        torch.from_numpy(theta),
        torch.from_numpy(iv.astype(np.float32)),
        torch.from_numpy(mask),
    )


@torch.no_grad()
def score(model, theta, iv, mask) -> float:
    pred = model(theta)
    return ivrmse_bps(pred.float(), iv, mask)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="runs/four_two/best.pt")
    ap.add_argument("--h5", default="data/four_two_val.h5")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--eval", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    base = BatesSurrogate.from_checkpoint(args.checkpoint).cuda().eval()
    print(f"[model] params={n_params(base):,}  weight_mb={weight_bytes(base)/1e6:.2f}")
    theta, iv, mask = load_batch(args.h5, args.calib + args.eval, args.seed)
    theta, iv, mask = theta.cuda(), iv.cuda(), mask.cuda()
    calib, ev = theta[: args.calib], theta[args.calib :]
    ev_iv, ev_mask = iv[args.calib :], mask[args.calib :]
    fp32 = score(base, ev, ev_iv, ev_mask)
    print(f"[fp32] ivrmse={fp32:.2f} bps  b1={bench(base, ev[:1]):.3f} ms")

    best_name, best_delta = "fp32", 0.0
    for name, cfg in quant_configs():
        model, err = try_quantize(base, name, cfg, calib)
        if model is None:
            print(f"{name}: SKIP {err.splitlines()[-1][:160]}")
            continue
        sc = score(model, ev, ev_iv, ev_mask)
        delta = sc - fp32
        print(f"{name}: ivrmse={sc:.2f} bps  delta={delta:+.2f}  b1={bench(model, ev[:1]):.3f} ms")
        if abs(delta) <= 2.0 and (best_name == "fp32" or abs(delta) < abs(best_delta)):
            best_name, best_delta = name, delta
        compressed, cerr = try_compress(model)
        if cerr:
            print(f"  compress: {cerr.splitlines()[-1][:140]}")
        else:
            try:
                csc = score(compressed, ev, ev_iv, ev_mask)
                print(f"  +compress ivrmse={csc:.2f} bps  delta={csc - fp32:+.2f}")
            except Exception as exc:
                print(f"  +compress forward failed: {type(exc).__name__}: {exc}")

    sparse, serr = try_magnitude_sparsity(base)
    if sparse is None:
        print(f"sparse_magnitude: SKIP {serr.splitlines()[-1][:160]}")
    else:
        try:
            sc = score(sparse, ev, ev_iv, ev_mask)
            print(f"sparse_magnitude: ivrmse={sc:.2f} bps  delta={sc - fp32:+.2f}")
        except Exception as exc:
            print(f"sparse_magnitude forward failed: {type(exc).__name__}: {exc}")

    print(
        f"[decision] production graph stays FP32. "
        f"closest recipe={best_name} ({best_delta:+.2f} bps). "
        "Weight-only INT8 can stay inside 2 bps but is slower at batch 1, "
        "which is the calibration loop."
    )


if __name__ == "__main__":
    main()

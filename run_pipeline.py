#!/usr/bin/env python3
"""
Deep calibration full pipeline.

Steps:
  1. Generate synthetic training data from guided param bank
  2. Generate validation set
  3. Train BatesSurrogate
  4. Export to ONNX
  5. Run Python accuracy / diagnostic tests (diagnostics/diag_accuracy.py)
  6. Run Rust integration tests (cargo test --release --ignored)

Usage:
  python run_pipeline.py                            # default: 9M train + 2M val, 200 epochs
  python run_pipeline.py --N 200000 --epochs 50    # fast smoke test
  python run_pipeline.py --skip-datagen            # reuse existing HDF5
  python run_pipeline.py --skip-training           # reuse existing checkpoint
  python run_pipeline.py --skip-export             # skip ONNX export
  python run_pipeline.py --skip-accuracy           # skip Python accuracy diagnostics
  python run_pipeline.py --skip-rust-tests         # skip Rust integration tests
  python run_pipeline.py --accuracy-only           # only run accuracy tests (Python + Rust)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATAGEN_DIR = ROOT / "training data creation"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
RUNS_DIR = MODEL_DIR / "runs"
DIAG_DIR = ROOT / "diagnostics"
OPTIONS_PRICER_DIR = ROOT.parent / "options-pricer"

DEFAULT_GUIDED_BANK = str(DATA_DIR / "guided_param_bank_halfyear_2026_04_08.h5")
DEFAULT_GUIDED_WEIGHT = 0.7
DEFAULT_GUIDED_JITTER = 0.03
DEFAULT_CHUNK = 16_000
DEFAULT_N = 9_000_000
DEFAULT_EPOCHS = 200
DEFAULT_BATCH = 8196
DEFAULT_WIDTH = 512
DEFAULT_N_BLOCKS = 6
DEFAULT_SEED = 42

PY = sys.executable


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path, step_name: str) -> None:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[pipeline] {step_name}")
    print(f"[run] {' '.join(cmd)}")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[pipeline] FAILED: {step_name}  (exit {result.returncode})")
        sys.exit(result.returncode)

    print(f"\n[pipeline] {step_name} done  ({elapsed:.0f}s)")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_datagen(args: argparse.Namespace) -> None:
    """Generate training HDF5 from guided param bank."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    common = [
        PY, "heston_datagen.py",
        "--grid", "high41x14",
        "--nan-policy", "market_consistent",
        "--chunk", str(args.chunk),
        "--seed", str(args.seed),
        "--guided-bank", args.guided_bank,
        "--guided-weight", str(args.guided_weight),
        "--guided-jitter", str(args.guided_jitter),
    ]

    # Training set
    run(
        common + ["--N", str(args.N), "--out", args.train_h5],
        cwd=DATAGEN_DIR,
        step_name=f"datagen  (N={args.N:,}  → {args.train_h5})",
    )

    # Validation set
    val_h5 = args.train_h5.replace(".h5", "_val.h5")
    run(
        common + ["--val", "--val-N", str(args.val_N), "--out", args.train_h5],
        cwd=DATAGEN_DIR,
        step_name=f"datagen val (→ {val_h5})",
    )


def step_train(args: argparse.Namespace) -> None:
    """Train the surrogate network."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    val_h5 = args.train_h5.replace(".h5", "_val.h5")
    val_args = ["--val-h5", val_h5] if Path(val_h5).exists() else []

    cmd = [
        PY, "-m", "model.train",
        "--h5", args.train_h5,
        "--out-dir", str(RUNS_DIR),
        "--width", str(args.width),
        "--n-blocks", str(args.n_blocks),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--preload", "auto",
        # Run 2 fixes: vega loss + weight floor 0.05 + PINN lambdas
        "--data-loss", "vega",
        "--weight-floor", str(args.weight_floor),
        "--lr", str(args.lr),
        "--lambda-cal", str(args.lambda_cal),
        "--lambda-bfly", str(args.lambda_bfly),
        "--lambda-ts", str(args.lambda_ts),
    ] + val_args

    run(cmd, cwd=ROOT, step_name=f"train  ({args.epochs} epochs)")


def step_export(args: argparse.Namespace) -> None:
    """Export trained checkpoint to ONNX."""
    checkpoint = RUNS_DIR / "best.pt"
    if not checkpoint.exists():
        print(f"[pipeline] ERROR: checkpoint not found: {checkpoint}")
        sys.exit(1)

    adapter = RUNS_DIR / "best_pinn_adapter.pt"
    cmd = [
        PY, "-m", "model.export",
        "--checkpoint", str(checkpoint),
        "--out-dir", str(MODEL_DIR),
        "--format", "onnx",
    ]
    if adapter.exists():
        cmd += ["--adapter", str(adapter)]

    run(cmd, cwd=ROOT, step_name="export → ONNX")


def step_accuracy(args: argparse.Namespace) -> None:
    """Run Python diagnostic accuracy tests (diag_accuracy.py)."""
    onnx = MODEL_DIR / "bates_surrogate.onnx"
    if not onnx.exists():
        print(f"[pipeline] ERROR: ONNX model not found: {onnx}")
        print("[pipeline] Run export step first (or --skip-export if already exported).")
        sys.exit(1)

    run(
        [PY, "diag_accuracy.py"],
        cwd=DIAG_DIR,
        step_name="Python accuracy diagnostics (5 tests: κ-θ grid, v₀-θ grid, per-param bias, corr matrix, prior pull)",
    )


def step_rust_tests(args: argparse.Namespace) -> None:
    """Run Rust integration tests (Phase 5c / 6): surrogate round-trip, κ sensitivity, CTMC deep smoke."""
    if not OPTIONS_PRICER_DIR.exists():
        print(f"[pipeline] WARNING: options-pricer not found at {OPTIONS_PRICER_DIR} — skipping Rust tests")
        return

    onnx = MODEL_DIR / "bates_surrogate.onnx"
    if not onnx.exists():
        print(f"[pipeline] ERROR: ONNX model not found: {onnx}")
        sys.exit(1)

    # Tests ordered by Phase 5c priority: round-trip → κ sensitivity → grid recovery → v0/θ → pricing smoke → benchmark
    rust_tests = [
        "synthetic_round_trip_recovers_known_params",
        "surrogate_kappa_sensitivity",
        "kappa_theta_grid_recovery",
        "v0_theta_identifiability",
        "price_american_option_deep_smoke",
        "jnj_american_put_full_comparison",
    ]

    for test in rust_tests:
        cmd = [
            "cargo", "test", "--release",
            "-p", "numerical-methods",
            test,
            "--", "--ignored", "--nocapture",
        ]
        run(cmd, cwd=OPTIONS_PRICER_DIR, step_name=f"Rust test: {test}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Pipeline control
    ap.add_argument("--skip-datagen",    action="store_true", help="Skip data generation")
    ap.add_argument("--skip-training",   action="store_true", help="Skip model training")
    ap.add_argument("--skip-export",     action="store_true", help="Skip ONNX export")
    ap.add_argument("--skip-accuracy",   action="store_true", help="Skip Python accuracy diagnostics")
    ap.add_argument("--skip-rust-tests", action="store_true", help="Skip Rust integration tests")
    ap.add_argument("--accuracy-only",   action="store_true",
                    help="Run only accuracy tests (Python diag + Rust tests); skips datagen/train/export")

    # Datagen
    ap.add_argument("--N", type=int, default=DEFAULT_N,
                    help=f"Training samples (default {DEFAULT_N:,})")
    ap.add_argument("--val-N", type=int, default=2_000_000,
                    help="Validation samples (default 2,000,000)")
    ap.add_argument("--train-h5", type=str,
                    default=str(DATA_DIR / "heston_pipeline.h5"),
                    help="Output HDF5 path for training data")
    ap.add_argument("--guided-bank", type=str, default=DEFAULT_GUIDED_BANK,
                    help="Guided param bank HDF5 (default: latest halfyear bank)")
    ap.add_argument("--guided-weight", type=float, default=DEFAULT_GUIDED_WEIGHT,
                    help=f"Fraction of samples from guided bank (default {DEFAULT_GUIDED_WEIGHT})")
    ap.add_argument("--guided-jitter", type=float, default=DEFAULT_GUIDED_JITTER,
                    help=f"Jitter scale on guided params (default {DEFAULT_GUIDED_JITTER})")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                    help=f"GPU batch size for datagen (default {DEFAULT_CHUNK})")

    # Training
    ap.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size",   type=int,   default=DEFAULT_BATCH)
    ap.add_argument("--width",        type=int,   default=DEFAULT_WIDTH)
    ap.add_argument("--n-blocks",     type=int,   default=DEFAULT_N_BLOCKS)
    ap.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    ap.add_argument("--weight-floor", type=float, default=0.05,
                    help="Vega weight floor (Run 2 fix A, default 0.05)")
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--lambda-cal",   type=float, default=0.01)
    ap.add_argument("--lambda-bfly",  type=float, default=0.005)
    ap.add_argument("--lambda-ts",    type=float, default=0.003)

    args = ap.parse_args()

    if args.accuracy_only:
        args.skip_datagen = True
        args.skip_training = True
        args.skip_export = True

    t_total = time.time()

    print(f"\n{'='*60}")
    print("[pipeline] Deep Calibration — Full Pipeline")
    print(f"{'='*60}")
    print(f"  Python:       {PY}")
    print(f"  Root:         {ROOT}")
    print(f"  Pricer:       {OPTIONS_PRICER_DIR}")
    print(f"  Guided bank:  {args.guided_bank}")
    print(f"  Train data:   {args.train_h5}")
    print(f"  N samples:    {args.N:,}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Steps:  datagen={not args.skip_datagen}  train={not args.skip_training}"
          f"  export={not args.skip_export}"
          f"  py-accuracy={not args.skip_accuracy}  rust-tests={not args.skip_rust_tests}")
    print(f"{'='*60}\n")

    if not args.skip_datagen:
        if not Path(args.guided_bank).exists():
            print(f"[pipeline] ERROR: guided bank not found: {args.guided_bank}")
            print("[pipeline] Pass --guided-bank <path> or generate one first with "
                  "training data creation/run_full_real_guided_pipeline.py")
            sys.exit(1)
        step_datagen(args)

    if not args.skip_training:
        if not Path(args.train_h5).exists():
            print(f"[pipeline] ERROR: training data not found: {args.train_h5}")
            print("[pipeline] Run without --skip-datagen first.")
            sys.exit(1)
        step_train(args)

    if not args.skip_export:
        step_export(args)

    if not args.skip_accuracy:
        step_accuracy(args)

    if not args.skip_rust_tests:
        step_rust_tests(args)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"[pipeline] All steps complete in {elapsed/60:.1f} min")
    print(f"  ONNX model:  {MODEL_DIR / 'bates_surrogate.onnx'}")
    print(f"  Checkpoint:  {RUNS_DIR / 'best.pt'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

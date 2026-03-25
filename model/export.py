"""
model/export.py
===============
Export a trained BatesSurrogate to TorchScript and ONNX (opset 17).

Usage
-----
  # Export from best checkpoint
  python -m model.export --checkpoint model/runs/best.pt

  # TorchScript only
  python -m model.export --checkpoint model/runs/best.pt --format torchscript

  # ONNX only
  python -m model.export --checkpoint model/runs/best.pt --format onnx

Output
------
  model/bates_surrogate.pt    (TorchScript, optimised for inference)
  model/bates_surrogate.onnx  (ONNX opset 17, dynamic batch axis)

TorchScript is recommended for Rust integration via tch-rs.
ONNX is recommended for production via ort (ONNX Runtime), which has a
smaller binary footprint than libtorch.

References
----------
  deep_calibration_research.tex §6.4-6.5 (lines 1184-1213)
  deep_calibration_research.tex §7 (Rust interop)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.network import BatesSurrogate


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------

def export_torchscript(
    model:       BatesSurrogate,
    output_path: str | Path = "model/bates_surrogate.pt",
    optimize:    bool = True,
) -> str:
    """
    Export model to TorchScript via torch.jit.trace.

    The surrogate is a pure feedforward MLP with no conditional branches,
    so trace (rather than script) is appropriate and produces a more
    compact graph.

    Args:
        model:       Trained BatesSurrogate (will be set to eval mode).
        output_path: Destination .pt file.
        optimize:    If True, apply torch.jit.optimize_for_inference.

    Returns:
        Absolute path to the saved file (str).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu().eval()
    dummy = torch.zeros(1, model.n_params, dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    if optimize:
        traced = torch.jit.optimize_for_inference(traced)

    traced.save(str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"[torchscript] saved → {output_path}  ({size_mb:.1f} MB)")
    return str(output_path.resolve())


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model:        BatesSurrogate,
    output_path:  str | Path = "model/bates_surrogate.onnx",
    opset_version: int = 18,
    verify:       bool = True,
) -> str:
    """
    Export model to ONNX (opset 17) with a dynamic batch dimension.

    Performs a numerical consistency check after export using onnxruntime:
      max|pytorch_out − ort_out| < 1e-4

    Requires:
      pip install onnx onnxscript onnxruntime

    Args:
        model:         Trained BatesSurrogate.
        output_path:   Destination .onnx file.
        opset_version: ONNX opset.  17 is recommended for PyTorch 2.5+.
        verify:        Run onnx.checker + ort numerical consistency check.

    Returns:
        Absolute path to the saved file (str).

    Raises:
        ImportError:  If onnx or onnxruntime are not installed.
        AssertionError: If numerical consistency check fails.
    """
    try:
        import onnx
    except ImportError as e:
        raise ImportError(
            "onnx is required for ONNX export.  "
            "Install with: pip install onnx onnxscript"
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu().eval()
    dummy = torch.zeros(1, model.n_params, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            input_names   = ["parameters"],
            output_names  = ["iv_surface"],
            dynamic_axes  = {
                "parameters": {0: "batch_size"},
                "iv_surface": {0: "batch_size"},
            },
            opset_version = opset_version,
            do_constant_folding = True,
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f"[onnx] saved → {output_path}  ({size_mb:.1f} MB)")

    if verify:
        # 1. ONNX model validity check
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("[onnx] model validity: OK")

        # 2. Numerical consistency vs PyTorch
        try:
            import onnxruntime as ort
        except ImportError:
            print("[onnx] onnxruntime not installed — skipping numerical check")
            return str(output_path.resolve())

        import numpy as np

        sess = ort.InferenceSession(str(output_path))

        # Test batch sizes 1 and 4
        for batch in [1, 4]:
            x_np  = np.random.randn(batch, model.n_params).astype(np.float32)
            x_pt  = torch.from_numpy(x_np)

            with torch.no_grad():
                pt_out = model(x_pt).numpy()

            ort_out = sess.run(None, {"parameters": x_np})[0]

            max_err = float(np.abs(pt_out - ort_out).max())
            print(f"[onnx] batch={batch}  max|pytorch−ort|={max_err:.2e}", end="")
            assert max_err < 1e-4, (
                f"ONNX numerical error too large: {max_err:.2e} ≥ 1e-4"
            )
            print("  ✓")

    return str(output_path.resolve())


# ---------------------------------------------------------------------------
# Convenience: load checkpoint and export
# ---------------------------------------------------------------------------

def load_and_export(
    checkpoint_path: str | Path,
    out_dir:         str | Path = "model/",
    formats:         list[str]  | None = None,
) -> dict[str, str]:
    """
    Load a training checkpoint, reconstruct BatesSurrogate, and export.

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by train.py.
        out_dir:         Directory for exported files.
        formats:         List of formats: ["torchscript", "onnx"].
                         Defaults to both.

    Returns:
        Dict mapping format name to absolute output file path.
    """
    if formats is None:
        formats = ["torchscript", "onnx"]

    out_dir = Path(out_dir)
    model   = BatesSurrogate.from_checkpoint(checkpoint_path)
    print(f"[export] loaded checkpoint: {checkpoint_path}")
    print(f"[export] model: {model.n_parameters():,} parameters")

    results: dict[str, str] = {}

    if "torchscript" in formats:
        pt_path = out_dir / "bates_surrogate.pt"
        results["torchscript"] = export_torchscript(model, pt_path)

    if "onnx" in formats:
        onnx_path = out_dir / "bates_surrogate.onnx"
        results["onnx"] = export_onnx(model, onnx_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Export trained BatesSurrogate to TorchScript and/or ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Training checkpoint path (e.g. model/runs/best.pt)")
    ap.add_argument("--out-dir",    type=str, default="model/",
                    help="Output directory for exported files")
    ap.add_argument("--format",     type=str, default="both",
                    choices=["torchscript", "onnx", "both"],
                    help="Export format")
    ap.add_argument("--no-verify",  action="store_true",
                    help="Skip ONNX numerical verification step")
    args = ap.parse_args()

    fmts = ["torchscript", "onnx"] if args.format == "both" else [args.format]
    paths = load_and_export(
        checkpoint_path = args.checkpoint,
        out_dir         = args.out_dir,
        formats         = fmts,
    )

    print("\n[export] complete:")
    for fmt, path in paths.items():
        print(f"  {fmt:12s} → {path}")

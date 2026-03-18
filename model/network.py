"""
model/network.py
================
BatesSurrogate  —  feedforward IV-surface surrogate for the Bates (SVJ) model.

Architecture (default):
  Input  θ : (B, 10)   8 Bates params + r + q, all normalised to [0, 1]
  Stem      : Linear(10→512) → LayerNorm → SiLU
  Body      : 6 × ResBlock(512)  [pre-norm, skip + SiLU]
  Head      : Linear(512→686) → softplus(β=3) + 0.01
  Output    : (B, 686)  implied volatilities ∈ (0.01, ~3.0)

~3.9 M parameters.  BF16-safe (pre-norm avoids gradient spikes).

Also exports:
  GridConstants  —  fixed 49-point moneyness / 14-point maturity grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Grid constants  (high41x14 preset — 49 strikes × 14 maturities)
# ---------------------------------------------------------------------------

# Hardcoded defaults match the high41x14 preset in heston_datagen.py
_DEFAULT_LOG_MONEYNESS = np.linspace(-0.80, 0.40, 49, dtype=np.float32)
_DEFAULT_MATURITIES = np.array(
    [
        1 / 52, 2 / 52, 3 / 52,
        1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 9 / 12,
        1.0, 1.25, 1.5, 2.0, 3.0,
    ],
    dtype=np.float32,
)


@dataclass
class GridConstants:
    """Fixed (K, T) grid arrays shared by network, loss, and calibrate modules."""

    log_moneyness: torch.Tensor   # (NK,)  float32
    maturities: torch.Tensor      # (NT,)  float32
    NK: int = field(init=False)
    NT: int = field(init=False)
    N_FLAT: int = field(init=False)

    def __post_init__(self) -> None:
        self.NK = int(self.log_moneyness.shape[0])
        self.NT = int(self.maturities.shape[0])
        self.N_FLAT = self.NK * self.NT

    def to(self, device: torch.device | str) -> "GridConstants":
        return GridConstants(
            log_moneyness=self.log_moneyness.to(device),
            maturities=self.maturities.to(device),
        )

    @classmethod
    def default(cls) -> "GridConstants":
        """Returns the high41x14 grid (hardcoded)."""
        return cls(
            log_moneyness=torch.from_numpy(_DEFAULT_LOG_MONEYNESS),
            maturities=torch.from_numpy(_DEFAULT_MATURITIES),
        )

    @classmethod
    def from_h5(cls, h5_path: str | Path) -> "GridConstants":
        """Loads grid arrays from HDF5 attributes (falls back to defaults)."""
        with h5py.File(h5_path, "r") as f:
            log_m = np.array(
                f.attrs.get("log_moneyness", _DEFAULT_LOG_MONEYNESS), dtype=np.float32
            )
            mats = np.array(
                f.attrs.get("maturities", _DEFAULT_MATURITIES), dtype=np.float32
            )
        return cls(
            log_moneyness=torch.from_numpy(log_m),
            maturities=torch.from_numpy(mats),
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Pre-norm residual block:
        h = LayerNorm(x)
        h = SiLU(Linear(h))
        h = Linear(h)
        out = SiLU(x + h)        ← post-add SiLU

    Pre-norm (LayerNorm before the linear) is more stable than post-norm
    under BF16 autocast at depth ≥ 6 (avoids gradient spikes from the
    normalisation being applied to un-normalised residuals).
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln  = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return F.silu(x + h)


# ---------------------------------------------------------------------------
# Main surrogate
# ---------------------------------------------------------------------------

class BatesSurrogate(nn.Module):
    """
    Feedforward pricing surrogate  θ → Σ(θ).

    Maps 10-dimensional Bates/Heston parameter vectors (normalised to [0, 1])
    to a 686-dimensional flattened implied-volatility surface (49 strikes ×
    14 maturities).  Output values are strictly positive via
    ``softplus(x, β=3) + 0.01``.

    Args:
        n_params:  Input dimension.  Default 10 (8 Bates + r + q).
        n_outputs: Output dimension. Default 686 (49 × 14).
        width:     Hidden width.     Default 512.
        n_blocks:  Residual blocks.  Default 6 (≥ 6 requires residual connections).
    """

    def __init__(
        self,
        n_params: int = 10,
        n_outputs: int = 686,
        width: int = 512,
        n_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.n_params  = n_params
        self.n_outputs = n_outputs
        self.width     = width
        self.n_blocks  = n_blocks

        # Stem: project input to hidden width
        self.stem = nn.Sequential(
            nn.Linear(n_params, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )

        # Body: residual blocks
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])

        # Head: project to output
        self.head = nn.Linear(width, n_outputs)

        # Weight initialisation (PyTorch defaults are Xavier uniform for Linear,
        # which is fine; we tweak the final head to start near a plausible IV)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: (B, n_params) float32 or bfloat16, values in [0, 1].
        Returns:
            iv_pred: (B, n_outputs) float32, implied volatilities > 0.01.
        """
        x = self.stem(theta)
        for block in self.blocks:
            x = block(x)
        raw = self.head(x)
        # softplus(β=3) is a smooth, everywhere-differentiable lower bound;
        # + 0.01 ensures IVs never collapse to zero during early training.
        return F.softplus(raw, beta=3) + 0.01

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "BatesSurrogate":
        """Reconstructs model from a training checkpoint dict."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        model = cls(
            n_params  = cfg.get("n_params",  10),
            n_outputs = cfg.get("n_outputs", 686),
            width     = cfg.get("width",     512),
            n_blocks  = cfg.get("n_blocks",  6),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid  = GridConstants.default()
    model = BatesSurrogate()
    print(f"Parameters: {model.n_parameters():,}")
    print(f"Grid: NK={grid.NK}, NT={grid.NT}, N_FLAT={grid.N_FLAT}")

    x  = torch.rand(4, 10)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    assert out.shape == (4, 686)
    assert out.min() > 0.0

    # BF16 autocast test
    if torch.cuda.is_available():
        model = model.cuda()
        x_cuda = x.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_bf16 = model(x_cuda)
        print(f"BF16 output dtype: {out_bf16.dtype}  shape: {out_bf16.shape}")

    print("network.py: all checks passed.")

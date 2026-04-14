"""
model/network.py
================
Grid-aware IV-surface surrogate for Heston/Bates calibration.

Default architecture:
    Input θ: (B, 7)  -> trunk MLP (pre-norm residual blocks)
    Head:
        1) factorized grid head (low-rank strike x maturity composition)
        2) small residual correction head on the flattened surface
    Output: softplus(raw, beta=3) + 0.01

The factorized head is tailored to implied-vol surfaces where most variation
is low-rank across strike/maturity, improving sample efficiency versus a
single flat linear projection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

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


class SwiGLU(nn.Module):
    """SwiGLU module suitable for nn.Sequential.

    Expects last-dimension to be 2 * dim; returns SiLU(a) * b (dim).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """Functional SwiGLU: splits the last dim in half and applies gating.
    """
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b

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
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = swiglu(h)
        h = self.fc2(h)
        return F.silu(x + h)


class GridFactorizedHead(nn.Module):
    """Low-rank strike x maturity head with additive row/column biases.

    Given trunk features h in R^width:
      A = Wk(h) in R^(NK x R)
      B = Wt(h) in R^(NT x R)
      S_ij = <A_i, B_j> + bk_i + bt_j + bg

    A small residual linear head is added by the main model to recover
    high-frequency structure not captured by the low-rank term.
    """

    def __init__(self, width: int, nk: int, nt: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.nk = nk
        self.nt = nt
        self.rank = rank

        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

        self.k_factors = nn.Linear(width, nk * rank)
        self.t_factors = nn.Linear(width, nt * rank)
        self.k_bias = nn.Linear(width, nk)
        self.t_bias = nn.Linear(width, nt)
        self.global_bias = nn.Linear(width, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        bsz = h.shape[0]
        z = self.dropout(self.norm(h))

        a = self.k_factors(z).view(bsz, self.nk, self.rank)
        b = self.t_factors(z).view(bsz, self.nt, self.rank)
        kb = self.k_bias(z).view(bsz, self.nk, 1)
        tb = self.t_bias(z).view(bsz, 1, self.nt)
        gb = self.global_bias(z).view(bsz, 1, 1)

        raw_grid = torch.einsum("bnr,bmr->bnm", a, b) + kb + tb + gb
        return raw_grid.reshape(bsz, self.nk * self.nt)


# ---------------------------------------------------------------------------
# Main surrogate
# ---------------------------------------------------------------------------

class BatesSurrogate(nn.Module):
    """
    Feedforward pricing surrogate θ -> (Sigma(theta), theta_aux).

    Maps input vectors (6: kappa, theta, sigma_v, rho, v0, carry_norm, all normalised to
    [0, 1]) to both:
      1. Flattened IV surface (686 = 49 × 14)
      2. Auxiliary parameter predictions (5 Heston, for regularization during calibration)

    Risk-free rate (r) and dividend yield (q) are market observables, not calibration
    parameters. They are applied separately during pricing, not part of the model input.

    Dual-head architecture improves parameter identifiability by constraining
    the latent representations to preserve the input-parameter relationship.

    Args:
        n_params:  Input dimension.  Default 5 (5 Heston: kappa, theta, sigma, rho, v0).
        n_outputs: Output dimension. Default 686 (49 × 14).
        width:     Hidden width.     Default 512.
        n_blocks:  Residual blocks.  Default 6.
        nk:        Number of strikes in grid. Default 49.
        nt:        Number of maturities in grid. Default 14.
        rank:      Low-rank dimension for the factorized head. Default 24.
    """

    def __init__(
        self,
        n_params: int = 6,
        n_outputs: int = 686,
        width: int = 512,
        n_blocks: int = 6,
        nk: int = 49,
        nt: int = 14,
        rank: int = 24,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.n_params  = n_params
        self.n_outputs = n_outputs
        self.width     = width
        self.n_blocks  = n_blocks
        self.nk        = nk
        self.nt        = nt
        self.rank      = rank

        if self.nk * self.nt != self.n_outputs:
            raise ValueError(
                f"Expected nk*nt == n_outputs, got {self.nk}*{self.nt} != {self.n_outputs}"
            )

        # Stem: project input to hidden width
        self.stem = nn.Sequential(
            nn.Linear(n_params, width * 2),
            nn.LayerNorm(width * 2),
            SwiGLU(),
            nn.Dropout(dropout),
        )
        
        # Body: residual blocks
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])

        # Grid-aware head plus a residual correction on the flattened surface.
        self.grid_head = GridFactorizedHead(width, nk=self.nk, nt=self.nt, rank=rank, dropout=dropout)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(width, n_outputs),
        )

        # Auxiliary parameter prediction head (dual-head for identifiability).
        # Predicts the 5 Heston parameters [kappa, theta, sigma, rho, v0] in [0,1].
        # During training, this guides the latent representation toward meaningful
        # parameter space, improving calibration robustness.
        self.param_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 5),  # Output 5 Heston parameters
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

        # Weight initialisation (PyTorch defaults are Xavier uniform for Linear,
        # which is fine; we tweak the final head to start near a plausible IV)
        for layer in [
            self.grid_head.k_factors,
            self.grid_head.t_factors,
            self.grid_head.k_bias,
            self.grid_head.t_bias,
            self.grid_head.global_bias,
        ]:
            nn.init.xavier_uniform_(layer.weight, gain=0.05)
            nn.init.zeros_(layer.bias)

        final = self.residual_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("residual_head final layer must be nn.Linear")
        nn.init.zeros_(final.bias)
        nn.init.xavier_uniform_(final.weight, gain=0.1)

        # Initialize param_head to output ~0.5 (center of [0,1])
        param_final = self.param_head[-2]  # Linear layer before Sigmoid
        if not isinstance(param_final, nn.Linear):
            raise TypeError("param_head linear layer not found")
        nn.init.zeros_(param_final.bias)
        nn.init.xavier_uniform_(param_final.weight, gain=0.1)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Main forward pass: returns IV surface only.

        Args:
            theta: (B, n_params) float32 or bfloat16, values in [0, 1].
        Returns:
            iv_pred: (B, n_outputs) float32, implied volatilities > 0.01.
        """
        x = self.stem(theta)
        for block in self.blocks:
            x = block(x)
        raw = self.grid_head(x) + self.residual_head(x)
        # softplus(β=3) is a smooth, everywhere-differentiable lower bound;
        # + 0.01 ensures IVs never collapse to zero during early training.
        return F.softplus(raw, beta=3) + 0.01

    def forward_dual(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dual-head forward pass: returns both IV surface and parameter predictions.

        Used during training to improve parameter identifiability.

        Args:
            theta: (B, n_params) float32 or bfloat16, values in [0, 1].
        Returns:
            (iv_pred, param_pred):
              - iv_pred: (B, n_outputs) implied volatilities
              - param_pred: (B, 5) predicted Heston parameters in [0, 1]
        """
        x = self.stem(theta)
        for block in self.blocks:
            x = block(x)
        raw = self.grid_head(x) + self.residual_head(x)
        iv_pred = F.softplus(raw, beta=3) + 0.01
        param_pred = self.param_head(x)
        return iv_pred, param_pred

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_compatible_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load checkpoints from both legacy and current architectures.

        Legacy checkpoints used a single flat head named ``head``. We map that
        into the residual head's final linear layer and leave the factorized
        head at its initialisation.
        """
        sd = dict(state_dict)

        # Legacy mapping: head.{weight,bias} -> residual_head.4.{weight,bias}
        if "head.weight" in sd and "residual_head.4.weight" not in sd:
            sd["residual_head.4.weight"] = sd.pop("head.weight")
        if "head.bias" in sd and "residual_head.4.bias" not in sd:
            sd["residual_head.4.bias"] = sd.pop("head.bias")

        missing, unexpected = self.load_state_dict(sd, strict=False)

        # Allow missing grid-head, residual-head, and param-head parameters when loading legacy checkpoints.
        allowed_missing_prefixes = (
            "grid_head.",
            "residual_head.0.",
            "residual_head.1.",
            "residual_head.3.",
            "param_head.",  # New dual-head; old checkpoints won't have it
        )
        bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes)]
        bad_unexpected = [k for k in unexpected if not k.startswith("head.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Incompatible checkpoint state_dict. "
                f"Missing keys: {bad_missing}; Unexpected keys: {bad_unexpected}"
            )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "BatesSurrogate":
        """Reconstructs model from a training checkpoint dict."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        model = cls(
            n_params  = cfg.get("n_params",  5),
            n_outputs = cfg.get("n_outputs", 686),
            width     = cfg.get("width",     512),
            n_blocks  = cfg.get("n_blocks",  6),
            nk        = cfg.get("nk",        49),
            nt        = cfg.get("nt",        14),
            rank      = cfg.get("rank",      24),
            dropout   = cfg.get("dropout",   0.10),
        )
        model.load_compatible_state_dict(ckpt["model_state_dict"])
        return model


# ---------------------------------------------------------------------------
# PINN LoRA adapter
# ---------------------------------------------------------------------------

class PINNAdapter(nn.Module):
    """
    Low-rank residual correction for PINN fine-tuning.

    Intended use: after training a base BatesSurrogate with --no-pinn, freeze
    the base model and train only this adapter with PINN (calendar + butterfly)
    constraints.  The adapter applies a small learned correction to the base
    IV surface without touching the surrogate weights.

    Forward:
        iv_corrected = iv_base + up(silu(down(iv_base)))

    where down: N_FLAT → rank  and  up: rank → N_FLAT.
    Initialised so the correction starts near zero (identity pass-through).

    Args:
        n_flat: IV surface size (default 686 = 49 × 14).
        rank:   Bottleneck rank.  Larger = more capacity; 32–64 is usually enough.
    """

    def __init__(self, n_flat: int = 686, rank: int = 32) -> None:
        super().__init__()
        self.down = nn.Linear(n_flat, rank, bias=True)
        self.up   = nn.Linear(rank, n_flat, bias=False)

        nn.init.xavier_uniform_(self.down.weight, gain=0.01)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)  # start as identity (zero correction)

    def forward(self, iv: torch.Tensor) -> torch.Tensor:
        return iv + self.up(F.silu(self.down(iv)))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BatesSurrogateWithPINN(nn.Module):
    """
    Combines a frozen BatesSurrogate base with a trainable PINNAdapter.

    Used during PINN fine-tuning: the base model is loaded from a checkpoint
    and its parameters are frozen; only the adapter is updated.

    At export time this class is traced as a single ONNX model so the Rust
    inference code requires no changes.
    """

    def __init__(self, base: "BatesSurrogate", adapter: PINNAdapter) -> None:
        super().__init__()
        self.base    = base
        self.adapter = adapter

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        iv_base = self.base(theta)
        return self.adapter(iv_base)

    def freeze_base(self) -> None:
        """Freeze all base model parameters; adapter remains trainable."""
        for p in self.base.parameters():
            p.requires_grad_(False)

    def unfreeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad_(True)

    @classmethod
    def from_checkpoints(
        cls,
        base_checkpoint: str | Path,
        adapter_checkpoint: str | Path | None = None,
        rank: int = 32,
    ) -> "BatesSurrogateWithPINN":
        """Load base from checkpoint and optionally restore a saved adapter."""
        base    = BatesSurrogate.from_checkpoint(base_checkpoint)
        n_flat  = base.n_outputs
        adapter = PINNAdapter(n_flat=n_flat, rank=rank)
        if adapter_checkpoint is not None:
            ckpt = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
            adapter.load_state_dict(ckpt["adapter_state_dict"])
        return cls(base, adapter)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid  = GridConstants.default()
    model = BatesSurrogate()
    print(f"Parameters: {model.n_parameters():,}")
    print(f"Grid: NK={grid.NK}, NT={grid.NT}, N_FLAT={grid.N_FLAT}")

    x  = torch.rand(4, 7)
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

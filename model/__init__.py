"""
model/
------
Deep-calibration neural network package for the Bates/Heston surrogate.

Modules
-------
  network    BatesSurrogate, ResBlock, GridConstants
  loss       vega_weighted_mse, calendar_spread_penalty,
             durrleman_butterfly_penalty, total_loss, ivrmse_bps
  train      Training loop CLI (python -m model.train)
  calibrate  calibrate_single, calibrate_batch, CalibrateResult
  export     export_torchscript, export_onnx, load_and_export

Quick start
-----------
  from model.network import BatesSurrogate, GridConstants
  from model.calibrate import calibrate_single

  grid  = GridConstants.default()
  model = BatesSurrogate.from_checkpoint("model/runs/best.pt")
  result = calibrate_single(model, iv_market, mask_market, grid)
"""
from .network import BatesSurrogate, GridConstants, ResBlock
from .loss import (
    compute_vega_weights,
    vega_weighted_mse,
    calendar_spread_penalty,
    durrleman_butterfly_penalty,
    total_loss,
    ivrmse_bps,
    LossBreakdown,
)
from .calibrate import (
    calibrate_single,
    calibrate_batch,
    CalibrateResult,
    denormalize,
    normalize,
    PARAM_BOUNDS_PHYSICAL,
    PARAM_NAMES,
)
from .export import export_torchscript, export_onnx, load_and_export

__all__ = [
    "BatesSurrogate", "GridConstants", "ResBlock",
    "compute_vega_weights", "vega_weighted_mse",
    "calendar_spread_penalty", "durrleman_butterfly_penalty",
    "total_loss", "ivrmse_bps", "LossBreakdown",
    "calibrate_single", "calibrate_batch", "CalibrateResult",
    "denormalize", "normalize", "PARAM_BOUNDS_PHYSICAL", "PARAM_NAMES",
    "export_torchscript", "export_onnx", "load_and_export",
]

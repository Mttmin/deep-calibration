# Deep-calibration

Calibration of stochastic local volatility models using deep learning techniques, aiming to be used by my option pricer

## Synthetic Dataset

This project builds realistic synthetic implied-volatility (IV) surfaces in three steps:

1. Pull real option chains from Alpha Vantage for many tickers.
2. Calibrate Bates model parameters to those real surfaces.
3. Generate large synthetic datasets guided by those calibrated parameters.

The goal is to make synthetic surfaces look closer to real market behavior, especially short-maturity skew and wings.

### Main Command

Run the full pipeline from the `training data creation/` folder:

```bash
python run_full_real_guided_pipeline.py
```

To generate a larger synthetic set: use `--generate-N` (e.g. `--generate-N 100000`) to specify how many guided synthetic surfaces to create.

### Files created

Files are written to `data/`:

- `real_surfaces_<prefix>.h5`: real surfaces built from market data
- `guided_param_bank_<prefix>.h5`: top calibrated parameter candidates
- `heston_guided_<prefix>.h5`: final guided synthetic dataset

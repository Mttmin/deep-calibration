# Scenario C data pipeline rebuild plan

Goal: rebuild the Heston training data generation so the surrogate can be
trained to research-grade pricing accuracy across log-moneyness -0.8 to
+0.4 and maturities 1 week to 3 years, with honest masking of cells where
option prices are below meaningful precision.

The existing pipeline generates 10M surfaces via COS pricer with
HALF_ABS=3.0 and a naive fill for NaN cells. This produced wing label
corruption of ~70% pricing error at wing_1yr per audit D1 (2026-04-17).
This plan replaces the generator with a multi-method pricer, validates
with high-precision Monte Carlo, regenerates training data with honest
per-cell quality masks, and retrains the surrogate.

Estimated total work: 5 focused days. This plan is written for sequential
execution with three parallel tasks at the start.

---

## Ground rules

- Do not modify the training code (`model/train.py`, `model/loss.py`,
  `model/network.py`) during Phases 1-3. Training code changes happen in
  Phase 4 only.
- Existing pipeline code stays intact until the new pipeline is
  validated. Write new code alongside old code, do not replace in place.
- All new code goes under `training data creation/v2/` to preserve the
  old pipeline at `training data creation/`.
- Every phase ends with a concrete go/no-go check that determines whether
  to proceed or stop and report.
- When a step fails or produces unexpected results, stop and write a
  `BLOCKER_<phase>_<step>.md` report rather than attempting heroics.
- Use the existing `.conda` env unless explicitly adding a new
  dependency, and if adding one, justify it in the step's output.

---

## Phase 0: Parallel preparation (Day 1)

Three independent tasks that can run simultaneously. Each has its own
deliverable and does not depend on the others. Run these as three
parallel sub-agents or three separate terminal sessions.

### 0A: Literature review on asymptotic Heston pricing

Research and write a summary document at

`training data creation/v2/docs/asymptotic_methods.md` covering:

1. Forde-Jacquier-Lee asymptotic expansion for Heston implied volatility
   in the deep-OTM regime. Write the explicit formula for the leading
   order and next-to-leading order terms. Cite: Forde & Jacquier (2011),
   "Small-time asymptotics for implied volatility under the Heston
   model."
2. Lee's moment formula for large-moneyness IV asymptotics. Give the
   slope formula: for k -> +inf, sigma(k)^2 * T ~ psi_R(p*) * k where
   p* is the critical moment.
3. Gatheral's SVI and SSVI parameterizations as potential fallback wing
   extrapolation methods. Summary only, not for implementation here.
4. QE discretization for Heston Monte Carlo (Andersen 2008). Explain
   why Euler discretization fails for Heston (variance can go negative)
   and why QE is the standard choice.
5. Carr-Madan FFT method as an alternative to COS for specific regimes.
   When does it outperform COS?

Output: 2-3 page markdown document with equations in LaTeX, citations,
and a comparison table of which method is best in which regime.

Go/no-go: document exists and contains explicit formulas (not just
descriptions) for the Forde-Jacquier-Lee expansion and the QE scheme.

### 0B: Monte Carlo ground truth infrastructure

Build a standalone Heston Monte Carlo pricer at

`training data creation/v2/heston_mc.py` using the QE discretization
from Andersen 2008.

Requirements:

- Function signature: `price_european(kappa, theta, sigma, rho, v0, r, q, K, T, spot, n_paths, n_steps) -> price`
- Implement QE scheme faithfully including the psi_c threshold switching
  between quadratic-exponential and exponential sampling.
- Antithetic variates enabled by default for variance reduction.
- Support batched pricing: vectorize over K and T for a single parameter
  set to amortize path generation cost.
- Output includes both the price and the Monte Carlo standard error.

Validation before go/no-go:

- Price an ATM 1-year European call under Heston with known parameters
  (kappa=2, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04, r=0.03, q=0, K=100,
  spot=100). Compare to COS result at HALF_ABS=5.0, N_COS=512. Should
  agree to within 3 MC standard errors.
- Price a 30% OTM 1-week call with same parameters. Report price, MC SE,
  COS price. They should agree within 3 SE, or if not, document it for
  investigation in Phase 1.
- Price a 50% OTM 1-week call. MC price may be zero or near-zero. COS
  likely returns NaN or a numerically unstable value. Record both
  outputs.

Go/no-go: ATM 1Y agreement within 3 SE of COS, MC standard error on
50k paths for ATM 1Y below 0.1% of price.

Note: 10M path MC is for validation of the multi-method pricer in Phase
2, not for training data generation. 50k paths here is sanity-check
capacity. Do not attempt to generate training data with MC; it is far
too slow.

### 0C: Realistic parameter distribution audit

Before regenerating data, audit whether the current parameter sampling

covers realistic market regimes or wastes capacity on implausible Heston
parameter combinations.

Script: `training data creation/v2/diagnostics/audit_param_realism.py`

For the current `data/heston_pipeline.h5`:

1. Load all parameter sets.
2. For each, compute the Feller condition: 2 * kappa * theta >= sigma^2.
   Report the fraction of samples that violate Feller.
3. Compute the "long-run volatility" sqrt(theta) and report its
   distribution. Real markets have long-run vol between 10% and 30%
   typically; flag samples outside 5-50%.
4. Compute the variance-of-variance proxy sigma * sqrt(theta) /
   (2*kappa) and plot its distribution. This roughly measures how
   volatile volatility is relative to its mean; realistic values are
   0.5-2.0.
5. Cross-reference rho values. Real equity markets have rho between
   -0.9 and -0.3 typically; flag deviations.

Output: `training data creation/v2/docs/param_realism_report.md` with:

- Fraction of training parameters in "realistic" region (define
  realistic as Feller satisfied AND long-run vol in [0.10, 0.30] AND
  rho in [-0.9, -0.3]).
- Recommendation on whether parameter resampling is needed for Phase 3.

Go/no-go: report exists with a numeric answer to "what fraction of
training params are realistic." If the answer is below 40%, flag that
Phase 3 will need parameter resampling, not just multi-method pricing.

---

## Phase 1: Multi-method pricer (Day 2)

Single-threaded. Depends on 0A (need asymptotic formulas).

### 1A: Implement method dispatcher

File: `training data creation/v2/pricer.py`

Build a function
`price_cell(kappa, theta, sigma, rho, v0, r, q, K, T, spot) -> (price, method_flag, confidence)` that:

1. Attempts COS with HALF_ABS=5.0, N_COS=512 in float64.
2. If COS returns finite and price / spot > 1e-4, accept, method_flag =
   "cos_standard".
3. If COS returns finite but price / spot <= 1e-4 AND price / spot >
   1e-6, accept with method_flag = "cos_small_price",
   confidence = 0.7.
4. If COS returns NaN or inf, try COS with HALF_ABS=10.0, N_COS=2048.
   If accepts under step 2 or 3 criteria, method_flag =
   "cos_extended".
5. If still fails AND we are in the deep-OTM regime (|log(K/spot)| >
   0.3), try Forde-Jacquier-Lee asymptotic from Phase 0A. Accept if
   result is finite and positive, method_flag = "asymptotic_fjl",
   confidence = 0.5.
6. If all methods fail OR if the accepted price / spot < 1e-6, return
   (NaN, "unpricable", 0.0). This cell is genuinely below precision.

The confidence value is not used immediately but will be stored and may
be used in future training runs with confidence-weighted loss.

Write unit tests in `training data creation/v2/tests/test_pricer.py`:

- ATM 1Y should always return cos_standard.
- 50% OTM 1-week put should return either unpricable or
  asymptotic_fjl, never cos_standard.
- Consistency check: prices monotone decreasing in strike for calls,
  increasing for puts.

Go/no-go: all unit tests pass. On a grid of 100 random parameter sets,
the unpricable fraction is below 15% (if above 15%, the asymptotic
method is probably broken or the grid is genuinely too extreme).

### 1B: Benchmark the multi-method pricer vs COS

Script: `training data creation/v2/diagnostics/bench_pricer.py`

Compare old pricer (COS only, HALF_ABS=3.0) vs new pricer on 500 random
parameter sets over the full grid:

- Report fraction of cells each pricer marks as valid (non-NaN).
- For cells where both return finite values, compute mean and p99
  absolute difference in IV.
- Report per-grid-cell fraction of method_flag usage (what fraction of
  cells needed cos_standard vs cos_extended vs asymptotic vs
  unpricable, as a NK x NT heatmap).

Output: `training data creation/v2/docs/pricer_benchmark.md` with the
method usage heatmap saved as PNG.

Go/no-go: the new pricer returns valid (non-unpricable) values for at
least 80% of the full grid averaged across the 500 samples. If below
80%, escalate because the plan assumed the wings were fixable and that
may be wrong.

---

## Phase 2: Validate pricer vs MC (Day 3)

Single-threaded. Depends on 0B and Phase 1.

### 2A: High-precision MC validation

Script: `training data creation/v2/diagnostics/validate_with_mc.py`

Take 50 random parameter sets. For each, price the full 49 x 14 grid
with:

- The new multi-method pricer.
- 5 million-path MC using the QE scheme from 0B.

For each cell, compute the signed relative error:
`(pricer_price - mc_price) / max(mc_price, 1e-6)`

Report:

- Per-cell mean and std of relative errors (heatmap).
- Per-method mean and std of relative errors (cos_standard,
  cos_extended, asymptotic_fjl).
- List of cells where |relative error| > 5% despite the pricer marking
  them as valid (these are cases where the pricer has false confidence).

Output: `training data creation/v2/docs/mc_validation.md` with heatmaps
and failure case list.

Go/no-go:

- Cells marked cos_standard agree with MC to within 0.5% mean relative
  error, 2% p99.
- Cells marked cos_extended agree within 2% mean, 5% p99.
- Cells marked asymptotic_fjl agree within 5% mean, 10% p99.

If any method class fails its threshold, stop and report. Do not
regenerate training data with a method that does not meet accuracy
targets.

### 2B: Spot check against reference implementations

For 10 specific parameter sets taken from published Heston validation
tables (the "test basket" in Albrecher-Mayer-Schoutens-Tistaert 2007 if
available, otherwise standard benchmark cases from the COS paper itself),
compare the new pricer output to published values.

Output: table in `mc_validation.md` showing published price vs pricer
price vs MC price for each benchmark case.

Go/no-go: all benchmark cases within 0.1% of published values.

---

## Phase 3: Regenerate training data (Day 4, morning)

Single-threaded. Depends on Phase 2.

### 3A: Parameter sampling update (conditional on 0C)

If Phase 0C flagged that realistic parameter coverage is below 40%:
implement a realistic sampling policy at
`training data creation/v2/sampling.py` that:

- Samples kappa from log-uniform on [0.3, 10]
- Samples theta from log-uniform on [0.01, 0.16] (vol^2 from 10% to 40%)
- Samples sigma from [0.1, 1.5] with rejection on Feller
- Samples rho from truncated normal centered at -0.6 with std 0.3,
  clipped to [-0.95, -0.1]
- Samples v0 from log-uniform on [0.005, 0.20] (vol from 7% to 45%)

Otherwise skip this step and use existing sampling.

### 3B: Regenerate 10M training surfaces

Script: `training data creation/v2/generate_v2.py`

Using the multi-method pricer from Phase 1 and the sampling from 3A:

- Generate 10M training surfaces.
- Store IV surface (NaN for unpricable cells).
- Store a new `quality_mask` dataset of shape (N, NK, NT) with integer
  codes: 0=cos_standard, 1=cos_extended, 2=asymptotic_fjl,
  3=unpricable.
- Store `confidence` dataset of shape (N, NK, NT) float32.
- Keep the old `raw_cell_mask` for backward compatibility but it will
  no longer be the primary mask.

Output: `data/heston_pipeline_v2.h5`

Also regenerate val: `data/heston_pipeline_v2_val.h5` with 1M surfaces
using a different random seed.

Go/no-go:

- File exists, correct shape, no corrupted blocks.
- Unpricable fraction overall below 10% for the val file.
- Unpricable fraction in the wing corner (strike idx 0-5, mat idx 0-2)
  below 40% (down from 86% in the old pipeline).

If the unpricable fraction in the wing corner is still above 40%,
consider it a partial success and note that deepest wings cannot be
priced; the mask will honestly reflect this.

---

## Phase 4: Training with v2 data (Day 4, afternoon + Day 5)

Single-threaded. Depends on Phase 3.

### 4A: Minimal training code updates

File: `model/train.py` and `model/loss.py`.

Changes:

- Dataset loading: update BatesDataset to read `quality_mask` from v2
  files and treat code 3 (unpricable) as mask=False.
- Loss function: optionally use confidence values in weighting (already
  supported via `--confidence` flag, just verify it works with the new
  confidence schema).
- Best-checkpoint gating: lower from ep >= 100 to ep >= 3 as previously
  decided.

Do not change the loss function beyond these. Do not change the
architecture. Do not change the optimizer or scheduler.

### 4B: Train v2 baseline

Launch:

```
.conda/bin/python -m model.train \
  --h5 data/heston_pipeline_v2.h5 \
  --val-h5 data/heston_pipeline_v2_val.h5 \
  --data-loss ivrmse \
  --preload auto \
  --lr 3e-4 \
  --lambda-cal 0.01 --lambda-bfly 0.005 --lambda-ts 0.003 \
  --epochs 100 \
  --out-dir runs/v2_baseline
```

Run to completion or early stop. Record best val IVRMSE and best epoch.

### 4C: Evaluate v2 model downstream

Run the Run A pricing error check (audit_d2_surrogate_cos_price_error)
against the v2 model.

Compute additionally: the same pricing error computation against the v2
H5 (should be much better than old because labels are now accurate).

Output: `runs/v2_baseline/eval_report.md` comparing:

- Old pipeline best: val IVRMSE 129 bps, wing 1Y 9.3% mean pricing error
- V2 pipeline best: val IVRMSE ???, wing 1Y ??? mean pricing error
- V2 pipeline vs COS ground truth: wing 1Y ???
- V2 pipeline vs MC ground truth on 20 cases: wing 1Y ???

Go/no-go:

- If val IVRMSE below 50 bps AND wing 1Y pricing error below 3% mean:
  success, proceed to calibration testing.
- If val IVRMSE 50-100 bps: partial success, identify remaining
  bottleneck.
- If val IVRMSE above 100 bps despite clean data: architecture or
  optimization issue that was previously hidden by label noise. Escalate.

### 4D: Calibration sanity check

Run the existing Rust and Python calibration tests against the v2 model:

- `surrogate_kappa_sensitivity` test
- `kappa_theta_grid_recovery` test
- `synthetic_round_trip` test
- Python `diag_accuracy` calibration on 50 random samples

Output: table comparing recovery accuracy old vs v2.

Go/no-go: kappa recovery RMSE below 0.5 (from current 1.42). If not,
kappa identifiability has structural limits beyond label quality (likely
grid maturity range) and a separate plan is needed.

---

## Deliverables checklist

Phase 0:

- `training data creation/v2/docs/asymptotic_methods.md`
- `training data creation/v2/heston_mc.py`
- `training data creation/v2/docs/param_realism_report.md`

Phase 1:

- `training data creation/v2/pricer.py`
- `training data creation/v2/tests/test_pricer.py`
- `training data creation/v2/diagnostics/bench_pricer.py`
- `training data creation/v2/docs/pricer_benchmark.md`

Phase 2:

- `training data creation/v2/diagnostics/validate_with_mc.py`
- `training data creation/v2/docs/mc_validation.md`

Phase 3:

- `training data creation/v2/sampling.py` (if needed)
- `training data creation/v2/generate_v2.py`
- `data/heston_pipeline_v2.h5`
- `data/heston_pipeline_v2_val.h5`

Phase 4:

- `runs/v2_baseline/best.pt`
- `runs/v2_baseline/eval_report.md`
- Updated calibration test results

Final:

- Single summary document `SCENARIO_C_RESULTS.md` at repo root with
  before/after comparison table and honest statement of remaining
  limitations.

---

## Escalation triggers

Stop and report (do not continue) if:

- Phase 0B: MC and COS disagree by more than 10% on the ATM benchmark.
  Suggests an MC implementation bug.
- Phase 1B: unpricable fraction above 30% even with multi-method.
  Suggests the scenario C target is not achievable without Monte Carlo
  fallback in the pricer itself (which would make data generation too
  slow).
- Phase 2A: any method class fails its accuracy threshold against MC.
  Do not regenerate data with an inaccurate method.
- Phase 3B: wing corner unpricable fraction still above 40%. Acceptable
  but requires a conversation about honest scope before proceeding.
- Phase 4C: val IVRMSE above 100 bps despite clean data. Suggests
  architecture or optimization problems that were masked by label
  corruption.

---

## What this plan does not address

- Calibration-side improvements (loss functions, optimizer choice,
  regularization). Scoped out; calibration is only used as a diagnostic.
- Architecture changes (SSVI parameterization, larger models, different
  heads). Only if Phase 4 fails.
- Grid extension (longer maturities, wider strikes). Only if Phase 4
  shows identifiability constraints.
- PINN weight rebalancing. The train_log analysis showed calendar PINN
  contributes nothing at current weights; this can be addressed later
  if butterfly violations become the pricing bottleneck.

These are separate plans if they become necessary.

"""Generate all figures for the Scenario-C LaTeX report."""
from __future__ import annotations
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True, parents=True)
sys.path.insert(0, str(REPO))

from model.network import BatesSurrogate  # noqa: E402

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
})

V2_PARAM_LO = np.array([0.30, 0.01, 0.10, -0.90, 0.02], dtype=np.float64)
V2_PARAM_HI = np.array([10.0, 0.16, 1.50, -0.30, 0.20], dtype=np.float64)
V2_R_LO, V2_R_HI = 0.00, 0.05
PARAM_NAMES = [r"$\kappa$", r"$\theta$", r"$\sigma_v$", r"$\rho$", r"$v_0$", r"$r$"]


def normalise_np(params: np.ndarray) -> np.ndarray:
    heston = params[:, :5]
    r = params[:, 5:6]
    hn = (heston - V2_PARAM_LO) / (V2_PARAM_HI - V2_PARAM_LO + 1e-12)
    rn = (r - V2_R_LO) / (V2_R_HI - V2_R_LO + 1e-12)
    qn = np.zeros_like(rn)
    return np.concatenate([hn, rn, qn], axis=1).astype(np.float32)


def fig_grid_and_pricable():
    """Grid overview + pricable region."""
    with h5py.File(REPO / "data/heston_mc_reference.h5", "r") as f:
        pricable = f["pricable_region"][:].astype(bool)
    nk, nt = pricable.shape
    log_k = np.linspace(-0.80, 0.40, nk)
    maturities = np.array([1/52, 2/52, 3/52, 1/12, 2/12, 3/12, 4/12, 6/12, 9/12, 1.0, 1.25, 1.5, 2.0, 3.0])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    im = ax.pcolormesh(
        np.arange(nt + 1),
        np.arange(nk + 1),
        pricable.astype(float),
        cmap="Greens",
        vmin=0, vmax=1,
        edgecolors="none",
    )
    ax.set_xlabel("Maturity index (T in years shown below)")
    ax.set_ylabel("Log-moneyness index (k shown on right)")
    ax.set_title(f"Pricable region mask: {int(pricable.sum())}/{pricable.size} cells"
                 f" ({100*pricable.mean():.1f}%)")
    ax.set_xticks(np.arange(nt) + 0.5)
    ax.set_xticklabels([f"{t:.2f}" for t in maturities], rotation=45, ha="right", fontsize=7)
    sel = np.linspace(0, nk - 1, 7).astype(int)
    ax.set_yticks(sel + 0.5)
    ax.set_yticklabels([f"{log_k[i]:+.2f}" for i in sel], fontsize=7)
    ax.set_aspect("auto")
    fig.savefig(FIG / "pricable_region.pdf")
    plt.close(fig)
    print("wrote pricable_region.pdf")


def fig_sample_surface():
    """Example IV surface (pricer, MC, model) on one parameter set."""
    with h5py.File(REPO / "data/heston_mc_reference.h5", "r") as f:
        params = f["params"][:]
        mc_iv = f["mc_iv"][:].astype(np.float32)
        pr_iv = f["pricer_iv"][:].astype(np.float32)
        pricable = f["pricable_region"][:].astype(bool)

    # Pick a set with high pricable coverage and reasonable params
    valid_frac = np.array([
        (np.isfinite(mc_iv[i]) & np.isfinite(pr_iv[i]) & pricable).mean()
        for i in range(mc_iv.shape[0])
    ])
    idx = int(np.argmax(valid_frac))
    p = params[idx]
    nk, nt = pricable.shape
    log_k = np.linspace(-0.80, 0.40, nk)
    maturities = np.array([1/52, 2/52, 3/52, 1/12, 2/12, 3/12, 4/12, 6/12, 9/12, 1.0, 1.25, 1.5, 2.0, 3.0])

    model = BatesSurrogate.from_checkpoint(str(REPO / "runs/v2_baseline/best.pt"))
    model.eval()
    theta = torch.from_numpy(normalise_np(params[idx:idx + 1]))
    with torch.no_grad():
        iv_nn = model(theta).float().cpu().numpy().reshape(nk, nt)

    surfaces = {
        "Pricer (COS)": pr_iv[idx],
        "MC (QE)":      mc_iv[idx],
        "Surrogate":    iv_nn,
    }
    vmin = min(np.nanmin(s[pricable]) for s in surfaces.values())
    vmax = max(np.nanmax(s[pricable]) for s in surfaces.values())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (name, s) in zip(axes, surfaces.items()):
        im = ax.pcolormesh(
            np.arange(nt + 1), np.arange(nk + 1),
            np.where(pricable, s, np.nan),
            cmap="viridis", vmin=vmin, vmax=vmax, edgecolors="none",
        )
        ax.set_title(name)
        ax.set_xlabel("Maturity T")
        ax.set_xticks(np.arange(nt) + 0.5)
        ax.set_xticklabels([f"{t:.2f}" for t in maturities], rotation=45, ha="right", fontsize=6)
    axes[0].set_ylabel("Log-moneyness k")
    sel = np.linspace(0, nk - 1, 7).astype(int)
    axes[0].set_yticks(sel + 0.5)
    axes[0].set_yticklabels([f"{log_k[i]:+.2f}" for i in sel], fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.82, label="Implied volatility")
    fig.suptitle(
        rf"Example surface: $\kappa$={p[0]:.2f}, $\theta$={p[1]:.3f}, "
        rf"$\sigma_v$={p[2]:.2f}, $\rho$={p[3]:.2f}, $v_0$={p[4]:.3f}, $r$={p[5]:.3f}"
    )
    fig.savefig(FIG / "sample_surface.pdf")
    plt.close(fig)
    print("wrote sample_surface.pdf")


def fig_training_curves():
    df = pd.read_csv(REPO / "runs/v2_baseline/train_log.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    ax = axes[0]
    ax.plot(df["epoch"], df["val_ivrmse_pricable_bps"], label="val (pricable)", lw=1.4)
    ax.plot(df["epoch"], df["val_ivrmse_full_bps"], label="val (full grid)", lw=1.0, alpha=0.7)
    train_bps = np.sqrt(df["train_vega"]) * 1e4
    ax.plot(df["epoch"], train_bps, label="train (vega RMSE proxy)", lw=1.0, alpha=0.7, ls="--")
    ax.axhline(35.99, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("IV RMSE (bps)")
    ax.set_title("Training curves — v2 baseline")
    ax.set_yscale("log"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(df["epoch"], df["train_cal"], label=r"$L_{\mathrm{cal}}$", lw=1.0)
    ax.plot(df["epoch"], df["train_bfly"], label=r"$L_{\mathrm{bfly}}$", lw=1.0)
    ax.plot(df["epoch"], df["train_ts"], label=r"$L_{\mathrm{ts}}$", lw=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Raw penalty (unweighted)")
    ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title("Soft-constraint penalties")
    fig.savefig(FIG / "training_curves.pdf")
    plt.close(fig)
    print("wrote training_curves.pdf")


def fig_error_histograms():
    with h5py.File(REPO / "data/heston_mc_reference.h5", "r") as f:
        params = f["params"][:]
        mc_iv = f["mc_iv"][:].astype(np.float32)
        pr_iv = f["pricer_iv"][:].astype(np.float32)
        pricable = f["pricable_region"][:].astype(bool)

    model = BatesSurrogate.from_checkpoint(str(REPO / "runs/v2_baseline/best.pt")).cuda()
    model.eval()
    theta = torch.from_numpy(normalise_np(params)).cuda()
    with torch.no_grad():
        iv_pred = model(theta).float().cpu().numpy().reshape(mc_iv.shape)

    m = pricable[None, :, :] & np.isfinite(mc_iv) & np.isfinite(pr_iv) & np.isfinite(iv_pred)
    d_mm = (iv_pred - mc_iv)[m] * 1e4
    d_mp = (iv_pred - pr_iv)[m] * 1e4
    d_mc_p = (mc_iv - pr_iv)[m] * 1e4

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    bins = np.linspace(-300, 300, 121)
    ax = axes[0]
    for label, d, c in [("MC vs pricer", d_mc_p, "tab:green"),
                        ("Model vs pricer", d_mp, "tab:orange"),
                        ("Model vs MC", d_mm, "tab:blue")]:
        ax.hist(np.clip(d, bins[0], bins[-1]), bins=bins, histtype="step",
                label=f"{label} (RMSE {np.sqrt((d**2).mean()):.1f})", lw=1.4, color=c)
    ax.axvline(-50, color="k", ls=":", lw=0.6); ax.axvline(50, color="k", ls=":", lw=0.6)
    ax.set_xlabel("IV difference (bps)"); ax.set_ylabel("Cell count")
    ax.set_title("IV-difference distributions (pricable cells)")
    ax.legend(fontsize=8)

    ax = axes[1]
    a = np.abs(d_mm)
    qs = np.linspace(0, 100, 201)
    ax.plot(qs, np.percentile(a, qs), lw=1.4, label="|model − MC|")
    ax.plot(qs, np.percentile(np.abs(d_mp), qs), lw=1.0, alpha=0.7, label="|model − pricer|")
    ax.plot(qs, np.percentile(np.abs(d_mc_p), qs), lw=1.0, alpha=0.7, label="|MC − pricer|")
    ax.axhline(50, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Percentile"); ax.set_ylabel("|IV diff| (bps)")
    ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title(f"Tail curves — 50 bps gate holds on {100*(a < 50).mean():.2f}% of cells")
    fig.savefig(FIG / "error_distributions.pdf")
    plt.close(fig)
    print("wrote error_distributions.pdf")


def fig_calibration_recovery():
    """Parameter recovery scatter from calibration sanity."""
    with h5py.File(REPO / "data/heston_mc_reference.h5", "r") as f:
        params_true = f["params"][:].astype(np.float64)
        mc_iv = f["mc_iv"][:].astype(np.float32)
        pricable = f["pricable_region"][:].astype(bool)
    n, nk, nt = mc_iv.shape
    device = "cuda"
    model = BatesSurrogate.from_checkpoint(str(REPO / "runs/v2_baseline/best.pt")).to(device)
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)

    lo_full = np.concatenate([V2_PARAM_LO, [V2_R_LO]])
    hi_full = np.concatenate([V2_PARAM_HI, [V2_R_HI]])
    lo = torch.tensor(lo_full, device=device, dtype=torch.float32)
    hi = torch.tensor(hi_full, device=device, dtype=torch.float32)

    mc_iv_t = torch.from_numpy(mc_iv).to(device).reshape(n, nk * nt)
    mask = (torch.from_numpy(pricable).to(device).reshape(nk * nt).unsqueeze(0)
            & torch.isfinite(mc_iv_t))
    mc_iv_t = torch.where(mask, mc_iv_t, torch.zeros_like(mc_iv_t))
    n_valid = mask.sum(dim=1).clamp_min(1).float()

    torch.manual_seed(0)
    z = torch.zeros(n, 6, device=device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        u = torch.sigmoid(z)
        phys = lo + (hi - lo) * u
        q_col = torch.zeros(n, 1, device=device)
        theta_in = torch.cat([u, q_col], dim=1)
        iv_pred = model(theta_in).float()
        sq = (iv_pred - mc_iv_t) ** 2
        per = (sq * mask.float()).sum(dim=1) / n_valid
        per.mean().backward()
        opt.step()

    with torch.no_grad():
        phys = (lo + (hi - lo) * torch.sigmoid(z)).cpu().numpy().astype(np.float64)

    names_plain = [r"$\kappa$", r"$\theta$", r"$\sigma_v$", r"$\rho$", r"$v_0$", r"$r$"]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.8))
    for i, ax in enumerate(axes.flat):
        lo_i, hi_i = lo_full[i], hi_full[i]
        rng_e = phys[:, i] - params_true[:, i]
        rel95 = np.percentile(np.abs(rng_e), 95) / (hi_i - lo_i) * 100
        ax.scatter(params_true[:, i], phys[:, i], s=12, alpha=0.55)
        xs = np.linspace(lo_i, hi_i, 100)
        ax.plot(xs, xs, "k--", lw=0.7, label="identity")
        ax.set_xlim(lo_i, hi_i); ax.set_ylim(lo_i, hi_i)
        ax.set_xlabel(f"true {names_plain[i]}")
        ax.set_ylabel(f"recovered {names_plain[i]}")
        ax.set_title(f"{names_plain[i]}: p95 rel err {rel95:.1f}%", fontsize=10)
    fig.suptitle("Calibration sanity — parameter recovery on 200 MC surfaces")
    fig.savefig(FIG / "calibration_recovery.pdf")
    plt.close(fig)
    print("wrote calibration_recovery.pdf")


def fig_prior_samples():
    """Prior samples scatter matrix (subset)."""
    with h5py.File(REPO / "data/heston_mc_reference.h5", "r") as f:
        params = f["params"][:]
    cols = [(0, 2, r"$\kappa$", r"$\sigma_v$"),
            (0, 1, r"$\kappa$", r"$\theta$"),
            (3, 4, r"$\rho$",   r"$v_0$"),
            (1, 4, r"$\theta$", r"$v_0$")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.3))
    for ax, (i, j, xl, yl) in zip(axes, cols):
        ax.scatter(params[:, i], params[:, j], s=10, alpha=0.55)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    feller_k = np.linspace(V2_PARAM_LO[0], V2_PARAM_HI[0], 100)
    for theta_fix in (0.02, 0.04, 0.08, 0.16):
        axes[0].plot(feller_k, np.sqrt(2 * feller_k * theta_fix), ls=":", lw=0.7,
                     label=rf"Feller $\theta$={theta_fix}")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Prior samples (200 MC reference sets) — Scenario-C realistic prior")
    fig.savefig(FIG / "prior_samples.pdf")
    plt.close(fig)
    print("wrote prior_samples.pdf")


if __name__ == "__main__":
    fig_grid_and_pricable()
    fig_sample_surface()
    fig_training_curves()
    fig_error_histograms()
    fig_calibration_recovery()
    fig_prior_samples()
    print(f"all figures written to {FIG}")

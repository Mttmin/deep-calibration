"""
Heston/Bates Simulated Data Explorer & Alpha Vantage Comparator

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy.interpolate import griddata


DEFAULT_H5 = Path(__file__).resolve().parent.parent / "data" / "heston_train_val.h5"
API_KEYS_PATH = Path(__file__).resolve().parent.parent / "api_keys.json"

FALLBACK_LOG_MONEYNESS = np.array([-0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40])
FALLBACK_MATURITIES = np.array([1 / 12, 3 / 12, 6 / 12, 1.0, 1.5, 2.0])
FALLBACK_PARAM_KEYS = ["kappa", "theta", "sigma_v", "rho", "v0"]

PARAM_LABELS = {
    "kappa": "kappa (mean-rev)",
    "theta": "theta (long-var)",
    "sigma_v": "sigma_v (vol-of-vol)",
    "rho": "rho (corr)",
    "v0": "v0 (init-var)",
    "lambda_j": "lambda_j (jump intensity)",
    "mu_j": "mu_j (mean jump)",
    "sigma_j": "sigma_j (jump vol)",
    "r": "r (risk-free)",
    "q": "q (dividend)",
}


def _decode_list(value) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    for x in list(value):
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _pretty_label(key: str) -> str:
    return PARAM_LABELS.get(key, key)


def maturity_label(t: float) -> str:
    weeks = t * 52.0
    months = t * 12.0
    if weeks <= 2.5:
        return f"{int(round(weeks))}W"
    if t < 1.0:
        return f"{int(round(months))}M"
    if abs(t - round(t)) < 1e-9:
        return f"{int(round(t))}Y"
    return f"{t:g}Y"


def _load_api_key() -> str:
    if API_KEYS_PATH.exists():
        try:
            keys = json.loads(API_KEYS_PATH.read_text())
            return keys.get("Alpha_vantage", "")
        except (json.JSONDecodeError, KeyError):
            return ""
    return ""


st.set_page_config(
    page_title="IV Surface Data Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("IV Surface Data Explorer")
mode = st.sidebar.radio(
    "Mode",
    ["Simulated Data", "Real vs Simulated", "Parameter Space"],
    index=0,
)


@st.cache_resource
def load_h5(path: str):
    """Load HDF5 datasets and metadata, compatible with legacy and Bates files."""
    with h5py.File(path, "r", swmr=True) as f:
        params = f["params"][:]
        iv_surface = f["iv_surface"][:]
        cell_mask = f["cell_mask"][:]
        raw_cell_mask = f["raw_cell_mask"][:] if "raw_cell_mask" in f else None
        valid_count = f["valid_count"][:]
        market_params = f["market_params"][:] if "market_params" in f else None
        attrs = dict(f.attrs)

    log_m = np.array(attrs.get("log_moneyness", FALLBACK_LOG_MONEYNESS), dtype=np.float64)
    mats = np.array(attrs.get("maturities", FALLBACK_MATURITIES), dtype=np.float64)
    param_keys = _decode_list(attrs.get("param_names", FALLBACK_PARAM_KEYS))
    if not param_keys:
        param_keys = FALLBACK_PARAM_KEYS

    nk = int(attrs.get("NK", iv_surface.shape[1]))
    nt = int(attrs.get("NT", iv_surface.shape[2]))
    model_type = str(attrs.get("model_type", "heston"))

    meta = {
        "attrs": attrs,
        "log_moneyness": log_m,
        "maturities": mats,
        "maturity_labels": [maturity_label(float(t)) for t in mats],
        "param_keys": param_keys,
        "param_labels": [_pretty_label(k) for k in param_keys],
        "market_keys": ["r", "q"] if market_params is not None else [],
        "NK": nk,
        "NT": nt,
        "model_type": model_type,
        "atm_idx": int(np.argmin(np.abs(log_m))),
    }
    return params, market_params, iv_surface, cell_mask, raw_cell_mask, valid_count, meta


def fetch_alpha_vantage(symbol: str, api_key: str, date: str | None = None) -> dict | None:
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol,
        "apikey": api_key,
    }
    if date:
        params["date"] = date

    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "data" not in data:
            st.error(f"API error: {data.get('Note', data.get('Information', 'Unknown error'))}")
            return None
        return data
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def parse_av_chain(data: dict) -> tuple[pd.DataFrame, float]:
    rows = []
    for contract in data.get("data", []):
        try:
            rows.append(
                {
                    "expiration": contract["expiration"],
                    "strike": float(contract["strike"]),
                    "type": contract["type"],
                    "last": float(contract.get("last", 0) or 0),
                    "bid": float(contract.get("bid", 0) or 0),
                    "ask": float(contract.get("ask", 0) or 0),
                    "mark": float(contract.get("mark", 0) or 0),
                    "volume": int(contract.get("volume", 0) or 0),
                    "open_interest": int(contract.get("open_interest", 0) or 0),
                    "implied_volatility": float(contract.get("implied_volatility", 0) or 0),
                }
            )
        except (ValueError, KeyError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df, 0.0

    # Remove clearly bad IV quotes that dominate interpolation and distance.
    df = df[(df["implied_volatility"] > 0.01) & (df["implied_volatility"] < 5.0)].copy()
    if df.empty:
        return df, 0.0

    df["expiration"] = pd.to_datetime(df["expiration"])
    df["mid"] = np.where(
        (df["bid"] > 0) & (df["ask"] > 0),
        0.5 * (df["bid"] + df["ask"]),
        np.where(df["mark"] > 0, df["mark"], df["last"]),
    )

    underlying = 0.0
    if "meta_data" in data:
        underlying = float(data["meta_data"].get("underlying_price", 0) or 0)
    if underlying <= 0 and not df.empty and df["open_interest"].sum() > 0:
        underlying = float(df.loc[df["open_interest"].idxmax(), "strike"])

    return df, underlying


def build_real_iv_surface(
    df: pd.DataFrame,
    underlying: float,
    data_date: pd.Timestamp,
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    if df.empty or underlying <= 0:
        return np.full((len(log_moneyness), len(maturities)), np.nan)

    df = df[df["implied_volatility"] > 0].copy()
    df["log_moneyness"] = np.log(df["strike"] / underlying)
    df["days_to_exp"] = (df["expiration"] - data_date).dt.days
    df["maturity_years"] = df["days_to_exp"] / 365.25
    df = df[df["maturity_years"] > 0]
    if df.empty:
        return np.full((len(log_moneyness), len(maturities)), np.nan)

    points = df[["log_moneyness", "maturity_years"]].values
    values = df["implied_volatility"].values
    grid_m, grid_t = np.meshgrid(log_moneyness, maturities, indexing="ij")
    iv_linear = griddata(points, values, (grid_m, grid_t), method="linear")
    iv_nearest = griddata(points, values, (grid_m, grid_t), method="nearest")
    iv_grid = np.where(np.isfinite(iv_linear), iv_linear, iv_nearest)

    # Keep realistic annualized IV range to avoid outlier-driven nearest matches.
    iv_grid = np.clip(iv_grid, 0.01, 3.0)
    return iv_grid


def plot_iv_surface_3d(
    iv: np.ndarray,
    title: str,
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
    mask: np.ndarray | None = None,
) -> go.Figure:
    iv_plot = iv.copy()
    if mask is not None:
        iv_plot[~mask] = np.nan

    fig = go.Figure(
        data=[
            go.Surface(
                x=maturities,
                y=log_moneyness,
                z=iv_plot * 100,
                colorscale="Viridis",
                colorbar=dict(title="IV (%)"),
                hovertemplate=(
                    "Maturity: %{x:.3f}Y<br>"
                    "Log-moneyness: %{y:.2f}<br>"
                    "IV: %{z:.2f}%<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Maturity (years)",
            yaxis_title="Log-moneyness",
            zaxis_title="IV (%)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        height=550,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_iv_heatmap(
    iv: np.ndarray,
    title: str,
    log_moneyness: np.ndarray,
    maturity_labels: list[str],
    mask: np.ndarray | None = None,
) -> go.Figure:
    iv_plot = iv.copy()
    if mask is not None:
        iv_plot[~mask] = np.nan

    fig = go.Figure(
        data=go.Heatmap(
            x=maturity_labels,
            y=[f"{m:+.2f}" for m in log_moneyness],
            z=iv_plot * 100,
            colorscale="Viridis",
            colorbar=dict(title="IV (%)"),
            hovertemplate=(
                "Maturity: %{x}<br>"
                "Log-moneyness: %{y}<br>"
                "IV: %{z:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Maturity",
        yaxis_title="Log-moneyness",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_smile_overlay(
    iv: np.ndarray,
    title: str,
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
    maturity_labels: list[str],
    mask: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for ti, (_, label) in enumerate(zip(maturities, maturity_labels)):
        col = iv[:, ti]
        valid = np.isfinite(col) if mask is None else (mask[:, ti] & np.isfinite(col))
        if not valid.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=log_moneyness[valid],
                y=col[valid] * 100,
                mode="lines+markers",
                name=label,
                line=dict(color=colors[ti % len(colors)], width=2),
                marker=dict(size=5),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Log-moneyness",
        yaxis_title="IV (%)",
        height=420,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_term_structure(
    iv: np.ndarray,
    title: str,
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Dark2

    target_levels = np.array([-0.30, -0.10, 0.00, 0.10, 0.30])
    key_idx = np.unique([int(np.argmin(np.abs(log_moneyness - t))) for t in target_levels])

    for idx, si in enumerate(key_idx.tolist()):
        row = iv[si, :]
        valid = np.isfinite(row)
        if not valid.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=maturities[valid],
                y=row[valid] * 100,
                mode="lines+markers",
                name=f"m={log_moneyness[si]:+.2f}",
                line=dict(color=colors[idx % len(colors)], width=2),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Maturity (years)",
        yaxis_title="IV (%)",
        height=420,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _render_param_metrics(keys: list[str], values: np.ndarray) -> None:
    cols_per_row = 4
    for start in range(0, len(keys), cols_per_row):
        row_keys = keys[start : start + cols_per_row]
        row_vals = values[start : start + cols_per_row]
        cols = st.columns(len(row_keys))
        for col, key, val in zip(cols, row_keys, row_vals):
            col.metric(_pretty_label(key), f"{float(val):.4f}")


def page_simulated():
    st.header("Simulated IV Surfaces")

    h5_path = st.sidebar.text_input("HDF5 path", value=str(DEFAULT_H5))
    if not Path(h5_path).exists():
        st.warning(f"File not found: {h5_path}")
        return

    params, market_params, iv_surface, cell_mask, raw_cell_mask, valid_count, meta = load_h5(h5_path)
    attrs = meta["attrs"]
    log_m = meta["log_moneyness"]
    mats = meta["maturities"]
    mat_labels = meta["maturity_labels"]
    param_keys = meta["param_keys"]
    n_cells = int(meta["NK"] * meta["NT"])
    N = len(params)

    fill_rate = attrs.get("cell_fill_rate_pct", np.nanmean(cell_mask) * 100)
    st.sidebar.markdown(f"**Model:** {meta['model_type']}")
    st.sidebar.markdown(f"**NaN policy:** {attrs.get('nan_policy', 'mask')}")
    st.sidebar.markdown(f"**Samples:** {N:,}")
    st.sidebar.markdown(f"**Cell fill rate:** {float(fill_rate):.1f}%")
    st.sidebar.markdown(f"**Grid:** {meta['NK']} strikes x {meta['NT']} maturities")

    if raw_cell_mask is not None:
        show_raw_mask = st.sidebar.checkbox(
            "Show raw invertible mask overlay",
            value=True,
            help="Use raw_cell_mask (pre-fill) for plotting and valid-cell counts",
        )
    else:
        show_raw_mask = False

    st.sidebar.markdown("---")
    sample_mode = st.sidebar.radio("Sample selection", ["Random", "By index", "By parameters"])

    if sample_mode == "Random":
        if st.sidebar.button("New random sample"):
            st.session_state["sim_idx"] = int(np.random.randint(0, N))
        idx = st.session_state.get("sim_idx", 0)
    elif sample_mode == "By index":
        idx = st.sidebar.number_input("Sample index", 0, N - 1, 0)
    else:
        st.sidebar.markdown("**Filter parameters:**")
        param_lo = np.array(attrs.get("param_lo", np.min(params, axis=0)), dtype=np.float64)
        param_hi = np.array(attrs.get("param_hi", np.max(params, axis=0)), dtype=np.float64)

        filters = []
        for i, (key, lo, hi) in enumerate(zip(param_keys, param_lo, param_hi)):
            r = st.sidebar.slider(
                _pretty_label(key),
                float(lo),
                float(hi),
                (float(lo), float(hi)),
                key=f"pf_{i}",
            )
            filters.append((params[:, i] >= r[0]) & (params[:, i] <= r[1]))

        mask_filter = np.all(filters, axis=0)
        matching = np.where(mask_filter)[0]
        st.sidebar.markdown(f"**Matching:** {len(matching):,} / {N:,}")
        if len(matching) == 0:
            st.warning("No samples match the selected parameter ranges.")
            return
        idx = int(matching[0])
        if st.sidebar.button("Random from filtered"):
            idx = int(np.random.choice(matching))
            st.session_state["sim_idx"] = idx

    p = params[idx]
    rq = market_params[idx] if market_params is not None else None
    iv = iv_surface[idx]
    mask = raw_cell_mask[idx] if (show_raw_mask and raw_cell_mask is not None) else cell_mask[idx]
    vc = int(valid_count[idx])

    if show_raw_mask and raw_cell_mask is not None:
        vc_raw = int(raw_cell_mask[idx].sum())
        st.caption(f"Raw invertible cells: {vc_raw}/{n_cells}")

    st.subheader("Model Parameters")
    _render_param_metrics(param_keys, p)
    if rq is not None:
        st.subheader("Market Parameters")
        _render_param_metrics(meta["market_keys"], rq)

    st.caption(f"Sample #{idx:,}  |  Valid cells: {vc}/{n_cells}")

    tab_3d, tab_heat, tab_smile, tab_term = st.tabs(
        ["3D Surface", "Heatmap", "Smile Curves", "Term Structure"]
    )

    with tab_3d:
        st.plotly_chart(
            plot_iv_surface_3d(iv, f"IV Surface - Sample #{idx}", log_m, mats, mask),
            use_container_width=True,
        )
    with tab_heat:
        st.plotly_chart(
            plot_iv_heatmap(iv, f"IV Heatmap - Sample #{idx}", log_m, mat_labels, mask),
            use_container_width=True,
        )
    with tab_smile:
        st.plotly_chart(
            plot_smile_overlay(iv, f"Volatility Smiles - Sample #{idx}", log_m, mats, mat_labels, mask),
            use_container_width=True,
        )
    with tab_term:
        st.plotly_chart(
            plot_term_structure(iv, f"Term Structure - Sample #{idx}", log_m, mats),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("Batch Statistics")

    n_batch = min(5000, N)
    batch_idx = np.random.default_rng(0).choice(N, n_batch, replace=False)
    batch_iv = iv_surface[batch_idx]

    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure()
        for ti, label in enumerate(mat_labels):
            vals = batch_iv[:, :, ti].flatten()
            vals = vals[np.isfinite(vals)]
            fig.add_trace(
                go.Violin(
                    y=vals * 100,
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        fig.update_layout(
            title="IV Distribution by Maturity",
            yaxis_title="IV (%)",
            showlegend=False,
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        atm_idx = int(meta["atm_idx"])
        atm_ivs = batch_iv[:, atm_idx, :]

        fig = go.Figure()
        for pct, opacity in [(5, 0.1), (25, 0.2)]:
            lo = np.nanpercentile(atm_ivs * 100, pct, axis=0)
            hi = np.nanpercentile(atm_ivs * 100, 100 - pct, axis=0)
            fig.add_trace(go.Scatter(x=mats, y=hi, mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(
                go.Scatter(
                    x=mats,
                    y=lo,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba(99,110,250,{opacity})",
                    name=f"{pct}th-{100-pct}th pctl",
                )
            )

        median = np.nanmedian(atm_ivs * 100, axis=0)
        fig.add_trace(
            go.Scatter(
                x=mats,
                y=median,
                mode="lines+markers",
                name="Median",
                line=dict(color="rgb(99,110,250)", width=2),
            )
        )
        fig.update_layout(
            title="ATM IV Term Structure (distribution across samples)",
            xaxis_title="Maturity (years)",
            yaxis_title="ATM IV (%)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)


def page_comparison():
    st.header("Real vs Simulated Comparison")

    st.sidebar.markdown("### Alpha Vantage")
    api_key = st.sidebar.text_input("API Key", value=_load_api_key(), type="password")
    symbol = st.sidebar.text_input("Symbol", value="SPY")
    av_date = st.sidebar.text_input("Date (YYYY-MM-DD)", value="", help="Leave empty for latest")

    h5_path = st.sidebar.text_input("HDF5 path", value=str(DEFAULT_H5), key="h5_comp")

    if not api_key:
        st.info(
            "Enter your Alpha Vantage API key in the sidebar to fetch real options data. "
            "Get a free key at https://www.alphavantage.co/support/#api-key."
        )
        return
    if not Path(h5_path).exists():
        st.warning(f"Simulated data not found: {h5_path}")
        return

    with st.spinner(f"Fetching {symbol} options from Alpha Vantage..."):
        raw = fetch_alpha_vantage(symbol, api_key, av_date or None)
    if raw is None:
        return

    df, underlying = parse_av_chain(raw)
    if df.empty:
        st.error("No option contracts returned.")
        return

    st.success(f"Fetched {len(df):,} contracts for {symbol} (underlying ~ ${underlying:.2f})")

    data_date_str = av_date or raw.get("meta_data", {}).get("date", "")
    data_date = pd.Timestamp(data_date_str) if data_date_str else pd.Timestamp.now()

    params, market_params, iv_surface, cell_mask, raw_cell_mask, _, meta = load_h5(h5_path)
    log_m = meta["log_moneyness"]
    mats = meta["maturities"]
    mat_labels = meta["maturity_labels"]

    if raw_cell_mask is not None:
        show_raw_mask = st.sidebar.checkbox(
            "Use raw invertible mask for matching",
            value=True,
            key="comp_use_raw_mask",
            help="Restrict distance metric to cells that were invertible before filling",
        )
    else:
        show_raw_mask = False

    real_iv = build_real_iv_surface(df, underlying, data_date, log_m, mats)
    real_flat = real_iv.flatten()
    real_valid = np.isfinite(real_flat)
    if real_valid.sum() < 5:
        st.warning("Too few valid IV points in the real data for meaningful comparison.")
        return

    # Weighted nearest-match metric:
    # - overweight short maturities (where equity skew is steepest)
    # - overweight put wing (log-moneyness < 0)
    grid_m, grid_t = np.meshgrid(log_m, mats, indexing="ij")
    short_w = 1.0 / np.sqrt(np.maximum(grid_t, 1.0 / 365.25))
    put_w = np.where(grid_m < 0.0, 1.8, 1.0)
    ultra_short_boost = np.where(grid_t <= (2.0 / 52.0), 1.5, 1.0)
    weights = (short_w * put_w * ultra_short_boost).astype(np.float64)
    w_flat = weights.flatten()

    sim_flat = iv_surface.reshape(len(iv_surface), -1)
    if show_raw_mask and raw_cell_mask is not None:
        sim_valid = raw_cell_mask.reshape(len(raw_cell_mask), -1)
    else:
        sim_valid = np.isfinite(sim_flat)
    active = sim_valid & real_valid[None, :]
    sqerr = (sim_flat - real_flat[None, :]) ** 2
    weighted_num = np.where(active, sqerr * w_flat[None, :], 0.0).sum(axis=1)
    weighted_den = np.where(active, w_flat[None, :], 0.0).sum(axis=1)

    total_real_weight = float(w_flat[real_valid].sum())
    coverage = weighted_den / max(total_real_weight, 1e-12)
    base_rmse = np.sqrt(weighted_num / np.clip(weighted_den, 1e-12, None))

    # Penalize matches that only overlap a tiny subset of the real surface.
    distances = base_rmse * (1.0 + 0.5 * np.maximum(0.0, 0.6 - coverage))
    distances[coverage < 0.20] = np.inf
    best_idx = int(np.nanargmin(distances))

    best_iv = iv_surface[best_idx]
    best_mask = raw_cell_mask[best_idx] if (show_raw_mask and raw_cell_mask is not None) else cell_mask[best_idx]
    best_params = params[best_idx]
    best_market = market_params[best_idx] if market_params is not None else None
    best_dist = float(distances[best_idx])
    best_cov = float(coverage[best_idx])

    st.markdown(
        f"**Closest simulated sample:** #{best_idx:,} "
        f"(weighted RMSE: {best_dist:.4f}, weighted coverage: {best_cov * 100:.1f}%)"
    )

    st.markdown("**Estimated model parameters:**")
    _render_param_metrics(meta["param_keys"], best_params)
    if best_market is not None:
        st.markdown("**Estimated market parameters:**")
        _render_param_metrics(meta["market_keys"], best_market)

    tab_smile, tab_3d, tab_diff, tab_raw = st.tabs(
        ["Smile Comparison", "3D Surfaces", "Difference", "Raw Market Data"]
    )

    with tab_smile:
        n_t = len(mat_labels)
        n_cols = min(4, n_t)
        n_rows = int(math.ceil(n_t / n_cols))
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=mat_labels,
            vertical_spacing=0.12,
            horizontal_spacing=0.06,
        )

        for ti, label in enumerate(mat_labels):
            row = ti // n_cols + 1
            col = ti % n_cols + 1

            real_col = real_iv[:, ti]
            valid_r = np.isfinite(real_col)
            if valid_r.any():
                fig.add_trace(
                    go.Scatter(
                        x=log_m[valid_r],
                        y=real_col[valid_r] * 100,
                        mode="lines+markers",
                        name=f"Real ({label})",
                        line=dict(color="rgb(239,85,59)", width=2),
                        marker=dict(size=6),
                        legendgroup="real",
                        showlegend=(ti == 0),
                    ),
                    row=row,
                    col=col,
                )

            sim_col = best_iv[:, ti]
            valid_s = best_mask[:, ti] & np.isfinite(sim_col)
            if valid_s.any():
                fig.add_trace(
                    go.Scatter(
                        x=log_m[valid_s],
                        y=sim_col[valid_s] * 100,
                        mode="lines+markers",
                        name=f"Sim ({label})",
                        line=dict(color="rgb(99,110,250)", width=2, dash="dash"),
                        marker=dict(size=6, symbol="diamond"),
                        legendgroup="sim",
                        showlegend=(ti == 0),
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            title=f"Volatility Smile: {symbol} Real vs Closest Simulated",
            height=max(420, 280 * n_rows),
            legend=dict(orientation="h", y=-0.05, itemsizing="constant"),
        )
        for i in range(1, n_t + 1):
            row = (i - 1) // n_cols + 1
            col = (i - 1) % n_cols + 1
            fig.update_xaxes(title_text="Log-moneyness" if row == n_rows else "", row=row, col=col)
            fig.update_yaxes(title_text="IV (%)" if col == 1 else "", row=row, col=col)

        st.plotly_chart(fig, use_container_width=True)

    with tab_3d:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                plot_iv_surface_3d(real_iv, f"Real: {symbol}", log_m, mats),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                plot_iv_surface_3d(best_iv, "Simulated (closest match)", log_m, mats, best_mask),
                use_container_width=True,
            )

    with tab_diff:
        diff_surface = real_iv - best_iv
        fig = go.Figure(
            data=go.Heatmap(
                x=mat_labels,
                y=[f"{m:+.2f}" for m in log_m],
                z=diff_surface * 100,
                colorscale="RdBu_r",
                zmid=0,
                colorbar=dict(title="Delta IV (pp)"),
                hovertemplate=(
                    "Maturity: %{x}<br>"
                    "Log-moneyness: %{y}<br>"
                    "Delta IV: %{z:+.2f} pp<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title="Difference (Real - Simulated) in percentage points",
            xaxis_title="Maturity",
            yaxis_title="Log-moneyness",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        valid_diff = diff_surface[np.isfinite(diff_surface)]
        if len(valid_diff) > 0:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Delta", f"{np.mean(valid_diff) * 100:+.2f} pp")
            c2.metric("RMSE", f"{np.sqrt(np.mean(valid_diff**2)) * 100:.2f} pp")
            c3.metric("Max |Delta|", f"{np.max(np.abs(valid_diff)) * 100:.2f} pp")
            c4.metric("Coverage", f"{np.isfinite(diff_surface).sum()}/{diff_surface.size} cells")

    with tab_raw:
        st.subheader(f"Raw Options Chain: {symbol}")

        df_plot = df[df["implied_volatility"] > 0].copy()
        df_plot["log_moneyness"] = np.log(df_plot["strike"] / underlying)
        df_plot["days_to_exp"] = (df_plot["expiration"] - data_date).dt.days

        expiries = sorted(df_plot["expiration"].unique())
        selected_expiries = expiries[:6] if len(expiries) > 6 else expiries

        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, exp in enumerate(selected_expiries):
            sub = df_plot[df_plot["expiration"] == exp].sort_values("strike")
            dte = int((exp - data_date).days)
            fig.add_trace(
                go.Scatter(
                    x=sub["log_moneyness"],
                    y=sub["implied_volatility"] * 100,
                    mode="markers",
                    name=f"{pd.Timestamp(exp).strftime('%Y-%m-%d')} ({dte}d)",
                    marker=dict(size=4, color=colors[i % len(colors)], opacity=0.7),
                )
            )

        fig.update_layout(
            title=f"Raw Market IV - {symbol}",
            xaxis_title="Log-moneyness (ln K/S)",
            yaxis_title="IV (%)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df[
                [
                    "expiration",
                    "strike",
                    "type",
                    "bid",
                    "ask",
                    "mid",
                    "volume",
                    "open_interest",
                    "implied_volatility",
                ]
            ]
            .sort_values(["expiration", "strike"])
            .head(500),
            use_container_width=True,
        )


def page_param_space():
    st.header("Parameter Space Coverage")

    h5_path = st.sidebar.text_input("HDF5 path", value=str(DEFAULT_H5), key="h5_param")
    if not Path(h5_path).exists():
        st.warning(f"File not found: {h5_path}")
        return

    params, _, iv_surface, cell_mask, raw_cell_mask, valid_count, meta = load_h5(h5_path)
    log_m = meta["log_moneyness"]
    mat_labels = meta["maturity_labels"]
    param_keys = meta["param_keys"]
    param_labels = meta["param_labels"]
    N = len(params)

    n_plot = min(10_000, N)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, n_plot, replace=False)
    p_sub = params[idx]
    vc_sub = valid_count[idx]

    df_params = pd.DataFrame(p_sub, columns=param_keys)
    df_params["valid_cells"] = vc_sub

    have_feller = all(k in param_keys for k in ["kappa", "theta", "sigma_v"])
    if have_feller:
        ik = param_keys.index("kappa")
        it = param_keys.index("theta")
        isv = param_keys.index("sigma_v")
        feller_ratio = (2 * p_sub[:, ik] * p_sub[:, it]) / (p_sub[:, isv] ** 2)
        df_params["feller_ratio"] = feller_ratio

    st.subheader("Parameter Distributions")

    n_p = len(param_keys)
    n_cols = min(4, n_p)
    n_rows = int(math.ceil(n_p / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=param_labels)
    for i in range(n_p):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.add_trace(
            go.Histogram(
                x=p_sub[:, i],
                nbinsx=60,
                marker_color="rgb(99,110,250)",
                opacity=0.7,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(height=max(280, 220 * n_rows), margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if all(k in param_keys for k in ["rho", "sigma_v"]):
            fig = px.scatter(
                df_params,
                x="rho",
                y="sigma_v",
                color="valid_cells",
                color_continuous_scale="Viridis",
                opacity=0.4,
                title="rho vs sigma_v (colored by valid cells)",
            )
            fig.update_layout(height=400)
            fig.update_traces(marker_size=3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("rho/sigma_v axes are not available in this dataset.")

    with col2:
        if have_feller:
            fig = px.scatter(
                df_params,
                x="kappa",
                y="theta",
                color="feller_ratio",
                color_continuous_scale="RdYlGn",
                opacity=0.4,
                title="kappa vs theta (colored by Feller ratio)",
            )
            fig.update_layout(height=400)
            fig.update_traces(marker_size=3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feller diagnostics are unavailable for this dataset schema.")

    st.subheader("Data Quality")
    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure(data=go.Histogram(x=valid_count, nbinsx=55, marker_color="rgb(0,204,150)"))
        fig.update_layout(
            title=f"Distribution of Valid Cells per Sample (out of {meta['NK'] * meta['NT']})",
            xaxis_title="Valid cells",
            yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        use_mask = raw_cell_mask if raw_cell_mask is not None else cell_mask
        nan_rate = 1.0 - use_mask.mean(axis=0)
        fig = go.Figure(
            data=go.Heatmap(
                x=mat_labels,
                y=[f"{m:+.2f}" for m in log_m],
                z=nan_rate * 100,
                colorscale="Reds",
                colorbar=dict(title="NaN %"),
            )
        )
        fig.update_layout(
            title="NaN Rate by Grid Cell (%)",
            xaxis_title="Maturity",
            yaxis_title="Log-moneyness",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("IV Characteristics vs Parameters")
    atm_idx = int(meta["atm_idx"])
    batch_iv = iv_surface[idx]
    atm_iv_short = batch_iv[:, atm_idx, 0]

    left_idx = int(np.argmin(np.abs(log_m - (-0.10))))
    right_idx = int(np.argmin(np.abs(log_m - 0.10)))
    skew_short = batch_iv[:, left_idx, 0] - batch_iv[:, right_idx, 0]

    df_corr = pd.DataFrame({"ATM IV (short)": atm_iv_short * 100, "Skew (short)": skew_short * 100})
    for key in ["rho", "sigma_v", "v0"]:
        if key in param_keys:
            df_corr[key] = p_sub[:, param_keys.index(key)]
    df_corr = df_corr.dropna()

    c1, c2 = st.columns(2)
    with c1:
        if "rho" in df_corr.columns:
            fig = px.scatter(
                df_corr,
                x="rho",
                y="Skew (short)",
                opacity=0.3,
                title="rho vs smile skew",
                trendline="ols",
            )
            fig.update_traces(marker_size=3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("rho is unavailable in this dataset.")

    with c2:
        if "v0" in df_corr.columns:
            fig = px.scatter(
                df_corr,
                x="v0",
                y="ATM IV (short)",
                opacity=0.3,
                title="v0 vs short-tenor ATM IV",
                trendline="ols",
            )
            fig.update_traces(marker_size=3)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("v0 is unavailable in this dataset.")


if mode == "Simulated Data":
    page_simulated()
elif mode == "Real vs Simulated":
    page_comparison()
elif mode == "Parameter Space":
    page_param_space()
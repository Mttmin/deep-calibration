"""
Build a real-options IV surface dataset from Alpha Vantage.

Examples
--------
# Latest chain per ticker (no explicit date)
python real_surface_dataset.py --tickers SPY,QQQ,IWM --out ../data/real_surfaces_latest.h5

# Daily snapshots over a date range
python real_surface_dataset.py --tickers-file ../data/tickers.txt \
  --start-date 2025-01-01 --end-date 2025-03-31 --step-days 7 \
  --out ../data/real_surfaces_q1_2025.h5
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import griddata
from tqdm import tqdm


GRID_PRESETS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "base25x10": (
        np.array([
            -0.50, -0.40, -0.30, -0.25, -0.20, -0.15, -0.12, -0.10, -0.08, -0.06, -0.04,
            -0.02, 0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
            0.40, 0.50,
        ], dtype=np.float64),
        np.array([1 / 52, 2 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0],
                 dtype=np.float64),
    ),
    "high41x14": (
        np.linspace(-0.80, 0.40, 49, dtype=np.float64),
        np.array([
            1 / 52, 2 / 52, 3 / 52,
            1 / 12, 2 / 12, 3 / 12, 4 / 12, 6 / 12, 9 / 12,
            1.0, 1.25, 1.5, 2.0, 3.0,
        ], dtype=np.float64),
    ),
}
DEFAULT_GRID_PRESET = "high41x14"
API_KEYS_PATH = Path(__file__).resolve().parent.parent / "api_keys.json"
DEFAULT_BIG_TICKERS = (
    "SPY,QQQ,IWM,DIA,XLF,XLE,XLK,XLY,XLI,XLV,XLP,XLU,XLB,XLRE,"
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,BRK.B,JPM,V,MA,UNH,XOM,"
    "JNJ,PG,HD,COST,ABBV,BAC,WMT,AVGO,CVX,KO,MRK,PEP,ADBE,NFLX,CRM"
)


@dataclass
class SurfaceRecord:
    ticker: str
    snapshot_date: str
    api_date: str
    underlying: float
    n_contracts: int
    n_points_used: int
    raw_coverage_pct: float
    filled_coverage_pct: float


def load_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key
    if API_KEYS_PATH.exists():
        data = json.loads(API_KEYS_PATH.read_text())
        key = data.get("Alpha_vantage", "")
        if key:
            return key
    raise ValueError("No Alpha Vantage API key found. Pass --api-key or set api_keys.json.")


def parse_tickers(tickers: str | None, tickers_file: str | None) -> list[str]:
    out: list[str] = []
    if tickers:
        out.extend([x.strip().upper() for x in tickers.split(",") if x.strip()])
    if tickers_file:
        text = Path(tickers_file).read_text()
        for line in text.splitlines():
            line = line.strip().upper()
            if line and not line.startswith("#"):
                out.append(line)
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError("No tickers provided. Use --tickers or --tickers-file.")
    return uniq


def build_dates(start_date: str | None, end_date: str | None, step_days: int) -> list[str | None]:
    if not start_date and not end_date:
        return [None]
    if not start_date or not end_date:
        raise ValueError("Provide both --start-date and --end-date, or neither.")

    d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
    d1 = datetime.strptime(end_date, "%Y-%m-%d").date()
    if d1 < d0:
        raise ValueError("--end-date must be >= --start-date")

    dates: list[str | None] = []
    cur = d0
    while cur <= d1:
        dates.append(cur.isoformat())
        cur += timedelta(days=step_days)
    return dates


def fetch_alpha_vantage(symbol: str, api_key: str, snapshot_date: str | None, timeout_s: int) -> dict | None:
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol,
        "apikey": api_key,
    }
    if snapshot_date:
        params["date"] = snapshot_date

    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    if "data" not in payload:
        return None
    return payload


def parse_chain(payload: dict) -> tuple[pd.DataFrame, float, str]:
    rows = []
    for c in payload.get("data", []):
        try:
            bid = float(c.get("bid", 0) or 0)
            ask = float(c.get("ask", 0) or 0)
            mark = float(c.get("mark", 0) or 0)
            last = float(c.get("last", 0) or 0)
            mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0) else (mark if mark > 0 else last)
            rows.append({
                "expiration": c["expiration"],
                "strike": float(c["strike"]),
                "type": str(c.get("type", "")).lower(),
                "bid": bid,
                "ask": ask,
                "mid": float(mid),
                "volume": int(c.get("volume", 0) or 0),
                "open_interest": int(c.get("open_interest", 0) or 0),
                "implied_volatility": float(c.get("implied_volatility", 0) or 0),
            })
        except (KeyError, ValueError, TypeError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df, 0.0, ""

    df = df[(df["implied_volatility"] > 0.01) & (df["implied_volatility"] < 5.0)].copy()
    if df.empty:
        return df, 0.0, ""

    df["expiration"] = pd.to_datetime(df["expiration"])

    meta = payload.get("meta_data", {})
    underlying = float(meta.get("underlying_price", 0) or 0)
    api_date = str(meta.get("date", ""))

    if underlying <= 0 and df["open_interest"].sum() > 0:
        underlying = float(df.loc[df["open_interest"].idxmax(), "strike"])

    return df, underlying, api_date


def build_surface(
    df: pd.DataFrame,
    underlying: float,
    asof_date: pd.Timestamp,
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Returns (iv_filled, cell_mask, raw_cell_mask, points_used).
    """
    if df.empty or underlying <= 0:
        shape = (len(log_moneyness), len(maturities))
        return np.full(shape, np.nan), np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool), 0

    work = df.copy()
    work["log_moneyness"] = np.log(work["strike"] / underlying)
    work["days_to_exp"] = (work["expiration"] - asof_date).dt.days
    work["maturity_years"] = work["days_to_exp"] / 365.25
    work = work[(work["maturity_years"] > 0) & np.isfinite(work["log_moneyness"])].copy()

    if work.empty:
        shape = (len(log_moneyness), len(maturities))
        return np.full(shape, np.nan), np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool), 0

    # Confidence proxy: tighter spreads + more liquidity => higher weight.
    spread = np.maximum(work["ask"].to_numpy() - work["bid"].to_numpy(), 0.0)
    rel_spread = spread / np.maximum(work["mid"].to_numpy(), 1e-6)
    liq = np.log1p(work["open_interest"].to_numpy() + work["volume"].to_numpy())
    liq_norm = liq / max(np.quantile(liq, 0.95), 1e-6)
    conf = np.exp(-2.0 * np.clip(rel_spread, 0.0, 5.0)) * np.clip(liq_norm, 0.0, 1.0)
    conf = np.where(np.isfinite(conf), conf, 0.0)

    points = work[["log_moneyness", "maturity_years"]].to_numpy()
    iv_values = work["implied_volatility"].to_numpy()

    grid_m, grid_t = np.meshgrid(log_moneyness, maturities, indexing="ij")

    iv_linear = griddata(points, iv_values, (grid_m, grid_t), method="linear")
    iv_nearest = griddata(points, iv_values, (grid_m, grid_t), method="nearest")
    iv_filled = np.where(np.isfinite(iv_linear), iv_linear, iv_nearest)
    iv_filled = np.clip(iv_filled, 0.01, 3.0)

    raw_mask = np.isfinite(iv_linear)
    cell_mask = np.isfinite(iv_filled)

    # If nearest failed too (rare), fill with median observed IV.
    if not np.all(cell_mask):
        med_iv = float(np.median(iv_values)) if len(iv_values) else 0.20
        iv_filled = np.where(cell_mask, iv_filled, med_iv)
        cell_mask = np.isfinite(iv_filled)

    # Convert confidence samples into a grid so downstream can weight losses.
    conf_linear = griddata(points, conf, (grid_m, grid_t), method="linear")
    conf_nearest = griddata(points, conf, (grid_m, grid_t), method="nearest")
    conf_grid = np.where(np.isfinite(conf_linear), conf_linear, conf_nearest)
    conf_grid = np.clip(np.where(np.isfinite(conf_grid), conf_grid, 0.0), 0.0, 1.0)

    # Save confidence into a module-level variable return style by attaching attr.
    build_surface.last_confidence = conf_grid  # type: ignore[attr-defined]
    return iv_filled.astype(np.float64), cell_mask.astype(bool), raw_mask.astype(bool), len(work)


def save_dataset(
    out_path: Path,
    records: list[SurfaceRecord],
    iv_surfaces: list[np.ndarray],
    cell_masks: list[np.ndarray],
    raw_masks: list[np.ndarray],
    confidences: list[np.ndarray],
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
    grid_preset: str,
    source: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(records)
    nk = len(log_moneyness)
    nt = len(maturities)

    ticker_dtype = h5py.string_dtype(encoding="utf-8")
    date_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "w") as f:
        f.attrs["source"] = source
        f.attrs["grid_preset"] = grid_preset
        f.attrs["log_moneyness"] = log_moneyness.tolist()
        f.attrs["maturities"] = maturities.tolist()
        f.attrs["NK"] = nk
        f.attrs["NT"] = nt
        f.attrs["N_total"] = n

        f.create_dataset("ticker", data=np.array([r.ticker for r in records], dtype=object), dtype=ticker_dtype)
        f.create_dataset("snapshot_date", data=np.array([r.snapshot_date for r in records], dtype=object), dtype=date_dtype)
        f.create_dataset("api_date", data=np.array([r.api_date for r in records], dtype=object), dtype=date_dtype)
        f.create_dataset("underlying", data=np.array([r.underlying for r in records], dtype=np.float64))
        f.create_dataset("n_contracts", data=np.array([r.n_contracts for r in records], dtype=np.int32))
        f.create_dataset("n_points_used", data=np.array([r.n_points_used for r in records], dtype=np.int32))

        f.create_dataset(
            "iv_surface",
            data=np.stack(iv_surfaces).astype(np.float64),
            compression="lzf",
            chunks=(min(256, n), nk, nt),
        )
        f.create_dataset(
            "cell_mask",
            data=np.stack(cell_masks).astype(bool),
            compression="lzf",
            chunks=(min(256, n), nk, nt),
        )
        f.create_dataset(
            "raw_cell_mask",
            data=np.stack(raw_masks).astype(bool),
            compression="lzf",
            chunks=(min(256, n), nk, nt),
        )
        f.create_dataset(
            "cell_confidence",
            data=np.stack(confidences).astype(np.float32),
            compression="lzf",
            chunks=(min(256, n), nk, nt),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", type=str, default=DEFAULT_BIG_TICKERS)
    ap.add_argument("--tickers-file", type=str, default=None)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--grid", type=str, default=DEFAULT_GRID_PRESET, choices=sorted(GRID_PRESETS.keys()))
    ap.add_argument("--min-points", type=int, default=30, help="Minimum valid chain points required")
    ap.add_argument("--sleep", type=float, default=12.5, help="Seconds to sleep between API calls")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--max-failures", type=int, default=50)
    ap.add_argument("--out", type=str, default="../data/real_surfaces.h5")
    args = ap.parse_args()

    api_key = load_api_key(args.api_key)
    tickers = parse_tickers(args.tickers, args.tickers_file)
    dates = build_dates(args.start_date, args.end_date, args.step_days)
    log_m, mats = GRID_PRESETS[args.grid]

    jobs = [(t, d) for t in tickers for d in dates]
    print(f"[config] tickers={len(tickers)} dates={len(dates)} jobs={len(jobs)} grid={args.grid} ({len(log_m)}x{len(mats)})")

    records: list[SurfaceRecord] = []
    iv_surfaces: list[np.ndarray] = []
    cell_masks: list[np.ndarray] = []
    raw_masks: list[np.ndarray] = []
    confidences: list[np.ndarray] = []

    failures = 0

    for ticker, snapshot_date in tqdm(jobs, desc="Fetching/building"):
        try:
            payload = fetch_alpha_vantage(ticker, api_key, snapshot_date, args.timeout)
            if payload is None:
                failures += 1
                continue

            df, underlying, api_date = parse_chain(payload)
            if df.empty or underlying <= 0:
                failures += 1
                continue

            asof = pd.Timestamp(snapshot_date if snapshot_date else (api_date or date.today().isoformat()))
            iv, cell_mask, raw_mask, points_used = build_surface(df, underlying, asof, log_m, mats)
            conf = getattr(build_surface, "last_confidence")

            if points_used < args.min_points:
                continue

            raw_cov = 100.0 * float(raw_mask.mean())
            fill_cov = 100.0 * float(cell_mask.mean())

            records.append(SurfaceRecord(
                ticker=ticker,
                snapshot_date=snapshot_date or "latest",
                api_date=api_date,
                underlying=float(underlying),
                n_contracts=int(len(df)),
                n_points_used=int(points_used),
                raw_coverage_pct=raw_cov,
                filled_coverage_pct=fill_cov,
            ))
            iv_surfaces.append(iv)
            cell_masks.append(cell_mask)
            raw_masks.append(raw_mask)
            confidences.append(conf.astype(np.float32))

            if args.sleep > 0:
                time.sleep(args.sleep)

        except Exception:
            failures += 1
            if failures >= args.max_failures:
                raise RuntimeError(f"Too many failures ({failures}). Aborting.")
            if args.sleep > 0:
                time.sleep(args.sleep)

    if not records:
        raise RuntimeError("No valid surfaces collected. Check API key, symbols, and date range.")

    out = Path(args.out)
    save_dataset(
        out_path=out,
        records=records,
        iv_surfaces=iv_surfaces,
        cell_masks=cell_masks,
        raw_masks=raw_masks,
        confidences=confidences,
        log_moneyness=log_m,
        maturities=mats,
        grid_preset=args.grid,
        source="alpha_vantage_historical_options",
    )

    raw_cov_mean = np.mean([r.raw_coverage_pct for r in records])
    print(f"[done] saved {len(records):,} surfaces -> {out}")
    print(f"[stats] mean raw coverage={raw_cov_mean:.1f}% failures={failures}")


if __name__ == "__main__":
    main()

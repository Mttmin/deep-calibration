"""
python run_full_real_guided_pipeline.py \
    --tickers SPY,QQQ,IWM,AAPL,MSFT \
  --start-date 2025-01-01 --end-date 2025-03-31 --step-days 7 \
    --prefix q1_2025
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_BIG_TICKERS = (
    "SPY,QQQ,IWM,DIA,XLF,XLE,XLK,XLY,XLI,XLV,XLP,XLU,XLB,XLRE,"
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,BRK.B,JPM,V,MA,UNH,XOM,"
    "JNJ,PG,HD,COST,ABBV,BAC,WMT,AVGO,CVX,KO,MRK,PEP,ADBE,NFLX,CRM"
)

# Pipeline defaults tuned 
DEFAULT_GRID = "high41x14"
DEFAULT_MIN_POINTS = 30
DEFAULT_SLEEP = 5
DEFAULT_MAX_SURFACES = 300
DEFAULT_CANDIDATES = 2048
DEFAULT_TOP_K = 8
DEFAULT_CALIB_BATCH = 256
DEFAULT_USE_RAW_MASK = True
DEFAULT_USE_CONFIDENCE = True
DEFAULT_GENERATE_N = 2_000_000
DEFAULT_GUIDED_WEIGHT = 0.7
DEFAULT_GUIDED_JITTER = 0.03
DEFAULT_NAN_POLICY = "market_consistent"
DEFAULT_CHUNK = 16000
DEFAULT_SEED = 42


def run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(cwd), check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", type=str, default=DEFAULT_BIG_TICKERS)
    ap.add_argument("--tickers-file", type=str, default=None)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--start-date", type=str, default=None)
    ap.add_argument("--end-date", type=str, default=None)
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--prefix", type=str, default="pilot")

    # Keep only a few practical overrides; everything else uses defaults above.
    ap.add_argument("--generate-N", type=int, default=DEFAULT_GENERATE_N,
                    help="Guided synthetic samples to generate (set 0 to skip phase 3)")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                    help="Seconds between Alpha Vantage requests")
    ap.add_argument("--quick", action="store_true",
                    help="Pilot mode: fewer calibrations/candidates and smaller generated set")

    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    real_h5 = data_dir / f"real_surfaces_{args.prefix}.h5"
    bank_h5 = data_dir / f"guided_param_bank_{args.prefix}.h5"
    synth_h5 = data_dir / f"heston_guided_{args.prefix}.h5"

    py = sys.executable

    if args.quick:
        max_surfaces = 60
        candidates = 512
        generate_n = min(args.generate_N, 200_000) if args.generate_N > 0 else 0
    else:
        max_surfaces = DEFAULT_MAX_SURFACES
        candidates = DEFAULT_CANDIDATES
        generate_n = args.generate_N

    phase1 = [
        py,
        "real_surface_dataset.py",
        "--grid", DEFAULT_GRID,
        "--min-points", str(DEFAULT_MIN_POINTS),
        "--sleep", str(args.sleep),
        "--out", str(real_h5),
    ]
    if args.tickers:
        phase1 += ["--tickers", args.tickers]
    if args.tickers_file:
        phase1 += ["--tickers-file", args.tickers_file]
    if args.api_key:
        phase1 += ["--api-key", args.api_key]
    if args.start_date and args.end_date:
        phase1 += [
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--step-days", str(args.step_days),
        ]

    phase2 = [
        py,
        "calibrate_real_surfaces.py",
        "--real-h5", str(real_h5),
        "--out", str(bank_h5),
        "--grid", DEFAULT_GRID,
        "--max-surfaces", str(max_surfaces),
        "--candidates", str(candidates),
        "--top-k", str(DEFAULT_TOP_K),
        "--batch", str(DEFAULT_CALIB_BATCH),
        "--seed", str(DEFAULT_SEED),
    ]
    if DEFAULT_USE_RAW_MASK:
        phase2 += ["--use-raw-mask"]
    if DEFAULT_USE_CONFIDENCE:
        phase2 += ["--use-confidence"]

    run(phase1, cwd=here)
    run(phase2, cwd=here)

    if generate_n > 0:
        phase3 = [
            py,
            "heston_datagen.py",
            "--grid", DEFAULT_GRID,
            "--nan-policy", DEFAULT_NAN_POLICY,
            "--N", str(generate_n),
            "--chunk", str(DEFAULT_CHUNK),
            "--seed", str(DEFAULT_SEED),
            "--guided-bank", str(bank_h5),
            "--guided-weight", str(DEFAULT_GUIDED_WEIGHT),
            "--guided-jitter", str(DEFAULT_GUIDED_JITTER),
            "--out", str(synth_h5),
        ]
        run(phase3, cwd=here)

    print("[done]")
    print(f"  real surfaces:   {real_h5}")
    print(f"  guided bank:     {bank_h5}")
    if args.generate_N > 0:
        print(f"  guided dataset:  {synth_h5}")


if __name__ == "__main__":
    main()

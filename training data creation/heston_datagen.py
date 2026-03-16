"""
Usage
-----
  python heston_datagen.py                         # 2M train set
  python heston_datagen.py --N 20000000            # 20M train set
  python heston_datagen.py --val                   # 200k val set
  python heston_datagen.py --check                 # sanity only
  python heston_datagen.py --chunk 2048            # lower VRAM (f64 is ~2x memory)
"""
from __future__ import annotations
import math, argparse, time
from pathlib import Path

import numpy as np
import torch
import h5py
from scipy.stats import qmc
from tqdm import tqdm


#  Fixed (K, T) grid

LOG_MONEYNESS: np.ndarray = np.array(
    [-0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40],
    dtype=np.float64,
)
MATURITIES: np.ndarray = np.array(
    [1 / 12, 3 / 12, 6 / 12, 1.0, 1.5, 2.0],
    dtype=np.float64,
)
NK: int = len(LOG_MONEYNESS)   # 9
NT: int = len(MATURITIES)      # 6
# Output dimension: 54 IVs per surface.
#
# 1W excluded: even at sigma=14%, options at |m|>=0.10 have price < 1e-12,
# making IV inversion physically meaningless regardless of numerical precision.
# All 54 remaining cells have < 3% NaN rate across the full parameter space.

IS_PUT_SIDE: np.ndarray = LOG_MONEYNESS < 0.0   # (NK,) bool


#  Parameter space

PARAM_BOUNDS: np.ndarray = np.array([
    [0.10, 10.0],   # κ   mean-reversion speed
    [0.02,  0.25],  # θ   long-run variance  (σ_LR ∈ [14%, 50%])
    [0.05,  1.50],  # σ_v vol-of-vol
    [-0.95, 0.10],  # ρ   equity skew almost always negative
    [0.02,  0.25],  # v₀  initial variance  (σ₀  ∈ [14%, 50%])
    # v0/theta min=0.02: ensures all 54 grid cells produce invertible prices.
    # v0=0.01 (10% vol) → m=±0.20 at T=6M is borderline → NaN at ~60% of draws.
], dtype=np.float64)
PARAM_NAMES = ["kappa", "theta", "sigma_v", "rho", "v0"]


#  COS / IV hyperparameters

N_COS:    int   = 256     # Cosine terms; 256 gives < 0.0001% error in float64
HALF_ABS: float = 3.0     # ABSOLUTE truncation [a,b]=[-3,+3] for ALL maturities.
                          # L*sqrt(T) with L=12 gives b=17 at T=2Y → e^17=2.4e7
                          # → catastrophic cancellation in chi. Fixed width avoids this.
N_NEWTON: int   = 20      # Newton-Raphson IV steps; converges in ~6 at f64 precision
IV_LO:    float = 1e-6
IV_HI:    float = 10.0
CHUNK:    int   = 4_096   # GPU batch size; f64 uses 2x memory vs f32


#  Parameter sampling — scrambled Sobol + Feller rejection

def sample_params(N: int, seed: int = 42) -> np.ndarray:
    """
    Returns (N, 5) float64 Heston parameters via scrambled Sobol sequence.

    Feller condition 2κθ ≥ σ_v²: when violated, v_t can hit zero, the CIR
    diffusion degenerates, and the characteristic function may produce NaN.
    Oversample 4× and discard violators; typical pass rate 55-70%.
    """
    lo, hi  = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    sampler = qmc.Sobol(d=5, scramble=True, seed=seed)
    n_raw   = int(2 ** math.ceil(math.log2(N * 4)))
    raw     = sampler.random(n_raw)
    pts     = qmc.scale(raw, lo, hi)   # float64

    κ, θ, σ = pts[:, 0], pts[:, 1], pts[:, 2]
    ok      = (2.0 * κ * θ) >= σ**2
    valid   = pts[ok]

    if len(valid) < N:
        raise RuntimeError(
            f"Feller rejection left {len(valid):,} from {n_raw:,}. "
            "Lower σ_v upper bound or raise κ/θ lower bounds."
        )
    print(
        f"[sample] Sobol draws: {n_raw:,}  |  "
        f"Feller pass: {ok.mean()*100:.1f}%  |  keeping: {N:,}"
    )
    return valid[:N]   # (N, 5) float64


#  Heston CF — Little Trap formulation, float64 / complex128

def heston_cf(
    u:      torch.Tensor,   # (Nc,)  real frequencies, float64
    T:      float,
    params: torch.Tensor,   # (B, 5) float64
    r:      float,
    q:      float,
) -> torch.Tensor:           # (B, Nc) complex128
    """
    φ(u) = E_Q[e^{iu·ln(S_T/S_0)}] under Heston, Little Trap formulation.

    FLOAT64 NOTE: uses torch.complex128 (= two float64 values internally).
    On consumer GPUs (RTX 5090) f64 throughput ≈ 1/64 of f32, but generation
    is still fast enough for a one-time offline job.

    Little Trap (Albrecher et al. 2007):
        ξ  = κ − iuρσ_v
        d  = √(ξ² + σ_v²·u(u+i))
        g  = (ξ−d) / (ξ+d)
        logQ = ln((1−g·e^{−dT}) / (1−g))   ← branch-stable form
    """
    κ  = params[:, 0:1]   # (B, 1)
    θ  = params[:, 1:2]
    σv = params[:, 2:3]
    ρ  = params[:, 3:4]
    v0 = params[:, 4:5]

    uc   = u.to(dtype=torch.complex128)          # (Nc,)
    ξ    = κ - 1j * ρ * σv * uc                 # (B, Nc)
    d    = torch.sqrt(ξ**2 + σv**2 * uc * (uc + 1j))
    g    = (ξ - d) / (ξ + d)
    e_dT = torch.exp(-d * T)
    logQ = torch.log((1.0 - g * e_dT) / (1.0 - g))

    A = 1j * uc * (r - q) * T
    B = (κ * θ / σv**2) * ((ξ - d) * T - 2.0 * logQ)
    C = (v0  / σv**2)   * (ξ - d) * (1.0 - e_dT) / (1.0 - g * e_dT)

    return torch.exp(A + B + C)   # (B, Nc) complex128


#  COS call payoff coefficients V_k / K  (precomputed once, float64)

def cos_call_Vk(dev: torch.device) -> torch.Tensor:   # (Nc,) float64
    """
    V_k / K = (2/(b−a)) · [χ_k(0,b) − ψ_k(0,b)]
    K factors out; integral runs from y=0 to y=b (call payoff support).
    Half-weight on k=0 included here.
    """
    a, b  = -HALF_ABS, HALF_ABS
    ba    = b - a
    k     = torch.arange(N_COS, device=dev, dtype=torch.float64)
    alpha = k * math.pi / ba
    ang0  = -alpha * a

    cos0   = torch.cos(ang0)
    sin0   = torch.sin(ang0)
    coskpi = torch.cos(k * math.pi)

    denom    = (1.0 + alpha**2).clone(); denom[0] = 1.0
    chi      = (math.exp(b) * coskpi - cos0 - alpha * sin0) / denom
    chi[0]   = math.exp(b) - 1.0

    psi      = torch.zeros(N_COS, device=dev, dtype=torch.float64)
    psi[0]   = b
    psi[1:]  = -sin0[1:] / alpha[1:]

    Vk       = (2.0 / ba) * (chi - psi)
    Vk[0]   *= 0.5
    return Vk


#  Vectorised COS call pricer — one maturity, B params × NK strikes

def cos_call_prices(
    params:  torch.Tensor,   # (B, 5) float64
    Vk:      torch.Tensor,   # (Nc,)  float64
    T:       float,
    S0:      float,
    K_grid:  torch.Tensor,   # (NK,)  float64
    r:       float,
    q:       float,
    k_idx:   torch.Tensor,   # (Nc,)  int
) -> torch.Tensor:            # (B, NK) float64
    """
    GEMM decomposition: no (B, NK, Nc) tensor ever materialised.
        Re[φ·e^{−iu_k a}·e^{iu_k x_l}]·Vk = c.real @ W_cos − c.imag @ W_sin
    where c_base = φ·e^{−iu_k a} is (B,Nc) complex, W_{cos,sin} are (Nc,NK).
    Memory: O(B·Nc + Nc·NK) not O(B·Nc·NK).
    """
    a, b  = -HALF_ABS, HALF_ABS
    ba    = b - a
    u_k   = k_idx.double() * math.pi / ba           # (Nc,) float64

    phi     = heston_cf(u_k, T, params, r, q)       # (B, Nc) complex128
    phase_a = torch.exp(-1j * u_k.to(torch.complex128) * a)
    c_base  = phi * phase_a[None, :]                 # (B, Nc)

    x_l       = torch.log(torch.tensor(S0, device=params.device, dtype=torch.float64) / K_grid)
    phase_mat = u_k[None, :] * x_l[:, None]         # (NK, Nc)

    W_cos = (torch.cos(phase_mat) * Vk[None, :]).T  # (Nc, NK)
    W_sin = (torch.sin(phase_mat) * Vk[None, :]).T  # (Nc, NK)

    price_sum = c_base.real @ W_cos - c_base.imag @ W_sin   # (B, NK) float64
    prices    = math.exp(-r * T) * K_grid[None, :] * price_sum
    return prices.clamp(min=0.0)


#  OTM conversion via put-call parity

def call_to_otm(
    call_prices: torch.Tensor,   # (B, NK) float64
    K_grid:      torch.Tensor,   # (NK,)   float64
    T:           float,
    S0:          float,
    r:           float,
    q:           float,
    is_put:      torch.Tensor,   # (NK,)   bool
) -> torch.Tensor:                # (B, NK) float64
    disc  = math.exp(-r * T)
    fwd_d = S0 * math.exp(-q * T)
    put   = (call_prices + (K_grid * disc - fwd_d)[None, :]).clamp(min=0.0)
    return torch.where(is_put[None, :], put, call_prices)


#  BS price + vega, float64

_INV_SQRT2   = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _ncdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT2))


def _bs_price_vega(
    sigma:  torch.Tensor,   # (B, NK) float64
    F:      float,
    K:      torch.Tensor,   # (NK,)   float64
    T:      float,
    r:      float,
    is_put: torch.Tensor,   # (NK,)   bool
) -> tuple[torch.Tensor, torch.Tensor]:
    sqT  = math.sqrt(T); disc = math.exp(-r * T)
    F_t  = torch.full_like(sigma, F)
    d1   = (torch.log(F_t / K) + 0.5 * sigma**2 * T) / (sigma * sqT + 1e-30)
    d2   = d1 - sigma * sqT
    call = disc * (F_t * _ncdf(d1) - K * _ncdf(d2))
    put  = call + (K * disc - F_t * disc)
    vega = disc * F_t * sqT * torch.exp(-0.5 * d1**2) * _INV_SQRT2PI
    return torch.where(is_put[None, :], put, call), vega


#  IV inversion — Newton-Raphson, GPU-batched, float64

def prices_to_iv(
    otm:    torch.Tensor,   # (B, NK) float64 OTM prices
    K_grid: torch.Tensor,   # (NK,)   float64
    T:      float,
    S0:     float,
    r:      float,
    q:      float,
    is_put: torch.Tensor,   # (NK,)   bool
) -> torch.Tensor:           # (B, NK) float64, NaN where non-invertible
    """
    Newton-Raphson in float64 with Brenner-Subrahmanyam initial guess.
    20 iterations: converges to machine precision (~1e-14) in 6-8 steps.
    NaN flagged when price <= intrinsic + 1e-12 or residual > 1e-6.
    """
    F    = S0 * math.exp((r - q) * T)
    disc = math.exp(-r * T)

    call_int  = (F - K_grid * disc).clamp(min=0.0)
    put_int   = (K_grid * disc - F).clamp(min=0.0)
    intrinsic = torch.where(is_put, put_int, call_int)

    zero_mask = otm <= intrinsic[None, :] + 1e-12

    sigma = (otm / F) * math.sqrt(2.0 * math.pi / T)
    sigma = sigma.clamp(IV_LO, IV_HI)

    for _ in range(N_NEWTON):
        bs_p, v = _bs_price_vega(sigma, F, K_grid, T, r, is_put)
        sigma   = (sigma - (bs_p - otm) / v.clamp(min=1e-30)).clamp(IV_LO, IV_HI)

    bs_final, _ = _bs_price_vega(sigma, F, K_grid, T, r, is_put)
    bad = (bs_final - otm).abs() / (otm.abs() + 1e-15) > 1e-6

    sigma[zero_mask | bad] = float("nan")
    return sigma


#  Sanity check — compare GPU f64 COS vs numpy f64 reference

def sanity_check(dev: torch.device) -> None:
    """
    GPU (complex128) vs numpy (float64) reference at mid-range params.
    Expects < 0.001% price error (f64 truncation vs analytical).

    FLOAT32 WARNING: never test with sigma_v < 0.04 in f32 mode.
    In f64 this is not an issue (machine epsilon ~2.2e-16).
    """
    print("\n[sanity] GPU complex128 COS vs numpy float64 reference ...")

    def heston_cf_np(u, T, kappa, theta, sv, rho, v0, r=0., q=0.):
        xi=kappa-1j*rho*sv*u; d=np.sqrt(xi**2+sv**2*u*(u+1j))
        g=(xi-d)/(xi+d); e_dT=np.exp(-d*T); logQ=np.log((1-g*e_dT)/(1-g))
        return np.exp(1j*u*(r-q)*T + (kappa*theta/sv**2)*((xi-d)*T-2*logQ)
                      + (v0/sv**2)*(xi-d)*(1-e_dT)/(1-g*e_dT))

    def cos_f64_np(S0, K, T, kappa, theta, sv, rho, v0, r=0., q=0., N=256, H=3.0):
        a,b=-H,H; ba=b-a; k=np.arange(N,dtype=float); u_k=k*math.pi/ba
        phi=heston_cf_np(u_k,T,kappa,theta,sv,rho,v0,r,q)
        c=phi*np.exp(-1j*u_k*a); x=math.log(S0/K); alpha=k*math.pi/ba; ang0=-alpha*a
        denom=1+alpha**2; denom[0]=1.
        chi=(math.exp(b)*np.cos(k*np.pi)-np.cos(ang0)-alpha*np.sin(ang0))/denom; chi[0]=math.exp(b)-1
        psi=np.zeros(N); psi[0]=b; psi[1:]=-np.sin(ang0[1:])/alpha[1:]
        Vk=(2/ba)*(chi-psi); Vk_w=Vk.copy(); Vk_w[0]*=0.5
        s=np.sum((c.real*np.cos(u_k*x)-c.imag*np.sin(u_k*x))*Vk_w)
        return max(math.exp(-r*T)*K*s, max(S0*math.exp((r-q)*T)-K*math.exp(-r*T),0.))

    kappa, theta, sv, rho, v0 = 2.0, 0.04, 0.30, -0.70, 0.04
    S0, r, q = 1.0, 0.0, 0.0
    K_t    = torch.tensor([S0], device=dev, dtype=torch.float64)
    k_idx  = torch.arange(N_COS, device=dev)
    Vk     = cos_call_Vk(dev)
    par    = torch.tensor([[kappa, theta, sv, rho, v0]], device=dev, dtype=torch.float64)
    is_put = torch.tensor([False], device=dev)

    all_ok = True
    for T in MATURITIES:
        ref  = cos_f64_np(S0, S0, float(T), kappa, theta, sv, rho, v0, r, q)
        gpu  = cos_call_prices(par, Vk, float(T), S0, K_t, r, q, k_idx)[0, 0].item()
        pe   = abs(gpu - ref) / (ref + 1e-15) * 100
        otm  = torch.tensor([[gpu]], device=dev, dtype=torch.float64)
        iv   = prices_to_iv(otm, K_t, float(T), S0, r, q, is_put)[0, 0].item()
        ok   = pe < 0.001 and not math.isnan(iv)
        all_ok &= ok
        print(
            f"  T={T:.4f}  ref={ref:.8f}  gpu={gpu:.8f}  "
            f"err={pe:.5f}%  iv={iv:.5f}  {'OK' if ok else 'FAIL'}"
        )
    if not all_ok:
        raise RuntimeError("Sanity check FAILED")
    print("[sanity] PASSED\n")


#  Main generation loop

def generate(
    N:           int   = 2_000_000,
    output_path: str   = "heston_train.h5",
    S0:          float = 1.0,
    r:           float = 0.0,
    q:           float = 0.0,
    chunk_size:  int   = CHUNK,
    seed:        int   = 42,
) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}" + (
        f"  ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""
    ))

    sanity_check(dev)

    params_np = sample_params(N, seed=seed)                      # (N, 5) float64
    K_grid_np = S0 * np.exp(LOG_MONEYNESS)                       # (NK,)  float64
    K_grid    = torch.tensor(K_grid_np, device=dev, dtype=torch.float64)
    k_idx     = torch.arange(N_COS, device=dev)
    Vk        = cos_call_Vk(dev)                                  # (Nc,)  float64
    is_put    = torch.tensor(IS_PUT_SIDE, device=dev)             # (NK,)  bool

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = math.ceil(N / chunk_size)

    with h5py.File(out, "w") as f:
        f.attrs["param_names"]   = PARAM_NAMES
        f.attrs["log_moneyness"] = LOG_MONEYNESS.tolist()
        f.attrs["maturities"]    = MATURITIES.tolist()
        f.attrs["K_grid"]        = K_grid_np.tolist()
        f.attrs["S0"]            = S0
        f.attrs["r"]             = r
        f.attrs["q"]             = q
        f.attrs["N_COS"]         = N_COS
        f.attrs["HALF_ABS"]      = HALF_ABS
        f.attrs["N_total"]       = N
        f.attrs["param_lo"]      = PARAM_BOUNDS[:, 0].tolist()
        f.attrs["param_hi"]      = PARAM_BOUNDS[:, 1].tolist()
        f.attrs["NK"]            = NK
        f.attrs["NT"]            = NT

        cs_n = min(4096, N)
        f.create_dataset("params",
            shape=(N, 5), dtype="float64",
            chunks=(cs_n, 5), compression="lzf")
        f.create_dataset("iv_surface",
            shape=(N, NK, NT), dtype="float64",
            chunks=(min(512, N), NK, NT), compression="lzf")
        f.create_dataset("cell_mask",
            shape=(N, NK, NT), dtype="bool",
            chunks=(min(512, N), NK, NT), compression="lzf")
        f.create_dataset("valid_count",
            shape=(N,), dtype="uint8",
            chunks=(cs_n,))

        ds_p  = f["params"]
        ds_iv = f["iv_surface"]
        ds_cm = f["cell_mask"]
        ds_vc = f["valid_count"]

        total_valid_cells = 0
        t0 = time.time()

        for ci in tqdm(range(n_chunks), desc="Generating"):
            i0 = ci * chunk_size
            i1 = min(i0 + chunk_size, N)
            B  = i1 - i0

            params_t = torch.tensor(params_np[i0:i1], device=dev, dtype=torch.float64)
            iv_chunk = torch.full((B, NK, NT), float("nan"),
                                  dtype=torch.float64, device=dev)

            for ti, T in enumerate(MATURITIES.tolist()):
                call_p = cos_call_prices(params_t, Vk, T, S0, K_grid, r, q, k_idx)
                otm_p  = call_to_otm(call_p, K_grid, T, S0, r, q, is_put)
                iv_chunk[:, :, ti] = prices_to_iv(otm_p, K_grid, T, S0, r, q, is_put)

            mask_chunk = ~torch.isnan(iv_chunk)           # (B, NK, NT) bool
            vcount     = mask_chunk.sum(dim=(1, 2))       # (B,) int

            total_valid_cells += int(mask_chunk.sum().item())

            ds_p[i0:i1]  = params_np[i0:i1]
            ds_iv[i0:i1] = iv_chunk.cpu().numpy()
            ds_cm[i0:i1] = mask_chunk.cpu().numpy()
            ds_vc[i0:i1] = vcount.byte().cpu().numpy()

        elapsed = time.time() - t0
        max_cells = N * NK * NT
        f.attrs["gen_time_s"]         = elapsed
        f.attrs["total_valid_cells"]  = total_valid_cells
        f.attrs["cell_fill_rate_pct"] = 100.0 * total_valid_cells / max_cells

    fill_pct = 100.0 * total_valid_cells / max_cells
    size     = out.stat().st_size / 1e9
    print(f"\n[done]  cell fill rate: {fill_pct:.2f}%  ({total_valid_cells:,} / {max_cells:,})")
    print(f"[file]  {out}  ({size:.3f} GB)")
    print(f"[perf]  {elapsed/60:.1f} min  |  {N/elapsed:,.0f} samples/s")
    if dev.type == "cuda":
        print(f"[vram]  peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


#  PyTorch Dataset — lazy HDF5, fork-safe

class HestonDataset(torch.utils.data.Dataset):
    """
    Returns (params_norm, iv_flat, mask_flat):
      params_norm : (5,)       float32  normalised to [0,1] per parameter
      iv_flat     : (NK*NT,)   float32  IVs; NaN where cell_mask is False
      mask_flat   : (NK*NT,)   bool     True where IV is valid

    In the training loop, mask_flat is used to zero-out NaN cells in the loss:
        loss = (mask * (nn_out - iv_flat)**2).sum() / mask.sum().clamp(min=1)

    Lazy open: HDF5 file opened inside __getitem__ after DataLoader fork.
    """

    def __init__(self, h5_path: str, min_valid_cells: int = 30):
        """
        min_valid_cells: discard samples with fewer than this many valid cells.
        Default 30 (of 54): ensures ≥55% of the surface is populated.
        Set to 0 to keep all samples.
        """
        self.h5_path = h5_path
        self._h5     = None

        with h5py.File(h5_path, "r") as f:
            vc            = f["valid_count"][:]
            self.idx      = np.where(vc >= min_valid_cells)[0]
            self.param_lo = np.array(f.attrs["param_lo"], dtype=np.float32)
            self.param_hi = np.array(f.attrs["param_hi"], dtype=np.float32)

        print(
            f"[dataset] {h5_path}: {len(self.idx):,} samples "
            f"(min_valid_cells={min_valid_cells})"
        )

    def _open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        self._open()
        j = int(self.idx[i])

        params = self._h5["params"][j].astype(np.float32)
        iv     = self._h5["iv_surface"][j].astype(np.float32)   # (NK, NT)
        mask   = self._h5["cell_mask"][j]                        # (NK, NT) bool

        params = (params - self.param_lo) / (self.param_hi - self.param_lo + 1e-12)

        # Replace NaN with 0 in iv (loss will mask these out anyway)
        iv[~mask] = 0.0

        return (
            torch.from_numpy(params),
            torch.from_numpy(iv.flatten()),         # (NK*NT,)
            torch.from_numpy(mask.flatten()),        # (NK*NT,) bool
        )

    def __del__(self) -> None:
        if self._h5 is not None:
            try: self._h5.close()
            except Exception: pass


#  CLI

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--N",     type=int,   default=2_000_000)
    ap.add_argument("--out",   type=str,   default="heston_train.h5")
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--chunk", type=int,   default=CHUNK,
                    help=f"GPU batch size (default {CHUNK}; f64 uses 2x VRAM vs f32)")
    ap.add_argument("--S0",    type=float, default=1.0)
    ap.add_argument("--r",     type=float, default=0.0)
    ap.add_argument("--q",     type=float, default=0.0)
    ap.add_argument("--val",   action="store_true",
                    help="200k val set (seed+9999, appends _val.h5)")
    ap.add_argument("--check", action="store_true",
                    help="Sanity check only, no generation")
    args = ap.parse_args()

    if args.check:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sanity_check(dev)
    elif args.val:
        out = args.out.replace(".h5", "") + "_val.h5"
        generate(N=200_000, output_path=out, seed=args.seed + 9999,
                 chunk_size=args.chunk, S0=args.S0, r=args.r, q=args.q)
    else:
        generate(N=args.N, output_path=args.out, seed=args.seed,
                 chunk_size=args.chunk, S0=args.S0, r=args.r, q=args.q)
# src/gpv_assignment.py
"""
Problem set: Nonparametric Estimation of First Price Procurement Auction (GPV)
Outputs one plot with THREE step functions (low/median/high competition) and a CSV.

What this script does (per assignment):
  1) Takes your dataset of auctions with number of bidders and winning bids.
  2) Selects three competition levels at the 25th/50th/75th percentiles of N.
     If the exact N has too few obs, it pools nearby N (±1, then ±2).
  3) For each level, applies the GPV steps:
       - Estimate winning-bid ECDF/PDF (F_S, f_S)
       - Convert to single-bidder bid ECDF/PDF (F_B, f_B)
       - GPV inversion: c(b) = b - (1/(N-1)) * (1 - F_B(b)) / f_B(b)
       - Build F_C by mapping b -> c(b) with F_C(c(b)) = F_B(b)
       - (Optional) smooth pseudo-costs with triweight to get a PDF; we plot only CDFs
  4) Saves:
       - data/FC_steps.csv  (columns: level, N_label, N_used, c, FC)
       - data/FC_steps_plot.pdf  (single page: three step CDFs)

Accepts either:
  - CSV with columns: num_bidders, lowest_bid
  - or a .dat like professor's nlp.dat with columns: t N W Z (we only use N and W)

References to assignment:
  - three step functions for low/median/high competition (25th/50th/75th pct of N), pooling allowed.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ------------------------- utilities -------------------------

def _midrank_ecdf_on_sample(x_sorted: np.ndarray) -> np.ndarray:
    """ECDF evaluated on the sample points using mid-ranks (avoids 0/1)."""
    n = x_sorted.size
    return (np.arange(1, n + 1) - 0.5) / n

def _triweight_kde(samples: np.ndarray, eval_points: np.ndarray) -> np.ndarray:
    """
    Triweight kernel density estimate at eval_points.
    K(u) = (35/32)*(1 - u^2)^3 * 1(|u|<=1)
    Bandwidth: Silverman ROT (works well; robust and standard).
    """
    x = np.asarray(samples, float)
    z = np.asarray(eval_points, float)
    n = x.size
    if n < 2:
        return np.zeros_like(z)
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = sd if (sd > 0 and (iqr <= 0)) else min(sd, iqr / 1.349) if sd > 0 else (iqr / 1.349 if iqr > 0 else 1.0)
    h = 1.06 * sigma * (n ** (-1/5)) if sigma > 0 else max(1e-6, np.mean(np.abs(np.diff(np.unique(x)))) or 1.0)

    u = (z[:, None] - x[None, :]) / h
    mask = (np.abs(u) <= 1.0).astype(float)
    K = (35.0 / 32.0) * np.power(1.0 - u * u, 3) * mask  # (G, n)
    dens = K.sum(axis=1) / (n * h)
    return dens

def _pick_groups(N_series: pd.Series, min_obs: int = 12, max_band: int = 2):
    """
    Choose three competition 'levels' around the 25/50/75 percentiles of N.
    Pool to [N0-k, N0+k] with the smallest k (<= max_band) that reaches min_obs.
    Returns: list of dicts {level, N0, Nmin, Nmax, label}
    """
    Ns = N_series.to_numpy()
    q25, q50, q75 = np.quantile(Ns, [0.25, 0.50, 0.75], method="nearest")
    
    # Try to get distinct values - if percentiles coincide, spread them out
    targets = [int(q25), int(q50), int(q75)]
    uniq = np.unique(Ns)
    
    # If q25 == q50, try to use distinct N values
    if targets[0] == targets[1] and len(uniq) >= 3:
        # Use the lower value for low, keep median, use higher for high
        low_candidates = uniq[uniq <= targets[0]]
        if len(low_candidates) >= 2:
            targets[0] = int(low_candidates[-2])  # Second lowest
    
    levels = [("low", targets[0]), ("median", targets[1]), ("high", targets[2])]
    
    groups = []
    for label, tgt in levels:
        N0 = int(uniq[np.argmin(np.abs(uniq - tgt))])
        band = 0
        while True:
            Nmin, Nmax = N0 - band, N0 + band
            mask = (Ns >= Nmin) & (Ns <= Nmax)
            if mask.sum() >= min_obs or band >= max_band:
                groups.append(dict(level=label, N0=N0, Nmin=Nmin, Nmax=Nmax, label=f"N∈[{Nmin},{Nmax}]"))
                break
            band += 1
    return groups

# ------------------------- GPV core -------------------------

def _triweight_kernel_pdf(pseudo_bids: np.ndarray) -> np.ndarray:
    """
    Triweight kernel density estimator for pseudo-costs (following GPV notebook).
    This matches the pseudo_pdf function from the assignment notebook.
    """
    sorted_bids = np.sort(pseudo_bids)
    n = len(pseudo_bids)
    pseudo_pdf = np.zeros_like(pseudo_bids)
    
    # Bandwidth calculation (Silverman's rule adapted for triweight)
    delta = 2.978 * 1.06 * (np.var(pseudo_bids)**(1/2)) ** (-1/6)
    
    for r in range(n):
        # Calculate kernel weights
        obj_triw = (1/delta) * (sorted_bids - sorted_bids[r])
        triweightker = np.where(np.abs(obj_triw) <= 1, (35/32) * (1 - obj_triw**2)**3, 0)
        striweightker = (1/delta) * np.sum(triweightker)
        pseudo_pdf[r] = (1/n) * striweightker
    
    return pseudo_pdf

def _empirical_cdf(values: np.ndarray) -> np.ndarray:
    """
    Simple empirical CDF (following the assignment notebook approach).
    Returns F(x) for sorted values.
    """
    n = values.shape[0]
    # Return the empirical CDF: F(x_i) = (i+1)/n for the i-th ordered value
    return np.arange(1, n + 1) / n

def _cost_cdf_from_wins(wins: np.ndarray, N_use: int, downsample: int = 140):
    """
    GPV estimation following the exact methodology from the assignment notebook.
    
    Steps (as in GPVprocurement.ipynb):
    1. Estimate F_w (ECDF of winning bids) and f_w (kernel density)
    2. Calculate pseudo-costs: c_win = w - (1/(N-1)) * (1-F_w) / f_w
    3. Estimate F_z (ECDF of pseudo-costs) and f_z (triweight kernel)
    4. Transform to cost distribution: F_c = 1 - (1 - F_z)^(1/N)
    """
    w = np.sort(np.asarray(wins, float))
    n = w.size
    if n < 5:
        return np.array([]), np.array([])

    # Step 1: Estimate empirical CDF and PDF of winning bids
    # Use ECDF (returns F including the initial 0)
    Fw_full = _empirical_cdf(w)
    Fw = Fw_full  # F_w evaluated at sorted w
    
    # Estimate f_w using Gaussian KDE with bandwidth adjustment for robust estimation
    # Use Scott's or Silverman's rule for more stable bandwidth
    kde = gaussian_kde(w, bw_method='scott')
    fw = kde.evaluate(w)
    # Ensure f_w is bounded away from zero to avoid division issues
    fw = np.clip(fw, 1e-10, None)

    # Step 2: Calculate pseudo-costs (c_win in the notebook)
    # Add safeguard: clip the ratio to avoid extreme values
    ratio = (1.0 - Fw) / fw
    # Cap the ratio at a reasonable maximum to prevent explosion
    max_ratio = np.percentile(w, 95) - np.percentile(w, 5)  # Use range as scale
    ratio = np.clip(ratio, 0, max_ratio)
    
    c_win = w - (1.0 / max(N_use - 1, 1)) * ratio

    # Step 3: Estimate pseudo CDF and PDF
    # Sort c_win and get its ECDF
    c_sorted = np.sort(c_win)
    Fz_full = _empirical_cdf(c_sorted)
    Fz = Fz_full
    
    # Step 4: Transform to cost CDF
    # Following notebook: Fc = 1 - (1 - Fz)^(1/N)
    Fc = 1.0 - np.power(1.0 - Fz, 1.0 / N_use)
    
    # Ensure monotonicity and bounds
    Fc = np.clip(np.maximum.accumulate(Fc), 0.0, 1.0)

    # Downsample if too many points
    if c_sorted.size > downsample:
        step = max(1, c_sorted.size // downsample)
        c_sorted = c_sorted[::step]
        Fc = Fc[::step]
    
    return c_sorted, Fc

    # (Optional) triweight smoothing of pseudo-costs if you want a smooth f_C:
    # fz = _triweight_kde(c, c) ; Fc_smooth = 1 - np.power(1 - _midrank_ecdf_on_sample(c), 1.0/N_use)
    # For the assignment, only the step CDFs are requested.

    # Downsample for lighter CSV/plot if many points
    if c_sorted.size > downsample:
        step = max(1, c_sorted.size // downsample)
        c_sorted = c_sorted[::step]
        FC = FC[::step]
    return c_sorted, FC

# ------------------------- I/O and pipeline -------------------------

def _load_dataset(path: str) -> pd.DataFrame:
    """
    Accepts:
      - CSV with columns: num_bidders, lowest_bid
      - DAT with columns: t N W Z  (we only use N, W)
    Returns df with columns: num_bidders, lowest_bid
    """
    p = Path(path)
    if p.suffix.lower() == ".dat":
        df = pd.read_csv(p, delim_whitespace=True, header=None, names=["t", "N", "W", "Z"])
        out = pd.DataFrame({"num_bidders": df["N"].astype(int), "lowest_bid": df["W"].astype(float)})
        return out
    else:
        df = pd.read_csv(p)
        cols = {c.strip().lower(): c for c in df.columns}
        if "num_bidders" in cols and "lowest_bid" in cols:
            return df.rename(columns={cols["num_bidders"]: "num_bidders", cols["lowest_bid"]: "lowest_bid"})[
                ["num_bidders", "lowest_bid"]
            ].copy()
        # try professor-like columns
        if "n" in cols and "w" in cols:
            return df.rename(columns={cols["n"]: "num_bidders", cols["w"]: "lowest_bid"})[
                ["num_bidders", "lowest_bid"]
            ].copy()
        raise ValueError("Input must have columns 'num_bidders' and 'lowest_bid' (or 'N' and 'W').")

def run(in_path: str, out_csv: str, out_pdf: str, min_obs: int = 12, max_band: int = 2):
    df = _load_dataset(in_path)
    # clean: keep only positive bids, but keep N>=1 for percentile calculation
    df = df[(df["lowest_bid"].astype(float) > 0)].copy()
    df["num_bidders"] = df["num_bidders"].astype(int)
    df["lowest_bid"] = df["lowest_bid"].astype(float)

    # pick 3 competition levels (using ALL data including N=1 for percentiles)
    groups = _pick_groups(df["num_bidders"], min_obs=min_obs, max_band=max_band)

    # Now filter out N=1 for estimation (GPV requires N>=2)
    df = df[df["num_bidders"] >= 2].copy()

    rows = []
    curves = []

    for g in groups:
        mask = (df["num_bidders"] >= g["Nmin"]) & (df["num_bidders"] <= g["Nmax"])
        wins = df.loc[mask, "lowest_bid"].to_numpy()
        Ns = df.loc[mask, "num_bidders"].to_numpy()
        if wins.size < 5:
            print(f"[WARN] Too few observations for {g['level']} ({g['label']}); skipping.")
            continue
        N_use = int(np.median(Ns))  # central N in the pooled band
        c, FC = _cost_cdf_from_wins(wins, N_use)
        if c.size == 0:
            print(f"[WARN] No curve produced for {g['level']} ({g['label']}).")
            continue

        curves.append((g["level"], f"{g['label']} (N≈{N_use})", c, FC))
        for ci, fi in zip(c, FC):
            rows.append(dict(level=g["level"], N_label=g["label"], N_used=N_use, c=ci, FC=fi))
        print(f"[OK] {g['level']:>6} | {g['label']:<10} | used N={N_use:<2} | obs={wins.size:<4} | grid={c.size}")

    # save CSV
    out_csv_p = Path(out_csv); out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv_p, index=False)

    # plot the three step functions in ONE figure (as required)
    if curves:
        plt.figure(figsize=(7.5, 5.2), dpi=140)
        colors = {'low': 'C0', 'median': 'C1', 'high': 'C2'}
        
        for level, label, c, FC in curves:
            plt.step(c, FC, where="post", 
                    label=f"{level.capitalize()}  {label}",
                    color=colors.get(level, 'black'),
                    linewidth=2)
        
        plt.xlabel("Cost, c", fontsize=11)
        plt.ylabel(r"$\hat F_C(c)$", fontsize=11)
        plt.title("Estimated Cost CDFs by Competition Level", fontsize=12)
        plt.grid(True, alpha=0.35)
        plt.legend(fontsize=9)
        plt.ylim(0, 1.05)  # Show full CDF range
        
        out_pdf_p = Path(out_pdf); out_pdf_p.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(out_pdf_p, bbox_inches="tight"); plt.close()

    print(f"[DONE] wrote: {out_csv} and {out_pdf}")

# ------------------------- CLI -------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPV assignment runner (three cost-CDF step functions).")
    ap.add_argument("--in", dest="in_path", default="data/bid_min_and_count.csv",
                    help="Input data: CSV with columns (num_bidders, lowest_bid) or .dat with N,W")
    ap.add_argument("--out-csv", default="data/FC_steps.csv", help="Output CSV path")
    ap.add_argument("--out-pdf", default="data/FC_steps_plot.pdf", help="Output PDF path")
    ap.add_argument("--min-obs", type=int, default=12, help="Min obs per group before pooling")
    ap.add_argument("--max-band", type=int, default=2, help="Max pooling half-width for N (±band)")
    args = ap.parse_args()
    run(args.in_path, args.out_csv, args.out_pdf, min_obs=args.min_obs, max_band=args.max_band)
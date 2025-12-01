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

Accepts CSV with columns: num_bidders, lowest_bid
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ------------------------- utilities -------------------------
def _pick_groups(N_series: pd.Series, min_obs: int = 12, max_band: int = 2):
    """
    Select three competition level groups for GPV estimation.
    
    Strategy:
        1. Find target N values at 25th, 50th, and 75th percentiles
        2. For each target, start with exact N value
        3. If fewer than min_obs, expand to [N-k, N+k] incrementally
        4. Stop when reaching min_obs or max_band is exceeded
    
    This pooling strategy balances two objectives:
        - Having enough observations for reliable estimation
        - Keeping competition levels relatively homogeneous
    
    Args:
        N_series: Series of number of bidders across all auctions
        min_obs: Minimum observations required per group (default: 12)
        max_band: Maximum pooling radius (default: 2, means ±2 bidders)
        
    Returns:
        List of dicts with keys: level, N0, Nmin, Nmax, label
            level: 'low', 'median', or 'high'
            N0: Target number of bidders
            Nmin, Nmax: Actual range used after pooling
            label: Human-readable description
    """
    Ns = N_series.to_numpy()
    
    # Find target N values at key percentiles
    q25, q50, q75 = np.quantile(Ns, [0.25, 0.50, 0.75], method="nearest")
    
    # Handle case where percentiles coincide (all data clustered at same N)
    targets = [int(q25), int(q50), int(q75)]
    uniq = np.unique(Ns)
    
    # If low and median targets are the same, try to spread them out
    if targets[0] == targets[1] and len(uniq) >= 3:
        low_candidates = uniq[uniq <= targets[0]]
        if len(low_candidates) >= 2:
            targets[0] = int(low_candidates[-2])  # Use second-lowest value
    
    levels = [("low", targets[0]), ("median", targets[1]), ("high", targets[2])]
    
    groups = []
    for label, tgt in levels:
        # Find closest actual N value to target
        N0 = int(uniq[np.argmin(np.abs(uniq - tgt))])
        
        # Expand pooling window until we have enough observations
        band = 0
        while True:
            Nmin, Nmax = N0 - band, N0 + band
            mask = (Ns >= Nmin) & (Ns <= Nmax)
            
            # Stop if we have enough obs or reached max pooling width
            if mask.sum() >= min_obs or band >= max_band:
                groups.append(dict(
                    level=label, 
                    N0=N0, 
                    Nmin=Nmin, 
                    Nmax=Nmax, 
                    label=f"N∈[{Nmin},{Nmax}]"
                ))
                break
            band += 1
    
    return groups


# ------------------------- GPV core -------------------------

def _empirical_cdf(values: np.ndarray) -> np.ndarray:
    """
    Calculate simple empirical CDF following the assignment notebook.
    
    For sorted values x_1 ≤ x_2 ≤ ... ≤ x_n:
        F(x_i) = i / n
    
    This gives the proportion of data at or below each value.
    
    Args:
        values: Sorted array of data values
        
    Returns:
        Array of CDF values F(x_i) = i/n for i=1,...,n
    """
    n = values.shape[0]
    return np.arange(1, n + 1) / n


def _cost_cdf_from_wins(wins: np.ndarray, N_use: int, downsample: int = 140):
    """
    GPV nonparametric estimator for the cost distribution from winning bids.
    
    Theoretical foundation:
        In a first-price auction with N bidders who observe private costs c drawn
        from F_C and bid optimally, the winning bid w follows a different distribution
        F_W. GPV (1992) provides a method to recover F_C from observed winning bids.
    
    Four-step GPV procedure:
    
    1. Estimate F_w and f_w (winning bid distribution)
       - F_w: empirical CDF of observed winning bids
       - f_w: kernel density estimate (Gaussian KDE with Scott's bandwidth)
    
    2. Calculate pseudo-costs (intermediate transformation)
       - Formula: c_win = w - (1/(N-1)) * (1 - F_w(w)) / f_w(w)
       - This accounts for the strategic markup bidders add above their costs
       - The term (1-F_w)/f_w is the inverse hazard rate
    
    3. Estimate F_z and f_z (pseudo-cost distribution)
       - F_z: empirical CDF of calculated pseudo-costs
       - f_z: could use triweight kernel (optional, not used in final CDF)
    
    4. Transform to cost CDF
       - Formula: F_c(c) = 1 - (1 - F_z(c))^(1/N)
       - This accounts for the fact that w is the minimum of N bids
       - Order statistics relationship: if Z ~ F_z, then min(Z_1,...,Z_N) follows
         a distribution related to F_z^N
    
    Robustness features:
        - Clips f_w away from zero to prevent division issues
        - Bounds the ratio (1-F_w)/f_w to prevent extreme pseudo-costs
        - Enforces monotonicity in final F_c
        - Downsamples output for computational efficiency
    
    Args:
        wins: Array of observed winning bids
        N_use: Number of bidders (central value if pooled)
        downsample: Maximum number of output points (default: 140)
        
    Returns:
        Tuple of (c_sorted, Fc):
            c_sorted: Grid of cost values
            Fc: Estimated CDF values at those costs
    """
    # Sort winning bids
    w = np.sort(np.asarray(wins, float))
    n = w.size
    
    # Need minimum sample size for reliable estimation
    if n < 5:
        return np.array([]), np.array([])

    # ===== STEP 1: Estimate F_w and f_w =====
    # Empirical CDF of winning bids
    Fw = _empirical_cdf(w)
    
    # Kernel density estimate of f_w
    # Scott's rule chooses bandwidth to minimize AMISE for normal reference
    kde = gaussian_kde(w, bw_method='scott')
    fw = kde.evaluate(w)
    
    # Ensure density is bounded away from zero (prevents division by zero)
    fw = np.clip(fw, 1e-10, None)

    # ===== STEP 2: Calculate pseudo-costs =====
    # Inverse hazard rate: (1 - F_w) / f_w
    ratio = (1.0 - Fw) / fw
    
    # Cap ratio to prevent extreme values from density estimation artifacts
    # Use the interdecile range as a scale reference
    max_ratio = np.percentile(w, 95) - np.percentile(w, 5)
    ratio = np.clip(ratio, 0, max_ratio)
    
    # GPV inversion formula for pseudo-costs
    # c = w - markup, where markup = (1/(N-1)) * inverse_hazard_rate
    c_win = w - (1 / (N_use - 1.0)) * ratio

    # ===== STEP 3: Estimate F_z (pseudo-cost ECDF) =====
    c_sorted = np.sort(c_win)
    Fz = _empirical_cdf(c_sorted)
    
    # ===== STEP 4: Transform to cost CDF =====
    # Account for order statistics: F_c = 1 - (1 - F_z)^(1/N)
    # This inverts the relationship F_w(w) ≈ F_c(c)^N for the minimum bid
    Fc = 1.0 - np.power(1.0 - Fz, 1.0 / N_use)
    
    # Enforce monotonicity (sometimes needed due to numerical issues)
    # Ensure F_c is non-decreasing and stays in [0, 1]
    Fc = np.clip(np.maximum.accumulate(Fc), 0.0, 1.0)

    # Downsample if we have too many points (for efficiency)
    if c_sorted.size > downsample:
        step = max(1, c_sorted.size // downsample)
        c_sorted = c_sorted[::step]
        Fc = Fc[::step]
    
    return c_sorted, Fc


# ------------------------- I/O and pipeline -------------------------

def _load_dataset(path: str) -> pd.DataFrame:
    """
    Load auction data from CSV file.
    
    Accepts CSV with columns: num_bidders, lowest_bid
    (Also accepts N, W as alternative column names)
    
    Args:
        path: File path to CSV file
        
    Returns:
        DataFrame with standardized columns: num_bidders, lowest_bid
        
    Raises:
        ValueError: If required columns are not found or file is not CSV
    """
    p = Path(path)
    
    # Verify it's a CSV file
    if p.suffix.lower() not in [".csv"]:
        raise ValueError(
            f"Only CSV files are supported. Got: {p.suffix}\n"
            f"Please provide a CSV file with columns 'num_bidders' and 'lowest_bid'."
        )
    
    # Load CSV
    df = pd.read_csv(p)
    
    # Create case-insensitive column lookup
    cols = {c.strip().lower(): c for c in df.columns}
    
    # Try standard column names
    if "num_bidders" in cols and "lowest_bid" in cols:
        return df.rename(columns={
            cols["num_bidders"]: "num_bidders", 
            cols["lowest_bid"]: "lowest_bid"
        })[["num_bidders", "lowest_bid"]].copy()
    
    # Try alternative column names (N, W)
    if "n" in cols and "w" in cols:
        return df.rename(columns={
            cols["n"]: "num_bidders", 
            cols["w"]: "lowest_bid"
        })[["num_bidders", "lowest_bid"]].copy()
    
    # If we get here, required columns weren't found
    raise ValueError(
        "CSV must have columns 'num_bidders' and 'lowest_bid' "
        "(or 'N' and 'W'). "
        f"Found columns: {list(df.columns)}"
    )


def run(in_path: str, out_csv: str, out_pdf: str, min_obs: int = 12, 
        max_band: int = 2, remove_outliers: bool = True, 
        outlier_percentile: float = 95.0):
    """
    Main pipeline: load data, estimate cost CDFs, save results.
    
    Workflow:
        1. Load and clean data
        2. Remove outliers (optional)
        3. Filter to N ≥ 3 (N=2 causes numerical issues)
        4. Select three competition level groups
        5. Estimate cost CDF for each group using GPV
        6. Save CSV with cost grid and CDF values
        7. Create plot with three step functions
    
    Args:
        in_path: Input CSV file path
        out_csv: Output CSV path for numerical results
        out_pdf: Output PDF path for plot
        min_obs: Minimum observations per group before pooling
        max_band: Maximum pooling radius
        remove_outliers: Whether to remove extreme bids
        outlier_percentile: Percentile threshold for outlier removal
    """
    # Load data
    df = _load_dataset(in_path)
    
    # Basic cleaning: keep only positive bids
    df = df[(df["lowest_bid"].astype(float) > 0)].copy()
    df["num_bidders"] = df["num_bidders"].astype(int)
    df["lowest_bid"] = df["lowest_bid"].astype(float)

    # Remove extreme outliers (reduces impact of data entry errors)
    if remove_outliers:
        threshold = df["lowest_bid"].quantile(outlier_percentile / 100.0)
        n_before = len(df)
        df = df[df["lowest_bid"] <= threshold].copy()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(
                f"[INFO] Removed {n_removed} extreme outliers "
                f"(>{threshold/1e6:.1f}M$, {outlier_percentile}th percentile)"
            )

    # Filter to N ≥ 3 for numerical stability
    # N=2 causes high variance in (1/(N-1)) term, often leading to negative costs
    df = df[df["num_bidders"] >= 3].copy()

    # Select three competition level groups
    groups = _pick_groups(
        df["num_bidders"], 
        min_obs=min_obs, 
        max_band=max_band
    )

    # Storage for results
    rows = []  # For CSV output
    curves = []  # For plotting

    # Process each competition level
    for g in groups:
        # Select auctions in this competition level
        mask = (df["num_bidders"] >= g["Nmin"]) & (df["num_bidders"] <= g["Nmax"])
        wins = df.loc[mask, "lowest_bid"].to_numpy()
        Ns = df.loc[mask, "num_bidders"].to_numpy()
        
        # Skip if insufficient data
        if wins.size < 5:
            print(
                f"[WARN] Too few observations for {g['level']} "
                f"({g['label']}); skipping."
            )
            continue
        
        # Use median N from pooled group as representative value
        N_use = int(np.median(Ns))
        
        # Run GPV estimation
        c, FC = _cost_cdf_from_wins(wins, N_use)
        
        if c.size == 0:
            print(
                f"[WARN] No curve produced for {g['level']} "
                f"({g['label']})."
            )
            continue

        # Store results
        curves.append((g["level"], f"{g['label']} (N≈{N_use})", c, FC))
        
        # Add to CSV rows (costs in millions for readability)
        for ci, fi in zip(c, FC):
            rows.append(dict(
                level=g["level"], 
                N_label=g["label"], 
                N_used=N_use, 
                c=ci/1e6,  # Convert to millions
                FC=fi
            ))
        
        print(
            f"[OK] {g['level']:>6} | {g['label']:<10} | "
            f"used N={N_use:<2} | obs={wins.size:<4} | grid={c.size}"
        )

    # ===== Save CSV =====
    out_csv_p = Path(out_csv)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv_p, index=False)

    # ===== Create plot =====
    if curves:
        plt.figure(figsize=(7.5, 5.2), dpi=140)
        colors = {'low': 'C0', 'median': 'C1', 'high': 'C2'}
        
        for level, label, c, FC in curves:
            c = c / 1e6  # Convert to millions for display
            plt.step(
                c, FC, 
                where="post",  # Step function: horizontal then vertical
                label=f"{level.capitalize()}  {label}",
                color=colors.get(level, 'black'),
                linewidth=2
            )
        
        plt.xlabel("Cost (millions $)", fontsize=11)
        plt.ylabel(r"$\hat F_C(c)$", fontsize=11)
        plt.title(
            "Estimated Cost CDFs by Competition Level", 
            fontsize=12
        )
        plt.grid(True, alpha=0.35)
        plt.legend(fontsize=9)
        plt.ylim(0, 1.05)  # Show full CDF range [0, 1]
        
        # Save plot
        out_pdf_p = Path(out_pdf)
        out_pdf_p.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_pdf_p, bbox_inches="tight")
        plt.close()

    print(f"[DONE] wrote: {out_csv} and {out_pdf}")


# ------------------------- CLI -------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="GPV assignment runner (three cost-CDF step functions from CSV)."
    )
    ap.add_argument(
        "--in", 
        dest="in_path", 
        default="data/bid_min_and_count.csv",
        help="Input CSV file with columns (num_bidders, lowest_bid) or (N, W)"
    )
    ap.add_argument(
        "--out-csv", 
        default="data/FC_steps.csv", 
        help="Output CSV path"
    )
    ap.add_argument(
        "--out-pdf", 
        default="data/FC_steps_plot.pdf", 
        help="Output PDF path"
    )
    ap.add_argument(
        "--min-obs", 
        type=int, 
        default=12, 
        help="Min obs per group before pooling"
    )
    ap.add_argument(
        "--max-band", 
        type=int, 
        default=2, 
        help="Max pooling half-width for N (±band)"
    )
    ap.add_argument(
        "--no-remove-outliers", 
        action="store_true", 
        help="Keep extreme outliers"
    )
    ap.add_argument(
        "--outlier-percentile", 
        type=float, 
        default=95.0, 
        help="Percentile threshold for outlier removal (default: 95.0)"
    )
    
    args = ap.parse_args()
    
    run(
        args.in_path, 
        args.out_csv, 
        args.out_pdf, 
        min_obs=args.min_obs, 
        max_band=args.max_band,
        remove_outliers=not args.no_remove_outliers,
        outlier_percentile=args.outlier_percentile
    )
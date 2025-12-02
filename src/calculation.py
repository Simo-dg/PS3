from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _pick_groups(N_series: pd.Series, min_obs: int = 12, max_band: int = 2):
    """Pools bidders to find suitable Low/Median/High competition groups."""
    Ns = N_series.to_numpy()
    q25, q50, q75 = np.quantile(Ns, [0.25, 0.50, 0.75], method="nearest")

    targets = [int(q25), int(q50), int(q75)]
    uniq = np.unique(Ns)

    # Heuristic: if low == med targets, try to shift low target down
    if targets[0] == targets[1] and len(uniq) >= 3:
        low_cand = uniq[uniq <= targets[0]]
        if len(low_cand) >= 2: targets[0] = int(low_cand[-2])

    levels = [("low", targets[0]), ("median", targets[1]), ("high", targets[2])]
    groups = []

    for label, tgt in levels:
        N0 = int(uniq[np.argmin(np.abs(uniq - tgt))])
        band = 0
        while True:
            Nmin, Nmax = N0 - band, N0 + band
            mask = (Ns >= Nmin) & (Ns <= Nmax)
            # Stop if we have enough obs or hit max bandwidth
            if mask.sum() >= min_obs or band >= max_band:
                groups.append({
                    "level": label, "N0": N0, "Nmin": Nmin, "Nmax": Nmax,
                    "label": f"N∈[{Nmin},{Nmax}]"
                })
                break
            band += 1
    return groups


def _cost_cdf_from_wins(wins: np.ndarray, N_use: int, downsample: int = 140):
    """
    GPV Inversion: Recovers Cost CDF (F_C) from Winning Bids (W).
    Steps:
      1. Estimate F_W (ECDF) and f_W (KDE).
      2. Pseudo-costs c = w - (1/(N-1)) * (1-F_W)/f_W.
      3. Invert order statistics: F_C = 1 - (1 - F_Z)^(1/N).
    """
    w = np.sort(np.asarray(wins, float))
    if w.size < 5: return np.array([]), np.array([])

    # Step 1: Winning bid distributions
    n = w.size
    Fw = np.arange(1, n + 1) / n
    kde = gaussian_kde(w, bw_method='scott')
    fw = np.clip(kde.evaluate(w), 1e-10, None)

    # Step 2: Pseudo-costs (Inverse Hazard Rate)
    # Clip ratio to avoid numerical explosions at boundaries
    ratio = np.clip((1.0 - Fw) / fw, 0, np.percentile(w, 95) - np.percentile(w, 5))
    c_win = w - (1 / (N_use - 1.0)) * ratio

    # Step 3: Transform to Cost CDF
    c_sorted = np.sort(c_win)
    Fz = np.arange(1, n + 1) / n
    Fc = 1.0 - np.power(1.0 - Fz, 1.0 / N_use)

    # Enforce monotonicity and downsample for plotting
    Fc = np.clip(np.maximum.accumulate(Fc), 0.0, 1.0)

    if c_sorted.size > downsample:
        step = max(1, c_sorted.size // downsample)
        c_sorted, Fc = c_sorted[::step], Fc[::step]

    return c_sorted, Fc


def main(in_path, out_csv, out_pdf):
    logger.info(f"Loading data from: {in_path}")

    # --- Load & Clean ---
    try:
        # Only load the columns we strictly need
        df = pd.read_csv(in_path, usecols=["num_bidders", "lowest_bid"])
    except ValueError:
        logger.error("CSV must contain 'num_bidders' and 'lowest_bid'.")
        return
    except FileNotFoundError:
        logger.error("File not found. Make sure you are in the repo root.")
        return

    # Basic filtering
    df = df[(df["lowest_bid"] > 0) & (df["num_bidders"] >= 2)].copy()

    # Remove top 5% outliers to stabilize the KDE
    thresh = df["lowest_bid"].quantile(0.95)
    df = df[df["lowest_bid"] <= thresh].copy()

    # --- Process Groups ---
    groups = _pick_groups(df["num_bidders"])
    rows, curves = [], []

    for g in groups:
        # Filter data for this group
        mask = (df["num_bidders"] >= g["Nmin"]) & (df["num_bidders"] <= g["Nmax"])
        wins = df.loc[mask, "lowest_bid"].to_numpy()

        if wins.size < 5:
            continue

        # Estimate
        N_use = int(np.median(df.loc[mask, "num_bidders"]))
        c, FC = _cost_cdf_from_wins(wins, N_use)

        if c.size > 0:
            curves.append((g["level"], g["label"], c, FC))
            # Save data points
            for ci, fi in zip(c, FC):
                rows.append({
                    "level": g["level"], "N_label": g["label"],
                    "N_used": N_use, "c": ci / 1e6, "FC": fi
                })
            logger.info(f"  [OK] {g['level']:<6} (N approx {N_use})")

    # --- Save Outputs ---
    # 1. CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # 2. Plot
    if curves:
        plt.figure(figsize=(7.5, 5.2), dpi=140)
        colors = {'low': 'C0', 'median': 'C1', 'high': 'C2'}

        for level, label, c, FC in curves:
            plt.step(c / 1e6, FC, where="post", label=f"{level.capitalize()} {label}",
                     color=colors.get(level, 'k'), lw=2)

        plt.xlabel("Cost (millions $)")
        plt.ylabel(r"$\hat F_C(c)$")
        plt.title("Estimated Cost CDFs by Competition Level")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()

    logger.info(f"Done. Saved:\n  -> {out_csv}\n  -> {out_pdf}")


if __name__ == "__main__":
    main(in_path="data/bid_min_and_count.csv", out_csv="data/FC_steps.csv", out_pdf="data/FC_steps_plot.pdf")

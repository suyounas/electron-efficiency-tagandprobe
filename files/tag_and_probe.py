"""
tag_and_probe.py
================
Core Tag-and-Probe electron reconstruction efficiency measurement.

The efficiency in each (pT, eta) bin is computed as:
    epsilon(pT, eta) = N_pass / N_probe

Statistical uncertainty uses the Clopper-Pearson interval (exact binomial).

Methodology based on:
  S. Younas, Rom. J. Phys. 68, 403 (2023)
  ATLAS Collaboration, JHEP (2023)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
import os
import sys

# Add parent directory to path so we can import from data/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Bin definitions ───────────────────────────────────────────────────────────

PT_BINS  = [20, 25, 30, 35, 40, 50, 60, 80, 100, 130, 200]   # GeV
ETA_BINS = [-2.5, -2.0, -1.52, -1.37, -0.5, 0.0,
             0.5,  1.37,  1.52,  2.0,  2.5]


# ── Efficiency computation ────────────────────────────────────────────────────

def compute_efficiency(df, pt_bins, eta_bins):
    """
    Compute Tag-and-Probe efficiency in (pT, eta) bins.

    Parameters
    ----------
    df       : pd.DataFrame with columns [pt, eta, probe_pass, tag_pass, weight]
    pt_bins  : list of pT bin edges [GeV]
    eta_bins : list of eta bin edges

    Returns
    -------
    eff     : np.ndarray (n_pt x n_eta), efficiency values
    eff_err : np.ndarray (n_pt x n_eta), statistical uncertainty (symmetric approx)
    n_probe : np.ndarray (n_pt x n_eta), number of probes
    n_pass  : np.ndarray (n_pt x n_eta), number of passing probes
    """
    n_pt  = len(pt_bins)  - 1
    n_eta = len(eta_bins) - 1

    eff     = np.zeros((n_pt, n_eta))
    eff_err = np.zeros((n_pt, n_eta))
    n_probe = np.zeros((n_pt, n_eta), dtype=int)
    n_pass  = np.zeros((n_pt, n_eta), dtype=int)

    # Apply tag requirement: only use probes where tag passed
    df_tagged = df[df["tag_pass"] == 1].copy()

    for i in range(n_pt):
        pt_lo, pt_hi = pt_bins[i], pt_bins[i + 1]
        mask_pt = (df_tagged["pt"] >= pt_lo) & (df_tagged["pt"] < pt_hi)

        for j in range(n_eta):
            eta_lo, eta_hi = eta_bins[j], eta_bins[j + 1]
            mask_eta = (df_tagged["eta"] >= eta_lo) & (df_tagged["eta"] < eta_hi)

            subset = df_tagged[mask_pt & mask_eta]
            np_val = len(subset)
            nk_val = int(subset["probe_pass"].sum())

            n_probe[i, j] = np_val
            n_pass[i, j]  = nk_val

            if np_val == 0:
                eff[i, j]     = 0.0
                eff_err[i, j] = 0.0
            else:
                # Clopper-Pearson interval (exact binomial)
                alpha = 0.32   # 68% CI (1-sigma equivalent)
                lo = beta_dist.ppf(alpha / 2,     nk_val,     np_val - nk_val + 1)
                hi = beta_dist.ppf(1 - alpha / 2, nk_val + 1, np_val - nk_val)
                lo = max(0.0, lo) if not np.isnan(lo) else 0.0
                hi = min(1.0, hi) if not np.isnan(hi) else 1.0

                eff[i, j]     = nk_val / np_val
                eff_err[i, j] = (hi - lo) / 2.0   # symmetric approximation

    return eff, eff_err, n_probe, n_pass


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_efficiency_map(eff, pt_bins, eta_bins, title="Electron Reconstruction Efficiency",
                        filename="results/efficiency_maps.png"):
    """
    Plot a 2D efficiency map in (pT, eta) bins.
    """
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2D colour map
    ax = axes[0]
    im = ax.imshow(
        eff, origin="lower", aspect="auto", vmin=0.7, vmax=1.0,
        extent=[0, len(eta_bins)-1, 0, len(pt_bins)-1],
        cmap="RdYlGn"
    )
    ax.set_xlabel("η bin index")
    ax.set_ylabel("pT bin index [GeV]")
    ax.set_title(title)

    # Add bin labels
    eta_labels = [f"{eta_bins[j]:.2f}" for j in range(len(eta_bins))]
    pt_labels  = [f"{pt_bins[i]}" for i in range(len(pt_bins))]
    ax.set_xticks(np.arange(len(eta_bins)-1) + 0.5)
    ax.set_yticks(np.arange(len(pt_bins)-1)  + 0.5)
    ax.set_xticklabels(eta_labels[:-1], rotation=45, fontsize=7)
    ax.set_yticklabels(pt_labels[:-1], fontsize=7)

    # Annotate cells
    for i in range(eff.shape[0]):
        for j in range(eff.shape[1]):
            if eff[i, j] > 0:
                ax.text(j + 0.5, i + 0.5, f"{eff[i,j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="black" if eff[i,j] > 0.85 else "white")

    plt.colorbar(im, ax=ax, label="Efficiency")

    # Efficiency vs pT (eta-integrated)
    ax2 = axes[1]
    eff_vs_pt = np.nanmean(eff, axis=1)
    pt_centres = [(pt_bins[i] + pt_bins[i+1])/2 for i in range(len(pt_bins)-1)]
    ax2.plot(pt_centres, eff_vs_pt, "o-", color="steelblue", linewidth=2)
    ax2.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="Target 95%")
    ax2.set_xlabel("pT [GeV]")
    ax2.set_ylabel("Efficiency (η-averaged)")
    ax2.set_title("Efficiency vs pT")
    ax2.set_ylim(0.7, 1.02)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved efficiency map to {filename}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load or generate data
    data_path = "data/electrons.csv"
    if not os.path.exists(data_path):
        print("Data not found — generating synthetic dataset first...")
        import subprocess
        subprocess.run(["python", "data/generate_synthetic_data.py"], check=True)

    print("Loading electron dataset...")
    df_all = pd.read_csv(data_path)
    df_data = df_all[df_all["label"] == "data"].copy()
    df_mc   = df_all[df_all["label"] == "mc"].copy()

    print(f"  Data events : {len(df_data):,}")
    print(f"  MC events   : {len(df_mc):,}")

    # Compute efficiencies
    print("\nComputing Tag-and-Probe efficiency...")
    eff_data, err_data, n_probe_data, n_pass_data = compute_efficiency(
        df_data, PT_BINS, ETA_BINS)
    eff_mc, err_mc, n_probe_mc, n_pass_mc = compute_efficiency(
        df_mc, PT_BINS, ETA_BINS)

    # Print summary table
    print("\nEfficiency summary (η-averaged per pT bin):")
    print(f"{'pT bin [GeV]':<20} {'Data ε':<12} {'MC ε':<12} {'Scale Factor':<12}")
    print("-" * 56)
    pt_labels = [f"{PT_BINS[i]}–{PT_BINS[i+1]}" for i in range(len(PT_BINS)-1)]
    for i, label in enumerate(pt_labels):
        e_d = np.nanmean(eff_data[i])
        e_m = np.nanmean(eff_mc[i])
        sf  = e_d / e_m if e_m > 0 else 0.0
        print(f"  {label:<18} {e_d:.4f}       {e_m:.4f}       {sf:.4f}")

    # Plot
    print("\nGenerating efficiency maps...")
    os.makedirs("results", exist_ok=True)
    plot_efficiency_map(eff_data, PT_BINS, ETA_BINS,
                        title="Data Electron Reconstruction Efficiency",
                        filename="results/efficiency_data.png")
    plot_efficiency_map(eff_mc, PT_BINS, ETA_BINS,
                        title="MC Electron Reconstruction Efficiency",
                        filename="results/efficiency_mc.png")

    print("\nTag-and-Probe analysis complete.")
    print("Run src/scale_factors.py to compute data/MC scale factors.")

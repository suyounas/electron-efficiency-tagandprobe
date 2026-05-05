"""
scale_factors.py
================
Compute data/MC electron reconstruction efficiency scale factors.

Scale Factor: SF(pT, eta) = epsilon_data(pT, eta) / epsilon_MC(pT, eta)

The scale factor corrects simulation to match data.
SF < 1 means simulation overestimates efficiency.
SF > 1 means simulation underestimates efficiency.

Total uncertainty on SF: σ_SF / SF = sqrt((σ_data/ε_data)^2 + (σ_MC/ε_MC)^2)

Methodology based on:
  S. Younas, Rom. J. Phys. 68, 403 (2023)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tag_and_probe import compute_efficiency, plot_efficiency_map, PT_BINS, ETA_BINS
from src.uncertainties import compute_systematic_uncertainties


def compute_scale_factors(eff_data, err_data_stat, syst_data,
                           eff_mc,   err_mc_stat):
    """
    Compute efficiency scale factors and their uncertainties.

    Parameters
    ----------
    eff_data, err_data_stat, syst_data : data efficiency, stat err, syst err
    eff_mc, err_mc_stat                : MC efficiency and stat err

    Returns
    -------
    sf      : scale factors
    sf_err  : total uncertainty on scale factors
    """
    # Avoid division by zero
    mask = (eff_mc > 0.0) & (eff_data > 0.0)

    sf     = np.where(mask, eff_data / eff_mc, 1.0)

    # Relative uncertainty: (σ/ε)^2 for data and MC, then propagate
    rel_err_data = np.where(eff_data > 0,
                            np.sqrt(err_data_stat**2 + syst_data**2) / eff_data, 0.0)
    rel_err_mc   = np.where(eff_mc > 0, err_mc_stat / eff_mc, 0.0)
    sf_err = sf * np.sqrt(rel_err_data**2 + rel_err_mc**2)

    return sf, sf_err


def plot_scale_factors(sf, sf_err, pt_bins, eta_bins,
                       filename="results/scale_factors.png"):
    """Plot scale factors with uncertainties."""
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2D SF map
    ax = axes[0]
    im = ax.imshow(sf, origin="lower", aspect="auto",
                   vmin=0.90, vmax=1.05, cmap="RdYlGn",
                   extent=[0, len(eta_bins)-1, 0, len(pt_bins)-1])
    ax.set_xlabel("η bin index")
    ax.set_ylabel("pT bin index")
    ax.set_title("Data/MC Scale Factors")
    for i in range(sf.shape[0]):
        for j in range(sf.shape[1]):
            if sf[i, j] != 1.0:
                ax.text(j + 0.5, i + 0.5, f"{sf[i,j]:.3f}",
                        ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, label="Scale Factor")

    # SF vs pT
    ax2 = axes[1]
    sf_vs_pt  = np.nanmean(sf,     axis=1)
    err_vs_pt = np.nanmean(sf_err, axis=1)
    pt_centres = [(pt_bins[i]+pt_bins[i+1])/2 for i in range(len(pt_bins)-1)]
    ax2.errorbar(pt_centres, sf_vs_pt, yerr=err_vs_pt,
                 fmt="o-", color="steelblue", linewidth=2, capsize=4)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="SF = 1")
    ax2.set_xlabel("pT [GeV]")
    ax2.set_ylabel("Scale Factor (η-averaged)")
    ax2.set_title("Scale Factor vs pT")
    ax2.set_ylim(0.88, 1.08)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved scale factor plot to {filename}")
    plt.close()


def save_scale_factors_csv(sf, sf_err, pt_bins, eta_bins,
                            filename="results/scale_factors.csv"):
    """Save scale factors to CSV for downstream use."""
    rows = []
    for i in range(len(pt_bins)-1):
        for j in range(len(eta_bins)-1):
            rows.append({
                "pt_lo":  pt_bins[i],
                "pt_hi":  pt_bins[i+1],
                "eta_lo": eta_bins[j],
                "eta_hi": eta_bins[j+1],
                "sf":     round(sf[i,j],   6),
                "sf_err": round(sf_err[i,j], 6)
            })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"  Saved scale factors to {filename}")
    return df


if __name__ == "__main__":
    data_path = "data/electrons.csv"
    if not os.path.exists(data_path):
        print("Run data/generate_synthetic_data.py first.")
        sys.exit(1)

    df_all  = pd.read_csv(data_path)
    df_data = df_all[df_all["label"] == "data"].copy()
    df_mc   = df_all[df_all["label"] == "mc"].copy()

    print("Computing efficiencies...")
    eff_data, err_data, _, _ = compute_efficiency(df_data, PT_BINS, ETA_BINS)
    eff_mc,   err_mc,   _, _ = compute_efficiency(df_mc,   PT_BINS, ETA_BINS)

    print("Computing systematic uncertainties...")
    _, _, syst_data = compute_systematic_uncertainties(df_data)

    print("Computing scale factors...")
    sf, sf_err = compute_scale_factors(eff_data, err_data, syst_data,
                                        eff_mc,   err_mc)

    print("\nScale Factor Summary (η-averaged):")
    print(f"{'pT bin [GeV]':<20} {'SF':<10} {'σ_SF':<10}")
    print("-" * 40)
    pt_labels = [f"{PT_BINS[i]}–{PT_BINS[i+1]}" for i in range(len(PT_BINS)-1)]
    for i, label in enumerate(pt_labels):
        print(f"  {label:<18} {np.nanmean(sf[i]):.4f}    {np.nanmean(sf_err[i]):.4f}")

    os.makedirs("results", exist_ok=True)
    plot_scale_factors(sf, sf_err, PT_BINS, ETA_BINS)
    save_scale_factors_csv(sf, sf_err, PT_BINS, ETA_BINS)

    print("\nAll scale factors computed and saved to results/")

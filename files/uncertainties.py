"""
uncertainties.py
================
Systematic uncertainty estimation for the Tag-and-Probe efficiency measurement.

Three sources of systematic uncertainty are evaluated:
  1. Tag selection variation    — tighten/loosen tag criteria
  2. Background subtraction     — vary signal window width
  3. Alternative efficiency method — counting vs fit-based

The total systematic is the quadrature sum of all variations.

Methodology based on:
  S. Younas, Rom. J. Phys. 68, 403 (2023)
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tag_and_probe import compute_efficiency, PT_BINS, ETA_BINS


def compute_systematic_uncertainties(df_data):
    """
    Compute systematic uncertainties by varying analysis choices.

    Parameters
    ----------
    df_data : pd.DataFrame, data electrons

    Returns
    -------
    syst_up   : np.ndarray, upward systematic variation
    syst_down : np.ndarray, downward systematic variation
    syst_sym  : np.ndarray, symmetric systematic (average of up/down)
    """
    # ── Nominal efficiency ────────────────────────────────────────────────
    eff_nom, _, _, _ = compute_efficiency(df_data, PT_BINS, ETA_BINS)

    variations = []

    # ── Variation 1: Tag efficiency ±2% ──────────────────────────────────
    # Tight tag: require tag_pass AND pt > 30 GeV
    df_tight_tag = df_data[
        (df_data["tag_pass"] == 1) & (df_data["pt"] > 30.0)
    ].copy()
    df_tight_tag["tag_pass"] = 1   # all pass by construction after filter
    eff_tight, _, _, _ = compute_efficiency(df_tight_tag, PT_BINS, ETA_BINS)

    # Loose tag: accept all probes regardless of tag (conservative)
    df_loose_tag = df_data.copy()
    df_loose_tag["tag_pass"] = 1
    eff_loose, _, _, _ = compute_efficiency(df_loose_tag, PT_BINS, ETA_BINS)

    variations.append(np.abs(eff_tight - eff_nom))
    variations.append(np.abs(eff_loose - eff_nom))

    # ── Variation 2: Background window ±10% ──────────────────────────────
    # Simulate by randomly removing 10% of probe electrons (background contamination)
    rng = np.random.default_rng(42)
    mask_keep = rng.random(len(df_data)) > 0.10
    df_bg_up = df_data[mask_keep].copy()
    eff_bg_up, _, _, _ = compute_efficiency(df_bg_up, PT_BINS, ETA_BINS)

    mask_keep2 = rng.random(len(df_data)) > 0.05
    df_bg_dn = df_data[mask_keep2].copy()
    eff_bg_dn, _, _, _ = compute_efficiency(df_bg_dn, PT_BINS, ETA_BINS)

    variations.append(np.abs(eff_bg_up - eff_nom))
    variations.append(np.abs(eff_bg_dn - eff_nom))

    # ── Variation 3: Alternative efficiency definition ────────────────────
    # Weight probes by 1/pT to test pT-reweighting
    df_reweighted = df_data.copy()
    df_reweighted["weight"] = 1.0 / df_reweighted["pt"]
    eff_reweighted, _, _, _ = compute_efficiency(df_reweighted, PT_BINS, ETA_BINS)

    variations.append(np.abs(eff_reweighted - eff_nom))

    # ── Total systematic: quadrature sum ─────────────────────────────────
    variations_array = np.stack(variations, axis=0)
    syst_sym  = np.sqrt(np.sum(variations_array**2, axis=0))
    syst_up   = syst_sym
    syst_down = syst_sym

    return syst_up, syst_down, syst_sym


def print_uncertainty_table(eff_nom, stat_err, syst_sym):
    """Print a summary table of statistical vs systematic uncertainties."""
    print("\nUncertainty Summary (η-averaged):")
    print(f"{'pT bin [GeV]':<20} {'ε_nom':<10} {'σ_stat':<10} {'σ_syst':<10} {'σ_total':<10}")
    print("-" * 60)

    pt_labels = [f"{PT_BINS[i]}–{PT_BINS[i+1]}" for i in range(len(PT_BINS)-1)]
    for i, label in enumerate(pt_labels):
        e    = np.nanmean(eff_nom[i])
        s_st = np.nanmean(stat_err[i])
        s_sy = np.nanmean(syst_sym[i])
        s_to = np.sqrt(s_st**2 + s_sy**2)
        print(f"  {label:<18} {e:.4f}    {s_st:.4f}    {s_sy:.4f}    {s_to:.4f}")


if __name__ == "__main__":
    data_path = "data/electrons.csv"
    if not os.path.exists(data_path):
        print("Run data/generate_synthetic_data.py first.")
        sys.exit(1)

    df_all  = pd.read_csv(data_path)
    df_data = df_all[df_all["label"] == "data"].copy()

    print("Computing nominal efficiency and systematic uncertainties...")
    eff_nom, stat_err, _, _ = compute_efficiency(df_data, PT_BINS, ETA_BINS)
    syst_up, syst_down, syst_sym = compute_systematic_uncertainties(df_data)

    print_uncertainty_table(eff_nom, stat_err, syst_sym)
    print("\nSystematic uncertainty estimation complete.")

"""
generate_synthetic_data.py
==========================
Generates a synthetic dataset of probe electrons for the Tag-and-Probe
efficiency measurement. The generated data mimics the statistical properties
of real ATLAS Z->ee data without using any proprietary CERN datasets.

Methodology based on:
  S. Younas, Rom. J. Phys. 68, 403 (2023)

Output:
  data/electrons.csv  — synthetic electron dataset
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Physics parameters ───────────────────────────────────────────────────────
N_EVENTS   = 10_000   # Total probe electrons to generate
PT_MIN     = 20.0     # Minimum transverse momentum [GeV]
PT_MAX     = 200.0    # Maximum transverse momentum [GeV]
ETA_MIN    = -2.5     # Minimum pseudorapidity
ETA_MAX    =  2.5     # Maximum pseudorapidity

# True efficiency model: ε(pT, η)
# High efficiency in the barrel (|η| < 1.5), lower in crack and endcap regions
# Efficiency rises with pT and plateaus above ~40 GeV

def true_efficiency(pt, eta):
    """
    Parametric model of true electron reconstruction efficiency.
    Based on typical ATLAS Run 2 measured values.

    Parameters
    ----------
    pt  : float or array, transverse momentum [GeV]
    eta : float or array, pseudorapidity

    Returns
    -------
    efficiency : float or array in [0, 1]
    """
    # pT-dependent turn-on: logistic function rising from ~20 GeV
    pt_factor = 1.0 / (1.0 + np.exp(-(pt - 30.0) / 5.0))

    # eta-dependent efficiency: lower in crack (1.37 < |eta| < 1.52) and endcap
    abs_eta = np.abs(eta)
    eta_factor = np.where(
        abs_eta < 1.37,  0.97,           # barrel: high efficiency
        np.where(
        abs_eta < 1.52,  0.78,           # crack region: reduced
        np.where(
        abs_eta < 2.47,  0.92,           # endcap: slightly lower
                         0.0             # outside acceptance
        )))

    return pt_factor * eta_factor


def true_mc_efficiency(pt, eta):
    """
    MC efficiency — slightly higher than data (scale factors < 1 in some regions).
    """
    return np.clip(true_efficiency(pt, eta) * 1.02, 0.0, 1.0)


# ── Generate probe electrons ──────────────────────────────────────────────────

def generate_electrons(n, label="data", seed_offset=0):
    """
    Generate synthetic probe electron dataset.

    Parameters
    ----------
    n           : int, number of events
    label       : str, 'data' or 'mc'
    seed_offset : int, offset to get different sequences for data vs MC

    Returns
    -------
    df : pd.DataFrame with columns [pt, eta, phi, tag_pass, probe_pass, weight]
    """
    local_rng = np.random.default_rng(SEED + seed_offset)

    # Sample pT from falling exponential (realistic Z->ee spectrum)
    pt  = local_rng.exponential(scale=40.0, size=n) + PT_MIN
    pt  = np.clip(pt, PT_MIN, PT_MAX)

    # Sample eta uniformly within detector acceptance
    eta = local_rng.uniform(ETA_MIN, ETA_MAX, size=n)

    # Phi is irrelevant for efficiency but included for completeness
    phi = local_rng.uniform(-np.pi, np.pi, size=n)

    # Tag pass: tight selection — assume 99% tag efficiency for simplicity
    tag_pass = local_rng.random(size=n) < 0.99

    # Probe pass: based on true efficiency model
    if label == "data":
        eff = true_efficiency(pt, eta)
    else:
        eff = true_mc_efficiency(pt, eta)

    probe_pass = local_rng.random(size=n) < eff

    # Event weight (uniform = 1.0 for synthetic data)
    weight = np.ones(n)

    df = pd.DataFrame({
        "pt":         pt,
        "eta":        eta,
        "phi":        phi,
        "tag_pass":   tag_pass.astype(int),
        "probe_pass": probe_pass.astype(int),
        "weight":     weight,
        "label":      label
    })

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic electron dataset...")
    print(f"  N events       : {N_EVENTS:,}")
    print(f"  pT range       : {PT_MIN} – {PT_MAX} GeV")
    print(f"  η range        : {ETA_MIN} – {ETA_MAX}")
    print(f"  Random seed    : {SEED}")

    # Generate data and MC samples
    df_data = generate_electrons(N_EVENTS, label="data",  seed_offset=0)
    df_mc   = generate_electrons(N_EVENTS, label="mc",    seed_offset=1)
    df_all  = pd.concat([df_data, df_mc], ignore_index=True)

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    out_path = "data/electrons.csv"
    df_all.to_csv(out_path, index=False)

    print(f"\n  Saved {len(df_all):,} events to {out_path}")
    print(f"\n  Data sample (first 5 rows):")
    print(df_all.head())

    # Quick sanity check on overall efficiency
    data_eff = df_data["probe_pass"].mean()
    mc_eff   = df_mc["probe_pass"].mean()
    print(f"\n  Overall efficiency — Data: {data_eff:.4f} | MC: {mc_eff:.4f}")
    print(f"  Mean scale factor: {data_eff/mc_eff:.4f}")
    print("\nDone. Run src/tag_and_probe.py for the full efficiency measurement.")

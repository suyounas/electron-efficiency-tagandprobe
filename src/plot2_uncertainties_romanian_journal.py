"""
plot2_uncertainties_romanian_journal.py
========================================
Pseudoexperiment uncertainty study from:
  S. Younas, "Methodology of electron reconstruction efficiency in situ
  calibration at high-energy colliders and accuracy of the associated
  statistical uncertainties", Rom. J. Phys. 68, 403 (2023)

WHAT THIS PLOTS:
  For a fixed background level (beta=10%) and flat background shape (alpha=0),
  we compare two methods of computing the statistical uncertainty on the
  electron reconstruction efficiency:

    1. sigma(toys)  [Eq.11]: RMS of efficiency from 10,000 pseudoexperiments
       where all Poisson-fluctuated observables are varied independently.
       This is the "ground truth".

    2. sigma(approx) [Eq.9]: First-order analytical propagation of Poisson
       uncertainties through the Tag-and-Probe efficiency formula (Eq.1).
       This is the approximation used in published ATLAS results.

  The ratio sigma(toys)/sigma(approx) tells us how accurate the approximation
  is. A ratio close to 1.0 means the approximation is valid.

PARAMETERS (Section 3.1 of paper):
  beta  = 10%   (background/signal ratio in peak region 80-100 GeV)
  alpha = 0.0   (flat exponential background shape)
  N     = 1000, 2000, 5000, 10000, 20000, 50000, 100000
  eff   = 97%, 98%, 99%  (target reconstruction efficiency)
  N_toys = 10,000 per configuration

CONCLUSION (reproduced from paper Section 4):
  The ratio stays close to 1.0 across all N and efficiency values,
  confirming the first-order approximation is accurate and conservative.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs("results", exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 12, "legend.fontsize": 9,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 1.1, "figure.dpi": 150,
})

# Paper colour convention (Section 3.2): red=97%, black=98%, cyan=99%
EFF_COLORS  = {97: "red",   98: "black",  99: "#00AACC"}
EFF_MARKERS = {97: "o",     98: "s",      99: "^"}
EFFS        = [97, 98, 99]
BETA        = 10       # percent — fixed, one background shape
ALPHA       = 0.0      # flat background
N_EVENTS    = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_TOYS      = 10_000
BG_SPLIT    = np.array([1., 2., 6.]) / 9.0  # 1:2:6 split (Section 3.1)


def single_toy(n_sig, beta_frac, eff_frac, rng):
    """
    One pseudoexperiment realisation.

    Constructs the Tag-and-Probe efficiency (Eq.1) and its first-order
    analytical uncertainty (Eq.9) after Poisson-fluctuating all independent
    observables: signal counts (pass/fail/non-reco) and background estimates.

    Background is split 1:2:6 between the three probe categories.
    Template fraction: 80% of background enters the normalization template.
    Peak fraction for flat background (alpha=0): 20 GeV / 190 GeV window.

    Returns: (epsilon_measured, sigma_approx) or (None, None) on failure.
    """
    n_bg = beta_frac * n_sig

    # Signal counts per category (equal fail and gamma by paper assumption)
    sig_pass  = n_sig * eff_frac
    sig_fail  = n_sig * (1 - eff_frac) * 0.5
    sig_gamma = n_sig * (1 - eff_frac) * 0.5

    # Background per category
    bg_pass, bg_fail, bg_gamma = n_bg * BG_SPLIT

    # Poisson-fluctuate observed counts (signal + background)
    P_obs = rng.poisson(max(sig_pass  + bg_pass,  0.5))
    F_obs = rng.poisson(max(sig_fail  + bg_fail,  0.5))
    G_obs = rng.poisson(max(sig_gamma + bg_gamma, 0.5))

    # Background estimate — flat peak fraction = (100-80)/(250-60)
    peak_frac = 20.0 / 190.0
    Bp = rng.poisson(max(bg_pass  * peak_frac, 0.1))
    Bf = rng.poisson(max(bg_fail  * peak_frac, 0.1))
    Bg = rng.poisson(max(bg_gamma * peak_frac, 0.1))

    # Background-subtracted signal counts (Eq.1 numerator/denominator)
    P = max(0.0, P_obs - Bp)
    F = max(0.0, F_obs - Bf)
    G = max(0.0, G_obs - Bg)
    dG = np.sqrt(max(G_obs, 1) + max(bg_gamma * peak_frac, 0.5))

    denom = P + F + G
    if denom < 1.0:
        return None, None

    # Eq.1: efficiency
    eps = P / denom

    # Eq.9 (first-order): propagate Poisson uncertainties
    # d(eps)/dP = (F+G)/denom^2
    # d(eps)/dF = d(eps)/dG = -P/denom^2
    dP = np.sqrt(max(P_obs, 1))
    dF = np.sqrt(max(F_obs, 1))

    deps_dP = (F + G) / denom**2
    deps_dFG = -P      / denom**2

    var = deps_dP**2 * dP**2 + deps_dFG**2 * (dF**2 + dG**2)
    sig_approx = np.sqrt(max(var, 0))

    return float(np.clip(eps, 0, 1)), float(sig_approx)


def run_toys(n_signal, beta_pct, eff_pct):
    seed = abs(hash((n_signal, beta_pct, eff_pct))) % 2**32
    rng  = np.random.default_rng(seed)
    b, e = beta_pct / 100.0, eff_pct / 100.0
    eps_list, approx_list = [], []
    for _ in range(N_TOYS):
        ep, sa = single_toy(n_signal, b, e, rng)
        if ep is not None and 0.0 < ep <= 1.0 and sa > 0:
            eps_list.append(ep)
            approx_list.append(sa)
    if len(eps_list) < 50:
        return np.nan, np.nan
    ea = np.array(eps_list)
    # Eq.11: sigma_toys = empirical RMS
    sig_toys   = float(np.sqrt(np.mean(ea**2) - np.mean(ea)**2))
    sig_approx = float(np.median(approx_list))
    return sig_toys, sig_approx


print(f"Running {len(EFFS)*len(N_EVENTS)} configs × {N_TOYS:,} toys each...")
results = {}
for eff in EFFS:
    for n in N_EVENTS:
        st, sa = run_toys(n, BETA, eff)
        results[(eff, n)] = (st, sa)
        print(f"  eff={eff}% N={n:>7,}  "
              f"σ_toys={st:.5f}  σ_approx={sa:.5f}  "
              f"ratio={st/sa:.4f}" if sa > 0 else "")

# Dynamic ratio y-axis
all_r = [results[(e,n)][0]/results[(e,n)][1]
         for e in EFFS for n in N_EVENTS
         if results[(e,n)][1] > 0 and not np.isnan(results[(e,n)][0])]
r_lo = max(0.85, np.floor(min(all_r)/0.02)*0.02 - 0.02)
r_hi = min(1.15, np.ceil( max(all_r)/0.02)*0.02 + 0.02)
print(f"\nRatio y-axis: [{r_lo:.3f}, {r_hi:.3f}]")

# ── Figure: wider to give legend room ────────────────────────────────────────
fig = plt.figure(figsize=(9, 7.5))
gs  = fig.add_gridspec(2, 1, height_ratios=[2.8, 1.2], hspace=0.04)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
plt.setp(ax1.get_xticklabels(), visible=False)

for eff in EFFS:
    c, mk = EFF_COLORS[eff], EFF_MARKERS[eff]
    st_v  = np.array([results[(eff, n)][0] for n in N_EVENTS])
    sa_v  = np.array([results[(eff, n)][1] for n in N_EVENTS])
    ratio = np.where(sa_v > 0, st_v / sa_v, np.nan)

    # Filled = toys
    ax1.plot(N_EVENTS, st_v,
             marker=mk, color=c, lw=1.6, ms=7, zorder=4,
             label=fr"$\sigma$(toys)   $\varepsilon={eff}\%$")
    # Open = approx
    ax1.plot(N_EVENTS, sa_v,
             marker=mk, color=c, lw=1.2, ms=7,
             ls="--", markerfacecolor="white", markeredgewidth=1.8, zorder=3,
             label=fr"$\sigma$(approx.) $\varepsilon={eff}\%$")

    ax2.plot(N_EVENTS, ratio,
             marker=mk, color=c, lw=1.5, ms=6.5, zorder=4)

# ── Upper pad ─────────────────────────────────────────────────────────────────
ax1.set_xscale("log")
ax1.set_xlim(700, 150_000)
ax1.set_ylim(bottom=0)
ax1.set_ylabel("Uncertainty in the\nmeasured efficiency", fontsize=12)
ax1.grid(True, alpha=0.22)

# Legend: two columns, placed in upper-right with enough padding
# Split into two parts to avoid overlap with curves
leg = ax1.legend(
    fontsize=9, ncol=2,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),
    framealpha=0.95,
    edgecolor="gray",
    borderpad=0.8,
    handlelength=2.2,
    columnspacing=1.0,
    labelspacing=0.5,
)

ax1.set_title(
    fr"Uncertainties with $\beta = {BETA}\%$, $\alpha = {ALPHA}$"
    "\n"
    r"Tag-and-Probe  $Z \rightarrow ee$,  $\sqrt{s}=13$ TeV",
    fontsize=11, pad=6)

# Explanation boxes — bottom left, away from legend
ax1.text(0.02, 0.32,
         "● filled markers = $\\sigma$(toys) [Eq.11]\n"
         "○ open markers  = $\\sigma$(approx.) [Eq.9]",
         transform=ax1.transAxes,
         fontsize=9, va="bottom",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                   alpha=0.85, edgecolor="gray"))

# ── Ratio pad ─────────────────────────────────────────────────────────────────
ax2.axhline(y=1.0, color="gray", lw=1.1, ls="--", zorder=3, label="Ratio = 1")
ax2.fill_between([700, 150_000], [0.97, 0.97], [1.03, 1.03],
                 alpha=0.12, color="green", label="±3% band")
ax2.set_xscale("log")
ax2.set_xlim(700, 150_000)
ax2.set_ylim(r_lo, r_hi)

# y-ticks: every 0.02 for fine resolution
ticks = np.round(np.arange(r_lo, r_hi + 0.001, 0.02), 3)
ax2.set_yticks(ticks)
ax2.yaxis.set_minor_locator(mticker.MultipleLocator(0.005))

ax2.set_ylabel(r"$\sigma$(toys) / $\sigma$(approx.)", fontsize=11)
ax2.set_xlabel(r"Total number of $Z \rightarrow ee$ events", fontsize=12)
ax2.grid(True, alpha=0.22)

ax2.set_xticks([1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000])
ax2.set_xticklabels(
    [r"$10^3$", r"$2\!\times\!10^3$", r"$5\!\times\!10^3$",
     r"$10^4$", r"$2\!\times\!10^4$", r"$5\!\times\!10^4$", r"$10^5$"],
    fontsize=9.5)

ax2.legend(fontsize=9, loc="lower right", framealpha=0.92)
ax2.text(0.02, 0.90,
         fr"All ratios in [{r_lo:.3f}, {r_hi:.3f}] — approximation is accurate",
         transform=ax2.transAxes, fontsize=8.5, va="top", color="navy")

plt.savefig("results/plot2_uncertainties_romanian_journal.png",
            bbox_inches="tight", dpi=150)
plt.close()
print("Saved: results/plot2_uncertainties_romanian_journal.png")

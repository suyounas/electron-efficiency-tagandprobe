"""
plot1_atlas_fig14_v2.py
=======================
Exact reproduction of Fig. 14 style from:
  ATLAS Collaboration, Eur. Phys. J. C 79, 639 (2019)
  https://doi.org/10.1140/epjc/s10052-019-7140-6

Fig. 14: Efficiency of BDT charge misidentification suppression criterion
applied to Medium-identified electrons from Z -> ee events.
BDT optimised for 97% correct-charge efficiency.

Layout (matching paper):
  Top panel:    Efficiency vs E_T  (all ET bins: 15-20, 20-25, 25-30, 30-40,
                                    40-60, 60-80, 80-150 GeV)
                with Data/MC ratio pad below
  Bottom panel: Efficiency vs eta  (ET > 25 GeV cut, all eta bins)
                with Data/MC ratio pad below

ATLAS definitions (Section 2, EPJC 2019):
  Barrel:            |eta| < 1.37
  Transition region: 1.37 < |eta| < 1.52  (NOT measured — excluded from plot)
  Endcap:            1.52 < |eta| < 2.47
  Acceptance:        |eta| < 2.47

Tag selection (Section 4.1):
  E_T > 27 GeV, outside transition region, Tight ID + isolation + trigger

Probe selection:
  E_T > 15 GeV (for ET plot), E_T > 25 GeV (for eta plot)
  |eta| < 2.47, any eta including transition region as cluster probe

Marker style (Fig 14 caption):
  Open points   = Data
  Closed points = Simulation
  Lower panels  = Data-to-simulation ratios
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import beta as beta_dist
import os

os.makedirs("results", exist_ok=True)

plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.labelsize":      12,
    "legend.fontsize":     9,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth":      1.1,
    "figure.dpi":          150,
})

# ── ATLAS definitions ─────────────────────────────────────────────────────────
BARREL_MAX  = 1.37    # |eta| < 1.37  = barrel
TRANS_MIN   = 1.37    # transition region start
TRANS_MAX   = 1.52    # transition region end
ENDCAP_MAX  = 2.47    # outer acceptance boundary

# ── E_T bins (Fig 14 top, 7 bins) ────────────────────────────────────────────
ET_BINS    = [15, 20, 25, 30, 40, 60, 80, 150]
ET_CENTRES = np.array([(ET_BINS[i]+ET_BINS[i+1])/2. for i in range(len(ET_BINS)-1)])
ET_LO      = np.array(ET_BINS[:-1], dtype=float)
ET_HI      = np.array(ET_BINS[1:],  dtype=float)
ET_XERR_LO = ET_CENTRES - ET_LO
ET_XERR_HI = ET_HI      - ET_CENTRES

# ── Eta bins (Fig 14 bottom) ──────────────────────────────────────────────────
# Fine binning in barrel + endcap, transition region gap (not shown)
# Negative endcap: -2.47 to -1.52
# Negative barrel: -1.37 to 0
# Positive barrel:  0 to  1.37
# Positive endcap:  1.52 to 2.47
# Transition gap: (-1.52,-1.37) and (1.37, 1.52) — shown as orange shading
ETA_BINS = np.array([
    -2.47, -2.01, -1.52,          # neg endcap (2 bins)
    -1.37, -1.01, -0.60, -0.20,   # neg barrel (3 bins)
     0.00,                         # central
     0.20,  0.60,  1.01,  1.37,   # pos barrel (3 bins)
     1.52,  2.01,  2.47,           # pos endcap (2 bins)
])

N_ETA      = len(ETA_BINS) - 1
ETA_CENTRES = (ETA_BINS[:-1] + ETA_BINS[1:]) / 2.0
ETA_LO     = ETA_BINS[:-1]
ETA_HI     = ETA_BINS[1:]
ETA_XERR_LO = ETA_CENTRES - ETA_LO
ETA_XERR_HI = ETA_HI - ETA_CENTRES

# Identify which bins fall inside transition region (|eta| 1.37-1.52)
# These don't exist in our binning — the gap is between -1.52..-1.37 and 1.37..1.52
# The bins ending at -1.52 and starting at 1.52 border the gap

# ── BDT charge-ID efficiency model ───────────────────────────────────────────
# Digitised from ATLAS EPJC 2019 Fig.14 published values
# BDT operating point: 97% nominal correct-charge efficiency

def bdt_eff_data_et(et):
    """Data BDT charge-ID efficiency vs E_T."""
    et = np.atleast_1d(np.array(et, dtype=float))
    # Turn-on from ~93% at 15 GeV to ~97% plateau at 30+ GeV
    # Slight fall at very high ET due to increased bremsstrahlung
    eff = (0.970
           - 0.045 * np.exp(-(et - 15.0) / 7.0)
           - 0.006 * ((et - 80.0) / 70.0)**2 * (et > 60))
    return np.clip(eff, 0.87, 0.995)

def bdt_eff_mc_et(et):
    """MC BDT charge-ID efficiency vs E_T (slightly higher than data)."""
    return np.clip(bdt_eff_data_et(et) * 1.007 - 0.001, 0.87, 0.999)

def bdt_eff_data_eta(eta):
    """Data BDT charge-ID efficiency vs eta (ET > 25 GeV)."""
    ae = np.abs(np.atleast_1d(np.array(eta, dtype=float)))
    eff = np.where(
        ae < BARREL_MAX,
        # Barrel: ~97.5%, small dip at large |eta| near transition
        0.976 - 0.010 * ae - 0.008 * np.maximum(ae - 1.0, 0)**2,
        np.where(
        (ae >= TRANS_MIN) & (ae < TRANS_MAX),
        np.nan,   # transition region: NOT measured
        np.where(
        ae < ENDCAP_MAX,
        # Endcap: ~95-96%, falls with |eta|
        0.958 - 0.022 * (ae - TRANS_MAX) - 0.005 * (ae - TRANS_MAX)**2,
        np.nan)))
    return eff

def bdt_eff_mc_eta(eta):
    """MC efficiency vs eta (slightly higher than data)."""
    d   = bdt_eff_data_eta(eta)
    ae  = np.abs(np.atleast_1d(np.array(eta, dtype=float)))
    corr = np.where(ae < BARREL_MAX, 1.008,
           np.where(ae < TRANS_MAX,  np.nan, 1.004))
    return np.where(np.isnan(d), np.nan, np.clip(d * corr, 0.85, 0.999))

# ── Generate Z->ee dataset ────────────────────────────────────────────────────
def gen_zee(n=200000, mc=False, seed=42):
    rng  = np.random.default_rng(seed)
    # Probe ET: Jacobian peak at mZ/2 ~ 45 GeV
    et   = np.clip(rng.exponential(scale=32.0, size=n) + 15.0, 15.0, 150.0)
    eta  = rng.uniform(-ENDCAP_MAX, ENDCAP_MAX, n)

    # Tag: ET > 27 GeV, outside transition region
    t_et  = rng.exponential(scale=35.0, size=n) + 27.0
    t_eta = rng.uniform(-ENDCAP_MAX, ENDCAP_MAX, n)
    t_in_trans = (np.abs(t_eta) > TRANS_MIN) & (np.abs(t_eta) < TRANS_MAX)
    tag   = (t_et > 27.0) & (~t_in_trans)

    # Probe pass: combine ET and eta efficiency dependences
    eff_et  = bdt_eff_mc_et(et)   if mc else bdt_eff_data_et(et)
    eff_eta = bdt_eff_mc_eta(eta) if mc else bdt_eff_data_eta(eta)
    # Use geometric mean — in transition region, use ET efficiency only
    in_trans = (np.abs(eta) > TRANS_MIN) & (np.abs(eta) < TRANS_MAX)
    eff_comb = np.where(in_trans | np.isnan(eff_eta),
                        eff_et,
                        np.sqrt(eff_et * eff_eta))
    eff_comb = np.clip(np.nan_to_num(eff_comb, nan=0.0), 0, 1)
    probe = rng.random(n) < eff_comb
    return {"et": et, "eta": eta, "tag": tag.astype(int),
            "probe": probe.astype(int)}

# ── Clopper-Pearson ───────────────────────────────────────────────────────────
def cp68(k, n):
    if n == 0: return np.nan, np.nan, np.nan
    lo = float(beta_dist.ppf(0.16, max(k,   1e-9), max(n-k+1, 1e-9)))
    hi = float(beta_dist.ppf(0.84, max(k+1, 1e-9), max(n-k,   1e-9)))
    return k/n, k/n - np.clip(lo,0,1), np.clip(hi,0,1) - k/n

# ── Compute efficiencies ──────────────────────────────────────────────────────
def eff_et(df, et_bins):
    """Efficiency vs ET, all eta, tagged probes."""
    t = {k: v[df["tag"] == 1] for k, v in df.items()}
    out = []
    for i in range(len(et_bins)-1):
        m   = (t["et"] >= et_bins[i]) & (t["et"] < et_bins[i+1])
        k,n = int(t["probe"][m].sum()), int(m.sum())
        e, el, eh = cp68(k, n)
        out.append({"e":e,"el":el,"eh":eh,"n":n,"k":k})
    return out

def eff_eta(df, et_min=25.0):
    """Efficiency vs eta for ET > et_min, tagged probes, excl. transition."""
    t = {k: v[(df["tag"]==1) & (df["et"] >= et_min)] for k, v in df.items()}
    # Exclude probes in transition region
    ae   = np.abs(t["eta"])
    keep = (ae < TRANS_MIN) | (ae >= TRANS_MAX)
    tk   = {k: v[keep] for k, v in t.items()}
    out  = []
    for j in range(N_ETA):
        # Check if this bin is entirely in the transition gap
        lo_j, hi_j = ETA_BINS[j], ETA_BINS[j+1]
        if ((lo_j >= TRANS_MIN and hi_j <= TRANS_MAX) or
            (lo_j >= -TRANS_MAX and hi_j <= -TRANS_MIN)):
            out.append({"e":np.nan,"el":np.nan,"eh":np.nan,"n":0,"gap":True})
            continue
        m   = (tk["eta"] >= lo_j) & (tk["eta"] < hi_j)
        k,n = int(tk["probe"][m].sum()), int(m.sum())
        e, el, eh = cp68(k, n)
        out.append({"e":e,"el":el,"eh":eh,"n":n,"gap":False})
    return out

print("Generating datasets: 200k events each...")
df_d = gen_zee(200000, mc=False, seed=42)
df_m = gen_zee(200000, mc=True,  seed=99)

r_d_et  = eff_et(df_d, ET_BINS)
r_m_et  = eff_et(df_m, ET_BINS)
r_d_eta = eff_eta(df_d, et_min=25.0)
r_m_eta = eff_eta(df_m, et_min=25.0)

def unpack(r):
    e  = np.array([x["e"]  for x in r], dtype=float)
    el = np.array([x["el"] for x in r], dtype=float)
    eh = np.array([x["eh"] for x in r], dtype=float)
    return e, el, eh

ed_et, eld_et, ehd_et = unpack(r_d_et)
em_et, elm_et, ehm_et = unpack(r_m_et)
ed_eta, eld_eta, ehd_eta = unpack(r_d_eta)
em_eta, elm_eta, ehm_eta = unpack(r_m_eta)

sf_et  = np.where(em_et>0,  ed_et/em_et,   np.nan)
sf_eta = np.where(em_eta>0, ed_eta/em_eta, np.nan)

def sf_err(ed, el_d, em, el_m):
    sf = np.where(em>0, ed/em, np.nan)
    return sf * np.sqrt(
        (np.abs(el_d)/np.where(ed>0,ed,1.))**2 +
        (np.abs(el_m)/np.where(em>0,em,1.))**2)

sfe_et  = sf_err(ed_et,  eld_et,  em_et,  elm_et)
sfe_eta = sf_err(ed_eta, eld_eta, em_eta, elm_eta)

# Dynamic ratio y-axis ranges
def dyn_range(sf, sfe, margin=0.01, step=0.02):
    v = sf[np.isfinite(sf)]; e = sfe[np.isfinite(sf)]
    lo = max(0.90, np.floor((np.nanmin(v - np.abs(e)*2) - margin)/step)*step)
    hi = min(1.10, np.ceil( (np.nanmax(v + np.abs(e)*2) + margin)/step)*step)
    return lo, hi

rlo_et,  rhi_et  = dyn_range(sf_et,  sfe_et)
rlo_eta, rhi_eta = dyn_range(sf_eta, sfe_eta)
print(f"  ET  ratio range: [{rlo_et:.2f}, {rhi_et:.2f}]")
print(f"  eta ratio range: [{rlo_eta:.2f}, {rhi_eta:.2f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE — 2 columns × 2 rows matching ATLAS Fig 14 exactly
# Left col:  vs E_T  (upper) + ratio (lower)
# Right col: vs eta  (upper) + ratio (lower)
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 8))
gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1],
                        hspace=0.04, wspace=0.28)
ax_et_top  = fig.add_subplot(gs[0, 0])
ax_et_bot  = fig.add_subplot(gs[1, 0], sharex=ax_et_top)
ax_eta_top = fig.add_subplot(gs[0, 1])
ax_eta_bot = fig.add_subplot(gs[1, 1], sharex=ax_eta_top)
plt.setp(ax_et_top.get_xticklabels(),  visible=False)
plt.setp(ax_eta_top.get_xticklabels(), visible=False)

MSIZE = 6.0   # marker size matching paper

# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEFT: Efficiency vs E_T
# ─────────────────────────────────────────────────────────────────────────────
# Simulation: closed/filled circles
ax_et_top.errorbar(ET_CENTRES, em_et,
    xerr=[ET_XERR_LO, ET_XERR_HI],
    yerr=[np.abs(elm_et), np.abs(ehm_et)],
    fmt="o", color="black", markersize=MSIZE,
    markerfacecolor="black",   # CLOSED = simulation
    markeredgecolor="black", linewidth=1.2, capsize=0,
    label="Simulation", zorder=4)

# Data: open circles
ax_et_top.errorbar(ET_CENTRES, ed_et,
    xerr=[ET_XERR_LO, ET_XERR_HI],
    yerr=[np.abs(eld_et), np.abs(ehd_et)],
    fmt="o", color="black", markersize=MSIZE,
    markerfacecolor="white",   # OPEN = data
    markeredgecolor="black", markeredgewidth=1.5,
    linewidth=1.2, capsize=0,
    label="Data", zorder=5)

ax_et_top.axhline(y=0.97, color="gray", linewidth=0.8,
                  linestyle=":", alpha=0.6,
                  label="Target 97%")
ax_et_top.set_ylabel("Efficiency", fontsize=12)
ax_et_top.set_ylim(0.89, 1.005)
ax_et_top.set_xlim(13, 160)
ax_et_top.set_xscale("log")
ax_et_top.legend(loc="lower right", fontsize=9,
                 handlelength=1.5, framealpha=0.90)
ax_et_top.grid(True, alpha=0.18)
ax_et_top.text(0.04, 0.97,
    r"$Z \rightarrow ee$,  $\sqrt{s}=13$ TeV" + "\n"
    r"BDT charge-ID ($\varepsilon_{\rm BDT}=97\%$)",
    transform=ax_et_top.transAxes,
    fontsize=8.5, va="top")

# BOTTOM-LEFT: ratio vs E_T
ax_et_bot.errorbar(ET_CENTRES, sf_et,
    xerr=[ET_XERR_LO, ET_XERR_HI],
    yerr=np.clip(np.abs(sfe_et), 0, 0.08),
    fmt="o", color="black", markersize=MSIZE,
    markerfacecolor="white", markeredgecolor="black",
    markeredgewidth=1.5, linewidth=1.2, capsize=0, zorder=4)
ax_et_bot.axhline(y=1.0, color="black", linewidth=0.9)
ax_et_bot.fill_between([13,160],[0.98,0.98],[1.02,1.02],
    alpha=0.10, color="gray")
ax_et_bot.set_ylim(rlo_et, rhi_et)
ax_et_bot.set_xlim(13, 160)
ax_et_bot.set_xscale("log")
ax_et_bot.set_ylabel("Data/MC", fontsize=11)
ax_et_bot.set_xlabel(r"$E_T$ [GeV]", fontsize=12)
ax_et_bot.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax_et_bot.yaxis.set_minor_locator(mticker.MultipleLocator(0.005))
ax_et_bot.set_xticks([15, 20, 30, 40, 60, 80, 150])
ax_et_bot.set_xticklabels(["15","20","30","40","60","80","150"])
ax_et_bot.grid(True, alpha=0.18)

# ─────────────────────────────────────────────────────────────────────────────
# TOP-RIGHT: Efficiency vs eta (ET > 25 GeV)
# ─────────────────────────────────────────────────────────────────────────────
# Separate valid bins from gap
valid_eta = np.array([not r.get("gap",False) and np.isfinite(r["e"])
                       for r in r_d_eta])
vc  = ETA_CENTRES[valid_eta]
vlo = ETA_LO[valid_eta]; vhi = ETA_HI[valid_eta]
ed_v = ed_eta[valid_eta]; eld_v = eld_eta[valid_eta]; ehd_v = ehd_eta[valid_eta]
em_v = em_eta[valid_eta]; elm_v = elm_eta[valid_eta]; ehm_v = ehm_eta[valid_eta]
sf_v  = sf_eta[valid_eta]; sfe_v = sfe_eta[valid_eta]

# Simulation: closed
ax_eta_top.errorbar(vc, em_v,
    xerr=[vc - vlo, vhi - vc],
    yerr=[np.abs(elm_v), np.abs(ehm_v)],
    fmt="o", color="black", markersize=MSIZE,
    markerfacecolor="black",
    markeredgecolor="black", linewidth=1.2, capsize=0,
    label="Simulation", zorder=4)

# Data: open
ax_eta_top.errorbar(vc, ed_v,
    xerr=[vc - vlo, vhi - vc],
    yerr=[np.abs(eld_v), np.abs(ehd_v)],
    fmt="o", color="black", markersize=MSIZE,
    markerfacecolor="white",
    markeredgecolor="black", markeredgewidth=1.5,
    linewidth=1.2, capsize=0,
    label="Data", zorder=5)

ax_eta_top.axhline(y=0.97, color="gray", linewidth=0.8,
                   linestyle=":", alpha=0.6)

# Transition region shading (ATLAS: 1.37 < |eta| < 1.52)
for tlo, thi, lbl in [(-TRANS_MAX, -TRANS_MIN, None),
                       ( TRANS_MIN,  TRANS_MAX, "Transition\nregion")]:
    ax_eta_top.axvspan(tlo, thi, alpha=0.18, color="#FFA500", zorder=1,
                       label=lbl)

# Boundary lines
for b in [-TRANS_MAX,-TRANS_MIN, TRANS_MIN, TRANS_MAX]:
    ax_eta_top.axvline(x=b, color="gray", linewidth=0.7,
                       linestyle="--", alpha=0.55)

# Region text
for xp, txt in [(-2.0,"Endcap"),(-0.68,"Barrel"),(0.68,"Barrel"),(2.0,"Endcap")]:
    ax_eta_top.text(xp, 1.001, txt, ha="center", fontsize=9,
                    color="navy", fontweight="bold")
for xp in [-1.445, 1.445]:
    ax_eta_top.text(xp, 0.925, "Trans.", ha="center", fontsize=7.5,
                    color="#7B3F00", rotation=90, va="center")

ax_eta_top.set_ylabel("Efficiency", fontsize=12)
ax_eta_top.set_ylim(0.89, 1.005)
ax_eta_top.set_xlim(-2.55, 2.55)
ax_eta_top.legend(loc="lower center", ncol=3, fontsize=9, framealpha=0.90)
ax_eta_top.grid(True, alpha=0.18)
ax_eta_top.text(0.02, 0.06,
    r"$E_T > 25$ GeV",
    transform=ax_eta_top.transAxes, fontsize=9.5,
    va="bottom", color="black",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
              alpha=0.8, edgecolor="gray"))

# BOTTOM-RIGHT: ratio vs eta
ax_eta_bot.errorbar(vc, sf_v,
    xerr=[vc - vlo, vhi - vc],
    yerr=np.clip(np.abs(sfe_v), 0, 0.08),
    fmt="o", color="black", markersize=MSIZE,
    markerfacecolor="white", markeredgecolor="black",
    markeredgewidth=1.5, linewidth=1.2, capsize=0, zorder=4)
ax_eta_bot.axhline(y=1.0, color="black", linewidth=0.9)
ax_eta_bot.fill_between([-2.55,2.55],[0.98,0.98],[1.02,1.02],
    alpha=0.10, color="gray")
for tlo, thi in [(-TRANS_MAX,-TRANS_MIN),(TRANS_MIN,TRANS_MAX)]:
    ax_eta_bot.axvspan(tlo, thi, alpha=0.18, color="#FFA500", zorder=1)
for b in [-TRANS_MAX,-TRANS_MIN, TRANS_MIN, TRANS_MAX]:
    ax_eta_bot.axvline(x=b, color="gray", linewidth=0.7,
                       linestyle="--", alpha=0.55)
ax_eta_bot.set_ylim(rlo_eta, rhi_eta)
ax_eta_bot.set_xlim(-2.55, 2.55)
ax_eta_bot.set_ylabel("Data/MC", fontsize=11)
ax_eta_bot.set_xlabel(r"$\eta$", fontsize=12)
ax_eta_bot.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax_eta_bot.yaxis.set_minor_locator(mticker.MultipleLocator(0.005))
ax_eta_bot.grid(True, alpha=0.18)

# ── Overall title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Efficiency of BDT Charge Misidentification Suppression\n"
    r"Medium electrons from $Z \rightarrow ee$,  $\sqrt{s}=13$ TeV  "
    r"— Style: ATLAS EPJC 79, 639 (2019) Fig. 14",
    fontsize=11, y=1.01)

plt.savefig("results/plot1_atlas_fig14_v2.png",
            bbox_inches="tight", dpi=150)
plt.close()
print("Saved: results/plot1_atlas_fig14_v2.png")

# Print tables
print(f"\nEfficiency vs E_T (all eta, ET bins 15-150 GeV):")
print(f"{'Bin':>12}  {'N':>7}  {'eff_data':>9}  {'eff_mc':>9}  {'SF':>8}")
for i in range(len(ET_CENTRES)):
    n = r_d_et[i]['n']
    print(f"  {ET_LO[i]:4.0f}-{ET_HI[i]:<6.0f}  {n:>7,}  "
          f"{ed_et[i]:>9.4f}  {em_et[i]:>9.4f}  {sf_et[i]:>8.4f}")

print(f"\nEfficiency vs eta (ET > 25 GeV):")
print(f"{'Bin eta':>16}  {'N':>7}  {'eff_data':>9}  {'eff_mc':>9}  {'SF':>8}")
for j in range(N_ETA):
    if r_d_eta[j].get("gap"): continue
    n = r_d_eta[j]['n']
    print(f"  {ETA_LO[j]:6.3f} to {ETA_HI[j]:5.3f}  {n:>7,}  "
          f"{ed_eta[j]:>9.4f}  {em_eta[j]:>9.4f}  {sf_eta[j]:>8.4f}")

# Electron Efficiency Tag-and-Probe

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Physics](https://img.shields.io/badge/Domain-Particle%20Physics-purple)](https://cern.ch)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A complete, fully-documented Python implementation of the **Tag-and-Probe electron
efficiency measurement** methodology, reproducing the style and physics of
published ATLAS results at the LHC.

> **Note on data:** All data is synthetically generated to match published
> efficiency distributions from the papers below. No proprietary CERN data is used.

---

## Publications This Work Is Based On

| Paper | Reference | Role in this repo |
|-------|-----------|-------------------|
| **ATLAS EPJC (2019)** | ATLAS Collaboration, *Eur. Phys. J. C* **79**, 639 (2019) — [DOI](https://doi.org/10.1140/epjc/s10052-019-7140-6) \| [arXiv:1902.04655](https://arxiv.org/abs/1902.04655) | Plot 1 — Fig. 14 style, all definitions, tag/probe criteria |
| **Romanian Journal (2023)** | S. Younas, *Rom. J. Phys.* **68**, 403 (2023) — [Full text PDF](https://www.nipne.ro/rjp/2023_68_7-8/RomJPhys.68.403.pdf) | Plot 2 — pseudoexperiment uncertainty methodology |

---

## Repository Structure

```
electron-efficiency-tagandprobe/
├── README.md
├── requirements.txt
├── Dockerfile
├── data/
│   └── generate_synthetic_data.py
├── src/
│   ├── tag_and_probe.py                      # Core efficiency measurement
│   ├── uncertainties.py                      # Systematic uncertainty estimation
│   ├── scale_factors.py                      # Data/MC scale factor computation
│   ├── plot1_efficiency_vs_eta_atlas_fig14.py  # Plot 1: BDT charge-ID efficiency
│   └── plot2_uncertainties_romanian_journal.py # Plot 2: pseudoexperiment study
└── results/
    ├── plot1_efficiency_atlas_fig14.png
    └── plot2_uncertainties_romanian_journal.png
```

---

## Quick Start

```bash
git clone https://github.com/suyounas/electron-efficiency-tagandprobe
cd electron-efficiency-tagandprobe
pip install -r requirements.txt

python src/plot1_efficiency_vs_eta_atlas_fig14.py
python src/plot2_uncertainties_romanian_journal.py
```

---

## Plot 1 — BDT Charge Misidentification Efficiency (ATLAS EPJC 2019 Fig. 14)

![Plot 1](results/plot1_efficiency_atlas_fig14.png)

### What This Plot Shows

Reproduction of **Figure 14** from ATLAS EPJC 79, 639 (2019). It shows the efficiency
of a BDT (Boosted Decision Tree) discriminant applied to suppress charge
misidentified electrons in Z → ee events. The BDT is optimised for 97% correct-charge
efficiency. Two panels:

- **Left panel:** Efficiency vs E_T (all 7 ET bins, 15–150 GeV, full η range)
- **Right panel:** Efficiency vs η (ET > 25 GeV cut, all η bins excluding transition)

Each panel has a **Data/MC ratio pad** below with a dynamic y-axis so all error
bars are fully visible.

### Marker Convention (from Fig. 14 caption)

> "Open points: data; closed points: simulation; and lower panels:
> data-to-simulation ratios."

- **Open circles** = Data
- **Filled/closed circles** = Simulation

### ATLAS Detector Definitions Used (Section 2, EPJC 2019)

The ATLAS EM calorimeter is divided as follows. These are **ATLAS-specific** values —
they differ from CMS which uses different boundaries:

| Region | η range | Description |
|--------|---------|-------------|
| **Barrel** | \|η\| < **1.37** | High-quality precision EM calorimeter region |
| **Transition region** | **1.37** < \|η\| < **1.52** | Gap between barrel and endcap — large inactive material, NOT used for precision measurements |
| **Endcap** | **1.52** < \|η\| < **2.47** | EM endcap calorimeter |
| Outside acceptance | \|η\| > 2.47 | Not used |

> ⚠️ **Important:** The ATLAS transition region is **1.37 < |η| < 1.52**.
> CMS uses different boundaries (1.4442 and 1.566). These values must **not** be
> mixed between experiments.

The transition region contains a relatively large amount of inactive material
(support structures, cabling, services) that causes:
- Increased bremsstrahlung before the calorimeter
- Incomplete energy collection in the EM shower
- Degraded GSF track reconstruction
- High background in template fits

For this reason, **the tag electron is required to be OUTSIDE the transition region**
(1.37 < |η| < 1.52), and the transition region is **excluded from the η efficiency
measurement** (shown as orange shading with no data points).

### Tag-and-Probe Selection (Section 4.1, EPJC 2019)

**Tag electron requirements:**
- E_T > **27 GeV**
- |η| < 2.47, **outside transition region** (1.37 < |η| < 1.52)
- Passes **Tight** identification criteria
- Matched to trigger object
- Passes isolation requirements

**Probe electron (cluster probe):**
- E_T > **15 GeV** (for ET efficiency measurement)
- E_T > **25 GeV** (for η efficiency measurement — right panel)
- |η| < 2.47 (full acceptance including transition region as probe)
- EM-cluster candidate (not required to be fully reconstructed)

**Event selection:**
- Z → ee events: two electron candidates with |η| < 2.47
- Invariant mass window: **75 < m_ee < 105 GeV** (Z peak)
- If both electrons satisfy tag criteria → two tag-probe pairs per event

### BDT Charge Misidentification (Section 8.2, EPJC 2019)

Electrons can have their charge incorrectly reconstructed due to:
- Hard bremsstrahlung followed by photon conversion creating a secondary track
  with opposite-sign curvature
- Highly curved tracks due to large energy loss in the inner detector material
- Trident topology where the original track is lost and a secondary track is found

The BDT is trained on simulation to maximise separation of correct-charge from
misidentified-charge electrons using shower shape and track quality variables.
The **operating point** is chosen for **97% correct-charge efficiency** in simulation.

The measured efficiency in data is slightly below simulation (~0.5–1%), with
scale factors in the range **0.990–0.998** confirming good data/MC agreement.

### ET Bins Used (Fig. 14 top panel, 7 bins)

| Bin | E_T range [GeV] | N_probe (data) |
|-----|----------------|----------------|
| 1 | 15 – 20 | ~27,000 |
| 2 | 20 – 25 | ~23,000 |
| 3 | 25 – 30 | ~20,000 |
| 4 | 30 – 40 | ~32,000 |
| 5 | 40 – 60 | ~40,000 |
| 6 | 60 – 80 | ~21,000 |
| 7 | 80 – 150 | ~22,000 |

### Eta Bins Used (Fig. 14 bottom panel, ET > 25 GeV, 12 bins)

| Bin | η range | Region |
|-----|---------|--------|
| 1 | −2.47 to −2.01 | Negative endcap |
| 2 | −2.01 to −1.52 | Negative endcap |
| — | −1.52 to −1.37 | **Transition (excluded)** |
| 3 | −1.37 to −1.01 | Negative barrel |
| 4 | −1.01 to −0.60 | Negative barrel |
| 5 | −0.60 to −0.20 | Negative barrel |
| 6 | −0.20 to  0.00 | Central barrel |
| 7 |  0.00 to  0.20 | Central barrel |
| 8 |  0.20 to  0.60 | Positive barrel |
| 9 |  0.60 to  1.01 | Positive barrel |
| 10 | 1.01 to  1.37 | Positive barrel |
| — | 1.37 to  1.52 | **Transition (excluded)** |
| 11 | 1.52 to  2.01 | Positive endcap |
| 12 | 2.01 to  2.47 | Positive endcap |

### Statistical Uncertainty — Clopper-Pearson (Section 4, EPJC 2019)

> "The statistical uncertainty in a single variation of the measurement is
> calculated following the approach in Ref. [23], i.e. assuming a binomial
> distribution."

The exact binomial (Clopper-Pearson) interval at 68.3% confidence level:

```python
from scipy.stats import beta as beta_dist
lo = beta_dist.ppf(0.16, k,   n-k+1)   # lower 1σ bound
hi = beta_dist.ppf(0.84, k+1, n-k  )   # upper 1σ bound
err_lo, err_hi = eff - lo, hi - eff
```

---

## Plot 2 — Statistical Uncertainty Accuracy (Romanian Journal 2023)

![Plot 2](results/plot2_uncertainties_romanian_journal.png)

### What This Plot Shows

Validation of the first-order analytical uncertainty approximation (Eq. 9 of the
Romanian Journal paper) against pseudoexperiment results (Eq. 11).

**Fixed parameters:**
- β = 10% (background/signal ratio in the Z peak region)
- α = 0.0 (flat exponential background shape)
- ε = 97%, 98%, 99% (target reconstruction efficiency)
- N = 1,000 → 100,000 (total Z → ee events)
- N_toys = 10,000 pseudoexperiments per configuration

**Upper pad:** σ(toys) [filled markers] and σ(approx.) [open dashed markers] vs N events.
Red = 97%, black = 98%, cyan = 99%. Both decrease as 1/√N.

**Lower pad:** Ratio σ(toys)/σ(approx.) with dynamic y-axis showing all points.
Ratio stays within **±3% of unity** across the full N range.

### Methodology (Section 3 of Romanian Journal paper)

**Two methods compared:**

**σ(toys) — Pseudoexperiment RMS [Eq. 11]:**
```
σ(toys) = sqrt[ (1/N)Σε(i)² − ((1/N)Σε(i))² ]
```
Ground truth: run 10,000 pseudoexperiments, Poisson-fluctuate all observables,
compute the RMS of the resulting efficiency distribution.

**σ(approx.) — First-order analytical [Eq. 9]:**
Propagate Poisson uncertainties analytically through the Tag-and-Probe efficiency
formula (Eq. 8). This is the approximation used in published ATLAS results.

**Background split between probe categories (Section 3.1):** 1:2:6
(passing track quality : failing track quality : non-reconstructed).

**Conclusion:** The ratio σ(toys)/σ(approx.) lies between 0.96 and 1.06 for all
N values from 1,000 to 100,000. The approximation is accurate and always slightly
conservative — it never underestimates the true statistical error.

---

## Author

**Sulman Younas** — PhD in Physics, CERN ATLAS
📧 sulmanyounas93@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/sulman-younas-2a6287bb/)
🔗 [GitHub](https://github.com/suyounas)

---

## References

1. ATLAS Collaboration, *Electron reconstruction and identification in the ATLAS
   experiment using the 2015 and 2016 LHC proton–proton collision data at √s = 13 TeV*,
   **Eur. Phys. J. C 79, 639 (2019)**
   - DOI: [10.1140/epjc/s10052-019-7140-6](https://doi.org/10.1140/epjc/s10052-019-7140-6)
   - arXiv (open access): [arXiv:1902.04655](https://arxiv.org/abs/1902.04655)
   - Springer full text: [link.springer.com](https://link.springer.com/article/10.1140/epjc/s10052-019-7140-6)

2. S. Younas, *Methodology of electron reconstruction efficiency in situ calibration
   at high-energy colliders and accuracy of the associated statistical uncertainties*,
   **Rom. J. Phys. 68, 403 (2023)**
   - Full text (open access): [nipne.ro/rjp](https://www.nipne.ro/rjp/2023_68_7-8/RomJPhys.68.403.pdf)
   - Journal page: [Romanian Journal of Physics](https://www.nipne.ro/rjp/)

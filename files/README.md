# Electron Efficiency Tag-and-Probe

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Physics](https://img.shields.io/badge/Domain-Particle%20Physics-purple)](https://cern.ch)
[![ATLAS](https://img.shields.io/badge/Experiment-ATLAS%20%2F%20CMS-orange)](https://atlas.cern.ch)

A complete Python reimplementation of the **Tag-and-Probe electron reconstruction 
efficiency measurement** methodology used at the ATLAS and CMS experiments at CERN.

> **Based on the methodology published in:**  
> S. Younas, *Rom. J. Phys.* **68**, 403 (2023)  
> [https://www.nipne.ro/rjp/2023_68_7-8/RomJPhys.68.403.pdf](https://www.nipne.ro/rjp/2023_68_7-8/RomJPhys.68.403.pdf)

> **Related ATLAS publication:**  
> Electron and photon efficiencies in LHC Run 2 with the ATLAS experiment — *JHEP* (2023)  
> [https://cds.cern.ch/record/2798075](https://cds.cern.ch/record/2798075)

---

## What is Tag-and-Probe?

The **Tag-and-Probe (T&P)** method is the standard technique for measuring electron 
reconstruction and identification efficiency at hadron colliders using data-driven 
Z → ee decays. It avoids relying purely on simulation by extracting efficiency directly 
from recorded collision data.

- **Tag electron:** Passes tight selection criteria (high-quality, unambiguous electron)
- **Probe electron:** The other electron from the Z decay — efficiency measured on probes
- **Efficiency:** fraction of probe electrons that pass the reconstruction/ID criteria

The measured efficiency is compared to simulation (Monte Carlo) to derive **scale factors** 
that correct simulation to match data.

---

## ⚠️ Note on Data

> All data in this repository is **synthetically generated** to match published efficiency 
> distributions from the Romanian Journal of Physics paper. No proprietary CERN data is used.  
> This reimplementation is for educational and portfolio purposes, demonstrating the 
> statistical methodology in a fully reproducible, open-source form.

---

## Repository Structure

```
electron-efficiency-tagandprobe/
├── README.md
├── requirements.txt
├── Dockerfile
├── data/
│   └── generate_synthetic_data.py    # Synthetic electron dataset generator
├── src/
│   ├── tag_and_probe.py              # Core efficiency calculation
│   ├── uncertainties.py              # Systematic uncertainty estimation
│   └── scale_factors.py             # Data/MC scale factor computation
├── notebooks/
│   └── efficiency_analysis.ipynb    # Full walkthrough notebook
└── results/
    └── efficiency_maps.png          # Output efficiency maps
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/suyounas/electron-efficiency-tagandprobe
cd electron-efficiency-tagandprobe

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python data/generate_synthetic_data.py

# Run the full analysis
python src/tag_and_probe.py

# Compute scale factors
python src/scale_factors.py
```

Or run everything in Docker:

```bash
docker build -t electron-efficiency .
docker run electron-efficiency
```

---

## Physics Background

The electron reconstruction efficiency ε is defined as:

```
ε(pT, η) = N_pass / N_probe
```

where:
- `N_probe` = number of probe electrons in a (pT, η) bin
- `N_pass`  = number of probes passing the reconstruction criterion

Efficiency is measured in bins of:
- **pT** (transverse momentum): 20 – 200 GeV
- **η** (pseudorapidity): −2.5 to 2.5

**Scale factors** correct simulation to data:

```
SF(pT, η) = ε_data(pT, η) / ε_MC(pT, η)
```

---

## Results

The pipeline produces:

- **2D efficiency maps** binned in (pT, η)
- **Statistical uncertainties** from Poisson counting
- **Systematic uncertainties** from tag selection variation and background subtraction
- **Scale factors** with combined statistical + systematic errors

---

## Author

**Sulman Younas** — Physics PhD, CERN ATLAS  
📧 sulmanyounas93@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sulman-younas-2a6287bb/)

---

## Citation

If you use this methodology, please cite the original paper:

```
S. Younas, Rom. J. Phys. 68, 403 (2023)
```

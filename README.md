# Paper 2: Facility-Level Structural Drivers of HIV Treatment Outcomes

## Overview

Analysis code for:

> **"Facility-Level Structural Drivers of HIV Treatment Outcomes: A Multi-Level Analysis of 27,288 Patients from a Nigerian HIV Programme and Implications for PEPFAR and Global Fund Programming"**
>
> Lakshmi Kalyani Chinthala  
> Independent Researcher, Golden Gate University  
> ORCID: [0009-0009-8736-6673](https://orcid.org/0009-0009-8736-6673)  
> Submitted to: BMJ Global Health

---

## Research Question

Which facility-level characteristics (care level, ownership type, funding model) are independently associated with poor HIV treatment outcomes after adjustment for patient-level clinical factors?

---

## Dataset

**Quality of Care HIV Dataset** (same as Paper 1)
- Source: [Kaggle — iogbonna (2022)](https://www.kaggle.com/datasets/iogbonna/quality-of-care-dataset-for-hiv-clients)
- 27,288 HIV-positive patients on ART
- Nigerian national HIV programme, July 2006 – December 2018
- Place in `data/` directory before running

---

## Key Results

| Finding | Result |
|---------|--------|
| Primary HC vs Tertiary (adjusted OR) | 1.95 (95% CI: 1.45–2.61, p<0.001) |
| NGO funding (adjusted OR) | 1.24 (95% CI: 1.10–1.39, p<0.001) |
| Federal funding (adjusted OR) | 1.25 (95% CI: 1.06–1.46, p=0.005) |
| Sex — Female (adjusted OR) | 0.87 (95% CI: 0.79–0.96, p=0.003) |
| ICC (facility-level clustering) | 2.2% |
| LR test — facility vars improve fit | chi-squared=53.6, p<0.001 |
| Excess poor outcomes (sub-tertiary) | ~397 in this dataset |

> **Note:** Primary HC finding is preliminary (n=521, 1.9% of sample). External validation required.

---

## Scripts (run in order)

```
paper2/src/
├── 01_feature_engineering.py      # Outcome variables + facility features
├── 02_descriptive_analysis.py     # Table 1 + chi-squared tests
├── 03_logistic_regression.py      # Main model + HC3 robust SEs
├── 04_icc_model_comparison.py     # ICC + AIC/BIC comparison
├── 05_interaction_analysis.py     # Funding × facility level interactions
├── 06_sensitivity_analyses.py     # 4 pre-specified sensitivity models
├── 07_facility_typology.py        # Positive deviant analysis
├── 08_missing_data.py             # Missing data characterisation
├── 09_economic_implications.py    # Excess outcomes + cost estimates
└── 10_figures.py                  # All 12 publication figures (300 DPI)
```

## Quick Start

```bash
# From repo root
pip install -r requirements.txt

# Place QualityOfCare.xlsx in data/

python paper2/src/01_feature_engineering.py
python paper2/src/02_descriptive_analysis.py
python paper2/src/03_logistic_regression.py
python paper2/src/04_icc_model_comparison.py
python paper2/src/05_interaction_analysis.py
python paper2/src/06_sensitivity_analyses.py
python paper2/src/07_facility_typology.py
python paper2/src/08_missing_data.py
python paper2/src/09_economic_implications.py
python paper2/src/10_figures.py
```

---

## Methodological Notes

- **Clustering:** 11 facility-level clusters precluded GEE or mixed-effects logistic regression (both require ≥20–30 clusters). HC3 heteroscedasticity-robust standard errors used throughout.
- **ICC:** Estimated from null linear probability mixed model — 2.2% of variance attributable to facility level.
- **SMOTE:** Not applied in this paper (outcomes modelled at natural prevalence for ecological validity).
- **STROBE:** Reporting guidelines followed throughout.

---

## Part of a Two-Paper Series

| Paper | Focus | Journal |
|-------|-------|---------|
| Paper 1 | Patient-level ML prediction (AUC 0.963) | npj Digital Medicine (under review) |
| Paper 2 | Facility-level health systems analysis | BMJ Global Health (submitted) |

---

## License

MIT — see root [LICENSE](../../LICENSE)

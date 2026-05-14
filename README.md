# SmartDaaS — HIV Programme Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartdaas-hiv-validation-afzaaz9drhjvfdyycem8ea.streamlit.app)
![Version](https://img.shields.io/badge/version-0.3-blue)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![License](https://img.shields.io/badge/license-MIT-green)

> ⚠️ **Research prototype only. Not validated for clinical decision-making. Do not use for individual patient treatment decisions without clinical review and appropriate regulatory approval.**

---

## What is SmartDaaS?

SmartDaaS is a research-validated HIV programme intelligence platform that combines **patient-level adherence risk prediction** (Paper 1) with **facility-level structural performance analytics** (Paper 2) into a unified decision support tool for HIV programme managers, PEPFAR implementing partners, and national HIV programme officers.

**The core question SmartDaaS answers:**
> *"Which patients at which facilities are most likely to fail treatment — and what should we do about it right now?"*

---

## What it does

| Feature | Description |
|---|---|
| 📊 **Patient Risk Scoring** | Upload a CSV of patient records → get individual risk scores (0–100%) for ART non-adherence or poor outcomes |
| 🧠 **SHAP Explainability** | Per-patient clinical reasoning — "PT-0141 is HIGH risk because: Prior ART interruption (+0.175), WHO Stage 3 (+0.109)" |
| 🏥 **Facility Intelligence** | Risk-adjusted facility benchmarking — identifies structurally underperforming facilities independent of patient case-mix |
| 🎯 **Decision Intelligence** | Today's call list, facility priority ranking, what-if intervention simulator, weekly programme summary exports |
| 💰 **Economic Calculator** | Quantifies excess poor outcomes and avoidable programme costs. ROI projections for targeted interventions |
| 📋 **Sample Data** | Downloadable CSV template with column reference guide and auto column-name mapping |

---

## Research Foundation

This platform is built on two peer-reviewed studies currently under review:

**Paper 1 — Under review at npj Digital Medicine**
> Chinthala LK. *Real-World Validation of Machine Learning Models for HIV Treatment Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis of 192,732 Clinical Records.* 2026.

| Metric | Value |
|---|---|
| AUC (cross-validation) | 0.963 |
| AUC (temporal validation) | 0.772 |
| Sensitivity | 87.3% |
| Specificity | 95.7% |
| Training records | 192,732 |

**Paper 2 — Under review at BMJ Global Health**
> Chinthala LK. *Facility-Level Structural Drivers of HIV Treatment Outcomes: A Multi-Level Analysis of 27,288 Patients from a Nigerian HIV Programme and Implications for PEPFAR and Global Fund Programming.* 2026.

Key findings:
- Primary health centres: **1.95× higher adjusted odds** of composite poor outcome vs tertiary hospitals (OR 1.95, 95% CI 1.45–2.61, p<0.001)
- NGO-funded facilities: independently **24% higher odds** (OR 1.24, 95% CI 1.10–1.39, p<0.001)
- Female sex independently protective (OR 0.87, p=0.003)

---

## Model Features

The patient risk model uses 15 clinical variables available from routine HIV programme records:

| Feature | Description |
|---|---|
| `Age` | Patient age in years at ART initiation |
| `sex_female` | Binary: 1=Female, 0=Male |
| `Cd4AtStart` | CD4 cell count (cells/µL) at ART start |
| `MostRecentCd4Count` | Most recent CD4 count (cells/µL) |
| `CD4_improvement` | Change in CD4 since ART start |
| `stage_start_num` | WHO clinical stage at ART initiation (1–4) |
| `WeightAtStart` | Patient weight (kg) at ART initiation |
| `weight_change` | Change in weight (kg) since ART start |
| `BMI_start` | BMI at ART initiation |
| `days_to_ART` | Days from HIV diagnosis to ART start |
| `had_interruption` | Binary: prior ART interruption documented |
| `opp_infection` | Binary: opportunistic infection documented |
| `side_effects` | Binary: side effects reported |
| `tb_positive` | Binary: TB positive status |
| `stage_worsened` | Binary: WHO stage worsened since ART start |

---

## Risk Tiers

| Tier | Score | Recommended Action |
|---|---|---|
| 🔴 HIGH | ≥70% | Urgent adherence counselling within 48hrs, peer navigator, clinical officer review |
| 🟡 MEDIUM | 40–69% | Adherence counselling within 2 weeks, attendance review, SMS reminder |
| 🟢 LOW | <40% | Standard care pathway, routine follow-up |

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/Kchinthala15/smartdaas-hiv-validation.git
cd smartdaas-hiv-validation

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**Requirements:**
- Python 3.9+
- `cv_results.pkl` and `prepped_data.pkl` must be in the repo root (included)

---

## Requirements

```
streamlit>=1.32.0,<2.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
scikit-learn>=1.3.0,<2.0.0
shap==0.44.1
matplotlib>=3.7.0,<4.0.0
openpyxl>=3.1.0,<4.0.0
```

---

## Deploying to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `app.py`
5. Click Deploy

---

## Target Users

- HIV programme managers (PEPFAR-supported settings)
- PEPFAR implementing partners (APIN, FHI 360, Jhpiego, JSI, Palladium)
- Global Fund grantees
- National HIV programme officers and MoH analytics teams
- Global health researchers

---

## PEPFAR MER Indicator Alignment

| MER Indicator | SmartDaaS Signal |
|---|---|
| TX_CURR | Base cohort for risk scoring |
| TX_PVLS | Low CD4/high risk → proxy for VL failure risk |
| TX_RTT | High risk patients flagged for re-engagement |
| TX_ML | ART interruption predictor (top SHAP feature) |
| TX_NEW | Delayed ART (>90 days) flagged in facility analysis |

---

## Limitations

1. **Single-country training data** — facility-level analysis uses Nigerian programme data (2006–2018). External validation in other settings is required before broad application.
2. **Temporal validation AUC 0.772** — performance degrades on future data as expected. This is the honest real-world estimate.
3. **15 features only** — does not capture socioeconomic status, geographic remoteness, or drug supply chain quality.
4. **Cross-sectional design** — causal inference is not possible from observational data.
5. **Research prototype** — not validated for clinical deployment. Regulatory review required before clinical use.

---

## Interested in a Pilot?

SmartDaaS is seeking one PEPFAR implementing partner for a paid analytical pilot:

> *"Provide your programme's DHIS2/EMR export. We run our models and deliver a facility + patient risk intelligence report."*

**Contact:** chinthalakalyani1@gmail.com
**ORCID:** 0009-0009-8736-6673

---

## Citation

If you use SmartDaaS in research or programme evaluation, please cite:

```
Chinthala LK. Real-World Validation of Machine Learning Models for HIV Treatment
Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis of
192,732 Clinical Records. Under review at npj Digital Medicine, 2026.

Chinthala LK. Facility-Level Structural Drivers of HIV Treatment Outcomes: A
Multi-Level Analysis of 27,288 Patients from a Nigerian HIV Programme and
Implications for PEPFAR and Global Fund Programming. Under review at BMJ
Global Health, 2026.
```

---

## License

MIT — see [LICENSE](LICENSE)

---

*SmartDaaS v0.3 · Research Prototype · Not for Clinical Use*
*Lakshmi Kalyani Chinthala · Golden Gate University · San Francisco, CA*

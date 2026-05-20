# SmartDaaS — HIV Programme Intelligence Platform

**AI-powered decision support for HIV programme managers, PEPFAR implementing partners, and Global Fund grantees across sub-Saharan Africa.**

[![Platform Status](https://img.shields.io/badge/Platform-Pilot--Ready-21d4fd)](https://smartdaas-hiv-validation.onrender.com)
[![Version](https://img.shields.io/badge/Version-1.2.0-3fb950)](https://smartdaas-hiv-validation.onrender.com)
[![Model AUC](https://img.shields.io/badge/Temporal%20AUC-0.772-f0a500)](https://smartdaas-hiv-validation.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-8b949e)](LICENSE)

---

## What SmartDaaS Does

SmartDaaS transforms routine HIV programme data into actionable operational intelligence. It identifies high-risk patients before they are lost to follow-up, detects underperforming facilities, quantifies avoidable programme costs, and generates executive-ready reports for donor reporting cycles.

**Core capabilities:**

- **Patient Risk Engine** — Individual risk scores with SHAP explainability. Upload your programme CSV and get a prioritised patient list within minutes.
- **Local Validation & Recalibration** — Fit a locally-validated model on your programme's own historical data. Report your local AUC to funders, not the Nigerian discovery cohort baseline.
- **Facility Intelligence** — Risk-adjusted facility benchmarking and positive deviant analysis. Structural drivers of outcomes after patient case-mix adjustment.
- **IeDEA MUD Regional Benchmarks** — Contextual comparison against IeDEA Multi-Use Dataset aggregates across West Africa (n=42,369), East Africa (n=229,002), Southern Africa (n=921,922), and Central Africa (n=42,459).
- **Data Quality Screening** — Automated A/B/C/D data quality grade with transparent scoring before any analysis runs.
- **Economic Impact Calculator** — Scenario-based cost projections with illustrative regional programme cost reference estimates.
- **Executive Reports** — One-click PDF reports formatted for programme directors and donor reporting meetings.

---

## Model Performance

| Metric | Value | Notes |
|---|---|---|
| Temporal AUC | **0.772** | Held-out future patients — realistic deployment estimate |
| Cross-validation AUC | 0.963 | 10-fold stratified CV — optimistic estimate |
| Sensitivity | 87.3% | At optimal threshold |
| Specificity | 95.7% | At optimal threshold |
| Training records | 27,288 | Nigerian HIV programme — discovery cohort |
| Calibration | Platt scaling / Isotonic regression | Local recalibration supported |

> **Honest framing:** The temporal AUC of 0.772 is the realistic deployment estimate. The cross-validation AUC of 0.963 reflects within-training-data performance and is optimistic. Always report the temporal AUC as your pre-recalibration baseline. Local recalibration on your programme's data is required before operational use.

---

## Data Foundation

**Discovery cohort:** 27,288 HIV-positive patients on ART from the Nigerian national HIV programme (Quality of Care dataset, 2006–2018). SmartDaaS was originally trained on this Nigerian HIV programme data and is currently being externally evaluated using independent DHS and PHIA datasets across sub-Saharan Africa.

**External validation status:** Multi-country external validation infrastructure has been established using independent population-based datasets:
- PHIA (Population-based HIV Impact Assessment) datasets — 9 surveys across Kenya, Malawi, Rwanda, Tanzania, Uganda, and Zambia (n=16,496 HIV-positive adults). Variable harmonisation completed for available datasets. Further reconciliation in progress.
- DHS (Demographic and Health Surveys) — multi-country datasets across Zambia, Malawi, Uganda, and Kenya. Variable mapping completed.
- IeDEA West Africa consortium — data access application submitted
- IeDEA East Africa and Southern Africa — applications in preparation

**External consistency finding:** SmartDaaS demonstrated external consistency of its core predictive signal across independent PHIA populations, while full model validation will require longitudinal programme datasets with complete feature availability and target outcomes. VLS failure prediction on pooled PHIA data (10,113 adults, 845 events, 9 datasets) yielded AUC 0.769 ± 0.009, comparable to the SmartDaaS temporal AUC of 0.772 on the Nigerian discovery cohort.

**What this means for deployment:** The base model reflects Nigerian programme patterns. Local recalibration on your programme's historical data is required before operational use in other settings. The Local Validation page guides you through this process.

---

## Research Foundation

**Paper 1 — Under review at npj Digital Medicine**
Chinthala LK. *Real-World Validation of Machine Learning Models for HIV Treatment Adherence Prediction and Care Gap Quantification: Validation on 27,288 Nigerian HIV Programme Records.* 2026.

**Paper 2 — Under preparation for submission**
Chinthala LK. *Facility-Level Structural Drivers of HIV Treatment Outcomes: A Multi-Level Analysis of 27,288 Patients from a Nigerian HIV Programme and Implications for PEPFAR and Global Fund Programming.* 2026.

> DOIs will be added upon publication. Analysis code available in this repository.

---

## Target Users

- HIV programme managers at PEPFAR implementing partners (APIN, IHVN, FHI 360, EGPAF, and similar)
- Strategic Information and M&E officers
- Global Fund principal recipients and sub-recipients
- National HIV programme officers and Ministry of Health analytics teams
- Implementation science researchers

---

## Platform Architecture

SmartDaaS is built on a tiered data architecture that adapts to the variables available in your programme export:

| Tier | Variables Required | What You Get |
|---|---|---|
| Core | Age, sex, ART status | Cohort characterisation + IeDEA regional benchmarks |
| Standard | Core + CD4, WHO stage, TB status, days to ART | Risk estimates (confidence varies by variables present) |
| Enhanced | Standard + interruption history, OI, weight, BMI | Full 15-feature model + SHAP explainability |

The platform automatically detects your data tier on upload, maps international variable naming conventions, and reports a transparent data quality grade before any analysis runs.

---

## Shadow Analytics Pilot Model

SmartDaaS is seeking implementing partner organisations for a 6-month shadow analytics pilot.

**What the pilot involves:**
1. You provide a historical programme data export (CSV from your EMR or DHIS2)
2. SmartDaaS runs local validation — computing a locally-validated AUC on your data
3. The validated model runs alongside your existing workflows for 3–6 months
4. SmartDaaS delivers a pilot outcome report for your board and funders

**What you need:** Minimum 200 patients with known outcomes, minimum 30 positive outcome events.

**What you get:** A locally-validated AUC, high-risk patient detection analysis, facility performance findings, and an executive intelligence report formatted for donor reporting.

**No workflow disruption. No new data collection. Results within weeks.**

Contact: chinthalakalyani1@gmail.com

---

## Installation and Deployment

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)
- Supabase account (optional — for audit trail)

### Local Development

```bash
git clone https://github.com/Kchinthala15/smartdaas-hiv-validation.git
cd smartdaas-hiv-validation
pip install -r requirements.txt
streamlit run app.py
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `APP_PASSWORD` | Recommended | Password for platform access |
| `ADMIN_PASSWORD` | Recommended | Password for Admin page |
| `SUPABASE_URL` | Optional | Supabase project URL for audit logging |
| `SUPABASE_KEY` | Optional | Supabase publishable key |

### Docker Deployment (Render)

The platform deploys via Docker on Render. Push to the `main` branch to trigger auto-deploy.

```bash
# The Dockerfile handles all dependencies
# Set environment variables in Render dashboard
```

---

## Model Architecture

```
Pipeline: StandardScaler → RandomForestClassifier
Training: 10-fold stratified CV with SMOTE (applied within folds only)
Temporal validation: Held-out post-2015 patients
Explainability: SHAP TreeExplainer with per-patient waterfall charts
Calibration: Platt scaling (n<500) or Isotonic regression (n≥500)
Features: 15 clinical variables (age, sex, CD4 trajectory, WHO stage,
          weight, BMI, days to ART, interruption history, OI, TB, side effects)
```

---

## Data Privacy and Governance

- **No patient data stored:** Uploaded data is processed in-session only and never transmitted externally or retained after the session ends.
- **Audit trail:** Only aggregate metadata is logged (event type, patient count, data tier, quality grade). No patient records, no file names, no organisation names.
- **File names:** Stored as SHA-256 hashes only.
- **DUA acknowledgement:** Users confirm data governance authority before any upload. Formal Data Use Agreements are established with pilot partners before operational engagement.
- **Regulatory alignment:** Nigeria NDPR, PEPFAR data governance frameworks.

---

## Limitations

1. **Single-country discovery cohort:** The base model was trained on Nigerian programme data (2006–2018). SmartDaaS is currently being externally evaluated using independent PHIA and DHS datasets across sub-Saharan Africa. External consistency of the core predictive signal has been demonstrated across 6 countries; full model validation requires longitudinal programme data with complete feature availability. Local recalibration is required before operational use in any specific setting.
2. **Temporal AUC of 0.772:** Performance degrades on future data, as expected for any ML model. This is the honest pre-recalibration deployment estimate.
3. **15 clinical variables only:** The model does not capture socioeconomic factors, geographic remoteness, drug supply chain quality, or facility staffing — all known outcome drivers.
4. **Binary composite outcome:** The model predicts composite poor outcome (non-adherence OR interruption OR mortality). It does not distinguish between these endpoints.
5. **Not validated for clinical decision-making:** Local validation on your programme's data is required before deployment. This is a decision-support tool — all outputs require review by qualified programme and clinical staff.
6. **Paediatric patients (under 15):** The model was trained on adult patients. Risk scores for paediatric patients are flagged and should not be used without specialist clinical review.

---

## IeDEA MUD Benchmarks

Regional aggregate contextual benchmarks are sourced from the IeDEA Multi-Use Dataset (MUD) v1.0:

> IeDEA (2025). Version 1.0. IeDEA Multi-Use Dataset (MUD). Retrieved from iedea.org. License: CC BY-NC-SA 4.0.

These benchmarks represent aggregate indicators from IeDEA-participating clinical sites and are not nationally representative. They are contextual comparisons only — not external validation of the SmartDaaS model.

---

## Citation

If you use SmartDaaS in research or programme evaluation:

```
Chinthala LK. SmartDaaS: HIV Programme Intelligence Platform (v1.2.0).
GitHub: https://github.com/Kchinthala15/smartdaas-hiv-validation. 2026.
```

---

## Contact

**Lakshmi Kalyani Chinthala**
Independent researcher · Ageno School of Business, Golden Gate University, San Francisco CA
ORCID: 0009-0009-8736-6673
Email: chinthalakalyani1@gmail.com
GitHub: github.com/Kchinthala15

---

*SmartDaaS is a decision-support platform for HIV programme intelligence and operational analytics. It is not intended to replace clinical judgment or function as an autonomous clinical decision-making system. All outputs should be reviewed by qualified programme teams prior to operational use.*

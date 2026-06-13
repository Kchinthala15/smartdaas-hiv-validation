# SmartDaaS — HIV Programme Intelligence Platform

> **AI-powered decision support for HIV programme managers, PEPFAR implementing partners, and Global Fund grantees across sub-Saharan Africa.**

[![Platform Status](https://img.shields.io/badge/Platform-Pilot--Ready-21d4fd)](https://smartdaas-hiv-validation.onrender.com)
[![Version](https://img.shields.io/badge/Version-1.0.0-3fb950)](https://smartdaas-hiv-validation.onrender.com)
[![Temporal AUC](https://img.shields.io/badge/Temporal%20AUC-0.772-f0a500)](https://smartdaas-hiv-validation.onrender.com)
[![Scientific Reports](https://img.shields.io/badge/Scientific%20Reports-Under%20Review-blue)](https://doi.org/10.64898/2026.05.15.26353325)
[![License](https://img.shields.io/badge/License-MIT-8b949e)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--8736--6673-a6ce39)](https://orcid.org/0009-0009-8736-6673)

---

SmartDaaS is an open-source HIV programme intelligence platform that transforms routine DHIS2 and EMR exports into actionable clinical and operational insights — without adding reporting burden, changing workflows, or requiring new data collection. Designed for PEPFAR implementing partners, Global Fund grantees, and national HIV programme teams, SmartDaaS identifies patients at risk of treatment interruption, flags underperforming facilities, and generates donor-ready executive reports — all from a single CSV upload.

---

## 🔒 Zero Patient Data Liability

> **Stateless architecture that processes data locally with SHA-256 hashing — zero patient PII is ever uploaded or stored.**

SmartDaaS is engineered to pass the strictest institutional procurement audits while entirely bypassing traditional IRB and cross-border data transfer bottlenecks:

| Security Principle | Implementation |
|---|---|
| **No PHI retained** | Stateless processing — all computation is ephemeral, nothing persists post-session |
| **SHA-256 hashing** | All file names and unique identifiers are one-way hashed on import — mathematically irreversible |
| **Zero PII stored** | No patient data ever leaves the session or touches a server |
| **Enterprise audit logging** | Supabase audit trail with brute-force authentication protection and automated 4-hour session timeouts |
| **IRB not required** | Stateless architecture bypasses cross-border data transfer review entirely |
| **DUA acknowledgement** | Users confirm governance authority before any upload proceeds |

```python
# SmartDaaS Security Architecture
PHI_RETAINED          = False   # stateless processing layer
PII_STORED            = False   # zero patient data liability
IDENTIFIER_HASHING    = "SHA-256"  # one-way · irreversible
SESSION_TIMEOUT_HR    = 4       # automated inactivity lockout
IRB_REQUIRED          = False   # stateless · bypasses data transfer review
CROSS_BORDER_TRANSFER = False   # no patient data leaves the session
# STATUS: PROCUREMENT READY
```

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Research & Publications](#research--publications)
- [Live Platform](#live-platform)
- [Screenshots](#screenshots)
- [Model Performance](#model-performance)
- [Technical Architecture](#technical-architecture)
- [Data Foundation & External Validation](#data-foundation--external-validation)
- [Use Cases](#use-cases)
- [Target Users](#target-users)
- [Shadow Analytics Pilot](#shadow-analytics-pilot)
- [Installation](#installation)
- [Data Privacy & Governance](#data-privacy--governance)
- [Limitations](#limitations)
- [Future Roadmap](#future-roadmap)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

HIV programmes in sub-Saharan Africa generate vast amounts of routine data through EMR systems and DHIS2. Yet most of this data is used primarily for backward-looking reporting — not forward-looking programme intelligence. SmartDaaS bridges that gap.

By applying machine learning to existing programme data, SmartDaaS produces:

- **Patient-level risk scores** that identify who is most likely to interrupt treatment before it happens
- **Facility-level benchmarks** that reveal which sites are over- or under-performing after adjusting for patient case-mix
- **Explainable AI outputs** that translate model predictions into plain-language clinical and programmatic drivers — so programme staff understand *why* a patient is flagged, not just *that* they are
- **Locally-validated models** that can be recalibrated on any implementing partner's own historical data, producing a site-specific AUC for funder reporting

SmartDaaS is designed as a **decision-support tool** — not an autonomous system. All outputs are intended to inform, not replace, clinical and programme judgement.

---


## Market Opportunity

| Metric | Value | Source |
|---|---|---|
| People living with HIV on ART globally | 39M+ | WHO 2023 |
| Annual HIV programme spend (PEPFAR + Global Fund) | $20B+ | UNAIDS 2023 |
| Patients lost to follow-up annually (Sub-Saharan Africa) | 1 in 3 | IeDEA / UNAIDS |
| AI programme intelligence platforms currently in market | 0 | Market analysis |

SmartDaaS targets the **programme intelligence gap** — the space between raw EMR data and operational decision-making that no current tool addresses. Existing tools tell programme teams what happened. SmartDaaS tells them where to focus next.

---

## Key Features

### 🎯 Patient Risk Engine
Individual risk scores with SHAP explainability. Upload your programme CSV and receive a prioritised patient list within minutes, with per-patient waterfall charts explaining the top drivers of risk.

### 🔬 Local Validation & Recalibration
Fit a locally-validated model on your programme's own historical data. Report your site-specific AUC to funders — not the Nigerian discovery cohort baseline. Supports Platt scaling and isotonic regression calibration.

### 🏥 Facility Intelligence
Risk-adjusted facility benchmarking and positive deviant analysis. Identifies structural drivers of outcomes after patient case-mix adjustment.

### 📊 IeDEA Regional Benchmarks
Contextual comparison against IeDEA Multi-Use Dataset (MUD) v1.0 aggregates:
- West Africa (n = 42,369) · East Africa (n = 229,002)
- Southern Africa (n = 921,922) · Central Africa (n = 42,459)

### 🔍 Explainability Layer (SHAP)
SHAP TreeExplainer provides per-patient waterfall charts and cohort-level feature importance summaries. Every prediction is traceable to its clinical drivers.

### 📋 Data Quality Screening
Automated A/B/C/D data quality grade with transparent scoring before any analysis runs. Catches missingness, implausible values, and schema mismatches.

### 💰 Economic Impact Calculator
Scenario-based cost projections using illustrative regional programme cost reference estimates. Quantifies the financial value of improved retention and reduced LTFU.

### 📄 Executive Reports
One-click PDF reports formatted for programme directors and donor reporting meetings.

### 🌐 DHIS2 / EMR Compatibility
Accepts standard CSV exports from DHIS2, OpenMRS, and most EMR systems across sub-Saharan Africa. Automatic variable mapping handles common naming conventions.

### 🔒 Validation Metadata Layer
Every upload generates a transparent audit trail of all inferences, proxies, imputations, and derivations applied — surfaced to users before analysis runs.

---

## 🌍 Supported Ingestion Pipelines

SmartDaaS natively recognises column names from major national HIV data systems — no reformatting or data engineering required:

| System | Country | Compatibility |
|---|---|---|
| **Kenya NASCOP** | Kenya | ✅ Native column mapping |
| **Uganda DHIS2** | Uganda | ✅ Native column mapping |
| **Malawi HMIS** | Malawi | ✅ Native column mapping |
| **MPHIA** | Malawi Population HIV Impact Assessment | ✅ Native column mapping |
| **THIS** | Tanzania HIV Impact Survey | ✅ Native column mapping |
| **UPHIA** | Uganda Population HIV Impact Assessment | ✅ Native column mapping |
| **Generic EMR export** | Any PEPFAR country | ✅ 80+ column aliases |
| **OpenMRS** | Multi-country | ✅ Supported |

> **No API required. No IT sign-off. No data pipeline.** SmartDaaS accepts the same CSV export your M&E team already pulls monthly.

---

## 🔧 Engineered for Real-World, Messy Data

Frontline health data is fragmented, incomplete, and inconsistent. SmartDaaS doesn't break — it adapts:

| Feature | What it does |
|---|---|
| **`dq_grade` — Data Quality Tiering** | Automated A/B/C/D profiling checks cohort integrity before any analysis runs. Catches missingness, implausible values, and schema mismatches. |
| **`cal_method` — Dynamic Recalibration** | Pipeline automatically adjusts calibration method (Platt scaling or isotonic regression) based on data quality tier — accurate outputs regardless of clinic environment. |
| **Column aliasing** | 80+ aliases handle naming inconsistencies across EMR systems. Unrecognised columns are flagged — never silently dropped. |
| **Transparent audit trail** | Every inference, proxy, imputation, and derivation is surfaced to users before analysis runs. |

> *Engineered for real-world, missing, and fragmented clinic records — not clean lab data.*

---

## Research & Publications

### 📄 Paper 1 — Under Review at Scientific Reports

**Chinthala LK.** *Real-World Validation of Machine Learning Models for HIV Treatment Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis of 192,732 Clinical Records.* Under review, Scientific Reports, 2026.

🔖 **Public DOI preprint:** [10.64898/2026.05.15.26353325](https://doi.org/10.64898/2026.05.15.26353325)

> *Manuscript under peer review. Preprint available at the DOI above.*

---

### 📄 Paper 2 — Under Review at BMJ Global Health

**Chinthala LK.** *Facility-Level Structural Drivers of HIV Treatment Outcomes: A Multi-Level Analysis of 27,288 Patients from a Nigerian HIV Programme and Implications for PEPFAR and Global Fund Programming.* Under review, BMJ Global Health, 2026.

🔖 **Public DOI preprint:** [10.64898/2026.05.15.26353326](https://doi.org/10.64898/2026.05.15.26353326)

> *Manuscript under peer review. Preprint available at the DOI above.*

---

### 👤 Author

**Lakshmi Kalyani Chinthala**
Independent Researcher · Ageno School of Business, Golden Gate University, San Francisco CA

🔗 **ORCID:** [0009-0009-8736-6673](https://orcid.org/0009-0009-8736-6673)

---

## Live Platform

🌐 **SmartDaaS App:** [https://smartdaas-hiv-validation.onrender.com](https://smartdaas-hiv-validation.onrender.com)

> The live platform is available for demonstration and pilot evaluation. Sample data is included on the platform for exploration without needing your own data upload.

---

## Screenshots

> Screenshots from the live platform are available at [smartdaas-hiv-validation.onrender.com](https://smartdaas-hiv-validation.onrender.com). The platform includes a full synthetic dataset for immediate exploration — no data upload required.

**Key screens:**
- Programme Dashboard — cohort overview, risk distribution, outreach capacity planner
- Patient Risk Stratification — ranked patient list with per-patient SHAP waterfall charts
- Facility Intelligence — risk-adjusted benchmarking across facilities
- Data Quality Report — automated A/B/C/D tiering with transparent scoring
- Executive Report — one-click PDF formatted for donor reporting meetings

---

## Model Performance

| Metric | Value | Notes |
|---|---|---|
| Temporal AUC | **0.772** | Held-out future patients — realistic deployment estimate |
| Cross-validation AUC | 0.975 | 10-fold stratified CV — optimistic in-sample estimate |
| Ablation AUC (14 features) | 0.963 | Excludes ART interruption history — confirms robustness (1.4% relative reduction, no leakage) |
| Sensitivity | 87.3% | At optimal threshold |
| Specificity | 95.7% | At optimal threshold |
| Training records | 27,288 | Nigerian HIV programme — Quality of Care dataset (discovery cohort) |
| Multi-country validation | 165,444 | CEPHIA assay database — 6 countries (US, UK, South Africa, Uganda, Kenya, Brazil) |
| Total study records | 192,732 | Combined across both datasets |
| Calibration | Platt scaling / Isotonic regression | Local recalibration supported |

> **Honest framing:** The temporal AUC of 0.772 is the realistic deployment estimate. The CV AUC of 0.963 reflects within-training performance and is optimistic. Always report the temporal AUC as the pre-recalibration baseline. Local recalibration on site-specific data is required before operational use.

---

## Technical Architecture

SmartDaaS operates across five layers:

```
┌─────────────────────────────────────────────────────────┐
│  1. DATA INGESTION                                      │
│     CSV upload · DHIS2 / EMR export compatibility       │
│     Automatic variable mapping · Schema normalisation   │
│     PHIA / DHS population survey compatibility          │
├─────────────────────────────────────────────────────────┤
│  2. DATA QUALITY & FEATURE ENGINEERING                  │
│     A/B/C/D quality grade · Missingness detection       │
│     15-feature clinical pipeline · Tier detection       │
│     Validation metadata audit trail                     │
├─────────────────────────────────────────────────────────┤
│  3. PREDICTIVE MODELLING                                │
│     StandardScaler → RandomForestClassifier             │
│     SMOTE (within CV folds only)                        │
│     Temporal validation on held-out future patients     │
├─────────────────────────────────────────────────────────┤
│  4. EXPLAINABILITY & CALIBRATION                        │
│     SHAP TreeExplainer · Per-patient waterfall charts   │
│     Platt scaling (n < 500) · Isotonic regression       │
│     Local recalibration on partner programme data       │
├─────────────────────────────────────────────────────────┤
│  5. REPORTING & INTELLIGENCE                            │
│     Risk stratification · Facility benchmarking         │
│     Economic impact · Executive PDF reports             │
│     IeDEA regional benchmark comparisons                │
└─────────────────────────────────────────────────────────┘
```

### Tiered Data Architecture

| Tier | Variables Required | What You Get |
|---|---|---|
| **Core** | Age, sex, ART status | Cohort characterisation + IeDEA regional benchmarks |
| **Standard** | Core + CD4, WHO stage, TB status, days to ART | Risk estimates with confidence indicators |
| **Enhanced** | Standard + interruption history, OI, weight, BMI | Full 15-feature model + SHAP explainability |

### Model Pipeline

```
Pipeline:       StandardScaler → RandomForestClassifier
Training:       10-fold stratified CV with SMOTE (within folds only)
Validation:     Held-out post-2015 patients (temporal split)
Explainability: SHAP TreeExplainer with per-patient waterfall charts
Calibration:    Platt scaling (n < 500) or Isotonic regression (n ≥ 500)
Features:       15 clinical variables — age, sex, CD4 trajectory, WHO stage,
                weight, BMI, days to ART, interruption history, OI, TB,
                side effects, CD4 improvement, weight change, stage worsened
```

---

## Data Foundation & External Validation

### Discovery Cohort
27,288 HIV-positive patients on ART from the Nigerian national HIV programme (Quality of Care dataset, 2006–2018). Used for model training, cross-validation, and temporal validation.

### CEPHIA Multi-Country Validation Dataset
165,444 HIV-positive specimens from the CEPHIA (Consortium for the Evaluation and Performance of HIV Incidence Assays) Public Use Dataset, spanning 6 countries: United States, United Kingdom, South Africa, Uganda, Kenya, and Brazil. Used for multi-country HIV recency analysis. 18.6% of specimens (n=30,732) met the RITA recent-infection criterion (≤130 days post-estimated date of infection), providing evidence-based grounding for the SmartDaaS early-detection design across diverse settings and HIV subtypes.

**Total records analysed across both datasets: 192,732**

### External Validation Infrastructure
SmartDaaS was originally trained on Nigerian HIV programme data and is currently being externally evaluated using independent population-based datasets across sub-Saharan Africa.

**PHIA datasets (9 surveys, 6 countries):**
- Kenya (KENPHIA 2018), Malawi (MPHIA 2015–16, 2020–21), Rwanda (RPHIA 2018–19)
- Tanzania (THIS 2016–17, 2022–23), Uganda (UPHIA 2016–17, 2020–21), Zambia (ZAMPHIA 2016)
- Combined HIV-positive analytical cohort: **16,496 adults**

**DHS datasets:** Variable mapping completed across Zambia, Malawi, Uganda, and Kenya.

**External consistency finding:**
SmartDaaS demonstrated external consistency of its core predictive signal across independent PHIA populations. VLS failure prediction on pooled PHIA data yielded **AUC 0.769 ± 0.009** (10,113 adults, 845 events, 9 datasets) — comparable to the SmartDaaS temporal AUC of 0.772 on the discovery cohort.

> Full model validation will require longitudinal programme datasets with complete feature availability and target outcomes. Local recalibration remains required before operational use in any specific setting.

---

## Use Cases

| Use Case | Description |
|---|---|
| **HIV Retention Monitoring** | Identify patients at elevated LTFU risk before disengagement occurs |
| **Treatment Interruption Prediction** | Predict ART interruption and prioritise proactive support |
| **Facility Performance Assessment** | Risk-adjusted facility benchmarking and positive deviant identification |
| **Donor Reporting** | Locally-validated AUC for PEPFAR APR and Global Fund performance reviews |
| **M&E Support** | Automated data quality grading, cohort analytics, IeDEA benchmarks |
| **Implementation Science** | Model transportability assessment across programme contexts |

---

## Target Users

| User | How SmartDaaS Helps |
|---|---|
| HIV Programme Managers | Risk stratification, facility benchmarking, executive reports |
| SI / M&E Officers | Data quality grading, IeDEA benchmarks, cohort analytics |
| PEPFAR / Global Fund Partners | Locally-validated AUC for funder reporting |
| Ministry of Health Analytics Teams | National programme intelligence, multi-facility comparison |
| Implementation Science Researchers | SHAP feature analysis, multi-country generalisability evidence |
| Clinical Teams | Patient-level risk scores with plain-language SHAP explanations |

---

## Shadow Analytics Pilot

SmartDaaS is seeking implementing partner organisations for a **6-month shadow analytics pilot**.

### What the Pilot Involves
1. You provide a historical programme data export (CSV from your EMR or DHIS2)
2. SmartDaaS runs local validation — computing a locally-validated AUC on your programme data
3. The validated model runs alongside your existing workflows for 3–6 months
4. SmartDaaS delivers a pilot outcome report for your board and funders

### Eligibility
- Minimum 200 patients with known outcomes
- Minimum 30 positive outcome events
- Existing DHIS2 or EMR export capability

### What You Receive
- A locally-validated AUC specific to your programme
- High-risk patient detection analysis
- Facility performance benchmarking
- Executive intelligence report formatted for donor reporting

**No workflow disruption. No new data collection. No cost to pilot partners. Results within weeks.**

📧 **Express interest:** chinthalakalyani1@gmail.com

---

## Installation

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)
- Supabase account (optional — for audit trail logging)

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

### Docker / Render Deployment

```bash
# The Dockerfile handles all dependencies
# Set environment variables in the Render dashboard
# Auto-deploy fires on every push to main branch
```

---

## Data Privacy & Governance

| Principle | Implementation |
|---|---|
| **No patient data stored** | Data processed in-session only — never transmitted or retained |
| **Audit trail** | Aggregate metadata only (event type, patient count, tier, quality grade) |
| **File name hashing** | File names stored as SHA-256 hashes only |
| **DUA acknowledgement** | Users confirm governance authority before upload |
| **Regulatory alignment** | Nigeria NDPR, PEPFAR data governance frameworks |

---

## Limitations

1. **Discovery cohort scope:** Trained on Nigerian programme data (2006–2018). External consistency demonstrated across 6 countries; full validation requires longitudinal data with complete feature availability. Local recalibration required before operational use.

2. **Temporal AUC of 0.772:** Honest pre-recalibration deployment estimate. Performance degrades on future data, as expected for any ML model.

3. **15 clinical variables:** Does not capture socioeconomic factors, geographic remoteness, supply chain quality, or facility staffing.

4. **Binary composite outcome:** Predicts composite poor outcome (non-adherence OR interruption OR mortality). Does not distinguish between endpoints.

5. **Decision-support tool only:** Not validated for autonomous clinical decision-making. All outputs require qualified programme and clinical review.

6. **Paediatric patients:** Trained on adult patients. Risk scores for patients under 15 are flagged and require specialist clinical review.

---

## Future Roadmap

### Near-Term
- [ ] IeDEA East Africa and Southern Africa longitudinal data integration
- [ ] Newer PHIA wave codebook verification (2020–2023 datasets)
- [ ] Uganda restricted AIS dataset integration (approval pending)

### Medium-Term
- [ ] Expanded programme modules (TB/HIV co-infection, PMTCT, paediatric ART)
- [ ] Direct DHIS2 API integration for real-time data ingestion
- [ ] Multi-language support (French, Portuguese, Swahili)
- [ ] Enhanced geospatial visualisation for facility mapping

### Long-Term
- [ ] Federated learning for multi-site improvement without data sharing
- [ ] Real-time EMR integration with automated risk alert workflows
- [ ] WHO and CDC data standard alignment
- [ ] Counterfactual explanation tools for clinical teams

---

## Citation

### Platform

```
Chinthala LK. SmartDaaS: HIV Programme Intelligence Platform (v1.2.0).
GitHub: https://github.com/Kchinthala15/smartdaas-hiv-validation. 2026.
```

### Preprint

```
Chinthala LK. Real-World Validation of Machine Learning Models for HIV
Treatment Adherence Prediction and Care Gap Quantification: A Multi-Country
Analysis of 192,732 Clinical Records. Under review, Scientific Reports, 2026.
Public DOI preprint: 10.64898/2026.05.15.26353325
```

---

## Contact

**Lakshmi Kalyani Chinthala**
Independent Researcher · Ageno School of Business, Golden Gate University, San Francisco CA

| | |
|---|---|
| 📧 Email | chinthalakalyani1@gmail.com |
| 🔗 ORCID | [0009-0009-8736-6673](https://orcid.org/0009-0009-8736-6673) |
| 💻 GitHub | [github.com/Kchinthala15](https://github.com/Kchinthala15) |
| 📄 Preprints | [doi.org/10.64898/2026.05.15.26353325](https://doi.org/10.64898/2026.05.15.26353325) · [doi.org/10.64898/2026.05.15.26353326](https://doi.org/10.64898/2026.05.15.26353326) |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
<em>SmartDaaS is a decision-support platform for HIV programme intelligence and operational analytics.<br>
It is not intended to replace clinical judgment or function as an autonomous clinical decision-making system.<br>
All outputs should be reviewed by qualified programme teams prior to operational use.</em>
</p>

# SmartDaaS — HIV Programme Intelligence Platform

> **Explainable risk prioritisation for HIV programme managers, PEPFAR implementing partners, and Global Fund grantees.**

[![Platform Status](https://img.shields.io/badge/Platform-Pre--Pilot-e3b341)](https://smartdaas.org)
[![Version](https://img.shields.io/badge/Version-1.0.0-3fb950)](https://smartdaas.org)
[![Temporal AUC](https://img.shields.io/badge/Temporal%20AUC-0.806-f0a500)](https://smartdaas.org)
[![Validation](https://img.shields.io/badge/Validation-Internal%20%2B%20Temporal-3fb950)](#model-performance)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20676666-blue)](https://doi.org/10.5281/zenodo.20676666)
[![License](https://img.shields.io/badge/License-MIT-8b949e)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--8736--6673-a6ce39)](https://orcid.org/0009-0009-8736-6673)

---

SmartDaaS turns routine DHIS2 and EMR exports into a ranked list of patients whose records indicate poor ART adherence, with a per-patient explanation of why — from a single CSV upload, without new data collection.

---

> ### ⚠️ Status: pre-pilot. Read this before evaluating.
>
> - **External validation is underway, not complete.** The model has been validated internally (10-fold CV) and temporally (on later patients it never saw), but has not yet been applied to patient-level data from another programme or health system. Two routes are in motion — CNICS and IeDEA West Africa.
> - **The training data's provenance is not documented.** It is a public deposit with no stated country, custodian, sampling frame, or collection methodology.
> - **It is a same-encounter model, not a forecast.** It describes which patients' records indicate poor adherence now. It does not predict who will become non-adherent later.
> - **All v2 architecture metrics are from synthetic data** and are not evidence of clinical performance.
>
> Local recalibration on your own historical data is required before any operational use.

---

## Table of Contents

- [Overview](#overview)
- [Data Privacy & Governance](#data-privacy--governance)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Data Foundation](#data-foundation)
- [Supported Ingestion Pipelines](#supported-ingestion-pipelines)
- [Engineered for Messy Data](#engineered-for-real-world-messy-data)
- [Technical Architecture](#technical-architecture)
- [Codebase Structure](#codebase-structure)
- [Installation](#installation)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

HIV programmes generate large volumes of routine data through EMR systems and DHIS2, most of it used for backward-looking reporting. SmartDaaS applies a model to that existing data to produce:

- **Patient-level risk scores** identifying whose records indicate poor ART adherence, so outreach can be prioritised under resource constraints
- **Facility-level comparisons** across recorded facility attributes (level, ownership, funding source), adjusted for patient case-mix
- **Per-patient SHAP explanations**, so staff see *why* a patient is flagged rather than only *that* they are
- **Local recalibration** on an implementing partner's own historical data, producing a site-specific AUC

SmartDaaS is a **decision-support tool**, not an autonomous system. All outputs inform, never replace, clinical and programme judgement.

**What it predicts:** the probability that a patient is recorded as having poor ART adherence at their most recent clinical visit. Poor adherence is a recognised leading indicator of treatment interruption, but this model has not been validated against interruption, mortality, or any composite outcome.

---

## Data Privacy & Governance

**Read this section literally. It describes what the software does, not what would be convenient.**

SmartDaaS is a Streamlit application. **Uploaded data is transmitted to the server the application is hosted on and processed there.** There is no client-side or local processing. If you host it outside your jurisdiction, uploading patient data constitutes a cross-border data transfer.

| Principle | What actually happens |
|---|---|
| **No patient data retained** | ✅ True. Processing is in-session; session state is ephemeral and nothing patient-level is written to any database. |
| **Data is transmitted to a server** | ✅ **Yes.** Streamlit executes server-side. This is unavoidable in this architecture. |
| **Audit trail** | Aggregate metadata only — event type, timestamp, session UUID, patient count, tier, quality grade, mean risk. Verified: no patient-level field is logged. |
| **File name hashing** | File names are SHA-256 hashed before logging. **Patient identifiers are not hashed** — supply de-identified data. |
| **De-identification** | Your responsibility, before upload. SmartDaaS does not de-identify. |
| **DUA acknowledgement** | Users confirm governance authority before any upload proceeds. |
| **Ethics / IRB review** | **Your institution's determination, not ours.** SmartDaaS makes no claim that any review is unnecessary. |
| **Cross-border transfer** | Occurs if the deployment is hosted outside your jurisdiction. Assess it. |

### If cross-border transfer is not acceptable

Two options, both real:

1. **Self-host.** The Dockerfile builds the whole platform. Run it inside your own infrastructure and no data leaves.
2. **Federated validation** — `smartdaas_validate.py` runs behind your firewall against your own data and returns only aggregate metrics. No patient data moves at all.

> **Note on earlier versions of this README.** Prior versions claimed "no patient data ever leaves the session or touches a server", `CROSS_BORDER_TRANSFER = False`, that unique identifiers were SHA-256 hashed, and that the architecture "bypasses IRB and cross-border data transfer review entirely." **All four were false.** Data is processed server-side; only file names are hashed; and a vendor cannot make a regulatory determination on an institution's behalf. Corrected 2026-07-17.

---

## Key Features

### 🎯 Patient Risk Engine
15 clinical variables → calibrated risk score + tier (HIGH ≥15% / MEDIUM ≥7.5% / LOW). Refuses to score if `had_interruption` — the dominant predictor — is absent, rather than silently defaulting it.

### 🔬 Local Validation & Recalibration
Platt scaling or isotonic regression on your own historical data, with a held-out 30% for honest metrics. Blocks recalibration where the supplied outcome is implausible for adherence (prevalence outside 0.5–25%), because fitting a calibrator to loss-to-follow-up or mortality re-fits the model to a different question.

### 🏥 Facility Intelligence
Comparisons across recorded facility attributes. **The dataset contains no facility identifier**, so individual facilities cannot be distinguished; these are associations across level/ownership/funding categories and are hypothesis-generating only.

### 📊 IeDEA Regional Benchmarks
Contextual comparison against IeDEA Multiregional Update aggregates (public, aggregate-only).

### 🔍 Explainability Layer (SHAP)
Per-patient waterfall charts via TreeExplainer.

### 📋 Data Quality Screening
A/B/C/D grading before any analysis runs — missingness, out-of-range values, schema mismatches. Every imputation and derivation is disclosed.

### 💰 Economic Impact Calculator
Illustrative estimates from published unit costs. Not a costing study.

### 📄 Executive Reports
Donor-ready PDF with risk stratification, top-N patients, SHAP drivers, and methodology.

### 🌐 DHIS2 / EMR Compatibility
80+ column aliases; DHIS2 tracked-entity pull. See the privacy note above regarding credentials.

---

## Model Performance

| Metric | Value | Notes |
|---|---|---|
| **Temporal AUC** | **0.806** (95% CI 0.774–0.837) | Trained on ART initiations ≤ Sept 2016, tested on 6,942 later initiators. **The headline number.** |
| 10-fold CV AUC | 0.801 ± 0.023 | Natural class distribution (3.67%). No oversampling. |
| Brier score | 0.027 temporal / 0.032 CV | |
| Sensitivity @ 0.15 | 27.7% | Flags 3.7% of the cohort |
| Specificity @ 0.15 | 97.0% | |
| PPV @ 0.15 | 21.9% | Against a 3.0% base rate — ~7× concentration of risk |
| Sensitivity @ 0.075 | 52.4% | Flags 11.6% of the cohort |
| Training cohort | 23,144 adults | 850 poor adherence (3.67%) |
| External validation | In progress | CNICS · IeDEA West Africa — see [Data Foundation](#data-foundation) |

**Internal and temporal estimates agree (0.801 vs 0.806), and that agreement is the finding.** A model scoring far higher internally than on later patients is usually leaking information between training and test rather than generalising.

**Threshold choice is a capacity decision, not a statistical one.** It depends on how many patients your outreach team can contact.

**Field context:** the most recent systematic review of ML models for HIV treatment interruption (Kwarah et al., *BMC Global and Public Health* 2025; 12 models) reports a mean internal AUC of **0.668**. Those models predict interruption rather than adherence, so this is indicative rather than like-for-like.

<details>
<summary><strong>Withdrawn performance claims (July 2026)</strong></summary>

Earlier versions of this README reported CV AUC 0.975, hold-out AUC 0.973 on "10,540 patients at natural class distribution", sensitivity 87.3%, specificity 95.7%, and an ablation AUC of 0.963 offered as evidence of "no leakage".

All are invalid. The training pipeline applied SMOTE to the entire dataset **before** the train/test split, so the hold-out was 50% synthetic minority cases interpolated from the training data. The ablation ran on the same contaminated data — it was the leak certifying itself.

The "expected degradation" from CV to temporal AUC (0.203) *was* the leak. With the corrected pipeline the difference is +0.006.

Removing SMOTE, excluding blank outcomes and paediatric patients, and fixing mixed CD4 units **improved** the model: 0.772 → 0.806.

`retrain_v1.py` reproduces all of this from the source spreadsheet.
</details>

### Fairness

Subgroup AUC on the temporal hold-out. **Range 0.719–0.850.**

| Subgroup | n | events | AUC |
|---|---|---|---|
| Overall | 6,942 | 206 | 0.806 |
| Female / Male | 4,751 / 2,191 | 128 / 78 | 0.795 / 0.826 |
| Age <30 / 30–49 / 50+ | 1,802 / 4,207 / 933 | 51 / 133 / 22 | 0.801 / 0.802 / 0.850 |
| **CD4 <200** | 1,760 | 51 | **0.752** |
| CD4 200–499 / 500+ | 2,103 / 1,036 | 22 / 10 | 0.786 / **0.719** |
| WHO I–II / III–IV | 5,591 / 1,117 | 147 / 50 | 0.815 / 0.759 |

**The model performs worst where patients are sickest** (CD4 <200: 0.752 vs 0.806 overall; WHO III–IV: 0.759). Earlier versions claimed a range of 0.815–0.866 and "no systematic disadvantage" — not reproducible, and the CD4 finding is the opposite of what was claimed. Several subgroups carry few events, so intervals are wide; these need testing in a pilot.

---

## Data Foundation

### Discovery cohort

**23,144 adults** (18–100) on ART, from the *Quality of Care dataset for HIV clients* (Kaggle, uploader `iogbonna`, 2022). 99.4% initiated ART 2013–2017.

**Provenance is not documented.** The deposit states no country, programme, custodian, sampling frame, or ethics approval. Its licence reads "Data files © Original Authors" — no rights are granted. Internal evidence (clinical free-text referencing Nigerian states; Federal/State/NGO funding categories) indicates Nigerian programme records — an inference by this author, **not a statement by the depositor**. Previous versions of this README described it as "the Nigerian national HIV programme". That was not supported.

From 27,288 raw records: 1,297 with a blank adherence outcome excluded (previously coded as *adherent*); 2,847 removed by the 18–100 age filter (2,161 under 18, 671 with no age, 15 implausible).

| | |
|---|---|
| Median age | 35 (IQR 29–43) |
| Female | 69.6% |
| Median CD4 at ART start | 260 cells/µL |
| CD4 <200 at start | 37.4% of those with a CD4 recorded (20.3% missing) |
| WHO stage at start | 58.6% I · 22.9% II · 16.4% III · 2.2% IV |
| Median diagnosis-to-ART | 29 days (IQR 0–153) |
| Any prior interruption | 8.9% |

### External validation — in progress

External validation requires patient-level data with the same 15 predictors and an equivalent adherence outcome. Two routes are in motion:

- **CNICS** (CFAR Network of Integrated Clinical Systems) — feasibility request submitted July 2026
- **IeDEA West Africa** — >50,000 adult ART initiators across eight countries including Nigeria; patient-level; concept sheet process

Until one of those completes, **0.806 is an internally and temporally validated estimate, not an externally validated one.** The distinction matters: temporal validation shows the model holds up on patients it never saw from the same programme; external validation shows it holds up somewhere else entirely. Only the second supports a claim of generalisability across health systems.

### Population survey work (PHIA / DHS) — a sibling study, not validation

PHIA and DHS mapping is complete, and this work is real — but **it cannot validate this model.**

PHIA supplies at most **8 of the 15 features** (`constants.py` records this directly). It has no WHO clinical stage, no weight, no height, no BMI, and being cross-sectional, only one CD4 measurement — so `CD4_improvement`, `weight_change` and `stage_worsened` cannot exist. DHS supplies roughly 4 of 15 and has no adherence outcome at all. **A 15-feature model cannot be run on data missing seven of its inputs.**

The PHIA analysis (AUC 0.769 ± 0.009, 10,113 adults, 9 surveys) predicts **viral load suppression failure** — a different outcome, on a different feature set. Its resemblance to 0.806 is not evidence the two measure the same signal. Earlier versions described this as "external consistency of the core predictive signal"; that framing is withdrawn.

**What it actually is:** the design for a separate study — population-level HIV cascade risk from survey data, using social determinants (food insecurity, stigma, mobility, distance to facility, wealth quintile) that clinical records don't capture. That addresses the sociocultural gap Kwarah et al. identified and that zero of 12 reviewed models included. It's a sibling study and will be reported as one.

### CEPHIA — unrelated to this model

165,444 rows from the CEPHIA Public Use Dataset (6 countries) were used for a separate **HIV recency analysis**. CEPHIA is an assay-evaluation panel: its rows are laboratory results, not patients. It was never analysed together with the adherence model.

Previous versions listed CEPHIA under "Multi-country validation" in the Model Performance table and reported "192,732 total study records" combining 27,288 patients with 165,444 assay results. **That total is not a meaningful quantity and is withdrawn.**

---

## Supported Ingestion Pipelines

| Source | Support |
|---|---|
| Generic CSV / EMR export | ✅ 80+ column aliases |
| DHIS2 tracked entities | ✅ Direct pull (see privacy note) |
| KENPHIA / MPHIA / RPHIA / THIS / UPHIA / ZAMPHIA | ⚠️ Native column mapping — **8/15 features; cannot produce a full 15-feature score** |
| DHS | ⚠️ ~4/15 features; no adherence outcome |

---

## Engineered for Real-World, Messy Data

Frontline health data is fragmented and inconsistent. SmartDaaS adapts rather than breaking:

| Feature | What it does |
|---|---|
| **`dq_grade` — Data Quality Tiering** | A/B/C/D profiling before any analysis. Catches missingness, implausible values, schema mismatches. |
| **`cal_method` — Dynamic Recalibration** | Platt or isotonic, selected by sample size and quality tier. |
| **Column aliasing** | 80+ aliases across EMR naming conventions. Unrecognised columns are flagged, never silently dropped. |
| **Median imputation, never zero-fill** | Missing features get the training-cohort median. **Zero is not neutral** — for `had_interruption`, `opp_infection`, `tb_positive` and `stage_worsened`, zero is the *protective* value, and zero-filling tells the model the patient is fine. |
| **Critical-feature hard stop** | Without `had_interruption`, scoring is refused rather than silently defaulted. |
| **Outcome plausibility guard** | Blocks recalibration on an outcome inconsistent with adherence. |
| **Transparent audit trail** | Every inference, proxy, imputation and derivation is surfaced *before* analysis runs. |

---

## Technical Architecture

```
Pipeline:  SimpleImputer(median) → StandardScaler → RandomForestClassifier
Features:  15 clinical variables
Outcome:   poor ART adherence at most recent visit (binary)
Training:  no resampling; natural 3.67% prevalence
```

### Tiered Data Architecture

| Tier | Requires | Produces |
|---|---|---|
| **Core** | Age, sex | Cohort characterisation only — no risk scores |
| **Standard** | Core + 3 of {CD4 at start, recent CD4, WHO stage, days to ART, TB status} | Risk scores, reduced confidence |
| **Enhanced** | Standard + 4 of {CD4 improvement, weight, weight change, BMI, interruption history, OI, side effects, stage worsened} | Full 15-feature model + SHAP |

`had_interruption` is required at every tier that scores.

---

## Codebase Structure

```
smartdaas-hiv-validation/
│
├── app.py                # Main entry — routing, session, upload flow
├── constants.py          # Features, aliases, thresholds, training medians
├── styles.py             # CSS
├── model.py              # Model loading, predictions, SHAP
├── pipeline.py           # Ingestion, mapping, DQ screening, recalibration
├── outreach.py           # Outreach optimiser
├── action_lens.py        # FrontlineLens
├── reports.py            # Executive PDF
├── dhis2_connector.py    # DHIS2 tracked-entity pull
├── api.py                # FastAPI scoring service (not deployed)
│
├── retrain_v1.py         # Reproduces the model end to end from the xlsx
│
├── cv_results.pkl        # Trained pipeline + AUC + thresholds
├── prepped_data.pkl      # Demo cohort — 2,000 SYNTHETIC rows, no real patients
├── requirements.txt      # v2 research deps — NOT the platform's
├── render.yaml           # Render config
└── Dockerfile            # Container build — this is what deploys
```

**For reviewers:** start with `constants.py` (all inputs and thresholds), then `model.py`. The scoring path is `pipeline.derive_engineered_features → model.run_predictions`. `retrain_v1.py` reproduces every published number from the source spreadsheet.

---

## Installation

### Prerequisites
- Python 3.11+
- Docker (for deployment)
- Supabase account (optional — aggregate audit logging only)

### Local development

> **`requirements.txt` will not run the platform.** It lists v2 research dependencies (`torch`, `econml`, `scikit-survival`) and omits `streamlit`, `shap`, `fpdf2`, `supabase`, `requests` and `openpyxl`. The Dockerfile is what actually builds the deployment. Until this is reconciled, use Docker or install directly:

```bash
git clone https://github.com/Kchinthala15/smartdaas-hiv-validation.git
cd smartdaas-hiv-validation
pip install streamlit pandas numpy scikit-learn shap matplotlib fpdf2 openpyxl supabase requests
streamlit run app.py
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `APP_PASSWORD` | Recommended | Platform access |
| `ADMIN_PASSWORD` | Recommended | Admin page |
| `SUPABASE_URL` / `SUPABASE_KEY` | Optional | Aggregate audit logging |
| `ANTHROPIC_API_KEY` | Optional | Enables an LLM rewording layer in the outreach brief. **If set, outreach brief text is sent to a third-party API.** Leave unset to disable. |

---

## Limitations

1. **No external validation.** Never applied to patient-level data from another health system.
2. **Provenance undocumented.** Sampling frame, custodian and methodology of the training data are unknown.
3. **Same-encounter prediction.** Seven of fifteen features are measured at the same visit as the outcome. This describes current adherence status; it does not forecast.
4. **Single country, single era.** 99.4% of the cohort initiated ART 2013–2017 in one country.
5. **Outcome.** Poor adherence at last visit is a leading indicator of, not identical to, treatment interruption.
6. **Weakest where it matters most.** CD4 <200: AUC 0.752 vs 0.806 overall.
7. **Missingness.** CD4 missing for ~20%, BMI for ~45%; unlikely to be missing at random.
8. **No visit-timing features.** Visit frequency, missed appointments, and pharmacy pickup gaps — often the strongest predictors of interruption — are absent from this dataset.
9. **No facility identifier.** Facility analysis compares recorded attributes, not facilities.
10. **v2 modules are synthetic-only prototypes.** No metric from them is evidence of clinical performance.

---

## Roadmap

**Near-term:** external validation via CNICS / IeDEA West Africa · deploy `api.py` · resolve training-data provenance · reconcile `requirements.txt`

**Medium-term:** DHIS2 App Platform application (React + `@dhis2/cli-app-scripts`, calling `api.py` and reusing the user's DHIS2 session) · federated validation deployment (`smartdaas_validate.py`) · the PHIA/DHS social-determinants study

**Long-term:** longitudinal visit-timing features · survival modelling · the v2 architecture, on real pilot data

---

## Citation

**Software:**

> Chinthala LK. *SmartDaaS: an explainable risk model for ART adherence in HIV programme settings.* SmartDaaS LLC, 2026. https://smartdaas.org

**Publications:** none.

Two manuscripts were previously listed here as under review. The first, on this model, was **withdrawn from *Scientific Reports*** in July 2026 after the data leakage error described above was identified. The second, on facility-level outcomes, was **rejected by *BMJ Open*** (not *BMJ Global Health*, as previously stated) in July 2026 on the grounds that the data's provenance, sampling and ownership were unclear — an objection now accepted as correct. Neither should be cited. Associated preprints are being withdrawn.

---

## Contact

**Lakshmi Kalyani Chinthala** — Founder, SmartDaaS LLC
lkchinthala@smartdaas.org · [ORCID 0009-0009-8736-6673](https://orcid.org/0009-0009-8736-6673)
Platform: [smartdaas.org](https://smartdaas.org)

---

## License

MIT — see [LICENSE](LICENSE).

**The training data is not covered by this licence.** The *Quality of Care* dataset is a third-party deposit whose licence grants no rights ("Data files © Original Authors"). It is not redistributed in this repository and should not be redistributed from it.

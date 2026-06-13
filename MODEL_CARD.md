# Model Card: SmartDaaS HIV Treatment Adherence Risk Model

## Model Details

**Developed by:** Lakshmi Kalyani Chinthala, MS Business Analytics (Golden Gate University), Independent HIV Analytics Researcher
**Model date:** 2026
**Model type:** Random Forest classifier
**Model version:** v1 (production)
**Explainability:** SHAP (SHapley Additive exPlanations) — per-patient feature attribution
**License:** MIT
**Contact:** lkchinthala@smartdaas.org

---

## Intended Use

**Primary intended use:** Population-level risk prioritisation tool for HIV implementing partners. SmartDaaS ingests routine, de-identified patient-level programme data and produces a ranked list of patients at elevated risk of poor ART adherence — an established early-warning indicator that precedes treatment interruption — each accompanied by the principal contributing factors (via SHAP), to guide weekly outreach prioritisation under resource constraints.

**Primary intended users:** HIV programme teams, strategic information units, and field outreach coordinators within PEPFAR- and Global Fund-supported implementing partner organisations.

**Out-of-scope uses:**
- Individual clinical treatment decisions. SmartDaaS is a prioritisation and operational planning tool, not a diagnostic or clinical decision-support device, and has not been validated for that purpose.
- Use without local recalibration on the deploying programme's own data is not recommended; performance on a new population without recalibration is not guaranteed to match reported metrics.
- Use outside sub-Saharan African HIV treatment programme contexts has not been evaluated.

---

## What the Model Predicts

The model predicts the probability that a patient will be recorded as having **"Poor" ART adherence at their next clinical assessment** (`ArvAdherenceLatestLevel = "Poor"`). Poor adherence is an established leading indicator of subsequent treatment interruption: patients identified as at risk of poor adherence represent the population for whom proactive outreach is most likely to prevent later disengagement from care. SmartDaaS therefore frames this output operationally as **"treatment interruption risk"** — patients flagged are those showing early signs of disengagement who warrant intensified support before interruption occurs.

---

## Training Data

**Source:** Quality of Care dataset for HIV clients (Ogbonna, 2022), a de-identified, publicly available dataset of HIV-positive patients enrolled on ART within the Nigerian national HIV programme.

**Sample size:** 27,288 patients across 11 facility groups (3 facility levels × ownership/funding types), data period July 2006 – December 2018. Natural poor-adherence prevalence: 3.4% (940/27,288); SMOTE applied to training folds only to address class imbalance.

**Population characteristics:** Mean age 35.2 years; 61.4% female; 36.7% presenting with CD4 <200 cells/µL at ART initiation; WHO stage at ART start — 31.2% Stage I, 38.7% Stage II, 24.1% Stage III, 6.0% Stage IV.

---

## Features

15 input features, derived from routine clinical and demographic data captured at ART initiation and at most recent visit:

| Feature | Description |
|---|---|
| `Age` | Patient age (years) |
| `sex_female` | Sex (binary: female = 1) |
| `Cd4AtStart` | CD4 count at ART initiation |
| `MostRecentCd4Count` | Most recent CD4 count |
| `CD4_improvement` | Change in CD4 since ART start |
| `stage_start_num` | WHO clinical stage at ART start (1–4) |
| `WeightAtStart` | Weight at ART initiation (kg) |
| `weight_change` | Change in weight since ART start |
| `BMI_start` | BMI at ART initiation |
| `days_to_ART` | Days from HIV diagnosis to ART initiation |
| `had_interruption` | History of prior treatment interruption (binary) |
| `opp_infection` | Presence of opportunistic infection (binary) |
| `side_effects` | Reported ART side effects (binary) |
| `tb_positive` | TB status (binary) |
| `stage_worsened` | WHO stage worsened since ART start (binary) |

All 15 features are derived from data routinely captured in standard HIV programme EMR systems; no additional data collection is required.

**On `had_interruption`:** This is the single most important predictor (SHAP mean |value| = 0.163; feature importance 0.381). It reflects **prior** treatment interruption history, recorded a mean of 448 days before the adherence outcome being predicted. This temporal separation, combined with ablation testing (below), confirms this is a genuine predictive signal rather than outcome leakage.

---

## Quantitative Performance

| Metric | Value | Evaluation set |
|---|---|---|
| Cross-validation AUC (15 features) | 0.975 (95% CI: 0.970–0.975) | Nigerian cohort, 10-fold stratified CV (n=27,288) |
| Hold-out AUC | 0.973 | 20% hold-out test set (n=10,540) |
| Temporal validation AUC | 0.772 (95% CI: 0.744–0.802) | Trained on records up to Sept 2016 (n=19,084), tested on later records (n=8,179) |
| Sensitivity | 87.3% (95% CI: 86.4–88.2%) | Hold-out, natural class distribution |
| Specificity | 95.7% (95% CI: 95.2–96.2%) | Hold-out, natural class distribution |
| Brier score | 0.079 | Hold-out (uninformative baseline = 0.25) |
| Ablation AUC (14 features, excl. `had_interruption`) | 0.963 | Confirms robustness — 1.4% relative reduction, no evidence of leakage |
| External validation AUC | 0.769 ± 0.009 | PHIA population surveys, 9 datasets / 6 countries, n=10,113 adults |
| Subgroup AUC range | 0.815–0.866 | Across sex, age, CD4 strata, WHO stage (max difference 0.051) |

**Temporal validation** trains the model on an earlier time period within the Nigerian cohort and evaluates it on a later, unseen time period — a stricter test of generalisability than random train/test splitting. The 0.203 reduction from CV to temporal AUC reflects genuine distributional shift across years and is reported as the realistic pre-recalibration deployment estimate.

**External validation** applies the model architecture to independent Population-based HIV Impact Assessment (PHIA) survey data spanning 9 surveys across 6 countries (Kenya, Malawi, Rwanda, Tanzania, Uganda, Zambia), assessing whether the predictive signal generalises beyond the original training population.

**Decision curve analysis** confirms positive net clinical benefit over treat-all and treat-none strategies across threshold probabilities 0.03–0.45.

---

## Local Recalibration

SmartDaaS is designed to be recalibrated to each new implementing partner's population prior to generating operational outputs. This design choice is reinforced by exploratory analysis (see `SmartDaaS_Socioeconomic_Exploratory_Note`) showing that the predictive contribution of additional candidate variables (e.g., socioeconomic factors) varies in direction and magnitude across countries and over time within the same country — indicating that a single fixed global model is unlikely to perform optimally across all deployment contexts without local adaptation.

---

## Data Quality Tiers

On upload, SmartDaaS evaluates the completeness of the input dataset against the 15-feature requirement and assigns a tier (Core / Standard / Enhanced) reflecting which outputs can be reliably generated. Datasets with substantial missingness in required fields are flagged rather than silently processed, and patient-level risk scores are not generated where required clinical variables are absent.

---

## Ethical Considerations

- **Data privacy:** SmartDaaS operates on de-identified, patient-level data. No direct identifiers (names, national IDs, contact information) are required or processed.
- **Data governance:** Use under a Data Use Agreement specifying scope, storage, retention, and deletion terms; the data-providing organisation retains full ownership of its data at all times.
- **Equity:** Subgroup fairness analysis found AUC consistent across sex, age, CD4 strata, and WHO clinical stage (range 0.815–0.866, max difference 0.051), with no systematic disadvantage to any demographic or clinical subgroup, including the highest-risk CD4 <200 group (AUC 0.829). The local recalibration design further mitigates the risk of a model trained on one population performing inequitably on another.
- **Transparency:** Per-patient SHAP explanations are provided so that the contributing factors behind each risk flag are visible to the user, rather than presenting an unexplained score.

---

## Known Limitations

- The current feature set is derived from baseline and most-recent-visit clinical data and does not include longitudinal visit/appointment-timing data (e.g., visit frequency, missed-appointment patterns, pharmacy pickup gaps). Literature on HIV treatment interruption prediction indicates visit-timing features are often strong predictors; incorporating them is a planned area of future work contingent on availability of longitudinal visit-level data from deploying programmes.
- The training dataset is drawn from a single country (Nigeria); while external validation spans 6 countries via PHIA survey data, the underlying training population is Nigerian.
- The model's primary outcome — poor adherence at last visit — is a leading indicator of, but not identical to, treatment interruption. `ArvAdherenceLatestLevel` was also evaluated as a candidate *feature* for predicting other outcomes and excluded in that context due to near-concurrent measurement with interruption; as the model's *target* variable here, with `had_interruption` as a temporally-prior predictor (mean 448 days separation), this construction is supported by ablation testing.
- As with any model trained on routine programme data, missingness in key clinical variables (e.g., CD4 count, 17–21% missing across facility types in the training data) may not be missing at random and could affect performance in populations with different missingness patterns.

---

## Recommendations

- Always recalibrate to the deploying programme's own population before relying on risk scores for operational decisions.
- Use SHAP explanations to support, not replace, clinical and programmatic judgement.
- Where longitudinal visit-level data is available, evaluate whether incorporating visit-timing features improves performance for that specific population — this is an open research question rather than an assumed improvement.

---

## Citation

Chinthala LK. "Real-World Validation of Machine Learning Models for HIV Treatment Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis of 192,732 Clinical Records." Under review, Scientific Reports, 2026.

Chinthala LK. "Facility-Level Structural Drivers of HIV Treatment Outcomes: A Multi-Level Analysis of 27,288 Patients from a Nigerian HIV Programme and Implications for PEPFAR and Global Fund Programming." Under review, BMJ Global Health, 2026.

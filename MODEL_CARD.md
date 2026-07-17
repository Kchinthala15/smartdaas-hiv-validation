# Model Card: SmartDaaS HIV Treatment Adherence Risk Model

## Model Details

**Developed by:** Lakshmi Kalyani Chinthala, MS Business Analytics (Golden Gate University), Independent HIV Analytics Researcher
**Model date:** 2026 (retrained 2026-07-16)
**Model type:** Random Forest classifier — `SimpleImputer(median) → StandardScaler → RandomForestClassifier`
**Model version:** v1.0
**Explainability:** SHAP (SHapley Additive exPlanations) — per-patient feature attribution
**License:** MIT
**Contact:** lkchinthala@smartdaas.org
**Platform:** https://smartdaas.org

---

## Intended Use

**Primary intended use:** Population-level risk prioritisation for HIV implementing partners. SmartDaaS ingests routine, de-identified patient-level programme data and produces a ranked list of patients at elevated risk of poor ART adherence, each accompanied by the principal contributing factors (via SHAP), to guide outreach prioritisation under resource constraints.

**Primary intended users:** HIV programme teams, strategic information units, and field outreach coordinators within PEPFAR- and Global Fund-supported implementing partner organisations.

**Out-of-scope uses:**
- Individual clinical treatment decisions. This is a prioritisation tool, not a diagnostic or clinical decision-support device, and has not been validated for that purpose.
- Use without local recalibration on the deploying programme's own data.
- **Any operational use at all, at present.** The model has never been externally validated on patient-level data from another health system, and the provenance of its training data is not documented (see below). It is a pre-pilot tool.
- Patients under 18. The model is trained on adults only.

---

## What the Model Predicts

The model predicts the probability that a patient is recorded as having **"Poor" ART adherence at their most recent clinical visit** (`ArvAdherenceLatestLevel = "Poor"`).

**This is a same-encounter prediction, not a forecast.** Six of the fifteen features — `MostRecentCd4Count`, `CD4_improvement`, `weight_change`, `opp_infection`, `tb_positive`, `stage_worsened` — are recorded at the same visit as the outcome. `Age` is recorded at data collection, also concurrent with the outcome. The model describes which patients *are* poorly adherent given a clinical picture; it does not predict who *will become* so.

Earlier versions of this card stated the model predicts adherence "at their next clinical assessment." That was incorrect. Nothing in the data supports a prospective claim.

Poor adherence is a recognised leading indicator of subsequent treatment interruption, but this model has not been validated against interruption, mortality, or any composite outcome.

---

## Training Data

**Source:** *Quality of Care dataset for HIV clients* (Kaggle, uploaded by user `iogbonna`, 2022-12-04).

**Provenance is not documented.** The deposit states no country, no programme, no custodian, no sampling frame, no collection methodology, and no ethics approval. Its licence reads "Data files © Original Authors" — no rights are granted. Internal evidence (clinical free-text referencing Nigerian states and localities; Federal/State/NGO funding categories) indicates Nigerian HIV programme records, but this is an inference drawn by the author, not a statement by the depositor. **Any claim that this dataset comes from "the Nigerian national HIV programme" is not supported by its source, and previous versions of this card made that claim in error.**

**Cohort after cleaning: 23,144 patients.** From 27,288 raw records:

| Step | n | removed |
|---|---|---|
| Raw records | 27,288 | — |
| Blank adherence value excluded (outcome unknown; previously coded as *adherent*) | 25,991 | 1,297 |
| Age filter, adults 18–100 | **23,144** | **2,847** |

The age filter removes 2,161 patients under 18 (paediatric WHO staging, CD4% and BMI are not comparable to adult measures), 671 with no recorded age, and 15 with implausible ages (up to 1,982,014).

**Poor-adherence prevalence: 3.67%** (850/23,144).

**Time period:** ART initiations span 2006–2018, but **99.4% of the cohort initiated between 2013 and 2017**. Fewer than 50 patients precede 2013.

**Population characteristics — computed on the 23,144-patient cohort**, not the raw file:

| | |
|---|---|
| Median age | 35 years (IQR 29–43) |
| Female | 16,101 (**69.6%**) |
| Median CD4 at ART start | 260 cells/µL |
| CD4 <200 at ART start | 37.4% of those with a CD4 recorded (20.3% missing) |
| WHO stage at ART start (of 22,440 non-missing) | **58.6% I · 22.9% II · 16.4% III · 2.2% IV** |
| Median diagnosis-to-ART interval | **29 days** (IQR 0–153; 21.0% missing) |
| Diagnosis-to-ART >90 days | 34.0% of non-missing |
| Any prior ART interruption | 2,065 (8.9%) |
| `MostRecentCd4Count` missing | 25.8% |
| `BMI_start` missing | 44.8% |

Previous versions of this card reported 61.4% female, mean age 35.2, a WHO distribution of 31.2/38.7/24.1/6.0, and a 74-day median diagnosis-to-ART interval with 47.3% exceeding 90 days. None of those is reproducible from the dataset.

**A note on the diagnosis-to-ART interval.** 4,940 raw records have an ART start date *preceding* the HIV confirmation date, by up to 3,006 days. `01_data_preprocessing.py` clipped these to 0, recoding them as same-day initiations, which pulls the apparent median down to 7 days. Excluding them instead gives **29 days**. Neither 74 (as published) nor 7 (a clipping artefact) is the right number.

**Known data quality issues:** 659 CD4 values are recorded as percentages rather than counts (`Cd4Unit = '%'`) and are excluded; height is recorded in mixed units and is missing for 43.7% of patients; 4,940 records have an ART start date preceding the HIV confirmation date; date columns are a mixture of strings and datetimes, so some day/month transpositions are likely.

**Resampling: none.** The model is trained without SMOTE. Earlier versions applied SMOTE to the full dataset before splitting, which invalidated the reported metrics. Removing it both fixed the leak and improved AUC.

---

## Features

15 features, from data routinely captured in standard HIV programme EMR systems:

| Feature | Description | Recorded at |
|---|---|---|
| `Age` | Patient age (years) | data collection |
| `sex_female` | Sex (binary: female = 1) | — |
| `Cd4AtStart` | CD4 count at ART initiation | ART start |
| `MostRecentCd4Count` | Most recent CD4 count | last visit |
| `CD4_improvement` | Change in CD4 since ART start | last visit |
| `stage_start_num` | WHO clinical stage at ART start (1–4) | ART start |
| `WeightAtStart` | Weight at ART initiation (kg) | ART start |
| `weight_change` | Change in weight since ART start | last visit |
| `BMI_start` | BMI at ART initiation | ART start |
| `days_to_ART` | Days from HIV diagnosis to ART initiation | ART start |
| `had_interruption` | Any prior treatment interruption (binary) | ever |
| `opp_infection` | Opportunistic infection present (binary) | last visit |
| `side_effects` | Reported ART side effects (binary) | last visit |
| `tb_positive` | TB status (binary) | last visit |
| `stage_worsened` | WHO stage worsened since ART start (binary) | last visit |

**On `had_interruption`:** the model's strongest predictor (feature importance 0.315; SHAP mean |value| 0.0176 — roughly twice the next feature). **It is required.** Its training median is 0, the protective value, so imputing it is indistinguishable from asserting the patient has never interrupted; doing so collapses the high-risk tier from ~800 patients to ~17. The platform refuses to score without it.

Earlier versions of this card cited a "mean 448-day separation" between `had_interruption` and the outcome as evidence against leakage. **That figure cannot be computed from this dataset** — it contains no last-visit date, so no predictor-to-outcome interval is derivable. The claim is withdrawn. The honest position: `had_interruption` records *any* prior interruption with no measurable temporal relationship to the outcome, and its dominance should be interpreted with that in mind.

---

## Quantitative Performance

| Metric | Value | Evaluation set |
|---|---|---|
| 10-fold CV AUC | **0.801 ± 0.023** | 23,144 adults, natural class distribution (3.67%) |
| CV Brier score | 0.032 | as above |
| **Temporal validation AUC** | **0.806 (95% CI 0.774–0.837)** | trained on ART initiations ≤ Sept 2016; tested on 6,942 later initiators |
| Temporal Brier score | 0.027 | as above |
| Sensitivity @ threshold 0.15 | 27.7% | temporal hold-out |
| Specificity @ threshold 0.15 | 97.0% | temporal hold-out |
| PPV @ threshold 0.15 | 21.9% (base rate 3.0%) | temporal hold-out |
| Sensitivity @ threshold 0.075 | 52.4% (flags 11.6% of cohort) | temporal hold-out |
| External validation | **none** | — |

**Internal and temporal estimates agree (0.801 vs 0.806).** That agreement is the substantive finding. A model scoring far higher internally than on later patients is usually leaking information between training and test rather than generalising; agreement across an internal and a temporal split is the evidence that the signal is real.

**Threshold choice is a programme decision, not a statistical one.** It depends on how many patients an outreach team can contact. 0.15 flags 3.7% of a cohort and concentrates risk roughly sevenfold; 0.075 flags 11.6% and catches half the poor-adherence patients.

**Field context:** the most recent systematic review of ML models for HIV treatment interruption (Kwarah et al., *BMC Global and Public Health*, 2025; 12 models) reports a mean internal AUC of 0.668. Those models predict interruption rather than adherence, so the comparison is indicative rather than like-for-like.

### Numbers withdrawn

The following appeared in earlier versions of this card and are not valid. They were produced by a pipeline that applied SMOTE to the full dataset before the train/test split, so the reported "hold-out" of 10,540 patients was 50% synthetic minority cases interpolated from the training data, not a natural-prevalence sample:

> CV AUC 0.975 · hold-out AUC 0.973 (n=10,540) · sensitivity 87.3% · specificity 95.7% · Brier 0.079 · ablation AUC 0.963

The "0.203 reduction from CV to temporal AUC" was described as genuine distributional shift. It was the leak. With the corrected pipeline the difference is +0.006.

**External validation AUC 0.769 (PHIA) is also withdrawn as a validation claim.** PHIA surveys supply at most 8 of the 15 features (`constants.py` documents this) and use a different outcome (self-reported missed doses). A 15-feature model cannot be applied to data lacking its inputs. The PHIA work is a separate analysis on different features against a different outcome, and will be reported as such.

---

## Fairness and Subgroup Performance

Subgroup AUC on the temporal hold-out (n=6,942, 206 poor-adherence events):

| Subgroup | n | events | AUC |
|---|---|---|---|
| Overall | 6,942 | 206 | **0.806** |
| Female | 4,751 | 128 | 0.795 |
| Male | 2,191 | 78 | 0.826 |
| Age <30 | 1,802 | 51 | 0.801 |
| Age 30–49 | 4,207 | 133 | 0.802 |
| Age 50+ | 933 | 22 | 0.850 |
| **CD4 <200** | 1,760 | 51 | **0.752** |
| CD4 200–499 | 2,103 | 22 | 0.786 |
| **CD4 500+** | 1,036 | 10 | **0.719** |
| WHO I–II | 5,591 | 147 | 0.815 |
| WHO III–IV | 1,117 | 50 | 0.759 |

**Range 0.719–0.850; maximum difference 0.131.**

**The model performs worst in the group with the most advanced disease** (CD4 <200: 0.752, against 0.806 overall) and in patients presenting at WHO stage III–IV (0.759). Earlier versions of this card reported a range of 0.815–0.866, a maximum difference of 0.051, "no systematic disadvantage to any demographic or clinical subgroup," and an AUC of 0.829 for the CD4 <200 group. None of those figures is reproducible, and the direction of the CD4 finding is the opposite of what was claimed.

Several subgroups carry few events (CD4 500+ has 10), so confidence intervals are wide and these differences are not established as significant. They are reported because they are what the data shows, and they should be tested properly in any pilot.

---

## Local Recalibration

SmartDaaS is designed to be recalibrated to each implementing partner's population before generating operational outputs. The platform blocks recalibration where the supplied outcome is implausible for ART adherence (prevalence outside 0.5–25%), because fitting a calibrator to a different construct — loss-to-follow-up, mortality, or a composite — re-fits the model to a different question rather than merely producing a poor number.

---

## Data Quality Tiers

On upload, SmartDaaS assigns a tier (Core / Standard / Enhanced) reflecting which outputs can be generated. Datasets with substantial missingness are flagged rather than silently processed; patient-level risk scores are not generated where required clinical variables are absent. Features missing from an upload are imputed with the training-cohort median — never with zero, which is the protective value for four of the binary features — and every imputation is disclosed to the user.

---

## Ethical Considerations

- **Data privacy:** SmartDaaS processes de-identified patient-level data. No direct identifiers are required.
- **Data governance:** use under a Data Use Agreement; the data-providing organisation retains ownership at all times.
- **Training data:** the training dataset's free-text clinical fields contain unredacted narrative detail (named facilities, patient travel, family circumstances). These are quasi-identifiers. The model does not use them, but their presence means the source deposit should not be characterised as fully de-identified, and its public availability may not have been authorised by whoever holds the data.
- **Equity:** see Fairness above. The model is weakest where patients are sickest. This is a limitation, not a resolved question.
- **Transparency:** per-patient SHAP explanations accompany every risk flag.

---

## Known Limitations

1. **No external validation.** The model has never been applied to patient-level data from another programme, country, or health system.
2. **Provenance not documented.** The training data's sampling frame, custodian, and collection methodology are unknown.
3. **Same-encounter prediction.** Seven of fifteen features are measured concurrently with the outcome. This is a description of current adherence status, not a forecast.
4. **Single-country, single-era.** 99.8% of the cohort initiated ART 2013–2017 in one country.
5. **Outcome.** Poor adherence at last visit is a leading indicator of, not identical to, treatment interruption. The model has not been validated against interruption or mortality.
6. **Missingness.** CD4 is missing for 17–21% of the training cohort and is unlikely to be missing at random.
7. **No visit-timing features.** Visit frequency, missed appointments and pharmacy pickup gaps are often strong predictors of interruption and are absent from this dataset.
8. **Weakest where it matters most.** See Fairness.

---

## Reproducibility

`retrain_v1.py` reproduces the model end to end from the source spreadsheet: cohort construction, training, temporal validation, threshold derivation, and the saved artefacts. Every cleaning decision is an explicit, editable line.

---

## Citation

**Software:**

> Chinthala LK. *SmartDaaS: an explainable risk model for ART adherence in HIV programme settings.* SmartDaaS LLC, 2026. https://smartdaas.org

**Publications:** none. Two manuscripts were previously listed here. The first, on this model, was withdrawn from *Scientific Reports* in July 2026 after the data leakage error above was identified. The second, on facility-level outcomes, was rejected by *BMJ Open* in July 2026 on the grounds that the data's provenance, sampling and ownership were unclear — an objection this card now accepts as correct. Neither should be cited.

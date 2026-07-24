"""
SmartDaaS v1.0 — Constants
Lakshmi Kalyani Chinthala | Founder & Independent Researcher
ORCID: 0009-0009-8736-6673

All model features, column aliases, labels, tier definitions,
data quality thresholds, and recalibration constants.
Centralised here so any reviewer can immediately understand
the model's inputs and configuration without reading app logic.
"""

# ─────────────────────────────────────────────────────────────
# MODEL FEATURES
# The 15 clinical variables used by the Random Forest classifier.
# Order matters — must match the order used during training.
# ─────────────────────────────────────────────────────────────
FEATURES = [
    'Age', 'sex_female', 'Cd4AtStart', 'MostRecentCd4Count', 'CD4_improvement',
    'stage_start_num', 'WeightAtStart', 'weight_change', 'BMI_start', 'days_to_ART',
    'had_interruption', 'opp_infection', 'side_effects', 'tb_positive', 'stage_worsened'
]

# ─────────────────────────────────────────────────────────────
# TRAINING MEDIANS
# The exact values the model's own internal imputer uses. Any feature
# missing from an upload is filled with these — NOT with 0.
#
# 0 is not a neutral value. For had_interruption, opp_infection,
# tb_positive and stage_worsened, 0 is the PROTECTIVE value: zero-filling
# tells the model the patient is fine.
#
# Regenerate with retrain_v1.py whenever the model is retrained:
#   model.named_steps['imp'].statistics_
# ─────────────────────────────────────────────────────────────
TRAINING_MEDIANS = {
    'Age':                     35.0000,
    'sex_female':               1.0000,
    'Cd4AtStart':             261.0000,
    'MostRecentCd4Count':     387.0000,
    'CD4_improvement':         52.0000,
    'stage_start_num':          1.0000,
    'WeightAtStart':           57.0000,
    'weight_change':            2.0000,
    'BMI_start':               22.0604,
    'days_to_ART':             29.0000,
    'had_interruption':         0.0000,
    'opp_infection':            0.0000,
    'side_effects':             0.0000,
    'tb_positive':              0.0000,
    'stage_worsened':           0.0000,
}

# ─────────────────────────────────────────────────────────────
# CRITICAL FEATURES — must be present in the upload, never imputed.
#
# had_interruption is the model's dominant predictor (importance 0.381)
# and its training median is 0.0 — the protective value. Imputing it is
# indistinguishable from zero-filling it, and collapses the HIGH tier
# from ~800 patients to ~17 while sending ~238 genuinely poor-adherent
# patients to LOW. If it is absent, refuse to score.
# ─────────────────────────────────────────────────────────────
CRITICAL_FEATURES = {'had_interruption'}

# Model is trained on adults 18-100. Under-18s are out of scope:
# paediatric WHO staging differs, CD4% replaces absolute count, and BMI
# needs age-sex z-scores. Was 15; corrected to match the training cohort.
PEDIATRIC_AGE_CUTOFF = 18

# ─────────────────────────────────────────────────────────────
# COLUMN ALIASES
# Maps common alternative column names → expected feature names.
# Keys = what users / EMR systems might call them (lowercased).
# Values = what the model expects.
# ─────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    'age': 'Age',
    'patient_age': 'Age',
    'sex': 'sex_female',
    'gender': 'sex_female',
    'female': 'sex_female',
    'is_female': 'sex_female',
    'cd4_at_start': 'Cd4AtStart',
    'cd4atstart': 'Cd4AtStart',
    'cd4_start': 'Cd4AtStart',
    'baseline_cd4': 'Cd4AtStart',
    'most_recent_cd4': 'MostRecentCd4Count',
    'mostrecentcd4': 'MostRecentCd4Count',
    'cd4_recent': 'MostRecentCd4Count',
    'latest_cd4': 'MostRecentCd4Count',
    'cd4_improvement': 'CD4_improvement',
    'cd4improvement': 'CD4_improvement',
    'cd4_change': 'CD4_improvement',
    'who_stage': 'stage_start_num',
    'stage_start': 'stage_start_num',
    'who_clinical_stage': 'stage_start_num',
    'clinical_stage': 'stage_start_num',
    'weight_at_start': 'WeightAtStart',
    'weightatstart': 'WeightAtStart',
    'baseline_weight': 'WeightAtStart',
    'weight_start': 'WeightAtStart',
    'weightchange': 'weight_change',
    'weight_delta': 'weight_change',
    'bmi_start': 'BMI_start',
    'bmi_at_start': 'BMI_start',
    'baseline_bmi': 'BMI_start',
    'days_to_art': 'days_to_ART',
    'diagnosis_to_art': 'days_to_ART',
    'days_diagnosis_to_art': 'days_to_ART',
    'art_delay': 'days_to_ART',
    'had_interruption': 'had_interruption',
    'art_interruption': 'had_interruption',
    'interruption': 'had_interruption',
    'prior_interruption': 'had_interruption',
    'opp_infection': 'opp_infection',
    'opportunistic_infection': 'opp_infection',
    'oi': 'opp_infection',
    'side_effects': 'side_effects',
    'side_effect': 'side_effects',
    'adverse_effects': 'side_effects',
    'tb_positive': 'tb_positive',
    'tb': 'tb_positive',
    'tuberculosis': 'tb_positive',
    'tb_status': 'tb_positive',
    'stage_worsened': 'stage_worsened',
    'stage_worsening': 'stage_worsened',
    'clinical_deterioration': 'stage_worsened',

    # ── International / WHO-aligned variants ──────────────────
    'age_years': 'Age',
    'age_at_art': 'Age',
    'age_at_enrollment': 'Age',
    'patient_age_years': 'Age',

    'sex_at_birth': 'sex_female',
    'biological_sex': 'sex_female',
    'patient_sex': 'sex_female',
    'gender_female': 'sex_female',

    'cd4_baseline': 'Cd4AtStart',
    'cd4_count_at_art': 'Cd4AtStart',
    'cd4_art_initiation': 'Cd4AtStart',
    'cd4_enrol': 'Cd4AtStart',
    'cd4_enrollment': 'Cd4AtStart',
    'cd4_last': 'MostRecentCd4Count',
    'last_cd4': 'MostRecentCd4Count',
    'current_cd4': 'MostRecentCd4Count',
    'cd4_follow_up': 'MostRecentCd4Count',
    'cd4_delta': 'CD4_improvement',
    'cd4_gain': 'CD4_improvement',

    # WHO Stage — Kenya NASCOP / Uganda DHIS2 / Malawi HMIS variants
    'who_stage_at_art': 'stage_start_num',
    'clinical_stage_at_art': 'stage_start_num',
    'who_clinical_stage_at_start': 'stage_start_num',
    'art_stage': 'stage_start_num',
    'stage': 'stage_start_num',
    'hiv_stage': 'stage_start_num',

    # Weight / BMI
    'weight_kg': 'WeightAtStart',
    'weight_baseline': 'WeightAtStart',
    'art_weight': 'WeightAtStart',
    'weight_kg_change': 'weight_change',
    'weight_gain_loss': 'weight_change',
    'bmi': 'BMI_start',
    'bmi_baseline': 'BMI_start',
    'body_mass_index': 'BMI_start',

    # Days to ART
    'art_initiation_delay': 'days_to_ART',
    'days_hiv_to_art': 'days_to_ART',
    'time_to_art': 'days_to_ART',
    'linkage_days': 'days_to_ART',

    # Interruption
    'treatment_interruption': 'had_interruption',
    'art_gap': 'had_interruption',
    'lost_to_followup': 'had_interruption',
    'ltfu': 'had_interruption',

    # OI / TB
    'oi_present': 'opp_infection',
    'opportunistic_infection_present': 'opp_infection',
    'tb_coinfection': 'tb_positive',
    'tuberculosis_status': 'tb_positive',
    'tb_screen_positive': 'tb_positive',

    # ART status inference helpers (not model features — used for tier detection)
    'art_start_date': '__art_inferred__',
    'date_art_started': '__art_inferred__',
    'art_initiation_date': '__art_inferred__',
    'regimen_at_start': '__art_inferred__',
    'current_regimen': '__art_inferred__',
    'art_regimen': '__art_inferred__',
    'arvs': '__art_inferred__',

    # ── PHIA / population-survey dataset compatibility ─────────────────────
    # IMPORTANT VALIDATION CAVEATS:
    #   1. cd4count = most-recent survey CD4, NOT CD4 at ART start.
    #   2. tbdiagn = self-reported TB, not lab-confirmed.
    #   3. arvinterr = direct interruption variable (PHIA 2020+ only).
    #   4. arvsmissdays>0 is a proxy for had_interruption.
    #   5. PHIA supports 8/15 features only — this is NOT sufficient to
    #      externally validate the 15-feature model. Any PHIA analysis is a
    #      separate study, not a validation of this model.
    'cd4count': 'MostRecentCd4Count',
    'cd4cat': 'MostRecentCd4Count',
    'arvinterr': 'had_interruption',
    'tbdiagn': 'tb_positive',
    'tbclinvisit': 'tb_positive',
}

# ─────────────────────────────────────────────────────────────
# FEATURE DISPLAY LABELS
# Human-readable names shown in the UI and reports.
# ─────────────────────────────────────────────────────────────
FEAT_LABELS = {
    'Age': 'Age (years)',
    'sex_female': 'Sex (Female=1)',
    'Cd4AtStart': 'CD4 at ART Start',
    'MostRecentCd4Count': 'Most Recent CD4',
    'CD4_improvement': 'CD4 Improvement',
    'stage_start_num': 'WHO Stage (1–4)',
    'WeightAtStart': 'Weight at Start (kg)',
    'weight_change': 'Weight Change (kg)',
    'BMI_start': 'BMI at Start',
    'days_to_ART': 'Days: Diagnosis to ART',
    'had_interruption': 'Prior ART Interruption',
    'opp_infection': 'Opportunistic Infection',
    'side_effects': 'Side Effects Reported',
    'tb_positive': 'TB Positive',
    'stage_worsened': 'Clinical Stage Worsened',
}

FEAT_DESCRIPTIONS = {
    'Age': 'Patient age in years at ART initiation',
    'sex_female': 'Binary: 1=Female, 0=Male',
    'Cd4AtStart': 'CD4 cell count (cells/µL) at ART start',
    'MostRecentCd4Count': 'Most recent CD4 count (cells/µL)',
    'CD4_improvement': 'Change in CD4 count since ART start (can be negative)',
    'stage_start_num': 'WHO clinical stage at ART initiation (1, 2, 3, or 4)',
    'WeightAtStart': 'Patient weight in kg at ART initiation',
    'weight_change': 'Change in weight (kg) since ART start (can be negative)',
    'BMI_start': 'BMI at ART initiation (kg/m²)',
    'days_to_ART': 'Days between HIV diagnosis and ART start (0 = same day)',
    'had_interruption': 'Binary: 1=prior ART interruption documented, 0=none',
    'opp_infection': 'Binary: 1=opportunistic infection documented, 0=none',
    'side_effects': 'Binary: 1=side effects reported, 0=none',
    'tb_positive': 'Binary: 1=TB positive, 0=negative/unknown',
    'stage_worsened': 'Binary: 1=WHO stage worsened since ART start, 0=stable/improved',
}

FEAT_RANGES = {
    'Age': '18–80',
    'sex_female': '0 or 1',
    'Cd4AtStart': '0–1500',
    'MostRecentCd4Count': '0–1500',
    'CD4_improvement': '-500 to +800',
    'stage_start_num': '1, 2, 3, or 4',
    'WeightAtStart': '30–150',
    'weight_change': '-30 to +30',
    'BMI_start': '12–50',
    'days_to_ART': '0–3650',
    'had_interruption': '0 or 1',
    'opp_infection': '0 or 1',
    'side_effects': '0 or 1',
    'tb_positive': '0 or 1',
    'stage_worsened': '0 or 1',
}

# ─────────────────────────────────────────────────────────────
# CLINICAL INTERVENTION RECOMMENDATIONS
# Shown per risk tier on the Patient Risk and SHAP pages.
# ─────────────────────────────────────────────────────────────
INTERVENTIONS = {
    'HIGH': [
        "🔴 Schedule urgent adherence counselling within 48 hours",
        "🔴 Activate peer navigator support",
        "🔴 Review regimen tolerability and side effects",
        "🔴 Arrange pill count / home visit",
        "🔴 Escalate to clinical officer for review",
    ],
    'MEDIUM': [
        "🟡 Schedule adherence counselling within 2 weeks",
        "🟡 Review last clinic attendance pattern",
        "🟡 Assess social support and transport barriers",
        "🟡 Send SMS reminder for next appointment",
    ],
    'LOW': [
        "🟢 Continue standard care pathway",
        "🟢 Routine follow-up at next scheduled visit",
        "🟢 Reinforce adherence education at next visit",
    ]
}

# ─────────────────────────────────────────────────────────────
# ECONOMIC CONSTANTS
# ─────────────────────────────────────────────────────────────
COST_PER_POOR_OUTCOME = 1850  # USD — Menzies et al. AIDS 2011, Nigeria-specific PEPFAR data

# ─────────────────────────────────────────────────────────────
# RISK THRESHOLDS
# Default operating thresholds — overridden by local calibration.
# Derived from the temporal holdout (train ≤ Sept 2016, test after;
# n=6,942, prevalence 2.97%).
#
#   HIGH   (>= 0.1291): Sensitivity 29.3%, Specificity 97.0%, PPV 22.8%
#                       flags 3.7% of patients (237 of 6,382)
#   MEDIUM (>= 0.0474): Sensitivity 54.3%, Specificity 89.7%, PPV 13.5%
#                       flags 11.6% of patients (741 of 6,382)
#
# v1.1: these are on the CALIBRATED probability scale. They were chosen to
# preserve v1's FLAGGING RATE, so clinic workload is unchanged — what changes
# is which patients are flagged, not how many.
#
# Lower the thresholds to catch more patients at the cost of more
# outreach volume. This is a programme capacity decision.
# ─────────────────────────────────────────────────────────────
BASELINE_THRESHOLD = 0.1291   # HIGH
MEDIUM_THRESHOLD   = 0.0474   # MEDIUM

# ─────────────────────────────────────────────────────────────
# DATA TIER DEFINITIONS
# Determines what analysis is available based on uploaded features.
# ─────────────────────────────────────────────────────────────
TIER_CORE_REQUIRED = {'Age', 'sex_female'}

TIER_STANDARD_QUALIFYING = {
    'Cd4AtStart', 'MostRecentCd4Count', 'stage_start_num',
    'days_to_ART', 'tb_positive'
}
TIER_STANDARD_MIN = 3

TIER_ENHANCED_QUALIFYING = {
    'CD4_improvement', 'WeightAtStart', 'weight_change', 'BMI_start',
    'had_interruption', 'opp_infection', 'side_effects', 'stage_worsened'
}
TIER_ENHANCED_MIN = 4

# ─────────────────────────────────────────────────────────────
# ART STATUS INFERENCE TRIGGERS
# Column names that indicate the upload contains ART patient data
# even when no explicit art_status column is present.
# ─────────────────────────────────────────────────────────────
ART_INFERENCE_TRIGGERS_LOWER = {
    'art_start_date', 'date_art_started', 'art_initiation_date',
    'dateartstarted', 'regimen_at_start', 'regimenAtStart'.lower(),
    'current_regimen', 'art_regimen', 'arvs',
    'days_to_art', 'days_to_ART'.lower(), 'diagnosis_to_art',
    'cd4atstart', 'cd4_at_start', 'Cd4AtStart'.lower(),
    'stage_start_num', 'who_stage', 'clinical_stage',
    # PHIA / population-survey ART indicators
    'arvscurrent',
    'arvstakenev',
    'arvftm',
    'arvfty',
    'arvinterr',
    'arvsmissdays',
    'artselfreported',
    'artinitiated12months',
}

# ─────────────────────────────────────────────────────────────
# DATA QUALITY — VALID RANGES PER FEATURE
# Used for out-of-range detection in data quality screening.
# NOTE: Age lower bound is 18. The model is trained on adults only
# (18–100). Paediatric patients are out of scope and should be
# flagged rather than scored.
# ─────────────────────────────────────────────────────────────
FEATURE_VALID_RANGES = {
    'Age':               (18, 100),
    'sex_female':        (0, 1),
    'Cd4AtStart':        (0, 2000),
    'MostRecentCd4Count':(0, 2000),
    'CD4_improvement':   (-1500, 1500),
    'stage_start_num':   (1, 4),
    'WeightAtStart':     (10, 300),
    'weight_change':     (-100, 100),
    'BMI_start':         (8, 80),
    'days_to_ART':       (0, 10000),
    'had_interruption':  (0, 1),
    'opp_infection':     (0, 1),
    'side_effects':      (0, 1),
    'tb_positive':       (0, 1),
    'stage_worsened':    (0, 1),
}

# Clinically important features — missing these hurts quality grade more
HIGH_IMPORTANCE_FEATURES = {
    'Cd4AtStart', 'MostRecentCd4Count', 'stage_start_num',
    'had_interruption', 'tb_positive'
}

# ─────────────────────────────────────────────────────────────
# RECALIBRATION CONSTANTS
# ─────────────────────────────────────────────────────────────
RECAL_MIN_PATIENTS     = 200   # Minimum patients for recalibration
RECAL_MIN_POS_EVENTS   = 30    # Minimum positive outcome events
RECAL_MAX_OUTCOME_MISS = 0.40  # Maximum allowed outcome missingness
RECAL_MAX_FEAT_MISS    = 0.50  # Maximum allowed feature missingness (warning only)
RECAL_ISOTONIC_MIN     = 500   # Min patients to use isotonic vs Platt scaling
BOOTSTRAP_N            = 1000  # Bootstrap iterations for AUC confidence intervals

# Temporal holdout AUC — primary operational estimate.
# 15 features, adults 18–100, n=21,273 (773 poor adherence, 3.63%).
# Trained on ART initiations <= 2016-09-14; tested on 6,382 later initiators.
# Sigmoid-calibrated. 95% CI 0.778–0.844. 10-fold CV on the same cohort: 0.802.
#
# NOTE ON THE SPLIT: this splits on ART INITIATION date. The outcome
# (ArvAdherenceLatestLevel) has no timestamp anywhere in the source data, so
# this tests generalisation across initiation eras and follow-up durations,
# NOT calendar time. It is not a prospective simulation.
BASELINE_AUC           = 0.8124
BASELINE_AUC_CI        = (0.7783, 0.8442)
BASELINE_CV_AUC        = 0.8019
BASELINE_BRIER         = 0.0259
CALIBRATION_SLOPE      = 1.096   # 1.000 is perfect; v1 was 1.402 (uncalibrated)

# ─────────────────────────────────────────────────────────────
# MODEL VERSION AND LOCK
# Regenerate with retrain_v1_1.py; the lock hash must reproduce.
# ─────────────────────────────────────────────────────────────
MODEL_VERSION          = 'v1.1'
MODEL_LOCK_SHA256      = '8f8b4d23aafac6fcba5824e0a837bee8dab8a09a80123e9e20c5e0ffa47b359b'

# TB status values that count as positive. v1 searched for 'positive|active',
# which matched NOTHING in the source column, making tb_positive a constant
# zero for every patient. 'IPT' is preventive therapy (patients WITHOUT active
# TB) and 'Presumptive TB' is unconfirmed — both correctly stay 0.
TB_POSITIVE_VALUES     = {'confirmed tb', 'tb treatment'}

# ─────────────────────────────────────────────────────────────
# OUTCOME COLUMN DETECTION
# Recognised outcome column name hints (lowercased, stripped).
# ─────────────────────────────────────────────────────────────
OUTCOME_NAME_HINTS = {
    'poor_outcome', 'composite_outcome', 'outcome', 'art_outcome',
    'poor_adherence', 'treatment_interrupted', 'interruption',
    'had_poor_outcome', 'bad_outcome', 'adverse_outcome',
    'ltfu', 'lost_to_followup', 'treatment_failure',
    'mortality', 'dead', 'patient_dead', 'died',
    'art_interruption', 'non_adherent', 'not_adherent',
    'label', 'target', 'y', 'outcome_binary',
}

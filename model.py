"""
SmartDaaS v1.0 — Model
Lakshmi Kalyani Chinthala | Founder & Independent Researcher

Model loading, prediction, SHAP explanation, and sample data generation.

Exports:
    model, SCALER, CLF, MODEL_AUC, X_DEMO, Y_DEMO, MODEL_OK, MODEL_ERR
    run_predictions(df_in)
    safe_feature_importances()
    compute_shap(X_raw, n_sample=80)
    compute_shap_single(x_row)
    generate_sample_csv(n=20, seed=42)
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

from constants import (
    FEATURES, BASELINE_THRESHOLD, MEDIUM_THRESHOLD, TRAINING_MEDIANS,
)
from pipeline import apply_calibration

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with open('cv_results.pkl', 'rb') as f:
            cv = pickle.load(f)
        with open('prepped_data.pkl', 'rb') as f:
            prep = pickle.load(f)
        model = cv['rf_model']          # calibrated — USE FOR SCORING
        shap_base = cv['shap_model']    # uncalibrated — EXPLANATION ONLY

        # v1.1: rf_model is now a CalibratedClassifierCV, which has no
        # .named_steps. SHAP needs the raw pipeline parts, so they come from
        # shap_base instead. Never score with shap_base — its probabilities are
        # on the old compressed scale and the thresholds do not apply to it.
        from sklearn.calibration import CalibratedClassifierCV
        if not isinstance(model, CalibratedClassifierCV):
            raise RuntimeError(
                "cv_results.pkl rf_model is not calibrated. The thresholds in "
                "constants.py are on the calibrated scale and do not apply.")

        imputer = shap_base.named_steps.get('imp', None)
        scaler = shap_base.named_steps['sc']
        clf = shap_base.named_steps['clf']
        auc = float(cv['auc'])
        X_demo = prep['X']
        y_demo = prep['y']
        return model, imputer, scaler, clf, auc, X_demo, y_demo, True, ""
    except Exception as e:
        return None, None, None, None, 0.0, None, None, False, str(e)

model, IMPUTER, SCALER, CLF, MODEL_AUC, X_DEMO, Y_DEMO, MODEL_OK, MODEL_ERR = load_model()


def _prep_for_shap(X):
    """Apply the pipeline's imputer before scaling.

    compute_shap() calls SCALER.transform() directly, skipping the pipeline's
    imputer step. Any NaN would pass straight through and produce NaN SHAP
    values, which have the right length and so survive the len() guard below —
    the page would silently fall back to global feature importances while
    looking like per-patient explanations.
    """
    import numpy as _np
    X = _np.asarray(X, dtype=float)
    if IMPUTER is not None:
        X = IMPUTER.transform(X)
    return SCALER.transform(X)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# DATA PIPELINE — imported from pipeline.py
# ─────────────────────────────────────────────────────────────
def run_predictions(df_in):
    # Safety net: ensure all features exist and are numeric before indexing.
    # IMPORTANT: derive_engineered_features() must be called before this function
    # to compute CD4_improvement, weight_change, had_interruption, stage_worsened.
    # Calling run_predictions() directly without that step will silently use 0.0
    # for any missing engineered features, which reduces prediction accuracy.
    # The main upload pipeline (Patient Risk page) always calls
    # derive_engineered_features() first — this guard is a safety net only.
    df_in = df_in.copy()
    _n_imputed = 0
    for f in FEATURES:
        if f not in df_in.columns:
            df_in[f] = TRAINING_MEDIANS[f]
            _n_imputed += 1
        else:
            df_in[f] = pd.to_numeric(df_in[f], errors='coerce').fillna(TRAINING_MEDIANS[f])
    if _n_imputed > 0:
        import warnings as _warnings
        _warnings.warn(
            f"run_predictions: {_n_imputed} feature(s) were missing and filled with "
            f"the training median. Call derive_engineered_features() before "
            f"run_predictions() so features are derived and critical gaps are caught.",
            stacklevel=2
        )
    X = df_in[FEATURES].values.astype(float)
    probs_raw = model.predict_proba(X)[:, 1]

    # Apply local calibration if recalibration has been run this session
    calibrator = st.session_state.get('calibrator', None)
    use_isotonic = st.session_state.get('use_isotonic', False)
    local_threshold = st.session_state.get('local_threshold', BASELINE_THRESHOLD)

    if calibrator is not None:
        probs = apply_calibration(probs_raw, calibrator, use_isotonic)
        threshold_lo = local_threshold * 0.6   # ~40% of threshold = medium
        threshold_hi = local_threshold          # local optimal threshold
    else:
        probs = probs_raw
        # Derived from the temporal holdout — see constants.py. The previous
        # values (0.4 / 0.7) belonged to the SMOTE-trained model, whose scores
        # were inflated to a 50% prevalence scale. The current model's highest
        # score is ~0.39, so 0.7 would place every patient in LOW.
        threshold_lo = MEDIUM_THRESHOLD
        threshold_hi = BASELINE_THRESHOLD

    df_in = df_in.copy()
    df_in['risk_score'] = probs
    df_in['risk_pct'] = (probs * 100).round(1)
    df_in['risk_label'] = pd.cut(
        probs,
        bins=[-0.001, threshold_lo, threshold_hi, 1.001],
        labels=['LOW', 'MEDIUM', 'HIGH']
    ).astype(str)
    return df_in, X, probs


def safe_feature_importances():
    """Always returns a valid 1D array of length len(FEATURES). Never crashes."""
    n = len(FEATURES)
    try:
        if CLF is not None and hasattr(CLF, 'feature_importances_'):
            fi = np.array(CLF.feature_importances_).flatten()
            if len(fi) == n:
                return fi
    except Exception:
        pass
    # Ultimate fallback: uniform importances
    return np.ones(n) / n


def compute_shap(X_raw, n_sample=80):
    """Always returns (mean_shap array of length len(FEATURES), explainer_or_None)."""
    n = len(FEATURES)
    try:
        import shap
        if SCALER is None or CLF is None:
            raise ValueError("Model components not loaded")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_raw), min(n_sample, len(X_raw)), replace=False)
        X_s = _prep_for_shap(X_raw[idx])
        exp = shap.TreeExplainer(CLF)
        sv = exp.shap_values(X_s)
        # Handle both old (list) and new (array) SHAP output formats
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        result = np.abs(sv).mean(axis=0).flatten()
        if len(result) == n:
            return result, exp
    except Exception:
        pass
    return safe_feature_importances(), None


def compute_shap_single(x_row):
    """Always returns (shap array of length len(FEATURES), is_real_shap bool)."""
    n = len(FEATURES)
    try:
        import shap
        if SCALER is None or CLF is None:
            raise ValueError("Model components not loaded")
        x_scaled = _prep_for_shap(x_row.reshape(1, -1))
        exp = shap.TreeExplainer(CLF)
        sv = exp.shap_values(x_scaled)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        result = np.array(sv).flatten()
        if len(result) == n:
            return result, True
    except Exception:
        pass
    return safe_feature_importances(), False


def generate_sample_csv(n=20, seed=42):
    """Generate a realistic sample CSV template."""
    rng = np.random.RandomState(seed)
    data = {
        'patient_id': [f'PT-{i:04d}' for i in range(1, n + 1)],
        'Age': rng.randint(20, 65, n).astype(float),
        'sex_female': rng.randint(0, 2, n).astype(float),
        'Cd4AtStart': rng.randint(50, 800, n).astype(float),
        'MostRecentCd4Count': rng.randint(100, 900, n).astype(float),
        'CD4_improvement': rng.randint(-100, 400, n).astype(float),
        'stage_start_num': rng.randint(1, 5, n).astype(float),
        'WeightAtStart': np.round(rng.uniform(45, 95, n), 1),
        'weight_change': np.round(rng.uniform(-5, 10, n), 1),
        'BMI_start': np.round(rng.uniform(17, 32, n), 1),
        'days_to_ART': rng.randint(0, 365, n).astype(float),
        'had_interruption': rng.randint(0, 2, n).astype(float),
        'opp_infection': rng.randint(0, 2, n).astype(float),
        'side_effects': rng.randint(0, 2, n).astype(float),
        'tb_positive': rng.randint(0, 2, n).astype(float),
        'stage_worsened': rng.randint(0, 2, n).astype(float),
    }
    return pd.DataFrame(data)

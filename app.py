"""
SmartDaaS v1.0 — AI-Powered HIV Programme Intelligence Platform
Lakshmi Kalyani Chinthala | Independent Researcher & Founder
ORCID: 0009-0009-8736-6673

SmartDaaS is a decision-support platform for HIV programme intelligence,
operational analytics, and facility benchmarking. Designed for PEPFAR
implementing partners, Global Fund grantees, and national HIV programme officers.
Not intended to replace clinical judgment. Local validation required before
deployment in real-world programme environments.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from io import StringIO, BytesIO
import datetime
import os
import hashlib
import uuid
from dhis2_connector import render_dhis2_connector

def get_secret(key, default=None):
    """
    Get secret from st.secrets (Streamlit Cloud) or
    os.environ (Render/Docker) with graceful fallback.
    """
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

# ─────────────────────────────────────────────────────────────
# SUPABASE CONNECTION (graceful fallback if not configured)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_supabase():
    try:
        from supabase import create_client
        url = get_secret("SUPABASE_URL")
        key = get_secret("SUPABASE_KEY")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None

def _hash_filename(filename):
    """Return SHA-256 hash of filename. Prevents logging sensitive info."""
    if not filename:
        return "no_file"
    return hashlib.sha256(filename.encode()).hexdigest()[:16]


def _get_session_id():
    """Return stable session ID for this Streamlit session."""
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())[:12]
    return st.session_state['session_id']


def log_event(supabase, event_type, payload):
    """
    Central audit logging function. Fails silently.
    Never logs patient data — only aggregate metadata.
    """
    if supabase is None:
        return
    try:
        record = {
            "event_at":   datetime.datetime.utcnow().isoformat(),
            "event_type": str(event_type),
            "session_id": _get_session_id(),
        }
        record.update({k: v for k, v in payload.items()})
        supabase.table("audit_log").insert(record).execute()
    except Exception:
        pass


def log_upload(supabase, n_patients, n_high, n_medium, avg_risk,
               source="", tier=None, dq_grade=None,
               art_inferred=False, n_pediatric=0, dua_ack=False):
    """Log an upload event. source is hashed before logging."""
    if supabase is None:
        return
    log_event(supabase, "upload", {
        "file_hash":    _hash_filename(source),
        "n_patients":   int(n_patients),
        "n_high_risk":  int(n_high),
        "n_medium_risk":int(n_medium),
        "avg_risk":     round(float(avg_risk), 3),
        "tier":         tier or "unknown",
        "dq_grade":     dq_grade or "unknown",
        "art_inferred": bool(art_inferred),
        "n_pediatric":  int(n_pediatric),
        "dua_ack":      bool(dua_ack),
    })


def log_recalibration(supabase, n_patients, n_positive, prevalence,
                      local_auc, cal_method, passed=True,
                      failed_checks=None):
    """Log a recalibration attempt."""
    if supabase is None:
        return
    log_event(supabase, "recalibration", {
        "n_patients":    int(n_patients),
        "n_positive":    int(n_positive),
        "prevalence":    round(float(prevalence), 4),
        "local_auc":     round(float(local_auc), 4) if local_auc else None,
        "cal_method":    cal_method or "",
        "passed":        bool(passed),
        "failed_checks": str(failed_checks or []),
    })


def log_report(supabase, n_patients, report_type,
               org_type=None, region=None):
    """
    Log a report generation event.
    No organisation name or hash stored — anonymous by design.
    org_type: e.g. "implementing_partner", "ministry", "ngo", "research"
    region: e.g. "West Africa", "East Africa", "Southern Africa"
    """
    if supabase is None:
        return
    log_event(supabase, "report", {
        "n_patients":  int(n_patients),
        "report_type": str(report_type),
        "org_type":    str(org_type) if org_type else "not_specified",
        "region":      str(region) if region else "not_specified",
    })


def log_page_view(supabase, page_name):
    """Log a page navigation event."""
    if supabase is None:
        return
    log_event(supabase, "page_view", {
        "page": str(page_name),
    })

# ─────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────
def check_password():
    """
    Returns True if the user is authenticated.
    Uses Streamlit secrets for password storage.
    Falls back to open access if no password is configured
    (safe for demo/development).
    Includes brute-force protection: 5 failed attempts triggers a lockout.
    """
    # If no password configured in secrets, allow access
    correct_password = get_secret("APP_PASSWORD")
    if not correct_password:
        return True  # No password set — open access (dev mode)

    # Already authenticated this session
    if st.session_state.get("authenticated"):
        # ── Session timeout: 4 hours of inactivity ──────────────────────
        last_active = st.session_state.get("last_active")
        if last_active:
            elapsed = (datetime.datetime.utcnow() - last_active).total_seconds()
            if elapsed > 14400:  # 4 hours
                st.session_state["authenticated"] = False
                st.session_state["last_active"] = None
                st.warning("Your session has expired. Please log in again.")
            else:
                st.session_state["last_active"] = datetime.datetime.utcnow()
                return True
        else:
            st.session_state["last_active"] = datetime.datetime.utcnow()
            return True

    # ── Brute-force protection ───────────────────────────────────────────
    MAX_ATTEMPTS = 5
    attempts = st.session_state.get("login_attempts", 0)
    if attempts >= MAX_ATTEMPTS:
        st.error(
            "Access temporarily locked after 5 failed attempts. "
            "Please refresh the page or contact chinthalakalyani1@gmail.com for access."
        )
        return False

    # Show login screen
    st.markdown("""
    <style>
    .login-container {
        max-width: 420px; margin: 8vh auto; padding: 2.5rem;
        background: #111820; border: 1px solid #00e5ff33;
        border-radius: 12px; text-align: center;
    }
    .login-brand { font-family:'IBM Plex Mono',monospace; font-size:2rem;
        font-weight:600; color:#00e5ff; margin-bottom:0.25rem; }
    .login-sub { color:#8fa0b0; font-size:1rem; margin-bottom:2rem; }
    </style>
    <div class="login-container">
        <div class="login-brand">SmartDaaS</div>
        <div class="login-sub">AI-Powered HIV Programme Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Access SmartDaaS")
        password_input = st.text_input(
            "Access code",
            type="password",
            placeholder="Enter your access code",
            label_visibility="collapsed"
        )
        if st.button("→ Enter Platform", use_container_width=True, type="primary"):
            if password_input == correct_password:
                st.session_state["authenticated"] = True
                st.session_state["last_active"] = datetime.datetime.utcnow()
                st.session_state["login_attempts"] = 0
                st.rerun()
            else:
                st.session_state["login_attempts"] = st.session_state.get("login_attempts", 0) + 1
                remaining = max(0, MAX_ATTEMPTS - st.session_state["login_attempts"])
                if remaining > 0:
                    st.error(f"Incorrect access code. {remaining} attempt{'s' if remaining != 1 else ''} remaining.")
                else:
                    st.error("Access locked. Refresh the page or contact chinthalakalyani1@gmail.com.")

        st.markdown("""<div style='text-align:center;margin-top:1.5rem;
            font-size:0.75rem;color:#484f58'>
            Decision-support platform · Not a clinical tool<br>
            chinthalakalyani1@gmail.com
        </div>""", unsafe_allow_html=True)

    return False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDaaS — HIV Programme Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# AUTHENTICATION GATE — nothing renders until authenticated
# ─────────────────────────────────────────────────────────────
if not check_password():
    st.stop()

# ─────────────────────────────────────────────────────────────
# SUPABASE INIT
# ─────────────────────────────────────────────────────────────
supabase = init_supabase()

# ─────────────────────────────────────────────────────────────
# CSS — injected from styles.py
# ─────────────────────────────────────────────────────────────
from styles import inject_css
inject_css()

# ── placeholder so we can find the end of old block ──

# ─────────────────────────────────────────────────────────────
# CONSTANTS — imported from constants.py
# ─────────────────────────────────────────────────────────────
from constants import (
    FEATURES, COLUMN_ALIASES, FEAT_LABELS, FEAT_DESCRIPTIONS, FEAT_RANGES,
    INTERVENTIONS, COST_PER_POOR_OUTCOME, BASELINE_THRESHOLD,
    TIER_CORE_REQUIRED, TIER_STANDARD_QUALIFYING, TIER_STANDARD_MIN,
    TIER_ENHANCED_QUALIFYING, TIER_ENHANCED_MIN, ART_INFERENCE_TRIGGERS_LOWER,
    FEATURE_VALID_RANGES, HIGH_IMPORTANCE_FEATURES,
    RECAL_MIN_PATIENTS, RECAL_MIN_POS_EVENTS, RECAL_MAX_OUTCOME_MISS,
    RECAL_MAX_FEAT_MISS, RECAL_ISOTONIC_MIN, BOOTSTRAP_N,
    BASELINE_AUC, OUTCOME_NAME_HINTS,
)

# ── Temporary dict to mark end of removed block ──


# ─────────────────────────────────────────────────────────────
# DUA ACKNOWLEDGEMENT GATE — Stage 2
# Session-reset consent checkpoint before real data upload.
# ─────────────────────────────────────────────────────────────

DUA_STATEMENTS = [
    {
        "key": "dua_authority",
        "text": (
            "I confirm I have the authority to use this patient data for "
            "programme analytics purposes and that its use complies with "
            "applicable data governance requirements, including my "
            "organisation's data policies and relevant national regulations "
            "(including Nigeria's NDPR and applicable PEPFAR data governance "
            "frameworks where relevant)."
        ),
    },
    {
        "key": "dua_decision_support",
        "text": (
            "I understand that SmartDaaS is a decision-support tool only. "
            "All outputs require review by qualified programme and clinical "
            "staff before operational use. SmartDaaS outputs should not be "
            "used as the sole basis for individual patient treatment decisions."
        ),
    },
    {
        "key": "dua_privacy",
        "text": (
            "I confirm that I have reviewed SmartDaaS's privacy-by-design "
            "architecture. I understand that patient data uploaded in this "
            "session is processed in-memory only, is not stored on any server, "
            "is not transmitted to any external party, and is permanently and "
            "automatically deleted when this browser session ends or the page "
            "is closed. No patient-level data is retained between sessions."
        ),
    },
]


def is_dua_acknowledged():
    """Check if all DUA statements are acknowledged in this session."""
    return all(
        st.session_state.get(s["key"], False)
        for s in DUA_STATEMENTS
    )


def render_dua_gate(context="upload"):
    """
    Render the DUA acknowledgement gate.
    Returns True if all statements are acknowledged and upload may proceed.
    Returns False if any statement is unacknowledged.
    Once acknowledged, shows a compact one-line confirmation instead of
    re-rendering all three checkboxes.
    context: 'upload' (Patient Risk) or 'report' (Executive Report)
    """
    if is_dua_acknowledged():
        # Already acknowledged this session — show compact confirmation only
        ack_at = st.session_state.get('dua_acknowledged_at', '')
        st.markdown(
            f'<div style="background:#0d1f17;border:1px solid #3fb95066;'
            f'border-radius:6px;padding:8px 14px;font-size:0.82rem;color:#3fb950;'
            f'margin-bottom:8px">'
            f'✅ Data governance acknowledged for this session. '
            f'You may upload programme data.</div>',
            unsafe_allow_html=True
        )
        return True

    st.markdown("""<div style="border:1px solid #f0a500;border-radius:8px;
        padding:16px 20px;background:#1c1a10;margin:12px 0">
        <div style="color:#f0a500;font-weight:700;font-size:0.95rem;
            margin-bottom:12px">
            ⚠️ Data Governance Acknowledgement Required
        </div>
        <p style="color:#adbac7;font-size:0.95rem;margin:0 0 14px 0">
        Before uploading real patient data, please read and acknowledge
        each of the following statements. This acknowledgement applies
        to this session only and will be requested again in future sessions.
        </p>
        <p style="color:#8b949e;font-size:0.75rem;margin:0 0 14px 0">
        <em>For formal pilot engagements, a separate Data Use Agreement (DUA)
        will be established between SmartDaaS and your organisation before
        any programme data is ingested operationally.</em>
        </p>
    </div>""", unsafe_allow_html=True)

    all_checked = True
    for stmt in DUA_STATEMENTS:
        checked = st.checkbox(
            stmt["text"],
            key=stmt["key"],
            value=st.session_state.get(stmt["key"], False),
        )
        if not checked:
            all_checked = False

    if all_checked:
        st.session_state['dua_acknowledged_at'] = (
            datetime.datetime.utcnow().isoformat()
        )
        st.success(
            "✅ All statements acknowledged. You may now upload patient data."
        )
        return True
    else:
        remaining = sum(
            1 for s in DUA_STATEMENTS
            if not st.session_state.get(s["key"], False)
        )
        st.warning(
            f"Please acknowledge all {remaining} remaining "
            f"statement{'s' if remaining > 1 else ''} to proceed."
        )
        return False


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with open('cv_results.pkl', 'rb') as f:
            cv = pickle.load(f)
        with open('prepped_data.pkl', 'rb') as f:
            prep = pickle.load(f)
        model = cv['rf_model']
        scaler = model.named_steps['sc']
        clf = model.named_steps['clf']
        auc = float(cv['auc'])
        X_demo = prep['X']
        y_demo = prep['y']
        return model, scaler, clf, auc, X_demo, y_demo, True, ""
    except Exception as e:
        return None, None, None, 0.0, None, None, False, str(e)

model, SCALER, CLF, MODEL_AUC, X_DEMO, Y_DEMO, MODEL_OK, MODEL_ERR = load_model()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# DATA PIPELINE — imported from pipeline.py
# ─────────────────────────────────────────────────────────────
from pipeline import (
    normalize_columns,
    preprocess_phia_compatible,
    build_validation_metadata,
    render_validation_metadata,
    detect_art_status,
    detect_tier,
    check_pediatric_patients,
    render_tier_report,
    run_data_quality_screening,
    render_data_quality_report,
    render_iedea_benchmarks,
    derive_engineered_features,
    render_recalibration_page,
    detect_outcome_columns,
    normalize_outcome_column,
    validate_outcome_column,
    bootstrap_auc,
    find_optimal_threshold,
    sanitize_feature_matrix,
    run_recalibration,
    apply_calibration,
    generate_synthetic_recal_data,
)

def run_predictions(df_in):
    # Safety net: ensure all features exist and are numeric before indexing.
    # IMPORTANT: derive_engineered_features() must be called before this function
    # to compute CD4_improvement, weight_change, had_interruption, stage_worsened.
    # Calling run_predictions() directly without that step will silently use 0.0
    # for any missing engineered features, which reduces prediction accuracy.
    # The main upload pipeline (Patient Risk page) always calls
    # derive_engineered_features() first — this guard is a safety net only.
    df_in = df_in.copy()
    _n_zero_filled = 0
    for f in FEATURES:
        if f not in df_in.columns:
            df_in[f] = 0.0
            _n_zero_filled += 1
        else:
            df_in[f] = pd.to_numeric(df_in[f], errors='coerce').fillna(0.0)
    if _n_zero_filled > 3:
        import warnings as _warnings
        _warnings.warn(
            f"run_predictions: {_n_zero_filled} features were missing and filled "
            f"with 0.0. Call derive_engineered_features() before run_predictions() "
            f"to ensure all features are correctly computed.",
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
        threshold_lo = 0.4
        threshold_hi = 0.7

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
        X_s = SCALER.transform(X_raw[idx])
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
        x_scaled = SCALER.transform(x_row.reshape(1, -1))
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




# ═════════════════════════════════════════════════════════════════════════
# SMARTDAAS FEATURE BLOCK — Outreach Optimiser + Programme Brief + Nav
# ═════════════════════════════════════════════════════════════════════════
# Self-contained. No new dependencies: uses fpdf2 (already in requirements)
# for the PDF brief; the optional AI-wording layer uses `anthropic` only if
# it is installed and an ANTHROPIC_API_KEY is configured — otherwise it is
# silently skipped and the deterministic brief is shown.
# ═════════════════════════════════════════════════════════════════════════

# ── Outreach Optimiser constants ─────────────────────────────────────────
_OO_CONTACT_MINUTES = {'HIGH': 45, 'MEDIUM': 20, 'LOW': 5}
_OO_MODIFIABLE_FEATURES = {
    'side_effects': 3.0, 'had_interruption': 2.5, 'opp_infection': 2.0,
    'tb_positive': 2.0, 'stage_worsened': 1.5, 'days_to_ART': 1.2,
    'weight_change': 1.1,
}
_OO_BASE_REDUCTION_PROB = 0.23
_OO_COL_HIGH, _OO_COL_MEDIUM, _OO_COL_LOW = '#ff453a', '#ffb300', '#30d158'
_OO_COL_ACCENT, _OO_COL_TEXT, _OO_COL_MUTED = '#00e5ff', '#cdd9e5', '#8b949e'


def _oo_intervention_leverage(row):
    leverage = 1.0
    for feat, weight in _OO_MODIFIABLE_FEATURES.items():
        if feat in row.index:
            val = pd.to_numeric(row[feat], errors='coerce')
            if pd.notna(val) and val > 0.5:
                leverage += (weight - 1.0) * 0.3
    return min(leverage, 3.0)


def _oo_urgency(row):
    urgency = 1.0
    if 'MostRecentCd4Count' in row.index:
        cd4 = pd.to_numeric(row['MostRecentCd4Count'], errors='coerce')
        if pd.notna(cd4):
            if cd4 < 100:   urgency += 0.8
            elif cd4 < 200: urgency += 0.5
            elif cd4 < 350: urgency += 0.2
    if 'stage_start_num' in row.index:
        stage = pd.to_numeric(row['stage_start_num'], errors='coerce')
        if pd.notna(stage) and stage >= 3:
            urgency += 0.3
    return min(urgency, 2.0)


def _oo_top_modifiable_factor(row):
    LABELS = {
        'side_effects':     'Side effects -> regimen review',
        'had_interruption': 'Prior interruption -> re-engagement',
        'opp_infection':    'Opportunistic infection -> co-management',
        'tb_positive':      'TB positive -> TB/HIV integration',
        'stage_worsened':   'Clinical deterioration -> urgent review',
        'days_to_ART':      'Late ART start -> linkage support',
        'weight_change':    'Weight loss -> nutritional support',
    }
    for feat in sorted(_OO_MODIFIABLE_FEATURES,
                       key=_OO_MODIFIABLE_FEATURES.get, reverse=True):
        if feat in row.index:
            val = pd.to_numeric(row[feat], errors='coerce')
            if pd.notna(val) and val > 0.5:
                return LABELS.get(feat, feat)
    return 'Standard retention follow-up'


def _oo_contact_type_label(tier):
    return {
        'HIGH':   'Home visit / Urgent counselling (45 min)',
        'MEDIUM': 'Phone call + brief session (20 min)',
        'LOW':    'SMS reminder (5 min)',
    }.get(tier, 'Standard outreach')


def _oo_build_action_plan(df, n_workers, days_available,
                          hours_per_worker_per_day, include_medium=True):
    df = df.copy()
    total_minutes = n_workers * days_available * hours_per_worker_per_day * 60
    df['_leverage'] = df.apply(_oo_intervention_leverage, axis=1)
    df['_urgency']  = df.apply(_oo_urgency, axis=1)
    df['_priority_raw'] = (
        df.get('risk_score', 0.0).astype(float) * df['_leverage'] * df['_urgency']
    )
    p_max, p_min = df['_priority_raw'].max(), df['_priority_raw'].min()
    if p_max > p_min:
        df['priority_score'] = (
            (df['_priority_raw'] - p_min) / (p_max - p_min) * 100).round(1)
    else:
        df['priority_score'] = 50.0

    tiers = ['HIGH', 'MEDIUM'] if include_medium else ['HIGH']
    candidates = df[df['risk_label'].isin(tiers)].copy()
    candidates = candidates.sort_values('priority_score', ascending=False)

    rows, used_minutes, rank = [], 0.0, 1
    for _, row in candidates.iterrows():
        tier = str(row['risk_label'])
        mins = _OO_CONTACT_MINUTES.get(tier, 20)
        if used_minutes + mins > total_minutes:
            break
        if tier == 'HIGH':
            proj = round(float(row.get('risk_score', 0.5)) *
                         _OO_BASE_REDUCTION_PROB * float(row['_leverage']), 3)
        else:
            proj = round(float(row.get('risk_score', 0.3)) *
                         _OO_BASE_REDUCTION_PROB * 0.5 * float(row['_leverage']), 3)
        rows.append({
            'rank': rank,
            'patient_id': row.get('patient_id', f'PT-{rank:04d}'),
            'risk_pct': round(float(row.get('risk_pct',
                              row.get('risk_score', 0) * 100)), 1),
            'risk_label': tier,
            'priority_score': round(float(row['priority_score']), 1),
            'leverage': round(float(row['_leverage']), 2),
            'urgency': round(float(row['_urgency']), 2),
            'contact_type': _oo_contact_type_label(tier),
            'estimated_minutes': mins,
            'top_modifiable_factor': _oo_top_modifiable_factor(row),
            'projected_interruption_reduction': proj,
        })
        used_minutes += mins
        rank += 1
    return pd.DataFrame(rows), used_minutes, total_minutes


def _oo_impact_estimates(plan_df, full_cohort_df, cost_per_poor_outcome=1850.0):
    if plan_df.empty:
        return {}
    n_planned = len(plan_df)
    n_high = int((plan_df['risk_label'] == 'HIGH').sum())
    n_medium = int((plan_df['risk_label'] == 'MEDIUM').sum())
    n_full_high = (int((full_cohort_df['risk_label'] == 'HIGH').sum())
                   if 'risk_label' in full_cohort_df.columns else n_high)
    prevented = plan_df['projected_interruption_reduction'].sum()
    cost = prevented * cost_per_poor_outcome
    coverage = (n_high / n_full_high * 100) if n_full_high > 0 else 0
    mins = plan_df['estimated_minutes'].sum()
    return {
        'n_planned': n_planned, 'n_high': n_high, 'n_medium': n_medium,
        'n_full_high': n_full_high, 'coverage_pct': round(coverage, 1),
        'interruptions_prevented': round(prevented, 1),
        'cost_savings_usd': round(cost, 0),
        'total_mins_planned': int(mins), 'total_hrs_planned': round(mins / 60, 1),
    }


# ── Narrative brief: deterministic fact extraction + template ────────────
def _nb_extract_facts(df_scored):
    facts = {'n_total': int(len(df_scored))}
    n = facts['n_total']
    if n == 0:
        return facts
    labels = df_scored.get('risk_label', pd.Series([], dtype=str))
    facts['n_high'] = int((labels == 'HIGH').sum())
    facts['n_medium'] = int((labels == 'MEDIUM').sum())
    facts['n_low'] = int((labels == 'LOW').sum())
    facts['pct_high'] = round(facts['n_high'] / n * 100, 1)
    facts['pct_medium'] = round(facts['n_medium'] / n * 100, 1)
    facts['pct_low'] = round(facts['n_low'] / n * 100, 1)
    if 'risk_score' in df_scored.columns:
        facts['mean_risk_pct'] = round(
            pd.to_numeric(df_scored['risk_score'], errors='coerce').mean() * 100, 1)
    else:
        facts['mean_risk_pct'] = None

    def _pct(col):
        if col in df_scored.columns:
            v = pd.to_numeric(df_scored[col], errors='coerce')
            return round((v > 0.5).mean() * 100, 1)
        return None
    facts['pct_interruption'] = _pct('had_interruption')
    facts['pct_side_effects'] = _pct('side_effects')
    facts['pct_tb'] = _pct('tb_positive')
    facts['pct_opp_infection'] = _pct('opp_infection')

    if 'stage_start_num' in df_scored.columns:
        stage = pd.to_numeric(df_scored['stage_start_num'], errors='coerce')
        facts['pct_advanced_stage'] = round((stage >= 3).mean() * 100, 1)
    else:
        facts['pct_advanced_stage'] = None
    if 'MostRecentCd4Count' in df_scored.columns:
        cd4 = pd.to_numeric(df_scored['MostRecentCd4Count'], errors='coerce')
        facts['pct_low_cd4'] = round((cd4 < 200).mean() * 100, 1)
    else:
        facts['pct_low_cd4'] = None
    if 'sex_female' in df_scored.columns and 'risk_score' in df_scored.columns:
        sf = pd.to_numeric(df_scored['sex_female'], errors='coerce')
        mr = df_scored.loc[sf < 0.5, 'risk_score'].mean()
        fr = df_scored.loc[sf >= 0.5, 'risk_score'].mean()
        if pd.notna(mr) and pd.notna(fr):
            facts['male_risk_pct'] = round(mr * 100, 1)
            facts['female_risk_pct'] = round(fr * 100, 1)
            facts['sex_gap_pp'] = round((mr - fr) * 100, 1)
        else:
            facts['sex_gap_pp'] = None
    else:
        facts['sex_gap_pp'] = None
    return facts


def _nb_build_template(facts, impact=None, params=None):
    n = facts.get('n_total', 0)
    if n == 0:
        return {'full_text': 'No patient data available to summarise.'}
    pct_high = facts.get('pct_high', 0)
    n_high = facts.get('n_high', 0)
    headline = (f"Of {n:,} patients in this cohort, {n_high:,} ({pct_high}%) are at "
                f"HIGH risk of treatment interruption and warrant proactive outreach.")
    situation = (
        f"This cohort of {n:,} patients comprises {facts.get('n_high',0):,} HIGH-risk "
        f"({pct_high}%), {facts.get('n_medium',0):,} MEDIUM-risk "
        f"({facts.get('pct_medium',0)}%), and {facts.get('n_low',0):,} LOW-risk "
        f"({facts.get('pct_low',0)}%) patients.")
    if facts.get('mean_risk_pct') is not None:
        situation += (f" The mean predicted interruption risk across the cohort is "
                      f"{facts['mean_risk_pct']}%.")

    clauses = []
    if facts.get('pct_interruption') is not None and facts['pct_interruption'] > 15:
        clauses.append(f"{facts['pct_interruption']}% have a documented prior ART "
                       f"interruption -- the strongest single predictor of future "
                       f"disengagement")
    if facts.get('pct_advanced_stage') is not None and facts['pct_advanced_stage'] > 20:
        clauses.append(f"{facts['pct_advanced_stage']}% presented at WHO Stage 3-4, "
                       f"indicating late presentation")
    if facts.get('pct_low_cd4') is not None and facts['pct_low_cd4'] > 25:
        clauses.append(f"{facts['pct_low_cd4']}% have a most-recent CD4 below 200 "
                       f"cells/microlitre")
    if facts.get('pct_tb') is not None and facts['pct_tb'] > 10:
        clauses.append(f"{facts['pct_tb']}% are TB-positive, requiring TB/HIV "
                       f"co-management")
    if facts.get('pct_side_effects') is not None and facts['pct_side_effects'] > 20:
        clauses.append(f"{facts['pct_side_effects']}% report treatment side effects "
                       f"that may be addressable through regimen review")
    if clauses:
        drivers = "Key risk drivers in this cohort: " + "; ".join(clauses) + "."
    else:
        drivers = ("No single risk driver dominates this cohort; risk is distributed "
                   "across multiple factors. Review individual patient profiles for "
                   "specifics.")
    if facts.get('sex_gap_pp') is not None and facts['sex_gap_pp'] >= 3:
        drivers += (f" Male patients average {facts['male_risk_pct']}% risk versus "
                    f"{facts['female_risk_pct']}% for female patients (a "
                    f"{facts['sex_gap_pp']} percentage-point gap), suggesting "
                    f"male-targeted retention efforts may be warranted.")

    if impact and impact.get('n_planned', 0) > 0:
        nw = params.get('n_workers', '-') if params else '-'
        dd = params.get('days', '-') if params else '-'
        action = (f"With your stated capacity of {nw} outreach worker(s) over {dd} "
                  f"day(s), SmartDaaS recommends contacting {impact['n_planned']} "
                  f"patients this week ({impact.get('n_high',0)} HIGH-risk, "
                  f"{impact.get('n_medium',0)} MEDIUM-risk). This covers "
                  f"{impact.get('coverage_pct',0)}% of all HIGH-risk patients within "
                  f"{impact.get('total_hrs_planned','-')} hours of outreach time.")
        if impact.get('interruptions_prevented') is not None:
            action += (f" If all planned contacts are completed, the illustrative "
                       f"estimate is {impact['interruptions_prevented']} interruptions "
                       f"prevented next month, corresponding to approximately "
                       f"${impact.get('cost_savings_usd',0):,.0f} in avoidable "
                       f"programme costs (planning estimate only -- not for funder "
                       f"reporting without local validation).")
    else:
        action = (f"Recommended next step: prioritise the {n_high:,} HIGH-risk patients "
                  f"for proactive outreach. Use the capacity inputs above to fit this "
                  f"list to your available staff and generate a ranked weekly plan.")

    full = (f"{headline}\n\nSITUATION. {situation}\n\nRISK DRIVERS. {drivers}\n\n"
            f"RECOMMENDED ACTION. {action}")
    return {'full_text': full}


def _nb_verify_numbers(source, enhanced):
    import re
    def nums(t):
        return {r.replace(',', '') for r in re.findall(r'\d[\d,]*\.?\d*', t)}
    return len(nums(source) - nums(enhanced)) == 0


def _nb_enhance_with_api(full_text):
    """Optional. Returns (text, was_enhanced). Silent no-op if unavailable."""
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return full_text, False
    try:
        import anthropic
    except Exception:
        return full_text, False
    sys_prompt = (
        "You are rephrasing a pre-computed HIV programme brief. Rephrase into smooth "
        "professional prose. STRICT RULES: (1) do not change, add, remove, or round ANY "
        "number; every figure must appear exactly as given. (2) add no clinical claim or "
        "recommendation not already present. (3) preserve all caveats. (4) max 4 short "
        "paragraphs. (5) no greeting or signature. If unsure, reproduce verbatim.")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1000,
            system=sys_prompt,
            messages=[{"role": "user",
                       "content": "Rephrase this brief:\n\n" + full_text}])
        out = "".join(b.text for b in msg.content
                      if getattr(b, "type", None) == "text").strip()
        if out and _nb_verify_numbers(full_text, out):
            return out, True
        return full_text, False
    except Exception:
        return full_text, False


def _nb_format_html(text):
    import html
    safe = html.escape(text)
    for label in ['SITUATION.', 'RISK DRIVERS.', 'RECOMMENDED ACTION.']:
        safe = safe.replace(label, f'<strong style="color:#00e5ff">{label}</strong>')
    return safe


def _nb_pdf_brief(narrative_text, plan_df, impact, params, org_name=""):
    """Build the brief + action plan as a PDF using fpdf2 (already installed)."""
    from fpdf import FPDF

    def _s(t):
        if not isinstance(t, str):
            t = str(t)
        return (t.replace('\u2014', '-').replace('\u2013', '-')
                 .replace('\u2192', '->').replace('\u00b7', '.')
                 .replace('\u2019', "'").replace('\u2018', "'")
                 .replace('\u201c', '"').replace('\u201d', '"')
                 .replace('\u2026', '...').replace('\u00b5', 'u')
                 .replace('\u00d7', 'x')
                 .encode('latin-1', errors='replace').decode('latin-1'))

    TEAL = (10, 125, 140)
    DARK = (34, 34, 34)
    GREY = (120, 120, 120)

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_left_margin(18)
    pdf.set_right_margin(18)

    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 10, _s("SmartDaaS - Programme Brief"), ln=True)

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*GREY)
    import datetime as _dt
    sub = (f"{org_name} . " if org_name else "") + \
          _dt.datetime.utcnow().strftime('%d %B %Y, %H:%M UTC')
    pdf.cell(0, 6, _s(sub), ln=True)
    pdf.ln(3)

    blocks = narrative_text.replace('\r\n', '\n').split('\n\n')
    label_map = {'SITUATION.': 'Situation', 'RISK DRIVERS.': 'Risk drivers',
                 'RECOMMENDED ACTION.': 'Recommended action'}
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        matched = False
        for raw, nice in label_map.items():
            if block.startswith(raw):
                pdf.set_font('Helvetica', 'B', 11)
                pdf.set_text_color(*TEAL)
                pdf.cell(0, 7, _s(nice), ln=True)
                pdf.set_font('Helvetica', '', 10)
                pdf.set_text_color(*DARK)
                pdf.multi_cell(0, 5.5, _s(block[len(raw):].strip()))
                pdf.ln(1.5)
                matched = True
                break
        if not matched:
            pdf.set_font('Helvetica', '', 11)
            pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 6, _s(block))
            pdf.ln(2)

    if plan_df is not None and len(plan_df) > 0:
        pdf.ln(2)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(*TEAL)
        pdf.cell(0, 7, _s("Weekly action plan - patient contact list"), ln=True)

        widths = [10, 26, 14, 18, 44, 62]
        headers = ['#', 'Patient ID', 'Risk%', 'Tier', 'Contact type', 'Primary focus']
        pdf.set_font('Helvetica', 'B', 7.5)
        pdf.set_fill_color(*TEAL)
        pdf.set_text_color(255, 255, 255)
        for w, h in zip(widths, headers):
            pdf.cell(w, 6, _s(h), border=0, fill=True)
        pdf.ln()

        pdf.set_font('Helvetica', '', 7.5)
        pdf.set_text_color(*DARK)
        max_rows = 40
        for i, (_, r) in enumerate(plan_df.head(max_rows).iterrows()):
            fill = (i % 2 == 1)
            if fill:
                pdf.set_fill_color(242, 246, 247)
            cells = [
                str(r['rank']), str(r['patient_id']),
                f"{r['risk_pct']:.0f}", str(r['risk_label']),
                str(r['contact_type']).split(' (')[0],
                str(r['top_modifiable_factor']),
            ]
            for w, c in zip(widths, cells):
                pdf.cell(w, 5.5, _s(c[:40]), border=0, fill=fill)
            pdf.ln()
        if len(plan_df) > max_rows:
            pdf.ln(1)
            pdf.set_font('Helvetica', 'I', 7)
            pdf.set_text_color(*GREY)
            pdf.multi_cell(0, 4, _s(
                f"Showing first {max_rows} of {len(plan_df)} planned contacts. "
                f"Full list available in the CSV export."))

    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 4, _s(
        "Projected impact figures are illustrative planning estimates only and are not "
        "for funder reporting without local validation. SmartDaaS is a decision-support "
        "tool; all outputs require review by qualified programme and clinical staff "
        "before operational use. Patient data is processed in-session only and is not "
        "stored or transmitted."))

    out = pdf.output(dest='S')
    if isinstance(out, str):
        return out.encode('latin-1')
    return bytes(out)


def render_narrative_block(df_scored, plan_df=None, impact=None,
                           params=None, supabase=None, org_name=""):
    """Embedded programme brief — top of the Outreach Optimiser results."""
    if df_scored is None or len(df_scored) == 0:
        return
    facts = _nb_extract_facts(df_scored)
    template = _nb_build_template(facts, impact, params)
    display_text = template['full_text']
    enhanced = False

    if get_secret("ANTHROPIC_API_KEY") and st.session_state.get('oo_enhance_brief'):
        display_text, enhanced = _nb_enhance_with_api(template['full_text'])

    st.markdown(f"""
<div style="background:#111820;border:1px solid #00e5ff55;border-radius:10px;
    padding:1.5rem 1.75rem;margin:0 0 1rem 0;line-height:1.75;font-size:0.96rem;
    color:#e2eaf3">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;
        color:#00e5ff;text-transform:uppercase;letter-spacing:2.5px;
        margin-bottom:0.75rem">
        Programme Brief - auto-generated - every figure traceable to your data
    </div>
    <div style="white-space:pre-wrap">{_nb_format_html(display_text)}</div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        import streamlit.components.v1 as components
        safe_js = (display_text.replace('\\', '\\\\').replace('`', '\\`')
                   .replace('$', '\\$'))
        components.html(f"""
            <button id="cp" style="width:100%;padding:8px 12px;font-size:0.85rem;
                background:#1e2530;color:#00e5ff;border:1px solid #00e5ff55;
                border-radius:6px;cursor:pointer;font-family:sans-serif">
                Copy brief</button>
            <script>
            const b=document.getElementById('cp');
            b.onclick=()=>{{navigator.clipboard.writeText(`{safe_js}`).then(()=>{{
                b.textContent='Copied';setTimeout(()=>b.textContent='Copy brief',1800);}});}};
            </script>""", height=48)
    with c2:
        try:
            pdf_bytes = _nb_pdf_brief(display_text, plan_df, impact or {},
                                      params or {}, org_name)
            st.download_button(
                "Download PDF brief", data=pdf_bytes,
                file_name=f"smartdaas_brief_{datetime.date.today()}.pdf",
                mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.button("PDF unavailable", disabled=True, use_container_width=True,
                      help=f"{type(e).__name__}")
    with c3:
        if get_secret("ANTHROPIC_API_KEY"):
            st.toggle("Smoother wording (AI - figures verified)",
                      key="oo_enhance_brief",
                      help="Rephrases the brief; numbers are verified so the AI "
                           "cannot change any figure.")
        if enhanced:
            st.caption("AI-enhanced wording. All figures verified against the template.")

    if supabase is not None:
        try:
            supabase.table("audit_log").insert({
                "event_at": datetime.datetime.utcnow().isoformat(),
                "event_type": "narrative_brief",
                "n_patients": int(facts.get('n_total', 0)),
                "report_type": "ai_enhanced" if enhanced else "template",
            }).execute()
        except Exception:
            pass


# ── Demo cohort for when no data is scored yet ───────────────────────────
def _oo_demo_cohort(n=200, seed=42):
    rng = np.random.RandomState(seed)
    n_high = int(n * 0.22); n_medium = int(n * 0.35); n_low = n - n_high - n_medium

    def _block(size, rmean, rstd, stage_high=False):
        risk = np.clip(rng.normal(rmean, rstd, size), 0.01, 0.99)
        return pd.DataFrame({
            'patient_id': [f'PT-{i:04d}' for i in range(size)],
            'risk_score': risk, 'risk_pct': (risk * 100).round(1),
            'Age': rng.randint(20, 60, size).astype(float),
            'sex_female': rng.randint(0, 2, size).astype(float),
            'Cd4AtStart': rng.randint(50, 600, size).astype(float),
            'MostRecentCd4Count': rng.randint(50, 700, size).astype(float),
            'CD4_improvement': rng.randint(-200, 400, size).astype(float),
            'stage_start_num': rng.choice([1,2,3,4] if stage_high else [1,2],
                                          size).astype(float),
            'WeightAtStart': rng.randint(45, 90, size).astype(float),
            'weight_change': rng.randint(-10, 10, size).astype(float),
            'BMI_start': rng.uniform(16, 32, size).round(1),
            'days_to_ART': rng.randint(0, 400, size).astype(float),
            'had_interruption': rng.choice([0,1], size, p=[0.6,0.4]).astype(float),
            'opp_infection': rng.choice([0,1], size, p=[0.75,0.25]).astype(float),
            'side_effects': rng.choice([0,1], size, p=[0.65,0.35]).astype(float),
            'tb_positive': rng.choice([0,1], size, p=[0.82,0.18]).astype(float),
            'stage_worsened': rng.choice([0,1], size, p=[0.78,0.22]).astype(float),
        })
    parts = [_block(n_high, 0.82, 0.08, True), _block(n_medium, 0.52, 0.07),
             _block(n_low, 0.18, 0.09)]
    idx = 0
    for p in parts:
        p['patient_id'] = [f'PT-{idx+i:04d}' for i in range(len(p))]; idx += len(p)
    df = pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    df['risk_label'] = pd.cut(df['risk_score'], bins=[-0.001, 0.4, 0.7, 1.001],
                              labels=['LOW', 'MEDIUM', 'HIGH']).astype(str)
    return df


def render_outreach_optimiser(supabase=None):
    """Outreach Optimiser page — capacity-constrained weekly action plan."""
    st.markdown("""
<div style="background:#111820;border:1px solid #00e5ff33;border-radius:10px;
    padding:1.5rem 2rem;margin-bottom:1.25rem">
    <p style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#00e5ff;
        text-transform:uppercase;letter-spacing:3px;margin:0 0 0.4rem 0">
        SmartDaaS . New</p>
    <h2 style="color:#e6edf3;font-size:1.4rem;font-weight:700;margin:0 0 0.5rem 0">
        Outreach Optimiser</h2>
    <p style="color:#cdd9e5;font-size:0.95rem;margin:0;line-height:1.6">
        Converts patient risk scores into a <strong>capacity-constrained weekly action
        plan</strong> for your outreach team. Not just who is high-risk -
        <em>exactly who to contact this week, in what order, and why</em>,
        fitted to your actual staff capacity.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""<div style="background:#0c2014;border-left:3px solid #30d158;
        padding:0.75rem 1rem;border-radius:0 6px 6px 0;font-size:0.9rem;
        color:#30d158;margin-bottom:1rem">
        <strong>What makes this different:</strong>
        Every other HIV analytics platform stops at "these 500 patients are high risk."
        SmartDaaS asks: <em>how many outreach workers do you actually have, and how many
        hours this week?</em> Then it builds the plan around your real capacity.
    </div>""", unsafe_allow_html=True)

    df_scored = st.session_state.get('df_scored', None)
    using_demo = False
    if df_scored is None or not isinstance(df_scored, pd.DataFrame) or len(df_scored) == 0:
        st.markdown("""<div style="background:#1c1a10;border:1px solid #f0a50066;
            border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1rem">
            <p style="color:#f0a500;font-weight:700;font-size:0.95rem;margin:0 0 0.5rem 0">
            No programme data loaded yet</p>
            <p style="color:#cdd9e5;font-size:0.9rem;margin:0 0 0.75rem 0;line-height:1.6">
            To build an action plan for your patients, go to <strong>Patient Risk</strong>
            in the sidebar, upload your programme CSV, and run the risk scoring. Then come
            back here - the Outreach Optimiser will automatically use your cohort.</p>
            <p style="color:#8b949e;font-size:0.85rem;margin:0">
            In the meantime, the plan below runs on a <strong>synthetic demo cohort of
            200 patients</strong> so you can explore how the tool works.</p>
            </div>""", unsafe_allow_html=True)
        df_scored = _oo_demo_cohort(200, 42)
        using_demo = True

    if not {'risk_score', 'risk_label'} <= set(df_scored.columns):
        st.markdown("""<div style="background:#2a1010;border:1px solid #ff453a66;
            border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1rem">
            <p style="color:#ff453a;font-weight:700;font-size:0.95rem;margin:0 0 0.5rem 0">
            Risk scoring has not been run on this data yet</p>
            <p style="color:#cdd9e5;font-size:0.9rem;margin:0;line-height:1.6">
            Your data was uploaded but risk scores have not been calculated. Please go to
            <strong>Patient Risk</strong>, run the scoring, then return here.</p>
            </div>""", unsafe_allow_html=True)
        return

    if 'patient_id' not in df_scored.columns:
        df_scored = df_scored.copy()
        df_scored['patient_id'] = [f'PT-{i:04d}' for i in range(len(df_scored))]
    if 'risk_pct' not in df_scored.columns:
        df_scored = df_scored.copy()
        df_scored['risk_pct'] = (df_scored['risk_score'] * 100).round(1)

    n_total = len(df_scored)
    n_high = int((df_scored['risk_label'] == 'HIGH').sum())
    n_medium = int((df_scored['risk_label'] == 'MEDIUM').sum())
    n_low = int((df_scored['risk_label'] == 'LOW').sum())

    st.markdown('<p class="section-hdr">Cohort Snapshot</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, val, lbl, color in [
        (cols[0], f'{n_total:,}', 'Total patients', '#cdd9e5'),
        (cols[1], f'{n_high:,}', 'HIGH risk', _OO_COL_HIGH),
        (cols[2], f'{n_medium:,}', 'MEDIUM risk', _OO_COL_MEDIUM),
        (cols[3], f'{n_low:,}', 'LOW risk', _OO_COL_LOW)]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val" '
                        f'style="color:{color}">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Your Outreach Capacity This Week</p>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-box">Enter your <em>actual</em> available outreach
        staff and time. SmartDaaS fits the action plan to this capacity.</div>""",
                unsafe_allow_html=True)
    cc = st.columns(4)
    with cc[0]:
        n_workers = st.number_input('Outreach workers available', 1, 50, 2,
                                    help='CHWs, peer navigators, adherence counsellors')
    with cc[1]:
        days_avail = st.number_input('Working days in window', 1, 30, 5,
                                     help='Days until next reporting deadline')
    with cc[2]:
        hrs_per_day = st.number_input('Outreach hours per worker per day',
                                      0.5, 8.0, 3.0, step=0.5)
    with cc[3]:
        include_medium = st.checkbox('Include MEDIUM-risk\n(after HIGH)', value=True)

    cap_hrs = n_workers * days_avail * hrs_per_day
    st.markdown(f'<div style="background:#1e2530;border:1px solid #00e5ff33;'
                f'border-radius:6px;padding:8px 14px;font-size:0.85rem;color:#00e5ff;'
                f'margin:4px 0 12px 0">Total outreach capacity: <strong>{cap_hrs:.1f} '
                f'hours</strong> ({cap_hrs*60:.0f} minutes) across {n_workers} worker(s) '
                f'over {days_avail} day(s)</div>', unsafe_allow_html=True)

    if st.button("Build This Week's Action Plan", type='primary',
                 use_container_width=True):
        st.session_state['oo_run'] = True
        st.session_state['oo_params'] = {
            'n_workers': int(n_workers), 'days': int(days_avail),
            'hrs': float(hrs_per_day), 'include_medium': bool(include_medium)}

    if not st.session_state.get('oo_run'):
        st.markdown('<div style="text-align:center;padding:2rem;color:#484f58;'
                    'font-size:0.9rem">Set your staff capacity above and click Build.'
                    '</div>', unsafe_allow_html=True)
        return

    params = st.session_state.get('oo_params', {
        'n_workers': int(n_workers), 'days': int(days_avail),
        'hrs': float(hrs_per_day), 'include_medium': bool(include_medium)})

    with st.spinner('Building capacity-constrained action plan...'):
        plan_df, used_mins, total_mins = _oo_build_action_plan(
            df_scored, params['n_workers'], params['days'], params['hrs'],
            params['include_medium'])
        impact = _oo_impact_estimates(plan_df, df_scored)

    if plan_df.empty:
        st.warning("No HIGH or MEDIUM risk patients found, or capacity is too low to "
                   "schedule any contacts. Try increasing available hours.")
        return

    # Programme brief at the top of results
    try:
        render_narrative_block(df_scored, plan_df, impact, params, supabase)
    except Exception as _e:
        st.caption(f"Programme brief unavailable ({type(_e).__name__}).")

    st.markdown('<p class="section-hdr">This Week\'s Plan - At a Glance</p>',
                unsafe_allow_html=True)
    st.markdown(f"""<div style="background:#111820;border:1px solid #00e5ff44;
        border-radius:10px;padding:1.25rem 1.75rem;margin-bottom:1rem">
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#00e5ff;
        margin:0 0 0.75rem 0">SmartDaaS recommends contacting
        <strong style="font-size:1.4rem">{impact['n_planned']}</strong> patients this week
        ({impact['n_high']} HIGH . {impact['n_medium']} MEDIUM)</p>
        <p style="color:#cdd9e5;font-size:0.9rem;margin:0;line-height:1.7">
        Out of <strong>{impact['n_full_high']}</strong> total HIGH-risk patients, this plan
        covers <strong>{impact['coverage_pct']}%</strong> within your available capacity of
        <strong>{impact['total_hrs_planned']} hours</strong>. Estimated interruptions
        prevented if all contacts are completed:
        <strong style="color:{_OO_COL_LOW}">{impact['interruptions_prevented']}</strong>
        (illustrative).</p></div>""", unsafe_allow_html=True)

    mc = st.columns(4)
    for col, val, lbl, color in [
        (mc[0], str(impact['n_planned']), 'Patients to contact', _OO_COL_ACCENT),
        (mc[1], f"{impact['coverage_pct']}%", 'HIGH-risk coverage', _OO_COL_MEDIUM),
        (mc[2], str(impact['interruptions_prevented']),
         'Est. interruptions prevented', _OO_COL_LOW),
        (mc[3], f"${impact['cost_savings_usd']:,.0f}", 'Est. cost saved', '#21d4fd')]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val" '
                        f'style="color:{color}">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True)
    st.caption("Illustrative estimates only. Based on a conservative 23% interruption "
               "reduction per contacted HIGH-risk patient (PEPFAR retention literature) "
               "and $1,850/poor outcome (Menzies et al. 2011). Not for funder reporting "
               "without local validation.")

    # Capacity gauge
    st.markdown('<p class="section-hdr">Capacity Utilisation</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 1.2), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    pct = min(used_mins / total_mins, 1.0) if total_mins > 0 else 0
    ax.barh(0, 1.0, height=0.5, color='#30363d')
    ax.barh(0, pct, height=0.5,
            color=_OO_COL_HIGH if pct > 0.9 else _OO_COL_MEDIUM if pct > 0.7 else _OO_COL_LOW)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8, color=_OO_COL_MUTED)
    ax.set_title(f'Outreach capacity used: {pct*100:.0f}% ({used_mins/60:.1f} of '
                 f'{total_mins/60:.1f} hrs)', color='#e6edf3', fontsize=9, pad=6)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Priority scatter
    st.markdown('<p class="section-hdr">Priority Score Distribution</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box" style="font-size:0.85rem">Priority = risk '
                'probability x intervention leverage (modifiable factors) x clinical '
                'urgency (CD4, WHO stage). Higher = greater outreach ROI.</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    for tier, color in {'HIGH': _OO_COL_HIGH, 'MEDIUM': _OO_COL_MEDIUM}.items():
        sub = plan_df[plan_df['risk_label'] == tier]
        if len(sub) > 0:
            ax.scatter(sub['rank'], sub['priority_score'], c=color, label=tier,
                       s=40, alpha=0.85, zorder=3)
    ax.set_xlabel('Outreach rank', color=_OO_COL_MUTED, fontsize=9)
    ax.set_ylabel('Priority score (0-100)', color=_OO_COL_MUTED, fontsize=9)
    ax.set_title("Patient priority scores - this week's plan", color='#e6edf3',
                 fontsize=10, pad=8)
    ax.legend(fontsize=8, facecolor='#161b22', labelcolor=_OO_COL_TEXT)
    ax.tick_params(colors=_OO_COL_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.grid(axis='y', color='#30363d', linewidth=0.5, alpha=0.5)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Modifiable factors
    st.markdown('<p class="section-hdr">Where to Focus Interventions</p>',
                unsafe_allow_html=True)
    counts = plan_df['top_modifiable_factor'].value_counts().head(6)
    fig, ax = plt.subplots(figsize=(9, 3), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=_OO_COL_ACCENT,
                   alpha=0.8, height=0.55)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(val),
                va='center', fontsize=8, color=_OO_COL_TEXT)
    ax.set_xlabel('Patients in plan', color=_OO_COL_MUTED, fontsize=9)
    ax.set_title('Top intervention focus areas this week', color='#e6edf3',
                 fontsize=10, pad=8)
    ax.tick_params(colors=_OO_COL_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.set_xlim(0, counts.values.max() * 1.25)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Action plan table
    st.markdown('<p class="section-hdr">Weekly Action Plan - Patient List</p>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box" style="font-size:0.85rem">Patients ranked by '
                'priority - highest outreach ROI first. Work down this list in order.'
                '</div>', unsafe_allow_html=True)
    disp = plan_df[['rank', 'patient_id', 'risk_pct', 'risk_label', 'priority_score',
                    'contact_type', 'top_modifiable_factor', 'estimated_minutes']].copy()
    disp['risk_pct'] = disp['risk_pct'].apply(lambda x: f'{x:.1f}')
    disp['priority_score'] = disp['priority_score'].apply(lambda x: f'{x:.1f}')
    disp.columns = ['Rank', 'Patient ID', 'Risk %', 'Tier', 'Priority Score',
                    'Contact Type', 'Primary Focus', 'Est. Minutes']

    def _tier_color(val):
        c = {'HIGH': _OO_COL_HIGH, 'MEDIUM': _OO_COL_MEDIUM,
             'LOW': _OO_COL_LOW}.get(val, _OO_COL_TEXT)
        return f'color: {c}; font-weight: bold'
    try:
        styled = disp.style.map(_tier_color, subset=['Tier'])
    except Exception:
        styled = disp.style.applymap(_tier_color, subset=['Tier'])
    st.dataframe(styled, use_container_width=True, height=420)

    # Exports
    st.markdown('<p class="section-hdr">Export Action Plan</p>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    with ec1:
        export = plan_df[['rank', 'patient_id', 'risk_pct', 'risk_label',
                          'priority_score', 'contact_type', 'top_modifiable_factor',
                          'estimated_minutes',
                          'projected_interruption_reduction']].copy()
        export.columns = ['Priority Rank', 'Patient ID', 'Risk Score (%)', 'Risk Tier',
                          'Priority Score', 'Contact Type', 'Primary Intervention Focus',
                          'Est. Time (min)', 'Projected Risk Reduction']
        st.download_button("Download Action Plan (CSV)",
                           data=export.to_csv(index=False).encode(),
                           file_name=f"smartdaas_outreach_plan_{datetime.date.today()}.csv",
                           mime="text/csv", use_container_width=True)
        st.caption("For CHW team use.")
    with ec2:
        summary = pd.DataFrame([
            ['Generated', datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')],
            ['Workers', params['n_workers']], ['Days', params['days']],
            ['Hrs/worker/day', params['hrs']],
            ['Total capacity (hrs)', impact['total_hrs_planned']],
            ['Patients in plan', impact['n_planned']],
            ['HIGH risk', impact['n_high']], ['MEDIUM risk', impact['n_medium']],
            ['HIGH-risk coverage', f"{impact['coverage_pct']}%"],
            ['Est. interruptions prevented', impact['interruptions_prevented']],
            ['Est. cost savings', f"${impact['cost_savings_usd']:,.0f}"],
        ], columns=['Metric', 'Value'])
        st.download_button("Download Summary (CSV)",
                           data=summary.to_csv(index=False).encode(),
                           file_name=f"smartdaas_outreach_summary_{datetime.date.today()}.csv",
                           mime="text/csv", use_container_width=True)
        st.caption("For programme director / donor reporting.")

    # Audit log
    if supabase is not None:
        try:
            supabase.table("audit_log").insert({
                "event_at": datetime.datetime.utcnow().isoformat(),
                "event_type": "outreach_plan",
                "n_patients": int(n_total),
                "n_high_risk": int(impact['n_high']),
                "report_type": "demo" if using_demo else "real",
            }).execute()
        except Exception:
            pass

    if using_demo:
        st.markdown("""<div style="background:#231a00;border-left:3px solid #ffb300;
            padding:0.75rem 1rem;border-radius:0 6px 6px 0;font-size:0.85rem;
            color:#ffb300;margin-top:1rem">Demo mode: based on synthetic data. Upload
            your programme data on the Patient Risk page for a real action plan.</div>""",
            unsafe_allow_html=True)


# ── Restructured three-tier sidebar navigation ──────────────────────────
_NAV_PAGES = [
    "🏠 Home", "❤️ Patient Risk", "🎯 Outreach Optimiser", "📄 Executive Report",
    "🏥 Facility Intelligence", "👥 Cohort Intelligence", "✅ Local Validation",
    "🧬 Model Transparency", "🔍 SHAP Explainability", "📘 Model Info",
    "📁 Sample Data", "🔐 Admin",
]
_NAV_T2_START = 5   # nth-child index of "Facility Intelligence"
_NAV_T3_START = 11  # nth-child index of "Sample Data"


def render_sidebar_nav(default_page="🏠 Home"):
    """Three-tier nav as a single radio. Returns the selected page key."""
    t2, t3 = _NAV_T2_START, _NAV_T3_START
    st.markdown(f"""
<style>
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]
    label:nth-child({t2}), [data-testid="stSidebar"] [data-testid="stRadio"]
    div[role="radiogroup"] label:nth-child({t3}) {{
    margin-top:14px !important;border-top:1px solid #1e2530 !important;
    padding-top:16px !important;position:relative;}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]
    label:nth-child({t2})::before {{content:"VALIDATION & METHODOLOGY";
    position:absolute;top:2px;left:4px;font-size:0.58rem;color:#5a6675;
    letter-spacing:1.6px;}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]
    label:nth-child({t3})::before {{content:"OPERATOR TOOLS";position:absolute;
    top:2px;left:4px;font-size:0.58rem;color:#5a6675;letter-spacing:1.6px;}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]
    label:nth-child({t3}) p, [data-testid="stSidebar"] [data-testid="stRadio"]
    div[role="radiogroup"] label:nth-child({t3+1}) p {{color:#6e7b8a !important;}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]
    label:nth-child(-n+4) p {{font-weight:500 !important;}}
</style>
""", unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Navigation</p>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.66rem;color:#00e5ff;letter-spacing:1.6px;'
                'text-transform:uppercase;padding:2px 0 2px 4px">Start here</div>'
                '<div style="font-size:0.66rem;color:#484f58;padding:0 0 8px 4px;'
                'line-height:1.4">Upload data -> build outreach plan -> send report.'
                '</div>', unsafe_allow_html=True)
    try:
        idx = _NAV_PAGES.index(default_page)
    except ValueError:
        idx = 0
    return st.radio("", _NAV_PAGES, index=idx,
                    label_visibility="collapsed", key="nav_main")


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="smartdaas-header">
    <p class="brand-name">SmartDaaS</p>
    <p class="brand-sub">AI-Powered HIV Programme Intelligence · Operational Analytics · Decision Support</p>
    <span class="version-tag">v1.0 · Decision-Support Platform · Pilot-Ready</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    def nav_group(label):
        st.markdown(
            f'<div style="font-size:0.62rem;color:#484f58;text-transform:uppercase;'
            f'letter-spacing:1.8px;padding:10px 0 3px 4px;margin-top:2px;'
            f'border-top:1px solid #21212a">{label}</div>',
            unsafe_allow_html=True
        )

    # ── Restructured three-tier navigation (page keys unchanged) ──
    page = render_sidebar_nav()

    # Local validation badge
    if st.session_state.get('recal_done'):
        local_auc = st.session_state.get('local_auc', 0)
        st.markdown(
            f'<div style="background:#0d2137;border:1px solid #3fb950;'
            f'border-radius:6px;padding:6px 10px;margin-top:4px;'
            f'font-size:0.75rem;color:#3fb950;text-align:center">'
            f'✅ Local AUC: {local_auc:.3f}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<p class="section-hdr">System Status</p>', unsafe_allow_html=True)
    if MODEL_OK:
        st.markdown(f"""
        <div class="metric-box" style="border-color:#3fb950">
            <div class="metric-val" style="color:#3fb950">0.772</div>
            <div class="metric-lbl">Model AUC</div>
        </div>""", unsafe_allow_html=True)
        st.success("✓ Model loaded")
    else:
        st.error(f"Model error: {MODEL_ERR[:80]}")
        st.info("Ensure cv_results.pkl and prepped_data.pkl are in the repo root.")

    st.markdown("---")
    st.markdown("""<div class="info-box">
        <strong>SmartDaaS v1.0</strong><br>
        Paper 1 → <a href="https://doi.org/10.64898/2026.05.15.26353325" style="color:#21d4fd">Preprint</a><br>
        Paper 2 → <a href="https://doi.org/10.64898/2026.05.15.26353326" style="color:#21d4fd">Preprint</a>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    db_status = "✓ Connected" if supabase is not None else "○ Demo mode"
    db_color = "#3fb950" if supabase is not None else "#8b949e"
    st.markdown(f'<div style="font-size:0.75rem;color:{db_color}">DB: {db_status}</div>',
                unsafe_allow_html=True)

    # Logout
    if st.button("← Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

# ── Mobile bottom navigation bar ─────────────────────────────
# Shown only on mobile (CSS hides it on desktop).
# Taps the corresponding sidebar radio label via JS so Streamlit
# treats it as a real nav event and reruns with the correct page.
_MOBILE_NAV = [
    ("🏠", "Home",     "🏠 Home"),
    ("❤️", "Risk",     "❤️ Patient Risk"),
    ("🎯", "Outreach", "🎯 Outreach Optimiser"),
    ("🏥", "Facility", "🏥 Facility Intelligence"),
    ("📄", "Report",   "📄 Executive Report"),
]
_current_page = st.session_state.get("nav_main", "🏠 Home")
_nav_html = "".join(
    '<a class="mobile-nav-item{active}" '
    'onclick="(function(){{var labels=window.parent.document'
    '.querySelectorAll(\'[data-testid=stSidebar] [data-testid=stRadio] label\');'
    'for(var i=0;i<labels.length;i++){{if(labels[i].innerText.trim().indexOf(\'{icon}\')===0)'
    '{{labels[i].click();break;}}}}}})()">'
    '<span class="nav-icon">{icon}</span><span>{label}</span></a>'.format(
        icon=icon, label=label,
        active=" active" if _current_page == key else ""
    )
    for icon, label, key in _MOBILE_NAV
)
st.markdown(
    f'<div class="mobile-nav-bar">{_nav_html}</div>',
    unsafe_allow_html=True
)

# ═════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠 Home":

    # ── VERSION BANNER — update this string on each deployment ──


    # ── HERO SECTION ──────────────────────────────────────
    st.markdown("""<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d2137 100%);
        border:1px solid #21d4fd22;border-radius:12px;padding:2rem 2.5rem;margin-bottom:1.5rem">
        <p style="font-size:0.8rem;color:#21d4fd;font-family:'IBM Plex Mono',monospace;
        text-transform:uppercase;letter-spacing:3px;margin:0 0 0.5rem 0">
        Predictive Intelligence for Public Health Programmes</p>
        <h2 style="color:#e6edf3;font-size:1.6rem;font-weight:700;margin:0 0 0.75rem 0">
        AI-powered intelligence for HIV programmes</h2>
        <p style="color:#cdd9e5;font-size:1rem;margin:0 0 1rem 0;line-height:1.6">
        Identify high-risk patients. Detect underperforming facilities.
        Quantify avoidable costs. Drive better outcomes.</p>
        <p style="color:#adbac7;font-size:1rem;margin:0;line-height:1.6">
        SmartDaaS combines patient-level risk prediction with facility intelligence
        and economic insights into one platform. Built on programme data from
        <strong style="color:#21d4fd">27,288 HIV patients on ART</strong> (Nigerian national HIV
        programme — discovery cohort) with local recalibration support for any country context.
        Designed for PEPFAR implementing partners, Global Fund grantees, and national
        HIV programme offices across sub-Saharan Africa.</p>
    </div>""", unsafe_allow_html=True)

    # ── KEY METRICS ───────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    for col, val, lbl in [
        (c1, "0.772", "Temporal Validation AUC"),
        (c2, "87.3%", "Sensitivity"),
        (c3, "27K", "Training Records"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── KEY CAPABILITIES ──────────────────────────────────
    st.markdown('<p class="section-hdr">Key Capabilities</p>', unsafe_allow_html=True)
    caps = [
        ("📊", "Patient Risk Engine",
         "Upload your data and instantly get individual risk scores with clear SHAP explanations. Know who needs attention before they are lost to follow-up."),
        ("🏥", "Facility Intelligence",
         "Identify care gaps and structural drivers of poor performance. Risk-adjusted benchmarking separates facility quality from patient case-mix."),
        ("💰", "Economic Impact Calculator",
         "Quantify excess poor outcomes and potential cost savings. Three scenario presets built on published PEPFAR cost data."),
        ("🎯", "Intervention Engine",
         "Specific, tiered recommendations aligned with PEPFAR MER indicators TX_CURR, TX_PVLS, TX_ML. CHW outreach, clinical review, retention tracing."),
        ("📄", "Executive Reports",
         "Professional one-click PDF reports for donors and leadership. Ready to present at quarterly review meetings."),
    ]
    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3, c1, c2]
    for i, (icon, title, desc) in enumerate(caps):
        with cols[i]:
            st.markdown(f"""<div style="background:#161b22;border:1px solid #444c56;
                border-radius:8px;padding:1rem;margin-bottom:0.75rem;min-height:130px">
                <p style="font-size:1rem;font-weight:600;color:#21d4fd;margin:0 0 0.4rem 0">
                {icon} {title}</p>
                <p style="font-size:0.95rem;color:#cdd9e5;margin:0;line-height:1.5">{desc}</p>
            </div>""", unsafe_allow_html=True)

    # ── LIVE PREVIEW ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-hdr">Live Platform Preview</p>', unsafe_allow_html=True)
    st.markdown("See SmartDaaS working — live demo data, no upload required.")

    if MODEL_OK:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(200, len(X_DEMO)), replace=False)
        df_prev = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        X_prev = df_prev[FEATURES].values.astype(float)
        probs_prev = model.predict_proba(X_prev)[:, 1]
        df_prev['risk_pct'] = (probs_prev * 100).round(1)
        df_prev['risk_label'] = ['HIGH' if p>=0.7 else 'MEDIUM' if p>=0.4 else 'LOW'
                                  for p in probs_prev]
        n_h = (df_prev['risk_label']=='HIGH').sum()
        n_m = (df_prev['risk_label']=='MEDIUM').sum()
        n_l = (df_prev['risk_label']=='LOW').sum()

        c_chart, c_risk, c_top = st.columns(3)
        with c_chart:
            fig, ax = plt.subplots(figsize=(3.5, 2.5), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            ax.hist(probs_prev*100, bins=15, color='#21d4fd', alpha=0.8, edgecolor='#0d1117')
            ax.axvline(70, color='#f85149', lw=1.5, linestyle='--')
            ax.axvline(40, color='#e3b341', lw=1.5, linestyle='--')
            ax.set_xlabel('Risk Score (%)', color='#adbac7', fontsize=8)
            ax.set_ylabel('Patients', color='#adbac7', fontsize=8)
            ax.set_title('Risk Distribution — 200 demo patients',
                         color='#e6edf3', fontsize=8, pad=6)
            ax.tick_params(colors='#adbac7', labelsize=7)
            for sp in ax.spines.values(): sp.set_color('#444c56')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with c_risk:
            st.markdown(f'<div class="risk-high" style="margin-bottom:0.5rem"><div class="risk-number" style="font-size:1.4rem">{n_h}</div><div class="risk-label">HIGH Risk ≥70%</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="risk-medium" style="margin-bottom:0.5rem"><div class="risk-number" style="font-size:1.4rem">{n_m}</div><div class="risk-label">MEDIUM Risk</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="risk-low"><div class="risk-number" style="font-size:1.4rem">{n_l}</div><div class="risk-label">LOW Risk</div></div>', unsafe_allow_html=True)

        with c_top:
            st.markdown("**Top 5 highest risk patients:**")
            top5 = df_prev.nlargest(5, 'risk_pct')[['risk_pct','risk_label']].copy()
            top5.index = [f"PT-{i:04d}" for i in range(len(top5))]
            top5.columns = ['Risk %', 'Tier']
            top5['Tier'] = top5['Tier'].map(
                {'HIGH':'🔴 HIGH','MEDIUM':'🟡 MED','LOW':'🟢 LOW'})
            st.dataframe(top5, use_container_width=True, height=210)
    else:
        st.info("Model loading — refresh in a moment.")

    # ── CTA ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""<div style="background:linear-gradient(135deg,#0d1f17,#0d2137);
        border:1px solid #21d4fd33;border-radius:10px;padding:1.5rem;text-align:center;
        margin-bottom:1rem">
        <p style="color:#21d4fd;font-size:1.1rem;font-weight:600;margin:0 0 0.5rem 0">
        Ready to see it in action?</p>
        <p style="color:#cdd9e5;margin:0 0 1rem 0">
        Upload your patient data or try with demo data →
        go to <strong>❤️ Patient Risk</strong> in the sidebar</p>
        <p style="color:#adbac7;font-size:0.95rem;margin:0">
        Or see the <strong>🛡️ Pilot Model</strong> page to learn about our
        Shadow Analytics Implementation Programme
        </p>
    </div>""", unsafe_allow_html=True)

    # ── RESEARCH FOUNDATION ───────────────────────────────
    st.markdown('<p class="section-hdr">Research Foundation</p>', unsafe_allow_html=True)
    st.markdown("""
> **Paper 1 — Under review at Scientific Reports**
> Chinthala LK. *Real-World Validation of Machine Learning Models for HIV Treatment Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis of 192,732 Clinical Records.* 2026.
> 📎 Code: [github.com/Kchinthala15/smartdaas-hiv-validation](https://github.com/Kchinthala15/smartdaas-hiv-validation)
> 📄 [Preprint — doi.org/10.64898/2026.05.15.26353325](https://doi.org/10.64898/2026.05.15.26353325)

> **Paper 2 — Submitted to BMJ Global Health**
> Chinthala LK. *Facility-Level Structural Drivers of HIV Treatment Outcomes:
> A Multi-Level Analysis of 27,288 Patients from a Nigerian HIV Programme and
> Implications for PEPFAR and Global Fund Programming.* 2026.
> 📄 [Preprint — doi.org/10.64898/2026.05.15.26353326](https://doi.org/10.64898/2026.05.15.26353326)

**Target users:** HIV programme managers · PEPFAR implementing partners ·
Global Fund grantees · National HIV programme officers · MoH analytics teams
""")

    # ── ABOUT THE DEVELOPER ───────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-hdr">About the Developer</p>', unsafe_allow_html=True)
    st.markdown("""<div style="background:#161b22;border:1px solid #444c56;
        border-radius:8px;padding:1.5rem;margin-bottom:1rem">
        <p style="color:#e6edf3;font-size:1rem;font-weight:600;margin:0 0 0.5rem 0">
        Lakshmi Kalyani Chinthala</p>
        <p style="color:#cdd9e5;margin:0 0 0.75rem 0;font-size:1rem">
        Founder, SmartDaaS &nbsp;·&nbsp; Independent Researcher, San Francisco, CA.
        Specialising in machine learning
        applications for global health programme management, with a focus on HIV/AIDS
        in sub-Saharan Africa.</p>
        <p style="color:#adbac7;font-size:0.95rem;margin:0">
        📧 chinthalakalyani1@gmail.com &nbsp;·&nbsp;
        🔗 <a href="https://github.com/Kchinthala15/smartdaas-hiv-validation"
        style="color:#21d4fd">GitHub</a> &nbsp;·&nbsp;
        🆔 ORCID: 0009-0009-8736-6673 &nbsp;·&nbsp;
        📍 San Francisco, CA, USA</p>
    </div>""", unsafe_allow_html=True)

    # ── DATA POLICY ───────────────────────────────────────
    st.markdown('<p class="section-hdr">Data Privacy & Handling Policy</p>',
                unsafe_allow_html=True)
    st.markdown("""<div style="background:#0d1f17;border:1px solid #3fb95044;
        border-radius:8px;padding:1.25rem;margin-bottom:1rem">
        <p style="color:#3fb950;font-weight:600;margin:0 0 0.5rem 0">
        🔒 Privacy by Design</p>
        <p style="color:#cdd9e5;font-size:0.875rem;margin:0;line-height:1.6">
        <strong>No data storage:</strong> Patient data is processed within your browser
        session and never stored, transmitted externally, or retained after session ends.<br><br>
        <strong>No third-party sharing:</strong> Your programme data is never shared
        with any third party.<br><br>
        <strong>Pilot deployments:</strong> A formal Data Use Agreement (DUA) is
        established with all pilot partners before any data is ingested. All pilots
        operate under Nigeria's NDPR and relevant PEPFAR data governance frameworks.
        </p>
    </div>""", unsafe_allow_html=True)

    # ── DISCLAIMER ────────────────────────────────────────
    st.markdown("""<div class="warn-box">
    ⚠️ SmartDaaS is a decision-support platform designed to assist HIV programme teams.
    All outputs should be reviewed by qualified programme and clinical staff.
    Local validation on your programme's data is strongly recommended before full operational use.
    </div>""", unsafe_allow_html=True)



# ═════════════════════════════════════════════════════════════
# PAGE 7 — SAMPLE DATA (moved up — referenced on Home page)
# ═════════════════════════════════════════════════════════════
elif page == "📁 Sample Data":
    st.markdown("""
### Sample Data & CSV Template

Download the template below, fill it with your patient data, then upload it on the
**❤️ Patient Risk** page to get risk scores.
""")

    st.markdown('<p class="section-hdr">Download Template</p>', unsafe_allow_html=True)

    sample_df = generate_sample_csv(n=20)

    st.markdown("""<div class="template-box">
<strong style="color:#21d4fd">SmartDaaS Patient Risk Template</strong><br><br>
This CSV contains 20 example patients showing the correct format and value ranges.
Replace the data with your own patients. Keep the column headers exactly as shown.
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download CSV Template (20 example patients)",
            data=sample_df.to_csv(index=False).encode(),
            file_name="smartdaas_patient_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        # Empty template with headers only
        empty_df = pd.DataFrame(columns=['patient_id'] + FEATURES)
        st.download_button(
            label="📥 Download Empty Template (headers only)",
            data=empty_df.to_csv(index=False).encode(),
            file_name="smartdaas_empty_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown('<p class="section-hdr">Preview (first 5 rows)</p>', unsafe_allow_html=True)
    st.dataframe(sample_df.head(), use_container_width=True)

    st.markdown('<p class="section-hdr">Column Reference Guide</p>', unsafe_allow_html=True)

    col_guide = pd.DataFrame({
        'Column Name': FEATURES,
        'Label': [FEAT_LABELS[f] for f in FEATURES],
        'Description': [FEAT_DESCRIPTIONS[f] for f in FEATURES],
        'Valid Range': [FEAT_RANGES[f] for f in FEATURES],
    })
    st.dataframe(col_guide, use_container_width=True, height=500)

    st.markdown('<p class="section-hdr">Accepted Column Name Variants</p>', unsafe_allow_html=True)
    st.markdown("""
SmartDaaS v1.0 automatically recognises common alternative column names.
For example, `age`, `patient_age`, and `Age` all map correctly. Here are the most common alternatives:
""")

    alias_examples = {
        'age / patient_age': 'Age',
        'sex / gender / female / is_female': 'sex_female',
        'cd4_at_start / cd4_start / baseline_cd4': 'Cd4AtStart',
        'most_recent_cd4 / cd4_recent / latest_cd4': 'MostRecentCd4Count',
        'who_stage / clinical_stage / who_clinical_stage': 'stage_start_num',
        'weight_at_start / baseline_weight': 'WeightAtStart',
        'days_to_art / diagnosis_to_art / art_delay': 'days_to_ART',
        'art_interruption / interruption / prior_interruption': 'had_interruption',
        'opportunistic_infection / oi': 'opp_infection',
        'tb / tuberculosis / tb_status': 'tb_positive',
    }

    alias_df = pd.DataFrame({
        'Your column name(s)': list(alias_examples.keys()),
        'Maps to': list(alias_examples.values()),
    })
    st.dataframe(alias_df, use_container_width=True)

    st.markdown("""<div class="info-box">
💡 If your column names are not in the list above and the auto-mapping fails,
the app will show you exactly which columns are missing and what they should be called.
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE 2 — PATIENT RISK
# ═════════════════════════════════════════════════════════════
elif page == "❤️ Patient Risk":
    st.markdown("""
### Patient Risk Engine

**Identify who needs attention most — before they are lost to follow-up.**

Upload your patient dataset and get instant risk scores for poor outcomes —
adherence issues, treatment interruption, or clinical deterioration.
Each score comes with a full SHAP explanation so your team knows exactly why a patient is flagged.
""")
    if not MODEL_OK:
        st.error("Model not loaded. Ensure cv_results.pkl and prepped_data.pkl are in the repo root.")
        st.stop()

    # -- DUA gate --------------------------------------------------
    # Checkboxes shown only if not yet acknowledged this session.
    # Once all 3 are ticked the page rerenders with the upload widget.
    if not is_dua_acknowledged():
        st.markdown('<p class="section-hdr">Data Governance</p>', unsafe_allow_html=True)
        dua_ok = render_dua_gate(context="upload")
        if not dua_ok:
            st.info("Please acknowledge all three statements above to unlock upload.")
            st.markdown('<p class="section-hdr">Or explore with demo data</p>', unsafe_allow_html=True)
            use_demo_only = st.checkbox(
                "Use demo data to explore the platform",
                value=False,
                help="200 patients from the training set - no upload required"
            )
            if not use_demo_only:
                st.stop()
            uploaded = None
            use_demo = True
        else:
            # Just acknowledged - fall through to upload widget below
            uploaded = None
            use_demo = False
    else:
        # Already acknowledged this session - show compact banner and proceed
        render_dua_gate(context="upload")
        uploaded = None
        use_demo = False

    # -- Upload widget (only shown when DUA is acknowledged) -------
    if is_dua_acknowledged():
        st.markdown('<p class="section-hdr">Upload Programme Data</p>', unsafe_allow_html=True)
        col_up, col_demo = st.columns([2, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Upload your patient CSV",
                type=['csv'],
                help="Use the template from the Sample Data page. Column names are flexible."
            )
        with col_demo:
            st.markdown("<br>", unsafe_allow_html=True)
            use_demo = st.checkbox(
                "Use demo data instead",
                value=False,
                help="200 patients from the training set - shows how the platform works"
            )
        if use_demo:
            uploaded = None
    # ── DHIS2 Direct Connection ───────────────────────────
    if is_dua_acknowledged():
        st.markdown('<p class="section-hdr">Or Connect Directly to DHIS2</p>', unsafe_allow_html=True)
        df_from_dhis2 = render_dhis2_connector()
        if df_from_dhis2 is not None and not use_demo:
            uploaded = None
            st.session_state['dhis2_active'] = True
        else:
            if not st.session_state.get('dhis2_active'):
                df_from_dhis2 = None

    # ── LOAD DATA ─────────────────────────────────────────
    df_input = None
    mappings_info = []

    # ── If df_scored already exists in session state and no new upload,
    # use it directly — this prevents the DUA gate from showing on
    # dropdown rerenders (selectbox changes trigger full page rerender
    # but the uploaded file object is lost; df_scored persists).
    _existing_scored = st.session_state.get('df_scored', None)
    _new_upload = uploaded is not None and not use_demo
    _dhis2_available = st.session_state.get('dhis2_df') is not None and not use_demo and uploaded is None

    if not _new_upload and not _dhis2_available and not use_demo and _existing_scored is not None:
        df_input = _existing_scored.copy()
        X_raw = df_input[FEATURES].values.astype(float)
        probs = model.predict_proba(X_raw)[:, 1]
        st.info(
            f"ℹ️ Showing previously uploaded cohort ({len(df_input):,} patients). "
            f"Upload a new file above to refresh."
        )

    # Check for DHIS2 data first
    _dhis2_df = st.session_state.get('dhis2_df') if is_dua_acknowledged() else None
    if df_input is None and _dhis2_df is not None and not use_demo and uploaded is None:
        try:
            df_raw = _dhis2_df.copy()
            st.markdown(f"""<div class="info-box">
            🔗 DHIS2 data loaded — {len(df_raw):,} patients pulled directly from your instance
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading DHIS2 data: {e}")
            df_raw = None

    elif df_input is None and uploaded is not None and not use_demo:
        try:
            # ── File size gate: max 50MB ─────────────────────────────────
            MAX_FILE_MB = 50
            MAX_ROWS = 100_000
            uploaded.seek(0, 2)
            file_size_mb = uploaded.tell() / (1024 * 1024)
            uploaded.seek(0)
            if file_size_mb > MAX_FILE_MB:
                st.error(
                    f"File too large ({file_size_mb:.1f} MB). "
                    f"Maximum upload size is {MAX_FILE_MB} MB. "
                    f"Please split your data into smaller batches."
                )
                st.stop()

            df_raw = pd.read_csv(uploaded)

            # ── Column name sanitization ─────────────────────────────────────
            # Strip dangerous characters from column names before any mapping.
            # Prevents injection via crafted CSV headers.
            def _sanitize_col_names(df):
                import re
                clean = {}
                for col in df.columns:
                    # Strip leading/trailing whitespace
                    c = str(col).strip()
                    # Remove null bytes and control characters
                    c = re.sub(r'[\x00-\x1f\x7f]', '', c)
                    # Replace semicolons, backticks, quotes that could cause issues
                    c = re.sub(r'[`;\'\"\\]', '_', c)
                    # Collapse multiple spaces/underscores
                    c = re.sub(r'\s+', '_', c)
                    c = re.sub(r'_+', '_', c)
                    # Truncate excessively long column names
                    c = c[:120] if len(c) > 120 else c
                    clean[col] = c if c else f'col_{list(df.columns).index(col)}'
                df = df.rename(columns=clean)
                return df

            df_raw = _sanitize_col_names(df_raw)

            # ── Row count gate ───────────────────────────────────────────
            if len(df_raw) > MAX_ROWS:
                st.error(
                    f"Upload contains {len(df_raw):,} rows. "
                    f"Maximum is {MAX_ROWS:,} rows per session. "
                    f"Please split into batches and analyse separately."
                )
                st.stop()

            st.markdown(f"""<div class="info-box">
            📂 File uploaded — {len(df_raw):,} rows,
            {len(df_raw.columns)} columns detected
            </div>""", unsafe_allow_html=True)

            # ── Step 0: PHIA / population-survey pre-processing ───
            # Detects PHIA-compatible column patterns and applies
            # recodes, proxies, composites, and derivations before
            # column normalisation. Stores derivation log in session.
            # Skipped silently if no PHIA-compatible columns detected.
            _progress = st.progress(0, text="Analysing data structure...")
            try:
                _progress.progress(10, text="Step 1 of 7 — Pre-processing...")
                df_raw, _derivation_log = preprocess_phia_compatible(df_raw)
                st.session_state['_phia_derivation_log'] = _derivation_log

                # ── Step 1: Column normalisation ──────────────────────
                _progress.progress(25, text="Step 2 of 7 — Mapping column names...")
                df_mapped, missing, mappings_applied = normalize_columns(df_raw)

                if mappings_applied:
                    with st.expander(f"ℹ️ Auto-mapped {len(mappings_applied)} column name(s)", expanded=False):
                        for orig, mapped in mappings_applied:
                            if mapped != '__art_inferred__':
                                st.markdown(f"- `{orig}` → `{mapped}`")

                # ── Step 2: ART status detection ──────────────────────
                _progress.progress(38, text="Step 3 of 7 — Detecting ART status...")
                art_confirmed, art_inferred, art_note = detect_art_status(df_raw)

                # ── Step 3: Tier detection ────────────────────────────
                _progress.progress(50, text="Step 4 of 7 — Detecting data tier...")
                tier, present, missing_core, standard_present, enhanced_present, tier_notes = \
                    detect_tier(df_mapped, art_confirmed, art_inferred)

                # ── Step 4: Pediatric flag ────────────────────────────
                _progress.progress(60, text="Step 5 of 7 — Checking patient population...")
                pediatric_indices = check_pediatric_patients(df_mapped)

                _progress.progress(70, text="Step 6 of 7 — Running data quality screening...")
            except Exception as _proc_err:
                _progress.empty()
                st.error(f"Processing error: {str(_proc_err)}")
                st.stop()

            # ── Step 5: Render tier report ────────────────────────
            st.markdown("---")
            can_proceed = render_tier_report(
                tier, present, missing_core, standard_present,
                enhanced_present, art_confirmed, art_inferred,
                art_note, pediatric_indices, df_mapped
            )
            st.markdown("---")

            if not can_proceed:
                _progress.empty()
                st.stop()

            # ── Step 6: Data quality screening ───────────────────
            available_features = [f for f in FEATURES if f in df_mapped.columns]
            dq_results = run_data_quality_screening(df_mapped, available_features)

            # Render the quality report
            render_data_quality_report(dq_results, len(df_mapped))

            # Apply imputation from quality results
            for feat, info in dq_results['missing'].items():
                if feat in df_mapped.columns:
                    df_mapped[feat] = df_mapped[feat].fillna(info['impute_val'])

            # Store quality results in session
            st.session_state['dq_grade'] = dq_results['grade']
            st.session_state['dq_results'] = dq_results

            # ── Step 6b: Validation metadata ─────────────────────
            _progress.progress(85, text="Step 7 of 7 — Building validation metadata...")
            # Build and render a transparent audit trail of all
            # inferences, proxies, imputations, and derivations.
            # Exposes caveats to pilot partners and funders.
            # Stored in session for access by other pages (e.g. Executive Report).
            _derivation_log = st.session_state.get('_phia_derivation_log', [])
            _val_meta = build_validation_metadata(
                df_raw=df_raw,
                df_mapped=df_mapped,
                mappings_applied=mappings_applied,
                missing_features=missing,
                derivation_log=_derivation_log,
                dq_results=dq_results,
                tier=tier,
            )
            st.session_state['validation_metadata'] = _val_meta
            render_validation_metadata(_val_meta)

            _progress.progress(100, text="✅ Data processed successfully!")
            _progress.empty()

            # ── Step 7: Derive engineered features + fill any remaining gaps ─
            # Runs for ALL tiers. Derives CD4_improvement, weight_change,
            # had_interruption, stage_worsened from raw columns where possible.
            # Any feature still missing is filled with 0 (neutral default).
            df_mapped, derived_feats, defaulted_feats = derive_engineered_features(df_mapped)

            if derived_feats:
                st.info(
                    f"ℹ️ **{len(derived_feats)} feature(s) derived** from your upload: "
                    f"{', '.join(derived_feats)}."
                )
            if defaulted_feats:
                st.caption(
                    f"⚙️ {len(defaulted_feats)} feature(s) not present in upload — "
                    f"set to neutral default (0) for prediction: "
                    f"{', '.join([FEAT_LABELS.get(f, f) for f in defaulted_feats])}. "
                    f"This may reduce prediction specificity for affected patients."
                )

            # ── Step 8: Patient ID ────────────────────────────────
            if 'patient_id' not in df_mapped.columns:
                df_mapped['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_mapped))]

            # ── Step 9: Store tier in session for downstream pages ─
            st.session_state['upload_tier'] = tier
            st.session_state['pediatric_indices'] = pediatric_indices
            st.session_state['art_inferred'] = art_inferred
            st.session_state['art_note'] = art_note

            # ── Step 10: Gate risk scoring by tier ───────────────
            if tier == 'CORE':
                df_input = df_mapped
                st.info(
                    "**Core Tier upload:** Cohort characterisation and IeDEA MUD regional "
                    "aggregate contextual benchmarks are available below. Patient-level risk "
                    "scores are not generated for Core tier uploads because the clinical "
                    "variables required for reliable risk prediction are not present. "
                    "To unlock risk scoring, add CD4 count, WHO clinical stage, TB status, "
                    "and days from diagnosis to ART start."
                )
            else:
                df_input = df_mapped
                if tier == 'STANDARD':
                    st.warning(
                        "**Standard Tier upload:** Risk estimates will be generated "
                        "using partial feature availability. Prediction confidence and "
                        "stability may vary depending on which clinical variables are "
                        "present. Interpret scores alongside clinical judgement and "
                        "local programme context."
                    )

            n_patients = len(df_input)
            st.success(f"✓ **{n_patients:,} patients** loaded — Tier: **{tier}**")

            # Log upload to Supabase
            log_upload(
                supabase, n_patients, 0, 0, 0.0,
                source=uploaded.name,
                tier=tier,
                dq_grade=dq_results.get('grade', 'unknown'),
                art_inferred=art_inferred,
                n_pediatric=len(pediatric_indices),
                dua_ack=is_dua_acknowledged(),
            )

        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.markdown(
                "Make sure it is a valid CSV. "
                "Download the template from **📁 Sample Data** if needed."
            )
            st.stop()

    elif df_input is None and (use_demo or uploaded is None):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(200, len(X_DEMO)), replace=False)
        df_input = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        df_input['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_input))]
        st.markdown("""<div class="info-box">
        🔬 <strong>Demo mode:</strong> 200 patients from the training set.
        Upload your own CSV above to analyse real patients.
        </div>""", unsafe_allow_html=True)

    if df_input is None:
        st.info("Upload a CSV above or check 'Use demo data' to begin.")
        st.stop()

    # ── RUN PREDICTIONS (gated by tier) ───────────────────
    current_tier = st.session_state.get('upload_tier', 'ENHANCED')

    # Show calibration status
    if st.session_state.get('recal_done'):
        local_auc = st.session_state.get('local_auc', 0)
        local_thresh = st.session_state.get('local_threshold', 0.70)
        st.success(
            f"✅ **Local calibration active** — Using locally-validated model "
            f"(AUC: {local_auc:.3f}, threshold: {local_thresh:.3f}). "
            f"Risk scores reflect your programme's local calibration, not the "
            f"Nigerian discovery cohort baseline."
        )
    else:
        st.info(
            "ℹ️ Using baseline model (Nigerian discovery cohort, AUC: 0.772). "
            "Run **✅ Local Validation** to adapt performance to your programme population."
        )

    if current_tier == 'CORE' and uploaded is not None and not use_demo:
        # Core tier: no risk scores — show cohort summary only
        st.markdown('<p class="section-hdr">Cohort Summary</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">'
                       f'{len(df_input):,}</div><div class="metric-lbl">Total Patients</div>'
                       f'</div>', unsafe_allow_html=True)
        if 'Age' in df_input.columns:
            with c2:
                st.markdown(f'<div class="metric-box"><div class="metric-val">'
                           f'{pd.to_numeric(df_input["Age"], errors="coerce").median():.0f}'
                           f'</div><div class="metric-lbl">Median Age</div>'
                           f'</div>', unsafe_allow_html=True)
        if 'sex_female' in df_input.columns:
            pct_female = pd.to_numeric(
                df_input['sex_female'], errors='coerce').mean() * 100
            with c3:
                st.markdown(f'<div class="metric-box"><div class="metric-val">'
                           f'{pct_female:.0f}%</div>'
                           f'<div class="metric-lbl">Female</div>'
                           f'</div>', unsafe_allow_html=True)
        st.info(
            "**Core Tier:** Patient risk scores require CD4 count, WHO clinical stage, "
            "TB status, and days from diagnosis to ART. See the upgrade guidance above."
        )
        st.stop()

    if df_input is not None and 'risk_pct' not in df_input.columns:
        with st.spinner("Running risk predictions..."):
            df_input, X_raw, probs = run_predictions(df_input)
        # Make the scored cohort available to other pages
        st.session_state['df_scored'] = df_input.copy()
    elif df_input is not None and 'risk_pct' in df_input.columns:
        # Already scored — recompute X_raw and probs for SHAP
        X_raw = df_input[FEATURES].values.astype(float)
        probs = model.predict_proba(X_raw)[:, 1]

    # Standard tier confidence disclosure
    if current_tier == 'STANDARD' and uploaded is not None and not use_demo:
        st.warning(
            "**Standard Tier upload:** Risk estimates were generated using partial "
            "feature availability. Prediction confidence and stability may vary "
            "depending on which clinical variables are present. Missing Enhanced tier "
            "variables were imputed with neutral defaults. Interpret all scores "
            "alongside clinical judgement and local programme context."
        )

    n_high = (df_input['risk_label'] == 'HIGH').sum()
    n_med = (df_input['risk_label'] == 'MEDIUM').sum()
    n_low = (df_input['risk_label'] == 'LOW').sum()
    n_tot = len(df_input)

    # Update Supabase log with accurate risk breakdown
    if uploaded is not None and not use_demo:
        log_upload(
            supabase, n_tot, n_high, n_med,
            float(df_input['risk_pct'].mean()),
            source=uploaded.name,
            tier=st.session_state.get('upload_tier', 'unknown'),
            dq_grade=st.session_state.get('dq_grade', 'unknown'),
            art_inferred=st.session_state.get('art_inferred', False),
            n_pediatric=len(st.session_state.get('pediatric_indices', [])),
            dua_ack=is_dua_acknowledged(),
        )

    # ── SUMMARY CARDS ─────────────────────────────────────
    st.markdown('<p class="section-hdr">Risk Summary</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="risk-high"><div class="risk-number">{n_high}</div><div class="risk-label">High Risk ≥70%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="risk-medium"><div class="risk-number">{n_med}</div><div class="risk-label">Medium Risk 40–69%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="risk-low"><div class="risk-number">{n_low}</div><div class="risk-label">Low Risk &lt;40%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{n_high/n_tot*100:.0f}%</div><div class="metric-lbl">High Risk Rate</div></div>', unsafe_allow_html=True)

    # ── INLINE SHAP PREVIEW ───────────────────────────────
    st.markdown('<p class="section-hdr">🧠 Quick SHAP Preview — Top High Risk Patient</p>',
                unsafe_allow_html=True)
    st.markdown("Why is the highest-risk patient flagged? No need to navigate away.")

    top_idx = int(df_input['risk_pct'].values.argmax())
    top_row = df_input.iloc[top_idx]
    top_x = X_raw[top_idx]
    top_sv, sv_ok = compute_shap_single(top_x)
    top_sv_arr = np.array(top_sv).flatten()

    sv_order = [int(i) for i in np.argsort(np.abs(top_sv_arr))[-8:] if int(i) < len(FEATURES)]
    sv_vals = [float(top_sv_arr[i]) for i in sv_order]
    sv_names = [FEAT_LABELS.get(FEATURES[i], FEATURES[i]) for i in sv_order]
    sv_colors = ['#f85149' if v > 0 else '#3fb950' for v in sv_vals]

    c_info, c_shap = st.columns([1, 2])
    with c_info:
        risk_color = "#f85149" if top_row['risk_label'] == 'HIGH' else "#e3b341"
        st.markdown(f"""<div style="background:#161b22;border:1px solid #f8514933;
            border-radius:8px;padding:1rem;text-align:center">
            <div style="font-size:2rem;font-weight:700;color:{risk_color};
            font-family:'IBM Plex Mono',monospace">{top_row['risk_pct']:.1f}%</div>
            <div style="font-size:0.75rem;color:#adbac7;text-transform:uppercase">
            {top_row['patient_id']} · {top_row['risk_label']} RISK</div>
        </div>""", unsafe_allow_html=True)
        top3_idx = [int(x) for x in np.argsort(np.abs(top_sv_arr))[-3:][::-1]
                    if int(x) < len(FEATURES)]
        st.markdown("**Top contributing risk factors:**")
        for ii in top3_idx:
            direction = "↑ increases" if top_sv_arr[ii] > 0 else "↓ reduces"
            st.markdown(f"- {FEAT_LABELS.get(FEATURES[ii], FEATURES[ii])}: {direction} risk")

    with c_shap:
        fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        ax.barh(range(len(sv_names)), sv_vals, color=sv_colors,
                height=0.6, edgecolor='#0d1117')
        ax.axvline(0, color='#8b949e', lw=0.8)
        ax.set_yticks(range(len(sv_names)))
        ax.set_yticklabels(sv_names, fontsize=8, color='#e6edf3')
        ax.set_xlabel('SHAP Value', color='#adbac7', fontsize=8)
        ax.tick_params(colors='#adbac7', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#444c56')
        ax.set_title(f'Why {top_row["patient_id"]} is {top_row["risk_label"]} risk',
                     color='#e6edf3', fontsize=9, pad=6)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("""<div class="info-box">
    💡 Go to <strong>🔍 SHAP Explainability</strong> for full per-patient waterfall charts
    across all 15 features for any patient in your cohort.
    </div>""", unsafe_allow_html=True)
    c_chart, c_table = st.columns([1, 2])
    with c_chart:
        st.markdown('<p class="section-hdr">Risk Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        ax.hist(probs, bins=25, color='#21d4fd', alpha=0.75, edgecolor='#0d1117')
        ax.axvline(0.7, color='#f85149', lw=1.8, linestyle='--', label='HIGH (0.70)')
        ax.axvline(0.4, color='#e3b341', lw=1.8, linestyle='--', label='MEDIUM (0.40)')
        ax.set_xlabel('Risk Score', color='#8b949e', fontsize=9)
        ax.set_ylabel('Patients', color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#8b949e')
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c_table:
        st.markdown('<p class="section-hdr">Patient Risk Scores</p>', unsafe_allow_html=True)
        show_cols = [c for c in ['patient_id', 'risk_pct', 'risk_label', 'Age',
                                  'Cd4AtStart', 'stage_start_num', 'had_interruption']
                     if c in df_input.columns]
        df_show = df_input[show_cols].copy()
        df_show['risk_label'] = df_show['risk_label'].map(
            {'HIGH': '🔴 HIGH', 'MEDIUM': '🟡 MEDIUM', 'LOW': '🟢 LOW'}
        ).fillna('🟢 LOW')
        df_show = df_show.sort_values('risk_pct', ascending=False)
        st.dataframe(df_show, height=260, use_container_width=True)

    # ── SHAP GLOBAL ───────────────────────────────────────
    st.markdown('<p class="section-hdr">Global Feature Importance (SHAP)</p>', unsafe_allow_html=True)
    with st.spinner("Computing SHAP explanations..."):
        mean_sv, shap_exp = compute_shap(X_raw)

    order = np.argsort(mean_sv)
    order = [int(i) for i in order if int(i) < len(FEATURES)]
    f_names = [FEAT_LABELS.get(FEATURES[i], FEATURES[i]) for i in order]
    f_vals = np.array([mean_sv[i] for i in order])
    f_colors = ['#21d4fd' if v >= np.percentile(f_vals, 60) else '#0072b2' for v in f_vals]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    ax.barh(range(len(f_names)), f_vals, color=f_colors, height=0.65, edgecolor='#0d1117', linewidth=0.3)
    for i, v in enumerate(f_vals):
        ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=7.5, color='#8b949e')
    ax.set_yticks(range(len(f_names)))
    ax.set_yticklabels(f_names, fontsize=9, color='#e6edf3')
    ax.set_xlabel('Mean |SHAP Value| — average impact on risk score', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.set_title('Global Feature Importance — All Patients', color='#e6edf3', fontsize=10, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── INDIVIDUAL PATIENT EXPLORER ───────────────────────
    st.markdown('<p class="section-hdr">Individual Patient Explorer</p>', unsafe_allow_html=True)
    pid_list = df_input['patient_id'].tolist()
    best_pos = int(df_input['risk_pct'].values.argmax())
    sel_id = st.selectbox("Select patient for explanation:", pid_list, index=best_pos)
    sel_pos = df_input[df_input['patient_id'] == sel_id].index[0]
    pos = df_input.index.get_loc(sel_pos)
    row = df_input.loc[sel_pos]
    pt_lbl = str(row['risk_label'])
    pt_sv, sv_ok = compute_shap_single(X_raw[pos])

    # ── Pediatric flag ────────────────────────────────────────
    pediatric_ids = st.session_state.get('pediatric_indices', [])
    is_pediatric = sel_pos in pediatric_ids
    if is_pediatric:
        patient_age = row.get('Age', '?')
        st.warning(
            f"⚠️ **Pediatric patient** (age {patient_age:.0f}): "
            f"This model was trained on patients aged 15 and above. "
            f"The risk score below is **not validated for paediatric HIV care**. "
            f"Do not use this score for clinical decision-making without review "
            f"by a qualified paediatric HIV clinician."
        )

    c_s, c_e = st.columns([1, 2])
    with c_s:
        cc = 'risk-high' if pt_lbl == 'HIGH' else 'risk-medium' if pt_lbl == 'MEDIUM' else 'risk-low'
        pediatric_badge = " · ⚠️ PAEDIATRIC" if is_pediatric else ""
        st.markdown(f"""<div class="{cc}" style="margin-bottom:1rem">
            <div class="risk-number">{row['risk_pct']:.1f}%</div>
            <div class="risk-label">{pt_lbl} RISK{pediatric_badge}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("**Key Clinical Values:**")
        for feat in ['Age', 'Cd4AtStart', 'MostRecentCd4Count', 'stage_start_num',
                     'days_to_ART', 'had_interruption', 'tb_positive', 'side_effects']:
            if feat in row:
                val = row[feat]
                st.markdown(f"- {FEAT_LABELS.get(feat, feat)}: **{val:.1f}**")
        st.markdown("**Recommended Actions:**")
        for action in INTERVENTIONS.get(pt_lbl, INTERVENTIONS['LOW']):
            st.markdown(action)

    with c_e:
        sv_order = [int(i) for i in np.argsort(np.abs(pt_sv))[-10:] if int(i) < len(FEATURES)]
        sv_vals = np.array([pt_sv[i] for i in sv_order])
        sv_names = [FEAT_LABELS.get(FEATURES[i], FEATURES[i]) for i in sv_order]
        sv_colors = ['#f85149' if float(v) > 0 else '#3fb950' for v in sv_vals]
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        ax.barh(range(len(sv_names)), sv_vals, color=sv_colors, height=0.65, edgecolor='#0d1117', linewidth=0.3)
        ax.axvline(0, color='#8b949e', lw=0.8)
        ax.set_yticks(range(len(sv_names)))
        ax.set_yticklabels(sv_names, fontsize=8.5, color='#e6edf3')
        ax.set_xlabel('SHAP Value (🔴 increases risk · 🟢 reduces risk)', color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        lbl_type = "SHAP" if sv_ok else "Feature Importance (SHAP unavailable)"
        ax.set_title(f"Clinical Feature Contributions: {sel_id} | Risk: {row['risk_pct']:.1f}% ({pt_lbl})",
                     color='#e6edf3', fontsize=9, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""<div class="info-box">
    ℹ️ <strong>SHAP Interpretation Note:</strong> Feature contributions reflect model-learned
    patterns from training data. Individual values should be interpreted alongside clinical
    judgement and local programme context. Feature contributions may vary across facility
    types and patient populations. Sex-based contributions reflect cohort-level statistical
    patterns — not individual clinical determinism. If facility tier data is available,
    interpret contributions in that structural context.
    <br><br><em>Local validation is required before deployment within real-world programme environments.</em>
    </div>""", unsafe_allow_html=True)

    # ── EXPORT ────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Export Results</p>', unsafe_allow_html=True)
    export = df_input[['patient_id', 'risk_pct', 'risk_label'] + FEATURES].copy()
    # ── Provenance metadata ──────────────────────────────
    export.insert(0, 'smartdaas_version', 'v1.0')
    export.insert(1, 'export_timestamp_utc', datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
    export.insert(2, 'session_id', _get_session_id())
    export.insert(3, 'model_auc_temporal', '0.772')
    export.insert(4, 'disclaimer', 'Decision-support output. Review by qualified programme staff required before operational use.')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "📥 All Risk Scores (CSV)",
            data=export.to_csv(index=False).encode(),
            file_name="smartdaas_risk_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        high_df = export[export['risk_label'] == 'HIGH']
        st.download_button(
            f"🚨 High Risk Only ({n_high} patients)",
            data=high_df.to_csv(index=False).encode(),
            file_name="smartdaas_high_risk.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        action_df = export[export['risk_label'].isin(['HIGH', 'MEDIUM'])].copy()
        action_df['recommended_action'] = action_df['risk_label'].map({
            'HIGH': INTERVENTIONS['HIGH'][0].replace("🔴 ", ""),
            'MEDIUM': INTERVENTIONS['MEDIUM'][0].replace("🟡 ", ""),
        })
        st.download_button(
            f"📋 Action List ({len(action_df)} patients)",
            data=action_df.to_csv(index=False).encode(),
            file_name="smartdaas_action_list.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═════════════════════════════════════════════════════════════
# PAGE 3 — SHAP EXPLAINABILITY
# ═════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Explainability":
    st.markdown("""
### Why Did the Model Flag This Patient?

Select any patient to see a full clinical explanation of their risk score.
Each factor is shown with its exact SHAP contribution — **red increases risk, green reduces it.**
""")

    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    # ── Use uploaded/scored data if available, else fall back to demo ──
    _scored = st.session_state.get('df_scored', None)
    if _scored is not None and len(_scored) > 0:
        df_shap = _scored.copy()
        # Ensure all feature columns exist and are numeric
        for f in FEATURES:
            if f not in df_shap.columns:
                df_shap[f] = 0.0
            else:
                df_shap[f] = pd.to_numeric(df_shap[f], errors='coerce').fillna(0.0)
        X_shap = df_shap[FEATURES].values.astype(float)
        probs_shap = model.predict_proba(X_shap)[:, 1]
        df_shap['risk_pct'] = (probs_shap * 100).round(1)
        if 'risk_label' not in df_shap.columns:
            df_shap['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                                     for p in probs_shap]
        if 'patient_id' not in df_shap.columns:
            df_shap['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_shap))]
        st.info(f"ℹ️ Showing SHAP explanations for your uploaded cohort ({len(df_shap):,} patients).")
    else:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(200, len(X_DEMO)), replace=False)
        df_shap = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        df_shap['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_shap))]
        X_shap = df_shap[FEATURES].values.astype(float)
        probs_shap = model.predict_proba(X_shap)[:, 1]
        df_shap['risk_pct'] = (probs_shap * 100).round(1)
        df_shap['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                                 for p in probs_shap]
        st.markdown("""<div class="info-box">
        🔬 <strong>Demo mode:</strong> Upload patient data on the Patient Risk page first
        to see SHAP explanations for your own cohort.
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Select Patient</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        risk_filter = st.selectbox("Filter by risk tier:", ["All patients", "HIGH only", "MEDIUM only", "LOW only"])
    with c2:
        sort_by = st.selectbox("Sort by:", ["Highest risk first", "Lowest risk first", "Patient ID"])

    df_filtered = df_shap.copy()
    if risk_filter == "HIGH only":
        df_filtered = df_filtered[df_filtered['risk_label'] == 'HIGH']
    elif risk_filter == "MEDIUM only":
        df_filtered = df_filtered[df_filtered['risk_label'] == 'MEDIUM']
    elif risk_filter == "LOW only":
        df_filtered = df_filtered[df_filtered['risk_label'] == 'LOW']
    if sort_by == "Highest risk first":
        df_filtered = df_filtered.sort_values('risk_pct', ascending=False)
    elif sort_by == "Lowest risk first":
        df_filtered = df_filtered.sort_values('risk_pct', ascending=True)

    if len(df_filtered) == 0:
        st.warning("No patients match this filter.")
        st.stop()

    sel_id = st.selectbox(f"Select patient ({len(df_filtered)} shown):", df_filtered['patient_id'].tolist(), index=0)
    sel_row = df_filtered[df_filtered['patient_id'] == sel_id].iloc[0]
    # Use positional index in X_shap array — reset_index ensures alignment
    df_shap_reset = df_shap.reset_index(drop=True)
    sel_pos = df_shap_reset[df_shap_reset['patient_id'] == sel_id].index[0]
    pt_lbl = sel_row['risk_label']
    pt_prob = sel_row['risk_pct']
    x_row = X_shap[sel_pos]
    pt_sv, sv_ok = compute_shap_single(x_row)

    # Top 3 drivers
    top3_idx = [int(x) for x in np.argsort(np.abs(pt_sv))[-3:][::-1] if int(x) < len(FEATURES)]
    pt_sv_arr = np.array(pt_sv).flatten()
    pt_sv_list = [float(pt_sv_arr[i]) if i < len(pt_sv_arr) else 0.0 for i in range(len(FEATURES))]
    drivers = []
    for ii in top3_idx:
        feat_name = FEAT_LABELS.get(FEATURES[ii], FEATURES[ii])
        direction = "increases" if pt_sv_list[ii] > 0 else "reduces"
        drivers.append(f"**{feat_name}** {direction} risk (SHAP: {pt_sv_list[ii]:+.3f})")

    # Risk badge
    cc = 'risk-high' if pt_lbl == 'HIGH' else 'risk-medium' if pt_lbl == 'MEDIUM' else 'risk-low'
    risk_color = "#f85149" if pt_lbl == "HIGH" else "#e3b341" if pt_lbl == "MEDIUM" else "#3fb950"

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.markdown(f"""<div class="{cc}" style="text-align:center;padding:1.5rem">
            <div class="risk-number">{pt_prob:.1f}%</div>
            <div class="risk-label">{pt_lbl} RISK</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        avg_all = df_shap['risk_pct'].mean()
        avg_high = df_shap[df_shap['risk_label'] == 'HIGH']['risk_pct'].mean()
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">{avg_all:.1f}%</div>
            <div class="metric-lbl">Cohort Average</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-box" style="margin-top:0.5rem">
            <div class="metric-val">{avg_high:.1f}%</div>
            <div class="metric-lbl">HIGH Risk Average</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div style="background:#161b22;border-left:4px solid {risk_color};
            padding:1rem;border-radius:0 8px 8px 0;">
            <strong style="color:{risk_color}">Why is {sel_id} {pt_lbl} risk?</strong><br><br>
            Top contributing risk factors:<br>
            1. {drivers[0]}<br>
            2. {drivers[1]}<br>
            3. {drivers[2]}
        </div>""", unsafe_allow_html=True)

    # SHAP waterfall
    st.markdown('<p class="section-hdr">SHAP Waterfall — Feature Contributions</p>', unsafe_allow_html=True)
    sv_order = [int(x) for x in np.argsort(np.abs(pt_sv)) if int(x) < len(FEATURES)]
    sv_vals = [float(pt_sv_arr[i]) for i in sv_order]
    sv_names_vals = []
    for ii in sv_order:
        val = float(sel_row[FEATURES[ii]])
        sv_names_vals.append(f"{FEAT_LABELS.get(FEATURES[ii], FEATURES[ii])} = {val:.1f}")
    sv_colors = ['#f85149' if float(v) > 0 else '#3fb950' for v in sv_vals]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    bars = ax.barh(range(len(sv_names_vals)), sv_vals, color=sv_colors, height=0.7, edgecolor='#0d1117', linewidth=0.3)
    for i, (bar, v) in enumerate(zip(bars, sv_vals)):
        x_pos = v + 0.002 if v >= 0 else v - 0.002
        ha = 'left' if v >= 0 else 'right'
        ax.text(x_pos, i, f'{v:+.4f}', va='center', ha=ha, fontsize=8.5, color='#e6edf3')
    ax.set_yticks(range(len(sv_names_vals)))
    ax.set_yticklabels(sv_names_vals, fontsize=9, color='#e6edf3')
    ax.axvline(0, color='#8b949e', lw=1.2)
    ax.set_xlabel('SHAP Value — contribution to risk score', color='#8b949e', fontsize=10)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    ax.set_title(f'SHAP Explanation: {sel_id} | Risk: {pt_prob:.1f}% ({pt_lbl})',
                 color='#e6edf3', fontsize=11, pad=12, fontweight='bold')
    baseline = probs_shap.mean() * 100
    ax.text(0.99, 0.02, f'Cohort baseline: {baseline:.1f}%',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#8b949e', style='italic')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Clinical profile table
    st.markdown('<p class="section-hdr">Full Clinical Profile</p>', unsafe_allow_html=True)
    profile_data = []
    for feat in FEATURES:
        val = sel_row[feat]
        feat_idx = FEATURES.index(feat)
        sv_val = pt_sv_list[feat_idx] if feat_idx < len(pt_sv_list) else 0.0
        impact = "🔴 Risk factor" if sv_val > 0.01 else "🟢 Protective" if sv_val < -0.01 else "⚪ Neutral"
        profile_data.append({
            'Feature': FEAT_LABELS.get(feat, feat),
            'Patient Value': f"{val:.1f}",
            'SHAP Value': f"{sv_val:+.4f}",
            'Clinical Impact': impact
        })
    profile_df = pd.DataFrame(profile_data).sort_values('SHAP Value', ascending=False)
    st.dataframe(profile_df, use_container_width=True, height=400)

    # Actions
    st.markdown('<p class="section-hdr">Recommended Clinical Actions</p>', unsafe_allow_html=True)
    for action in INTERVENTIONS.get(pt_lbl, INTERVENTIONS['LOW']):
        st.markdown(action)

    # Export
    st.markdown('<p class="section-hdr">Export Patient Explanation</p>', unsafe_allow_html=True)
    export_explanation = pd.DataFrame({
        'Patient ID': [sel_id], 'Risk Score (%)': [pt_prob], 'Risk Tier': [pt_lbl],
        'Top Driver 1': [drivers[0].replace("**", "")],
        'Top Driver 2': [drivers[1].replace("**", "")],
        'Top Driver 3': [drivers[2].replace("**", "")],
        'Recommended Action 1': [INTERVENTIONS[pt_lbl][0].replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")],
        'Recommended Action 2': [INTERVENTIONS[pt_lbl][1].replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")],
    })
    st.download_button(
        "📥 Download Patient Explanation (CSV)",
        data=export_explanation.to_csv(index=False).encode(),
        file_name=f"smartdaas_explanation_{sel_id}.csv",
        mime="text/csv"
    )


# ═════════════════════════════════════════════════════════════
# PAGE 4 — FACILITY INTELLIGENCE
# ═════════════════════════════════════════════════════════════
elif page == "🏥 Facility Intelligence":
    st.markdown("""
### Facility-Level HIV Programme Intelligence

Identifies structural drivers of poor outcomes across facility levels, ownership types,
and funding models.

> **Nigerian Discovery Cohort Findings** — the statistics below are derived from
> 27,288 patients in the Nigerian national HIV programme (Paper 2, submitted to BMJ Global Health).
> These findings represent hypothesis-generating evidence from a single-country discovery
> cohort. External validation across additional countries and health systems is ongoing.
> Results should not be assumed to generalise directly to other programme contexts.
""")

    # Load or generate facility data
    try:
        fac_df = pd.read_csv('facility_summary.csv')
        perf_df = pd.read_csv('facility_performance.csv')
    except Exception:
        fac_df = pd.DataFrame({
            'Health facility level': [
                'Primary health center', 'Primary health center', 'Primary health center',
                'Secondary health facility', 'Secondary health facility',
                'Secondary health facility', 'Secondary health facility',
                'Tertiary hospital', 'Tertiary hospital', 'Tertiary hospital', 'Tertiary hospital'
            ],
            'FacilityType': [
                'Faith Based', 'Private for profit', 'Public',
                'Faith Based', 'Private for profit', 'Private not for profit', 'Public',
                'Faith Based', 'Private for profit', 'Private not for profit', 'Public'
            ],
            'N': [12, 64, 445, 2923, 429, 91, 14539, 238, 3, 1, 8543],
            'poor_outcome': [0.0, 0.266, 0.142, 0.122, 0.126, 0.066, 0.123, 0.181, 0.333, 0.0, 0.100],
            'poor_adh': [0.0, 0.172, 0.040, 0.032, 0.035, 0.022, 0.040, 0.025, 0.0, 0.0, 0.026],
            'mortality': [0.0, 0.0, 0.011, 0.008, 0.002, 0.0, 0.010, 0.004, 0.0, 0.0, 0.007],
            'interrupted': [0.0, 0.156, 0.099, 0.094, 0.103, 0.044, 0.094, 0.168, 0.333, 0.0, 0.079],
            'cd4_mean': [459.9, 328.9, 377.8, 391.5, 287.3, 354.5, 442.7, 283.0, 171.0, 391.0, 506.8],
        })
        perf_df = None

    # Summary metrics
    st.markdown('<p class="section-hdr">Key Findings (Paper 2)</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-box"><div class="metric-val">1.95×</div><div class="metric-lbl">Primary HC vs Tertiary OR</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-box"><div class="metric-val">1.24×</div><div class="metric-lbl">NGO Funding OR</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-box"><div class="metric-val">2.2%</div><div class="metric-lbl">ICC (facility clustering)</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-box"><div class="metric-val">p&lt;0.001</div><div class="metric-lbl">LR Test: facility vars</div></div>', unsafe_allow_html=True)

    # Outcomes heatmap
    st.markdown('<p class="section-hdr">Composite Poor Outcome Rate by Facility Type</p>', unsafe_allow_html=True)
    levels = ['Primary health center', 'Secondary health facility', 'Tertiary hospital']
    types = ['Public', 'Faith Based', 'Private for profit', 'Private not for profit']
    hmap = np.full((len(types), len(levels)), np.nan)
    for _, row in fac_df.iterrows():
        lvl, ftp = row['Health facility level'], row['FacilityType']
        if lvl in levels and ftp in types:
            hmap[types.index(ftp), levels.index(lvl)] = row['poor_outcome'] * 100

    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    valid = hmap[~np.isnan(hmap)]
    vmax = max(valid.max(), 20) if len(valid) else 20
    im = ax.imshow(hmap, cmap='RdYlGn_r', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Primary\nHC', 'Secondary\nHF', 'Tertiary\nHosp'], fontsize=11, color='#e6edf3')
    ax.set_yticks(range(4))
    ax.set_yticklabels(types, fontsize=11, color='#e6edf3')
    ax.set_title('Composite Poor Outcome Rate (%) by Facility Level and Ownership', color='#e6edf3', fontsize=11, pad=10)
    for ti in range(4):
        for li in range(3):
            val = hmap[ti, li]
            if not np.isnan(val):
                ax.text(li, ti, f'{val:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold',
                        color='white' if val > vmax * 0.7 else 'black')
            else:
                ax.text(li, ti, 'n/a', ha='center', va='center', fontsize=9, color='#555555')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Poor outcome rate (%)', color='#8b949e', fontsize=9)
    cbar.ax.tick_params(colors='#8b949e', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Outcome bars
    st.markdown('<p class="section-hdr">Outcome Rates by Facility Level</p>', unsafe_allow_html=True)
    outcomes_map = {'poor_outcome': 'Composite', 'poor_adh': 'Poor Adherence', 'mortality': 'Mortality', 'interrupted': 'ART Interruption'}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), facecolor='#161b22')
    for ax_i, (col, label) in enumerate(outcomes_map.items()):
        ax = axes[ax_i]
        ax.set_facecolor('#161b22')
        lvl_rates, lvl_short = [], []
        for lvl in levels:
            sub = fac_df[fac_df['Health facility level'] == lvl]
            if len(sub) > 0:
                rate = np.average(sub[col], weights=sub['N']) * 100
                lvl_rates.append(rate)
                lvl_short.append(lvl.split()[0])
        colors_bar = ['#CC79A7', '#56B4E9', '#0072B2'][:len(lvl_rates)]
        bars = ax.bar(range(len(lvl_rates)), lvl_rates, color=colors_bar, width=0.55, edgecolor='#0d1117')
        for bar, v in zip(bars, lvl_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{v:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e6edf3')
        ax.set_xticks(range(len(lvl_short)))
        ax.set_xticklabels(lvl_short, fontsize=8, color='#8b949e')
        ax.set_ylabel('Rate (%)', fontsize=8, color='#8b949e')
        ax.set_title(label, color='#e6edf3', fontsize=9, fontweight='bold', pad=6)
        ax.tick_params(colors='#8b949e', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.set_ylim(0, max(lvl_rates) * 1.45 if lvl_rates else 25)
    plt.suptitle('Weighted Outcome Rates by Facility Level (n=27,288)', color='#e6edf3', fontsize=11, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Programme framework alignment
    st.markdown('<p class="section-hdr">Programme Reporting Framework Alignment</p>',
                unsafe_allow_html=True)
    st.markdown(
        "SmartDaaS signals map to both PEPFAR MER and Global Fund performance indicators. "
        "The table below shows how SmartDaaS outputs complement existing reporting frameworks."
    )

    tab_mer, tab_gf = st.tabs(["PEPFAR MER Indicators", "Global Fund Indicators"])

    with tab_mer:
        mer_data = {
            'MER Indicator': ['TX_CURR', 'TX_PVLS', 'TX_RTT', 'TX_ML', 'TX_NEW'],
            'Description': [
                'Currently on ART',
                'Viral load suppression',
                'Return to treatment',
                'Interruptions in treatment',
                'New ART initiations'
            ],
            'SmartDaaS Signal': [
                'Base cohort for risk scoring',
                'Low CD4/high risk → proxy for VL failure risk',
                'High risk patients flagged for re-engagement',
                'ART interruption predictor (top SHAP feature)',
                'Delayed ART (>90 days) flagged in facility analysis'
            ],
            'Tier Required': ['Core', 'Standard', 'Standard', 'Enhanced', 'Enhanced']
        }
        st.dataframe(pd.DataFrame(mer_data), use_container_width=True)
        st.caption(
            "PEPFAR MER alignment is based on Nigerian discovery cohort analysis. "
            "Indicator thresholds may differ across country operating plans."
        )

    with tab_gf:
        gf_data = {
            'Global Fund Indicator': [
                'ART Coverage Rate',
                'Viral Suppression Rate',
                'Retention on ART (12 months)',
                'Lost to Follow-Up Rate',
                'TB/HIV Co-management'
            ],
            'Description': [
                'Proportion of PLHIV on ART',
                '% of patients on ART with VL <1000 copies/ml',
                '% retained on ART at 12 months',
                '% of patients lost to follow-up',
                '% of TB patients receiving ART'
            ],
            'SmartDaaS Signal': [
                'Cohort size and coverage characterisation (Core tier)',
                'High risk patients → proxy for VL non-suppression risk',
                'Prior interruption flag → retention risk identification',
                'HIGH tier patients flagged before dropout occurs',
                'TB positive flag in risk scoring and SHAP explainability'
            ],
            'Tier Required': ['Core', 'Standard', 'Enhanced', 'Enhanced', 'Standard']
        }
        st.dataframe(pd.DataFrame(gf_data), use_container_width=True)
        st.caption(
            "Global Fund indicator alignment reflects conceptual mapping only. "
            "Formal alignment requires country-level grant performance framework review."
        )


# ═════════════════════════════════════════════════════════════
# PAGE 5 — ECONOMIC CALCULATOR
# ═════════════════════════════════════════════════════════════
elif page == "💰 Economic Calculator":
    st.markdown("""
### Programme Economic Impact Calculator

Estimates excess poor outcomes and avoidable costs attributable to sub-optimal
facility-level care. Based on Paper 2 findings (27,288 patients, Nigerian HIV programme).
""")

    st.markdown("""<div class="info-box">
📌 <strong>Basis:</strong> Primary health centre patients had 1.95× adjusted odds of poor outcome
vs tertiary hospitals (OR 1.95, 95% CI 1.45–2.61, p&lt;0.001). Unit cost: USD 880/patient/year
for comprehensive ART in PEPFAR programmes (Menzies et al., <em>AIDS</em> 2011); Nigeria facility
mean USD 231 (Bautista-Arredondo et al., <em>PLOS One</em> 2018). All estimates are scenario
projections — not guaranteed savings.
</div>""", unsafe_allow_html=True)

    # ── SCENARIO PRESETS ──────────────────────────────────
    st.markdown('<p class="section-hdr">Scenario Presets</p>', unsafe_allow_html=True)
    st.info(
        "Select a preset to populate parameters, or enter your own values below. "
        "All estimates are indicative only — based on Nigerian discovery cohort findings "
        "and published unit cost references. Adjust the cost parameter to reflect your "
        "country context using WHO regional estimates or your programme's actual costs."
    )

    # WHO regional cost reference
    WHO_REGIONAL_COSTS = {
        'West Africa (illustrative estimate)': 290,
        'East Africa (illustrative estimate)': 310,
        'Southern Africa (illustrative estimate)': 480,
        'Nigeria — PEPFAR comprehensive ART (Menzies 2011)': 880,
        'Nigeria — facility mean (Bautista-Arredondo 2018)': 231,
    }
    with st.expander("📊 Illustrative Regional Programme Cost Reference Estimates", expanded=False):
        cost_ref = pd.DataFrame([
            {'Region / Source': k, 'Unit Cost (USD/patient/year)': v}
            for k, v in WHO_REGIONAL_COSTS.items()
        ])
        st.dataframe(cost_ref, use_container_width=True)
        st.caption(
            "Values shown are indicative estimates intended for contextual planning and may "
            "vary by country, year, funding structure, and programme implementation setting. "
            "Sources: Menzies et al. AIDS 2011; Bautista-Arredondo et al. PLOS One 2018. "
            "Verify against your programme's actual cost data before use."
        )

    c1, c2, c3, c4 = st.columns(4)
    preset = None
    with c1:
        if st.button("🇳🇬 Nigeria Historical Baseline", use_container_width=True):
            preset = "nigeria"
    with c2:
        if st.button("📉 Conservative Scenario", use_container_width=True):
            preset = "conservative"
    with c3:
        if st.button("📈 Optimistic Intervention", use_container_width=True):
            preset = "optimistic"
    with c4:
        if st.button("🌍 Custom Programme", use_container_width=True):
            preset = "custom"

    if preset == "nigeria":
        default_total=27288; default_primary=2; default_secondary=66
        default_cost=880; default_tertiary=10
        st.markdown("""<div class="success-box">✓ <strong>Nigeria Historical Baseline:</strong>
        Parameters from Paper 2 (Nigerian discovery cohort). Primary HC poor outcome
        15.4% vs 10.2% tertiary. Cost: Menzies et al. 2011 PEPFAR comprehensive ART.
        </div>""", unsafe_allow_html=True)
    elif preset == "conservative":
        default_total=5000; default_primary=5; default_secondary=60
        default_cost=231; default_tertiary=10
        st.markdown("""<div class="info-box">📉 <strong>Conservative:</strong>
        Nigeria facility-level mean USD 231 (Bautista-Arredondo et al. 2018).
        Smaller programme, lower primary HC share.</div>""", unsafe_allow_html=True)
    elif preset == "optimistic":
        default_total=27288; default_primary=10; default_secondary=66
        default_cost=880; default_tertiary=8
        st.markdown("""<div class="info-box">📈 <strong>Optimistic:</strong>
        Higher primary HC representation (10%) with DSD reducing tertiary baseline to 8%.
        Best-case if targeted QI investment succeeds.</div>""", unsafe_allow_html=True)
    elif preset == "custom":
        default_total=10000; default_primary=5; default_secondary=65
        default_cost=310; default_tertiary=10
        st.markdown("""<div class="info-box">🌍 <strong>Custom Programme:</strong>
        Adjust all parameters for your programme context. Use the Illustrative
        Regional Cost Reference above as a starting point, or enter your programme's actual
        cost per patient per year.</div>""", unsafe_allow_html=True)
    else:
        default_total=27288; default_primary=2; default_secondary=66
        default_cost=880; default_tertiary=10

    st.markdown('<p class="section-hdr">Programme Parameters</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        n_total = st.number_input("Total ART patients in your programme", min_value=100, max_value=500000, value=default_total, step=500)
        pct_primary = st.slider("% at primary health centres", min_value=0, max_value=60, value=default_primary) / 100
        pct_secondary = st.slider("% at secondary health facilities", min_value=0, max_value=90, value=default_secondary) / 100
    with c2:
        cost_per_outcome = st.number_input("Annual cost per patient (USD)", min_value=100, max_value=5000, value=default_cost, step=50,
                                            help="Use 880 (Menzies 2011, PEPFAR comprehensive ART) or 231 (Nigeria facility mean, Bautista-Arredondo 2018)")
        tertiary_rate = st.slider("Baseline tertiary hospital poor outcome rate (%)", min_value=1, max_value=20, value=default_tertiary) / 100

    pct_tertiary = 1 - pct_primary - pct_secondary
    if pct_tertiary < 0:
        st.error("Primary + Secondary % exceeds 100%. Adjust sliders.")
        st.stop()

    n_primary = int(n_total * pct_primary)
    n_secondary = int(n_total * pct_secondary)
    n_tertiary = int(n_total * pct_tertiary)
    obs_primary = int(n_primary * 0.154)
    obs_secondary = int(n_secondary * 0.123)
    exp_primary = int(n_primary * tertiary_rate)
    exp_secondary = int(n_secondary * tertiary_rate)
    excess_primary = max(0, obs_primary - exp_primary)
    excess_secondary = max(0, obs_secondary - exp_secondary)
    total_excess = excess_primary + excess_secondary
    total_cost = total_excess * cost_per_outcome

    st.markdown('<p class="section-hdr">Estimated Impact</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="metric-val">{n_primary:,}</div><div class="metric-lbl">Patients at Primary HCs</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="metric-val">{total_excess:,}</div><div class="metric-lbl">Excess Poor Outcomes</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="metric-val">${total_cost/1000:.0f}K</div><div class="metric-lbl">Avoidable Cost (USD)</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="metric-val">${total_cost/n_total:.0f}</div><div class="metric-lbl">Cost per ART Patient</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    breakdown = pd.DataFrame({
        'Facility level': ['Primary HC', 'Secondary HF', 'Tertiary (baseline)', 'Total'],
        'Patients': [f'{n_primary:,}', f'{n_secondary:,}', f'{n_tertiary:,}', f'{n_total:,}'],
        'Observed poor outcomes': [f'{obs_primary:,}', f'{obs_secondary:,}', f'{int(n_tertiary*tertiary_rate):,}', f'{obs_primary+obs_secondary+int(n_tertiary*tertiary_rate):,}'],
        'Expected (if tertiary rate)': [f'{exp_primary:,}', f'{exp_secondary:,}', f'{int(n_tertiary*tertiary_rate):,}', f'{exp_primary+exp_secondary+int(n_tertiary*tertiary_rate):,}'],
        'Excess outcomes': [f'{excess_primary:,}', f'{excess_secondary:,}', '0', f'{total_excess:,}'],
        'Avoidable cost (USD)': [f'${excess_primary*cost_per_outcome:,.0f}', f'${excess_secondary*cost_per_outcome:,.0f}', '$0', f'${total_cost:,.0f}'],
    })
    st.dataframe(breakdown, use_container_width=True)

    st.markdown("""<div class="warn-box">
⚠️ <strong>Important limitations:</strong> Outcome rate estimates (15.4% primary HC, 10.2% tertiary)
are derived from the Nigerian national HIV programme discovery cohort (2006–2018) and have not been
externally validated in other country contexts. These figures may not reflect your programme's
actual outcome rates. All calculations are scenario projections for planning purposes only.
Do not use for budget commitments or donor reporting without local validation.
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE 6 — MODEL INFO
# ═════════════════════════════════════════════════════════════
elif page == "📘 Model Info":
    st.markdown("""
### Model Architecture and Validation

Full technical documentation of the SmartDaaS prediction model.
""")

    st.markdown('<p class="section-hdr">Model Performance</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, "0.772", "Temporal Validation AUC"),
        (c2, "87.3%", "Sensitivity"),
        (c3, "95.7%", "Specificity"),
        (c4, "27,288", "Training Records"),
    ]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Architecture</p>', unsafe_allow_html=True)
    st.markdown("""
- **Model type:** Random Forest Classifier
- **Pipeline:** StandardScaler → RandomForestClassifier
- **Training data:** 27,288 HIV-positive patients on ART (Nigerian national HIV programme, Quality of Care dataset — discovery cohort)
- **Companion analysis:** 165,444 CEPHIA specimens (HIV recency analysis — separate study, not used for ART outcome model training)
- **Features:** 15 clinical variables (see column guide on 📁 Sample Data page)
- **Cross-validation:** 5-fold stratified CV — AUC 0.963
- **Note:** Cross-validation AUC reflects internal model testing performance. Temporal validation AUC (0.772) reflects performance on future unseen patient records and should be treated as the primary operational estimate.
- **Temporal validation:** Held-out future patients (post-2015) — AUC 0.772
- **Explainability:** SHAP TreeExplainer with per-patient waterfall charts
- **Fairness:** Subgroup analysis across sex, age, WHO stage reported in Paper 1 (doi.org/10.64898/2026.05.15.26353325)
""")

    st.markdown('<p class="section-hdr">Important Limitations</p>', unsafe_allow_html=True)
    st.markdown("""
1. **Temporal validation AUC of 0.772** — performance degrades on future data, as expected for any ML model. This is the honest real-world estimate.
2. **Training data source** — Nigerian national HIV programme dataset (Quality of Care, 27,288 patients). Performance on data from different countries, health systems, or time periods may vary. Local validation is required before deployment.
3. **Not validated for clinical decision-making** — local validation is required before deployment in real-world programme environments. Prospective validation strengthens operational confidence.
4. **15 features only** — the model does not capture socioeconomic factors, geographic remoteness, or drug supply chain quality, which are known outcome drivers.
5. **Binary outcome** — the model predicts composite poor outcome (non-adherence OR interruption OR mortality). It does not distinguish between these outcomes.
""")

    st.markdown('<p class="section-hdr">Citation</p>', unsafe_allow_html=True)
    st.markdown("""
If you use SmartDaaS in research or programme evaluation, please cite:

> Chinthala LK. *Real-World Validation of Machine Learning Models for HIV Treatment
Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis of 192,732 Clinical Records.* Under review at Scientific Reports, 2026.
> 📄 [doi.org/10.64898/2026.05.15.26353325](https://doi.org/10.64898/2026.05.15.26353325)

> Chinthala LK. *Facility-Level Structural Drivers of HIV Treatment Outcomes: A Multi-Level
Analysis of 27,288 Patients from a Nigerian HIV Programme and Implications for PEPFAR
and Global Fund Programming.* Submitted to BMJ Global Health, 2026.
> 📄 [doi.org/10.64898/2026.05.15.26353326](https://doi.org/10.64898/2026.05.15.26353326)

**Code repository:** github.com/Kchinthala15/smartdaas-hiv-validation
""")


# ═════════════════════════════════════════════════════════════
# OUTREACH OPTIMISER  (capacity-constrained weekly action plan)
# ═════════════════════════════════════════════════════════════
elif page == "🎯 Outreach Optimiser":
    render_outreach_optimiser(supabase)


# ═════════════════════════════════════════════════════════════
# INTERVENTION RECOMMENDATION ENGINE
# ═════════════════════════════════════════════════════════════
elif page == "🎯 Intervention Engine":
    st.markdown("""
### Recommended Clinical Actions

Not just "HIGH risk" — but **why**, and **exactly what to do next**.
Each recommendation is generated from the patient's specific clinical profile
and maps to standard HIV programme intervention protocols.
""")

    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    # ── Use uploaded/scored data if available, else fall back to demo ──
    _scored = st.session_state.get('df_scored', None)
    if _scored is not None and len(_scored) > 0:
        df_ie = _scored.copy()
        for f in FEATURES:
            if f not in df_ie.columns:
                df_ie[f] = 0.0
            else:
                df_ie[f] = pd.to_numeric(df_ie[f], errors='coerce').fillna(0.0)
        X_ie = df_ie[FEATURES].values.astype(float)
        probs_ie = model.predict_proba(X_ie)[:, 1]
        df_ie['risk_pct'] = (probs_ie * 100).round(1)
        if 'risk_label' not in df_ie.columns:
            df_ie['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                                   for p in probs_ie]
        if 'patient_id' not in df_ie.columns:
            df_ie['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_ie))]
        st.info(f"ℹ️ Showing recommendations for your uploaded cohort ({len(df_ie):,} patients).")
    else:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(200, len(X_DEMO)), replace=False)
        df_ie = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        df_ie['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_ie))]
        X_ie = df_ie[FEATURES].values.astype(float)
        probs_ie = model.predict_proba(X_ie)[:, 1]
        df_ie['risk_pct'] = (probs_ie * 100).round(1)
        df_ie['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                               for p in probs_ie]
        st.markdown("""<div class="info-box">
        🔬 <strong>Demo mode:</strong> Upload patient data on the Patient Risk page
        to see recommendations for your own cohort.
        </div>""", unsafe_allow_html=True)

    # ── CLINICAL REASONING ENGINE ─────────────────────────
    def generate_recommendation(row):
        """
        Generate specific clinical recommendations based on patient profile.
        Returns (primary_recommendation, reasoning, urgency_days, protocol)
        """
        risk = row['risk_pct']
        reasons = []
        recs = []
        protocols = []

        # Analyse each clinical driver
        if row['had_interruption'] > 0.5:
            reasons.append("prior ART interruption documented")
            recs.append("Enhanced adherence counselling — interruption history protocol")
            protocols.append("ART Re-engagement Protocol")

        if row['stage_start_num'] >= 3:
            reasons.append(f"advanced WHO clinical stage ({row['stage_start_num']:.0f})")
            recs.append("Urgent clinical review — advanced disease staging")
            protocols.append("Advanced Disease Management")

        if row['Cd4AtStart'] < 200:
            reasons.append(f"severe immunosuppression at ART start (CD4={row['Cd4AtStart']:.0f})")
            recs.append("Opportunistic infection screening within 7 days")
            protocols.append("OI Prophylaxis Assessment")

        if row['CD4_improvement'] < 0:
            reasons.append(f"declining CD4 trend ({row['CD4_improvement']:.0f} cells/µL change)")
            recs.append("Viral load test within 14 days — possible treatment failure")
            protocols.append("Virological Failure Assessment")
        elif row['MostRecentCd4Count'] < 200:
            reasons.append(f"current CD4 critically low ({row['MostRecentCd4Count']:.0f} cells/µL)")
            recs.append("Viral load follow-up within 14 days + OI prophylaxis review")
            protocols.append("Immunological Monitoring")

        if row['opp_infection'] > 0.5:
            reasons.append("active opportunistic infection")
            recs.append("Immediate clinical review — OI management")
            protocols.append("Opportunistic Infection Management")

        if row['tb_positive'] > 0.5:
            reasons.append("TB co-infection")
            recs.append("TB-HIV co-treatment coordination within 48 hours")
            protocols.append("TB-HIV Co-treatment Protocol")

        if row['side_effects'] > 0.5:
            reasons.append("reported side effects")
            recs.append("Regimen tolerability review — consider switch assessment")
            protocols.append("Regimen Switch Assessment")

        if row['stage_worsened'] > 0.5:
            reasons.append("clinical stage worsening documented")
            recs.append("Urgent clinical review — treatment efficacy assessment")
            protocols.append("Treatment Failure Assessment")

        if row['days_to_ART'] > 90:
            reasons.append(f"delayed ART initiation ({row['days_to_ART']:.0f} days diagnosis-to-ART)")
            recs.append("Enhanced retention support — delayed initiators at higher dropout risk")
            protocols.append("Retention Support Protocol")

        if row['weight_change'] < -3:
            reasons.append(f"significant weight loss ({row['weight_change']:.1f} kg)")
            recs.append("Nutritional assessment + clinical review within 2 weeks")
            protocols.append("Nutritional Support Assessment")

        if row['sex_female'] < 0.5 and risk >= 0.7:
            reasons.append("male sex (lower adherence rates in this cohort)")
            recs.append("Male-friendly service delivery — flexible hours / community dispensing")
            protocols.append("Differentiated Service Delivery")

        # Default if no specific drivers
        if not reasons:
            if risk >= 0.7:
                reasons = ["multiple moderate risk factors in combination"]
                recs = ["Standard adherence counselling + attendance monitoring"]
                protocols = ["Standard Adherence Protocol"]
            else:
                reasons = ["profile within acceptable range"]
                recs = ["Standard care pathway — routine follow-up"]
                protocols = ["Standard Care"]

        # Urgency
        if risk >= 90:
            urgency = "Contact within 24 hours"
            urgency_color = "🔴"
        elif risk >= 80:
            urgency = "Contact within 48 hours"
            urgency_color = "🟠"
        elif risk >= 70:
            urgency = "Contact within 1 week"
            urgency_color = "🟡"
        elif risk >= 40:
            urgency = "Schedule within 2 weeks"
            urgency_color = "🟡"
        else:
            urgency = "Routine follow-up"
            urgency_color = "🟢"

        return {
            'primary_rec': recs[0] if recs else "Standard care",
            'all_recs': recs,
            'reasons': reasons,
            'protocols': list(set(protocols)),
            'urgency': f"{urgency_color} {urgency}",
            'n_drivers': len(reasons)
        }

    # ── FILTER + SELECT ───────────────────────────────────
    st.markdown('<p class="section-hdr">Patient Selection</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        tier_filter = st.selectbox("Risk tier:", ["HIGH + MEDIUM", "HIGH only", "All patients"])
    with c2:
        sort_opt = st.selectbox("Sort by:", ["Highest risk", "Most clinical drivers", "Patient ID"])
    with c3:
        show_n = st.slider("Show top N:", 5, 50, 15)

    df_show = df_ie.copy()
    if tier_filter == "HIGH only":
        df_show = df_show[df_show['risk_label'] == 'HIGH']
    elif tier_filter == "HIGH + MEDIUM":
        df_show = df_show[df_show['risk_label'].isin(['HIGH', 'MEDIUM'])]

    # Generate recommendations
    with st.spinner("Generating clinical recommendations..."):
        recs_list = [generate_recommendation(row) for _, row in df_show.iterrows()]
        df_show = df_show.copy()
        df_show['n_drivers'] = [r['n_drivers'] for r in recs_list]
        df_show['urgency'] = [r['urgency'] for r in recs_list]
        df_show['primary_rec'] = [r['primary_rec'] for r in recs_list]

    if sort_opt == "Highest risk":
        df_show = df_show.sort_values('risk_pct', ascending=False)
    elif sort_opt == "Most clinical drivers":
        df_show = df_show.sort_values('n_drivers', ascending=False)
    df_show = df_show.head(show_n).reset_index(drop=True)
    recs_list = [generate_recommendation(row) for _, row in df_show.iterrows()]

    # ── SUMMARY ───────────────────────────────────────────
    st.markdown('<p class="section-hdr">Cohort Summary</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="risk-high"><div class="risk-number">{(df_ie["risk_label"]=="HIGH").sum()}</div><div class="risk-label">HIGH Risk</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="risk-medium"><div class="risk-number">{(df_ie["risk_label"]=="MEDIUM").sum()}</div><div class="risk-label">MEDIUM Risk</div></div>', unsafe_allow_html=True)
    with c3:
        multi_driver = sum(1 for r in [generate_recommendation(row) for _, row in df_ie[df_ie['risk_label']=='HIGH'].iterrows()] if r['n_drivers'] >= 2)
        st.markdown(f'<div class="metric-box"><div class="metric-val">{multi_driver}</div><div class="metric-lbl">Multi-Driver HIGH</div></div>', unsafe_allow_html=True)
    with c4:
        tb_hiv = int((df_ie['tb_positive'] > 0.5).sum())
        st.markdown(f'<div class="metric-box"><div class="metric-val">{tb_hiv}</div><div class="metric-lbl">TB Co-infection</div></div>', unsafe_allow_html=True)

    # ── RECOMMENDATION CARDS ──────────────────────────────
    st.markdown('<p class="section-hdr">Recommended Actions</p>', unsafe_allow_html=True)

    for i, (_, row) in enumerate(df_show.iterrows()):
        rec = recs_list[i]
        risk_color = "#f85149" if row['risk_label'] == 'HIGH' else "#e3b341" if row['risk_label'] == 'MEDIUM' else "#3fb950"
        border_color = "f85149" if row['risk_label'] == 'HIGH' else "e3b341"

        with st.expander(f"{rec['urgency']}  ·  {row['patient_id']}  ·  Risk: {row['risk_pct']:.1f}%  ·  {rec['primary_rec'][:60]}...", expanded=(i < 3)):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"""<div style="background:#161b22;border:1px solid #{border_color};
                    border-radius:8px;padding:1rem;text-align:center">
                    <div style="font-size:2rem;font-weight:700;color:{risk_color};
                    font-family:'IBM Plex Mono',monospace">{row['risk_pct']:.1f}%</div>
                    <div style="font-size:0.75rem;color:#adbac7;text-transform:uppercase;
                    letter-spacing:1px">{row['risk_label']} RISK</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("**Clinical Profile:**")
                profile_items = [
                    ("Age", f"{row['Age']:.0f} years"),
                    ("CD4 at Start", f"{row['Cd4AtStart']:.0f} cells/µL"),
                    ("CD4 Change", f"{row['CD4_improvement']:+.0f} cells/µL"),
                    ("WHO Stage", f"Stage {row['stage_start_num']:.0f}"),
                    ("ART Delay", f"{row['days_to_ART']:.0f} days"),
                    ("Sex", "Female" if row['sex_female'] > 0.5 else "Male"),
                ]
                for label, val in profile_items:
                    st.markdown(f"- {label}: **{val}**")

            with c2:
                st.markdown("**Clinical Reasoning:**")
                for reason in rec['reasons']:
                    st.markdown(f"⚠️ {reason.capitalize()}")

                st.markdown("**Recommended Actions:**")
                for j, action in enumerate(rec['all_recs'], 1):
                    st.markdown(f"{j}. {action}")

                st.markdown("**Applicable Protocols:**")
                for protocol in rec['protocols']:
                    st.markdown(f"📋 {protocol}")

                st.markdown(f"**Urgency:** {rec['urgency']}")

    # ── EXPORT ────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Export Recommendations</p>', unsafe_allow_html=True)
    export_rows = []
    for i, (_, row) in enumerate(df_show.iterrows()):
        rec = recs_list[i]
        export_rows.append({
            'patient_id': row['patient_id'],
            'risk_pct': row['risk_pct'],
            'risk_label': row['risk_label'],
            'urgency': rec['urgency'],
            'n_clinical_drivers': rec['n_drivers'],
            'clinical_reasons': "; ".join(rec['reasons']),
            'primary_recommendation': rec['primary_rec'],
            'all_recommendations': " | ".join(rec['all_recs']),
            'protocols': "; ".join(rec['protocols']),
            'age': row['Age'],
            'cd4_at_start': row['Cd4AtStart'],
            'who_stage': row['stage_start_num'],
            'prior_interruption': 'Yes' if row['had_interruption'] > 0.5 else 'No',
            'tb_positive': 'Yes' if row['tb_positive'] > 0.5 else 'No',
        })

    export_df = pd.DataFrame(export_rows)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"📥 Download All Recommendations ({len(export_df)})",
            data=export_df.to_csv(index=False).encode(),
            file_name="smartdaas_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        urgent_df = export_df[export_df['urgency'].str.contains('24 hours|48 hours')]
        st.download_button(
            f"🚨 Download Urgent Only ({len(urgent_df)} patients)",
            data=urgent_df.to_csv(index=False).encode(),
            file_name="smartdaas_urgent_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ═════════════════════════════════════════════════════════════
# COHORT INTELLIGENCE DASHBOARD
# ═════════════════════════════════════════════════════════════
elif page == "👥 Cohort Intelligence":
    st.markdown("""
### Cohort Intelligence Dashboard

**Programme-level view — which subgroups are struggling, which are stable.**

Understand your cohort's risk distribution across demographic and clinical dimensions.
Identify deteriorating subgroups before they become programme failures.
""")

    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    # ── Use uploaded/scored data if available, else fall back to demo ──
    _scored = st.session_state.get('df_scored', None)
    if _scored is not None and len(_scored) > 0:
        df_ci = _scored.copy()
        for f in FEATURES:
            if f not in df_ci.columns:
                df_ci[f] = 0.0
            else:
                df_ci[f] = pd.to_numeric(df_ci[f], errors='coerce').fillna(0.0)
        X_ci = df_ci[FEATURES].values.astype(float)
        probs_ci = model.predict_proba(X_ci)[:, 1]
        df_ci['risk_pct'] = (probs_ci * 100).round(1)
        if 'risk_label' not in df_ci.columns:
            df_ci['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                                   for p in probs_ci]
        if 'patient_id' not in df_ci.columns:
            df_ci['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_ci))]
        st.info(f"ℹ️ Showing cohort intelligence for your uploaded dataset ({len(df_ci):,} patients).")
    else:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(300, len(X_DEMO)), replace=False)
        df_ci = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        df_ci['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_ci))]
        X_ci = df_ci[FEATURES].values.astype(float)
        probs_ci = model.predict_proba(X_ci)[:, 1]
        df_ci['risk_pct'] = (probs_ci * 100).round(1)
        df_ci['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                               for p in probs_ci]
        st.markdown("""<div class="info-box">
        🔬 <strong>Demo mode:</strong> Upload patient data on the Patient Risk page
        to see cohort intelligence for your own programme.
        </div>""", unsafe_allow_html=True)

    df_ci['sex_label'] = df_ci['sex_female'].apply(lambda x: 'Female' if x > 0.5 else 'Male')
    df_ci['age_group'] = pd.cut(df_ci['Age'], bins=[0, 25, 35, 45, 55, 100],
                                 labels=['<25', '25-34', '35-44', '45-54', '55+'])
    df_ci['who_stage_label'] = df_ci['stage_start_num'].apply(
        lambda x: f"Stage {int(x)}" if not pd.isna(x) else "Unknown")
    df_ci['cd4_group'] = pd.cut(df_ci['Cd4AtStart'],
                                 bins=[-1, 99, 199, 349, 499, 2000],
                                 labels=['<100', '100-199', '200-349', '350-499', '500+'])

    # ── PROGRAMME OVERVIEW ────────────────────────────────
    st.markdown('<p class="section-hdr">Programme Risk Overview</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    n_high = (df_ci['risk_label'] == 'HIGH').sum()
    n_med = (df_ci['risk_label'] == 'MEDIUM').sum()
    n_low = (df_ci['risk_label'] == 'LOW').sum()
    avg_risk = df_ci['risk_pct'].mean()
    pct_interruption = (df_ci['had_interruption'] > 0.5).mean() * 100

    with c1: st.markdown(f'<div class="risk-high"><div class="risk-number">{n_high}</div><div class="risk-label">HIGH Risk</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="risk-medium"><div class="risk-number">{n_med}</div><div class="risk-label">MEDIUM Risk</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="risk-low"><div class="risk-number">{n_low}</div><div class="risk-label">LOW Risk</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_risk:.1f}%</div><div class="metric-lbl">Avg Risk Score</div></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="metric-box"><div class="metric-val">{pct_interruption:.1f}%</div><div class="metric-lbl">Prior Interruption</div></div>', unsafe_allow_html=True)

    # ── SUBGROUP ANALYSIS ─────────────────────────────────
    st.markdown('<p class="section-hdr">Risk by Subgroup — Who Is Most Vulnerable?</p>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor='#161b22')
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    subgroups = [
        (axes[0, 0], 'sex_label', 'Risk by Sex', ['Female', 'Male'], ['#CC79A7', '#56B4E9']),
        (axes[0, 1], 'age_group', 'Risk by Age Group', ['<25', '25-34', '35-44', '45-54', '55+'], ['#21d4fd', '#0ea5c9', '#0891b2', '#0e7490', '#155e75']),
        (axes[1, 0], 'who_stage_label', 'Risk by WHO Clinical Stage', ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'], ['#3fb950', '#e3b341', '#f97316', '#f85149']),
        (axes[1, 1], 'cd4_group', 'Risk by CD4 at ART Start', ['<100', '100-199', '200-349', '350-499', '500+'], ['#f85149', '#f97316', '#e3b341', '#3fb950', '#21d4fd']),
    ]

    for ax, col, title, order, colors in subgroups:
        ax.set_facecolor('#161b22')
        grp = df_ci.groupby(col)['risk_pct'].mean().reindex(order).dropna()
        bars = ax.bar(range(len(grp)), grp.values, color=colors[:len(grp)],
                      width=0.6, edgecolor='#0d1117')
        for bar, v in zip(bars, grp.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8.5,
                    color='#e6edf3', fontweight='bold')
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp.index, fontsize=8, color='#cdd9e5', rotation=15)
        ax.set_ylabel('Avg Risk Score (%)', fontsize=8, color='#adbac7')
        ax.set_title(title, color='#e6edf3', fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors='#adbac7', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#444c56')
        ax.set_ylim(0, max(grp.values) * 1.35 if len(grp) > 0 else 100)

    plt.suptitle('Average Risk Score by Patient Subgroup', color='#e6edf3',
                 fontsize=12, fontweight='bold', y=0.98)
    st.pyplot(fig)
    plt.close()

    # ── HIGH RISK BREAKDOWN ───────────────────────────────
    st.markdown('<p class="section-hdr">HIGH Risk Patient Profile — What Are They Experiencing?</p>',
                unsafe_allow_html=True)

    df_high = df_ci[df_ci['risk_label'] == 'HIGH']
    df_low = df_ci[df_ci['risk_label'] == 'LOW']

    if len(df_high) > 0 and len(df_low) > 0:
        comparison_data = []
        binary_feats = ['had_interruption', 'opp_infection', 'side_effects',
                        'tb_positive', 'stage_worsened']
        feat_names = ['Prior Interruption', 'Opp. Infection', 'Side Effects',
                      'TB Positive', 'Stage Worsened']
        for feat, name in zip(binary_feats, feat_names):
            high_pct = (df_high[feat] > 0.5).mean() * 100
            low_pct = (df_low[feat] > 0.5).mean() * 100
            comparison_data.append({'Feature': name, 'HIGH Risk (%)': round(high_pct, 1),
                                     'LOW Risk (%)': round(low_pct, 1),
                                     'Difference': round(high_pct - low_pct, 1)})

        comp_df = pd.DataFrame(comparison_data).sort_values('Difference', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        x = range(len(comp_df))
        w = 0.35
        ax.bar([i - w/2 for i in x], comp_df['HIGH Risk (%)'], width=w,
               color='#f85149', label='HIGH Risk patients', edgecolor='#0d1117')
        ax.bar([i + w/2 for i in x], comp_df['LOW Risk (%)'], width=w,
               color='#3fb950', label='LOW Risk patients', edgecolor='#0d1117')
        for i in range(len(comp_df)):
            high_val = comp_df['HIGH Risk (%)'].iloc[i]
            low_val = comp_df['LOW Risk (%)'].iloc[i]
            ax.text(i - w/2, high_val + 0.5, f"{high_val:.0f}%", ha='center',
                    fontsize=8, color='#e6edf3', fontweight='bold')
            ax.text(i + w/2, low_val + 0.5, f"{low_val:.0f}%", ha='center',
                    fontsize=8, color='#e6edf3', fontweight='bold')
        ax.set_xticks(list(x))
        ax.set_xticklabels(comp_df['Feature'], fontsize=9, color='#cdd9e5')
        ax.set_ylabel('% of patients with factor', color='#adbac7', fontsize=9)
        ax.set_title('Clinical Risk Factors: HIGH vs LOW Risk Patients',
                     color='#e6edf3', fontsize=10, fontweight='bold', pad=10)
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#cdd9e5')
        ax.tick_params(colors='#adbac7', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#444c56')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(comp_df, use_container_width=True)

    # ── KEY INSIGHTS ──────────────────────────────────────
    st.markdown('<p class="section-hdr">Cohort Intelligence Insights</p>', unsafe_allow_html=True)

    insights = []
    # Sex gap
    female_risk = df_ci[df_ci['sex_label'] == 'Female']['risk_pct'].mean()
    male_risk = df_ci[df_ci['sex_label'] == 'Male']['risk_pct'].mean()
    if male_risk > female_risk + 3:
        insights.append(f"🔵 **Gender gap:** Male patients average {male_risk:.1f}% risk vs {female_risk:.1f}% for females (+{male_risk-female_risk:.1f}pp). Male-targeted interventions indicated.")

    # Advanced disease
    stage34_pct = (df_ci['stage_start_num'] >= 3).mean() * 100
    if stage34_pct > 20:
        insights.append(f"🔴 **Late presentation:** {stage34_pct:.1f}% of patients presented at WHO Stage 3–4. Early diagnosis programme strengthening needed.")

    # CD4 at start
    low_cd4_pct = (df_ci['Cd4AtStart'] < 200).mean() * 100
    if low_cd4_pct > 25:
        insights.append(f"🔴 **Severe immunosuppression:** {low_cd4_pct:.1f}% presented with CD4 <200 cells/µL. OI prophylaxis and viral load monitoring priorities.")

    # Interruption rate
    interruption_pct = (df_ci['had_interruption'] > 0.5).mean() * 100
    if interruption_pct > 15:
        insights.append(f"🟡 **High interruption history:** {interruption_pct:.1f}% have prior ART interruptions. Retention strengthening required.")

    if not insights:
        insights.append("✅ Cohort risk profile within expected ranges. Continue standard monitoring.")

    for insight in insights:
        st.markdown(insight)

    # Export
    st.markdown('<p class="section-hdr">Export Cohort Intelligence Report</p>',
                unsafe_allow_html=True)
    export_summary = df_ci.groupby(['risk_label', 'sex_label']).agg(
        count=('patient_id', 'count'),
        avg_risk=('risk_pct', 'mean'),
        avg_cd4=('Cd4AtStart', 'mean'),
        pct_interruption=('had_interruption', 'mean'),
    ).round(2).reset_index()
    export_summary.columns = ['Risk Tier', 'Sex', 'Count', 'Avg Risk (%)',
                               'Avg CD4 at Start', '% Prior Interruption']
    st.download_button(
        "📥 Download Cohort Intelligence Summary",
        data=export_summary.to_csv(index=False).encode(),
        file_name="smartdaas_cohort_intelligence.csv",
        mime="text/csv",
    )

    # ── IeDEA MUD Regional Aggregate Contextual Benchmarks ──
    st.markdown("---")
    render_iedea_benchmarks(df_upload=df_ci)


# ═════════════════════════════════════════════════════════════
# PDF EXECUTIVE REPORT GENERATOR
# ═════════════════════════════════════════════════════════════
elif page == "📄 Executive Report":
    st.markdown("""
### Executive Programme Intelligence Report

**One button. Professional PDF. Ready to hand to your programme director or implementing partner.**

Upload patient data (or use demo data), and SmartDaaS generates a complete
programme intelligence report — risk summary, facility findings, key recommendations,
and economic impact — formatted for executive and funder audiences.
""")


    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    # -- Data governance note (no checkboxes on this page) ---------
    # DUA is acknowledged once on Patient Risk page per session.
    # Executive Report simply shows a small notice and proceeds.
    st.markdown("""<div style='background:#0d1f17;border:1px solid #3fb95044;
        border-radius:6px;padding:8px 14px;font-size:0.82rem;color:#3fb950;margin-bottom:8px'>
        🔒 Patient data processed in-browser only · Not stored or transmitted externally · 
        Decision-support output — review by qualified programme staff required before operational use.
        </div>""", unsafe_allow_html=True)

    # -- Data source: prefer session state from Patient Risk page ----
    st.markdown('<p class="section-hdr">Report Data</p>', unsafe_allow_html=True)

    _scored = st.session_state.get('df_scored')
    _has_scored = (_scored is not None and
                   isinstance(_scored, pd.DataFrame) and
                   len(_scored) > 0 and
                   'risk_pct' in _scored.columns)

    if _has_scored:
        st.markdown(
            f"""<div style='background:#0d1f17;border:1px solid #3fb95044;
            border-radius:6px;padding:8px 14px;font-size:0.9rem;color:#3fb950;margin-bottom:8px'>
            ✅ Using dataset from Patient Risk page — {len(_scored):,} patients already scored.
            Navigate to Patient Risk to change dataset.
            </div>""", unsafe_allow_html=True
        )
        uploaded_rep = None
        use_demo_rep = False
    else:
        col_up, col_demo = st.columns([2, 1])
        with col_up:
            uploaded_rep = st.file_uploader(
                "Upload patient CSV for report", type=['csv'],
                help="Or score your data on the Patient Risk page first."
            )
        with col_demo:
            st.markdown("<br>", unsafe_allow_html=True)
            use_demo_rep = st.checkbox(
                "Use demo data instead",
                value=False,
                help="300 patients from the training set - no upload required"
            )
    st.markdown('<p class="section-hdr">Report Details</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        org_name = st.text_input("Organisation name", value="APIN Public Health Initiatives")
    with c2:
        programme_name = st.text_input("Programme / site name", value="Nigerian National HIV Programme")
    with c3:
        report_date = st.date_input("Report date", value=datetime.date.today())

    prepared_by = st.text_input("Prepared by", value="SmartDaaS Analytics · Lakshmi Kalyani Chinthala")

    # Cache uploaded file in session state so PDF generation rerender doesn't lose it
    if uploaded_rep is not None:
        st.session_state['rep_uploaded_bytes'] = uploaded_rep.read()
        st.session_state['rep_uploaded_name'] = uploaded_rep.name
    if use_demo_rep:
        st.session_state.pop('rep_uploaded_bytes', None)
        st.session_state.pop('rep_uploaded_name', None)

    has_real_upload = (
        not use_demo_rep and
        st.session_state.get('rep_uploaded_bytes') is not None
    )

    # Load data — priority: session state scored df > uploaded CSV > demo
    if _has_scored and not has_real_upload and not use_demo_rep:
        df_rep = _scored.copy()
        # Ensure all FEATURES columns present and numeric
        for f in FEATURES:
            if f not in df_rep.columns:
                df_rep[f] = 0.0
            else:
                df_rep[f] = pd.to_numeric(df_rep[f], errors='coerce').fillna(0.0)
        if 'patient_id' not in df_rep.columns:
            df_rep['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_rep))]
        # risk_pct and risk_label already computed on Patient Risk page
        X_rep = df_rep[FEATURES].values.astype(float)
        data_source = f"Patient Risk page dataset ({len(df_rep):,} patients)"
        rep_tier = st.session_state.get('upload_tier', 'ENHANCED')
        rep_pediatric = st.session_state.get('pediatric_indices', [])
    elif not has_real_upload:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(300, len(X_DEMO)), replace=False)
        df_rep = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        df_rep['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_rep))]
        data_source = "Demo dataset (300 patients from training set)"
        rep_tier = 'ENHANCED'
        rep_pediatric = []
    else:
        try:
            from io import BytesIO as _BIO
            df_raw = pd.read_csv(_BIO(st.session_state['rep_uploaded_bytes']))

            # ── PHIA / population-survey pre-processing ───────────
            df_raw, _rep_derivation_log = preprocess_phia_compatible(df_raw)
            st.session_state['_phia_derivation_log'] = _rep_derivation_log

            # ── Tier detection (same rules as Patient Risk page) ──
            df_rep, missing, mappings_applied = normalize_columns(df_raw)

            if mappings_applied:
                with st.expander(
                    f"ℹ️ Auto-mapped {len(mappings_applied)} column name(s)",
                    expanded=False
                ):
                    for orig, mapped in mappings_applied:
                        if mapped != '__art_inferred__':
                            st.markdown(f"- `{orig}` → `{mapped}`")

            art_confirmed, art_inferred, art_note = detect_art_status(df_raw)
            rep_tier, present, missing_core, std_present, enh_present, _ = \
                detect_tier(df_rep, art_confirmed, art_inferred)
            rep_pediatric = check_pediatric_patients(df_rep)

            st.markdown("---")
            can_proceed = render_tier_report(
                rep_tier, present, missing_core, std_present,
                enh_present, art_confirmed, art_inferred,
                art_note, rep_pediatric, df_rep
            )
            st.markdown("---")

            if not can_proceed:
                st.stop()

            # Core tier: no risk scores in report
            if rep_tier == 'CORE':
                st.info(
                    "**Core Tier upload:** The Executive Report will contain a "
                    "population summary only. Patient-level risk scores are not "
                    "generated for Core tier uploads. Add CD4, WHO stage, TB status, "
                    "and days to ART to unlock full report generation."
                )
                st.stop()

            # Derive engineered features + fill all gaps for ALL tiers
            df_rep, derived_feats_rep, defaulted_feats_rep = derive_engineered_features(df_rep)

            if rep_tier == 'STANDARD':
                st.warning(
                    "**Standard Tier upload:** Risk estimates in this report were "
                    "generated using partial feature availability. Prediction confidence "
                    "and stability may vary depending on which clinical variables are "
                    "present. Interpret all findings alongside clinical judgement and "
                    "local programme context."
                )

            # Data quality screening for uploaded report data
            available_feats = [f for f in FEATURES if f in df_rep.columns]
            dq_rep = run_data_quality_screening(df_rep, available_feats)
            render_data_quality_report(dq_rep, len(df_rep))

            # Validation metadata — surfaced in report for pilot partner audit
            _rep_val_meta = build_validation_metadata(
                df_raw=df_raw,
                df_mapped=df_rep,
                mappings_applied=mappings_applied,
                missing_features=missing,
                derivation_log=st.session_state.get('_phia_derivation_log', []),
                dq_results=dq_rep,
                tier=rep_tier,
            )
            st.session_state['validation_metadata'] = _rep_val_meta
            render_validation_metadata(_rep_val_meta)

            # Apply imputation from quality results
            for feat, info in dq_rep['missing'].items():
                if feat in df_rep.columns:
                    df_rep[feat] = df_rep[feat].fillna(info['impute_val'])

            # Fill any remaining nulls with neutral default
            for col in available_feats:
                if df_rep[col].isnull().any():
                    df_rep[col] = df_rep[col].fillna(df_rep[col].median())

            if 'patient_id' not in df_rep.columns:
                df_rep['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_rep))]

            data_source = (
                f"Uploaded dataset ({len(df_rep):,} patients — "
                f"{rep_tier} tier, Quality Grade: {dq_rep['grade']})"
            )

        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    # ── Run predictions ───────────────────────────────────
    # Safety net: ensure all features exist and are numeric
    for f in FEATURES:
        if f not in df_rep.columns:
            df_rep[f] = 0.0
        else:
            df_rep[f] = pd.to_numeric(df_rep[f], errors='coerce').fillna(0.0)
    X_rep = df_rep[FEATURES].values.astype(float)
    probs_rep = model.predict_proba(X_rep)[:, 1]
    df_rep['risk_pct'] = (probs_rep * 100).round(1)
    df_rep['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                             for p in probs_rep]

    n_total = len(df_rep)
    n_high = (df_rep['risk_label'] == 'HIGH').sum()
    n_med = (df_rep['risk_label'] == 'MEDIUM').sum()
    n_low = (df_rep['risk_label'] == 'LOW').sum()
    avg_risk = df_rep['risk_pct'].mean()
    pct_high = n_high / n_total * 100
    pct_interruption = (df_rep['had_interruption'] > 0.5).mean() * 100
    pct_tb = (df_rep['tb_positive'] > 0.5).mean() * 100
    pct_adv_disease = (df_rep['stage_start_num'] >= 3).mean() * 100
    pct_low_cd4 = (df_rep['Cd4AtStart'] < 200).mean() * 100
    # Consistent with Outreach Optimiser (23% assumed reduction)
    REDUCTION_RATE = 0.23
    # Conservative estimate using Menzies 2011 figure
    est_avoidable_cost = int(n_high * REDUCTION_RATE * COST_PER_POOR_OUTCOME)
    # Mid and upper estimates for methodology section
    est_avoidable_cost_mid   = int(n_high * REDUCTION_RATE * 3500)   # 2024 CPI-adjusted
    est_avoidable_cost_upper = int(n_high * REDUCTION_RATE * 5000)   # full re-engagement cost

    # Preview
    st.markdown('<p class="section-hdr">Report Preview</p>', unsafe_allow_html=True)
    st.markdown(f"""<div class="info-box">
    <strong>Report will include:</strong> Programme overview · Risk stratification summary ·
    Key clinical findings · Top 10 highest-risk patients · Facility intelligence (Paper 2) ·
    Economic impact estimate · Recommended actions · SmartDaaS methodology notes
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="metric-val">{n_total:,}</div><div class="metric-lbl">Total Patients</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="risk-high" style="padding:0.8rem"><div class="risk-number" style="font-size:1.6rem">{n_high}</div><div class="risk-label">HIGH Risk</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_risk:.1f}%</div><div class="metric-lbl">Avg Risk Score</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="metric-val">${est_avoidable_cost/1000:.0f}K</div><div class="metric-lbl">Est. Avoidable Cost</div></div>', unsafe_allow_html=True)

    # ── GENERATE PDF ──────────────────────────────────────
    st.markdown('<p class="section-hdr">Generate Report</p>', unsafe_allow_html=True)

    if st.button("📄 Generate Executive PDF Report", use_container_width=True, type="primary"):
        with st.spinner("Generating professional PDF report..."):
            try:
                from fpdf import FPDF

                # Sanitize all text going into PDF - replace chars Helvetica can't handle
                def _s(text):
                    if not isinstance(text, str):
                        text = str(text)
                    return (text
                        .replace('\u2014', '-')   # em dash
                        .replace('\u2013', '-')   # en dash
                        .replace('\u2192', '->')  # arrow
                        .replace('\u00b7', '.')   # middle dot
                        .replace('\u2019', "'")  # right single quote
                        .replace('\u2018', "'")  # left single quote
                        .replace('\u201c', '"')  # left double quote
                        .replace('\u201d', '"')  # right double quote
                        .replace('\u2026', '...')  # ellipsis
                        .replace('\u00b5', 'u')   # micro sign
                        .replace('\u00d7', 'x')   # multiplication sign
                        .replace('\u00b0', ' deg') # degree sign
                        .encode('latin-1', errors='replace').decode('latin-1')
                    )


                # ── COLOR PALETTE ────────────────────────────────────
                _BG        = (13, 17, 23)
                _CARD      = (30, 37, 48)
                _CYAN      = (0, 229, 255)
                _AMBER     = (255, 179, 0)
                _WHITE     = (255, 255, 255)
                _MUTED     = (176, 190, 197)
                _TEXT      = (226, 234, 243)
                _RED       = (255, 69, 58)
                _GREEN     = (48, 209, 88)
                _LOGO_DARK = (10, 22, 40)

                def _draw_bars(p, bx, by, bar_w, gap, scale=1.0):
                    bars = [
                        (0, 20, (0,136,170)), (1, 14, (0,136,170)),
                        (2,  8, _CYAN),       (3, 17, _CYAN),
                        (4, 11, (0,136,170)),
                    ]
                    for i, top_off, color in bars:
                        bar_h = 28*scale - top_off*scale
                        p.set_fill_color(*color)
                        p.rect(bx + i*(bar_w+gap), by + top_off*scale, bar_w, bar_h, 'F')

                def _smartdaas_text(p, x, y, size):
                    p.set_font('Helvetica', 'B', size)
                    sw = p.get_string_width('Smart')
                    p.set_text_color(*_WHITE)
                    p.set_xy(x, y); p.cell(sw, size*0.5, 'Smart')
                    p.set_text_color(*_CYAN)
                    p.set_xy(x+sw, y); p.cell(sw, size*0.5, 'DaaS')

                def _inner_header(p, page_title):
                    p.set_fill_color(*_LOGO_DARK)
                    p.rect(0, 0, 210, 16, 'F')
                    p.set_fill_color(*_CYAN)
                    p.rect(0, 16, 210, 0.8, 'F')
                    p.set_fill_color(*_BG)
                    p.rect(0, 16.8, 210, 297, 'F')
                    p.set_fill_color(*_CYAN)
                    p.rect(0, 0, 3, 297, 'F')
                    p.set_font('Helvetica', 'B', 11)
                    sw = p.get_string_width('Smart')
                    p.set_text_color(*_WHITE); p.set_xy(8, 4); p.cell(sw, 8, 'Smart')
                    p.set_text_color(*_CYAN);  p.set_xy(8+sw, 4); p.cell(sw, 8, 'DaaS')
                    p.set_font('Helvetica', 'B', 10)
                    p.set_text_color(*_MUTED)
                    p.set_xy(0, 4); p.cell(202, 8, page_title, align='R')

                def _section_title(p, title):
                    p.set_font('Helvetica', 'B', 14)
                    p.set_text_color(*_CYAN)
                    p.set_x(15); p.cell(180, 10, title, ln=True)
                    p.set_draw_color(*_CYAN)
                    p.set_line_width(0.5)
                    p.line(15, p.get_y(), 195, p.get_y())
                    p.ln(5)

                class SmartDaaSReport(FPDF):
                    def header(self): pass
                    def footer(self): pass

                pdf = SmartDaaSReport()

                # Patch pdf methods to auto-sanitize all text
                _orig_cell = pdf.cell
                _orig_multi = pdf.multi_cell
                def _safe_cell(*a, **kw):
                    a = tuple(_s(x) if isinstance(x, str) else x for x in a)
                    kw = {k: _s(v) if isinstance(v, str) else v for k, v in kw.items()}
                    return _orig_cell(*a, **kw)
                def _safe_multi(*a, **kw):
                    a = tuple(_s(x) if isinstance(x, str) else x for x in a)
                    kw = {k: _s(v) if isinstance(v, str) else v for k, v in kw.items()}
                    return _orig_multi(*a, **kw)
                pdf.cell = _safe_cell
                pdf.multi_cell = _safe_multi

                # ── COVER PAGE ───────────────────────────────────────
                pdf.set_auto_page_break(auto=False)
                pdf.add_page()
                pdf.set_fill_color(*_BG); pdf.rect(0, 0, 210, 297, 'F')
                pdf.set_fill_color(*_CYAN); pdf.rect(0, 0, 3, 297, 'F')

                # Header bar
                pdf.set_fill_color(*_LOGO_DARK); pdf.rect(0, 0, 210, 42, 'F')
                pdf.set_fill_color(*_CYAN); pdf.rect(0, 42, 210, 1.5, 'F')
                _draw_bars(pdf, 12, 7, 7, 2, scale=1.0)
                _smartdaas_text(pdf, 60, 10, 22)
                pdf.set_font('Helvetica', '', 9); pdf.set_text_color(0,171,187)
                pdf.set_xy(60, 25); pdf.cell(90, 8, 'HIV PROGRAMME INTELLIGENCE')
                pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_MUTED)
                pdf.set_xy(110, 12); pdf.cell(85, 7, 'Programme Intelligence', align='R')
                pdf.set_font('Helvetica', '', 10)
                pdf.set_xy(110, 21); pdf.cell(85, 8, _s(f'Executive Report  .  {report_date.strftime("%Y")}'), align='R')

                # Report title
                pdf.set_font('Helvetica', 'B', 22); pdf.set_text_color(*_WHITE)
                pdf.set_xy(15, 52); pdf.cell(180, 13, 'HIV Programme Intelligence Report')
                pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_MUTED)
                pdf.set_xy(15, 66); pdf.cell(180, 9, 'Predictive Risk  .  Facility Analytics  .  Economic Impact  .  Recommended Actions')
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.4)
                pdf.line(15, 75, 195, 75)

                # Org card
                pdf.set_fill_color(*_CARD); pdf.rect(15, 79, 180, 38, 'F')
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.3)
                pdf.line(15, 79, 15, 117)
                cover_details = [
                    ('Organisation', org_name),
                    ('Programme',    programme_name),
                    ('Report Date',  report_date.strftime('%d %B %Y')),
                    ('Prepared by',  prepared_by),
                ]
                for i, (lbl, val) in enumerate(cover_details):
                    pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_MUTED)
                    pdf.set_xy(22, 82+i*9); pdf.cell(42, 10, _s(lbl+':'))
                    pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(*_WHITE)
                    pdf.cell(1210, 8, _s(val))

                # Snapshot header
                pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_CYAN)
                pdf.set_xy(15, 122); pdf.cell(180, 8, 'PROGRAMME SNAPSHOT')
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.3)
                pdf.line(15, 128, 195, 128)

                # 4 metric cards
                snap_cards = [
                    (f'{n_total:,}',       'TOTAL PATIENTS', _AMBER),
                    (f'{n_high}',          'HIGH RISK',      _RED),
                    (f'{avg_risk:.1f}%',   'AVG RISK SCORE', _AMBER),
                    (f'${est_avoidable_cost//1000}K', 'AVOIDABLE COST', _CYAN),
                ]
                cw = 43
                for i, (val, lbl, vc) in enumerate(snap_cards):
                    cx = 15 + i*(cw+2); cy = 131
                    pdf.set_fill_color(*_CARD); pdf.rect(cx, cy, cw, 22, 'F')
                    pdf.set_fill_color(*vc);    pdf.rect(cx, cy, cw, 1.5, 'F')
                    pdf.set_font('Helvetica', 'B', 15); pdf.set_text_color(*vc)
                    pdf.set_xy(cx, cy+3); pdf.cell(cw, 9, val, align='C')
                    pdf.set_font('Helvetica', 'B', 9); pdf.set_text_color(*_WHITE)
                    pdf.set_xy(cx, cy+13); pdf.cell(cw, 8, lbl, align='C')

                # Risk bar
                bar_total = 180
                high_w = int(bar_total * pct_high/100)
                med_w  = int(bar_total * (n_med/n_total))
                low_w  = bar_total - high_w - med_w
                by_r = 157
                pdf.set_fill_color(*_RED);   pdf.rect(15, by_r, high_w, 9, 'F')
                pdf.set_fill_color(*_AMBER); pdf.rect(15+high_w, by_r, med_w, 9, 'F')
                pdf.set_fill_color(*_GREEN); pdf.rect(15+high_w+med_w, by_r, low_w, 9, 'F')
                pdf.set_font('Helvetica', 'B', 9); pdf.set_text_color(*_WHITE)
                # Always show all three labels — use coloured text below bar if segment too narrow
                pdf.set_xy(15, by_r+10)
                pdf.cell(high_w, 8, _s(f'{n_high} HIGH {pct_high:.1f}%') if high_w > 25 else '', align='C')
                pdf.cell(med_w,  8, _s(f'{n_med} MED {n_med/n_total*100:.1f}%') if med_w > 25 else '', align='C')
                pdf.cell(low_w,  8, _s(f'{n_low} LOW {n_low/n_total*100:.1f}%') if low_w > 25 else '', align='C')
                # For segments too narrow to label inline, show below in colour
                label_y = by_r + 19
                if high_w <= 25:
                    pdf.set_text_color(*_RED)
                    pdf.set_xy(15, label_y); pdf.cell(60, 7, _s(f'{n_high} HIGH {pct_high:.1f}%'))
                if med_w <= 25:
                    pdf.set_text_color(*_AMBER)
                    pdf.set_xy(80, label_y); pdf.cell(60, 7, _s(f'{n_med} MED {n_med/n_total*100:.1f}%'))
                if low_w <= 25:
                    pdf.set_text_color(*_GREEN)
                    pdf.set_xy(145, label_y); pdf.cell(50, 7, _s(f'{n_low} LOW {n_low/n_total*100:.1f}%'))

                # What's inside
                pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*_CYAN)
                pdf.set_xy(15, 180); pdf.cell(1100, 8, "WHAT'S INSIDE THIS REPORT")
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.4)
                pdf.line(15, 188, 195, 188)
                contents = [
                    ('01','Executive Summary',             'Risk stratification overview and key clinical findings'),
                    ('02','Facility Intelligence',         'Structural drivers of poor outcomes from 27,288 patients'),
                    ('03','Top 10 Highest Risk Patients',  'Patients requiring immediate contact and adherence support'),
                    ('04','Patient Risk Explanation',      'SHAP analysis — why the highest-risk patient was flagged'),
                    ('05','Recommended Programme Actions', 'Immediate, short-term and strategic interventions'),
                    ('06','Methodology & Limitations',     'Model validation, economic assumptions and data governance'),
                ]
                for i, (num, title, desc) in enumerate(contents):
                    y = 191 + i*12
                    pdf.set_fill_color(*_CYAN); pdf.rect(15, y+1, 11, 10, 'F')
                    pdf.set_font('Helvetica', 'B', 9); pdf.set_text_color(*_BG)
                    pdf.set_xy(15, y+2.5); pdf.cell(11, 9, num, align='C')
                    pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_WHITE)
                    pdf.set_xy(30, y+1); pdf.cell(165, 7, title)
                    pdf.set_font('Helvetica', '', 10); pdf.set_text_color(*_MUTED)
                    pdf.set_xy(30, y+7); pdf.cell(165, 6, _s(desc))

                # Disclaimer
                pdf.set_fill_color(35, 26, 0); pdf.rect(15, 270, 180, 20, 'F')
                pdf.set_draw_color(*_AMBER); pdf.set_line_width(0.5)
                pdf.line(15, 270, 15, 290)
                pdf.set_xy(21, 272); pdf.set_font('Helvetica', 'B', 11)
                pdf.set_text_color(*_AMBER); pdf.cell(30, 7, 'IMPORTANT')
                pdf.set_font('Helvetica', 'I', 10); pdf.set_text_color(*_TEXT)
                pdf.set_xy(21, 278)
                pdf.multi_cell(168, 7, _s('SmartDaaS v1.0 is a decision-support platform for HIV programme intelligence. Not intended to replace clinical judgment. All outputs require review by qualified programme teams before operational use. Local validation required before deployment.'))

                # ── EXECUTIVE SUMMARY ────────────────────────────────
                pdf.set_auto_page_break(auto=True, margin=18)
                pdf.add_page()
                _inner_header(pdf, 'EXECUTIVE SUMMARY')
                pdf.set_y(24); _section_title(pdf, 'EXECUTIVE SUMMARY')

                pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*_WHITE)
                pdf.set_x(15); pdf.cell(1100, 8, 'Programme Risk Overview', ln=True)
                pdf.ln(2)

                exec_metrics = [
                    ('Total Patients Analysed',              f'{n_total:,}',                          _WHITE),
                    ('HIGH Risk Patients (>=70%)',           f'{n_high} ({pct_high:.1f}%)',            _RED),
                    ('MEDIUM Risk Patients (40-69%)',        f'{n_med} ({n_med/n_total*100:.1f}%)',    _AMBER),
                    ('LOW Risk Patients (<40%)',             f'{n_low} ({n_low/n_total*100:.1f}%)',    _GREEN),
                    ('Average Risk Score',                   f'{avg_risk:.1f}%',                       _AMBER),
                    ('Patients with Prior ART Interruption', f'{pct_interruption:.1f}%',               _WHITE),
                    ('Patients with TB Co-infection',        f'{pct_tb:.1f}%',                         _WHITE),
                    ('Advanced Disease (WHO Stage 3-4)',     f'{pct_adv_disease:.1f}%',                _WHITE),
                    ('Severe Immunosuppression (CD4 <200)',  f'{pct_low_cd4:.1f}%',                    _WHITE),
                    ('Estimated Avoidable Cost (conservative)', f'USD {est_avoidable_cost:,}',     _CYAN),
                    ('Estimated Avoidable Cost (mid, 2024)',    f'USD {est_avoidable_cost_mid:,}',  _CYAN),
                    ('Estimated Avoidable Cost (upper)',        f'USD {est_avoidable_cost_upper:,}', _CYAN),
                ]
                for label, value, vc in exec_metrics:
                    pdf.set_fill_color(*_CARD)
                    pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_MUTED)
                    pdf.set_x(15); pdf.cell(125, 10, _s('  '+label), fill=True)
                    pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(*vc)
                    pdf.cell(55, 10, value, fill=True, ln=True)
                    pdf.ln(1)

                pdf.ln(4)
                pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*_WHITE)
                pdf.set_x(15); pdf.cell(1100, 8, 'Key Clinical Findings', ln=True)
                pdf.set_draw_color(*_GREEN); pdf.set_line_width(0.4)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(4)

                findings = [
                    f'{pct_high:.1f}% of patients are HIGH risk (>=70% predicted probability of poor ART outcome)',
                    f'{pct_interruption:.1f}% have documented prior ART interruptions - the strongest predictor in the SmartDaaS model',
                    f'{pct_adv_disease:.1f}% presented at WHO Stage 3 or 4, indicating late diagnosis and treatment initiation',
                    f'{pct_low_cd4:.1f}% had CD4 <200 cells/uL at ART start - severely immunosuppressed',
                ]
                if pct_tb > 5:
                    findings.append(f'{pct_tb:.1f}% are TB-HIV co-infected, requiring coordinated treatment protocols')
                for finding in findings:
                    pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                    pdf.set_x(15); pdf.cell(9, 7, chr(149))
                    pdf.multi_cell(163, 9, finding); pdf.ln(1)

                # ── FACILITY INTELLIGENCE ────────────────────────────
                pdf.add_page()
                _inner_header(pdf, 'FACILITY INTELLIGENCE')
                pdf.set_y(24); _section_title(pdf, 'FACILITY INTELLIGENCE')

                pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_MUTED)
                pdf.set_x(15)
                pdf.multi_cell(180, 8, _s('Based on Paper 2 analysis of 27,288 patients from the Nigerian national HIV programme (Chinthala 2026, submitted to BMJ Global Health).'))
                pdf.ln(4)

                # Forest plot
                findings_data = [
                    ('Primary HC vs Tertiary', 1.95, 1.45, 2.61, _RED),
                    ('NGO-Funded Facilities',  1.24, 1.10, 1.39, _AMBER),
                    ('Federal-Funded',         1.25, 1.06, 1.48, _AMBER),
                    ('Female Sex (protective)', 0.87, 0.79, 0.96, _GREEN),
                ]
                plot_y = pdf.get_y()
                scale_start = 77; scale_w = 100
                null_x = scale_start + scale_w * (1.0-0.7)/(2.8-0.7)
                pdf.set_font('Helvetica', '', 9); pdf.set_text_color(*_MUTED)
                for val, lbl in [(0.7,'0.7'),(1.0,'1.0'),(1.5,'1.5'),(2.0,'2.0'),(2.6,'2.6')]:
                    sx = scale_start + scale_w*(val-0.7)/(2.8-0.7)
                    pdf.set_xy(sx-4, plot_y-5); pdf.cell(10, 7, lbl, align='C')
                pdf.set_draw_color(*_MUTED); pdf.set_line_width(0.3)
                pdf.set_dash_pattern(dash=2, gap=2)
                pdf.line(null_x, plot_y, null_x, plot_y+len(findings_data)*18+4)
                pdf.set_dash_pattern(dash=0, gap=0)
                for i, (lbl, OR, lo, hi, color) in enumerate(findings_data):
                    fy = plot_y + 4 + i*18
                    pdf.set_fill_color(*_CARD); pdf.rect(15, fy, 180, 16, 'F')
                    pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_WHITE)
                    pdf.set_xy(17, fy+4); pdf.cell(510, 8, lbl)
                    pdf.set_text_color(*color)
                    pdf.set_xy(77, fy+4); pdf.cell(13, 10, f'{OR:.2f}')
                    lo_x = scale_start + scale_w*(lo-0.7)/(2.8-0.7)
                    hi_x = scale_start + scale_w*(hi-0.7)/(2.8-0.7)
                    or_x = scale_start + scale_w*(OR-0.7)/(2.8-0.7)
                    cy2 = fy+8
                    pdf.set_draw_color(*color); pdf.set_line_width(1.2)
                    pdf.line(lo_x, cy2, hi_x, cy2)
                    pdf.set_fill_color(*color)
                    pdf.rect(or_x-2, cy2-2, 4, 4, 'F')
                pdf.set_y(plot_y + len(findings_data)*18 + 10)
                pdf.set_font('Helvetica', 'I', 10); pdf.set_text_color(*_MUTED)
                pdf.set_x(15)
                pdf.cell(180, 8, _s('Square = OR estimate. Line = 95% CI. Dashed line = null (OR 1.0).'))
                pdf.ln(5)

                facility_findings = [
                    ('Primary HC vs Tertiary Hospital', 'OR 1.95 (95% CI 1.45-2.61)',
                     'Primary HCs have nearly double the odds of composite poor outcome after patient-level adjustment.',
                     'Structural quality improvement - staffing, drug supply, monitoring systems.'),
                    ('NGO-Funded Facilities', 'OR 1.24 (95% CI 1.10-1.39)',
                     'NGO-funded facilities show independently higher odds - may reflect higher case complexity.',
                     'Outcome-adjusted performance monitoring; investigate funding-to-quality translation.'),
                    ('Federal-Funded Facilities', 'OR 1.25 (95% CI 1.06-1.48)',
                     'Federal government funded facilities show similarly elevated risk.',
                     'Review programme management capacity and reporting burden at federal-funded sites.'),
                    ('Female Sex - Protective Effect', 'OR 0.87 (95% CI 0.79-0.96)',
                     'Female sex is independently protective overall, but advantage nearly disappears at primary HCs.',
                     'Male-targeted interventions at secondary/tertiary; structural improvements at primary HCs.'),
                ]
                for finding_title, stat, desc, action in facility_findings:
                    # If less than 40mm remaining on page, start a new one
                    if pdf.get_y() > 245:
                        pdf.add_page()
                        _inner_header(pdf, 'FACILITY INTELLIGENCE (CONTINUED)')
                        pdf.set_y(24)
                    pdf.set_fill_color(*_CARD)
                    pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_CYAN)
                    pdf.set_x(15); pdf.cell(180, 9, _s(f'  {finding_title}: {stat}'), fill=True, ln=True)
                    pdf.ln(1)
                    pdf.set_font('Helvetica', '', 11); pdf.set_text_color(*_TEXT)
                    pdf.set_x(15); pdf.multi_cell(180, 7, _s(f'  Finding: {desc}'))
                    pdf.set_text_color(*_GREEN)
                    pdf.set_x(15); pdf.multi_cell(180, 7, _s(f'  Action: {action}'))
                    pdf.ln(2)

                # ── TOP 10 HIGH RISK PATIENTS ────────────────────────
                pdf.add_page()
                _inner_header(pdf, 'TOP 10 HIGHEST RISK PATIENTS')
                pdf.set_y(24); _section_title(pdf, 'TOP 10 HIGHEST RISK PATIENTS')

                pdf.set_font('Helvetica', 'I', 12); pdf.set_text_color(*_MUTED)
                pdf.set_x(15); pdf.cell(180, 9, 'Patients requiring immediate contact and adherence support.', ln=True)
                pdf.ln(3)

                top10 = df_rep.nlargest(10, 'risk_pct').copy()

                # Compute real SHAP top driver per patient
                top10_indices = top10.index.tolist()
                top_drivers = []
                for idx in top10_indices:
                    try:
                        pos = df_rep.index.get_loc(idx)
                        sv, sv_ok = compute_shap_single(X_rep[pos])
                        sv_arr = np.abs(np.array(sv).flatten())
                        top_feat_idx = int(np.argmax(sv_arr))
                        top_feat = FEATURES[top_feat_idx] if top_feat_idx < len(FEATURES) else ''
                        top_drivers.append(FEAT_LABELS.get(top_feat, top_feat))
                    except Exception:
                        top_drivers.append('—')
                top10['top_driver'] = top_drivers

                # Shorten top driver labels for table display
                SHORT_DRIVER = {
                    'Prior ART Interruption': 'Prior Interruption',
                    'WHO Stage (1-4)': 'WHO Stage',
                    'CD4 at ART Start': 'CD4 at Start',
                    'Most Recent CD4': 'Recent CD4',
                    'CD4 Improvement': 'CD4 Change',
                    'Weight Change (kg)': 'Weight Change',
                    'Clinical Stage Worsened': 'Stage Worsened',
                    'Opportunistic Infection': 'Opp. Infection',
                    'Days: Diagnosis to ART': 'Dx to ART (days)',
                }
                top10['top_driver_short'] = top10['top_driver'].apply(
                    lambda x: SHORT_DRIVER.get(x, x)
                )

                pdf.set_fill_color(*_CARD)
                pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(*_CYAN)
                t_cols = [('Patient ID',30),('Risk %',20),('Age',12),('CD4',18),
                          ('Stage',16),('Prior Int.',18),('Top Driver',46),('Action',26)]
                pdf.set_x(15)
                for col_name, width in t_cols:
                    pdf.cell(width, 11, col_name, fill=True)
                pdf.ln()
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.3)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(2)

                for _, row in top10.iterrows():
                    urgency = "URGENT <24h" if row['risk_pct'] >= 90 else "48h contact" if row['risk_pct'] >= 80 else "This week"
                    cells = [
                        (str(row['patient_id']), 30, _TEXT),
                        (f"{row['risk_pct']:.1f}%", 20, _RED),
                        (f"{row['Age']:.0f}", 12, _TEXT),
                        (f"{row['Cd4AtStart']:.0f}", 18, _TEXT),
                        (f"Stage {row['stage_start_num']:.0f}", 16, _TEXT),
                        ("Yes" if row['had_interruption'] > 0.5 else "No", 18, _TEXT),
                        (str(row['top_driver_short'])[:22], 46, _AMBER),
                        (urgency, 26, _AMBER),
                    ]
                    pdf.set_x(15)
                    for val, width, vc in cells:
                        pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*vc)
                        pdf.cell(width, 9, val)
                    pdf.ln()
                    pdf.set_draw_color(*_CARD); pdf.set_line_width(0.2)
                    pdf.line(15, pdf.get_y(), 195, pdf.get_y())

                # ── SHAP EXPLAINABILITY — TOP PATIENT ────────────────
                try:
                    pdf.add_page()
                    _inner_header(pdf, 'PATIENT RISK EXPLANATION — SHAP ANALYSIS')
                    pdf.set_y(24); _section_title(pdf, 'PATIENT RISK EXPLANATION — SHAP ANALYSIS')

                    pdf.set_font('Helvetica', 'I', 12); pdf.set_text_color(*_MUTED)
                    pdf.set_x(15)
                    pdf.multi_cell(180, 8, _s(
                        'SHAP (SHapley Additive exPlanations) shows exactly which clinical factors '
                        'drove the risk score for the highest-risk patient. Red bars increase risk. '
                        'Green bars reduce risk. Each value is the precise contribution to the final score.'
                    ))
                    pdf.ln(4)

                    # Get top patient
                    top_pat = df_rep.nlargest(1, 'risk_pct').iloc[0]
                    top_pos = df_rep.index.get_loc(df_rep['risk_pct'].idxmax())
                    top_sv, sv_ok = compute_shap_single(X_rep[top_pos])
                    top_sv_arr = np.array(top_sv).flatten()

                    # Patient summary box
                    pdf.set_fill_color(*_CARD)
                    pdf.rect(15, pdf.get_y(), 180, 18, 'F')
                    y_box = pdf.get_y() + 4
                    pdf.set_xy(20, y_box)
                    pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*_RED)
                    pdf.cell(50, 9, _s(f"{top_pat['risk_pct']:.1f}% — HIGH RISK"))
                    pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                    pdf.cell(50, 9, _s(f"Patient: {top_pat['patient_id']}"))
                    pdf.cell(40, 9, _s(f"Age: {top_pat['Age']:.0f}"))
                    pdf.cell(40, 9, _s(f"CD4: {top_pat['Cd4AtStart']:.0f} cells/uL"))
                    pdf.ln(14)

                    # Build SHAP waterfall chart
                    sv_order = list(np.argsort(np.abs(top_sv_arr)))
                    sv_vals  = [float(top_sv_arr[i]) for i in sv_order if i < len(FEATURES)]
                    sv_names = []
                    for i in sv_order:
                        if i < len(FEATURES):
                            feat = FEATURES[i]
                            val  = float(top_pat[feat]) if feat in top_pat.index else 0.0
                            label = FEAT_LABELS.get(feat, feat)
                            sv_names.append(f"{label} = {val:.1f}")
                    sv_colors = ['#f85149' if v > 0 else '#3fb950' for v in sv_vals]

                    # Use short labels (no value) — value shown in bar annotation
                    sv_short_names = [FEAT_LABELS.get(FEATURES[i], FEATURES[i])
                                      for i in sv_order if i < len(FEATURES)]

                    fig, ax = plt.subplots(figsize=(11, 7), facecolor='#0d1117')
                    ax.set_facecolor('#0d1117')
                    bars = ax.barh(range(len(sv_short_names)), sv_vals,
                                   color=sv_colors, height=0.6,
                                   edgecolor='#161b22', linewidth=0.3)
                    x_range = max(abs(v) for v in sv_vals) if sv_vals else 0.1
                    min_bar_for_inline = x_range * 0.20
                    for i, (bar, v) in enumerate(zip(bars, sv_vals)):
                        offset = x_range * 0.03
                        abs_v = abs(v)
                        if abs_v >= min_bar_for_inline:
                            # Bar is long enough — annotate just beyond the bar end
                            x_pos = v + offset if v >= 0 else v - offset
                            ha = 'left' if v >= 0 else 'right'
                        else:
                            # Short bar — place annotation to right of zero line always
                            x_pos = x_range * 0.04
                            ha = 'left'
                        ax.text(x_pos, i, f'{v:+.4f}', va='center', ha=ha,
                                fontsize=10, color='#e6edf3', fontweight='bold')
                    ax.set_yticks(range(len(sv_short_names)))
                    ax.set_yticklabels(sv_short_names, fontsize=11, color='#cdd9e5')
                    ax.axvline(0, color='#8b949e', lw=1.5)
                    ax.set_xlabel('SHAP Value — contribution to risk score',
                                  color='#8b949e', fontsize=11)
                    ax.tick_params(colors='#8b949e', labelsize=10)
                    for sp in ax.spines.values():
                        sp.set_color('#21262d')
                    cohort_baseline = df_rep['risk_pct'].mean()
                    ax.set_title(
                        f"SHAP Explanation: {top_pat['patient_id']} | "
                        f"Risk: {top_pat['risk_pct']:.1f}% (HIGH) | "
                        f"Cohort baseline: {cohort_baseline:.1f}%",
                        color='#e6edf3', fontsize=12, pad=12, fontweight='bold'
                    )
                    plt.tight_layout(pad=1.5)

                    # Save to buffer and embed in PDF
                    img_buf = BytesIO()
                    fig.savefig(img_buf, format='png', dpi=200,
                                bbox_inches='tight', facecolor='#0d1117')
                    plt.close(fig)
                    img_buf.seek(0)

                    chart_y = pdf.get_y()
                    pdf.image(img_buf, x=15, y=chart_y, w=180)
                    pdf.set_y(chart_y + 125)
                    pdf.ln(4)

                    # Top 3 drivers as text summary below chart
                    sv_desc_order = list(reversed(np.argsort(np.abs(top_sv_arr))))
                    pdf.set_font('Helvetica', 'B', 12); pdf.set_text_color(*_CYAN)
                    pdf.set_x(15); pdf.cell(180, 9, _s('Top 3 Risk Drivers for this Patient:'), ln=True)
                    for rank, feat_idx in enumerate(sv_desc_order[:3], 1):
                        if feat_idx >= len(FEATURES):
                            continue
                        feat = FEATURES[feat_idx]
                        sv_val = float(top_sv_arr[feat_idx])
                        direction = 'increases risk' if sv_val > 0 else 'reduces risk'
                        label = FEAT_LABELS.get(feat, feat)
                        pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                        pdf.set_x(20)
                        pdf.cell(6, 9, _s(f'{rank}.'))
                        pdf.multi_cell(169, 9, _s(
                            f"{label} {direction} (SHAP: {sv_val:+.4f})"
                        ))
                    pdf.ln(3)
                    pdf.set_font('Helvetica', 'I', 10); pdf.set_text_color(*_MUTED)
                    pdf.set_x(15)
                    pdf.multi_cell(180, 7, _s(
                        'SHAP values are model-derived. Interpret alongside clinical judgement. '
                        'Feature contributions reflect patterns learned from the Nigerian discovery cohort '
                        'and may vary across populations and facility types.'
                    ))
                except Exception:
                    pass  # SHAP page is best-effort — never break report generation

                # ── RECOMMENDED ACTIONS ──────────────────────────────
                pdf.add_page()
                _inner_header(pdf, 'RECOMMENDED PROGRAMME ACTIONS')
                pdf.set_y(24); _section_title(pdf, 'RECOMMENDED PROGRAMME ACTIONS')

                action_sections = [
                    ('IMMEDIATE (This Week)', _RED, [
                        f'Contact the {n_high} HIGH risk patients - begin with the top 10 listed above',
                        'Activate peer navigator support for patients with prior interruption history',
                        'Schedule viral load tests for patients showing CD4 decline',
                        f'Prioritise TB-HIV co-treatment coordination for {int(pct_tb/100*n_total)} identified co-infected patients',
                    ]),
                    ('SHORT TERM (1-4 Weeks)', _AMBER, [
                        'Review regimen tolerability for patients with reported side effects',
                        'Site visit to primary health centres - structural quality assessment',
                        'Initiate adherence counselling for all MEDIUM risk patients',
                        'Review diagnosis-to-ART delays and implement fast-track protocols where feasible',
                    ]),
                    ('STRATEGIC (1-3 Months)', _GREEN, [
                        'Consider Differentiated Service Delivery (DSD) model expansion at primary HCs',
                        'Develop outcome-adjusted performance metrics for facility-level monitoring',
                        'Male engagement strategy - flexible hours, community dispensing, peer support',
                        'Apply SmartDaaS risk intelligence framework to PEPFAR MER quarterly reporting',
                    ]),
                ]
                for sec_title, color, actions in action_sections:
                    pdf.set_fill_color(*_CARD)
                    pdf.rect(15, pdf.get_y(), 180, 10, 'F')
                    pdf.set_draw_color(*color); pdf.set_line_width(0.5)
                    pdf.line(15, pdf.get_y(), 15, pdf.get_y()+10)
                    pdf.set_font('Helvetica', 'B', 13); pdf.set_text_color(*color)
                    pdf.set_x(20); pdf.cell(175, 10, sec_title, ln=True)
                    pdf.ln(2)
                    for action in actions:
                        pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                        pdf.set_x(20); pdf.cell(6, 9, chr(149))
                        pdf.multi_cell(164, 9, _s(action)); pdf.ln(1)
                    pdf.ln(5)

                # ── METHODOLOGY & LIMITATIONS ────────────────────────
                pdf.add_page()
                _inner_header(pdf, 'METHODOLOGY & LIMITATIONS')
                pdf.set_y(24); _section_title(pdf, 'METHODOLOGY & LIMITATIONS')

                pdf.set_font('Helvetica', '', 12); pdf.set_text_color(*_TEXT)
                pdf.set_x(15)
                pdf.multi_cell(180, 9, _s(
                    "Patient risk scores are generated by a Random Forest classifier trained on 27,288 HIV patients "
                    "from the Nigerian national HIV programme (discovery cohort). "
                    "Cross-validation AUC: 0.963 (5-fold stratified CV, balanced sample); sensitivity 87.3% "
                    "(95% CI: 86.4-88.2%) and specificity 95.7% (95% CI: 95.2-96.2%) at threshold 0.50 on "
                    "balanced training data (Chinthala LK. medRxiv 2026. doi:10.64898/2026.05.15.26353325). "
                    "Temporal validation AUC: 0.772 (95% CI: 0.744-0.802) on held-out post-2015 patients "
                    "never seen during training — the primary operational performance estimate. "
                    "At the deployment threshold of 0.70 on the temporal holdout, sensitivity is 72.8% and "
                    "specificity is 98.8%, reflecting a high-precision operating point that minimises false "
                    "alarms while reliably identifying the highest-risk patients for outreach. "
                    "SHAP (SHapley Additive exPlanations) values provide per-patient clinical reasoning. "
                    "Facility intelligence is based on multivariable logistic "
                    "regression with HC3 cluster-robust standard errors applied to 27,288 Nigerian HIV programme "
                    "patients (Chinthala 2026). Economic estimates apply a conservative 23% interruption reduction "
                    "assumption for contacted high-risk patients (PEPFAR retention literature). "
                    "Three cost scenarios are used: (1) Conservative — USD 1,850 per averted poor outcome "
                    "(Menzies et al., AIDS 2011, Nigeria-specific PEPFAR data); "
                    "(2) Mid — USD 3,500, reflecting CPI inflation adjustment to 2024 USD (~89% increase since 2009); "
                    "(3) Upper — USD 5,000, reflecting full programme cost of re-engagement including "
                    "viral load testing, tracing costs, and downstream second-line therapy risk "
                    "(Haacker et al., Health Affairs 2022; ACT model estimates). "
                    f"This report uses the conservative estimate (USD {est_avoidable_cost:,}) as the headline figure. "
                    f"Mid-range estimate: USD {est_avoidable_cost_mid:,}. Upper estimate: USD {est_avoidable_cost_upper:,}. "
                    "All findings are illustrative. Prospective validation is required before programmatic application. "
                    "SmartDaaS v1.0 is a decision-support platform for HIV programme intelligence. "
                    "All outputs require review by qualified programme teams prior to operational use. "
                    "Code: github.com/Kchinthala15/smartdaas-hiv-validation"
                ))
                pdf.ln(8)
                # Data source note
                pdf.set_fill_color(*_CARD); pdf.rect(15, pdf.get_y(), 180, 16, 'F')
                pdf.set_draw_color(*_CYAN); pdf.set_line_width(0.3)
                pdf.line(15, pdf.get_y(), 15, pdf.get_y()+16)
                pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*_CYAN)
                pdf.set_xy(20, pdf.get_y()+2); pdf.cell(170, 8, 'DATA SOURCE')
                pdf.set_font('Helvetica', '', 11); pdf.set_text_color(*_TEXT)
                pdf.set_xy(20, pdf.get_y()+6); pdf.cell(170, 8, _s(data_source))

                # Save to bytes
                pdf_bytes = bytes(pdf.output())
                pdf_buffer = BytesIO(pdf_bytes)

                st.success("✓ Report generated successfully!")
                log_report(supabase, n_total, "Executive PDF")
                st.download_button(
                    label="📥 Download Executive PDF Report",
                    data=pdf_buffer,
                    file_name=f"SmartDaaS_Report_{org_name.replace(' ','_')}_{report_date.strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"PDF generation error: {e}")
                st.info("If fpdf2 is not installed, run: pip install fpdf2")

    st.markdown("""<div class="warn-box">
    ⚠️ <strong>Decision-support output.</strong> This report is generated by an AI-powered analytics platform
    and should not replace clinical judgement or programme expertise. All findings require
    validation before operational use.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PROGRAMME DASHBOARD — single executive screen
# ═════════════════════════════════════════════════════════════
elif page == "📊 Programme Dashboard":
    st.markdown("""
### Programme Risk Overview

**Everything at a glance — for programme decisions and reporting.**
""")

    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    facility_pool = ["Kano General (Tertiary)", "Lagos Island GH (Tertiary)",
                     "Abuja Primary HC (Primary)", "Enugu State HF (Secondary)",
                     "Ibadan HC (Primary)", "Port Harcourt GH (Secondary)"]
    facility_levels = {"Kano General (Tertiary)": "Tertiary", "Lagos Island GH (Tertiary)": "Tertiary",
                       "Abuja Primary HC (Primary)": "Primary", "Enugu State HF (Secondary)": "Secondary",
                       "Ibadan HC (Primary)": "Primary", "Port Harcourt GH (Secondary)": "Secondary"}

    # ── Use uploaded/scored data if available, else fall back to demo ──
    _scored = st.session_state.get('df_scored', None)
    if _scored is not None and len(_scored) > 0:
        df_dash = _scored.copy()
        for f in FEATURES:
            if f not in df_dash.columns:
                df_dash[f] = 0.0
            else:
                df_dash[f] = pd.to_numeric(df_dash[f], errors='coerce').fillna(0.0)
        if 'patient_id' not in df_dash.columns:
            df_dash['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_dash))]
        X_dash = df_dash[FEATURES].values.astype(float)
        probs_dash = model.predict_proba(X_dash)[:, 1]
        df_dash['risk_pct'] = (probs_dash * 100).round(1)
        if 'risk_label' not in df_dash.columns:
            df_dash['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                                     for p in probs_dash]
        st.info(f"ℹ️ Showing dashboard for your uploaded cohort ({len(df_dash):,} patients).")
    else:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(300, len(X_DEMO)), replace=False)
        df_dash = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        df_dash['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_dash))]
        X_dash = df_dash[FEATURES].values.astype(float)
        probs_dash = model.predict_proba(X_dash)[:, 1]
        df_dash['risk_pct'] = (probs_dash * 100).round(1)
        df_dash['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                                 for p in probs_dash]
        st.markdown("""<div class="info-box">
        🔬 <strong>Demo mode:</strong> Upload patient data on the Patient Risk page
        to see your programme dashboard.
        </div>""", unsafe_allow_html=True)

    # Assign facilities (simulated — in deployment comes from DHIS2/EMR export)
    np.random.seed(42)
    df_dash['facility'] = np.random.choice(facility_pool, size=len(df_dash))
    df_dash['facility_level'] = df_dash['facility'].map(facility_levels)

    n_total = len(df_dash)
    n_high = (df_dash['risk_label']=='HIGH').sum()
    n_med = (df_dash['risk_label']=='MEDIUM').sum()
    n_low = (df_dash['risk_label']=='LOW').sum()
    avg_risk = df_dash['risk_pct'].mean()
    pct_interruption = (df_dash['had_interruption']>0.5).mean()*100
    est_cost = int(n_high * 0.3 * 880)

    # ── ROW 1: Key metrics ────────────────────────────────
    st.markdown('<p class="section-hdr">Programme Risk — Right Now</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(f'<div class="metric-box"><div class="metric-val">{n_total}</div><div class="metric-lbl">Total Patients</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="risk-high" style="padding:0.8rem"><div class="risk-number" style="font-size:1.6rem">{n_high}</div><div class="risk-label">HIGH Risk</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="risk-medium" style="padding:0.8rem"><div class="risk-number" style="font-size:1.6rem">{n_med}</div><div class="risk-label">MEDIUM Risk</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="risk-low" style="padding:0.8rem"><div class="risk-number" style="font-size:1.6rem">{n_low}</div><div class="risk-label">LOW Risk</div></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_risk:.1f}%</div><div class="metric-lbl">Avg Risk Score</div></div>', unsafe_allow_html=True)
    with c6: st.markdown(f'<div class="metric-box"><div class="metric-val">${est_cost/1000:.0f}K</div><div class="metric-lbl">Est. Avoidable Cost</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 2: Charts side by side ────────────────────────
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown('<p class="section-hdr">Risk Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        ax.hist(probs_dash*100, bins=20, color='#21d4fd', alpha=0.8, edgecolor='#0d1117')
        ax.axvline(70, color='#f85149', lw=2, linestyle='--', label='HIGH (70%)')
        ax.axvline(40, color='#e3b341', lw=2, linestyle='--', label='MEDIUM (40%)')
        ax.set_xlabel('Risk Score (%)', color='#adbac7', fontsize=9)
        ax.set_ylabel('Patients', color='#adbac7', fontsize=9)
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#adbac7')
        ax.tick_params(colors='#adbac7', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#444c56')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c_right:
        st.markdown('<p class="section-hdr">Facility Risk Burden</p>', unsafe_allow_html=True)
        fac_risk = df_dash.groupby('facility').agg(
            high_pct=('risk_label', lambda x: (x=='HIGH').mean()*100),
            level=('facility_level','first')
        ).sort_values('high_pct', ascending=True)
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        colors = ['#f85149' if l=='Primary' else '#e3b341' if l=='Secondary' else '#21d4fd'
                  for l in fac_risk['level']]
        names = [f.split(' (')[0][:20] for f in fac_risk.index]
        bars = ax.barh(range(len(names)), fac_risk['high_pct'], color=colors, height=0.6, edgecolor='#0d1117')
        for i, (bar, v) in enumerate(zip(bars, fac_risk['high_pct'])):
            ax.text(v+0.5, i, f'{v:.1f}%', va='center', fontsize=8, color='#e6edf3', fontweight='bold')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8, color='#cdd9e5')
        ax.set_xlabel('% HIGH Risk Patients', color='#adbac7', fontsize=8)
        ax.tick_params(colors='#adbac7', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#444c56')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── ROW 3: Top patients + Key actions ─────────────────
    c_pts, c_actions = st.columns(2)

    with c_pts:
        st.markdown('<p class="section-hdr">Top 5 — Immediate Contact Required</p>', unsafe_allow_html=True)
        top5 = df_dash.nlargest(5, 'risk_pct')[['patient_id','risk_pct','facility','Age','had_interruption']].copy()
        top5['Urgency'] = top5['risk_pct'].apply(lambda x: '🔴 Today' if x>=90 else '🟠 48hrs')
        top5.columns = ['Patient','Risk (%)','Facility','Age','Prior Interruption','Urgency']
        top5['Prior Interruption'] = top5['Prior Interruption'].apply(lambda x: '⚠️ Yes' if x>0.5 else '✓ No')
        st.dataframe(top5, use_container_width=True, height=220)

    with c_actions:
        st.markdown('<p class="section-hdr">Programme Action Summary</p>', unsafe_allow_html=True)
        urgent = (df_dash['risk_pct']>=90).sum()
        tb_hiv = (df_dash['tb_positive']>0.5).sum()
        adv_disease = (df_dash['stage_start_num']>=3).sum()
        interruptions = (df_dash['had_interruption']>0.5).sum()
        declining_cd4 = (df_dash['CD4_improvement']<0).sum()

        actions_data = [
            ('🔴 URGENT', f'{urgent} patients ≥90% risk', 'Phone contact within 24 hours'),
            ('🟠 HIGH', f'{n_high-urgent} patients 70-89%', 'Contact within 48 hours'),
            ('⚠️ TB-HIV', f'{tb_hiv} co-infected patients', 'Coordinate TB-HIV treatment'),
            ('📉 CD4 Decline', f'{declining_cd4} patients', 'Viral load follow-up within 14 days'),
            ('🏥 Adv. Disease', f'{adv_disease} at Stage 3-4', 'Clinical officer review required'),
            ('🔁 Interruptions', f'{interruptions} prior history', 'Enhanced retention support'),
        ]
        for icon_label, count, action in actions_data:
            st.markdown(f"**{icon_label}** — {count} → *{action}*")

    # ── ROW 4: Quick exports ───────────────────────────────
    st.markdown('<p class="section-hdr">Quick Exports</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        top_export = df_dash.nlargest(20, 'risk_pct')[['patient_id','risk_pct','risk_label','facility','Age']].copy()
        st.download_button("📥 Top 20 High Risk Patients",
                           data=top_export.to_csv(index=False).encode(),
                           file_name="smartdaas_top20_dashboard.csv",
                           mime="text/csv", use_container_width=True)
    with c2:
        fac_export = fac_risk.reset_index()[['facility','high_pct','level']].copy()
        fac_export.columns = ['Facility','High Risk (%)','Level']
        st.download_button("📥 Facility Risk Summary",
                           data=fac_export.to_csv(index=False).encode(),
                           file_name="smartdaas_facility_summary.csv",
                           mime="text/csv", use_container_width=True)
    with c3:
        full = df_dash[['patient_id','risk_pct','risk_label','facility','facility_level']+FEATURES]
        st.download_button("📥 Full Cohort Export",
                           data=full.sort_values('risk_pct',ascending=False).to_csv(index=False).encode(),
                           file_name="smartdaas_full_cohort_dashboard.csv",
                           mime="text/csv", use_container_width=True)

    st.markdown("""<div class="warn-box">
    ⚠️ Demo mode — facility assignments are simulated. In deployment, facility data
    comes from your programme's DHIS2/EMR export.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# MODEL TRANSPARENCY — Temporal Drift Monitor
# ═════════════════════════════════════════════════════════════
elif page == "🧬 Model Transparency":
    st.markdown("""
### Performance & Transparency

**SmartDaaS v1.0 delivers strong predictive performance — and shows its work.**

Every patient prediction includes a full SHAP explanation so your team understands
the clinical drivers. We believe transparency builds trust.
""")

    st.markdown("""<div class="info-box">
    💡 <strong>Important Note:</strong> Local validation on your programme\'s data is required
    for optimal performance. The temporal AUC (0.772) is the most realistic deployment estimate.
    After local retraining on your historical data, performance typically improves.
    </div>""", unsafe_allow_html=True)

    # ── PERFORMANCE PANEL ─────────────────────────────────
    st.markdown('<p class="section-hdr">Model Performance</p>', unsafe_allow_html=True)

    # ── Block 1: Cross-validation performance ─────────────
    st.markdown("""<div style="background:#1a2030;border:1px solid #21d4fd44;border-radius:8px;
        padding:0.75rem 1rem 0.5rem 1rem;margin-bottom:0.4rem">
        <div style="font-size:0.7rem;color:#21d4fd;font-family:'IBM Plex Mono',monospace;
            text-transform:uppercase;letter-spacing:2px;margin-bottom:0.6rem">
            Cross-Validation Performance — Training Data (5-fold stratified CV, balanced sample)
        </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cv_metrics = [
        (c1, "0.963",  "CV AUC",        "#21d4fd"),
        (c2, "87.3%",  "Sensitivity",   "#21d4fd"),
        (c3, "95.7%",  "Specificity",   "#21d4fd"),
        (c4, "0.079",  "Brier Score",   "#21d4fd"),
        (c5, "0.50",   "CV Threshold",  "#8b949e"),
    ]
    for col, val, lbl, color in cv_metrics:
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color:{color}">'
                f'{val}</div><div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.72rem;color:#8b949e;padding:0.3rem 0.25rem 0.1rem 0.25rem">
        Sensitivity 87.3% (95% CI: 86.4–88.2%) and Specificity 95.7% (95% CI: 95.2–96.2%)
        computed at threshold 0.50 on balanced training data.
        Source: Chinthala LK. <em>medRxiv</em> 2026. doi:10.64898/2026.05.15.26353325
    </div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Block 2: Temporal holdout performance ─────────────
    st.markdown("""<div style="background:#1a2010;border:1px solid #e3b34144;border-radius:8px;
        padding:0.75rem 1rem 0.5rem 1rem;margin-bottom:0.4rem">
        <div style="font-size:0.7rem;color:#e3b341;font-family:'IBM Plex Mono',monospace;
            text-transform:uppercase;letter-spacing:2px;margin-bottom:0.6rem">
            Temporal Holdout Performance — Held-Out Future Patients (post-2015, real-world deployment estimate)
        </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    temporal_metrics = [
        (c1, "0.772",  "Temporal AUC",      "#e3b341"),
        (c2, "0.191",  "Performance Gap",   "#f85149"),
        (c3, "72.8%",  "Sensitivity",       "#e3b341"),
        (c4, "98.8%",  "Specificity",       "#e3b341"),
        (c5, "0.70",   "Deploy Threshold",  "#e3b341"),
    ]
    for col, val, lbl, color in temporal_metrics:
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color:{color}">'
                f'{val}</div><div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.72rem;color:#8b949e;padding:0.3rem 0.25rem 0.1rem 0.25rem">
        Temporal AUC 0.772 (95% CI: 0.744–0.802) on held-out post-2015 patients never seen during training.
        Sensitivity 72.8% and Specificity 98.8% computed at deployment threshold 0.70 on temporal holdout.
        Performance gap vs CV AUC is expected — all ML models degrade on future data.
    </div></div>""", unsafe_allow_html=True)

    # ── WHAT THIS MEANS ───────────────────────────────────
    st.markdown('<p class="section-hdr">What These Numbers Mean</p>', unsafe_allow_html=True)

    st.markdown("""
**Cross-validation AUC (0.963)** is measured within the training data using 5-fold stratified CV
on a balanced sample. Sensitivity 87.3% and Specificity 95.7% are the corresponding CV metrics
at threshold 0.50. These are optimistic estimates — the model has seen similar patients.
Published in: Chinthala LK. *medRxiv* 2026. doi:10.64898/2026.05.15.26353325

**Temporal AUC (0.772)** measures how well the model predicts on patients from *later time periods*
it has never seen. This is the honest estimate of real-world deployment performance.
AUC 0.772 means the model correctly ranks 77.2% of patient pairs (high-risk above low-risk).
At the deployment threshold of 0.70, this yields Sensitivity 72.8% and Specificity 98.8% —
a high-precision operating point that minimises false alarms while reliably identifying
the highest-risk patients for outreach.

**The performance gap (0.191)** is expected and normal — all ML models degrade on future data.
It does not invalidate the model. After local recalibration on your programme's historical data,
performance typically improves toward the CV estimate.

**What AUC 0.772 means in practice:** For every 100 patient pairs where one truly has a poor
outcome and one does not, the model correctly identifies the higher-risk patient ~77 times.
This is substantially better than random (0.5) and comparable to validated clinical prediction
tools in HIV care.
""")

    # Visual: CV vs Temporal
    fig, ax = plt.subplots(figsize=(8, 3), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    categories = ['Cross-Validation\n(optimistic)', 'Temporal Validation\n(realistic)', 'Random\n(baseline)']
    values = [0.963, 0.772, 0.5]
    colors = ['#21d4fd', '#e3b341', '#444c56']
    bars = ax.barh(categories, values, color=colors, height=0.5, edgecolor='#0d1117')
    ax.axvline(0.5, color='#f85149', lw=1.5, linestyle='--', alpha=0.7)
    for bar, v in zip(bars, values):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
                va='center', fontsize=11, color='#e6edf3', fontweight='bold')
    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel('AUC-ROC', color='#adbac7', fontsize=10)
    ax.set_title('Model Performance: Cross-Validation vs Temporal Validation',
                 color='#e6edf3', fontsize=11, pad=10)
    ax.tick_params(colors='#adbac7', labelsize=9)
    for sp in ax.spines.values(): sp.set_color('#444c56')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── LOCAL VALIDATION REQUIREMENT ─────────────────────
    st.markdown('<p class="section-hdr">⚠️ Local Validation Requirement</p>', unsafe_allow_html=True)
    st.markdown("""<div class="warn-box">
    <strong>SmartDaaS requires local validation before operational use.</strong><br><br>
    The model was trained on the Nigerian national HIV programme dataset (Quality of Care dataset, 27,288 patients, 2006–2018 — discovery cohort).
    Performance on your programme's data — different population, different time period,
    different data quality — will differ from the figures shown here.<br><br>
    <strong>What this means for a pilot:</strong><br>
    In a shadow analytics pilot, SmartDaaS runs on 12 months of your historical closed programme
    data. We compare SmartDaaS-identified high-risk patients against known outcomes from that
    period. This generates a locally-validated AUC specific to your programme — the number
    you can confidently report to funders.
    </div>""", unsafe_allow_html=True)

    # ── PILOT RECALIBRATION PLAN ──────────────────────────
    st.markdown('<p class="section-hdr">Pilot Recalibration Plan</p>', unsafe_allow_html=True)

    steps = [
        ("Step 1 — Historical data ingestion", "Partner provides 12 months of closed programme data (CSV/DHIS2 export). SmartDaaS ingests with minimal friction — no workflow disruption."),
        ("Step 2 — Retrospective validation", "Model predictions on historical patients are compared against known outcomes. Local AUC, sensitivity, and specificity are computed for your specific programme."),
        ("Step 3 — Recalibration (if needed)", "If local AUC < 0.70, the model is recalibrated on partner-approved historical data only — no patient data leaves the partner's control."),
        ("Step 4 — Prospective shadow analytics", "Validated model runs alongside existing workflows for 3–6 months. SmartDaaS risk lists are compared against standard programme reports — identifying gaps spreadsheets missed."),
        ("Step 5 — Outcome report", "Summary report delivered: local AUC, high-risk patient detection rate, facility performance findings, economic impact estimate."),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}**")
        st.markdown(f"{desc}")
        st.markdown("")

    st.markdown("""<div class="info-box">
    All recalibration uses only partner-approved historical data.
    No patient data is transmitted outside the partner organisation's control.
    SmartDaaS operates as a shadow analytics layer — existing workflows are not disrupted.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PILOT MODEL PAGE — Shadow Analytics narrative
# ═════════════════════════════════════════════════════════════
elif page == "✅ Local Validation":
    if not MODEL_OK:
        st.error("Model not loaded. Cannot run recalibration.")
    else:
        render_recalibration_page(model, log_recalibration=log_recalibration, supabase=supabase)

elif page == "🛡️ Pilot Model":
    st.markdown("""
### SmartDaaS Implementation Programme
""")

    # Hero statement
    st.markdown("""<div style="background:linear-gradient(135deg,#161b22,#0d2137);
        border:1px solid #21d4fd44;border-radius:12px;padding:2rem;margin-bottom:1.5rem">
        <p style="font-size:1.6rem;color:#e6edf3;font-weight:700;margin:0 0 0.5rem 0">
        Designed for rapid implementation.</p>
        <p style="font-size:1rem;color:#21d4fd;font-weight:600;margin:0 0 0.75rem 0">
        Run SmartDaaS alongside your existing systems with zero disruption.</p>
        <p style="color:#cdd9e5;margin:0;font-size:0.95rem;line-height:1.6">
        SmartDaaS is designed for rapid implementation through our
        <strong>Shadow Analytics Programme</strong> — a low-risk way for HIV implementing
        partners to test and validate the platform using their own historical data.
        No workflow changes. No new data collection. Results within weeks.</p>
    </div>""", unsafe_allow_html=True)

    # What you get
    st.markdown('''<p class="section-hdr">What You Get</p>''', unsafe_allow_html=True)
    gets = [
        ("📋", "High-risk patient lists with explainable risk scores"),
        ("🏥", "Facility performance rankings and care-gap heatmaps"),
        ("💰", "Economic impact estimates — avoidable costs quantified"),
        ("📄", "Executive intelligence reports ready for donor reporting"),
        ("🎯", "Specific intervention recommendations aligned with PEPFAR MER indicators"),
    ]
    for icon, text in gets:
        st.markdown(f"**{icon} {text}**")

    # ── COMPARISON TABLE ──────────────────────────────────
    st.markdown('<p class="section-hdr">How It Compares to What You Have Now</p>',
                unsafe_allow_html=True)
    comparison = pd.DataFrame({
        'What you have now': [
            'Quarterly aggregate TX_CURR / TX_PVLS reports',
            'Manual facility comparisons in Excel',
            'Reactive follow-up after patient dropout',
            'Gut instinct for resource allocation',
            'Generic adherence counselling for all patients',
            'No case-mix adjustment in facility ranking',
        ],
        'What SmartDaaS adds': [
            'Weekly patient-level risk rankings — flag patients before they drop out',
            'Risk-adjusted facility benchmarking — separates quality from patient severity',
            'Predictive flagging before dropout — intervene while the patient is still engaged',
            'Evidence-based resource allocation — prioritise highest-burden facilities',
            'Tiered intervention recommendations — matched to each patient\'s specific drivers',
            'Structural performance intelligence — identify which facilities are truly failing',
        ]
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # What it is / isn't
    st.markdown('<p class="section-hdr">What the Pilot Is — and Is Not</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""**✅ What SmartDaaS IS:**
- A shadow analytics layer on historical programme data
- A patient-level risk stratification engine
- A facility benchmarking framework
- An executive intelligence reporting tool
- A validation study for AI-assisted HIV programme management
- A complement to PEPFAR MER reporting
""")
    with c2:
        st.markdown("""**❌ What SmartDaaS is NOT:**
- A replacement for clinical decision-making
- A real-time EMR or patient management system
- A validated clinical tool (requires local validation before operational use)
- A workflow disruption
- A requirement for new data collection
- Authoritative without local validation
""")

    # Pilot workflow
    st.markdown('<p class="section-hdr">The 4-Step Shadow Analytics Pilot</p>', unsafe_allow_html=True)

    pilot_steps = [
        ("📁", "Step 1 — Historical Data Ingestion", "12 months",
         "You provide your existing programme data export — CSV from your EMR system or DHIS2 aggregate export. SmartDaaS accepts common HIV programme data formats with automatic column mapping. No new data collection. No workflow changes. Time to first analysis: same day.",
         ["CSV / DHIS2 export from your existing system", "SmartDaaS auto-maps columns to model features", "Data stays within your organisation's control", "No patient data transmitted externally"]),

        ("🔬", "Step 2 — Retrospective Validation", "2-4 weeks",
         "SmartDaaS generates risk predictions for all patients in your historical dataset. We compare predictions against known outcomes from that period — patients who actually interrupted treatment, had poor adherence, or died. This produces a locally-validated performance estimate specific to your programme.",
         ["Local AUC computed on your programme's data", "Sensitivity and specificity benchmarked against your outcomes", "High-risk cluster profiles compared to standard reports", "Identifies gaps that aggregate TX_CURR / TX_PVLS missed"]),

        ("📊", "Step 3 — Prospective Shadow Analytics", "3-6 months",
         "The validated model runs alongside your existing workflows. Each reporting period, you export your programme data and SmartDaaS generates: patient risk rankings, facility performance benchmarks, intervention recommendations, and an executive report. Your team uses this intelligence alongside — not instead of — existing reports.",
         ["Weekly high-risk patient lists for CHW / clinical teams", "Monthly facility performance rankings", "Quarterly executive intelligence reports for donor reporting", "Comparison: SmartDaaS alerts vs standard follow-up protocols"]),

        ("📄", "Step 4 — Pilot Outcome Report", "End of pilot",
         "SmartDaaS delivers a comprehensive pilot outcome report documenting: validated model performance on your programme data, high-risk patient detection rates, facility-level structural findings, economic impact estimates, and recommendations for programme-level investment. This is the document you take to your board and funders.",
         ["Locally validated AUC and performance metrics", "High-risk patient detection rate vs standard protocols", "Facility performance findings and QI investment targets", "Economic impact estimate with your programme's actual costs", "Recommendations for PEPFAR MER integration"]),
    ]

    step_colors = [
        ("#00e5ff", "#00e5ff18", "#00e5ff44"),   # Step 1 — cyan
        ("#ffb300", "#ffb30018", "#ffb30044"),   # Step 2 — amber
        ("#30d158", "#30d15818", "#30d15844"),   # Step 3 — green
        ("#c084fc", "#c084fc18", "#c084fc44"),   # Step 4 — purple
    ]
    for idx, (icon, title, duration, description, deliverables) in enumerate(pilot_steps):
        col, col_bg, col_bd = step_colors[idx % len(step_colors)]
        step_num = idx + 1
        st.markdown(f"""<div style="background:#1e2530;border:1px solid {col_bd};
            border-left:5px solid {col};border-radius:0 8px 8px 0;
            padding:1.25rem 1.25rem 1rem 1.25rem;margin-bottom:0.5rem">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem">
                <div style="display:flex;align-items:center;gap:0.75rem">
                    <span style="background:{col};color:#0d1117;font-family:'IBM Plex Mono',monospace;
                        font-size:0.7rem;font-weight:700;padding:3px 8px;border-radius:4px;
                        letter-spacing:1px">STEP {step_num}</span>
                    <span style="font-size:1rem;font-weight:700;color:{col}">{title.split(' — ',1)[1] if ' — ' in title else title}</span>
                </div>
                <span style="background:{col_bg};border:1px solid {col_bd};color:{col};
                    font-size:0.75rem;padding:3px 10px;border-radius:4px;font-weight:600;
                    white-space:nowrap">{duration}</span>
            </div>
            <p style="color:#dde8f2;font-size:0.93rem;margin:0 0 0.6rem 0;line-height:1.6">{description}</p>
        </div>""", unsafe_allow_html=True)
        for d in deliverables:
            st.markdown(f'<p style="color:#c8d8e8;margin:0.15rem 0 0.15rem 1rem;font-size:0.9rem">✓ {d}</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Data requirements
    st.markdown('<p class="section-hdr">Data Requirements</p>', unsafe_allow_html=True)

    req_data = {
        'Variable': ['Age', 'Sex', 'CD4 at ART start', 'Most recent CD4',
                     'WHO clinical stage', 'ART start date', 'HIV diagnosis date',
                     'Weight at start', 'ART interruption history', 'TB status',
                     'Side effects', 'Opportunistic infections'],
        'Required?': ['✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes',
                      '✅ Yes', '⚡ Recommended', '⚡ Recommended', '⚡ Recommended',
                      '⚡ Recommended', '⚡ Recommended'],
        'Typically available in': ['All EMRs', 'All EMRs', 'Routine labs', 'Routine labs',
                                    'Clinical records', 'ART register', 'ART register',
                                    'Clinical records', 'ART register', 'TB register / EMR',
                                    'Clinical notes / EMR', 'Clinical notes / EMR'],
        'DHIS2 available?': ['✅', '✅', '✅', '✅', '✅', '✅', '✅', '⚡', '⚡', '✅', '❌', '⚡'],
    }
    st.dataframe(pd.DataFrame(req_data), use_container_width=True)

    st.markdown("""<div class="info-box">
    💡 <strong>Missing variables?</strong> SmartDaaS handles missing data via column-median
    imputation with a documented missingness report. The model can run with partial data —
    performance will be lower but the analysis remains valid and documented transparently.
    </div>""", unsafe_allow_html=True)

    # Contact
    st.markdown('<p class="section-hdr">Interested in a Pilot?</p>', unsafe_allow_html=True)
    st.markdown("""<div style="background:#161b22;border:1px solid #21d4fd44;border-radius:8px;padding:1.5rem">
        <p style="color:#e6edf3;font-size:1rem;font-weight:600;margin:0 0 0.5rem 0">
        SmartDaaS is seeking one analytical pilot partner for a 6-month shadow analytics engagement.</p>
        <p style="color:#cdd9e5;margin:0 0 1rem 0">
        Ideal partner: PEPFAR implementing partner or NGO managing ART programme in Nigeria
        or comparable sub-Saharan African setting. You provide historical programme data export.
        SmartDaaS delivers facility + patient risk intelligence reports.</p>
        <p style="color:#21d4fd;font-weight:600;margin:0">
        Contact: chinthalakalyani1@gmail.com<br>
        GitHub: github.com/Kchinthala15/smartdaas-hiv-validation<br>
        ORCID: 0009-0009-8736-6673</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="warn-box">
    ⚠️ SmartDaaS is a decision-support platform designed for HIV programme intelligence and operational analytics. It is not intended to replace clinical judgment or function as an autonomous clinical decision-making system. All outputs should be reviewed by qualified programme teams prior to operational use. SmartDaaS should not be used to make individual patient treatment decisions without clinical review, local validation, and appropriate regulatory oversight.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# ADMIN PAGE
# ═════════════════════════════════════════════════════════════
elif page == "🔐 Admin":
    st.markdown("### SmartDaaS Admin Dashboard")

    # Admin password check
    try:
        admin_password = get_secret("ADMIN_PASSWORD")
        if not st.session_state.get("admin_authenticated"):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("#### Admin Access Required")
                admin_input = st.text_input("Admin password", type="password",
                                             label_visibility="collapsed",
                                             placeholder="Admin password")
                if st.button("Enter Admin", use_container_width=True):
                    if admin_input == admin_password:
                        st.session_state["admin_authenticated"] = True
                        st.rerun()
                    else:
                        st.error("Incorrect admin password.")
            st.stop()
    except KeyError:
        pass  # No admin password set — show admin to anyone authenticated

    # ── SYSTEM STATUS ─────────────────────────────────────
    st.markdown('<p class="section-hdr">System Status</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        db_ok = supabase is not None
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val" style="color:{'#3fb950' if db_ok else '#f85149'}">
            {'✓ Live' if db_ok else '○ Demo'}</div>
            <div class="metric-lbl">Database</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val" style="color:#3fb950">✓ Live</div>
            <div class="metric-lbl">Model</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-val">v1.0</div>
            <div class="metric-lbl">Platform Version</div>
        </div>""", unsafe_allow_html=True)

    if supabase is None:
        st.markdown("""<div class="warn-box">
        ⚠️ <strong>Supabase not configured.</strong> Upload/report logging is disabled.
        Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets to enable persistence.
        See setup instructions below.
        </div>""", unsafe_allow_html=True)

    # ── UPLOAD HISTORY ────────────────────────────────────
    st.markdown('<p class="section-hdr">Upload History</p>', unsafe_allow_html=True)

    if supabase is not None:
        try:
            result = supabase.table("audit_log").select("*").order(
                "event_at", desc=True).limit(100).execute()
            if result.data:
                audit_df = pd.DataFrame(result.data)
                audit_df['event_at'] = pd.to_datetime(
                    audit_df['event_at']).dt.strftime('%Y-%m-%d %H:%M')

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                uploads = audit_df[audit_df['event_type'] == 'upload']
                recals  = audit_df[audit_df['event_type'] == 'recalibration']
                reports = audit_df[audit_df['event_type'] == 'report']
                with c1:
                    st.metric("Total events", len(audit_df))
                with c2:
                    st.metric("Uploads", len(uploads))
                with c3:
                    st.metric("Recalibrations", len(recals))
                with c4:
                    st.metric("Reports", len(reports))

                st.markdown("<br>", unsafe_allow_html=True)

                # Event type filter
                event_filter = st.selectbox(
                    "Filter by event type:",
                    ["All", "upload", "recalibration", "report",
                     "page_view"],
                    key="audit_filter"
                )
                if event_filter != "All":
                    display_df = audit_df[
                        audit_df['event_type'] == event_filter]
                else:
                    display_df = audit_df

                # Select safe display columns
                safe_cols = ['event_at', 'event_type', 'session_id',
                             'n_patients', 'tier', 'dq_grade',
                             'local_auc', 'passed', 'dua_ack',
                             'n_pediatric', 'report_type', 'page']
                show_cols = [c for c in safe_cols
                             if c in display_df.columns]
                st.dataframe(display_df[show_cols],
                             use_container_width=True)
                st.caption(
                    "File names are stored as SHA-256 hashes. "
                    "No patient data is logged. "
                    "Session IDs are random per-session identifiers."
                )

                # Tier distribution if uploads exist
                if len(uploads) > 0 and 'tier' in uploads.columns:
                    st.markdown(
                        '<p class="section-hdr">Upload Tier Distribution'
                        '</p>', unsafe_allow_html=True)
                    tier_counts = uploads['tier'].value_counts()
                    st.bar_chart(tier_counts)

                # Recalibration summary if exists
                if len(recals) > 0 and 'local_auc' in recals.columns:
                    st.markdown(
                        '<p class="section-hdr">Recalibration Summary'
                        '</p>', unsafe_allow_html=True)
                    passed_recals = recals[recals['passed'] == True]
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Attempts", len(recals))
                    with c2:
                        st.metric("Successful", len(passed_recals))
                    with c3:
                        if len(passed_recals) > 0:
                            avg_auc = pd.to_numeric(
                                passed_recals['local_auc'],
                                errors='coerce').mean()
                            st.metric("Mean local AUC",
                                      f"{avg_auc:.3f}" if pd.notna(avg_auc)
                                      else "—")
            else:
                st.info("No audit events recorded yet.")
        except Exception as e:
            # Fall back to old uploads table if audit_log doesn't exist yet
            try:
                result = supabase.table("uploads").select("*").order(
                    "uploaded_at", desc=True).limit(50).execute()
                if result.data:
                    st.dataframe(pd.DataFrame(result.data),
                                 use_container_width=True)
                    st.caption(
                        "Showing legacy uploads table. "
                        "Create audit_log table to enable full audit trail."
                    )
            except Exception:
                st.warning(f"Could not fetch audit log: {e}")
    else:
        st.info("Connect Supabase to see audit trail.")

    # ── REPORT HISTORY ────────────────────────────────────
    st.markdown('<p class="section-hdr">Report History</p>',
                unsafe_allow_html=True)

    if supabase is not None:
        try:
            result = supabase.table("audit_log").select("*").eq(
                "event_type", "report").order(
                "event_at", desc=True).limit(50).execute()
            if result.data:
                reports_df = pd.DataFrame(result.data)
                reports_df['event_at'] = pd.to_datetime(
                    reports_df['event_at']).dt.strftime('%Y-%m-%d %H:%M')
                safe_cols = ['event_at', 'session_id', 'n_patients',
                             'report_type', 'org_type', 'region']
                show_cols = [c for c in safe_cols
                             if c in reports_df.columns]
                st.dataframe(reports_df[show_cols],
                             use_container_width=True)
                st.metric("Total reports generated", len(reports_df))
            else:
                st.info("No reports generated yet.")
        except Exception as e:
            st.warning(f"Could not fetch report history: {e}")
    else:
        st.info("Connect Supabase to see report history.")

    # ── SUPABASE SETUP INSTRUCTIONS ───────────────────────
    st.markdown('<p class="section-hdr">Supabase Setup Instructions</p>',
                unsafe_allow_html=True)
    st.markdown("""
**Step 1 — Create Supabase account**
Go to [supabase.com](https://supabase.com) → Sign up with GitHub → Create new project.

**Step 2 — Create tables**
In Supabase → SQL Editor → New query → paste and run:
```sql
-- Primary audit log table (replaces uploads + reports)
CREATE TABLE audit_log (
    id           BIGSERIAL PRIMARY KEY,
    event_at     TIMESTAMPTZ DEFAULT NOW(),
    event_type   TEXT,
    session_id   TEXT,
    -- Upload fields
    file_hash    TEXT,
    n_patients   INTEGER,
    n_high_risk  INTEGER,
    n_medium_risk INTEGER,
    avg_risk     FLOAT,
    tier         TEXT,
    dq_grade     TEXT,
    art_inferred BOOLEAN,
    n_pediatric  INTEGER,
    dua_ack      BOOLEAN,
    -- Recalibration fields
    n_positive   INTEGER,
    prevalence   FLOAT,
    local_auc    FLOAT,
    cal_method   TEXT,
    passed       BOOLEAN,
    failed_checks TEXT,
    -- Report fields
    org_type     TEXT,
    region       TEXT,
    report_type  TEXT,
    -- Page view fields
    page         TEXT
);

-- Legacy tables (kept for backward compatibility)
CREATE TABLE IF NOT EXISTS uploads (
    id BIGSERIAL PRIMARY KEY,
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    n_patients INTEGER,
    n_high_risk INTEGER,
    n_medium_risk INTEGER,
    avg_risk_score FLOAT,
    data_source TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id BIGSERIAL PRIMARY KEY,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    org_name TEXT,
    n_patients INTEGER,
    report_type TEXT
);
```

**Step 3 — Get your credentials**
In Supabase → Project Settings → API → copy:
- Project URL (looks like `https://xxxx.supabase.co`)
- `anon` public key

**Step 4 — Add to Streamlit secrets**
In Streamlit Cloud → your app → Settings → Secrets → paste:
```toml
APP_PASSWORD = "your_pilot_access_code"
ADMIN_PASSWORD = "your_admin_password"
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your_anon_key"
```

**Step 5 — Redeploy**
Streamlit auto-redeploys when secrets change. Done.
""")

    # Admin logout
    st.markdown("---")
    if st.button("Lock Admin", use_container_width=False):
        st.session_state["admin_authenticated"] = False
        st.rerun()

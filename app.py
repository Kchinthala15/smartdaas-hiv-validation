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
    BASELINE_AUC, OUTCOME_NAME_HINTS, CRITICAL_FEATURES,
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
# ─────────────────────────────────────────────────────────────
# MODEL — imported from model.py
# ─────────────────────────────────────────────────────────────
from model import (
    load_model,
    model, SCALER, CLF, MODEL_AUC, X_DEMO, Y_DEMO, MODEL_OK, MODEL_ERR,
    run_predictions,
    safe_feature_importances,
    compute_shap,
    compute_shap_single,
    generate_sample_csv,
)

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





# ═════════════════════════════════════════════════════════════════════════
# SMARTDAAS FEATURE BLOCK — Outreach Optimiser + Programme Brief + Nav
# ═════════════════════════════════════════════════════════════════════════
# Self-contained. No new dependencies: uses fpdf2 (already in requirements)
# for the PDF brief; the optional AI-wording layer uses `anthropic` only if
# it is installed and an ANTHROPIC_API_KEY is configured — otherwise it is
# silently skipped and the deterministic brief is shown.
# ═════════════════════════════════════════════════════════════════════════

# ── Outreach Optimiser constants ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# OUTREACH OPTIMISER — imported from outreach.py
# ─────────────────────────────────────────────────────────────
from outreach import render_outreach_optimiser
from action_lens import render_action_lens

# ── Restructured three-tier sidebar navigation ──────────────────────────
_NAV_PAGES = [
    "🏠 Home", "❤️ Patient Risk", "🎯 Outreach Optimiser", "🧭 FrontlineLens",
    "📄 Executive Report",
    "🏥 Facility Intelligence", "👥 Cohort Intelligence", "✅ Local Validation",
    "🧬 Model Transparency", "🔍 SHAP Explainability", "📘 Model Info",
    "📁 Sample Data", "🔐 Admin",
]
_NAV_T2_START = 6   # nth-child index of "Facility Intelligence"
_NAV_T3_START = 12  # nth-child index of "Sample Data"


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
    label:nth-child(-n+5) p {{font-weight:500 !important;}}
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
            <div class="metric-val" style="color:#3fb950">0.806</div>
            <div class="metric-lbl">Model AUC</div>
        </div>""", unsafe_allow_html=True)
        st.success("✓ Model loaded")
    else:
        st.error(f"Model error: {MODEL_ERR[:80]}")
        st.info("Ensure cv_results.pkl and prepped_data.pkl are in the repo root.")

    st.markdown("---")
    st.markdown("""<div class="info-box">
        <strong>SmartDaaS v1.0</strong><br>
        <a href="https://github.com/Kchinthala15/smartdaas-hiv-validation" style="color:#21d4fd">Code &amp; methodology</a>
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
        <strong style="color:#21d4fd">23,144 adults on ART</strong> (a Nigerian HIV
        programme — discovery cohort) with local recalibration support for any country context.
        Designed for PEPFAR implementing partners, Global Fund grantees, and national
        HIV programme offices across sub-Saharan Africa.</p>
    </div>""", unsafe_allow_html=True)

    # ── KEY METRICS ───────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    for col, val, lbl in [
        (c1, "0.806", "Temporal Validation AUC"),
        (c2, "0.801", "Cross-Validation AUC"),
        (c3, "23K", "Training Records"),
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
> **Open methodology.** The model, the validation pipeline, and the data handling
> are published in full so the approach can be inspected and reproduced end to end.
> 📎 Code: [github.com/Kchinthala15/smartdaas-hiv-validation](https://github.com/Kchinthala15/smartdaas-hiv-validation)

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
            # ── Step 7b: Refuse to score if a CRITICAL feature is missing ──
            # had_interruption is the model's dominant predictor (importance
            # 0.381) and its training median is 0.0 — the protective value.
            # Imputing it is indistinguishable from zero-filling it: the HIGH
            # tier collapses from ~800 patients to ~17, and ~238 genuinely
            # poor-adherent patients are sent to LOW. Silence is worse than a
            # refusal here, so we refuse.
            _missing_critical = CRITICAL_FEATURES & set(defaulted_feats)
            if _missing_critical:
                _labels = ', '.join(sorted(FEAT_LABELS.get(f, f) for f in _missing_critical))
                st.error(
                    f"**Cannot generate risk scores — required field missing: {_labels}.**\n\n"
                    f"This is the model's strongest predictor. If it is absent, every patient "
                    f"is scored as though they have never interrupted treatment, and the "
                    f"high-risk group all but disappears — the tool would report that your "
                    f"cohort is fine rather than tell you it cannot see.\n\n"
                    f"**To enable scoring**, include one of: `had_interruption` (0/1), "
                    f"`art_interruption`, `treatment_interruptions` (a count), "
                    f"`n_interruptions`, or `treatment_gaps`.\n\n"
                    f"The tier report, data quality screening and validation metadata "
                    f"above remain available for this upload."
                )
                st.stop()

            if defaulted_feats:
                st.warning(
                    f"⚙️ **{len(defaulted_feats)} feature(s) not present in your upload** — "
                    f"imputed with the training-cohort median: "
                    f"{', '.join([FEAT_LABELS.get(f, f) for f in defaulted_feats])}. "
                    f"These patients are scored as an average patient on those fields, "
                    f"which reduces sensitivity: real risk can be masked. Supplying the "
                    f"missing fields will improve accuracy."
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
            "ℹ️ Using baseline model (Nigerian discovery cohort, AUC: 0.806). "
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
    export.insert(3, 'model_auc_temporal', '0.806')
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
> 27,288 patients in a Nigerian HIV programme.
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
    st.markdown('<p class="section-hdr">Key Findings — Facility Analysis</p>', unsafe_allow_html=True)
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
facility-level care. Derived from the discovery cohort (27,288 patients, a Nigerian HIV programme).
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
        Parameters from the facility analysis (Nigerian discovery cohort). Primary HC poor outcome
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
are derived from a Nigerian HIV programme discovery cohort (99.8% of ART initiations 2013–2017) and have not been
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
        (c1, "0.806", "Temporal Validation AUC"),
        (c2, "0.801", "Cross-Validation AUC"),
        (c3, "27.7%", "Sensitivity @ 0.15"),
        (c4, "23,144", "Training Records"),
    ]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Architecture</p>', unsafe_allow_html=True)
    st.markdown("""
- **Model type:** Random Forest Classifier
- **Pipeline:** SimpleImputer (training medians) → StandardScaler → RandomForestClassifier
- **Training data:** 23,144 adults aged 18–100 on ART (a Nigerian HIV programme, Quality of Care dataset — discovery cohort). Poor-adherence prevalence 3.67% (850 patients).
- **Outcome:** poor ART adherence recorded at the patient's most recent visit. Not treatment interruption, not mortality, not a composite.
- **Features:** 15 clinical variables (see column guide on 📁 Sample Data page)
- **Cross-validation:** 10-fold stratified CV, natural class distribution — AUC 0.801 (Brier 0.032)
- **Temporal validation:** trained on patients initiating ART up to September 2016, tested on 6,942 later initiators — AUC 0.806 (95% CI 0.774–0.837, Brier 0.027)
- **Note:** internal and temporal estimates agree (0.801 vs 0.806). That agreement is the point: a model that scores far higher internally than temporally is usually leaking information between training and test.
- **Explainability:** SHAP TreeExplainer with per-patient waterfall charts
""")

    st.markdown('<p class="section-hdr">Important Limitations</p>', unsafe_allow_html=True)
    st.markdown("""
1. **Temporal validation AUC of 0.806** — the honest real-world estimate, measured on patients the model never saw.
2. **Data provenance is not documented** — the Quality of Care dataset is a public deposit with no stated sampling frame, custodian, or collection methodology. Internal evidence (clinical free-text and funding structure) indicates Nigerian programme records. It has not been externally validated on patient-level data from any other health system.
3. **Not validated for clinical decision-making** — local validation is required before deployment in real-world programme environments. Prospective validation strengthens operational confidence.
4. **15 features only** — the model does not capture socioeconomic factors, geographic remoteness, or drug supply chain quality, which are known outcome drivers.
5. **Single outcome** — the model predicts poor ART adherence at the most recent visit. Poor adherence is a leading indicator of treatment interruption, but it is not the same thing, and the model has not been validated against interruption or mortality.
""")

    st.markdown('<p class="section-hdr">Citation</p>', unsafe_allow_html=True)
    st.markdown("""
If you use SmartDaaS in research or programme evaluation, please cite the software:

> Chinthala LK. *SmartDaaS: an explainable risk model for ART adherence in HIV
programme settings.* SmartDaaS LLC, 2026. https://smartdaas.org

**Code repository:** github.com/Kchinthala15/smartdaas-hiv-validation
""")


# ═════════════════════════════════════════════════════════════
# OUTREACH OPTIMISER  (capacity-constrained weekly action plan)
# ═════════════════════════════════════════════════════════════
elif page == "🎯 Outreach Optimiser":
    render_outreach_optimiser(supabase)


# ═════════════════════════════════════════════════════════════
# FRONTLINELENS  (risk -> explanation -> localized frontline action -> audit)
# ═════════════════════════════════════════════════════════════
elif page == "🧭 FrontlineLens":
    render_action_lens(supabase=supabase, log_event=log_event)


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
# ═════════════════════════════════════════════════════════════
# PDF EXECUTIVE REPORT GENERATOR
# ═════════════════════════════════════════════════════════════
elif page == "📄 Executive Report":
    from reports import render_executive_report
    render_executive_report(supabase=supabase, log_report=log_report)

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
    for optimal performance. The temporal AUC (0.806) is the most realistic deployment estimate.
    After local retraining on your historical data, performance typically improves.
    </div>""", unsafe_allow_html=True)

    # ── PERFORMANCE PANEL ─────────────────────────────────
    st.markdown('<p class="section-hdr">Model Performance</p>', unsafe_allow_html=True)

    # ── Block 1: Cross-validation performance ─────────────
    st.markdown("""<div style="background:#1a2030;border:1px solid #21d4fd44;border-radius:8px;
        padding:0.75rem 1rem 0.5rem 1rem;margin-bottom:0.4rem">
        <div style="font-size:0.7rem;color:#21d4fd;font-family:'IBM Plex Mono',monospace;
            text-transform:uppercase;letter-spacing:2px;margin-bottom:0.6rem">
            Internal Performance — 10-fold Stratified Cross-Validation (natural class distribution, 3.67%)
        </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cv_metrics = [
        (c1, "0.801",  "CV AUC",        "#21d4fd"),
        (c2, "23,144", "Adults",        "#21d4fd"),
        (c3, "850",    "Poor Adherence","#21d4fd"),
        (c4, "0.032",  "Brier Score",   "#21d4fd"),
        (c5, "3.67%",  "Prevalence",    "#8b949e"),
    ]
    for col, val, lbl, color in cv_metrics:
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color:{color}">'
                f'{val}</div><div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.72rem;color:#8b949e;padding:0.3rem 0.25rem 0.1rem 0.25rem">
        AUC 0.801 (SD 0.023) across 10 stratified folds of 23,144 adults, evaluated at the
        natural poor-adherence rate of 3.67% (850 patients). Oversampling is applied inside
        each training fold only, so no synthetic patient appears in a fold used for scoring.
    </div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Block 2: Temporal holdout performance ─────────────
    st.markdown("""<div style="background:#1a2010;border:1px solid #e3b34144;border-radius:8px;
        padding:0.75rem 1rem 0.5rem 1rem;margin-bottom:0.4rem">
        <div style="font-size:0.7rem;color:#e3b341;font-family:'IBM Plex Mono',monospace;
            text-transform:uppercase;letter-spacing:2px;margin-bottom:0.6rem">
            Temporal Holdout Performance — Patients Initiating ART After September 2016 (deployment estimate)
        </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    temporal_metrics = [
        (c1, "0.806",  "Temporal AUC",      "#e3b341"),
        (c2, "6,942",  "Held-Out Patients", "#e3b341"),
        (c3, "27.7%",  "Sensitivity",       "#e3b341"),
        (c4, "97.0%",  "Specificity",       "#e3b341"),
        (c5, "0.15",   "Deploy Threshold",  "#e3b341"),
    ]
    for col, val, lbl, color in temporal_metrics:
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color:{color}">'
                f'{val}</div><div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("""<div style="font-size:0.72rem;color:#8b949e;padding:0.3rem 0.25rem 0.1rem 0.25rem">
        Temporal AUC 0.806 (95% CI: 0.774–0.837) on 6,942 patients who started ART after the training window.
        Sensitivity 27.7% and Specificity 97.0% at deployment threshold 0.15; PPV 21.9% against a 3.0% base rate.
        Lowering the threshold to 0.075 raises sensitivity to 52.4% at the cost of flagging 11.6% of the cohort —
        this is a programme capacity decision, not a statistical one.
    </div></div>""", unsafe_allow_html=True)

    # ── WHAT THIS MEANS ───────────────────────────────────
    st.markdown('<p class="section-hdr">What These Numbers Mean</p>', unsafe_allow_html=True)

    st.markdown("""
**Cross-validation AUC (0.801)** is measured with 10-fold stratified CV across 23,144 adults, at
the natural poor-adherence rate of 3.67%. Oversampling is applied inside each training fold only,
so no synthetic patient ever appears in a fold used for scoring.

**Temporal AUC (0.806)** measures how well the model predicts on patients from *later time periods*
it has never seen — trained on those initiating ART up to September 2016, tested on 6,942 who
started after. AUC 0.806 means the model correctly ranks 80.6% of patient pairs (higher-risk above
lower-risk).

**The two agree (+0.006), and that is the finding.** A model that scores far higher on internal
validation than on future patients is usually leaking information between training and test rather
than generalising. Agreement across an internal and a temporal split is the evidence that the
signal is real.

**At the deployment threshold of 0.15**, the model flags 3.7% of patients with Sensitivity 27.7%,
Specificity 97.0% and PPV 21.9% — roughly 1 in 5 flagged patients genuinely has poor adherence,
against a 3.0% base rate. That is about a 7-fold concentration of risk. Lowering the threshold to
0.075 flags 11.6% and catches 52.4% of them. Which point is right depends on how many patients
your outreach team can contact, not on the statistics.

**Context:** the most recent systematic review of ML models for HIV treatment interruption
(Kwarah et al., *BMC Global and Public Health* 2025; 12 models) reports a mean internal AUC of
0.668. Those models predict interruption rather than adherence, so the comparison is indicative
rather than like-for-like.
""")

    # Visual: CV vs Temporal
    fig, ax = plt.subplots(figsize=(8, 3), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    categories = ['Cross-Validation\n(internal)', 'Temporal Validation\n(future patients)', 'Random\n(baseline)']
    values = [0.801, 0.806, 0.5]
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
    The model was trained on a public Nigerian HIV programme dataset (Quality of Care dataset; 23,144 adults after cleaning; ART initiations 2013–2017 for 99.8% of the cohort — discovery cohort). The deposit does not document its sampling frame, custodian, or collection methodology.
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

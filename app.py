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
    try:
        correct_password = get_secret("APP_PASSWORD")
    except Exception:
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
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; text-align: center;
    }
    .login-brand { font-family:'IBM Plex Mono',monospace; font-size:2rem;
        font-weight:600; color:#21d4fd; margin-bottom:0.25rem; }
    .login-sub { color:#8b949e; font-size:0.9rem; margin-bottom:2rem; }
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
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.smartdaas-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d2137 100%);
    border: 1px solid #21d4fd22; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
}
.brand-name { font-family:'IBM Plex Mono',monospace; font-size:2.2rem;
    font-weight:600; color:#21d4fd; letter-spacing:-1px; margin:0; }
.brand-sub { font-size:0.95rem; color:#adbac7; margin-top:0.25rem; font-weight:300; }
.version-tag { display:inline-block; background:#21d4fd22; border:1px solid #21d4fd66;
    color:#21d4fd; font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
    padding:2px 8px; border-radius:4px; margin-top:0.5rem; }
.metric-box { background:#161b22; border:1px solid #444c56; border-radius:8px;
    padding:1rem; text-align:center; margin-bottom:0.5rem; }
.metric-val { font-family:'IBM Plex Mono',monospace; font-size:1.8rem;
    font-weight:600; color:#21d4fd; }
.metric-lbl { font-size:0.75rem; color:#adbac7; text-transform:uppercase; letter-spacing:0.5px; }
.risk-high { background:#2d1115; border:1px solid #f8514933; border-radius:10px;
    padding:1rem; text-align:center; color:#f85149; }
.risk-medium { background:#1c1a0f; border:1px solid #e3b34133; border-radius:10px;
    padding:1rem; text-align:center; color:#e3b341; }
.risk-low { background:#0d1f17; border:1px solid #3fb95033; border-radius:10px;
    padding:1rem; text-align:center; color:#3fb950; }
.risk-number { font-family:'IBM Plex Mono',monospace; font-size:2.5rem; font-weight:600; }
.risk-label { font-size:0.8rem; opacity:0.8; text-transform:uppercase; letter-spacing:1px; }
.section-hdr { font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#21d4fd;
    text-transform:uppercase; letter-spacing:2px; border-bottom:1px solid #21d4fd44;
    padding-bottom:0.5rem; margin-bottom:1rem; margin-top:1.5rem; }
.info-box { background:#161b22; border-left:3px solid #21d4fd; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.85rem; color:#cdd9e5; margin:0.75rem 0; }
.warn-box { background:#1c1208; border-left:3px solid #e3b341; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.85rem; color:#e3b341; margin:0.75rem 0; }
.success-box { background:#0d1f17; border-left:3px solid #3fb950; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.85rem; color:#3fb950; margin:0.75rem 0; }
.template-box { background:#161b22; border:1px solid #21d4fd55; border-radius:8px;
    padding:1.5rem; margin:1rem 0; }
[data-testid="stSidebar"] { background-color:#0d1117; border-right:1px solid #21262d; }
[data-testid="stSidebar"] * { color: #cdd9e5 !important; }
[data-testid="stSidebar"] .section-hdr { color: #21d4fd !important; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
FEATURES = [
    'Age','sex_female','Cd4AtStart','MostRecentCd4Count','CD4_improvement',
    'stage_start_num','WeightAtStart','weight_change','BMI_start','days_to_ART',
    'had_interruption','opp_infection','side_effects','tb_positive','stage_worsened'
]

# Alternative column name mappings for robustness
# Keys = what users might call them, Values = what the model expects
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
    # Age
    'age_years': 'Age',
    'age_at_art': 'Age',
    'age_at_enrollment': 'Age',
    'patient_age_years': 'Age',

    # Sex
    'sex_at_birth': 'sex_female',
    'biological_sex': 'sex_female',
    'patient_sex': 'sex_female',
    'gender_female': 'sex_female',

    # CD4
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

    # WHO Stage - Kenya NASCOP / Uganda DHIS2 / Malawi HMIS variants
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
    'tb': 'tb_positive',
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
    # These aliases enable population-based HIV Impact Assessment (PHIA) datasets
    # and similar population surveys to flow through normalize_columns().
    #
    # IMPORTANT VALIDATION CAVEATS — read before using PHIA data:
    #   1. cd4count in PHIA = most-recent survey-measured CD4, NOT CD4 at ART start.
    #      Maps to MostRecentCd4Count only. Do NOT use as Cd4AtStart baseline.
    #   2. tbdiagn = self-reported TB diagnosis ("ever told by health worker you had TB").
    #      Not lab-confirmed. Different sensitivity/specificity from clinical TB records.
    #   3. arvinterr = direct treatment interruption variable (available in PHIA 2020+
    #      waves only: MPHIA 2020-21, THIS 2022-23, UPHIA 2020-21).
    #   4. arvsmissdays>0 is a proxy for had_interruption — handled in
    #      preprocess_phia_compatible() below, not here.
    #   5. PHIA supports validation of the core predictive signal (8/15 features).
    #      Full 15-feature model validation requires longitudinal EMR/programme data.

    # CD4 — PHIA biomarker variable (most-recent measured CD4)
    'cd4count': 'MostRecentCd4Count',          # Primary: direct PHIA biomarker variable
    'cd4cat': 'MostRecentCd4Count',             # Fallback only: ordinal CD4 category
                                                # Requires midpoint approximation:
                                                # <200→100, 200-349→275, 350-499→425, 500+→650
                                                # Flag rows using this with cd4_from_cat==1

    # Treatment interruption — PHIA 2020+ waves direct label
    'arvinterr': 'had_interruption',            # Direct interruption variable (newer waves)

    # TB — self-reported diagnosis (NOT lab-confirmed)
    'tbdiagn': 'tb_positive',                   # Primary: "ever told you had TB"
    'tbclinvisit': 'tb_positive',               # Fallback only: TB clinic visit proxy
                                                # Use where tbdiagn absent (e.g. UPHIA 2020)
}

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

COST_PER_POOR_OUTCOME = 1850  # USD (Menzies et al. 2011)
BASELINE_THRESHOLD = 0.70    # Default risk threshold — overridden by local calibration


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
            f'You may upload patient data.</div>',
            unsafe_allow_html=True
        )
        return True

    st.markdown("""<div style="border:1px solid #f0a500;border-radius:8px;
        padding:16px 20px;background:#1c1a10;margin:12px 0">
        <div style="color:#f0a500;font-weight:700;font-size:0.95rem;
            margin-bottom:12px">
            ⚠️ Data Governance Acknowledgement Required
        </div>
        <p style="color:#adbac7;font-size:0.85rem;margin:0 0 14px 0">
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

def normalize_columns(df):
    """
    Attempt to map uploaded column names to expected feature names.
    Returns (df_mapped, list_of_missing_cols, list_of_mappings_applied).
    Case-insensitive matching + alias lookup.
    """
    # Step 1: exact match first
    col_map = {}
    for col in df.columns:
        if col in FEATURES:
            col_map[col] = col

    # Step 2: case-insensitive + alias lookup for remaining
    needed = [f for f in FEATURES if f not in col_map.values()]
    for feat in needed:
        for col in df.columns:
            col_lower = col.lower().strip().replace(' ', '_')
            # Direct case-insensitive
            if col_lower == feat.lower():
                col_map[col] = feat
                break
            # Alias lookup
            if col_lower in COLUMN_ALIASES and COLUMN_ALIASES[col_lower] == feat:
                col_map[col] = feat
                break

    # Build renamed df
    rename_dict = {k: v for k, v in col_map.items() if k != v}
    df_mapped = df.rename(columns=rename_dict)

    mappings_applied = [(k, v) for k, v in rename_dict.items()]

    # Sex recode: handle M/F, Male/Female text strings -> binary 0/1
    # Partners commonly export sex as text - auto-recode so model gets numeric
    if 'sex_female' in df_mapped.columns:
        col = df_mapped['sex_female']
        numeric_attempt = pd.to_numeric(col, errors='coerce')
        # If conversion fails for most values, it's a text column
        if numeric_attempt.isna().sum() > len(col) * 0.5:
            col_str = col.astype(str).str.strip().str.lower()
            df_mapped['sex_female'] = col_str.map({
                'f': 1.0, 'female': 1.0, 'woman': 1.0, 'w': 1.0, '1': 1.0,
                'm': 0.0, 'male': 0.0, 'man': 0.0, '0': 0.0,
            })
            mappings_applied.append(('sex (M/F text)', 'sex_female (0=Male, 1=Female)'))

    missing = [f for f in FEATURES if f not in df_mapped.columns]

    return df_mapped, missing, mappings_applied


# ─────────────────────────────────────────────────────────────
# PHIA / POPULATION SURVEY PRE-PROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_phia_compatible(df):
    """
    Pre-processing pipeline for PHIA and compatible population-survey datasets.
    Must be called BEFORE normalize_columns().

    Handles transformations that require logic beyond simple column renaming:
      1. sex_female recode from PHIA gender coding (1=Male, 2=Female)
      2. had_interruption proxy from arvsmissdays (missed doses > 0)
      3. opp_infection composite from TB + STI proxy variables
      4. days_to_ART approximation from month/year fields
      5. Explicit flagging of all derived variables

    Returns (df_processed, derivation_log) where derivation_log is a list
    of strings describing every transformation applied, for transparency.

    VALIDATION CAVEATS preserved in derivation_log:
      - cd4count = most-recent survey CD4, not ART-start baseline
      - tbdiagn = self-reported TB, not lab-confirmed
      - days_to_ART = low-confidence derived feature (~±15 day error)
      - arvsmissdays proxy ≠ clinical treatment interruption
      - PHIA supports core signal validation only (8/15 SmartDaaS features)
    """
    import numpy as np
    df = df.copy()
    log = []
    cols = {c.lower().strip(): c for c in df.columns}

    # ── 1. sex_female recode ──────────────────────────────────────────────────
    # PHIA gender: 1=Male, 2=Female → SmartDaaS sex_female: 0=Male, 1=Female
    if 'gender' in cols and 'sex_female' not in cols:
        gc = cols['gender']
        gender_num = pd.to_numeric(df[gc], errors='coerce')
        df['sex_female'] = (gender_num == 2).astype(float)
        df.loc[gender_num.isna(), 'sex_female'] = np.nan
        log.append("sex_female: recoded from gender (PHIA: 1=Male→0, 2=Female→1)")

    # ── 2. had_interruption proxy from arvsmissdays ───────────────────────────
    # Use only if arvinterr (direct label) is absent or all-null.
    # arvinterr is handled via COLUMN_ALIASES rename; check post-rename name.
    has_direct = 'arvinterr' in cols
    has_proxy  = 'arvsmissdays' in cols
    if has_proxy and not has_direct and 'had_interruption' not in cols:
        mc = cols['arvsmissdays']
        miss = pd.to_numeric(df[mc], errors='coerce')
        df['had_interruption'] = (miss > 0).astype(float)
        df.loc[miss.isna(), 'had_interruption'] = np.nan
        df['_derived_interruption_proxy'] = 1  # flag column
        log.append(
            "had_interruption: PROXY derived from arvsmissdays>0. "
            "CAVEAT: self-reported missed doses, top-coded at 4. "
            "Not equivalent to clinical treatment interruption. "
            "Direct label (arvinterr) available in PHIA 2020+ waves only."
        )
    elif has_direct:
        log.append(
            "had_interruption: direct label (arvinterr) present — "
            "alias mapping will handle rename. No proxy needed."
        )

    # ── 3. opp_infection composite ────────────────────────────────────────────
    # Composite from TB diagnosis + active syphilis + STI diagnosis.
    # None of these individually equals clinical OI — composite is a proxy.
    if 'opp_infection' not in cols:
        sources_used = []
        oi = pd.Series(False, index=df.index, dtype=bool)
        any_source = False

        for src_key, src_label in [
            ('tbdiagn', 'tbdiagn==1'),
            ('activesyphilis', 'activesyphilis==1'),
            ('stddiag', 'stddiag==1'),
        ]:
            if src_key in cols:
                val = pd.to_numeric(df[cols[src_key]], errors='coerce')
                oi = oi | (val == 1).fillna(False)
                sources_used.append(src_label)
                any_source = True

        if any_source:
            df['opp_infection'] = oi.astype(float)
            df['_derived_opp_infection_composite'] = 1  # flag column
            log.append(
                f"opp_infection: COMPOSITE derived from [{', '.join(sources_used)}]. "
                "CAVEAT: narrower than clinical OI definition. "
                "tbdiagn = self-reported TB (not lab-confirmed). "
                "Composite proxy only."
            )

    # ── 4. days_to_ART approximation ─────────────────────────────────────────
    # Derived from first ART month/year and first HIV+ test month/year.
    # Month-year resolution only → ±15 day approximation error.
    # LOW-CONFIDENCE DERIVED FEATURE — flag explicitly.
    art_m = 'arvftm' in cols
    art_y = 'arvfty' in cols
    dx_m  = 'hivtfposm' in cols
    dx_y  = 'hivtfposy' in cols

    if art_m and art_y and dx_m and dx_y and 'days_to_ART' not in cols:
        arv_months = (
            pd.to_numeric(df[cols['arvfty']], errors='coerce') * 12 +
            pd.to_numeric(df[cols['arvftm']], errors='coerce')
        )
        dx_months = (
            pd.to_numeric(df[cols['hivtfposy']], errors='coerce') * 12 +
            pd.to_numeric(df[cols['hivtfposm']], errors='coerce')
        )
        days = ((arv_months - dx_months) * 30.44).round(0)
        df['days_to_ART'] = days.clip(0, 3650)
        df['_derived_days_to_art'] = 1  # flag column
        log.append(
            "days_to_ART: LOW-CONFIDENCE DERIVED from arvftm/fty and hivtfposm/y. "
            "Month-year resolution only — mid-month (day 15) assumed → ~±15 day error. "
            "Do not treat as equivalent to EMR-recorded ART initiation date. "
            "Rows flagged with _derived_days_to_art==1."
        )

    # ── 5. Persistent validation caveats ─────────────────────────────────────
    # These are appended to the log regardless of what was derived,
    # so callers always see the full caveat set.
    log.append(
        "PHIA VALIDATION SCOPE: This dataset supports validation of the core "
        "SmartDaaS predictive signal (up to 8 of 15 features). "
        "7 features are structurally absent from PHIA (CD4_improvement, "
        "stage_start_num, WeightAtStart, weight_change, BMI_start, "
        "side_effects, stage_worsened). "
        "Full 15-feature model validation requires longitudinal programme data."
    )
    if 'cd4count' in cols:
        log.append(
            "CD4 CAVEAT: cd4count in PHIA = most-recent survey-measured CD4, "
            "NOT CD4 at ART initiation. Maps to MostRecentCd4Count only. "
            "Standardised difference vs training cohort baseline CD4 is ~0.72 SD."
        )

    return df, log


# ─────────────────────────────────────────────────────────────
# VALIDATION METADATA LAYER
# ─────────────────────────────────────────────────────────────

def build_validation_metadata(df_raw, df_mapped, mappings_applied,
                               missing_features, derivation_log=None,
                               dq_results=None, tier=None):
    """
    Build a structured validation metadata object that tracks every
    inference, proxy, imputation, and derivation applied during processing.

    Returns a dict with the following structure:
    {
        "source_type": str,          # 'original' | 'phia_compatible' | 'mixed'
        "features_original": int,    # features present directly in upload
        "features_derived": int,     # features derived/proxied during processing
        "features_imputed": int,     # features filled with imputed defaults
        "features_missing": int,     # features structurally absent
        "derived_fields": [          # detail for each derived/proxied field
            {
                "feature": str,
                "method": str,       # 'alias_rename' | 'proxy' | 'composite' |
                                     # 'derived_approximate' | 'imputed_default'
                "source_fields": [],
                "confidence": str,   # 'high' | 'moderate' | 'low'
                "caveat": str
            }
        ],
        "warnings": [],              # list of caveat strings for UI display
        "phia_scope_note": str,      # canonical validation scope statement
        "audit_trail": []            # ordered log of all transformations
    }

    Designed for:
      - UI display in the platform (Data Quality / Model Transparency pages)
      - Pilot partner due diligence review
      - Grant/funder audit trail
      - Journal supplement material
    """
    import numpy as np

    meta = {
        "source_type": "original",
        "features_original": 0,
        "features_derived": 0,
        "features_imputed": 0,
        "features_missing": len(missing_features) if missing_features else 0,
        "derived_fields": [],
        "warnings": [],
        "phia_scope_note": "",
        "audit_trail": derivation_log or [],
    }

    cols_raw   = {c.lower().strip() for c in df_raw.columns}
    cols_mapped = set(df_mapped.columns)

    # ── Classify each model feature ──────────────────────────────────────────
    phia_proxy_flags = {
        '_derived_interruption_proxy',
        '_derived_opp_infection_composite',
        '_derived_days_to_art',
    }
    is_phia = any(f in df_mapped.columns for f in phia_proxy_flags)

    # Rename map: original col → model feature
    rename_map = {orig: mapped for orig, mapped in (mappings_applied or [])}

    for feat in FEATURES:
        if feat not in cols_mapped:
            # Structurally missing — already counted above
            continue

        # Check if it arrived via alias rename
        arrived_via_alias = feat in rename_map.values()

        # Check proxy flags
        if feat == 'had_interruption' and '_derived_interruption_proxy' in cols_mapped:
            meta["features_derived"] += 1
            meta["derived_fields"].append({
                "feature": "had_interruption",
                "method": "proxy",
                "source_fields": ["arvsmissdays"],
                "confidence": "moderate",
                "caveat": (
                    "Proxy derived from arvsmissdays>0. Self-reported missed doses "
                    "(top-coded at 4). Not equivalent to clinically documented "
                    "treatment interruption. Direct label (arvinterr) available "
                    "in PHIA 2020+ waves only."
                )
            })
            meta["warnings"].append(
                "had_interruption: PROXY (arvsmissdays>0) — not equivalent to "
                "clinical interruption record."
            )

        elif feat == 'opp_infection' and '_derived_opp_infection_composite' in cols_mapped:
            sources = []
            for src in ['tbdiagn', 'activesyphilis', 'stddiag']:
                if src in cols_raw:
                    sources.append(src)
            meta["features_derived"] += 1
            meta["derived_fields"].append({
                "feature": "opp_infection",
                "method": "composite",
                "source_fields": sources,
                "confidence": "moderate",
                "caveat": (
                    f"Composite from [{', '.join(sources)}]. Narrower than clinical OI "
                    "definition. tbdiagn = self-reported TB (not lab-confirmed)."
                )
            })
            meta["warnings"].append(
                "opp_infection: COMPOSITE proxy (TB + STI variables). "
                "Narrower than clinical OI. tbdiagn is self-reported."
            )

        elif feat == 'days_to_ART' and '_derived_days_to_art' in cols_mapped:
            meta["features_derived"] += 1
            meta["derived_fields"].append({
                "feature": "days_to_ART",
                "method": "derived_approximate",
                "source_fields": ["arvftm", "arvfty", "hivtfposm", "hivtfposy"],
                "confidence": "low",
                "caveat": (
                    "LOW-CONFIDENCE DERIVED. Month-year resolution only — "
                    "mid-month (day 15) assumed, introducing ~±15 day error. "
                    "Do not treat as equivalent to EMR-recorded ART initiation date."
                )
            })
            meta["warnings"].append(
                "days_to_ART: LOW-CONFIDENCE DERIVED from month/year fields "
                "(±15 day approximation). Not equivalent to EMR ART initiation date."
            )

        elif feat == 'tb_positive' and arrived_via_alias:
            # Arrived via tbdiagn → tb_positive alias
            src = next((o for o, m in rename_map.items() if m == 'tb_positive'), 'tbdiagn')
            meta["features_derived"] += 1
            meta["derived_fields"].append({
                "feature": "tb_positive",
                "method": "alias_rename",
                "source_fields": [src],
                "confidence": "moderate",
                "caveat": (
                    f"Mapped from '{src}'. "
                    "CAVEAT: self-reported TB diagnosis ('ever told by health worker "
                    "you had TB'). Not lab-confirmed. Different sensitivity/specificity "
                    "from clinical TB documentation."
                )
            })
            meta["warnings"].append(
                f"tb_positive: mapped from '{src}' — SELF-REPORTED TB diagnosis, "
                "not lab-confirmed."
            )

        elif feat == 'MostRecentCd4Count' and arrived_via_alias:
            src = next((o for o, m in rename_map.items() if m == 'MostRecentCd4Count'), None)
            if src == 'cd4count':
                meta["features_derived"] += 1
                meta["derived_fields"].append({
                    "feature": "MostRecentCd4Count",
                    "method": "alias_rename",
                    "source_fields": ["cd4count"],
                    "confidence": "high",
                    "caveat": (
                        "Mapped from cd4count (PHIA biomarker). "
                        "CAVEAT: PHIA cd4count = most-recent survey-measured CD4, "
                        "NOT CD4 at ART initiation. Standardised difference vs "
                        "training cohort baseline CD4 is ~0.72 SD. "
                        "Do not populate Cd4AtStart with this value."
                    )
                })
                meta["warnings"].append(
                    "MostRecentCd4Count: from PHIA cd4count — "
                    "most-recent survey CD4, NOT ART-start baseline. "
                    "Std diff vs training cohort: ~0.72 SD."
                )
            elif src == 'cd4cat':
                meta["features_derived"] += 1
                meta["derived_fields"].append({
                    "feature": "MostRecentCd4Count",
                    "method": "derived_approximate",
                    "source_fields": ["cd4cat"],
                    "confidence": "low",
                    "caveat": (
                        "Mapped from cd4cat (ordinal category) using midpoint "
                        "approximation: <200→100, 200-349→275, 350-499→425, 500+→650. "
                        "FALLBACK only. Treat as low-confidence CD4 estimate."
                    )
                })
                meta["warnings"].append(
                    "MostRecentCd4Count: from cd4cat ordinal category — "
                    "FALLBACK midpoint approximation. Low-confidence CD4 estimate."
                )
            else:
                meta["features_original"] += 1

        elif arrived_via_alias:
            # Standard alias rename — original data, just renamed
            meta["features_original"] += 1
            meta["derived_fields"].append({
                "feature": feat,
                "method": "alias_rename",
                "source_fields": [k for k, v in rename_map.items() if v == feat],
                "confidence": "high",
                "caveat": ""
            })

        elif dq_results and feat in dq_results.get('missing', {}):
            # Feature was present but partially/fully imputed by DQ step
            imp_info = dq_results['missing'][feat]
            meta["features_imputed"] += 1
            meta["derived_fields"].append({
                "feature": feat,
                "method": "imputed_default",
                "source_fields": [],
                "confidence": "low",
                "caveat": (
                    f"Imputed with {imp_info.get('impute_method','column median')} "
                    f"(value: {imp_info.get('impute_val','N/A')}) for "
                    f"{imp_info.get('pct_missing','?')}% missing values."
                )
            })

        else:
            meta["features_original"] += 1

    # ── Source type classification ────────────────────────────────────────────
    n_derived = meta["features_derived"]
    n_orig    = meta["features_original"]
    if n_derived == 0:
        meta["source_type"] = "original"
    elif n_orig == 0:
        meta["source_type"] = "phia_compatible"
    else:
        meta["source_type"] = "mixed"

    # ── PHIA scope note ───────────────────────────────────────────────────────
    n_structurally_absent = sum(
        1 for f in ['CD4_improvement','stage_start_num','WeightAtStart',
                    'weight_change','BMI_start','side_effects','stage_worsened']
        if f in (missing_features or [])
    )
    if n_structurally_absent >= 4 or is_phia:
        meta["phia_scope_note"] = (
            "SmartDaaS demonstrated external consistency of its core predictive "
            "signal across independent PHIA populations, while full model validation "
            "will require longitudinal programme datasets with complete feature "
            "availability and target outcomes. "
            f"{15 - meta['features_missing']} of 15 SmartDaaS features are available "
            f"in this upload ({meta['features_missing']} structurally absent)."
        )

    return meta


def render_validation_metadata(meta):
    """
    Render the validation metadata dict as a structured Streamlit UI component.
    Call after build_validation_metadata(), before or after DQ report.
    Designed for the Data Quality / Patient Risk pages.
    """
    if not meta:
        return

    has_derived   = meta["features_derived"] > 0
    has_imputed   = meta["features_imputed"] > 0
    has_warnings  = len(meta["warnings"]) > 0
    has_scope     = bool(meta.get("phia_scope_note"))

    # Only render if there's something noteworthy to show
    if not (has_derived or has_imputed or has_warnings or has_scope):
        return

    with st.expander(
        f"\U0001f50d Validation Metadata — "
        f"{meta['features_original']} original · "
        f"{meta['features_derived']} derived/proxied · "
        f"{meta['features_imputed']} imputed · "
        f"{meta['features_missing']} absent",
        expanded=has_warnings
    ):
        # Summary row
        cols = st.columns(4)
        cols[0].metric("Original features",  meta["features_original"])
        cols[1].metric("Derived / proxied",   meta["features_derived"],
                       delta=None if meta["features_derived"]==0 else "review caveats",
                       delta_color="off")
        cols[2].metric("Imputed (DQ fill)",   meta["features_imputed"],
                       delta=None if meta["features_imputed"]==0 else "low confidence",
                       delta_color="off")
        cols[3].metric("Structurally absent", meta["features_missing"],
                       delta=None if meta["features_missing"]==0 else "null-filled",
                       delta_color="off")

        st.markdown("---")

        # Scope note
        if has_scope:
            st.info(f"\U0001f4cb **Validation scope:** {meta['phia_scope_note']}")

        # Derived fields table
        if meta["derived_fields"]:
            st.markdown("**Feature provenance detail:**")
            conf_icons = {"high": "\U00002705", "moderate": "\U000026a0", "low": "\U0001f7e1"}
            for item in meta["derived_fields"]:
                if not item.get("caveat"):
                    continue  # skip clean alias renames with no caveat
                icon = conf_icons.get(item["confidence"], "\u2139\ufe0f")
                src  = ", ".join(f"`{s}`" for s in item["source_fields"]) if item["source_fields"] else "—"
                st.markdown(
                    f"{icon} **`{item['feature']}`** "
                    f"— method: *{item['method']}* "
                    f"— source: {src}  \n"
                    f"<small style='color:#8b949e'>{item['caveat']}</small>",
                    unsafe_allow_html=True
                )

        # Warnings
        if has_warnings:
            st.markdown("---")
            st.markdown("**Caveats for clinical interpretation:**")
            for w in meta["warnings"]:
                st.warning(w)

        # Audit trail
        if meta.get("audit_trail"):
            with st.expander("\U0001f4cb Full derivation audit trail", expanded=False):
                for i, entry in enumerate(meta["audit_trail"], 1):
                    st.markdown(f"**{i}.** {entry}")


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

ART_INFERENCE_TRIGGERS_LOWER = {
    'art_start_date', 'date_art_started', 'art_initiation_date',
    'dateartstarted', 'regimen_at_start', 'regimenAtStart'.lower(),
    'current_regimen', 'art_regimen', 'arvs',
    'days_to_art', 'days_to_ART'.lower(), 'diagnosis_to_art',
    'cd4atstart', 'cd4_at_start', 'Cd4AtStart'.lower(),
    'stage_start_num', 'who_stage', 'clinical_stage',
    # PHIA / population-survey ART indicators
    # Canonical variable names present across PHIA datasets
    'arvscurrent',          # "Are you currently taking ARVs?" (9/9 PHIA ind)
    'arvstakenev',          # "Have you ever taken ARVs?" (8/9 PHIA ind)
    'arvftm',               # Month first started ARVs (9/9 PHIA ind)
    'arvfty',               # Year first started ARVs (9/9 PHIA ind)
    'arvinterr',            # Treatment interruption (newer-wave PHIA ind)
    'arvsmissdays',         # Missed ARV doses (9/9 PHIA ind)
    'artselfreported',      # Self-reported ART status (PHIA bio)
    'artinitiated12months', # ART initiated past 12 months (PHIA)
}


def detect_art_status(df_original):
    cols_lower = {c.lower().strip().replace(' ', '_') for c in df_original.columns}
    explicit_names = {
        'art_status', 'on_art', 'receiving_art', 'art',
        'waspatientreceivingarv', 'was_patient_receiving_arv', 'arv_status'
    }
    if cols_lower & explicit_names:
        return True, False, ""
    matches = cols_lower & ART_INFERENCE_TRIGGERS_LOWER
    if matches:
        shown = sorted(matches)[:3]
        ellipsis = '...' if len(matches) > 3 else ''
        note = (
            f"ART status was inferred from ART-related clinical variables "
            f"({', '.join(shown)}{ellipsis}) because no explicit ART status "
            f"column was detected."
        )
        return False, True, note
    return False, False, (
        "No ART status column or ART-related variables detected. "
        "This upload cannot be confirmed as an ART patient cohort. "
        "Please add an 'art_status' column (1=on ART, 0=not on ART) or include "
        "ART clinical variables such as 'art_start_date', 'cd4_at_start', or 'days_to_art'."
    )


def detect_tier(df_mapped, art_confirmed, art_inferred):
    available = set(df_mapped.columns)
    if not art_confirmed and not art_inferred:
        return 'INSUFFICIENT', [], list(TIER_CORE_REQUIRED), [], [], [
            "Upload does not contain confirmed ART patient data. Risk scoring is not possible."
        ]
    missing_core = [f for f in TIER_CORE_REQUIRED if f not in available]
    if missing_core:
        return 'INSUFFICIENT', [], missing_core, [], [], [
            f"Missing required variables: {', '.join(missing_core)}. "
            "Age and sex are required for any analysis."
        ]
    standard_present = [f for f in TIER_STANDARD_QUALIFYING if f in available]
    enhanced_present = [f for f in TIER_ENHANCED_QUALIFYING if f in available]
    all_present = (
        [f for f in TIER_CORE_REQUIRED if f in available]
        + standard_present + enhanced_present
    )
    if len(standard_present) >= TIER_STANDARD_MIN and len(enhanced_present) >= TIER_ENHANCED_MIN:
        tier = 'ENHANCED'
    elif len(standard_present) >= TIER_STANDARD_MIN:
        tier = 'STANDARD'
    else:
        tier = 'CORE'
    return tier, all_present, [], standard_present, enhanced_present, []


def check_pediatric_patients(df_mapped):
    if 'Age' not in df_mapped.columns:
        return []
    try:
        ages = pd.to_numeric(df_mapped['Age'], errors='coerce')
        return list(df_mapped[ages < 15].index)
    except Exception:
        return []


def render_tier_report(tier, present, missing_core, standard_present,
                       enhanced_present, art_confirmed, art_inferred,
                       art_note, pediatric_indices, df_mapped):
    TIER_COLOURS = {
        'ENHANCED': '#21d4fd',
        'STANDARD': '#f0a500',
        'CORE': '#8b949e',
        'INSUFFICIENT': '#f85149'
    }
    TIER_LABELS = {
        'ENHANCED': 'Enhanced Tier — Full Analysis Available',
        'STANDARD': 'Standard Tier — Partial Feature Availability',
        'CORE': 'Core Tier — Cohort Characterisation Only',
        'INSUFFICIENT': 'Insufficient Data — Cannot Proceed'
    }
    TIER_CAPABILITIES = {
        'ENHANCED': [
            "Full 15-feature patient risk scores",
            "SHAP explainability per patient",
            "Full cohort intelligence dashboard",
            "IeDEA MUD regional aggregate contextual benchmarks",
            "Full executive report",
            "Intervention recommendations",
        ],
        'STANDARD': [
            "Risk estimates generated using partial feature availability — prediction confidence and stability may vary depending on which clinical variables are present",
            "Partial SHAP explainability",
            "Cohort intelligence dashboard",
            "IeDEA MUD regional aggregate contextual benchmarks",
            "Standard executive report",
            "Intervention recommendations (reduced specificity)",
        ],
        'CORE': [
            "Patient risk scores — NOT available (insufficient clinical variables)",
            "SHAP explainability — NOT available",
            "Basic cohort characterisation (age, sex distribution)",
            "IeDEA MUD regional aggregate contextual benchmarks",
            "Limited executive report (population summary only)",
            "To unlock risk scoring: add CD4, WHO stage, TB status, and days to ART",
        ],
        'INSUFFICIENT': [
            "No analysis available",
            "Please review missing variables and re-upload",
        ]
    }
    colour = TIER_COLOURS.get(tier, '#8b949e')
    label = TIER_LABELS.get(tier, tier)
    caps = TIER_CAPABILITIES.get(tier, [])
    tier_icons = {
        'ENHANCED': 'success', 'STANDARD': 'warning',
        'CORE': 'info', 'INSUFFICIENT': 'error'
    }
    getattr(st, tier_icons.get(tier, 'info'))(
        f"**Data Tier Detected: {label}**"
    )
    if art_inferred and art_note:
        st.warning(f"ℹ️ {art_note}")
    elif not art_confirmed and not art_inferred and art_note:
        st.error(f"❌ {art_note}")
    st.markdown("**What this upload enables:**")
    icon_map = {'ENHANCED': '✅', 'STANDARD': '⚡', 'CORE': '📊', 'INSUFFICIENT': '❌'}
    icon = icon_map.get(tier, '•')
    for cap in caps:
        if 'NOT available' in cap or 'not available' in cap or 'No analysis' in cap:
            st.markdown(f"❌ {cap}")
        elif 'unlock' in cap.lower() or 'To ' in cap:
            st.markdown(f"💡 {cap}")
        else:
            st.markdown(f"{icon} {cap}")
    if missing_core:
        st.error(f"**Missing required variables:** {', '.join(missing_core)}")
    if present:
        with st.expander("Variables detected in your upload", expanded=False):
            cols = st.columns(2)
            half = len(present) // 2
            with cols[0]:
                for f in present[:half]:
                    st.markdown(f"✅ {FEAT_LABELS.get(f, f)}")
            with cols[1]:
                for f in present[half:]:
                    st.markdown(f"✅ {FEAT_LABELS.get(f, f)}")
    missing_standard = [f for f in TIER_STANDARD_QUALIFYING if f not in standard_present]
    missing_enhanced = [f for f in TIER_ENHANCED_QUALIFYING if f not in enhanced_present]
    if tier in ('CORE', 'STANDARD') and (missing_standard or missing_enhanced):
        with st.expander("Variables that would upgrade your tier", expanded=False):
            if missing_standard:
                st.markdown("**To reach Standard tier, add:**")
                for f in missing_standard:
                    st.markdown(f"- {FEAT_LABELS.get(f, f)} (`{f}`)")
            if tier == 'STANDARD' and missing_enhanced:
                st.markdown("**To reach Enhanced tier, also add:**")
                for f in missing_enhanced:
                    st.markdown(f"- {FEAT_LABELS.get(f, f)} (`{f}`)")
    if pediatric_indices:
        n_ped = len(pediatric_indices)
        st.warning(
            f"**{n_ped} pediatric patient{'s' if n_ped > 1 else ''} detected** "
            f"(age < 15). This model was trained on patients aged 15 and above. "
            f"Risk scores for these patients are not validated for pediatric HIV care "
            f"and will be flagged individually. Clinical interpretation by a qualified "
            f"paediatric HIV clinician is required."
        )
    return tier != 'INSUFFICIENT'


# ─────────────────────────────────────────────────────────────
# DATA QUALITY SCREENING — Group 5
# ─────────────────────────────────────────────────────────────

# Valid ranges per feature — used for out-of-range detection
FEATURE_VALID_RANGES = {
    'Age':               (5, 100),
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


def run_data_quality_screening(df_mapped, available_features):
    """
    Run all data quality checks on the uploaded dataframe.
    Returns a structured results dict.
    """
    results = {
        'missing': {},
        'out_of_range': {},
        'duplicates': None,
        'constant_columns': [],
        'grade': None,
        'grade_reasons': [],
        'deductions': 0,
    }

    n_rows = len(df_mapped)
    if n_rows == 0:
        results['grade'] = 'D'
        results['grade_reasons'] = ['Upload contains no rows.']
        return results

    # ── 1. Missing value analysis ─────────────────────────
    total_missing_pct = 0
    high_importance_missing_pct = 0
    n_high_importance_checked = 0

    for feat in available_features:
        if feat not in df_mapped.columns:
            continue
        col = pd.to_numeric(df_mapped[feat], errors='coerce')
        n_missing = col.isnull().sum()
        pct_missing = n_missing / n_rows * 100
        if n_missing > 0:
            impute_val = col.median() if feat not in {
                'sex_female', 'had_interruption', 'opp_infection',
                'side_effects', 'tb_positive', 'stage_worsened'
            } else col.mode().iloc[0] if len(col.mode()) > 0 else 0
            impute_method = 'column median' if feat not in {
                'sex_female', 'had_interruption', 'opp_infection',
                'side_effects', 'tb_positive', 'stage_worsened'
            } else 'column mode'
            results['missing'][feat] = {
                'n_missing': int(n_missing),
                'pct_missing': round(pct_missing, 1),
                'impute_val': round(float(impute_val), 1) if pd.notna(impute_val) else 0,
                'impute_method': impute_method,
                'high_importance': feat in HIGH_IMPORTANCE_FEATURES,
            }
            total_missing_pct += pct_missing
            if feat in HIGH_IMPORTANCE_FEATURES:
                high_importance_missing_pct += pct_missing
                n_high_importance_checked += 1

    avg_missing = total_missing_pct / len(available_features) if available_features else 0
    avg_hi_missing = (high_importance_missing_pct / n_high_importance_checked
                      if n_high_importance_checked > 0 else 0)

    # ── 2. Out-of-range detection ─────────────────────────
    n_severe_range = 0
    for feat in available_features:
        if feat not in df_mapped.columns or feat not in FEATURE_VALID_RANGES:
            continue
        col = pd.to_numeric(df_mapped[feat], errors='coerce').dropna()
        lo, hi = FEATURE_VALID_RANGES[feat]
        out_mask = (col < lo) | (col > hi)
        n_out = int(out_mask.sum())
        if n_out > 0:
            pct_out = round(n_out / n_rows * 100, 1)
            out_vals = col[out_mask]
            results['out_of_range'][feat] = {
                'n_out': n_out,
                'pct_out': pct_out,
                'valid_range': f'{lo}–{hi}',
                'min_observed': round(float(out_vals.min()), 1),
                'max_observed': round(float(out_vals.max()), 1),
                'severity': 'severe' if pct_out > 5 else 'minor',
            }
            if pct_out > 5:
                n_severe_range += 1

    # ── 3. Duplicate patient ID detection ────────────────
    if 'patient_id' in df_mapped.columns:
        dup_count = int(df_mapped['patient_id'].duplicated().sum())
        if dup_count > 0:
            results['duplicates'] = {
                'n_duplicates': dup_count,
                'pct_duplicates': round(dup_count / n_rows * 100, 1),
            }

    # ── 4. Near-constant column detection ────────────────
    for feat in available_features:
        if feat not in df_mapped.columns:
            continue
        col = pd.to_numeric(df_mapped[feat], errors='coerce').dropna()
        if len(col) < 2:
            continue
        top_val_pct = col.value_counts(normalize=True).iloc[0] * 100
        if top_val_pct >= 95:
            results['constant_columns'].append({
                'feature': feat,
                'dominant_value': round(float(col.value_counts().index[0]), 1),
                'pct_dominant': round(top_val_pct, 1),
            })

    # ── 5. Grade calculation ──────────────────────────────
    deductions = 0
    reasons = []

    # Missing data deductions
    if avg_missing == 0:
        reasons.append('no missing values')
    elif avg_missing <= 5:
        deductions += 5
        reasons.append(f'{avg_missing:.1f}% average missingness (minor)')
    elif avg_missing <= 15:
        deductions += 15
        reasons.append(f'{avg_missing:.1f}% average missingness (moderate)')
    elif avg_missing <= 30:
        deductions += 25
        reasons.append(f'{avg_missing:.1f}% average missingness (high)')
    else:
        deductions += 40
        reasons.append(f'{avg_missing:.1f}% average missingness (severe)')

    # High-importance feature missingness
    if avg_hi_missing > 20:
        deductions += 15
        reasons.append(f'{avg_hi_missing:.1f}% missing in high-importance features')

    # Out-of-range deductions
    if n_severe_range == 0 and len(results['out_of_range']) == 0:
        reasons.append('no out-of-range values detected')
    elif n_severe_range == 0:
        deductions += 5
        reasons.append(f"{len(results['out_of_range'])} feature(s) with minor range violations")
    else:
        deductions += 15
        reasons.append(f'{n_severe_range} feature(s) with severe range violations (>5% of rows)')

    # Duplicate deductions
    if results['duplicates']:
        n_dup = results['duplicates']['n_duplicates']
        pct_dup = results['duplicates']['pct_duplicates']
        if pct_dup < 1:
            deductions += 5
            reasons.append(f'{n_dup} duplicate patient ID(s) detected (minor)')
        elif pct_dup < 5:
            deductions += 15
            reasons.append(f'{n_dup} duplicate patient IDs ({pct_dup}% of cohort)')
        else:
            deductions += 25
            reasons.append(f'{n_dup} duplicate patient IDs ({pct_dup}% of cohort — high)')
    else:
        reasons.append('no duplicate patient IDs')

    # Constant column deductions
    if results['constant_columns']:
        n_const = len(results['constant_columns'])
        deductions += n_const * 5
        feat_names = [FEAT_LABELS.get(c['feature'], c['feature'])
                      for c in results['constant_columns']]
        reasons.append(
            f"{n_const} near-constant column(s): {', '.join(feat_names)} "
            f"(possibly miscoded — check export)"
        )
    else:
        reasons.append('no near-constant columns detected')

    # Assign grade
    score = 100 - deductions
    if score >= 90:
        grade = 'A'
    elif score >= 75:
        grade = 'B'
    elif score >= 55:
        grade = 'C'
    else:
        grade = 'D'

    results['grade'] = grade
    results['grade_reasons'] = reasons
    results['deductions'] = deductions
    results['score'] = score

    return results


def render_data_quality_report(dq, n_rows):
    """
    Display the data quality screening report.
    """
    grade = dq['grade']
    score = dq.get('score', 100 - dq['deductions'])
    reasons = dq['grade_reasons']

    grade_colours = {'A': '#3fb950', 'B': '#21d4fd', 'C': '#f0a500', 'D': '#f85149'}
    grade_labels = {
        'A': 'Excellent — data appears well-formed',
        'B': 'Good — minor issues detected, review recommended',
        'C': 'Fair — moderate issues detected, review before use',
        'D': 'Poor — significant issues detected, review required',
    }
    colour = grade_colours.get(grade, '#8b949e')
    label = grade_labels.get(grade, grade)

    with st.expander(
        f"📋 Data Quality Screening — Grade {grade}: {label}",
        expanded=(grade in ('C', 'D'))
    ):
        # Grade header
        c_grade, c_detail = st.columns([1, 4])
        with c_grade:
            st.markdown(
                f'<div style="text-align:center;padding:12px;border:2px solid {colour};'
                f'border-radius:8px;">'
                f'<div style="color:{colour};font-size:2.5rem;font-weight:900">{grade}</div>'
                f'<div style="color:#8b949e;font-size:0.7rem">Screening Score: {score}/100</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with c_detail:
            st.markdown(
                f'**Data quality screening score — not a clinical validity score.**\n\n'
                f'Grade {grade}: ' + '; '.join(reasons) + '.'
            )
            if dq['deductions'] > 0:
                st.caption(
                    f"Total deductions: {dq['deductions']} points across "
                    f"{len([r for r in reasons if 'no ' not in r])} issue(s)."
                )

        st.markdown("---")

        # Missing values table
        if dq['missing']:
            st.markdown("**Missing Values — imputation applied:**")
            miss_rows = []
            for feat, info in dq['missing'].items():
                miss_rows.append({
                    'Variable': FEAT_LABELS.get(feat, feat),
                    'Missing': f"{info['n_missing']:,} ({info['pct_missing']}%)",
                    'Imputed with': f"{info['impute_method']} ({info['impute_val']})",
                    'High importance': '⚠️ Yes' if info['high_importance'] else 'No',
                })
            st.dataframe(pd.DataFrame(miss_rows), use_container_width=True)
        else:
            st.success("✅ No missing values detected.")

        # Out-of-range table
        if dq['out_of_range']:
            st.markdown("**Out-of-Range Values — rows retained, flagged for review:**")
            range_rows = []
            for feat, info in dq['out_of_range'].items():
                range_rows.append({
                    'Variable': FEAT_LABELS.get(feat, feat),
                    'Valid Range': info['valid_range'],
                    'Rows Affected': f"{info['n_out']:,} ({info['pct_out']}%)",
                    'Observed Min/Max': f"{info['min_observed']} / {info['max_observed']}",
                    'Severity': '🔴 Severe' if info['severity'] == 'severe' else '🟡 Minor',
                })
            st.dataframe(pd.DataFrame(range_rows), use_container_width=True)
            st.caption(
                "Rows with out-of-range values are retained. "
                "Review these values with your data manager before operational use."
            )
        else:
            st.success("✅ No out-of-range values detected.")

        # Duplicates
        if dq['duplicates']:
            n_dup = dq['duplicates']['n_duplicates']
            pct_dup = dq['duplicates']['pct_duplicates']
            st.warning(
                f"⚠️ **{n_dup:,} duplicate patient ID(s)** detected ({pct_dup}% of cohort). "
                f"Duplicates have not been removed. Review with your data manager — "
                f"duplicates may indicate repeated records, data export errors, or "
                f"patients attending multiple facilities."
            )
        else:
            st.success("✅ No duplicate patient IDs detected.")

        # Near-constant columns
        if dq['constant_columns']:
            st.markdown("**Near-Constant Columns — possible miscoding:**")
            for c in dq['constant_columns']:
                fname = FEAT_LABELS.get(c['feature'], c['feature'])
                st.warning(
                    f"⚠️ **{fname}**: {c['pct_dominant']}% of values are "
                    f"{c['dominant_value']}. This column may have been miscoded "
                    f"or exported incorrectly. Verify in your EMR system."
                )
        else:
            st.success("✅ No near-constant columns detected.")

        st.caption(
            f"Data quality screening applied to {n_rows:,} patients across "
            f"{len(dq['missing']) + len([f for f in FEATURE_VALID_RANGES if f not in dq['out_of_range']])} "
            f"features. This screening identifies structural data issues only and does not "
            f"assess clinical validity or programme representativeness."
        )



# ─────────────────────────────────────────────────────────────
# IeDEA MUD REGIONAL AGGREGATE CONTEXTUAL BENCHMARKS
# Source: IeDEA Multi-Use Dataset (MUD) v1.0, 2025
# Data through 2022. CC BY-NC-SA 4.0
# These are aggregate contextual benchmarks — NOT patient-level
# external validation.
# ─────────────────────────────────────────────────────────────

IEDEA_MUD_SUMMARY = {
    'WA': {
        'name': 'West Africa',
        'countries': 'Benin, Burkina Faso, Côte d\'Ivoire, Ghana, Mali, Nigeria, Senegal, Togo',
        'artstart_n': 42369,
        'cd4_art_median': 181.0,
        'cd4_art_pct_below200': 43.6,
        'vl_supp_6mo_perc': 84.6,
        'vl_supp_12mo_perc': 84.5,
    },
    'EA': {
        'name': 'East Africa',
        'countries': 'Kenya, Uganda, Tanzania',
        'artstart_n': 229002,
        'cd4_art_median': 197.0,
        'cd4_art_pct_below200': 30.2,
        'vl_supp_6mo_perc': 89.7,
        'vl_supp_12mo_perc': 87.6,
    },
    'SA': {
        'name': 'Southern Africa',
        'countries': 'South Africa, Zambia, Malawi, Lesotho, Mozambique, Zimbabwe',
        'artstart_n': 921922,
        'cd4_art_median': 218.0,
        'cd4_art_pct_below200': 20.5,
        'vl_supp_6mo_perc': 91.0,
        'vl_supp_12mo_perc': 89.7,
    },
    'CA': {
        'name': 'Central Africa',
        'countries': 'Burundi, Cameroon, DRC, Rwanda',
        'artstart_n': 42459,
        'cd4_art_median': 241.0,
        'cd4_art_pct_below200': 24.4,
        'vl_supp_6mo_perc': 91.9,
        'vl_supp_12mo_perc': 88.0,
    },
}

IEDEA_MUD_SEX = {
    'WA': {'Male': 85.0, 'Female': 84.2},
    'EA': {'Male': 86.3, 'Female': 88.3},
    'SA': {'Male': 88.0, 'Female': 90.7},
    'CA': {'Male': 87.9, 'Female': 88.1},
}

IEDEA_MUD_TREND = {
    'WA': {
        'years': [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,
                  2016,2017,2018,2019,2020,2021,2022],
        'vl_supp_12mo': [78.3,78.8,82.4,81.8,80.6,84.5,83.2,84.3,86.3,
                         89.9,88.7,88.8,90.0,88.1,90.1,93.2,90.1],
        'cd4_median': [121,125,157,163,171,181.5,207,210.5,207,256,
                       257,309,256,253,257,277.5,293.5],
    },
    'EA': {
        'years': [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,
                  2016,2017,2018,2019,2020,2021,2022],
        'vl_supp_12mo': [81.4,76.8,77.7,72.8,62.0,50.2,73.2,84.1,86.8,
                         84.6,84.6,85.8,91.2,93.8,95.7,94.6,95.4],
        'cd4_median': [100,117,140,146.62,169,198,215,262,321,290,
                       326,329,329,324,312,292,289],
    },
    'SA': {
        'years': [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,
                  2016,2017,2018,2019,2020,2021,2022],
        'vl_supp_12mo': [88.0,88.6,88.1,87.7,86.4,87.4,87.8,89.0,90.8,
                         90.7,90.3,89.6,89.1,90.8,92.7,93.6,93.0],
        'cd4_median': [127,140,148,154,167,195,217,239,269,284,
                       301,334,325,325,331,319,307],
    },
    'CA': {
        'years': [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,
                  2016,2017,2018,2019,2020,2021,2022],
        'vl_supp_12mo': [90.5,90.6,82.9,91.3,85.6,88.8,90.2,90.8,91.1,
                         85.9,84.2,84.5,85.6,91.5,94.1,93.8,96.8],
        'cd4_median': [144,175,201,211,223,249.5,247,250,286,273,
                       322,272,307,287,326,332,336],
    },
}

IEDEA_MUD_CITATION = (
    "IeDEA (2025). Version 1.0. IeDEA Multi-Use Dataset (MUD). "
    "Retrieved from iedea.org. License: CC BY-NC-SA 4.0."
)

IEDEA_MUD_NOTE = (
    "IeDEA MUD data represent aggregate indicators from IeDEA-participating "
    "clinical sites. These are regional contextual benchmarks — not nationally "
    "representative estimates and not patient-level external validation. "
    "Data through 2022. Site composition varies by region and year."
)


def render_iedea_benchmarks(df_upload=None, selected_region=None):
    """
    Render IeDEA MUD regional aggregate contextual benchmarks.
    If df_upload provided, compares cohort metrics against selected region.
    """
    st.markdown('<p class="section-hdr">IeDEA MUD Regional Aggregate Contextual Benchmarks</p>',
                unsafe_allow_html=True)

    st.markdown(f"""<div class="info-box">
    <strong>What this shows:</strong> Regional aggregate indicators from the IeDEA
    Multi-Use Dataset (MUD) v1.0 — covering {sum(v['artstart_n'] for v in IEDEA_MUD_SUMMARY.values()):,}
    patients across West, East, Southern, and Central Africa (data through 2022).
    These are <strong>contextual benchmarks only</strong> — not patient-level external validation
    of the SmartDaaS model. Data reflect IeDEA-participating sites and are not
    nationally representative.<br><br>
    <em>{IEDEA_MUD_CITATION}</em>
    </div>""", unsafe_allow_html=True)

    # ── Region selector ───────────────────────────────────
    region_options = {v['name']: k for k, v in IEDEA_MUD_SUMMARY.items()}
    region_options['All African Regions'] = 'ALL'

    default_region = 'All African Regions'
    if selected_region and selected_region in region_options.values():
        default_idx = list(region_options.values()).index(selected_region)
    else:
        default_idx = list(region_options.keys()).index('All African Regions')

    sel_name = st.selectbox(
        "Select region for comparison:",
        list(region_options.keys()),
        index=default_idx,
        key="iedea_region_sel"
    )
    sel_code = region_options[sel_name]

    # ── Regional summary cards ────────────────────────────
    if sel_code == 'ALL':
        regions_to_show = list(IEDEA_MUD_SUMMARY.keys())
    else:
        regions_to_show = [sel_code]

    cols = st.columns(len(regions_to_show))
    for i, reg in enumerate(regions_to_show):
        d = IEDEA_MUD_SUMMARY[reg]
        with cols[i]:
            st.markdown(f"""<div class="metric-box" style="text-align:center">
                <div style="color:#21d4fd;font-weight:700;font-size:0.85rem;margin-bottom:6px">
                    {d['name']}
                </div>
                <div style="color:#8b949e;font-size:0.7rem;margin-bottom:8px">
                    {d['countries'][:40]}{'...' if len(d['countries'])>40 else ''}
                </div>
                <div style="color:#e6edf3;font-size:1.1rem;font-weight:700">
                    {d['artstart_n']:,}
                </div>
                <div style="color:#8b949e;font-size:0.7rem">patients on ART</div>
                <hr style="border-color:#30363d;margin:8px 0">
                <div style="color:#3fb950;font-size:1rem;font-weight:700">
                    {d['vl_supp_12mo_perc']}%
                </div>
                <div style="color:#8b949e;font-size:0.7rem">VL suppression at 12mo</div>
                <div style="color:#e6edf3;font-size:0.9rem;margin-top:4px">
                    {d['cd4_art_median']:.0f}
                </div>
                <div style="color:#8b949e;font-size:0.7rem">Median CD4 at ART start</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison charts ─────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "VL Suppression Trend",
        "Sex-Disaggregated Outcomes",
        "CD4 at ART Start Trend"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        colours = {'WA': '#21d4fd', 'EA': '#3fb950', 'SA': '#f0a500', 'CA': '#f85149'}

        for reg in (regions_to_show if sel_code != 'ALL' else list(IEDEA_MUD_SUMMARY.keys())):
            td = IEDEA_MUD_TREND[reg]
            valid = [(y, v) for y, v in zip(td['years'], td['vl_supp_12mo']) if v is not None]
            if valid:
                ys, vs = zip(*valid)
                ax.plot(ys, vs, color=colours[reg], linewidth=2,
                        marker='o', markersize=3,
                        label=f"{IEDEA_MUD_SUMMARY[reg]['name']}")

        # If upload provided, overlay cohort VL suppression if available
        if df_upload is not None and 'vl_suppressed' in df_upload.columns:
            cohort_vl = pd.to_numeric(
                df_upload['vl_suppressed'], errors='coerce').mean() * 100
            ax.axhline(cohort_vl, color='#ffffff', linewidth=1.5,
                       linestyle='--',
                       label=f'Your cohort ({cohort_vl:.1f}%) — contextual reference only')

        ax.set_xlabel('Year of ART Initiation', color='#8b949e', fontsize=9)
        ax.set_ylabel('VL Suppression at 12 months (%)', color='#8b949e', fontsize=9)
        ax.set_title(
            'IeDEA MUD: Viral Load Suppression at 12 Months After ART Start\n'
            '(Regional aggregate contextual benchmark — not external validation)',
            color='#e6edf3', fontsize=9, pad=10
        )
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#cdd9e5')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.set_ylim(50, 100)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            "⚠️ Regional aggregate contextual comparison only — this display does not "
            "represent external validation or direct equivalence between datasets."
        )

    with tab2:
        regs = regions_to_show if sel_code != 'ALL' else list(IEDEA_MUD_SUMMARY.keys())
        male_vals = [IEDEA_MUD_SEX[r]['Male'] for r in regs]
        female_vals = [IEDEA_MUD_SEX[r]['Female'] for r in regs]
        reg_labels = [IEDEA_MUD_SUMMARY[r]['name'] for r in regs]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        x = range(len(regs))
        w = 0.35
        bars_m = ax.bar([i - w/2 for i in x], male_vals, width=w,
                        color='#21d4fd', label='Male', edgecolor='#0d1117')
        bars_f = ax.bar([i + w/2 for i in x], female_vals, width=w,
                        color='#f0a500', label='Female', edgecolor='#0d1117')

        for bar in bars_m:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.1f}%', ha='center',
                    fontsize=8, color='#e6edf3')
        for bar in bars_f:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.1f}%', ha='center',
                    fontsize=8, color='#e6edf3')

        ax.set_xticks(list(x))
        ax.set_xticklabels(reg_labels, fontsize=8, color='#cdd9e5')
        ax.set_ylabel('VL Suppression at 12 months (%)', color='#8b949e', fontsize=9)
        ax.set_title(
            'IeDEA MUD: VL Suppression by Sex\n'
            '(Regional aggregate — all years)',
            color='#e6edf3', fontsize=9, pad=10
        )
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#cdd9e5')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.set_ylim(75, 96)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(
            "_Note: In the Nigerian discovery cohort, female sex was associated with "
            "lower odds of poor outcome. IeDEA MUD shows female advantage in East and "
            "Southern Africa; West Africa shows a smaller and reversed pattern. "
            "These aggregate patterns provide contextual plausibility only._"
        )
        st.caption(
            "⚠️ Regional aggregate contextual comparison only — this display does not "
            "represent external validation or direct equivalence between datasets."
        )

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        colours = {'WA': '#21d4fd', 'EA': '#3fb950', 'SA': '#f0a500', 'CA': '#f85149'}

        for reg in (regions_to_show if sel_code != 'ALL' else list(IEDEA_MUD_SUMMARY.keys())):
            td = IEDEA_MUD_TREND[reg]
            valid = [(y, c) for y, c in zip(td['years'], td['cd4_median']) if c is not None]
            if valid:
                ys, cs = zip(*valid)
                ax.plot(ys, cs, color=colours[reg], linewidth=2,
                        marker='o', markersize=3,
                        label=f"{IEDEA_MUD_SUMMARY[reg]['name']}")

        ax.set_xlabel('Year of ART Initiation', color='#8b949e', fontsize=9)
        ax.set_ylabel('Median CD4 at ART Start (cells/µL)', color='#8b949e', fontsize=9)
        ax.set_title(
            'IeDEA MUD: Median CD4 Count at ART Initiation Over Time\n'
            '(Regional aggregate contextual benchmark)',
            color='#e6edf3', fontsize=9, pad=10
        )
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#cdd9e5')
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            "⚠️ Regional aggregate contextual comparison only — this display does not "
            "represent external validation or direct equivalence between datasets."
        )

    # ── Disclaimer ───────────────────────────────────────
    st.markdown(f"""<div class="warn-box" style="margin-top:12px">
    ⚠️ <strong>Important:</strong> {IEDEA_MUD_NOTE}
    </div>""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────
# LOCAL RECALIBRATION ENGINE — Stage 2
# Platt scaling calibration with full validation checks
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, roc_curve,
                              brier_score_loss, confusion_matrix)
from sklearn.utils import resample

# ── Constants ────────────────────────────────────────────────
RECAL_MIN_PATIENTS     = 200
RECAL_MIN_POS_EVENTS   = 30
RECAL_MAX_OUTCOME_MISS = 0.40
RECAL_MAX_FEAT_MISS    = 0.50
RECAL_ISOTONIC_MIN     = 500
BOOTSTRAP_N            = 1000
BASELINE_AUC           = 0.772
BASELINE_THRESHOLD     = 0.70

# Recognised outcome column names (lowercased, stripped)
OUTCOME_NAME_HINTS = {
    'poor_outcome', 'composite_outcome', 'outcome', 'art_outcome',
    'poor_adherence', 'treatment_interrupted', 'interruption',
    'had_poor_outcome', 'bad_outcome', 'adverse_outcome',
    'ltfu', 'lost_to_followup', 'treatment_failure',
    'mortality', 'dead', 'patient_dead', 'died',
    'art_interruption', 'non_adherent', 'not_adherent',
    'label', 'target', 'y', 'outcome_binary',
}


# ── 1. Outcome column detection ───────────────────────────────

def detect_outcome_columns(df):
    """
    Scan dataframe for likely outcome columns.
    Returns list of (col_name, detection_reason, value_counts) tuples.
    Detects both numeric 0/1 columns and text-valued programme columns
    (e.g. LTFU/Active/Retained/Dead) that can be normalised to binary.
    """
    # Text values that indicate a programme outcome column
    OUTCOME_TEXT_SIGNALS = {
        'ltfu', 'lost to follow-up', 'lost to follow up', 'lost',
        'dead', 'death', 'died', 'retained', 'active', 'defaulted',
        'interrupted', 'failure', 'transferred out', 'stopped',
        'suppressed', 'on treatment', 'enrolled', 'dropped',
    }

    candidates = []
    for col in df.columns:
        col_lower = col.lower().strip().replace(' ', '_')
        reason = None

        # Name-based detection
        if col_lower in OUTCOME_NAME_HINTS:
            reason = 'column name matches known outcome identifier'

        # Value-pattern detection — binary 0/1
        if reason is None:
            try:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                unique = set(vals.unique())
                if unique <= {0, 1, 0.0, 1.0} and len(vals) > 0:
                    pct_pos = (vals == 1).mean()
                    if 0.01 <= pct_pos <= 0.60:
                        reason = (f'binary 0/1 column with '
                                  f'{pct_pos*100:.1f}% positive rate')
            except Exception:
                pass

        # Text-value detection — programme status strings
        if reason is None:
            try:
                text_vals = df[col].dropna().astype(str).str.strip().str.lower()
                unique_text = set(text_vals.unique())
                matches = unique_text & OUTCOME_TEXT_SIGNALS
                if matches and len(unique_text) <= 10:
                    reason = (f'programme status column — '
                              f'recognised values: '
                              f'{", ".join(sorted(matches)[:4])}')
            except Exception:
                pass

        if reason:
            try:
                vc = df[col].value_counts().to_dict()
            except Exception:
                vc = {}
            candidates.append((col, reason, vc))

    # Exclude columns that are already model features
    candidates = [(c, r, v) for c, r, v in candidates if c not in FEATURES]
    return candidates


# ── 2. Outcome column validation ──────────────────────────────

def normalize_outcome_column(df, outcome_col):
    """
    Automatically normalise a programme outcome column to binary 0/1
    before validation and recalibration.

    Handles common real-world HIV programme export values:
      Poor outcome  → 1:  LTFU, Dead, Interrupted, Failure, Defaulted,
                          Transferred Out, Stopped, Discontinued, Yes, True, Y, 1
      Good outcome  → 0:  Retained, Active, Suppressed, Alive, Enrolled,
                          On Treatment, No, False, N, 0

    Returns (df_out, mapping_applied, ambiguous_values) where:
      - df_out          : copy of df with outcome_col recoded to 0/1 float
      - mapping_applied : dict of {original_value: mapped_value} for audit trail
      - ambiguous_values: list of values that could not be auto-mapped
    """
    # Canonical mappings — case-insensitive, stripped
    POOR_OUTCOME = {
        'ltfu', 'lost to follow-up', 'lost to follow up', 'lost',
        'dead', 'death', 'died',
        'interrupted', 'interruption', 'treatment interrupted',
        'failure', 'failed', 'treatment failure',
        'defaulted', 'default',
        'transferred out', 'transfer out', 'to',
        'stopped', 'stop', 'discontinued', 'discontinue',
        'drop', 'dropped', 'dropped out',
        'non-adherent', 'nonadherent',
        '1', 'yes', 'true', 'y',
    }
    GOOD_OUTCOME = {
        'retained', 'retain', 'active', 'on treatment', 'on art',
        'suppressed', 'virally suppressed', 'vls',
        'alive', 'living',
        'enrolled', 'current', 'in care',
        'adherent', 'compliant',
        '0', 'no', 'false', 'n',
    }

    df_out = df.copy()
    col = df_out[outcome_col].copy()

    # Fast path: genuine numeric integer/float dtype — check if already 0/1
    # Do this BEFORE dtype conversion to preserve int columns
    if pd.api.types.is_numeric_dtype(col):
        numeric = pd.to_numeric(col, errors='coerce')
        numeric_unique = set(numeric.dropna().unique())
        if numeric_unique <= {0, 1, 0.0, 1.0}:
            df_out[outcome_col] = numeric.astype(float)
            return df_out, {}, []

    # Normalise ArrowStringArray and other non-standard dtypes to plain object
    try:
        col = col.astype(object)
    except Exception:
        pass

    # Attempt text mapping
    mapping_applied = {}
    ambiguous_values = []
    recoded = []

    for raw_val in col:
        if pd.isna(raw_val):
            recoded.append(np.nan)
            continue
        # Try numeric first
        try:
            n = float(raw_val)
            if n in (0.0, 1.0):
                recoded.append(n)
                mapping_applied[str(raw_val)] = int(n)
                continue
        except (ValueError, TypeError):
            pass
        # Text normalisation
        norm = str(raw_val).strip().lower()
        if norm in POOR_OUTCOME:
            recoded.append(1.0)
            mapping_applied[str(raw_val)] = 1
        elif norm in GOOD_OUTCOME:
            recoded.append(0.0)
            mapping_applied[str(raw_val)] = 0
        else:
            recoded.append(np.nan)
            if str(raw_val) not in ambiguous_values:
                ambiguous_values.append(str(raw_val))

    df_out[outcome_col] = recoded
    return df_out, mapping_applied, ambiguous_values


def validate_outcome_column(df, outcome_col, available_features):
    """
    Run the four validation checks.
    Returns dict with pass/fail per check and overall can_proceed bool.
    """
    results = {
        'checks': {},
        'can_proceed': False,
        'warnings': [],
    }

    try:
        outcome = pd.to_numeric(df[outcome_col], errors='coerce')
    except Exception:
        results['checks']['outcome_readable'] = {
            'passed': False,
            'label': 'Outcome column readable',
            'detail': 'Could not parse outcome column as numeric.',
        }
        return results

    n_total = len(outcome)
    n_missing = outcome.isnull().sum()
    outcome_clean = outcome.dropna()
    n_clean = len(outcome_clean)

    # Check 1 — Sample size
    passed_n = n_clean >= RECAL_MIN_PATIENTS
    results['checks']['sample_size'] = {
        'passed': passed_n,
        'label': 'Sufficient sample size',
        'detail': (f'{n_clean:,} patients with known outcome '
                   f'(minimum required: {RECAL_MIN_PATIENTS:,})'),
        'value': n_clean,
        'threshold': RECAL_MIN_PATIENTS,
    }

    # Guard: if all outcome values are non-numeric / NaN, fail gracefully
    if n_clean == 0:
        results['checks']['outcome_events'] = {
            'passed': False,
            'label': 'Sufficient outcome events',
            'detail': (
                'Outcome column contains no valid numeric values after parsing. '
                'Expected binary 0/1. If your outcome column uses text values '
                '(e.g. "Active", "LTFU", "Dead"), please recode to 1=poor outcome, '
                '0=good outcome before uploading.'
            ),
            'value': 0,
            'threshold': RECAL_MIN_POS_EVENTS,
            'prevalence': 0,
        }
        results['can_proceed'] = False
        return results

    # Check 2 — Outcome events
    unique_vals = set(outcome_clean.unique())
    if not unique_vals <= {0, 1, 0.0, 1.0}:
        results['checks']['outcome_binary'] = {
            'passed': False,
            'label': 'Recognised outcome definition',
            'detail': (f'Outcome column contains unexpected values: '
                       f'{sorted(unique_vals)[:5]}. '
                       f'Expected binary 0/1 only.'),
        }
        return results

    n_pos = int((outcome_clean == 1).sum())
    n_neg = int((outcome_clean == 0).sum())
    passed_events = n_pos >= RECAL_MIN_POS_EVENTS
    prevalence_pct = (n_pos / n_clean * 100) if n_clean > 0 else 0.0
    results['checks']['outcome_events'] = {
        'passed': passed_events,
        'label': 'Sufficient outcome events',
        'detail': (f'{n_pos:,} positive outcome events '
                   f'(minimum required: {RECAL_MIN_POS_EVENTS:,}). '
                   f'Outcome prevalence: {prevalence_pct:.1f}%'),
        'value': n_pos,
        'threshold': RECAL_MIN_POS_EVENTS,
        'prevalence': n_pos / n_clean if n_clean > 0 else 0,
    }

    # Check 3 — Outcome missingness
    miss_pct = n_missing / n_total if n_total > 0 else 1.0
    passed_miss = miss_pct <= RECAL_MAX_OUTCOME_MISS
    results['checks']['outcome_missingness'] = {
        'passed': passed_miss,
        'label': 'Acceptable outcome missingness',
        'detail': (f'{n_missing:,} missing outcome values '
                   f'({miss_pct*100:.1f}% of cohort). '
                   f'Maximum allowed: {RECAL_MAX_OUTCOME_MISS*100:.0f}%'),
        'value': miss_pct,
        'threshold': RECAL_MAX_OUTCOME_MISS,
    }

    # Check 4 — Recognised outcome definition (binary confirmed)
    passed_def = True
    results['checks']['outcome_definition'] = {
        'passed': passed_def,
        'label': 'Recognised outcome definition',
        'detail': (f'Binary outcome confirmed: '
                   f'{n_pos:,} positive (1) and {n_neg:,} negative (0). '
                   f'Values: {sorted(unique_vals)}'),
    }

    # Check 5 — Feature missingness (warning only, not blocking)
    if available_features:
        feat_miss = []
        for feat in available_features:
            if feat in df.columns:
                miss = df[feat].isnull().mean()
                if miss > RECAL_MAX_FEAT_MISS:
                    feat_miss.append(
                        f'{FEAT_LABELS.get(feat, feat)} '
                        f'({miss*100:.0f}% missing)'
                    )
        if feat_miss:
            results['warnings'].append(
                f'High missingness in predictor features: '
                f'{", ".join(feat_miss[:3])}. '
                f'Recalibration will proceed but results may be less reliable.'
            )

    # Overall decision
    blocking = ['sample_size', 'outcome_events', 'outcome_missingness',
                'outcome_definition']
    results['can_proceed'] = all(
        results['checks'].get(c, {}).get('passed', False)
        for c in blocking
        if c in results['checks']
    )

    return results


# ── 3. Bootstrap AUC with confidence interval ─────────────────

def bootstrap_auc(y_true, y_prob, n_boot=BOOTSTRAP_N, seed=42):
    """
    Returns (auc, ci_lower, ci_upper) using percentile bootstrap.
    """
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt = np.array(y_true)[idx]
        yp = np.array(y_prob)[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except Exception:
            continue
    if len(aucs) < 10:
        base = roc_auc_score(y_true, y_prob)
        return base, None, None
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── 4. Find optimal threshold ────────────────────────────────

def find_optimal_threshold(y_true, y_prob, method='youden'):
    """
    Find threshold maximising Youden's J (sensitivity + specificity - 1).
    Returns (threshold, sensitivity, specificity).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx])
    # Compute confusion matrix at threshold
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
    }


# ── 5. Run recalibration ──────────────────────────────────────

def sanitize_feature_matrix(df, features=None):
    """
    Sanitize all model feature columns to numeric float before any
    matrix operation (.astype(float), model.predict_proba, etc.).

    Handles real-world programme export values that pass column normalisation
    but survive as strings into the feature matrix:
      T / True / Yes / Y / 1     → 1.0
      F / False / No / N / 0     → 0.0
      Female / Woman / F / W     → 1.0  (sex_female encoding)
      Male / Man / M             → 0.0
      Stage I/II/III/IV text     → 1.0/2.0/3.0/4.0
      Numeric strings ('3.5')    → 3.5
      Unparseable text           → 0.0 (neutral default, logged)

    Returns (df_clean, audit_log) where audit_log is a list of strings
    describing every column that was coerced and what values were found.
    Does NOT crash on unresolvable values — substitutes 0.0 and logs.
    """
    import re as _re
    df = df.copy()
    features = features or FEATURES
    audit_log = []

    # Value maps applied before generic numeric coercion
    BOOL_TRUE  = {'t', 'true', 'yes', 'y', '1', 'positive', 'pos'}
    BOOL_FALSE = {'f', 'false', 'no', 'n', '0', 'negative', 'neg', 'none', ''}
    FEMALE_STR = {'female', 'woman', 'w', '2'}
    MALE_STR   = {'m', 'male', 'man', '0', '1'}
    ROMAN      = {'i': 1.0, 'ii': 2.0, 'iii': 3.0, 'iv': 4.0}

    # Binary features — True=1, False=0
    BINARY_FEATURES = {
        'sex_female', 'had_interruption', 'opp_infection',
        'side_effects', 'tb_positive', 'stage_worsened',
    }

    for feat in features:
        if feat not in df.columns:
            continue

        col = df[feat]

        # Already numeric — just coerce safely
        if pd.api.types.is_numeric_dtype(col):
            df[feat] = pd.to_numeric(col, errors='coerce').fillna(0.0)
            continue

        # String / object column — needs parsing
        original_dtype = str(col.dtype)
        try:
            col_str = col.astype(str)
        except Exception:
            df[feat] = 0.0
            audit_log.append(f"{feat}: could not convert to string — filled with 0.0")
            continue

        unique_vals = col_str.dropna().unique()
        non_numeric = [v for v in unique_vals
                       if v not in ('nan', 'None', '')
                       and pd.to_numeric(v, errors='coerce') != pd.to_numeric(v, errors='coerce')]

        if not any(True for _ in non_numeric):
            # All values are numeric strings — simple coerce
            df[feat] = pd.to_numeric(col_str, errors='coerce').fillna(0.0)
            if len(non_numeric) == 0 and original_dtype != 'float64':
                audit_log.append(
                    f"{feat}: numeric strings coerced to float "
                    f"(dtype was {original_dtype})"
                )
            continue

        # Non-numeric strings present — apply feature-specific mapping
        offending = sorted(set(non_numeric))
        recoded = []
        unresolved = []

        for raw in col_str:
            s = str(raw).strip().lower()
            if s in ('nan', 'none', ''):
                recoded.append(0.0)
                continue

            # Try plain numeric first
            try:
                recoded.append(float(s))
                continue
            except (ValueError, TypeError):
                pass

            # sex_female special handling
            if feat == 'sex_female':
                if s in FEMALE_STR or s in ('t', 'true', 'yes', 'y'):
                    recoded.append(1.0)
                elif s in MALE_STR or s in ('f', 'false', 'no', 'n'):
                    recoded.append(0.0)
                else:
                    recoded.append(0.0)
                    if s not in unresolved: unresolved.append(s)
                continue

            # stage_start_num — Roman numerals and "Stage N" text
            if feat == 'stage_start_num':
                if s in ROMAN:
                    recoded.append(ROMAN[s])
                    continue
                # Extract roman numeral from compound string e.g. "stage iii", "who stage iv"
                roman_match = _re.search(r'\b(iv|iii|ii|i)\b', s)
                if roman_match:
                    recoded.append(ROMAN[roman_match.group(1)])
                    continue
                # Extract digit from string e.g. "stage 3", "who stage 3"
                m = _re.search(r'(\d)', s)
                if m:
                    recoded.append(float(m.group(1)))
                    continue
                recoded.append(2.0)  # default neutral stage
                if s not in unresolved: unresolved.append(s)
                continue

            # Binary features
            if feat in BINARY_FEATURES:
                if s in BOOL_TRUE:
                    recoded.append(1.0)
                elif s in BOOL_FALSE:
                    recoded.append(0.0)
                else:
                    recoded.append(0.0)
                    if s not in unresolved: unresolved.append(s)
                continue

            # Generic fallback — boolean-like then zero
            if s in BOOL_TRUE:
                recoded.append(1.0)
            elif s in BOOL_FALSE:
                recoded.append(0.0)
            else:
                recoded.append(0.0)
                if s not in unresolved: unresolved.append(s)

        df[feat] = recoded

        # Build audit entry
        mapping_summary = f"offending values: {offending}"
        if unresolved:
            audit_log.append(
                f"{feat}: {len(offending)} non-numeric value(s) found "
                f"{offending} — coerced to 0/1 where recognised; "
                f"unresolvable values {unresolved} → 0.0 (neutral default)"
            )
        else:
            audit_log.append(
                f"{feat}: {len(offending)} non-numeric value(s) "
                f"{offending} — all successfully coerced"
            )

    return df, audit_log


def run_recalibration(df_mapped, outcome_col, model, available_features):
    """
    Full recalibration pipeline.
    Returns recal_results dict with all metrics and calibrated model.
    """
    # ── Step 0: Sanitize feature matrix ──────────────────────
    # Coerces any non-numeric values in feature columns to float
    # before matrix operations. Prevents crashes on real-world
    # programme exports containing T/F, Yes/No, text categories.
    df_mapped, _sanitize_log = sanitize_feature_matrix(df_mapped, FEATURES)

    # Align outcome with available rows
    outcome = pd.to_numeric(df_mapped[outcome_col], errors='coerce')
    valid_mask = outcome.notna()
    df_valid = df_mapped[valid_mask].copy()
    y_true = outcome[valid_mask].astype(int).values

    # Ensure all features present
    for feat in FEATURES:
        if feat not in df_valid.columns:
            df_valid[feat] = 0

    X = df_valid[FEATURES].values.astype(float)

    # Base model predictions
    y_prob_base = model.predict_proba(X)[:, 1]

    # AUC before calibration
    auc_base = roc_auc_score(y_true, y_prob_base)

    # Decide calibration method
    n_clean = len(y_true)
    use_isotonic = n_clean >= RECAL_ISOTONIC_MIN

    # Fit calibration layer
    if use_isotonic:
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_prob_base, y_true)
        y_prob_cal = calibrator.predict(y_prob_base)
        cal_method = 'Isotonic Regression'
    else:
        calibrator = LogisticRegression(C=1e5, solver='lbfgs',
                                        max_iter=1000)
        calibrator.fit(y_prob_base.reshape(-1, 1), y_true)
        y_prob_cal = calibrator.predict_proba(
            y_prob_base.reshape(-1, 1))[:, 1]
        cal_method = 'Platt Scaling (Logistic Regression)'

    # AUC after calibration with bootstrap CI
    auc_cal, ci_lo, ci_hi = bootstrap_auc(y_true, y_prob_cal)

    # Optimal threshold
    thresh_metrics = find_optimal_threshold(y_true, y_prob_cal)

    # Brier score (calibration quality — lower is better)
    brier = brier_score_loss(y_true, y_prob_cal)

    # Outcome prevalence
    prevalence = float(y_true.mean())

    # ROC curve data for plotting
    fpr, tpr, _ = roc_curve(y_true, y_prob_cal)

    return {
        'calibrator': calibrator,
        'cal_method': cal_method,
        'use_isotonic': use_isotonic,
        'n_patients': n_clean,
        'n_positive': int(y_true.sum()),
        'prevalence': prevalence,
        'auc_base': float(auc_base),
        'auc_cal': float(auc_cal),
        'auc_ci_lo': ci_lo,
        'auc_ci_hi': ci_hi,
        'brier': float(brier),
        'threshold': thresh_metrics,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'baseline_auc': BASELINE_AUC,
        'outcome_col': outcome_col,
        'y_true': y_true.tolist(),
        'y_prob_cal': y_prob_cal.tolist(),
    }


# ── 6. Apply calibration to new predictions ───────────────────

def apply_calibration(probs_raw, calibrator, use_isotonic):
    """
    Apply stored calibration layer to new predicted probabilities.
    """
    try:
        if use_isotonic:
            return calibrator.predict(probs_raw)
        else:
            return calibrator.predict_proba(
                probs_raw.reshape(-1, 1))[:, 1]
    except Exception:
        return probs_raw


# ── 7. Synthetic test data generator ─────────────────────────

def generate_synthetic_recal_data(scenario='good', seed=42):
    """
    Generate synthetic data for testing recalibration.
    Scenario A: good  — 500 patients, 15% outcome, passes all checks
    Scenario B: few_events — 500 patients, 2% outcome, fails events check
    Scenario C: small_n — 150 patients, 15% outcome, fails sample size
    """
    rng = np.random.RandomState(seed)

    scenarios = {
        'good':       {'n': 500, 'prev': 0.15},
        'few_events': {'n': 500, 'prev': 0.02},
        'small_n':    {'n': 150, 'prev': 0.15},
    }
    cfg = scenarios.get(scenario, scenarios['good'])
    n, prev = cfg['n'], cfg['prev']

    df = pd.DataFrame({
        'Age':               rng.randint(20, 65, n).astype(float),
        'sex_female':        rng.randint(0, 2, n).astype(float),
        'Cd4AtStart':        rng.randint(50, 800, n).astype(float),
        'MostRecentCd4Count':rng.randint(100, 900, n).astype(float),
        'CD4_improvement':   rng.randint(-200, 400, n).astype(float),
        'stage_start_num':   rng.randint(1, 5, n).astype(float),
        'WeightAtStart':     rng.randint(40, 100, n).astype(float),
        'weight_change':     rng.uniform(-10, 10, n),
        'BMI_start':         rng.uniform(16, 35, n),
        'days_to_ART':       rng.randint(0, 365, n).astype(float),
        'had_interruption':  rng.randint(0, 2, n).astype(float),
        'opp_infection':     rng.randint(0, 2, n).astype(float),
        'side_effects':      rng.randint(0, 2, n).astype(float),
        'tb_positive':       rng.randint(0, 2, n).astype(float),
        'stage_worsened':    rng.randint(0, 2, n).astype(float),
        'patient_id':        [f'SYN-{i:04d}' for i in range(n)],
        'poor_outcome':      rng.binomial(1, prev, n).astype(float),
    })
    return df


# ── 8. Render recalibration UI ────────────────────────────────

def render_recalibration_page(model):
    """
    Full Local Validation page UI.
    """
    st.markdown("""
### 🔬 Local Validation & Recalibration

Validate SmartDaaS performance on your programme's own data and generate
a locally-calibrated risk model specific to your context.
""")

    st.markdown("""<div class="info-box">
    <strong>What this does:</strong> When you upload historical programme data with known
    patient outcomes, this module fits a calibration layer on top of the SmartDaaS base model.
    The result is a <strong>locally-validated AUC</strong> specific to your programme —
    the number you report to funders, not the Nigerian discovery cohort baseline of 0.772.<br><br>
    <strong>What you need:</strong> A CSV of historical patients where outcomes are already known
    (patients who completed treatment, interrupted, or died). Minimum 200 patients,
    minimum 30 positive outcome events.
    </div>""", unsafe_allow_html=True)

    # ── Baseline reference ─────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="metric-box"><div class="metric-val">0.772</div>'
            '<div class="metric-lbl">Baseline AUC (Nigerian cohort)</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            '<div class="metric-box"><div class="metric-val">27,288</div>'
            '<div class="metric-lbl">Discovery cohort patients</div></div>',
            unsafe_allow_html=True)
    with c3:
        local_auc = st.session_state.get('local_auc', None)
        if local_auc:
            st.markdown(
                f'<div class="metric-box" style="border-color:#3fb950">'
                f'<div class="metric-val" style="color:#3fb950">{local_auc:.3f}</div>'
                f'<div class="metric-lbl">Your local AUC ✓</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="metric-box" style="opacity:0.4">'
                '<div class="metric-val">—</div>'
                '<div class="metric-lbl">Your local AUC (pending)</div></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Test mode toggle ──────────────────────────────────
    st.markdown('<p class="section-hdr">Data Source</p>', unsafe_allow_html=True)
    use_synthetic = st.checkbox(
        "🧪 Use synthetic test data (for testing and demonstration)",
        value=False,
        help="Generates synthetic patient data with known outcomes to test the recalibration pipeline."
    )

    if use_synthetic:
        scenario = st.selectbox(
            "Select test scenario:",
            options=['good', 'few_events', 'small_n'],
            format_func=lambda x: {
                'good': 'Scenario A — Good data (500 patients, 15% outcome rate) — should PASS all checks',
                'few_events': 'Scenario B — Few outcome events (500 patients, 2% rate) — should FAIL events check',
                'small_n': 'Scenario C — Small sample (150 patients, 15% rate) — should FAIL sample size check',
            }[x]
        )
        df_recal = generate_synthetic_recal_data(scenario=scenario)
        st.info(
            f"🧪 Synthetic data generated: {len(df_recal):,} patients, "
            f"{int(df_recal['poor_outcome'].sum())} positive outcomes "
            f"({df_recal['poor_outcome'].mean()*100:.1f}% prevalence). "
            f"Outcome column: `poor_outcome`."
        )
        outcome_col_confirmed = 'poor_outcome'

    else:
        uploaded_recal = st.file_uploader(
            "Upload historical programme data with known outcomes (CSV)",
            type=['csv'],
            help="Must include patient clinical variables AND a known outcome column (0=good, 1=poor outcome).",
            key="recal_uploader"
        )
        if uploaded_recal is None:
            st.info(
                "Upload a CSV file containing historical patient data with known outcomes. "
                "The outcome column should be binary: 1 = poor outcome "
                "(non-adherence, interruption, or death), 0 = good outcome."
            )
            st.markdown("---")
            _render_recal_requirements()
            return

        try:
            df_raw = pd.read_csv(uploaded_recal)
            df_raw, _recal_log = preprocess_phia_compatible(df_raw)
            df_recal, _, _ = normalize_columns(df_raw)
            if 'patient_id' not in df_recal.columns:
                df_recal['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_recal))]
            st.success(f"✓ {len(df_recal):,} patients loaded from {uploaded_recal.name}")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        # Detect outcome columns
        candidates = detect_outcome_columns(df_recal)
        if not candidates:
            st.error(
                "No outcome column detected in your upload. "
                "Please add a column named `poor_outcome`, `outcome`, or similar "
                "containing binary values (1 = poor outcome, 0 = good outcome)."
            )
            return

        st.markdown('<p class="section-hdr">Outcome Column Selection</p>',
                    unsafe_allow_html=True)
        if len(candidates) == 1:
            col_name, reason, vc = candidates[0]
            st.info(f"Outcome column auto-detected: **`{col_name}`** — {reason}")
            outcome_col_confirmed = col_name
        else:
            col_options = [c[0] for c in candidates]
            col_reasons = {c[0]: c[1] for c in candidates}
            outcome_col_confirmed = st.selectbox(
                "Multiple outcome columns detected. Select the correct one:",
                options=col_options,
                format_func=lambda x: f"{x} — {col_reasons[x]}"
            )

    # ── Outcome normalisation ──────────────────────────────
    # Automatically recode common programme text values to binary 0/1
    # before validation. Surfaces mapping to user for transparency.
    st.markdown("---")
    df_recal, _outcome_mapping, _ambiguous = normalize_outcome_column(
        df_recal, outcome_col_confirmed
    )

    if _outcome_mapping:
        # Show what was recoded — user can verify
        recode_lines = []
        for orig, mapped in sorted(set(_outcome_mapping.items())):
            label = "poor outcome (1)" if mapped == 1 else "good outcome (0)"
            recode_lines.append(f"- `{orig}` → {label}")
        with st.expander(
            f"ℹ️ Outcome values auto-recoded ({len(set(_outcome_mapping.keys()))} "
            f"unique value(s) normalised to binary 0/1)", expanded=True
        ):
            st.markdown("\n".join(recode_lines))
            st.caption(
                "Review the mapping above. If any value has been assigned incorrectly, "
                "recode your outcome column manually before re-uploading."
            )

    if _ambiguous:
        # ── Interactive ambiguous value mapping ───────────────
        # Instead of forcing the user to leave the app and re-upload,
        # present a dropdown for each ambiguous value so they can
        # resolve the mapping in-session.
        st.warning(
            f"⚠️ **{len(_ambiguous)} outcome value(s) could not be auto-mapped.** "
            f"Please assign each value below — then recalibration will proceed."
        )
        st.markdown("**Assign ambiguous outcome values:**")

        _user_mappings = {}
        _all_resolved = True

        for _amb_val in _ambiguous:
            _user_choice = st.selectbox(
                f"What does `{_amb_val}` mean in your programme?",
                options=[
                    "— select —",
                    "Poor outcome (1) — LTFU / Death / Interruption / Failure",
                    "Good outcome (0) — Retained / Active / Suppressed / On Treatment",
                ],
                key=f"_amb_map_{_amb_val}"
            )
            if _user_choice == "— select —":
                _all_resolved = False
            elif _user_choice.startswith("Poor"):
                _user_mappings[_amb_val] = 1
            else:
                _user_mappings[_amb_val] = 0

        if not _all_resolved:
            st.info(
                "Assign all ambiguous values above to continue. "
                "If you are unsure, check your programme's outcome definitions."
            )
            st.stop()

        # Apply user mappings to the outcome column
        if _user_mappings:
            _col = df_recal[outcome_col_confirmed].copy()
            _recoded = []
            for _v in _col:
                if pd.isna(_v):
                    _recoded.append(np.nan)
                elif str(_v) in _user_mappings:
                    _recoded.append(float(_user_mappings[str(_v)]))
                else:
                    try:
                        _recoded.append(float(_v))
                    except (ValueError, TypeError):
                        _recoded.append(np.nan)
            df_recal[outcome_col_confirmed] = _recoded

            # Show confirmed mapping for audit trail
            _combined_map = {**_outcome_mapping,
                             **{k: v for k, v in _user_mappings.items()}}
            with st.expander(
                f"✅ Ambiguous values resolved — "
                f"{len(_user_mappings)} value(s) mapped by user", expanded=False
            ):
                for _orig, _mapped in sorted(_user_mappings.items()):
                    _lbl = "poor outcome (1)" if _mapped == 1 else "good outcome (0)"
                    st.markdown(f"- `{_orig}` → {_lbl} *(user-assigned)*")
                st.caption(
                    "This mapping is applied in-session only. "
                    "To make it permanent, update your outcome column "
                    "before re-uploading."
                )

    # ── Validation checks ─────────────────────────────────
    st.markdown('<p class="section-hdr">Validation Checks</p>',
                unsafe_allow_html=True)

    available_features = [f for f in FEATURES if f in df_recal.columns]
    val_results = validate_outcome_column(
        df_recal, outcome_col_confirmed, available_features)

    # Render validation checks
    check_icons = {True: '✅', False: '❌'}
    for check_key, check_data in val_results['checks'].items():
        passed = check_data['passed']
        st.markdown(
            f"{check_icons[passed]} **{check_data['label']}** — "
            f"{check_data['detail']}"
        )

    for warning in val_results['warnings']:
        st.warning(f"⚠️ {warning}")

    if not val_results['can_proceed']:
        st.error(
            "**Recalibration cannot proceed** — one or more required checks failed. "
            "Review the issues above, correct your data, and re-upload."
        )
        # Log blocked recalibration
        failed = [k for k, v in val_results['checks'].items()
                  if not v.get('passed', False)]
        n_pts = val_results['checks'].get(
            'sample_size', {}).get('value', 0)
        n_pos = val_results['checks'].get(
            'outcome_events', {}).get('value', 0)
        prev = val_results['checks'].get(
            'outcome_events', {}).get('prevalence', 0)
        log_recalibration(
            supabase, n_pts, n_pos, prev,
            local_auc=None, cal_method=None,
            passed=False, failed_checks=failed
        )
        return

    st.success("✅ All validation checks passed. Ready to run recalibration.")

    # ── Run recalibration ─────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-hdr">Run Recalibration</p>',
                unsafe_allow_html=True)

    prev = val_results['checks']['outcome_events']['prevalence']
    n_pts = val_results['checks']['sample_size']['value']
    cal_method_preview = ('Isotonic Regression'
                          if n_pts >= RECAL_ISOTONIC_MIN
                          else 'Platt Scaling (Logistic Regression)')

    st.markdown(
        f"**Programme:** {n_pts:,} patients · "
        f"Outcome prevalence: {prev*100:.1f}% · "
        f"Calibration method: {cal_method_preview}"
    )

    run_btn = st.button(
        "🔬 Run Local Recalibration",
        type="primary",
        use_container_width=True
    )

    if run_btn:
        with st.spinner("Running recalibration — fitting calibration layer and computing metrics..."):
            try:
                # Pre-sanitize and surface audit log before running
                _df_san, _san_log = sanitize_feature_matrix(df_recal, FEATURES)
                if _san_log:
                    with st.expander(
                        f"⚙️ Feature matrix coercion — "
                        f"{len(_san_log)} column(s) normalised to numeric",
                        expanded=False
                    ):
                        st.caption(
                            "The following feature columns contained non-numeric "
                            "values and were automatically coerced before recalibration."
                        )
                        for _entry in _san_log:
                            st.markdown(f"- {_entry}")

                recal = run_recalibration(
                    df_recal, outcome_col_confirmed,
                    model, available_features
                )

                # Store in session
                st.session_state['recal_results']   = recal
                st.session_state['local_auc']       = recal['auc_cal']
                st.session_state['local_threshold'] = recal['threshold']['threshold']
                st.session_state['calibrator']      = recal['calibrator']
                st.session_state['use_isotonic']    = recal['use_isotonic']
                st.session_state['recal_done']      = True

                # Log successful recalibration
                log_recalibration(
                    supabase,
                    n_patients=recal['n_patients'],
                    n_positive=recal['n_positive'],
                    prevalence=recal['prevalence'],
                    local_auc=recal['auc_cal'],
                    cal_method=recal['cal_method'],
                    passed=True,
                )

                st.success(
                    f"✅ Recalibration complete. "
                    f"Local AUC: **{recal['auc_cal']:.3f}** "
                    f"(baseline: {BASELINE_AUC})"
                )

            except Exception as e:
                st.error(f"Recalibration failed: {e}")
                return

    # ── Display results if recalibration done ─────────────
    if st.session_state.get('recal_done') and 'recal_results' in st.session_state:
        recal = st.session_state['recal_results']
        _render_recal_results(recal)


def _render_recal_requirements():
    """Show data requirements for recalibration."""
    st.markdown('<p class="section-hdr">Data Requirements</p>',
                unsafe_allow_html=True)
    reqs = [
        ('Minimum patients', f'{RECAL_MIN_PATIENTS:,}', 'Patients with known outcome'),
        ('Minimum positive events', f'{RECAL_MIN_POS_EVENTS:,}',
         'Patients with poor outcome = 1'),
        ('Maximum outcome missingness',
         f'{RECAL_MAX_OUTCOME_MISS*100:.0f}%',
         'Missing values in outcome column'),
        ('Outcome format', 'Binary 0/1',
         '1 = poor outcome, 0 = good outcome'),
        ('Recommended minimum', '500+',
         'Enables isotonic regression for better calibration'),
    ]
    req_df = pd.DataFrame(reqs,
                          columns=['Requirement', 'Threshold', 'Notes'])
    st.dataframe(req_df, use_container_width=True)


def _render_recal_results(recal):
    """Render full recalibration results and pilot validation summary."""
    st.markdown("---")
    st.markdown('<p class="section-hdr">Local Validation Results</p>',
                unsafe_allow_html=True)

    # ── Key metrics ───────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    ci_str = (f"(95% CI: {recal['auc_ci_lo']:.3f}–{recal['auc_ci_hi']:.3f})"
              if recal['auc_ci_lo'] else "")
    auc_delta = recal['auc_cal'] - recal['baseline_auc']
    delta_colour = '#3fb950' if auc_delta >= 0 else '#f85149'
    delta_sign = '+' if auc_delta >= 0 else ''

    with c1:
        st.markdown(
            f'<div class="metric-box" style="border-color:#3fb950">'
            f'<div class="metric-val" style="color:#3fb950">'
            f'{recal["auc_cal"]:.3f}</div>'
            f'<div class="metric-lbl">Local AUC</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val" style="color:{delta_colour}">'
            f'{delta_sign}{auc_delta:.3f}</div>'
            f'<div class="metric-lbl">vs Baseline (0.772)</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">'
            f'{recal["threshold"]["sensitivity"]*100:.1f}%</div>'
            f'<div class="metric-lbl">Sensitivity</div></div>',
            unsafe_allow_html=True)
    with c4:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">'
            f'{recal["threshold"]["specificity"]*100:.1f}%</div>'
            f'<div class="metric-lbl">Specificity</div></div>',
            unsafe_allow_html=True)
    with c5:
        st.markdown(
            f'<div class="metric-box">'
            f'<div class="metric-val">'
            f'{recal["brier"]:.3f}</div>'
            f'<div class="metric-lbl">Brier Score</div></div>',
            unsafe_allow_html=True)

    if ci_str:
        st.caption(f"Local AUC {ci_str} — based on {BOOTSTRAP_N:,} bootstrap samples.")

    # ── ROC curve ─────────────────────────────────────────
    st.markdown('<p class="section-hdr">ROC Curve — Local Validation</p>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    ax.plot(recal['fpr'], recal['tpr'],
            color='#21d4fd', lw=2,
            label=f"Local model (AUC = {recal['auc_cal']:.3f})")
    ax.plot([0, 1], [0, 1],
            color='#8b949e', lw=1, linestyle='--',
            label='Random classifier (AUC = 0.500)')
    ax.axhline(recal['threshold']['sensitivity'],
               color='#f0a500', lw=1, linestyle=':',
               label=f"Optimal threshold: {recal['threshold']['threshold']:.3f}")
    ax.set_xlabel('False Positive Rate (1 − Specificity)',
                  color='#8b949e', fontsize=9)
    ax.set_ylabel('True Positive Rate (Sensitivity)',
                  color='#8b949e', fontsize=9)
    ax.set_title(
        f'ROC Curve — Local Validation\n'
        f'(n={recal["n_patients"]:,} patients, '
        f'{recal["n_positive"]:,} positive outcomes, '
        f'{recal["prevalence"]*100:.1f}% prevalence)',
        color='#e6edf3', fontsize=9, pad=10)
    ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#cdd9e5')
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#30363d')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Confusion matrix at optimal threshold ─────────────
    st.markdown('<p class="section-hdr">Confusion Matrix at Optimal Threshold</p>',
                unsafe_allow_html=True)
    t = recal['threshold']
    cm_data = pd.DataFrame({
        '': ['Predicted: POOR OUTCOME', 'Predicted: GOOD OUTCOME'],
        'Actual: POOR OUTCOME': [f"✅ TP: {t['tp']:,}", f"❌ FN: {t['fn']:,}"],
        'Actual: GOOD OUTCOME': [f"❌ FP: {t['fp']:,}", f"✅ TN: {t['tn']:,}"],
    }).set_index('')
    st.dataframe(cm_data, use_container_width=True)
    st.caption(
        f"At threshold {t['threshold']:.3f} — "
        f"PPV: {t['ppv']*100:.1f}% · NPV: {t['npv']*100:.1f}% · "
        f"Calibration method: {recal['cal_method']}"
    )

    # ── Pilot Validation Summary card ─────────────────────
    st.markdown("---")
    st.markdown('<p class="section-hdr">Pilot Validation Summary</p>',
                unsafe_allow_html=True)
    st.markdown(
        "_This summary is formatted for funder and programme director audiences._"
    )

    perf_interp = (
        "substantially better than" if recal['auc_cal'] > recal['baseline_auc'] + 0.05
        else "consistent with" if abs(recal['auc_cal'] - recal['baseline_auc']) <= 0.05
        else "below"
    )

    brier_interp = (
        "excellent" if recal['brier'] < 0.10
        else "good" if recal['brier'] < 0.15
        else "moderate" if recal['brier'] < 0.20
        else "poor"
    )

    ci_str = (f" (95% CI: {recal['auc_ci_lo']:.3f}–{recal['auc_ci_hi']:.3f})"
              if recal['auc_ci_lo'] else "")

    # ── Two-column card layout — prevents text clustering ────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"""<div style="border:1px solid #3fb950;border-radius:8px;
            padding:16px;background:#0d2137;height:100%">
            <div style="color:#3fb950;font-size:0.85rem;font-weight:700;
                margin-bottom:10px;text-transform:uppercase;letter-spacing:1px">
                Performance Metrics
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:0.83rem">
            <tr><td style="padding:5px 6px;color:#8b949e;width:50%">Local AUC</td>
                <td style="padding:5px 6px;color:#3fb950;font-weight:700">
                {recal['auc_cal']:.3f}{ci_str}</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">vs Baseline (0.772)</td>
                <td style="padding:5px 6px;color:#cdd9e5">{perf_interp.capitalize()}</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Sensitivity</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['sensitivity']*100:.1f}%</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Specificity</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['specificity']*100:.1f}%</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">PPV</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['ppv']*100:.1f}%</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">NPV</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['npv']*100:.1f}%</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""<div style="border:1px solid #444c56;border-radius:8px;
            padding:16px;background:#161b22;height:100%">
            <div style="color:#21d4fd;font-size:0.85rem;font-weight:700;
                margin-bottom:10px;text-transform:uppercase;letter-spacing:1px">
                Validation Details
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:0.83rem">
            <tr><td style="padding:5px 6px;color:#8b949e;width:55%">Validation cohort</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['n_patients']:,} patients</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Outcome prevalence</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['prevalence']*100:.1f}%</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Positive events</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['n_positive']:,}</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Brier score</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['brier']:.3f} ({brier_interp})</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Optimal threshold</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['threshold']['threshold']:.3f}</td></tr>
            <tr><td style="padding:5px 6px;color:#8b949e">Calibration method</td>
                <td style="padding:5px 6px;color:#cdd9e5">{recal['cal_method'].split(" (")[0]}</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background:#1c1a10;border-left:3px solid #e3b341;
        border-radius:0 6px 6px 0;padding:10px 14px;margin-top:12px;font-size:0.8rem;color:#e3b341">
        ⚠️ This validation used historical retrospective data. Prospective validation is
        recommended before full operational deployment. All outputs require review by
        qualified programme and clinical staff before use.
    </div>""", unsafe_allow_html=True)

    # ── Download pilot summary ────────────────────────────
    summary_csv = pd.DataFrame([{
        'Metric': 'Local AUC',
        'Value': f"{recal['auc_cal']:.3f}",
        'Notes': f"95% CI: {recal['auc_ci_lo']:.3f}–{recal['auc_ci_hi']:.3f}" if recal['auc_ci_lo'] else 'Bootstrap CI unavailable',
    }, {
        'Metric': 'Baseline AUC (Nigerian cohort)',
        'Value': f"{recal['baseline_auc']:.3f}",
        'Notes': 'Pre-recalibration reference',
    }, {
        'Metric': 'Sensitivity at optimal threshold',
        'Value': f"{recal['threshold']['sensitivity']*100:.1f}%",
        'Notes': f"Threshold: {recal['threshold']['threshold']:.3f}",
    }, {
        'Metric': 'Specificity at optimal threshold',
        'Value': f"{recal['threshold']['specificity']*100:.1f}%",
        'Notes': '',
    }, {
        'Metric': 'PPV',
        'Value': f"{recal['threshold']['ppv']*100:.1f}%",
        'Notes': '',
    }, {
        'Metric': 'NPV',
        'Value': f"{recal['threshold']['npv']*100:.1f}%",
        'Notes': '',
    }, {
        'Metric': 'Brier Score',
        'Value': f"{recal['brier']:.4f}",
        'Notes': 'Lower is better. <0.10 excellent, <0.15 good',
    }, {
        'Metric': 'Validation patients',
        'Value': f"{recal['n_patients']:,}",
        'Notes': '',
    }, {
        'Metric': 'Positive outcome events',
        'Value': f"{recal['n_positive']:,}",
        'Notes': f"{recal['prevalence']*100:.1f}% prevalence",
    }, {
        'Metric': 'Calibration method',
        'Value': recal['cal_method'],
        'Notes': '',
    }])

    st.download_button(
        "📥 Download Pilot Validation Summary (CSV)",
        data=summary_csv.to_csv(index=False).encode(),
        file_name="smartdaas_pilot_validation_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Local validation complete. Risk scores on the Patient Risk page will now "
        "use locally-calibrated probabilities and the locally-optimised threshold "
        "for this session. Re-upload and re-run if your programme data changes."
    )


def derive_engineered_features(df):
    """
    Derive computed model features from raw uploaded columns where possible.
    Operates on a copy. Returns (df_out, list_of_derived, list_of_defaulted).

    Features computed:
      CD4_improvement   = MostRecentCd4Count - Cd4AtStart
      weight_change     = current_weight - WeightAtStart  (if both present)
      had_interruption  = 1 if treatment_interruptions >= 1 else 0
      stage_worsened    = 1 if current WHO stage > baseline WHO stage else 0

    Any feature still missing after derivation is filled with 0 (neutral default).
    """
    df = df.copy()
    derived = []
    defaulted = []

    # ── sex_female encoding ──────────────────────────────────
    # Convert string sex values to binary 0/1 before any numeric operations
    if 'sex_female' in df.columns:
        col = df['sex_female']
        if col.dtype == object or col.dtype.name == 'category':
            female_strings = {'f', 'female', 'woman', 'w', '1', 'yes', 'true'}
            df['sex_female'] = col.astype(str).str.lower().str.strip().map(
                lambda x: 1.0 if x in female_strings else (0.0 if x in {'m', 'male', 'man', '0', 'no', 'false'} else np.nan)
            ).fillna(0.0)

    # ── Binary categorical encoding ──────────────────────────
    # Features that may arrive as text categories — encode to 0/1
    # opp_infection: any value other than None/No/Negative/0 = 1
    if 'opp_infection' in df.columns:
        col = df['opp_infection']
        if col.dtype == object or col.dtype.name == 'category':
            none_strings = {'none', 'no', 'negative', '0', 'false', 'nan', ''}
            df['opp_infection'] = col.astype(str).str.lower().str.strip().map(
                lambda x: 0.0 if x in none_strings else 1.0
            )

    # side_effects: any value other than None/No/0 = 1
    if 'side_effects' in df.columns:
        col = df['side_effects']
        if col.dtype == object or col.dtype.name == 'category':
            none_strings = {'none', 'no', 'negative', '0', 'false', 'nan', ''}
            df['side_effects'] = col.astype(str).str.lower().str.strip().map(
                lambda x: 0.0 if x in none_strings else 1.0
            )

    # tb_positive: Yes/Positive/1 = 1, No/Negative/0 = 0
    if 'tb_positive' in df.columns:
        col = df['tb_positive']
        if col.dtype == object or col.dtype.name == 'category':
            pos_strings = {'yes', 'positive', '1', 'true', 'tb', 'tb positive'}
            df['tb_positive'] = col.astype(str).str.lower().str.strip().map(
                lambda x: 1.0 if x in pos_strings else 0.0
            )

    # who_stage / stage_start_num: extract numeric from text like "Stage 3" or "III"
    if 'stage_start_num' in df.columns:
        col = df['stage_start_num']
        if col.dtype == object or col.dtype.name == 'category':
            roman = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4}
            def parse_stage(x):
                s = str(x).lower().strip()
                # Roman numeral
                if s in roman: return float(roman[s])
                # "stage 3" or "who stage 3"
                import re
                m = re.search(r'(\d)', s)
                if m: return float(m.group(1))
                return np.nan
            df['stage_start_num'] = col.map(parse_stage).fillna(
                pd.to_numeric(col, errors='coerce')
            ).fillna(2.0)  # default to stage 2 if unparseable

    # ── General numeric coercion for all FEATURES ────────────
    # Ensures no string values reach .astype(float) in run_predictions
    for feat in FEATURES:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0.0)

    # ── CD4_improvement ─────────────────────────────────────
    if 'CD4_improvement' not in df.columns:
        if 'Cd4AtStart' in df.columns and 'MostRecentCd4Count' in df.columns:
            cd4_start   = pd.to_numeric(df['Cd4AtStart'],        errors='coerce')
            cd4_recent  = pd.to_numeric(df['MostRecentCd4Count'], errors='coerce')
            df['CD4_improvement'] = cd4_recent - cd4_start
            derived.append('CD4_improvement')
        else:
            df['CD4_improvement'] = 0
            defaulted.append('CD4_improvement')

    # ── weight_change ────────────────────────────────────────
    if 'weight_change' not in df.columns:
        # Try current_weight or weight_kg columns
        current_weight_col = next(
            (c for c in df.columns
             if c.lower().strip().replace(' ', '_') in
             {'weight_kg', 'current_weight', 'weight_current', 'weight_now',
              'recent_weight', 'weight_recent', 'weight'}),
            None
        )
        if current_weight_col and 'WeightAtStart' in df.columns:
            w_now   = pd.to_numeric(df[current_weight_col], errors='coerce')
            w_start = pd.to_numeric(df['WeightAtStart'],    errors='coerce')
            df['weight_change'] = w_now - w_start
            derived.append('weight_change')
        else:
            df['weight_change'] = 0
            defaulted.append('weight_change')

    # ── had_interruption ─────────────────────────────────────
    if 'had_interruption' not in df.columns:
        interruption_col = next(
            (c for c in df.columns
             if c.lower().strip().replace(' ', '_') in
             {'treatment_interruptions', 'interruptions', 'n_interruptions',
              'num_interruptions', 'number_of_interruptions', 'art_interruptions',
              'missed_appointments', 'treatment_gaps'}),
            None
        )
        if interruption_col:
            n_int = pd.to_numeric(df[interruption_col], errors='coerce').fillna(0)
            df['had_interruption'] = (n_int >= 1).astype(float)
            derived.append('had_interruption')
        else:
            df['had_interruption'] = 0
            defaulted.append('had_interruption')

    # ── stage_worsened ───────────────────────────────────────
    if 'stage_worsened' not in df.columns:
        # Look for a current WHO stage column distinct from stage_start_num
        current_stage_col = next(
            (c for c in df.columns
             if c.lower().strip().replace(' ', '_') in
             {'current_who_stage', 'who_stage_current', 'recent_who_stage',
              'who_stage_now', 'clinical_stage_current', 'stage_current'}),
            None
        )
        if current_stage_col and 'stage_start_num' in df.columns:
            stage_now   = pd.to_numeric(df[current_stage_col],  errors='coerce')
            stage_start = pd.to_numeric(df['stage_start_num'],  errors='coerce')
            df['stage_worsened'] = (stage_now > stage_start).astype(float)
            df['stage_worsened'] = df['stage_worsened'].fillna(0)
            derived.append('stage_worsened')
        else:
            df['stage_worsened'] = 0
            defaulted.append('stage_worsened')

    # ── Fill any remaining missing FEATURES with 0 ──────────
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0
            if feat not in defaulted and feat not in derived:
                defaulted.append(feat)

    return df, derived, defaulted


def run_predictions(df_in):
    # Safety net: ensure all features exist and are numeric before indexing.
    df_in = df_in.copy()
    for f in FEATURES:
        if f not in df_in.columns:
            df_in[f] = 0.0
        else:
            df_in[f] = pd.to_numeric(df_in[f], errors='coerce').fillna(0.0)
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
    st.markdown('<p class="section-hdr">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠 Home",
        "📊 Programme Dashboard",
        "📊 Patient Risk",
        "🧠 SHAP Explainability",
        "🎯 Intervention Engine",
        "🏥 Facility Intelligence",
        "📈 Cohort Intelligence",
        "💰 Economic Calculator",
        "📄 Executive Report",
        "🔬 Model Transparency",
        "🔬 Local Validation",
        "🤝 Pilot Model",
        "📖 Model Info",
        "📋 Sample Data",
        "🔐 Admin",
    ], label_visibility="collapsed")

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
            <div class="metric-lbl">Temporal AUC ← use this</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="color:#8b949e;font-size:0.7rem;text-align:center;margin-top:4px">
            Cross-val AUC: {MODEL_AUC:.3f} (optimistic — do not report)
        </div>""", unsafe_allow_html=True)
        st.success("✓ Model loaded")
    else:
        st.error(f"Model error: {MODEL_ERR[:80]}")
        st.info("Ensure cv_results.pkl and prepped_data.pkl are in the repo root.")

    st.markdown("---")
    st.markdown("""<div class="info-box">
        <strong>SmartDaaS v1.0</strong><br>
        Random Forest · 27,288 ART patients<br>
        AUC 0.963 (cross-val)<br>
        AUC 0.772 (temporal)<br><br>
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

# ═════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠 Home":

    # ── VERSION BANNER — update this string on each deployment ──
    st.markdown("""
    <div style="background:#0d2137;border:1px solid #21d4fd;border-radius:6px;
        padding:8px 16px;margin-bottom:1rem;display:flex;align-items:center;gap:12px">
        <span style="color:#21d4fd;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
            font-weight:700">v1.0 — 15 May 2026</span>
        <span style="color:#8b949e;font-size:0.75rem">
            Local recalibration engine · DUA acknowledgement gate ·
            Extended audit trail · Supabase event logging
        </span>
    </div>""", unsafe_allow_html=True)

    # ── HERO SECTION ──────────────────────────────────────
    st.markdown("""<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#0d2137 100%);
        border:1px solid #21d4fd22;border-radius:12px;padding:2rem 2.5rem;margin-bottom:1.5rem">
        <p style="font-size:0.8rem;color:#21d4fd;font-family:'IBM Plex Mono',monospace;
        text-transform:uppercase;letter-spacing:3px;margin:0 0 0.5rem 0">
        Smart Disease-as-a-Service</p>
        <h2 style="color:#e6edf3;font-size:1.6rem;font-weight:700;margin:0 0 0.75rem 0">
        AI-powered intelligence for HIV programmes</h2>
        <p style="color:#cdd9e5;font-size:1rem;margin:0 0 1rem 0;line-height:1.6">
        Identify high-risk patients. Detect underperforming facilities.
        Quantify avoidable costs. Drive better outcomes.</p>
        <p style="color:#adbac7;font-size:0.9rem;margin:0;line-height:1.6">
        SmartDaaS combines patient-level risk prediction with facility intelligence
        and economic insights into one platform. Built on programme data from
        <strong style="color:#21d4fd">27,288 HIV patients on ART</strong> (Nigerian national HIV
        programme — discovery cohort) with local recalibration support for any country context.
        Designed for PEPFAR implementing partners, Global Fund grantees, and national
        HIV programme offices across sub-Saharan Africa.</p>
    </div>""", unsafe_allow_html=True)

    # ── KEY METRICS ───────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, "0.963", "Cross-Val AUC"),
        (c2, "0.772", "Temporal AUC"),
        (c3, "87.3%", "Sensitivity"),
        (c4, "27K", "Training Records (discovery cohort)"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── TRACTION BANNER ───────────────────────────────────
    st.markdown("""<div style="background:linear-gradient(90deg,#0d2137,#0d1f17);
        border:1px solid #3fb95044;border-radius:10px;padding:1rem 2rem;
        margin:1rem 0;display:flex;justify-content:space-around;flex-wrap:wrap;gap:1rem">
        <div style="text-align:center">
            <div style="color:#3fb950;font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace">27,288</div>
            <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px">Patients trained on</div>
        </div>
        <div style="text-align:center">
            <div style="color:#3fb950;font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace">192,732</div>
            <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px">Records validated across</div>
        </div>
        <div style="text-align:center">
            <div style="color:#3fb950;font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace">6 countries</div>
            <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px">External consistency shown</div>
        </div>
        <div style="text-align:center">
            <div style="color:#3fb950;font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace">2 preprints</div>
            <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px">Under peer review</div>
        </div>
        <div style="text-align:center">
            <div style="color:#3fb950;font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace">USCHA 2026</div>
            <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px">Conference accepted</div>
        </div>
        <div style="text-align:center">
            <div style="color:#3fb950;font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace">$1.4B</div>
            <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;letter-spacing:1px">PEPFAR at risk annually</div>
        </div>
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
                <p style="font-size:0.85rem;color:#cdd9e5;margin:0;line-height:1.5">{desc}</p>
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
        <p style="color:#3fb950;font-size:1.3rem;font-weight:700;margin:0 0 0.5rem 0">
        🤝 Request a Pilot Partnership</p>
        <p style="color:#cdd9e5;margin:0 0 0.75rem 0;font-size:1rem">
        SmartDaaS is seeking one PEPFAR implementing partner for a <strong>6-month shadow analytics pilot.</strong><br>
        No workflow disruption. No new data collection. Results within weeks.</p>
        <p style="color:#21d4fd;font-size:1.1rem;font-weight:600;margin:0 0 1rem 0">
        📧 chinthalakalyani1@gmail.com</p>
        <p style="color:#adbac7;font-size:0.85rem;margin:0">
        Or explore the platform → <strong>📊 Patient Risk</strong> in the sidebar · See <strong>🤝 Pilot Model</strong> for pilot details
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

    # ── MARKET OPPORTUNITY ────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-hdr">Market Opportunity</p>', unsafe_allow_html=True)
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    for col, val, lbl, sub in [
        (c_m1, "39M+", "PLHIV on ART globally", "WHO 2023"),
        (c_m2, "$20B+", "Annual HIV programme spend", "PEPFAR + Global Fund"),
        (c_m3, "1 in 3", "Patients lost to follow-up", "Sub-Saharan Africa"),
        (c_m4, "0", "AI programme intelligence tools", "Currently in market"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val" style="font-size:1.4rem">{val}</div>
                <div class="metric-lbl">{lbl}</div>
                <div style="color:#484f58;font-size:0.68rem;margin-top:4px">{sub}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("""<div class="info-box" style="margin-top:0.75rem">
    SmartDaaS targets the <strong>programme intelligence gap</strong> in HIV — the space between
    raw EMR data and operational decision-making. No competitor currently offers patient-level
    AI risk stratification with facility benchmarking, SHAP explainability, and PEPFAR/Global Fund
    reporting alignment in a single deployable platform.
    </div>""", unsafe_allow_html=True)

    # ── ABOUT THE DEVELOPER ───────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-hdr">About the Developer</p>', unsafe_allow_html=True)
    st.markdown("""<div style="background:#161b22;border:1px solid #444c56;
        border-radius:8px;padding:1.5rem;margin-bottom:1rem">
        <p style="color:#e6edf3;font-size:1rem;font-weight:600;margin:0 0 0.5rem 0">
        Lakshmi Kalyani Chinthala</p>
        <p style="color:#cdd9e5;margin:0 0 0.75rem 0;font-size:0.9rem">
        Founder, SmartDaaS &nbsp;·&nbsp; Independent Researcher, San Francisco, CA.
        Specialising in machine learning
        applications for global health programme management, with a focus on HIV/AIDS
        in sub-Saharan Africa.</p>
        <p style="color:#adbac7;font-size:0.85rem;margin:0">
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
elif page == "📋 Sample Data":
    st.markdown("""
### Sample Data & CSV Template

Download the template below, fill it with your patient data, then upload it on the
**📊 Patient Risk** page to get risk scores.
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
elif page == "📊 Patient Risk":
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
        st.markdown('<p class="section-hdr">Upload Patient Data</p>', unsafe_allow_html=True)
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
    # ── LOAD DATA ─────────────────────────────────────────
    df_input = None
    mappings_info = []

    if uploaded is not None and not use_demo:
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
            df_raw, _derivation_log = preprocess_phia_compatible(df_raw)
            st.session_state['_phia_derivation_log'] = _derivation_log

            # ── Step 1: Column normalisation ──────────────────────
            df_mapped, missing, mappings_applied = normalize_columns(df_raw)

            if mappings_applied:
                with st.expander(f"ℹ️ Auto-mapped {len(mappings_applied)} column name(s)", expanded=False):
                    for orig, mapped in mappings_applied:
                        if mapped != '__art_inferred__':
                            st.markdown(f"- `{orig}` → `{mapped}`")

            # ── Step 2: ART status detection ──────────────────────
            art_confirmed, art_inferred, art_note = detect_art_status(df_raw)

            # ── Step 3: Tier detection ────────────────────────────
            tier, present, missing_core, standard_present, enhanced_present, tier_notes = \
                detect_tier(df_mapped, art_confirmed, art_inferred)

            # ── Step 4: Pediatric flag ────────────────────────────
            pediatric_indices = check_pediatric_patients(df_mapped)

            # ── Step 5: Render tier report ────────────────────────
            st.markdown("---")
            can_proceed = render_tier_report(
                tier, present, missing_core, standard_present,
                enhanced_present, art_confirmed, art_inferred,
                art_note, pediatric_indices, df_mapped
            )
            st.markdown("---")

            if not can_proceed:
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
                "Download the template from **📋 Sample Data** if needed."
            )
            st.stop()

    elif use_demo or uploaded is None:
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
            "Run **🔬 Local Validation** to calibrate for your programme's data."
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

    with st.spinner("Running risk predictions..."):
        df_input, X_raw, probs = run_predictions(df_input)

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
        st.markdown("**Top 3 drivers:**")
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
    💡 Go to <strong>🧠 SHAP Explainability</strong> for full per-patient waterfall charts
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
    sel_id = st.selectbox("Select patient (sorted by highest risk):", pid_list, index=best_pos)
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
elif page == "🧠 SHAP Explainability":
    st.markdown("""
### Why Did the Model Flag This Patient?

Select any patient to see a full clinical explanation of their risk score.
Each factor is shown with its exact SHAP contribution — **red increases risk, green reduces it.**
""")

    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_DEMO), min(200, len(X_DEMO)), replace=False)
    df_shap = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
    df_shap['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_shap))]
    X_shap = df_shap[FEATURES].values.astype(float)
    probs_shap = model.predict_proba(X_shap)[:, 1]
    df_shap['risk_pct'] = (probs_shap * 100).round(1)
    df_shap['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW' for p in probs_shap]

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
    sel_pos = df_shap[df_shap['patient_id'] == sel_id].index[0]
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
            Top 3 clinical drivers:<br>
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
elif page == "📖 Model Info":
    st.markdown("""
### Model Architecture and Validation

Full technical documentation of the SmartDaaS prediction model.
""")

    st.markdown('<p class="section-hdr">Model Performance</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, "0.963", "AUC Cross-Val"),
        (c2, "0.772", "AUC Temporal"),
        (c3, "87.3%", "Sensitivity"),
        (c4, "95.7%", "Specificity"),
        (c5, "27,288", "Training Records"),
    ]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Architecture</p>', unsafe_allow_html=True)
    st.markdown("""
- **Model type:** Random Forest Classifier
- **Pipeline:** StandardScaler → RandomForestClassifier
- **Training data:** 27,288 HIV-positive patients on ART (Nigerian national HIV programme, Quality of Care dataset — discovery cohort)
- **Companion analysis:** 165,444 CEPHIA specimens (HIV recency analysis — separate study, not used for ART outcome model training)
- **Features:** 15 clinical variables (see column guide on 📋 Sample Data page)
- **Cross-validation:** 5-fold stratified CV — AUC 0.963
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
# INTERVENTION RECOMMENDATION ENGINE
# ═════════════════════════════════════════════════════════════
elif page == "🎯 Intervention Engine":
    st.markdown("""
### Clinical Intervention Recommendation Engine

Not just "HIGH risk" — but **why**, and **exactly what to do next**.
Each recommendation is generated from the patient's specific clinical profile
and maps to standard HIV programme intervention protocols.
""")

    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    # Load data
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_DEMO), min(200, len(X_DEMO)), replace=False)
    df_ie = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
    df_ie['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_ie))]
    X_ie = df_ie[FEATURES].values.astype(float)
    probs_ie = model.predict_proba(X_ie)[:, 1]
    df_ie['risk_pct'] = (probs_ie * 100).round(1)
    df_ie['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                           for p in probs_ie]

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
    st.markdown('<p class="section-hdr">Clinical Recommendations</p>', unsafe_allow_html=True)

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
elif page == "📈 Cohort Intelligence":
    st.markdown("""
### Cohort Intelligence Dashboard

**Programme-level view — which subgroups are struggling, which are stable.**

Understand your cohort's risk distribution across demographic and clinical dimensions.
Identify deteriorating subgroups before they become programme failures.
""")

    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_DEMO), min(300, len(X_DEMO)), replace=False)
    df_ci = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
    df_ci['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_ci))]
    X_ci = df_ci[FEATURES].values.astype(float)
    probs_ci = model.predict_proba(X_ci)[:, 1]
    df_ci['risk_pct'] = (probs_ci * 100).round(1)
    df_ci['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW'
                           for p in probs_ci]
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

    # -- Upload widget ----------------------------------------------
    st.markdown('<p class="section-hdr">Report Data</p>', unsafe_allow_html=True)
    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploaded_rep = st.file_uploader(
            "Upload patient CSV for report", type=['csv'],
            help="Use the template from the Sample Data page."
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

    # Load data
    if use_demo_rep or uploaded_rep is None:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(300, len(X_DEMO)), replace=False)
        df_rep = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        df_rep['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_rep))]
        data_source = "Demo dataset (300 patients from training set)"
        rep_tier = 'ENHANCED'
        rep_pediatric = []
    else:
        try:
            df_raw = pd.read_csv(uploaded_rep)

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
    est_avoidable_cost = int(n_high * 0.3 * 1850)

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


                class SmartDaaSReport(FPDF):
                    def header(self):
                        self.set_fill_color(13, 17, 23)
                        self.rect(0, 0, 210, 297, 'F')
                        self.set_fill_color(33, 212, 253)
                        self.rect(0, 0, 3, 297, 'F')

                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Helvetica', 'I', 7)
                        self.set_text_color(139, 148, 158)
                        self.cell(0, 10, f'SmartDaaS v1.0  .  Decision-Support Platform  .  Page {self.page_no()}', align='C')

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

                pdf.set_auto_page_break(auto=True, margin=20)
                pdf.add_page()

                # ── TITLE PAGE ──────────────────────────────
                pdf.set_font('Helvetica', 'B', 28)
                pdf.set_text_color(33, 212, 253)
                pdf.set_y(35)
                pdf.cell(0, 12, 'SmartDaaS', align='C', ln=True)

                pdf.set_font('Helvetica', '', 14)
                pdf.set_text_color(230, 237, 243)
                pdf.cell(0, 8, 'HIV Programme Intelligence Report', align='C', ln=True)

                pdf.set_font('Helvetica', '', 10)
                pdf.set_text_color(173, 186, 199)
                pdf.ln(6)
                pdf.cell(0, 6, org_name, align='C', ln=True)
                pdf.cell(0, 6, programme_name, align='C', ln=True)
                pdf.cell(0, 6, _s(f'Report Date: {report_date.strftime("%d %B %Y")}'), align='C', ln=True)
                pdf.cell(0, 6, _s(f'Prepared by: {prepared_by}'), align='C', ln=True)
                pdf.cell(0, 6, _s(f'Data source: {data_source}'), align='C', ln=True)

                # Divider
                pdf.ln(8)
                pdf.set_draw_color(33, 212, 253)
                pdf.set_line_width(0.5)
                pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                pdf.ln(6)

                # Disclaimer
                pdf.set_font('Helvetica', 'I', 8)
                pdf.set_text_color(227, 179, 65)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(180, 5, 'IMPORTANT: SmartDaaS v1.0 is a decision-support platform designed for HIV programme intelligence and operational analytics. It is not intended to replace clinical judgment or function as an autonomous clinical decision-making system. All outputs should be reviewed and validated by qualified programme teams prior to operational use. Local validation is required before deployment within real-world programme environments.')

                # ── EXECUTIVE SUMMARY ────────────────────────
                pdf.add_page()
                pdf.set_font('Helvetica', 'B', 14)
                pdf.set_text_color(33, 212, 253)
                pdf.cell(0, 10, 'EXECUTIVE SUMMARY', ln=True)
                pdf.set_draw_color(33, 212, 253)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(4)

                # Key metrics table
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_text_color(230, 237, 243)
                pdf.cell(0, 7, 'Programme Risk Overview', ln=True)
                pdf.ln(2)

                metrics = [
                    ('Total Patients Analysed', f'{n_total:,}'),
                    ('HIGH Risk Patients (≥70%)', f'{n_high} ({pct_high:.1f}%)'),
                    ('MEDIUM Risk Patients (40-69%)', f'{n_med} ({n_med/n_total*100:.1f}%)'),
                    ('LOW Risk Patients (<40%)', f'{n_low} ({n_low/n_total*100:.1f}%)'),
                    ('Average Risk Score', f'{avg_risk:.1f}%'),
                    ('Patients with Prior ART Interruption', f'{pct_interruption:.1f}%'),
                    ('Patients with TB Co-infection', f'{pct_tb:.1f}%'),
                    ('Advanced Disease (WHO Stage 3-4)', f'{pct_adv_disease:.1f}%'),
                    ('Severe Immunosuppression (CD4 <200)', f'{pct_low_cd4:.1f}%'),
                    ('Estimated Avoidable Programme Cost', f'USD {est_avoidable_cost:,}'),
                ]

                for label, value in metrics:
                    pdf.set_fill_color(22, 27, 34)
                    pdf.set_text_color(173, 186, 199)
                    pdf.set_font('Helvetica', '', 9)
                    pdf.cell(110, 7, _s(f'  {label}'), border=0, fill=True)
                    pdf.set_text_color(33, 212, 253)
                    pdf.set_font('Helvetica', 'B', 9)
                    pdf.cell(70, 7, value, border=0, fill=True, ln=True)
                    pdf.ln(1)

                # ── KEY FINDINGS ──────────────────────────────
                pdf.ln(4)
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_text_color(230, 237, 243)
                pdf.cell(0, 7, 'Key Clinical Findings', ln=True)
                pdf.set_draw_color(63, 185, 80)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(4)

                findings = [
                    f'{pct_high:.1f}% of patients are HIGH risk (≥70% predicted probability of poor ART outcome)',
                    f'{pct_interruption:.1f}% have documented prior ART interruptions - the strongest predictor in the SmartDaaS model',
                    f'{pct_adv_disease:.1f}% presented at WHO Stage 3 or 4, indicating late diagnosis and treatment initiation',
                    f'{pct_low_cd4:.1f}% had CD4 <200 cells/uL at ART start - severely immunosuppressed',
                ]
                if pct_tb > 5:
                    findings.append(f'{pct_tb:.1f}% are TB-HIV co-infected, requiring coordinated treatment protocols')

                for finding in findings:
                    pdf.set_font('Helvetica', '', 9)
                    pdf.set_text_color(230, 237, 243)
                    pdf.cell(6, 6, chr(149), ln=False)
                    pdf.multi_cell(169, 6, finding)
                    pdf.ln(1)

                # ── FACILITY INTELLIGENCE ─────────────────────
                pdf.add_page()
                pdf.set_font('Helvetica', 'B', 14)
                pdf.set_text_color(33, 212, 253)
                pdf.cell(0, 10, 'FACILITY INTELLIGENCE', ln=True)
                pdf.set_draw_color(33, 212, 253)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(4)

                pdf.set_font('Helvetica', '', 9)
                pdf.set_text_color(173, 186, 199)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(180, 5, 'Based on Paper 2 analysis of 27,288 patients from the Nigerian national HIV programme (Chinthala 2026, submitted to BMJ Global Health). Findings are directionally consistent with evidence from South Africa (Bor et al. 2021, PLOS Medicine).')
                pdf.ln(4)

                facility_findings = [
                    ('Primary HC vs Tertiary Hospital', 'OR 1.95 (95% CI 1.45-2.61)', 'Primary HCs have nearly double the odds of composite poor outcome after patient-level adjustment', 'Structural quality improvement investment - staffing, drug supply, monitoring systems'),
                    ('NGO-Funded Facilities', 'OR 1.24 (95% CI 1.10-1.39)', 'NGO-funded facilities show independently higher odds - may reflect higher case complexity or administrative burden', 'Outcome-adjusted performance monitoring; investigate funding-to-quality translation mechanisms'),
                    ('Federal-Funded Facilities', 'OR 1.25 (95% CI 1.06-1.48)', 'Federal government funded facilities show similar independently elevated risk', 'Review programme management capacity and reporting burden at federal-funded sites'),
                    ('Female Sex - Protective Effect', 'OR 0.87 (95% CI 0.79-0.96)', 'Female sex is independently protective overall, but this advantage nearly disappears at primary health centres', 'Male-targeted interventions at secondary/tertiary; structural improvements at primary HCs benefit all'),
                ]

                for finding_title, stat, desc, action in facility_findings:
                    pdf.set_fill_color(22, 27, 34)
                    pdf.set_font('Helvetica', 'B', 9)
                    pdf.set_text_color(33, 212, 253)
                    pdf.set_x(pdf.l_margin)
                    pdf.cell(180, 7, _s(f'  {finding_title}: {stat}'), fill=True, ln=True)
                    pdf.set_font('Helvetica', '', 8.5)
                    pdf.set_text_color(230, 237, 243)
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(180, 5, _s(f'  Finding: {desc}'))
                    pdf.set_text_color(63, 185, 80)
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(180, 5, _s(f'  Action: {action}'))
                    pdf.ln(3)

                # ── TOP 10 HIGH RISK PATIENTS ─────────────────
                pdf.add_page()
                pdf.set_font('Helvetica', 'B', 14)
                pdf.set_text_color(33, 212, 253)
                pdf.cell(0, 10, 'TOP 10 HIGHEST RISK PATIENTS', ln=True)
                pdf.set_draw_color(33, 212, 253)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(2)

                pdf.set_font('Helvetica', 'I', 8)
                pdf.set_text_color(173, 186, 199)
                pdf.cell(0, 6, 'Patients requiring immediate contact and adherence support.', ln=True)
                pdf.ln(2)

                top10 = df_rep.nlargest(10, 'risk_pct')

                # Table header
                pdf.set_fill_color(22, 27, 34)
                pdf.set_font('Helvetica', 'B', 8)
                pdf.set_text_color(33, 212, 253)
                cols = [('Patient ID', 25), ('Risk Score', 20), ('Age', 12),
                        ('CD4 Start', 20), ('WHO Stage', 20), ('Interruption', 20), ('Action', 63)]
                for col_name, width in cols:
                    pdf.cell(width, 7, col_name, border=0, fill=True)
                pdf.ln()

                for _, row in top10.iterrows():
                    pdf.set_font('Helvetica', '', 8)
                    pdf.set_text_color(230, 237, 243)
                    urgency = "URGENT <24h" if row['risk_pct'] >= 90 else "48h contact" if row['risk_pct'] >= 80 else "This week"
                    cells = [
                        (str(row['patient_id']), 25),
                        (f"{row['risk_pct']:.1f}%", 20),
                        (f"{row['Age']:.0f}", 12),
                        (f"{row['Cd4AtStart']:.0f}", 20),
                        (f"Stage {row['stage_start_num']:.0f}", 20),
                        ("Yes" if row['had_interruption'] > 0.5 else "No", 20),
                        (urgency, 63),
                    ]
                    for val, width in cells:
                        pdf.cell(width, 6, val, border=0)
                    pdf.ln()
                    pdf.set_draw_color(48, 54, 61)
                    pdf.line(15, pdf.get_y(), 195, pdf.get_y())

                # ── RECOMMENDED ACTIONS ───────────────────────
                pdf.add_page()
                pdf.set_font('Helvetica', 'B', 14)
                pdf.set_text_color(33, 212, 253)
                pdf.cell(0, 10, 'RECOMMENDED PROGRAMME ACTIONS', ln=True)
                pdf.set_draw_color(33, 212, 253)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(4)

                action_sections = [
                    ('IMMEDIATE (This Week)', [
                        f'Contact the {n_high} HIGH risk patients - begin with the top 10 listed above',
                        'Activate peer navigator support for patients with prior interruption history',
                        'Schedule viral load tests for patients showing CD4 decline',
                        f'Prioritise TB-HIV co-treatment coordination for {int(pct_tb/100*n_total)} identified co-infected patients',
                    ]),
                    ('SHORT TERM (1-4 Weeks)', [
                        'Review regimen tolerability for patients with reported side effects',
                        f'Site visit to primary health centres - structural quality assessment',
                        'Initiate adherence counselling for all MEDIUM risk patients',
                        'Review diagnosis-to-ART delays and implement fast-track protocols where feasible',
                    ]),
                    ('STRATEGIC (1-3 Months)', [
                        'Consider Differentiated Service Delivery (DSD) model expansion at primary HCs',
                        'Develop outcome-adjusted performance metrics for facility-level monitoring',
                        'Male engagement strategy - flexible hours, community dispensing, peer support',
                        'Apply SmartDaaS risk intelligence framework to PEPFAR MER quarterly reporting',
                    ]),
                ]

                for section_title, actions in action_sections:
                    pdf.set_font('Helvetica', 'B', 10)
                    pdf.set_text_color(63, 185, 80)
                    pdf.cell(0, 8, section_title, ln=True)
                    for action in actions:
                        pdf.set_font('Helvetica', '', 9)
                        pdf.set_text_color(230, 237, 243)
                        pdf.cell(6, 6, chr(149), ln=False)
                        pdf.multi_cell(169, 6, action)
                    pdf.ln(3)

                # ── METHODOLOGY ───────────────────────────────
                pdf.ln(2)
                pdf.set_font('Helvetica', 'B', 10)
                pdf.set_text_color(33, 212, 253)
                pdf.cell(0, 8, 'METHODOLOGY & LIMITATIONS', ln=True)
                pdf.set_draw_color(33, 212, 253)
                pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                pdf.ln(4)

                pdf.set_font('Helvetica', '', 8.5)
                pdf.set_text_color(173, 186, 199)
                methodology_text = (
                    "Patient risk scores are generated by a Random Forest classifier trained on 27,288 HIV patients "
                    "from the Nigerian national HIV programme (discovery cohort). Cross-validation AUC: 0.963; temporal validation AUC: 0.772. "
                    "SHAP (SHapley Additive exPlanations) values provide per-patient clinical reasoning. "
                    "Facility intelligence is based on multivariable logistic regression with HC3 cluster-robust standard errors "
                    "applied to 27,288 Nigerian HIV programme patients (Chinthala 2026). "
                    "Economic estimates use USD 1,850 per averted poor outcome (Menzies et al., AIDS 2011). "
                    "All findings are observational and indicative. Prospective validation is required before programmatic application. "
                    "SmartDaaS v1.0 is a decision-support platform for HIV programme intelligence. "\
                    "All outputs require review by qualified programme teams prior to operational use. "
                    "Code: github.com/Kchinthala15/smartdaas-hiv-validation"
                )
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(180, 5, methodology_text)

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
### Programme Intelligence Dashboard

**Everything at a glance — for programme managers and donor reporting meetings.**
This is the screen you leave on during a presentation.
""")

    if not MODEL_OK:
        st.error("Model not loaded.")
        st.stop()

    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_DEMO), min(300, len(X_DEMO)), replace=False)
    df_dash = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
    df_dash['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_dash))]
    facility_pool = ["Kano General (Tertiary)","Lagos Island GH (Tertiary)",
                     "Abuja Primary HC (Primary)","Enugu State HF (Secondary)",
                     "Ibadan HC (Primary)","Port Harcourt GH (Secondary)"]
    facility_levels = {"Kano General (Tertiary)":"Tertiary","Lagos Island GH (Tertiary)":"Tertiary",
                       "Abuja Primary HC (Primary)":"Primary","Enugu State HF (Secondary)":"Secondary",
                       "Ibadan HC (Primary)":"Primary","Port Harcourt GH (Secondary)":"Secondary"}
    np.random.seed(42)
    df_dash['facility'] = np.random.choice(facility_pool, size=len(df_dash))
    df_dash['facility_level'] = df_dash['facility'].map(facility_levels)
    X_dash = df_dash[FEATURES].values.astype(float)
    probs_dash = model.predict_proba(X_dash)[:, 1]
    df_dash['risk_pct'] = (probs_dash * 100).round(1)
    df_dash['risk_label'] = ['HIGH' if p>=0.7 else 'MEDIUM' if p>=0.4 else 'LOW' for p in probs_dash]

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
elif page == "🔬 Model Transparency":
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

    c1,c2,c3,c4,c5 = st.columns(5)
    metrics = [
        (c1, "0.963", "Cross-Val AUC", "#21d4fd", "5-fold stratified CV on training data"),
        (c2, "0.772", "Temporal AUC", "#e3b341", "Held-out future patients (post-2015)"),
        (c3, "0.191", "Performance Gap", "#f85149", "CV minus temporal AUC — expected degradation"),
        (c4, "87.3%", "Sensitivity", "#21d4fd", "% HIGH risk patients correctly identified"),
        (c5, "95.7%", "Specificity", "#21d4fd", "% LOW risk patients correctly identified"),
    ]
    for col, val, lbl, color, tooltip in metrics:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:{color}">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    # ── WHAT THIS MEANS ───────────────────────────────────
    st.markdown('<p class="section-hdr">What These Numbers Mean</p>', unsafe_allow_html=True)

    st.markdown("""
**Cross-validation AUC (0.963)** measures how well the model predicts within the training data
using held-out folds. This is an optimistic estimate — the model has seen similar patients.

**Temporal AUC (0.772)** measures how well the model predicts on patients from *later time periods*
it has never seen. This is the more realistic estimate of real-world deployment performance.
AUC 0.772 means the model correctly ranks 77.2% of patient pairs (high-risk above low-risk).

**The performance gap (0.191)** is expected and normal — all ML models degrade on future data.
It does not invalidate the model. It means: the 0.772 figure, not 0.963, is your honest
estimate of operational performance before local retraining.

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
elif page == "🔬 Local Validation":
    if not MODEL_OK:
        st.error("Model not loaded. Cannot run recalibration.")
    else:
        render_recalibration_page(model)

elif page == "🤝 Pilot Model":
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

    for icon, title, duration, description, deliverables in pilot_steps:
        st.markdown(f"""<div style="background:#161b22;border:1px solid #444c56;
            border-left:4px solid #21d4fd;border-radius:0 8px 8px 0;
            padding:1.25rem;margin-bottom:1rem">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                <span style="font-size:1rem;font-weight:600;color:#e6edf3">{icon} {title}</span>
                <span style="background:#21d4fd18;border:1px solid #21d4fd44;color:#21d4fd;
                    font-size:0.75rem;padding:2px 8px;border-radius:4px">{duration}</span>
            </div>
            <p style="color:#cdd9e5;font-size:0.9rem;margin:0 0 0.75rem 0">{description}</p>
        </div>""", unsafe_allow_html=True)
        for d in deliverables:
            st.markdown(f"  ✓ {d}")
        st.markdown("")

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
    # ── FUNDING MILESTONES ─────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Funding Milestones & Use of Funds</p>', unsafe_allow_html=True)
    milestones = [
        ("Pre-seed / Grant ($150K–$500K)", [
            "2–3 shadow analytics pilots with PEPFAR implementing partners in Nigeria",
            "IeDEA longitudinal data integration for multi-country validation",
            "DHIS2 API direct integration (remove CSV upload friction)",
            "Part-time clinical advisor and data governance support",
        ]),
        ("Seed ($500K–$2M)", [
            "Full-time engineering and clinical partnerships team",
            "Multi-country expansion: Kenya, Uganda, South Africa",
            "TB/HIV co-infection and PMTCT programme modules",
            "PEPFAR implementing partner SaaS subscription model launch",
        ]),
        ("Series A ($2M+)", [
            "Ministry of Health direct integrations across 3+ countries",
            "Real-time EMR API connections (OpenMRS, DHIS2)",
            "Federated learning infrastructure for multi-site improvement",
            "WHO and Global Fund formal partnership development",
        ]),
    ]
    for stage, items in milestones:
        st.markdown(f"""<div style="background:#161b22;border-left:3px solid #21d4fd;
            border-radius:0 8px 8px 0;padding:1rem;margin-bottom:0.75rem">
            <div style="color:#21d4fd;font-weight:700;margin-bottom:0.5rem">{stage}</div>
        </div>""", unsafe_allow_html=True)
        for item in items:
            st.markdown(f"  ✓ {item}")
        st.markdown("")

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

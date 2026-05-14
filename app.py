"""
SmartDaaS MVP v0.3 — HIV Programme Intelligence Platform
Lakshmi Kalyani Chinthala | Golden Gate University
ORCID: 0009-0009-8736-6673

Fixes in v0.3:
- Robust CSV upload: case-insensitive column matching + helpful error messages
- Downloadable sample CSV template on Sample Data page
- Fixed use_demo / upload logic so upload always takes priority
- Column name auto-mapping for common alternatives
- Cleaner error states throughout
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
from io import StringIO

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
.brand-sub { font-size:0.95rem; color:#7d8590; margin-top:0.25rem; font-weight:300; }
.version-tag { display:inline-block; background:#21d4fd18; border:1px solid #21d4fd44;
    color:#21d4fd; font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
    padding:2px 8px; border-radius:4px; margin-top:0.5rem; }
.metric-box { background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:1rem; text-align:center; margin-bottom:0.5rem; }
.metric-val { font-family:'IBM Plex Mono',monospace; font-size:1.8rem;
    font-weight:600; color:#21d4fd; }
.metric-lbl { font-size:0.75rem; color:#7d8590; text-transform:uppercase; letter-spacing:0.5px; }
.risk-high { background:#2d1115; border:1px solid #f8514933; border-radius:10px;
    padding:1rem; text-align:center; color:#f85149; }
.risk-medium { background:#1c1a0f; border:1px solid #e3b34133; border-radius:10px;
    padding:1rem; text-align:center; color:#e3b341; }
.risk-low { background:#0d1f17; border:1px solid #3fb95033; border-radius:10px;
    padding:1rem; text-align:center; color:#3fb950; }
.risk-number { font-family:'IBM Plex Mono',monospace; font-size:2.5rem; font-weight:600; }
.risk-label { font-size:0.8rem; opacity:0.8; text-transform:uppercase; letter-spacing:1px; }
.section-hdr { font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#21d4fd;
    text-transform:uppercase; letter-spacing:2px; border-bottom:1px solid #21d4fd22;
    padding-bottom:0.5rem; margin-bottom:1rem; margin-top:1.5rem; }
.info-box { background:#161b22; border-left:3px solid #21d4fd; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.85rem; color:#8b949e; margin:0.75rem 0; }
.warn-box { background:#1c1208; border-left:3px solid #e3b341; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.85rem; color:#e3b341; margin:0.75rem 0; }
.success-box { background:#0d1f17; border-left:3px solid #3fb950; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.85rem; color:#3fb950; margin:0.75rem 0; }
.template-box { background:#161b22; border:1px solid #21d4fd33; border-radius:8px;
    padding:1.5rem; margin:1rem 0; }
[data-testid="stSidebar"] { background-color:#0d1117; border-right:1px solid #21262d; }
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
    missing = [f for f in FEATURES if f not in df_mapped.columns]

    return df_mapped, missing, mappings_applied


def run_predictions(df_in):
    X = df_in[FEATURES].values.astype(float)
    probs = model.predict_proba(X)[:, 1]
    df_in = df_in.copy()
    df_in['risk_score'] = probs
    df_in['risk_pct'] = (probs * 100).round(1)
    df_in['risk_label'] = pd.cut(
        probs,
        bins=[-0.001, 0.4, 0.7, 1.001],
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
    <p class="brand-sub">Smart Disease-as-a-Service · HIV Programme Intelligence Platform</p>
    <span class="version-tag">MVP v0.3 · Research Prototype · Not for Clinical Use</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-hdr">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠 Home",
        "📊 Patient Risk",
        "🧠 SHAP Explainability",
        "🏥 Facility Intelligence",
        "💰 Economic Calculator",
        "📖 Model Info",
        "📋 Sample Data",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p class="section-hdr">System Status</p>', unsafe_allow_html=True)
    if MODEL_OK:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{MODEL_AUC:.3f}</div>
            <div class="metric-lbl">Model AUC</div>
        </div>""", unsafe_allow_html=True)
        st.success("✓ Model loaded")
    else:
        st.error(f"Model error: {MODEL_ERR[:80]}")
        st.info("Ensure cv_results.pkl and prepped_data.pkl are in the repo root.")

    st.markdown("---")
    st.markdown("""<div class="info-box">
        <strong>SmartDaaS v0.3</strong><br>
        Random Forest · 192,732 records<br>
        AUC 0.963 (cross-val)<br>
        AUC 0.772 (temporal)<br><br>
        Paper 1 → <em>npj Digital Medicine</em><br>
        Paper 2 → <em>BMJ Global Health</em>
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
### AI-powered HIV Programme Intelligence

**Identify high-risk patients. Detect underperforming facilities. Quantify avoidable costs.**

SmartDaaS is a research-validated analytical framework built on 192,732 HIV patient records
from a Nigerian national ART programme. It combines patient-level adherence risk prediction
(Paper 1) with facility-level structural intelligence (Paper 2) into a unified programme
intelligence platform.
""")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, "0.963", "Cross-Val AUC"),
        (c2, "0.772", "Temporal AUC"),
        (c3, "87.3%", "Sensitivity"),
        (c4, "192K", "Training Records"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">What SmartDaaS Does</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**📊 Patient Risk**\n\nUpload patient records. Get individual risk scores with SHAP explanations. Export prioritised patient lists.")
    with c2:
        st.markdown("**🏥 Facility Intel**\n\nIdentify underperforming facilities. Care-gap heatmaps. Structural drivers after case-mix adjustment.")
    with c3:
        st.markdown("**💰 Economics**\n\nQuantify excess poor outcomes. Calculate avoidable programme costs. Build the case for investment.")
    with c4:
        st.markdown("**🎯 Actions**\n\nTier-specific intervention recommendations aligned with PEPFAR MER indicators TX_CURR, TX_PVLS, TX_ML.")

    st.markdown("---")
    st.markdown('<p class="section-hdr">Quick Start</p>', unsafe_allow_html=True)
    st.markdown("""
1. **Go to 📋 Sample Data** — download the CSV template
2. **Fill it** with your patient data (or use demo data)
3. **Go to 📊 Patient Risk** — upload and get risk scores instantly
4. **Export** your high-risk patient list for follow-up
""")

    st.markdown("---")
    st.markdown('<p class="section-hdr">Research Foundation</p>', unsafe_allow_html=True)
    st.markdown("""
> **Paper 1:** Chinthala LK. "Real-World Validation of Machine Learning Models for HIV
Treatment Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis
of 192,732 Clinical Records." *Under review at npj Digital Medicine, 2026.*

> **Paper 2:** Chinthala LK. "Facility-Level Structural Drivers of HIV Treatment Outcomes:
A Multi-Level Analysis of 27,288 Patients from a Nigerian HIV Programme and Implications
for PEPFAR and Global Fund Programming." *Under review at BMJ Global Health, 2026.*

**Target users:** HIV programme managers · PEPFAR implementing partners ·
Global Fund grantees · National HIV programme officers · MoH analytics teams
""")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Interested in a pilot?\nEmail: **chinthalakalyani1@gmail.com**\n\nGitHub: [Kchinthala15/smartdaas-hiv-validation](https://github.com/Kchinthala15/smartdaas-hiv-validation)")
    with col2:
        st.markdown("""<div class="warn-box">
⚠️ <strong>Research prototype only.</strong> SmartDaaS v0.3 is not validated for
clinical decision-making. Do not use to make individual patient treatment decisions
without clinical review and appropriate regulatory approval.
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
SmartDaaS v0.3 automatically recognises common alternative column names.
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
    if not MODEL_OK:
        st.error("Model not loaded. Ensure cv_results.pkl and prepped_data.pkl are in the repo root.")
        st.stop()

    st.markdown('<p class="section-hdr">Upload Patient Data</p>', unsafe_allow_html=True)

    # ── Upload / demo toggle ───────────────────────────────
    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload your patient CSV",
            type=['csv'],
            help="Use the template from 📋 Sample Data page. Column names are flexible."
        )
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = st.checkbox(
            "🔬 Use demo data instead",
            value=(uploaded is None),  # Default to demo only if nothing uploaded
            help="200 patients from the training set — shows how the platform works"
        )

    # ── LOAD DATA ─────────────────────────────────────────
    df_input = None
    mappings_info = []

    if uploaded is not None and not use_demo:
        try:
            df_raw = pd.read_csv(uploaded)
            st.markdown(f"""<div class="info-box">
            📂 File uploaded: <strong>{uploaded.name}</strong> — {len(df_raw):,} rows,
            {len(df_raw.columns)} columns detected
            </div>""", unsafe_allow_html=True)

            # Attempt column normalisation
            df_mapped, missing, mappings_applied = normalize_columns(df_raw)

            if mappings_applied:
                with st.expander(f"ℹ️ Auto-mapped {len(mappings_applied)} column name(s)", expanded=False):
                    for orig, mapped in mappings_applied:
                        st.markdown(f"- `{orig}` → `{mapped}`")

            if missing:
                st.error(f"❌ {len(missing)} required column(s) still missing after auto-mapping:")
                for m in missing:
                    st.markdown(f"- **`{m}`** — {FEAT_DESCRIPTIONS.get(m, '')} (valid range: {FEAT_RANGES.get(m, '')})")
                st.markdown("""<div class="warn-box">
                Go to <strong>📋 Sample Data</strong> to download the correct template,
                or check the Column Reference Guide there for all accepted column name variants.
                </div>""", unsafe_allow_html=True)
                st.stop()

            # Check for nulls
            null_counts = df_mapped[FEATURES].isnull().sum()
            has_nulls = null_counts[null_counts > 0]
            if len(has_nulls) > 0:
                with st.expander(f"⚠️ Missing values detected in {len(has_nulls)} column(s) — will be filled with column median"):
                    for col, cnt in has_nulls.items():
                        st.markdown(f"- `{col}`: {cnt} missing ({cnt/len(df_mapped)*100:.1f}%)")
                for col in has_nulls.index:
                    df_mapped[col] = df_mapped[col].fillna(df_mapped[col].median())

            if 'patient_id' not in df_mapped.columns:
                df_mapped['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_mapped))]

            df_input = df_mapped
            st.markdown(f"""<div class="success-box">
            ✓ <strong>{len(df_input):,} patients</strong> loaded and ready for risk scoring
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.markdown("Make sure it's a valid CSV. Download the template from **📋 Sample Data** if needed.")
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

    # ── RUN PREDICTIONS ───────────────────────────────────
    with st.spinner("Running risk predictions..."):
        df_input, X_raw, probs = run_predictions(df_input)

    n_high = (df_input['risk_label'] == 'HIGH').sum()
    n_med = (df_input['risk_label'] == 'MEDIUM').sum()
    n_low = (df_input['risk_label'] == 'LOW').sum()
    n_tot = len(df_input)

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

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHART + TABLE ─────────────────────────────────────
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

    c_s, c_e = st.columns([1, 2])
    with c_s:
        cc = 'risk-high' if pt_lbl == 'HIGH' else 'risk-medium' if pt_lbl == 'MEDIUM' else 'risk-low'
        st.markdown(f"""<div class="{cc}" style="margin-bottom:1rem">
            <div class="risk-number">{row['risk_pct']:.1f}%</div>
            <div class="risk-label">{pt_lbl} RISK</div>
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
        lbl_type = "SHAP" if sv_ok else "Feature Importance"
        ax.set_title(f'Why is {sel_id} flagged {pt_lbl} risk? [{lbl_type}]', color='#e6edf3', fontsize=9, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── EXPORT ────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Export Results</p>', unsafe_allow_html=True)
    export = df_input[['patient_id', 'risk_pct', 'risk_label'] + FEATURES].copy()
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
and funding models. Based on 27,288 patients from the Nigerian national HIV programme (Paper 2).
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

    # PEPFAR MER
    st.markdown('<p class="section-hdr">PEPFAR MER Indicator Alignment</p>', unsafe_allow_html=True)
    mer_data = {
        'MER Indicator': ['TX_CURR', 'TX_PVLS', 'TX_RTT', 'TX_ML', 'TX_NEW'],
        'Description': ['Currently on ART', 'Viral load suppression', 'Return to treatment', 'Interruptions in treatment', 'New ART initiations'],
        'SmartDaaS Signal': [
            'Base cohort for risk scoring',
            'Low CD4/high risk → proxy for VL failure risk',
            'High risk patients flagged for re-engagement',
            'ART interruption predictor (top SHAP feature)',
            'Delayed ART (>90 days) flagged in facility analysis'
        ],
        'Risk Level': ['All patients', 'HIGH', 'HIGH', 'HIGH/MEDIUM', 'Facility-level']
    }
    st.dataframe(pd.DataFrame(mer_data), use_container_width=True)


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
vs tertiary hospitals (OR 1.95, 95% CI 1.45–2.61, p&lt;0.001). Cost estimate: USD 1,850
per poor outcome (Menzies et al., <em>AIDS</em> 2011). All estimates are indicative.
</div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Programme Parameters</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        n_total = st.number_input("Total ART patients in your programme", min_value=100, max_value=500000, value=27288, step=500)
        pct_primary = st.slider("% at primary health centres", min_value=0, max_value=60, value=2) / 100
        pct_secondary = st.slider("% at secondary health facilities", min_value=0, max_value=90, value=66) / 100
    with c2:
        cost_per_outcome = st.number_input("Cost per poor outcome (USD)", min_value=500, max_value=10000, value=1850, step=50)
        tertiary_rate = st.slider("Baseline tertiary hospital poor outcome rate (%)", min_value=1, max_value=20, value=10) / 100

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
⚠️ These estimates are indicative only. They are based on observed rates from a single
Nigerian programme dataset (2006–2018) and assume your programme has similar structural
characteristics. Use for programme planning purposes only, not for budget commitments.
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
        (c5, "192,732", "Training Records"),
    ]:
        with col:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Architecture</p>', unsafe_allow_html=True)
    st.markdown("""
- **Model type:** Random Forest Classifier
- **Pipeline:** StandardScaler → RandomForestClassifier
- **Training data:** 192,732 HIV patient records (CEPHIA multi-country dataset)
- **Features:** 15 clinical variables (see column guide on 📋 Sample Data page)
- **Cross-validation:** 5-fold stratified CV — AUC 0.963
- **Temporal validation:** Held-out future patients (post-2015) — AUC 0.772
- **Explainability:** SHAP TreeExplainer with per-patient waterfall charts
- **Fairness:** Subgroup analysis across sex, age, WHO stage reported in Paper 1
""")

    st.markdown('<p class="section-hdr">Important Limitations</p>', unsafe_allow_html=True)
    st.markdown("""
1. **Temporal validation AUC of 0.772** — performance degrades on future data, as expected for any ML model. This is the honest real-world estimate.
2. **Training data source** — CEPHIA multi-country dataset. Performance on data from different settings may vary.
3. **Not validated for clinical decision-making** — this is a research prototype. Clinical deployment requires prospective validation and regulatory review.
4. **15 features only** — the model does not capture socioeconomic factors, geographic remoteness, or drug supply chain quality, which are known outcome drivers.
5. **Binary outcome** — the model predicts composite poor outcome (non-adherence OR interruption OR mortality). It does not distinguish between these outcomes.
""")

    st.markdown('<p class="section-hdr">Citation</p>', unsafe_allow_html=True)
    st.markdown("""
If you use SmartDaaS in research or programme evaluation, please cite:

> Chinthala LK. *Real-World Validation of Machine Learning Models for HIV Treatment
Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis of 192,732
Clinical Records.* Under review at npj Digital Medicine, 2026.

> Chinthala LK. *Facility-Level Structural Drivers of HIV Treatment Outcomes: A Multi-Level
Analysis of 27,288 Patients from a Nigerian HIV Programme and Implications for PEPFAR
and Global Fund Programming.* Under review at BMJ Global Health, 2026.

**Code repository:** github.com/Kchinthala15/smartdaas-hiv-validation
""")

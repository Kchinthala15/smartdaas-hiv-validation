# ───────────────────────────────────────────────────────────
#  SMARTDAAS MVP v0.2 — POLISHED HOME PAGE (BRIGHTER DARK THEME)
#  Chunk 1: Imports, Page Config, CSS, Model Loading, Header
# ───────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="SmartDaaS — HIV Adherence Risk",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS (BRIGHTER DARK THEME) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Slightly brighter main content area */
.main { background-color: #11161f; }
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Header */
.smartdaas-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d2137 100%);
    border: 1px solid #21d4fd22;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.smartdaas-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, #21d4fd08 0%, transparent 60%);
    pointer-events: none;
}
.brand-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #21d4fd;
    letter-spacing: -1px;
    margin: 0;
}
.brand-sub {
    font-size: 0.95rem;
    color: #7d8590;
    margin-top: 0.25rem;
    font-weight: 300;
}
.version-tag {
    display: inline-block;
    background: #21d4fd18;
    border: 1px solid #21d4fd44;
    color: #21d4fd;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-top: 0.5rem;
}

/* Metric cards */
.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #21d4fd;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Risk cards */
.risk-card {
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    border: 1px solid;
}
.risk-high {
    background: #2d1115;
    border-color: #f8514922;
    color: #f85149;
}
.risk-medium {
    background: #1c1a0f;
    border-color: #e3b34122;
    color: #e3b341;
}
.risk-low {
    background: #0d1f17;
    border-color: #3fb95022;
    color: #3fb950;
}
.risk-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.5rem;
    font-weight: 600;
}
.risk-label {
    font-size: 0.8rem;
    opacity: 0.8;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #21d4fd33;
    border-radius: 10px;
    padding: 1rem;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #21d4fd;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #21d4fd22;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Info box */
.info-box {
    background: #161b22;
    border-left: 3px solid #21d4fd;
    padding: 0.75rem 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #8b949e;
    margin: 1rem 0;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('cv_results.pkl', 'rb') as f:
            cv = pickle.load(f)
        model = cv['rf_model']
        with open('prepped_data.pkl', 'rb') as f:
            prep = pickle.load(f)
        features = prep['features']
        return model, features, cv['auc']
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None

model, FEATURES, MODEL_AUC = load_model()

# ── FEATURE LABELS ─────────────────────────────────────────
FEATURE_LABELS = {
    'Age': 'Age (years)',
    'sex_female': 'Sex (1=Female)',
    'Cd4AtStart': 'CD4 Count at ART Start',
    'MostRecentCd4Count': 'Most Recent CD4 Count',
    'CD4_improvement': 'CD4 Improvement',
    'stage_start_num': 'WHO Stage at Start (1-4)',
    'WeightAtStart': 'Weight at Start (kg)',
    'weight_change': 'Weight Change (kg)',
    'BMI_start': 'BMI at Start',
    'days_to_ART': 'Days: Diagnosis to ART',
    'had_interruption': 'Prior ART Interruption (0/1)',
    'opp_infection': 'Opportunistic Infection (0/1)',
    'side_effects': 'Side Effects Reported (0/1)',
    'tb_positive': 'TB Positive (0/1)',
    'stage_worsened': 'Clinical Stage Worsened (0/1)',
}

# ── HEADER BLOCK ───────────────────────────────────────────
st.markdown("""
<div class="smartdaas-header">
    <p class="brand-name">SmartDaaS</p>
    <p class="brand-sub">Smart Disease-as-a-Service · HIV Treatment Adherence Risk Platform</p>
    <span class="version-tag">MVP v0.2 · Research Prototype · Not for Clinical Use</span>
</div>
""", unsafe_allow_html=True)
# ───────────────────────────────────────────────────────────
#  CHUNK 2 — SIDEBAR + POLISHED HOME PAGE
# ───────────────────────────────────────────────────────────

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", ["🏠 Home", "📊 Predict Risk", "📖 Model Info", "📋 Sample Data"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p class="section-header">Model Status</p>', unsafe_allow_html=True)

    if model is not None:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{MODEL_AUC:.3f}</div>
            <div class="metric-lbl">Model AUC</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("✓ Model loaded")
    else:
        st.error("Model not loaded")

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>SmartDaaS</strong> uses a Random Forest model validated on 192,732 clinical records.
    Paper 1 under review at <em>npj Digital Medicine</em>.
    </div>
    """, unsafe_allow_html=True)

# ── HOME PAGE ─────────────────────────────────────────────
if page == "🏠 Home":

    # — ABOUT SECTION —
    st.markdown("""
    ### SmartDaaS — AI-powered HIV Programme Intelligence
    Identify high-risk patients and underperforming facilities.

    SmartDaaS is a research-validated AI framework built on 192,000+ patient records.
    It predicts adherence risk (AUC 0.772 temporal validation) and identifies structural facility weaknesses.
    This demo shows the analytical engine behind upcoming pilot deployments.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # — METRICS ROW —
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">0.963</div>
            <div class="metric-lbl">Cross-Val AUC</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">87.3%</div>
            <div class="metric-lbl">Sensitivity</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-val">192K</div>
            <div class="metric-lbl">Training Records</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # — WHAT SMARTDAAS DOES —
    st.markdown('<p class="section-header">What SmartDaaS Does</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📤 Upload**  
        Upload a CSV of patient records. Required columns match standard ART clinical variables.
        """)
    with col2:
        st.markdown("""
        **🤖 Predict**  
        The model scores each patient: High / Medium / Low adherence risk.
        """)
    with col3:
        st.markdown("""
        **🔍 Explain**  
        SHAP values show *why* each patient is flagged — which clinical factors drove their score.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # — RESEARCH FOUNDATION —
    st.markdown('<p class="section-header">Research Foundation</p>', unsafe_allow_html=True)
    st.markdown("""
    This tool implements the model from:

    > **Chinthala LK.** "Real-World Validation of Machine Learning Models for HIV Treatment
    Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis of 192,732
    Clinical Records." *Submitted to npj Digital Medicine, 2026.*

    **Target users:** HIV programme managers, PEPFAR implementing partners, Global Fund grantees,
    national HIV programme officers.
    """)

    # — CONTACT SECTION —
    st.markdown("---")
    st.markdown("""
    ### Interested in a pilot?
    Email: **chinthalakalyani1@gmail.com**
    """)

    # — EXISTING DISCLAIMER (kept exactly as you had it) —
    st.markdown("""
    <div class="info-box">
    ⚠️ <strong>Research prototype only.</strong> SmartDaaS v0.1 is not validated for clinical decision-making.
    Do not use to make individual patient treatment decisions without clinical review.
    </div>
    """, unsafe_allow_html=True)
# ───────────────────────────────────────────────────────────
#  CHUNK 3 — PREDICT RISK PAGE
# ───────────────────────────────────────────────────────────

elif page == "📊 Predict Risk":

    st.markdown('<p class="section-header">Upload Patient Data</p>', unsafe_allow_html=True)

    st.markdown("""
    Upload a CSV with patient records. Required columns:
    `Age, sex_female, Cd4AtStart, MostRecentCd4Count, CD4_improvement, stage_start_num,
    WeightAtStart, weight_change, BMI_start, days_to_ART, had_interruption,
    opp_infection, side_effects, tb_positive, stage_worsened`
    """)

    uploaded = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")

    # Demo mode toggle
    use_demo = st.checkbox("🔬 Use demo data (sample from training set)", value=(uploaded is None))

    if use_demo or uploaded is not None:
        if use_demo:
            # Load sample from training data
            with open('prepped_data.pkl', 'rb') as f:
                prep = pickle.load(f)
            X_demo = prep['X']
            y_demo = prep['y']
            # Take 200 random patients
            np.random.seed(42)
            idx = np.random.choice(len(X_demo), min(200, len(X_demo)), replace=False)
            df_input = pd.DataFrame(X_demo[idx], columns=FEATURES)
            df_input['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_input))]
            df_input['actual_outcome'] = y_demo[idx]
            st.info(f"Using demo data: {len(df_input)} patients from training set")
        else:
            df_input = pd.read_csv(uploaded)
            # Check columns
            missing_cols = [c for c in FEATURES if c not in df_input.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                st.stop()
            if 'patient_id' not in df_input.columns:
                df_input['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_input))]

        # ── RUN PREDICTIONS ───────────────────────────────
        with st.spinner("Running risk predictions..."):
            X = df_input[FEATURES].values
            probs = model.predict_proba(X)[:,1]
            df_input['risk_score'] = probs
            df_input['risk_pct'] = (probs * 100).round(1)

            # Risk tiers
            def tier(p):
                if p >= 0.7: return 'HIGH'
                elif p >= 0.4: return 'MEDIUM'
                return 'LOW'
            df_input['risk_label'] = [tier(p) for p in probs]

        # ── SUMMARY METRICS ───────────────────────────────
        st.markdown('<p class="section-header">Risk Summary</p>', unsafe_allow_html=True)
        n_high   = (df_input['risk_label']=='HIGH').sum()
        n_medium = (df_input['risk_label']=='MEDIUM').sum()
        n_low    = (df_input['risk_label']=='LOW').sum()
        n_total  = len(df_input)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="risk-card risk-high">
                <div class="risk-number">{n_high}</div>
                <div class="risk-label">High Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="risk-card risk-medium">
                <div class="risk-number">{n_medium}</div>
                <div class="risk-label">Medium Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="risk-card risk-low">
                <div class="risk-number">{n_low}</div>
                <div class="risk-label">Low Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            high_pct = n_high/n_total*100
            st.markdown(f"""
            <div class="metric-box" style="height:100%">
                <div class="metric-val">{high_pct:.0f}%</div>
                <div class="metric-lbl">High Risk Rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── RISK DISTRIBUTION CHART ───────────────────────
        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            st.markdown('<p class="section-header">Risk Distribution</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4,3), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            ax.hist(probs, bins=30, color='#21d4fd', alpha=0.7, edgecolor='#0d1117', linewidth=0.5)
            ax.axvline(0.7, color='#f85149', lw=1.5, linestyle='--', label='High risk threshold')
            ax.axvline(0.4, color='#e3b341', lw=1.5, linestyle='--', label='Medium threshold')
            ax.set_xlabel('Risk Score', color='#8b949e', fontsize=9)
            ax.set_ylabel('Patients', color='#8b949e', fontsize=9)
            ax.tick_params(colors='#8b949e', labelsize=8)
            for spine in ax.spines.values(): spine.set_color('#30363d')
            ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#8b949e')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── PATIENT TABLE ─────────────────────────────────
        with col_table:
            st.markdown('<p class="section-header">Patient Risk Scores</p>', unsafe_allow_html=True)
            display_cols = ['patient_id', 'risk_pct', 'risk_label', 'Age',
                           'Cd4AtStart', 'stage_start_num', 'had_interruption']
            display_cols = [c for c in display_cols if c in df_input.columns]

            def color_risk(val):
                if val == 'HIGH':   return 'background-color: #2d1115; color: #f85149'
                if val == 'MEDIUM': return 'background-color: #1c1a0f; color: #e3b341'
                return 'background-color: #0d1f17; color: #3fb950'

            df_display = df_input[display_cols].sort_values('risk_pct', ascending=False)
            styled = df_display.style.applymap(color_risk, subset=['risk_label'])
            st.dataframe(styled, height=280, use_container_width=True)

        # ── SHAP EXPLANATIONS ─────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Feature Importance (SHAP)</p>', unsafe_allow_html=True)

        with st.spinner("Computing SHAP explanations..."):
            sample_size = min(100, len(X))
            idx_s = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx_s]

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)
            if isinstance(sv, list): sv = sv[1]
            mean_shap = np.abs(sv).mean(axis=0)

        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        sorted_idx = np.argsort(mean_shap)
        feat_names = [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in sorted_idx]
        colors = ['#21d4fd' if mean_shap[i] >= np.percentile(mean_shap, 60)
                  else '#0072b2' for i in sorted_idx]
        bars = ax.barh(range(len(feat_names)), mean_shap[sorted_idx],
                       color=colors, height=0.65, edgecolor='#0d1117', linewidth=0.3)
        for i, (bar, val) in enumerate(zip(bars, mean_shap[sorted_idx])):
            ax.text(val + 0.001, i, f'{val:.4f}',
                    va='center', fontsize=7.5, color='#8b949e')
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(feat_names, fontsize=8.5, color='#e6edf3')
        ax.set_xlabel('Mean |SHAP Value|  (average impact on risk prediction)', color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#30363d')
        ax.set_title('Global Feature Importance — All Patients', color='#e6edf3', fontsize=10, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── INDIVIDUAL PATIENT EXPLORER ───────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Individual Patient Explorer</p>', unsafe_allow_html=True)

        patient_ids = df_input['patient_id'].tolist()
        selected_id = st.selectbox("Select patient to inspect:", patient_ids,
                                    index=df_input['risk_pct'].idxmax())

        pt_idx = df_input[df_input['patient_id']==selected_id].index[0]
        pt_pos = df_input.index.get_loc(pt_idx)
        pt_score = df_input.loc[pt_idx, 'risk_pct']
        pt_label = df_input.loc[pt_idx, 'risk_label']
        pt_X = X[pt_pos:pt_pos+1]
        pt_sv = explainer.shap_values(pt_X)
        if isinstance(pt_sv, list): pt_sv = pt_sv[1]
        pt_sv = pt_sv[0]

        col_score, col_explain = st.columns([1, 2])
        with col_score:
            card_class = f"risk-{'high' if pt_label=='HIGH' else 'medium' if pt_label=='MEDIUM' else 'low'}"
            st.markdown(f"""
            <div class="risk-card {card_class}" style="margin-bottom:1rem">
                <div class="risk-number">{pt_score:.1f}%</div>
                <div class="risk-label">{pt_label} RISK</div>
            </div>
            """, unsafe_allow_html=True)

            pt_data = df_input.loc[pt_idx, FEATURES]
            for feat in FEATURES[:8]:
                label = FEATURE_LABELS.get(feat, feat)
                val = pt_data[feat]
                st.markdown(f"**{label}:** `{val:.1f}`")

        with col_explain:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            sorted_pt = np.argsort(np.abs(pt_sv))
            top_n = 10
            top_idx = sorted_pt[-top_n:]
            vals = pt_sv[top_idx]
            names = [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in top_idx]
            colors_pt = ['#f85149' if v > 0 else '#3fb950' for v in vals]
            ax.barh(range(len(names)), vals, color=colors_pt,
                    height=0.65, edgecolor='#0d1117', linewidth=0.3)
            ax.axvline(0, color='#8b949e', lw=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8.5, color='#e6edf3')
            ax.set_xlabel('SHAP Value (red = increases risk, green = reduces risk)',
                         color='#8b949e', fontsize=8)
            ax.tick_params(colors='#8b949e', labelsize=8)
            for spine in ax.spines.values(): spine.set_color('#30363d')
            ax.set_title(f'Why is {selected_id} {pt_label} risk?',
                        color='#e6edf3', fontsize=10, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── EXPORT ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Export Results</p>', unsafe_allow_html=True)

        export_df = df_input[['patient_id','risk_pct','risk_label'] +
                              [c for c in FEATURES if c in df_input.columns]].copy()
        csv_bytes = export_df.to_csv(index=False).encode()

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Download Risk Scores (CSV)",
                data=csv_bytes,
                file_name="smartdaas_risk_scores.csv",
                mime="text/csv"
            )
        with col_dl2:
            high_risk_df = export_df[export_df['risk_label']=='HIGH']
            st.download_button(
                f"🚨 Download High Risk Patients Only ({n_high})",
                data=high_risk_df.to_csv(index=False).encode(),
                file_name="smartdaas_high_risk.csv",
                mime="text/csv"
            )
# ───────────────────────────────────────────────────────────
#  CHUNK 4 — MODEL INFO PAGE + SAMPLE DATA PAGE
# ───────────────────────────────────────────────────────────

elif page == "📖 Model Info":

    st.markdown('<p class="section-header">Model Architecture</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Algorithm:** Random Forest Classifier  
        **Trees:** 100 estimators  
        **Training set:** 192,732 patients  
        **Cross-validation:** 5-fold stratified  

        **Performance (Paper 1):**

        | Metric | Value |
        |--------|-------|
        | AUC (cross-val) | 0.963 |
        | AUC (temporal) | 0.772 |
        | Sensitivity | 87.3% |
        | Specificity | 95.7% |
        | Brier Score | 0.079 |
        """)

    with col2:
        st.markdown("**15 Clinical Features:**")
        for feat, label in FEATURE_LABELS.items():
            st.markdown(f"- `{feat}` — {label}")

    st.markdown("---")
    st.markdown('<p class="section-header">Important Caveats</p>', unsafe_allow_html=True)

    st.warning("""
    **This is a research prototype (v0.2).** Before any clinical deployment:
    - External validation on independent cohorts required  
    - Prospective clinical evaluation required  
    - Regulatory approval required (FDA, local health authority)  
    - HIPAA/data security review required  
    - Clinical ethics review required  

    **Citation:**  
    Chinthala LK. Real-world validation of ML models for HIV treatment adherence prediction.  
    Submitted to npj Digital Medicine. 2026.
    """)

# ───────────────────────────────────────────────────────────

elif page == "📋 Sample Data":

    st.markdown('<p class="section-header">Sample CSV Template</p>', unsafe_allow_html=True)
    st.markdown("Download this template, fill in your patient data, and upload it in the Predict tab.")

    template_df = pd.DataFrame({
        'patient_id': ['PT-0001','PT-0002','PT-0003'],
        'Age': [34, 45, 28],
        'sex_female': [1, 0, 1],
        'Cd4AtStart': [250, 450, 120],
        'MostRecentCd4Count': [380, 520, 95],
        'CD4_improvement': [130, 70, -25],
        'stage_start_num': [2, 1, 3],
        'WeightAtStart': [58.0, 72.0, 51.0],
        'weight_change': [2.5, -1.0, -3.0],
        'BMI_start': [21.3, 24.8, 19.6],
        'days_to_ART': [45, 120, 7],
        'had_interruption': [0, 1, 0],
        'opp_infection': [0, 0, 1],
        'side_effects': [1, 0, 0],
        'tb_positive': [0, 0, 1],
        'stage_worsened': [0, 0, 1],
    })

    st.dataframe(template_df, use_container_width=True)

    st.download_button(
        "📥 Download Template CSV",
        data=template_df.to_csv(index=False).encode(),
        file_name="smartdaas_template.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown('<p class="section-header">Column Definitions</p>', unsafe_allow_html=True)

    for feat, label in FEATURE_LABELS.items():
        st.markdown(f"**`{feat}`** — {label}")

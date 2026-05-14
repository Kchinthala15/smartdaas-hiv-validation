"""
SmartDaaS MVP v0.2 — HIV Treatment Adherence Risk Platform
Lakshmi Kalyani Chinthala | Golden Gate University
"""

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

st.set_page_config(
    page_title="SmartDaaS — HIV Adherence Risk",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.smartdaas-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d2137 100%);
    border: 1px solid #21d4fd22; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 2rem;
}
.brand-name { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem;
    font-weight: 600; color: #21d4fd; letter-spacing: -1px; margin: 0; }
.brand-sub { font-size: 0.95rem; color: #7d8590; margin-top: 0.25rem; font-weight: 300; }
.version-tag { display: inline-block; background: #21d4fd18; border: 1px solid #21d4fd44;
    color: #21d4fd; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    padding: 2px 8px; border-radius: 4px; margin-top: 0.5rem; }
.metric-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 1rem; text-align: center; }
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem;
    font-weight: 600; color: #21d4fd; }
.metric-lbl { font-size: 0.75rem; color: #7d8590; text-transform: uppercase; letter-spacing: 0.5px; }
.risk-card { border-radius: 10px; padding: 1.2rem 1.5rem; text-align: center; border: 1px solid; }
.risk-high   { background: #2d1115; border-color: #f8514922; color: #f85149; }
.risk-medium { background: #1c1a0f; border-color: #e3b34122; color: #e3b341; }
.risk-low    { background: #0d1f17; border-color: #3fb95022; color: #3fb950; }
.risk-number { font-family: 'IBM Plex Mono', monospace; font-size: 2.5rem; font-weight: 600; }
.risk-label  { font-size: 0.8rem; opacity: 0.8; margin-top: 0.3rem;
    text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #21262d; }
.section-header { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    color: #21d4fd; text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #21d4fd22; padding-bottom: 0.5rem; margin-bottom: 1rem; }
.info-box { background: #161b22; border-left: 3px solid #21d4fd; padding: 0.75rem 1rem;
    border-radius: 0 6px 6px 0; font-size: 0.85rem; color: #8b949e; margin: 1rem 0; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        with open('cv_results.pkl', 'rb') as f:
            cv = pickle.load(f)
        with open('prepped_data.pkl', 'rb') as f:
            prep = pickle.load(f)
        return cv['rf_model'], prep['features'], float(cv['auc'])
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None

model, FEATURES, MODEL_AUC = load_model()

LABELS = {
    'Age': 'Age (years)', 'sex_female': 'Sex (1=Female)',
    'Cd4AtStart': 'CD4 at ART Start', 'MostRecentCd4Count': 'Most Recent CD4',
    'CD4_improvement': 'CD4 Improvement', 'stage_start_num': 'WHO Stage (1-4)',
    'WeightAtStart': 'Weight at Start (kg)', 'weight_change': 'Weight Change (kg)',
    'BMI_start': 'BMI at Start', 'days_to_ART': 'Days: Diagnosis to ART',
    'had_interruption': 'Prior ART Interruption', 'opp_infection': 'Opportunistic Infection',
    'side_effects': 'Side Effects', 'tb_positive': 'TB Positive',
    'stage_worsened': 'Stage Worsened',
}

st.markdown("""
<div class="smartdaas-header">
    <p class="brand-name">SmartDaaS</p>
    <p class="brand-sub">Smart Disease-as-a-Service · HIV Treatment Adherence Risk Platform</p>
    <span class="version-tag">MVP v0.2 · Research Prototype · Not for Clinical Use</span>
</div>
""", unsafe_allow_html=True)

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
        </div>""", unsafe_allow_html=True)
        st.success("✓ Model loaded")
    else:
        st.error("Model not loaded")
    st.markdown("---")
    st.markdown("""<div class="info-box">
    <strong>SmartDaaS</strong> uses a Random Forest model validated on 192,732 clinical records.
    Paper 1 under review at <em>npj Digital Medicine</em>.
    </div>""", unsafe_allow_html=True)

# ── HOME ──────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    ### SmartDaaS — AI-powered HIV Programme Intelligence
    Identify high-risk patients and underperforming facilities.

    SmartDaaS is a research-validated AI framework built on 192,732 patient records.
    It predicts adherence risk (AUC 0.772 temporal validation) and identifies structural
    facility weaknesses. This demo shows the analytical engine behind upcoming pilot deployments.
    """)

    c1, c2, c3 = st.columns(3)
    for col, val, lbl in [(c1,"0.963","Cross-Val AUC"),(c2,"87.3%","Sensitivity"),(c3,"192K","Training Records")]:
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">What SmartDaaS Does</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("**📤 Upload**\nUpload a CSV of patient records.")
    with c2: st.markdown("**🤖 Predict**\nScores each patient: High / Medium / Low risk.")
    with c3: st.markdown("**🔍 Explain**\nSHAP values show why each patient is flagged.")

    st.markdown("---")
    st.markdown("""
    > **Chinthala LK.** "Real-World Validation of Machine Learning Models for HIV Treatment
    Adherence Prediction." *Submitted to npj Digital Medicine, 2026.*

    **Target users:** HIV programme managers, PEPFAR partners, Global Fund grantees.
    """)
    st.markdown("---")
    st.markdown("### Interested in a pilot?\nEmail: **chinthalakalyani1@gmail.com**")
    st.markdown("""<div class="info-box">
    ⚠️ <strong>Research prototype only.</strong> SmartDaaS v0.2 is not validated for
    clinical decision-making. Do not use for individual patient decisions without clinical review.
    </div>""", unsafe_allow_html=True)

# ── PREDICT ───────────────────────────────────────────────
elif page == "📊 Predict Risk":
    st.markdown('<p class="section-header">Upload Patient Data</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
    use_demo = st.checkbox("🔬 Use demo data", value=(uploaded is None))

    if use_demo or uploaded is not None:
        if use_demo:
            with open('prepped_data.pkl', 'rb') as f:
                prep = pickle.load(f)
            rng = np.random.RandomState(42)
            idx = rng.choice(len(prep['X']), min(200, len(prep['X'])), replace=False)
            df = pd.DataFrame(prep['X'][idx], columns=FEATURES)
            df['patient_id'] = [f"PT-{i:04d}" for i in range(len(df))]
            st.info(f"Demo mode: {len(df)} patients")
        else:
            df = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()
            if 'patient_id' not in df.columns:
                df['patient_id'] = [f"PT-{i:04d}" for i in range(len(df))]

        # Predictions
        X = df[FEATURES].values
        probs = model.predict_proba(X)[:, 1]
        df['risk_pct'] = (probs * 100).round(1)
        df['risk_label'] = ['HIGH' if p >= 0.7 else 'MEDIUM' if p >= 0.4 else 'LOW' for p in probs]

        # Summary cards
        n_high = (df['risk_label'] == 'HIGH').sum()
        n_med  = (df['risk_label'] == 'MEDIUM').sum()
        n_low  = (df['risk_label'] == 'LOW').sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="risk-card risk-high"><div class="risk-number">{n_high}</div><div class="risk-label">High Risk</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="risk-card risk-medium"><div class="risk-number">{n_med}</div><div class="risk-label">Medium Risk</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="risk-card risk-low"><div class="risk-number">{n_low}</div><div class="risk-label">Low Risk</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-box"><div class="metric-val">{n_high/len(df)*100:.0f}%</div><div class="metric-lbl">High Risk Rate</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Chart + Table
        c_chart, c_table = st.columns([1, 2])
        with c_chart:
            fig, ax = plt.subplots(figsize=(4,3), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            ax.hist(probs, bins=30, color='#21d4fd', alpha=0.7, edgecolor='#0d1117')
            ax.axvline(0.7, color='#f85149', lw=1.5, linestyle='--', label='High threshold')
            ax.axvline(0.4, color='#e3b341', lw=1.5, linestyle='--', label='Medium threshold')
            ax.set_xlabel('Risk Score', color='#8b949e', fontsize=9)
            ax.set_ylabel('Patients', color='#8b949e', fontsize=9)
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#8b949e')
            for sp in ax.spines.values(): sp.set_color('#30363d')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c_table:
            show_cols = [c for c in ['patient_id','risk_pct','risk_label','Age','Cd4AtStart','stage_start_num','had_interruption'] if c in df.columns]
            df_show = df[show_cols].copy()
            df_show['risk_label'] = df_show['risk_label'].map({'HIGH':'🔴 HIGH','MEDIUM':'🟡 MEDIUM','LOW':'🟢 LOW'})
            df_show = df_show.sort_values('risk_pct', ascending=False)
            st.dataframe(df_show, height=280, use_container_width=True)

        # SHAP — extract RF from pipeline, scale first
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Feature Importance (SHAP)</p>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP..."):
            n_sample = min(80, len(X))
            rng2 = np.random.RandomState(42)
            s_idx = rng2.choice(len(X), n_sample, replace=False)
            X_s = X[s_idx]
            scaler = model.named_steps['sc']
            clf    = model.named_steps['clf']
            X_s_scaled = scaler.transform(X_s)
            explainer  = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_s_scaled)
            if isinstance(sv, list): sv = sv[1]
            mean_sv = np.abs(sv).mean(axis=0)

        feat_list = list(FEATURES)
        order = np.argsort(mean_sv)
        f_names = [LABELS.get(feat_list[int(i)], feat_list[int(i)]) for i in order]
        f_vals  = mean_sv[order]
        f_colors = ['#21d4fd' if v >= np.percentile(mean_sv, 60) else '#0072b2' for v in f_vals]

        fig, ax = plt.subplots(figsize=(7,4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        ax.barh(range(len(f_names)), f_vals, color=f_colors, height=0.65, edgecolor='#0d1117')
        for i, v in enumerate(f_vals):
            ax.text(v+0.001, i, f'{v:.4f}', va='center', fontsize=7.5, color='#8b949e')
        ax.set_yticks(range(len(f_names))); ax.set_yticklabels(f_names, fontsize=8.5, color='#e6edf3')
        ax.set_xlabel('Mean |SHAP Value|', color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.set_title('Global Feature Importance', color='#e6edf3', fontsize=10, pad=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Individual patient
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Individual Patient Explorer</p>', unsafe_allow_html=True)
        pid_list = df['patient_id'].tolist()
        best_idx = int(df['risk_pct'].values.argmax())
        sel_id = st.selectbox("Select patient:", pid_list, index=best_idx)

        row = df[df['patient_id'] == sel_id].iloc[0]
        row_pos = df[df['patient_id'] == sel_id].index[0]
        pos = df.index.get_loc(row_pos)
        pt_X_scaled = scaler.transform(X[pos:pos+1])
        pt_sv = explainer.shap_values(pt_X_scaled)
        if isinstance(pt_sv, list): pt_sv = pt_sv[1]
        pt_sv = pt_sv[0]

        c_s, c_e = st.columns([1,2])
        with c_s:
            lbl = row['risk_label']
            cc = 'risk-high' if lbl=='HIGH' else 'risk-medium' if lbl=='MEDIUM' else 'risk-low'
            st.markdown(f'<div class="risk-card {cc}" style="margin-bottom:1rem"><div class="risk-number">{row["risk_pct"]:.1f}%</div><div class="risk-label">{lbl} RISK</div></div>', unsafe_allow_html=True)
            for feat in feat_list[:8]:
                st.markdown(f"**{LABELS.get(feat,feat)}:** `{row[feat]:.1f}`")

        with c_e:
            pt_order = np.argsort(np.abs(pt_sv))[-10:]
            pt_vals  = pt_sv[pt_order]
            pt_names = [LABELS.get(feat_list[int(i)], feat_list[int(i)]) for i in pt_order]
            pt_colors = ['#f85149' if v > 0 else '#3fb950' for v in pt_vals]
            fig, ax = plt.subplots(figsize=(6,4), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            ax.barh(range(len(pt_names)), pt_vals, color=pt_colors, height=0.65, edgecolor='#0d1117')
            ax.axvline(0, color='#8b949e', lw=0.8)
            ax.set_yticks(range(len(pt_names))); ax.set_yticklabels(pt_names, fontsize=8.5, color='#e6edf3')
            ax.set_xlabel('SHAP Value (red=increases risk, green=reduces risk)', color='#8b949e', fontsize=8)
            ax.tick_params(colors='#8b949e', labelsize=8)
            for sp in ax.spines.values(): sp.set_color('#30363d')
            ax.set_title(f'Why is {sel_id} {lbl} risk?', color='#e6edf3', fontsize=10, pad=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Export
        st.markdown("<br>", unsafe_allow_html=True)
        export = df[['patient_id','risk_pct','risk_label'] + feat_list].copy()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Download All Risk Scores", data=export.to_csv(index=False).encode(),
                               file_name="smartdaas_risk_scores.csv", mime="text/csv")
        with c2:
            high_df = export[export['risk_label']=='HIGH']
            st.download_button(f"🚨 Download High Risk Only ({n_high})",
                               data=high_df.to_csv(index=False).encode(),
                               file_name="smartdaas_high_risk.csv", mime="text/csv")

# ── MODEL INFO ────────────────────────────────────────────
elif page == "📖 Model Info":
    st.markdown('<p class="section-header">Model Architecture</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Algorithm:** Random Forest Classifier
        **Trees:** 100 estimators
        **Training set:** 192,732 patients
        **Cross-validation:** 5-fold stratified

        | Metric | Value |
        |--------|-------|
        | AUC (cross-val) | 0.963 |
        | AUC (temporal) | 0.772 |
        | Sensitivity | 87.3% |
        | Specificity | 95.7% |
        | Brier Score | 0.079 |
        """)
    with c2:
        st.markdown("**15 Clinical Features:**")
        for f, l in LABELS.items():
            st.markdown(f"- `{f}` — {l}")
    st.markdown("---")
    st.warning("""
    **Research prototype (v0.2).** Before clinical deployment:
    external validation, prospective evaluation, regulatory approval,
    HIPAA review, and ethics approval are all required.

    **Citation:** Chinthala LK. Real-world validation of ML models for HIV adherence.
    Submitted to npj Digital Medicine. 2026.
    """)

# ── SAMPLE DATA ───────────────────────────────────────────
elif page == "📋 Sample Data":
    st.markdown('<p class="section-header">Sample CSV Template</p>', unsafe_allow_html=True)
    template = pd.DataFrame({
        'patient_id':['PT-0001','PT-0002','PT-0003'],
        'Age':[34,45,28], 'sex_female':[1,0,1],
        'Cd4AtStart':[250,450,120], 'MostRecentCd4Count':[380,520,95],
        'CD4_improvement':[130,70,-25], 'stage_start_num':[2,1,3],
        'WeightAtStart':[58.0,72.0,51.0], 'weight_change':[2.5,-1.0,-3.0],
        'BMI_start':[21.3,24.8,19.6], 'days_to_ART':[45,120,7],
        'had_interruption':[0,1,0], 'opp_infection':[0,0,1],
        'side_effects':[1,0,0], 'tb_positive':[0,0,1], 'stage_worsened':[0,0,1],
    })
    st.dataframe(template, use_container_width=True)
    st.download_button("📥 Download Template", data=template.to_csv(index=False).encode(),
                       file_name="smartdaas_template.csv", mime="text/csv")
    st.markdown("---")
    for f, l in LABELS.items():
        st.markdown(f"**`{f}`** — {l}")

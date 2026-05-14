"""
SmartDaaS MVP v0.2 — HIV Programme Intelligence Platform
Lakshmi Kalyani Chinthala | Golden Gate University
ORCID: 0009-0009-8736-6673
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDaaS — HIV Programme Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  CSS
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
.risk-high   { background:#2d1115; border:1px solid #f8514933; border-radius:10px;
    padding:1rem; text-align:center; color:#f85149; }
.risk-medium { background:#1c1a0f; border:1px solid #e3b34133; border-radius:10px;
    padding:1rem; text-align:center; color:#e3b341; }
.risk-low    { background:#0d1f17; border:1px solid #3fb95033; border-radius:10px;
    padding:1rem; text-align:center; color:#3fb950; }
.risk-number { font-family:'IBM Plex Mono',monospace; font-size:2.5rem; font-weight:600; }
.risk-label  { font-size:0.8rem; opacity:0.8; text-transform:uppercase; letter-spacing:1px; }
.section-hdr { font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#21d4fd;
    text-transform:uppercase; letter-spacing:2px; border-bottom:1px solid #21d4fd22;
    padding-bottom:0.5rem; margin-bottom:1rem; margin-top:1.5rem; }
.info-box { background:#161b22; border-left:3px solid #21d4fd; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.85rem; color:#8b949e; margin:0.75rem 0; }
.warn-box { background:#1c1208; border-left:3px solid #e3b341; padding:0.75rem 1rem;
    border-radius:0 6px 6px 0; font-size:0.85rem; color:#e3b341; margin:0.75rem 0; }
.facility-card { background:#161b22; border:1px solid #30363d; border-radius:8px;
    padding:1rem; margin-bottom:0.5rem; }
[data-testid="stSidebar"] { background-color:#0d1117; border-right:1px solid #21262d; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
FEATURES = [
    'Age','sex_female','Cd4AtStart','MostRecentCd4Count','CD4_improvement',
    'stage_start_num','WeightAtStart','weight_change','BMI_start','days_to_ART',
    'had_interruption','opp_infection','side_effects','tb_positive','stage_worsened'
]

FEAT_LABELS = {
    'Age':'Age (years)', 'sex_female':'Sex (Female=1)',
    'Cd4AtStart':'CD4 at ART Start', 'MostRecentCd4Count':'Most Recent CD4',
    'CD4_improvement':'CD4 Improvement', 'stage_start_num':'WHO Stage (1–4)',
    'WeightAtStart':'Weight at Start (kg)', 'weight_change':'Weight Change (kg)',
    'BMI_start':'BMI at Start', 'days_to_ART':'Days: Diagnosis to ART',
    'had_interruption':'Prior ART Interruption', 'opp_infection':'Opportunistic Infection',
    'side_effects':'Side Effects Reported', 'tb_positive':'TB Positive',
    'stage_worsened':'Clinical Stage Worsened',
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
#  LOAD MODEL — safe with detailed error handling
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with open('cv_results.pkl','rb') as f:
            cv = pickle.load(f)
        with open('prepped_data.pkl','rb') as f:
            prep = pickle.load(f)
        model   = cv['rf_model']
        scaler  = model.named_steps['sc']
        clf     = model.named_steps['clf']
        auc     = float(cv['auc'])
        X_demo  = prep['X']
        y_demo  = prep['y']
        return model, scaler, clf, auc, X_demo, y_demo, True, ""
    except Exception as e:
        return None, None, None, 0.0, None, None, False, str(e)

model, SCALER, CLF, MODEL_AUC, X_DEMO, Y_DEMO, MODEL_OK, MODEL_ERR = load_model()


# ─────────────────────────────────────────────────────────────
#  HELPER — run predictions safely
# ─────────────────────────────────────────────────────────────
def run_predictions(df_in):
    X = df_in[FEATURES].values.astype(float)
    probs = model.predict_proba(X)[:, 1]
    df_in = df_in.copy()
    df_in['risk_score'] = probs
    df_in['risk_pct']   = (probs * 100).round(1)
    df_in['risk_label'] = pd.cut(
        probs,
        bins=[-0.001, 0.4, 0.7, 1.001],
        labels=['LOW','MEDIUM','HIGH']
    ).astype(str)
    return df_in, X, probs


# ─────────────────────────────────────────────────────────────
#  HELPER — SHAP (safe, always works)
# ─────────────────────────────────────────────────────────────
def compute_shap(X_raw, n_sample=80):
    """Always returns mean_shap array. Never crashes."""
    try:
        import shap
        rng   = np.random.RandomState(42)
        idx   = rng.choice(len(X_raw), min(n_sample, len(X_raw)), replace=False)
        X_s   = SCALER.transform(X_raw[idx])
        exp   = shap.TreeExplainer(CLF)
        sv    = exp.shap_values(X_s)
        if isinstance(sv, list): sv = sv[1]
        return np.abs(sv).mean(axis=0), exp
    except Exception:
        # Fallback: use RF feature importances
        return CLF.feature_importances_, None

def compute_shap_single(x_row):
    """SHAP for one patient. Never crashes."""
    try:
        import shap
        x_scaled = SCALER.transform(x_row.reshape(1,-1))
        exp = shap.TreeExplainer(CLF)
        sv  = exp.shap_values(x_scaled)
        if isinstance(sv, list): sv = sv[1]
        return sv[0], True
    except Exception:
        return CLF.feature_importances_, False


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="smartdaas-header">
    <p class="brand-name">SmartDaaS</p>
    <p class="brand-sub">Smart Disease-as-a-Service · HIV Programme Intelligence Platform</p>
    <span class="version-tag">MVP v0.2 · Research Prototype · Not for Clinical Use</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-hdr">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Home",
        "📊  Patient Risk",
        "🏥  Facility Intelligence",
        "💰  Economic Calculator",
        "📖  Model Info",
        "📋  Sample Data",
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
        st.error(f"Model error: {MODEL_ERR[:60]}")

    st.markdown("---")
    st.markdown("""<div class="info-box">
    <strong>SmartDaaS v0.2</strong><br>
    Random Forest · 192,732 records<br>
    AUC 0.963 (cross-val)<br>
    AUC 0.772 (temporal)<br><br>
    Paper 1 → <em>npj Digital Medicine</em><br>
    Paper 2 → <em>BMJ Global Health</em>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
    ### AI-powered HIV Programme Intelligence
    **Identify high-risk patients. Detect underperforming facilities. Quantify avoidable costs.**

    SmartDaaS is a research-validated analytical framework built on 192,732 HIV patient records
    from a Nigerian national ART programme. It combines patient-level adherence risk prediction
    (Paper 1) with facility-level structural intelligence (Paper 2) into a unified programme
    intelligence platform.
    """)

    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl in [
        (c1,"0.963","Cross-Val AUC"),
        (c2,"0.772","Temporal AUC"),
        (c3,"87.3%","Sensitivity"),
        (c4,"192K","Training Records"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">What SmartDaaS Does</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown("**📊 Patient Risk**\nUpload patient records. Get individual risk scores with SHAP explanations.")
    with c2: st.markdown("**🏥 Facility Intel**\nIdentify underperforming facilities. Care-gap heatmaps. Structural drivers.")
    with c3: st.markdown("**💰 Economics**\nQuantify excess poor outcomes. Calculate avoidable programme costs.")
    with c4: st.markdown("**🎯 Actions**\nTier-specific intervention recommendations aligned with PEPFAR MER indicators.")

    st.markdown("---")
    st.markdown('<p class="section-hdr">Research Foundation</p>', unsafe_allow_html=True)
    st.markdown("""
    > **Paper 1:** Chinthala LK. "Real-World Validation of Machine Learning Models for HIV
    Treatment Adherence Prediction." *Submitted to npj Digital Medicine, 2026.*

    > **Paper 2:** Chinthala LK. "Facility-Level Structural Drivers of HIV Treatment Outcomes:
    A Multi-Level Analysis of 27,288 Patients from a Nigerian HIV Programme."
    *Submitted to BMJ Global Health, 2026.*

    **Target users:** HIV programme managers · PEPFAR implementing partners ·
    Global Fund grantees · National HIV programme officers · MoH analytics teams
    """)

    st.markdown("---")
    st.markdown("### Interested in a pilot?\nEmail: **chinthalakalyani1@gmail.com**")
    st.markdown("""<div class="warn-box">
    ⚠️ <strong>Research prototype only.</strong> SmartDaaS v0.2 is not validated for
    clinical decision-making. Do not use to make individual patient treatment decisions
    without clinical review and appropriate regulatory approval.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 2 — PATIENT RISK
# ═════════════════════════════════════════════════════════════
elif page == "📊  Patient Risk":

    if not MODEL_OK:
        st.error("Model not loaded. Check cv_results.pkl and prepped_data.pkl are in the repo.")
        st.stop()

    st.markdown('<p class="section-hdr">Upload Patient Data</p>', unsafe_allow_html=True)
    st.markdown(f"""Upload a CSV with these columns:
    `{', '.join(FEATURES)}`""")

    uploaded = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
    use_demo = st.checkbox("🔬 Use demo data (200 patients from training set)",
                           value=(uploaded is None))

    # ── LOAD DATA ─────────────────────────────────────────
    df_input = None
    if use_demo:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_DEMO), min(200, len(X_DEMO)), replace=False)
        df_input = pd.DataFrame(X_DEMO[idx], columns=FEATURES)
        df_input['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_input))]
        st.info(f"Demo mode: {len(df_input)} patients loaded")

    elif uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in df_input.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
                st.markdown("Download a template from the **Sample Data** page.")
                st.stop()
            if 'patient_id' not in df_input.columns:
                df_input['patient_id'] = [f"PT-{i:04d}" for i in range(len(df_input))]
            st.success(f"✓ Uploaded: {len(df_input):,} patients")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    if df_input is None:
        st.info("Upload a CSV above or check 'Use demo data' to begin.")
        st.stop()

    # ── RUN PREDICTIONS ───────────────────────────────────
    with st.spinner("Running risk predictions..."):
        df_input, X_raw, probs = run_predictions(df_input)

    n_high = (df_input['risk_label']=='HIGH').sum()
    n_med  = (df_input['risk_label']=='MEDIUM').sum()
    n_low  = (df_input['risk_label']=='LOW').sum()
    n_tot  = len(df_input)

    # ── SUMMARY CARDS ─────────────────────────────────────
    st.markdown('<p class="section-hdr">Risk Summary</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="risk-high"><div class="risk-number">{n_high}</div><div class="risk-label">High Risk</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="risk-medium"><div class="risk-number">{n_med}</div><div class="risk-label">Medium Risk</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="risk-low"><div class="risk-number">{n_low}</div><div class="risk-label">Low Risk</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="metric-val">{n_high/n_tot*100:.0f}%</div><div class="metric-lbl">High Risk Rate</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHART + TABLE ─────────────────────────────────────
    c_chart, c_table = st.columns([1, 2])

    with c_chart:
        st.markdown('<p class="section-hdr">Risk Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4,3), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        ax.hist(probs, bins=25, color='#21d4fd', alpha=0.75, edgecolor='#0d1117')
        ax.axvline(0.7, color='#f85149', lw=1.8, linestyle='--', label='High (0.70)')
        ax.axvline(0.4, color='#e3b341', lw=1.8, linestyle='--', label='Medium (0.40)')
        ax.set_xlabel('Risk Score', color='#8b949e', fontsize=9)
        ax.set_ylabel('Patients', color='#8b949e', fontsize=9)
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#8b949e')
        for sp in ax.spines.values(): sp.set_color('#30363d')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with c_table:
        st.markdown('<p class="section-hdr">Patient Risk Scores</p>', unsafe_allow_html=True)
        show_cols = [c for c in ['patient_id','risk_pct','risk_label','Age',
                                  'Cd4AtStart','stage_start_num','had_interruption']
                     if c in df_input.columns]
        df_show = df_input[show_cols].copy()
        df_show['risk_label'] = df_show['risk_label'].map(
            {'HIGH':'🔴 HIGH','MEDIUM':'🟡 MEDIUM','LOW':'🟢 LOW'}
        ).fillna('🟢 LOW')
        df_show = df_show.sort_values('risk_pct', ascending=False)
        st.dataframe(df_show, height=260, use_container_width=True)

    # ── SHAP GLOBAL ───────────────────────────────────────
    st.markdown('<p class="section-hdr">Feature Importance (SHAP)</p>', unsafe_allow_html=True)

    with st.spinner("Computing SHAP explanations..."):
        mean_sv, shap_exp = compute_shap(X_raw)

    order  = np.argsort(mean_sv)
    f_names = [FEAT_LABELS.get(FEATURES[int(i)], FEATURES[int(i)]) for i in order]
    f_vals  = mean_sv[order]
    f_colors = ['#21d4fd' if v >= np.percentile(mean_sv, 60) else '#0072b2' for v in f_vals]

    fig, ax = plt.subplots(figsize=(7,4.5), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    bars = ax.barh(range(len(f_names)), f_vals, color=f_colors,
                   height=0.65, edgecolor='#0d1117', linewidth=0.3)
    for i, v in enumerate(f_vals):
        ax.text(v+0.0005, i, f'{v:.4f}', va='center', fontsize=7.5, color='#8b949e')
    ax.set_yticks(range(len(f_names)))
    ax.set_yticklabels(f_names, fontsize=9, color='#e6edf3')
    ax.set_xlabel('Mean |SHAP Value| — average impact on risk score',
                  color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.set_title('Global Feature Importance — All Patients',
                 color='#e6edf3', fontsize=10, pad=10)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── INDIVIDUAL PATIENT EXPLORER ───────────────────────
    st.markdown('<p class="section-hdr">Individual Patient Explorer</p>',
                unsafe_allow_html=True)

    pid_list  = df_input['patient_id'].tolist()
    best_pos  = int(df_input['risk_pct'].values.argmax())
    sel_id    = st.selectbox("Select patient:", pid_list, index=best_pos)

    sel_pos   = df_input[df_input['patient_id']==sel_id].index[0]
    pos       = df_input.index.get_loc(sel_pos)
    row       = df_input.loc[sel_pos]
    pt_lbl    = str(row['risk_label'])

    pt_sv, sv_ok = compute_shap_single(X_raw[pos])

    c_s, c_e = st.columns([1,2])
    with c_s:
        cc = 'risk-high' if pt_lbl=='HIGH' else 'risk-medium' if pt_lbl=='MEDIUM' else 'risk-low'
        st.markdown(f"""<div class="{cc}" style="margin-bottom:1rem">
            <div class="risk-number">{row['risk_pct']:.1f}%</div>
            <div class="risk-label">{pt_lbl} RISK</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Clinical Profile:**")
        for feat in FEATURES[:8]:
            val = row[feat]
            st.markdown(f"- {FEAT_LABELS.get(feat,feat)}: **{val:.1f}**")

        st.markdown("**Recommended Actions:**")
        for action in INTERVENTIONS.get(pt_lbl, INTERVENTIONS['LOW']):
            st.markdown(action)

    with c_e:
        sv_order = np.argsort(np.abs(pt_sv))[-10:]
        sv_vals  = pt_sv[sv_order]
        sv_names = [FEAT_LABELS.get(FEATURES[int(i)], FEATURES[int(i)]) for i in sv_order]
        sv_colors = ['#f85149' if v > 0 else '#3fb950' for v in sv_vals]

        fig, ax = plt.subplots(figsize=(6,4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        ax.barh(range(len(sv_names)), sv_vals, color=sv_colors,
                height=0.65, edgecolor='#0d1117', linewidth=0.3)
        ax.axvline(0, color='#8b949e', lw=0.8)
        ax.set_yticks(range(len(sv_names)))
        ax.set_yticklabels(sv_names, fontsize=8.5, color='#e6edf3')
        ax.set_xlabel('SHAP Value  (🔴 increases risk · 🟢 reduces risk)',
                      color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        lbl_type = "SHAP" if sv_ok else "Feature Importance (SHAP unavailable)"
        ax.set_title(f'Why is {sel_id} {pt_lbl} risk?  [{lbl_type}]',
                     color='#e6edf3', fontsize=9, pad=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── EXPORT ────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Export Results</p>', unsafe_allow_html=True)
    export = df_input[['patient_id','risk_pct','risk_label'] + FEATURES].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Download All Risk Scores (CSV)",
                           data=export.to_csv(index=False).encode(),
                           file_name="smartdaas_risk_scores.csv", mime="text/csv")
    with c2:
        high_df = export[export['risk_label']=='HIGH']
        st.download_button(f"🚨 Download High Risk Only ({n_high})",
                           data=high_df.to_csv(index=False).encode(),
                           file_name="smartdaas_high_risk.csv", mime="text/csv")


# ═════════════════════════════════════════════════════════════
#  PAGE 3 — FACILITY INTELLIGENCE
# ═════════════════════════════════════════════════════════════
elif page == "🏥  Facility Intelligence":

    st.markdown("""
    ### Facility-Level HIV Programme Intelligence
    Identifies structural drivers of poor outcomes across facility levels, ownership types,
    and funding models. Based on 27,288 patients from a Nigerian HIV programme (Paper 2).
    """)

    # Load facility data
    try:
        fac_df   = pd.read_csv('facility_summary.csv')
        perf_df  = pd.read_csv('facility_performance.csv')
        data_ok  = True
    except Exception:
        # Generate from built-in research data if CSVs not present
        fac_df = pd.DataFrame({
            'Health facility level': [
                'Primary health center','Primary health center','Primary health center',
                'Secondary health facility','Secondary health facility',
                'Secondary health facility','Secondary health facility',
                'Tertiary hospital','Tertiary hospital','Tertiary hospital','Tertiary hospital'
            ],
            'FacilityType': [
                'Faith Based','Private for profit','Public',
                'Faith Based','Private for profit','Private not for profit','Public',
                'Faith Based','Private for profit','Private not for profit','Public'
            ],
            'N': [12,64,445,2923,429,91,14539,238,3,1,8543],
            'poor_outcome': [0.0,0.266,0.142,0.122,0.126,0.066,0.123,0.181,0.333,0.0,0.100],
            'poor_adh':     [0.0,0.172,0.040,0.032,0.035,0.022,0.040,0.025,0.0,0.0,0.026],
            'mortality':    [0.0,0.0,0.011,0.008,0.002,0.0,0.010,0.004,0.0,0.0,0.007],
            'interrupted':  [0.0,0.156,0.099,0.094,0.103,0.044,0.094,0.168,0.333,0.0,0.079],
            'cd4_mean':     [459.9,328.9,377.8,391.5,287.3,354.5,442.7,283.0,171.0,391.0,506.8],
        })
        data_ok = True

    # ── SUMMARY METRICS ───────────────────────────────────
    st.markdown('<p class="section-hdr">Key Findings (Paper 2)</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-box"><div class="metric-val">1.95×</div><div class="metric-lbl">Primary HC vs Tertiary OR</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-box"><div class="metric-val">1.24×</div><div class="metric-lbl">NGO Funding OR</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-box"><div class="metric-val">2.2%</div><div class="metric-lbl">ICC (facility clustering)</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-box"><div class="metric-val">p&lt;0.001</div><div class="metric-lbl">LR Test: facility vars</div></div>', unsafe_allow_html=True)

    # ── OUTCOMES HEATMAP ──────────────────────────────────
    st.markdown('<p class="section-hdr">Composite Poor Outcome Rate by Facility Type</p>',
                unsafe_allow_html=True)

    levels = ['Primary health center','Secondary health facility','Tertiary hospital']
    types  = ['Public','Faith Based','Private for profit','Private not for profit']
    hmap   = np.full((len(types), len(levels)), np.nan)

    for _, row in fac_df.iterrows():
        lvl = row['Health facility level']
        ftp = row['FacilityType']
        if lvl in levels and ftp in types:
            li = levels.index(lvl)
            ti = types.index(ftp)
            hmap[ti, li] = row['poor_outcome'] * 100

    fig, ax = plt.subplots(figsize=(9,4), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    valid = hmap[~np.isnan(hmap)]
    vmax  = max(valid.max(), 20) if len(valid) else 20
    im    = ax.imshow(hmap, cmap='RdYlGn_r', vmin=0, vmax=vmax, aspect='auto')

    level_labels = ['Primary\nHC','Secondary\nHF','Tertiary\nHosp']
    ax.set_xticks(range(3)); ax.set_xticklabels(level_labels, fontsize=11, color='#e6edf3')
    ax.set_yticks(range(4)); ax.set_yticklabels(types, fontsize=11, color='#e6edf3')
    ax.set_title('Composite Poor Outcome Rate (%) by Facility Level and Ownership',
                 color='#e6edf3', fontsize=11, pad=10)

    for ti in range(4):
        for li in range(3):
            val = hmap[ti, li]
            if not np.isnan(val):
                ax.text(li, ti, f'{val:.1f}%', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if val > vmax*0.7 else 'black')
            else:
                ax.text(li, ti, 'n/a', ha='center', va='center',
                        fontsize=9, color='#555555')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Poor outcome rate (%)', color='#8b949e', fontsize=9)
    cbar.ax.tick_params(colors='#8b949e', labelsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── OUTCOME BARS BY FACILITY LEVEL ────────────────────
    st.markdown('<p class="section-hdr">Outcome Rates by Facility Level</p>',
                unsafe_allow_html=True)

    outcomes_map = {
        'poor_outcome':'Composite Poor Outcome',
        'poor_adh':'Poor Adherence',
        'mortality':'Mortality',
        'interrupted':'ART Interruption'
    }
    level_colors = {'Primary health center':'#CC79A7',
                    'Secondary health facility':'#56B4E9',
                    'Tertiary hospital':'#0072B2'}

    fig, axes = plt.subplots(1, 4, figsize=(14,4), facecolor='#161b22')
    fig.subplots_adjust(wspace=0.4)
    for ax_i, (col, label) in enumerate(outcomes_map.items()):
        ax = axes[ax_i]
        ax.set_facecolor('#161b22')
        lvl_rates = []
        lvl_labels_short = []
        for lvl in levels:
            sub = fac_df[fac_df['Health facility level']==lvl]
            if len(sub) > 0:
                rate = np.average(sub[col], weights=sub['N']) * 100
                lvl_rates.append(rate)
                lvl_labels_short.append(lvl.split()[0])
        colors_bar = ['#CC79A7','#56B4E9','#0072B2'][:len(lvl_rates)]
        bars = ax.bar(range(len(lvl_rates)), lvl_rates,
                      color=colors_bar, width=0.55, edgecolor='#0d1117')
        for bar, v in zip(bars, lvl_rates):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#e6edf3')
        ax.set_xticks(range(len(lvl_labels_short)))
        ax.set_xticklabels(lvl_labels_short, fontsize=8, color='#8b949e')
        ax.set_ylabel('Rate (%)', fontsize=8, color='#8b949e')
        ax.set_title(label, color='#e6edf3', fontsize=9, fontweight='bold', pad=6)
        ax.tick_params(colors='#8b949e', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        ax.set_ylim(0, max(lvl_rates)*1.45 if lvl_rates else 25)

    plt.suptitle('Weighted Outcome Rates by Facility Level (n=27,288)',
                 color='#e6edf3', fontsize=11, y=1.02)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── FACILITY PERFORMANCE TYPOLOGY ─────────────────────
    st.markdown('<p class="section-hdr">Facility Performance Typology</p>',
                unsafe_allow_html=True)
    st.markdown("""
    Risk-adjusted performance: facilities above the diagonal are **underperforming**
    (worse than their patient case-mix predicts). Facilities below are **positive deviants**
    (better than expected).
    """)

    try:
        type_colors = {
            'Positive Deviant\n(Better than expected)': '#3fb950',
            'Expected performer': '#21d4fd',
            'Underperformer\n(Worse than expected)': '#f85149',
        }
        fig, ax = plt.subplots(figsize=(8,6), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        lims = [0.05, 0.25]
        ax.plot(lims, lims, '--', color='#8b949e', lw=1.5, label='Perfect calibration')
        ax.fill_between(lims, [l*0.85 for l in lims], [l*1.15 for l in lims],
                        alpha=0.06, color='#21d4fd', label='±15% band')

        for _, row in perf_df.iterrows():
            ptype = str(row['performance_type']).strip()
            col   = type_colors.get(ptype, '#8b949e')
            n_val = row['n']
            size  = max(np.sqrt(n_val)*4, 40)
            ax.scatter(row['expected_rate'], row['observed_rate'],
                       s=size, color=col, alpha=0.85, edgecolor='#161b22', linewidth=0.8, zorder=5)
            if n_val > 50:
                lbl = (str(row['facility_group'])
                       .replace('Secondary health facility','Sec')
                       .replace('Tertiary hospital','Tert')
                       .replace('Primary health center','Prim')
                       .replace('Private for profit','Priv-Profit')
                       .replace('Faith Based','Faith')
                       .replace(' | ','\n'))
                ax.annotate(lbl, (row['expected_rate'], row['observed_rate']),
                            xytext=(6,4), textcoords='offset points',
                            fontsize=7.5, color='#8b949e')

        for ptype, col in type_colors.items():
            ax.scatter([],[],color=col,s=80,
                       label=ptype.replace('\n',' '), alpha=0.85)
        ax.set_xlabel('Expected Poor Outcome Rate (%) — risk-adjusted',
                      color='#8b949e', fontsize=10)
        ax.set_ylabel('Observed Poor Outcome Rate (%)', color='#8b949e', fontsize=10)
        ax.set_title('Facility Performance Typology\n(bubble size = patient volume)',
                     color='#e6edf3', fontsize=11, pad=10)
        ax.legend(fontsize=8, facecolor='#161b22', labelcolor='#8b949e', loc='upper left')
        ax.tick_params(colors='#8b949e', labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    except Exception:
        st.info("Facility performance typology chart requires facility_performance.csv in the repo.")

    # ── PEPFAR MER ALIGNMENT ──────────────────────────────
    st.markdown('<p class="section-hdr">PEPFAR MER Indicator Alignment</p>',
                unsafe_allow_html=True)
    mer_data = {
        'MER Indicator': ['TX_CURR','TX_PVLS','TX_RTT','TX_ML','TX_NEW'],
        'Description': [
            'Currently on ART','Viral load suppression',
            'Return to treatment','Interruptions in treatment',
            'New ART initiations'
        ],
        'SmartDaaS Signal': [
            'Base cohort for risk scoring',
            'Low CD4/high risk → proxy for VL failure risk',
            'High risk patients flagged for re-engagement',
            'ART interruption predictor (top SHAP feature)',
            'Delayed ART (>90 days) flagged in facility analysis'
        ],
        'Risk Level': ['All patients','HIGH','HIGH','HIGH/MEDIUM','Facility-level']
    }
    st.dataframe(pd.DataFrame(mer_data), use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 4 — ECONOMIC CALCULATOR
# ═════════════════════════════════════════════════════════════
elif page == "💰  Economic Calculator":

    st.markdown("""
    ### Programme Economic Impact Calculator
    Estimates excess poor outcomes and avoidable costs attributable to sub-optimal
    facility-level care. Based on Paper 2 findings. Inputs are adjustable.
    """)
    st.markdown("""<div class="info-box">
    📌 Based on Paper 2 findings: primary health centre patients had 1.95× adjusted odds
    of poor outcome vs tertiary hospitals (OR 1.95, 95% CI 1.45–2.61, p&lt;0.001).
    Cost estimate: USD 1,850 per poor outcome (Menzies et al., AIDS 2011).
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Programme Parameters</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        n_total = st.number_input(
            "Total ART patients in your programme",
            min_value=100, max_value=500000, value=27288, step=500
        )
        pct_primary = st.slider(
            "% at primary health centres",
            min_value=0, max_value=60, value=2
        ) / 100
        pct_secondary = st.slider(
            "% at secondary health facilities",
            min_value=0, max_value=90, value=66
        ) / 100

    with c2:
        cost_per_outcome = st.number_input(
            "Cost per poor outcome (USD)",
            min_value=500, max_value=10000, value=1850, step=50
        )
        tertiary_rate = st.slider(
            "Tertiary hospital poor outcome rate (%)",
            min_value=1, max_value=20, value=10
        ) / 100

    pct_tertiary  = 1 - pct_primary - pct_secondary
    if pct_tertiary < 0:
        st.error("Primary + Secondary % exceeds 100%. Adjust sliders.")
        st.stop()

    # Rates from Paper 2
    primary_rate   = 0.154
    secondary_rate = 0.123

    n_primary   = int(n_total * pct_primary)
    n_secondary = int(n_total * pct_secondary)
    n_tertiary  = int(n_total * pct_tertiary)

    exp_primary   = int(n_primary   * tertiary_rate)
    exp_secondary = int(n_secondary * tertiary_rate)
    obs_primary   = int(n_primary   * primary_rate)
    obs_secondary = int(n_secondary * secondary_rate)

    excess_primary   = max(0, obs_primary   - exp_primary)
    excess_secondary = max(0, obs_secondary - exp_secondary)
    total_excess     = excess_primary + excess_secondary
    total_cost       = total_excess * cost_per_outcome

    st.markdown('<p class="section-hdr">Estimated Impact</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="metric-val">{n_primary:,}</div><div class="metric-lbl">Patients at Primary HCs</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="metric-val">{total_excess:,}</div><div class="metric-lbl">Excess Poor Outcomes</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="metric-val">${total_cost/1000:.0f}K</div><div class="metric-lbl">Avoidable Cost (USD)</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="metric-val">${total_cost/n_total:.0f}</div><div class="metric-lbl">Cost per ART Patient</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Breakdown chart
    fig, axes = plt.subplots(1,2, figsize=(11,4.5), facecolor='#161b22')
    fig.subplots_adjust(wspace=0.4)

    # Patient distribution pie
    ax = axes[0]; ax.set_facecolor('#161b22')
    sizes  = [n_primary, n_secondary, n_tertiary]
    labels = [f'Primary\n{pct_primary*100:.0f}%',
              f'Secondary\n{pct_secondary*100:.0f}%',
              f'Tertiary\n{pct_tertiary*100:.0f}%']
    colors = ['#CC79A7','#56B4E9','#0072B2']
    patches, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct='%1.0f%%', startangle=90,
        textprops={'color':'#e6edf3','fontsize':9}
    )
    for at in autotexts: at.set_color('#0d1117'); at.set_fontweight('bold')
    ax.set_title('Patient Distribution\nby Facility Level',
                 color='#e6edf3', fontsize=10, pad=10)

    # Excess outcomes bar
    ax2 = axes[1]; ax2.set_facecolor('#161b22')
    cats  = ['Primary HC\n(observed)','Primary HC\n(if tertiary rate)',
             'Secondary HF\n(observed)','Secondary HF\n(if tertiary rate)']
    vals  = [obs_primary, exp_primary, obs_secondary, exp_secondary]
    bars_c = ['#f85149','#3fb950','#e3b341','#3fb950']
    bars_obj = ax2.bar(range(4), vals, color=bars_c, width=0.55, edgecolor='#0d1117')
    for bar, v in zip(bars_obj, vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{v:,}', ha='center', va='bottom', fontsize=9,
                 fontweight='bold', color='#e6edf3')
    ax2.set_xticks(range(4)); ax2.set_xticklabels(cats, fontsize=8, color='#8b949e')
    ax2.set_ylabel('Poor Outcomes', color='#8b949e', fontsize=9)
    ax2.set_title(f'Observed vs Expected Poor Outcomes\n(Excess = {total_excess:,} avoidable)',
                  color='#e6edf3', fontsize=10, pad=10)
    ax2.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax2.spines.values(): sp.set_color('#30363d')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown(f"""<div class="warn-box">
    ⚠️ <strong>Indicative estimates only.</strong>
    Based on Nigerian programme data. Primary HC rate = {primary_rate*100:.1f}%,
    Secondary HF rate = {secondary_rate*100:.1f}%, Tertiary reference rate = {tertiary_rate*100:.1f}%.
    Cost per poor outcome = USD {cost_per_outcome:,} (Menzies et al., 2011).
    These estimates should be validated with local programme data before use in funding proposals.
    </div>""", unsafe_allow_html=True)

    # Download summary
    summary_df = pd.DataFrame({
        'Parameter': ['Total ART patients','% at primary HCs','% at secondary HFs',
                      '% at tertiary','Primary HC poor outcome rate',
                      'Secondary HF poor outcome rate','Tertiary reference rate',
                      'Cost per poor outcome (USD)',
                      'Excess poor outcomes (primary)',
                      'Excess poor outcomes (secondary)',
                      'Total excess poor outcomes','Estimated avoidable cost (USD)'],
        'Value': [n_total, f'{pct_primary*100:.1f}%', f'{pct_secondary*100:.1f}%',
                  f'{pct_tertiary*100:.1f}%', '15.4%','12.3%',f'{tertiary_rate*100:.1f}%',
                  cost_per_outcome, excess_primary, excess_secondary,
                  total_excess, f'${total_cost:,.0f}']
    })
    st.download_button("📥 Download Economic Summary (CSV)",
                       data=summary_df.to_csv(index=False).encode(),
                       file_name="smartdaas_economic_summary.csv", mime="text/csv")


# ═════════════════════════════════════════════════════════════
#  PAGE 5 — MODEL INFO
# ═════════════════════════════════════════════════════════════
elif page == "📖  Model Info":

    st.markdown('<p class="section-hdr">Model Architecture</p>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Algorithm:** Random Forest Classifier
        **Trees:** 100 estimators | **Max depth:** 8
        **Training set:** 192,732 HIV patients on ART
        **Cross-validation:** 5-fold stratified
        **Leakage prevention:** Temporal hold-out validation

        | Metric | Value |
        |--------|-------|
        | AUC (5-fold CV) | 0.963 |
        | AUC (temporal) | 0.772 |
        | Sensitivity | 87.3% |
        | Specificity | 95.7% |
        | Brier Score | 0.079 |
        | Bootstrap 95% CI | 0.951–0.975 |
        """)
    with c2:
        st.markdown("**15 Clinical Predictor Variables:**")
        for feat, label in FEAT_LABELS.items():
            st.markdown(f"- `{feat}` — {label}")

    st.markdown('<p class="section-hdr">Methodological Notes</p>', unsafe_allow_html=True)
    st.markdown("""
    - **Class balance:** SMOTE applied (1:1 ratio) to handle adherence outcome imbalance
    - **Scaling:** StandardScaler within sklearn Pipeline (no leakage)
    - **Explainability:** SHAP TreeExplainer on extracted RF classifier
    - **Calibration:** Platt scaling applied post-hoc
    - **Temporal validation:** 80/20 chronological split (not random)
    """)

    st.markdown("---")
    st.warning("""
    **Research prototype (v0.2).** Before any clinical deployment:
    external validation on independent cohorts · prospective clinical evaluation ·
    regulatory approval (FDA / local health authority) · HIPAA/data security review ·
    clinical ethics approval.

    **Citation (Paper 1):**
    Chinthala LK. Real-world validation of ML models for HIV treatment adherence prediction
    and care gap quantification. Submitted to npj Digital Medicine. 2026.

    **Citation (Paper 2):**
    Chinthala LK. Facility-level structural drivers of HIV treatment outcomes.
    Submitted to BMJ Global Health. 2026.
    """)


# ═════════════════════════════════════════════════════════════
#  PAGE 6 — SAMPLE DATA
# ═════════════════════════════════════════════════════════════
elif page == "📋  Sample Data":

    st.markdown('<p class="section-hdr">CSV Template</p>', unsafe_allow_html=True)
    st.markdown("Download this template, fill in your patient data, and upload in Patient Risk.")

    template = pd.DataFrame({
        'patient_id':         ['PT-0001','PT-0002','PT-0003'],
        'Age':                [34,45,28],
        'sex_female':         [1,0,1],
        'Cd4AtStart':         [250,450,120],
        'MostRecentCd4Count': [380,520,95],
        'CD4_improvement':    [130,70,-25],
        'stage_start_num':    [2,1,3],
        'WeightAtStart':      [58.0,72.0,51.0],
        'weight_change':      [2.5,-1.0,-3.0],
        'BMI_start':          [21.3,24.8,19.6],
        'days_to_ART':        [45,120,7],
        'had_interruption':   [0,1,0],
        'opp_infection':      [0,0,1],
        'side_effects':       [1,0,0],
        'tb_positive':        [0,0,1],
        'stage_worsened':     [0,0,1],
    })

    st.dataframe(template, use_container_width=True)
    st.download_button("📥 Download Template CSV",
                       data=template.to_csv(index=False).encode(),
                       file_name="smartdaas_template.csv", mime="text/csv")

    st.markdown('<p class="section-hdr">Column Definitions</p>', unsafe_allow_html=True)
    for feat, label in FEAT_LABELS.items():
        st.markdown(f"**`{feat}`** — {label}")

    st.markdown('<p class="section-hdr">Encoding Guide</p>', unsafe_allow_html=True)
    st.markdown("""
    | Column | Values |
    |--------|--------|
    | sex_female | 1 = Female, 0 = Male |
    | stage_start_num | 1 = Stage I, 2 = II, 3 = III, 4 = IV |
    | had_interruption | 1 = Yes, 0 = No |
    | opp_infection | 1 = Yes, 0 = No |
    | side_effects | 1 = Yes, 0 = No |
    | tb_positive | 1 = Yes, 0 = No |
    | stage_worsened | 1 = Yes, 0 = No |
    """)

"""
app/demo.py — SmartDaaS Interactive Risk Prediction Demo
Streamlit web application for HIV treatment adherence risk scoring.

Usage:
    streamlit run app/demo.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="SmartDaaS — HIV Adherence Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLE ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1A5276, #2E86C1);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .risk-high   { background:#FADBD8; border-left:5px solid #E74C3C; padding:1rem; border-radius:5px; }
    .risk-medium { background:#FDEBD0; border-left:5px solid #E67E22; padding:1rem; border-radius:5px; }
    .risk-low    { background:#D5F5E3; border-left:5px solid #27AE60; padding:1rem; border-radius:5px; }
    .metric-box  { background:#EBF5FB; padding:1rem; border-radius:8px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:1.8rem;">🏥 SmartDaaS</h1>
    <p style="margin:0.3rem 0 0; opacity:0.9;">
        HIV Treatment Adherence Risk Predictor — Real-World ML Validation
    </p>
    <p style="margin:0.2rem 0 0; font-size:0.8rem; opacity:0.7;">
        Based on: "Real-World Validation of ML Models for HIV Treatment Adherence Prediction" 
        | 27,288 real patients | AUC 0.963
    </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR — PATIENT INPUT ───────────────────────────────
st.sidebar.header("📋 Patient Clinical Profile")
st.sidebar.markdown("Enter patient baseline characteristics at ART initiation:")

with st.sidebar:
    age = st.slider("Age (years)", 18, 75, 32)
    sex_female = st.selectbox("Sex", ["Female", "Male"]) == "Female"
    
    st.markdown("**CD4 Count**")
    cd4_start = st.number_input("CD4 at ART Start (cells/µL)", 5, 1500, 350)
    cd4_recent = st.number_input("Most Recent CD4 (cells/µL)", 5, 1500, 420)
    
    st.markdown("**Clinical Staging**")
    stage = st.selectbox("WHO Clinical Stage at Start", ["I", "II", "III", "IV"])
    stage_num = {"I":1,"II":2,"III":3,"IV":4}[stage]
    
    st.markdown("**Body Metrics**")
    weight_start = st.number_input("Weight at ART Start (kg)", 30.0, 150.0, 62.0)
    weight_recent = st.number_input("Most Recent Weight (kg)", 30.0, 150.0, 65.0)
    height = st.number_input("Height (cm)", 130.0, 210.0, 162.0)
    
    st.markdown("**Clinical History**")
    days_to_art = st.slider("Days from Diagnosis to ART Start", 0, 730, 45)
    opp_infection = st.checkbox("Opportunistic Infection Present")
    side_effects = st.checkbox("ART Side Effects Reported")
    tb_positive = st.checkbox("TB Co-infection")
    
    stage_last = st.selectbox("WHO Stage at Last Visit", ["I", "II", "III", "IV"])
    stage_worsened = {"I":1,"II":2,"III":3,"IV":4}[stage_last] > stage_num

# ── COMPUTE FEATURES ──────────────────────────────────────
bmi = weight_start / ((height/100)**2)
cd4_improvement = cd4_recent - cd4_start
weight_change = weight_recent - weight_start

features = np.array([[
    age, int(sex_female), cd4_start, cd4_recent, cd4_improvement,
    stage_num, weight_start, weight_change, bmi, days_to_art,
    int(opp_infection), int(side_effects), int(tb_positive), int(stage_worsened)
]])

feature_names = [
    'Age', 'Sex (Female)', 'CD4 at ART Start', 'Recent CD4',
    'CD4 Improvement', 'WHO Stage', 'Weight at Start', 'Weight Change',
    'BMI', 'Days to ART', 'Opp. Infection', 'Side Effects',
    'TB Positive', 'Stage Worsened'
]

# ── SIMULATE RISK SCORE (rule-based when no trained model available) ──
# In production: load trained model from results/cv_results.pkl
def compute_risk_score(f):
    """
    Rule-based risk approximation for demo purposes.
    In production, replace with: model.predict_proba(f)[0][1]
    """
    score = 0.05  # base rate ~5%
    
    # WHO stage contribution
    if f[0][5] == 3: score += 0.15
    elif f[0][5] == 4: score += 0.25
    
    # CD4 contribution
    if f[0][2] < 200: score += 0.12
    elif f[0][2] < 350: score += 0.06
    
    # CD4 trajectory
    if f[0][4] < 0: score += 0.10  # declining CD4
    elif f[0][4] > 100: score -= 0.05  # improving
    
    # Days to ART
    if f[0][9] > 180: score += 0.08
    elif f[0][9] > 90: score += 0.04
    
    # Comorbidities
    if f[0][10]: score += 0.08  # opp infection
    if f[0][12]: score += 0.06  # TB
    if f[0][11]: score += 0.04  # side effects
    if f[0][13]: score += 0.10  # stage worsened
    
    # Sex
    if not f[0][1]: score += 0.02  # male slightly higher
    
    # Age
    if f[0][0] < 25: score += 0.04
    
    return min(max(score, 0.01), 0.95)

risk_score = compute_risk_score(features)

# ── MAIN DISPLAY ──────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Poor Adherence Risk Score", f"{risk_score*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    risk_label = "HIGH" if risk_score > 0.25 else "MODERATE" if risk_score > 0.10 else "LOW"
    color = "#E74C3C" if risk_score > 0.25 else "#E67E22" if risk_score > 0.10 else "#27AE60"
    st.markdown(f'<div class="metric-box"><h3 style="color:{color}">{risk_label} RISK</h3></div>',
                unsafe_allow_html=True)

with col3:
    # CD4 trajectory indicator
    trajectory = "↑ Improving" if cd4_improvement > 20 else "↓ Declining" if cd4_improvement < -20 else "→ Stable"
    traj_color = "#27AE60" if "Improving" in trajectory else "#E74C3C" if "Declining" in trajectory else "#E67E22"
    st.markdown(f'<div class="metric-box"><b>CD4 Trajectory</b><br><span style="color:{traj_color};font-size:1.2rem">{trajectory}</span><br>Δ {cd4_improvement:+.0f} cells/µL</div>',
                unsafe_allow_html=True)

st.markdown("---")

# Risk interpretation
st.subheader("📊 Risk Interpretation & Clinical Action")
if risk_score > 0.25:
    st.markdown(f"""
    <div class="risk-high">
        <b>⚠️ HIGH RISK — Intensified monitoring recommended</b><br>
        Predicted poor adherence probability: <b>{risk_score*100:.1f}%</b><br><br>
        <b>Recommended actions:</b>
        <ul>
            <li>Schedule adherence counselling within 2 weeks</li>
            <li>Review ART regimen for tolerability</li>
            <li>Enable automated SMS/app reminders</li>
            <li>Consider differentiated service delivery model</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
elif risk_score > 0.10:
    st.markdown(f"""
    <div class="risk-medium">
        <b>⚡ MODERATE RISK — Enhanced monitoring</b><br>
        Predicted poor adherence probability: <b>{risk_score*100:.1f}%</b><br><br>
        <b>Recommended actions:</b>
        <ul>
            <li>Monthly adherence check-ins</li>
            <li>Peer support group referral</li>
            <li>Track CD4 trajectory at next visit</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="risk-low">
        <b>✅ LOW RISK — Standard monitoring</b><br>
        Predicted poor adherence probability: <b>{risk_score*100:.1f}%</b><br><br>
        Continue standard 3-monthly follow-up. Reassess if clinical status changes.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Feature contribution chart
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🔍 Key Risk Factors (This Patient)")
    
    # Approximate SHAP-like contributions
    contributions = {
        'WHO Stage at Start':    0.08 * stage_num,
        'CD4 Trajectory':        -0.005 * min(cd4_improvement, 0),
        'Days to ART':           min(days_to_art / 730 * 0.12, 0.12),
        'Opportunistic Infect.': 0.08 if opp_infection else 0,
        'Stage Worsened':        0.10 if stage_worsened else 0,
        'TB Co-infection':       0.06 if tb_positive else 0,
        'Side Effects':          0.04 if side_effects else 0,
        'CD4 at Start':          max((350-cd4_start)/350*0.10, 0),
    }
    
    df_contrib = pd.DataFrame(list(contributions.items()),
                               columns=['Factor', 'Contribution'])
    df_contrib = df_contrib[df_contrib['Contribution'] > 0.001].sort_values(
        'Contribution', ascending=True)
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors_bar = ['#E74C3C' if v > 0.05 else '#E67E22' if v > 0.02 else '#85929E'
                  for v in df_contrib['Contribution']]
    ax.barh(df_contrib['Factor'], df_contrib['Contribution'],
            color=colors_bar, edgecolor='white', height=0.6)
    ax.set_xlabel('Risk Contribution (approximate)')
    ax.set_title('Risk Factor Contributions', fontweight='bold', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_b:
    st.subheader("📈 Patient vs. Population")
    
    # Population reference values
    pop_stats = {
        'CD4 at Start': (452, cd4_start),
        'Days to ART':  (74,  days_to_art),
        'Age':          (35,  age),
        'BMI':          (22,  bmi),
    }
    
    fig2, axes = plt.subplots(2, 2, figsize=(5, 4))
    for ax_i, (label, (pop_val, pat_val)) in zip(axes.flatten(), pop_stats.items()):
        bars = ax_i.bar(['Population\nMedian', 'This\nPatient'],
                        [pop_val, pat_val],
                        color=['#92C5DE', '#0072B2'], width=0.5, edgecolor='white')
        ax_i.set_title(label, fontsize=8, fontweight='bold')
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)
        ax_i.tick_params(labelsize=7)
        for bar in bars:
            ax_i.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'{bar.get_height():.0f}', ha='center', fontsize=7)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# Model info
with st.expander("ℹ️ About This Model"):
    st.markdown(f"""
    **Model:** Random Forest Classifier (14 clinical features)
    
    **Validation Performance:**
    | Metric | Value |
    |--------|-------|
    | AUC-ROC (10-fold CV) | 0.9627 ± 0.0019 |
    | AUC-ROC (temporal validation) | 0.772 (95% CI: 0.744–0.802) |
    | Sensitivity | 87.3% |
    | Specificity | 95.7% |
    | Brier Score | 0.079 |
    
    **Dataset:** 27,288 real HIV patients on ART
    
    **Key design decision:** ART interruption history is intentionally excluded 
    from the primary model to maximise prospective deployability and avoid 
    temporal confounding.
    
    **Note:** This demo uses a rule-based approximation for illustration. 
    Production deployment requires the trained model from `results/cv_results.pkl`.
    
    **Code:** https://github.com/Kchinthala15/smartdaas-hiv-validation
    
    **Paper:** Chinthala, L.K. (2026). Real-World Validation of ML Models for 
    HIV Treatment Adherence Prediction. Submitted to npj Digital Medicine.
    
    ⚠️ *For research purposes only. Not for clinical use without prospective validation.*
    """)

st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; margin-top:2rem;">
    SmartDaaS Research Framework | Lakshmi Kalyani Chinthala | 
    <a href="https://github.com/Kchinthala15/smartdaas-hiv-validation">GitHub</a> | 
    ORCID: 0009-0009-8736-6673
</div>
""", unsafe_allow_html=True)

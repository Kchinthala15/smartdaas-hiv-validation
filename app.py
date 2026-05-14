import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import base64
import os

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartDaaS HIV Risk Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# GLOBAL CSS FOR POLISHED UI
# ---------------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #0E1117;
    color: #FAFAFA;
}

.section-header {
    font-size: 24px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 10px;
    color: #E0E0E0;
}

.metric-box {
    padding: 16px;
    border-radius: 10px;
    background-color: #111418;
    border: 1px solid #2A2D33;
    text-align: center;
    margin-bottom: 10px;
}

.metric-box h3 {
    margin: 0;
    font-size: 20px;
    color: #E8E8E8;
}

.metric-box p {
    margin: 0;
    font-size: 18px;
    color: #A0A0A0;
}

.nav-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.markdown("<p class='nav-title'>SmartDaaS</p>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict Risk", "Model Info", "Sample Data"]
)

# ---------------------------------------------------------
# UTILITY: LOAD MODEL + DATA
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_resource
def load_feature_matrix():
    return joblib.load("X.pkl")

model = load_model()
X = load_feature_matrix()
feature_names = X.columns.tolist()
# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "Home":
    st.title("SmartDaaS — HIV Risk Prediction Platform")

    st.markdown("<p class='section-header'>About SmartDaaS</p>", unsafe_allow_html=True)
    st.write("""
    SmartDaaS is an AI-driven HIV risk prediction platform designed to support Ministries of Health,
    implementing partners, and clinical programs in identifying individuals at elevated risk.
    It provides explainable AI, facility-level insights, and actionable intelligence for HIV programs.
    """)

    st.markdown("<p class='section-header'>Contact</p>", unsafe_allow_html=True)
    st.write("📧 **chinthalakalyani1@gmail.com**")

    st.markdown("<p class='section-header'>Model Performance</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-box'><h3>AUROC</h3><p>0.82</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-box'><h3>Accuracy</h3><p>0.76</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box'><h3>Recall</h3><p>0.71</p></div>", unsafe_allow_html=True)

    st.markdown("<p class='section-header'>Platform Features</p>", unsafe_allow_html=True)
    st.write("""
    - Predict individual HIV acquisition risk  
    - Explain predictions using SHAP  
    - Provide facility-level insights  
    - Support programmatic decision-making  
    - Designed for Ministries, NGOs, and implementing partners  
    """)

    st.markdown("<p class='section-header'>Why SmartDaaS?</p>", unsafe_allow_html=True)
    st.write("""
    SmartDaaS is built to be lightweight, scalable, and deployable in low-resource settings.
    It integrates seamlessly with existing HIV program workflows and supports targeted interventions.
    """)
# ---------------------------------------------------------
# PREDICT RISK PAGE
# ---------------------------------------------------------
if page == "Predict Risk":
    st.title("Predict HIV Risk")

    st.markdown("<p class='section-header'>Input Features</p>", unsafe_allow_html=True)

    # Collect user inputs
    user_input = {}
    for col in feature_names:
        user_input[col] = st.number_input(col, value=0.0)

    # ---------------------------------------------------------
    # RUN PREDICTION
    # ---------------------------------------------------------
    if st.button("Predict"):
        df_input = pd.DataFrame([user_input])
        risk_pct = model.predict_proba(df_input)[0][1] * 100

        st.markdown("<p class='section-header'>Prediction Result</p>", unsafe_allow_html=True)
        st.write(f"**Predicted HIV Risk: {risk_pct:.2f}%**")

        # ---------------------------------------------------------
        # SAFE STREAMLIT-NATIVE TABLE (NO ERRORS)
        # ---------------------------------------------------------
        df_display = df_input.copy()
        df_display["risk_pct"] = risk_pct

        # Create a risk label column
        def label_risk(val):
            if val >= 20:
                return "High"
            elif val >= 10:
                return "Medium"
            else:
                return "Low"

        df_display["risk_label"] = df_display["risk_pct"].apply(label_risk)

        # Streamlit-native safe coloring (NO pandas.style)
        st.dataframe(
            df_display,
            use_container_width=True,
            column_config={
                "risk_label": st.column_config.TextColumn(
                    "Risk Category",
                    help="Low / Medium / High",
                )
            }
        )

        # ---------------------------------------------------------
        # SHAP EXPLANATIONS (BAR CHART)
        # ---------------------------------------------------------
        st.markdown("<p class='section-header'>Feature Importance (SHAP)</p>", unsafe_allow_html=True)

        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(shap_values[0])
        }).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 10))
        ax.barh(shap_df["feature"], shap_df["importance"], color="#4c72b0")
        ax.set_xlabel("SHAP Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
# ---------------------------------------------------------
# MODEL INFO PAGE
# ---------------------------------------------------------
if page == "Model Info":
    st.title("Model Information")

    st.markdown("<p class='section-header'>Model Overview</p>", unsafe_allow_html=True)
    st.write("""
    The SmartDaaS HIV Risk Model is built using gradient boosting techniques optimized for 
    clinical prediction tasks. It leverages demographic, behavioral, and clinical features 
    to estimate the probability of HIV acquisition.
    """)

    st.markdown("<p class='section-header'>Training Details</p>", unsafe_allow_html=True)
    st.write("""
    - Algorithm: Gradient Boosting  
    - Explainability: SHAP (SHapley Additive exPlanations)  
    - Evaluation Metrics: AUROC, Accuracy, Recall  
    - Deployment: Streamlit-based lightweight inference  
    """)

    st.markdown("<p class='section-header'>Intended Use</p>", unsafe_allow_html=True)
    st.write("""
    This model is intended to support Ministries of Health, NGOs, and implementing partners 
    in identifying individuals who may benefit from targeted HIV prevention interventions.
    """)

    st.markdown("<p class='section-header'>Limitations</p>", unsafe_allow_html=True)
    st.write("""
    - Not a diagnostic tool  
    - Should be used alongside clinical judgment  
    - Requires appropriate contextualization for each setting  
    """)


# ---------------------------------------------------------
# SAMPLE DATA PAGE
# ---------------------------------------------------------
if page == "Sample Data":
    st.title("Sample Data")

    st.markdown("<p class='section-header'>Example Input Data</p>", unsafe_allow_html=True)
    st.write("""
    Below is a sample dataset illustrating the structure and format of inputs used by the model.
    """)

    if os.path.exists("sample_data.csv"):
        sample_df = pd.read_csv("sample_data.csv")
        st.dataframe(sample_df, use_container_width=True)
    else:
        st.warning("sample_data.csv not found. Please upload it to the app directory.")

    st.markdown("<p class='section-header'>Data Dictionary</p>", unsafe_allow_html=True)
    st.write("""
    - Each column corresponds to a model feature  
    - Values represent demographic or behavioral indicators  
    - Data should be numeric and preprocessed before inference  
    """)


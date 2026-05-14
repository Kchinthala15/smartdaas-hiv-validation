import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartDaaS HIV Risk", layout="wide")

# Custom CSS
st.markdown("""
<style>
.section-header {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
}
.metric-box {
    padding: 12px;
    border-radius: 8px;
    background-color: #111;
    border: 1px solid #333;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["Home", "Predict Risk", "Model Info", "Sample Data"])
if page == "Home":
    st.title("SmartDaaS — HIV Risk Prediction Platform")

    st.markdown("<p class='section-header'>About SmartDaaS</p>", unsafe_allow_html=True)
    st.write("""
    SmartDaaS is an AI-driven HIV risk prediction platform designed to support Ministries of Health,
    implementing partners, and clinical programs in identifying individuals at elevated risk.
    """)

    st.markdown("<p class='section-header'>Contact</p>", unsafe_allow_html=True)
    st.write("For pilots or collaboration:")
    st.write("📧 **chinthalakalyani1@gmail.com**")

    st.markdown("<p class='section-header'>Model Performance</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-box'><h3>AUROC</h3><p>0.82</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-box'><h3>Accuracy</h3><p>0.76</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box'><h3>Recall</h3><p>0.71</p></div>", unsafe_allow_html=True)
if page == "Predict Risk":
    st.title("Predict HIV Risk")

    # Load model + data
    model = joblib.load("model.pkl")
    X = joblib.load("X.pkl")
    feature_names = X.columns.tolist()

    st.markdown("<p class='section-header'>Input Features</p>", unsafe_allow_html=True)

    # User inputs
    user_input = {}
    for col in feature_names:
        user_input[col] = st.number_input(col, value=0.0)

    if st.button("Predict"):
        df_input = pd.DataFrame([user_input])
        risk_pct = model.predict_proba(df_input)[0][1] * 100

        st.markdown("<p class='section-header'>Prediction Result</p>", unsafe_allow_html=True)
        st.write(f"**Predicted HIV Risk: {risk_pct:.2f}%**")

        # Display table (plain, safe)
        df_display = df_input.copy()
        df_display["risk_pct"] = risk_pct
        st.dataframe(df_display, use_container_width=True)

        # SHAP EXPLANATIONS
        st.markdown("<p class='section-header'>Feature Importance (SHAP)</p>", unsafe_allow_html=True)

        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)

        # Bar chart SHAP
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(shap_values[0])
        }).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.barh(shap_df["feature"], shap_df["importance"], color="#4c72b0")
        ax.set_xlabel("SHAP Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
if page == "Model Info":
    st.title("Model Information")
    st.write("""
    This model is trained on clinical and demographic features to estimate HIV acquisition risk.
    It uses gradient boosting with SHAP explainability.
    """)

if page == "Sample Data":
    st.title("Sample Data")
    sample_df = pd.read_csv("sample_data.csv")
    st.dataframe(sample_df, use_container_width=True)

# SmartDaaS MVP v0.1
### HIV Treatment Adherence Risk Prediction Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartdaas.streamlit.app)

**Author:** Lakshmi Kalyani Chinthala | Golden Gate University  
**ORCID:** 0009-0009-8736-6673

---

## What it does

Upload a CSV of HIV patient records → get individual risk scores → see SHAP explanations for each patient.

**Risk tiers:**
- 🔴 **HIGH** (score ≥ 70%) — priority for adherence intervention
- 🟡 **MEDIUM** (score 40–69%) — enhanced monitoring
- 🟢 **LOW** (score < 40%) — standard care pathway

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploying to Streamlit Cloud (free)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `app.py`
5. Click Deploy

**Note:** The model file (`cv_results.pkl`) must be in the same directory as `app.py`.  
Download from: https://github.com/Kchinthala15/smartdaas-hiv-validation

---

## Model

Random Forest trained on 192,732 HIV patient records.

| Metric | Value |
|--------|-------|
| AUC (cross-validation) | 0.963 |
| AUC (temporal validation) | 0.772 |
| Sensitivity | 87.3% |
| Specificity | 95.7% |

**Paper:** Chinthala LK. "Real-World Validation of Machine Learning Models for HIV Treatment  
Adherence Prediction." Submitted to *npj Digital Medicine*, 2026.

---

## ⚠️ Disclaimer

Research prototype only. Not validated for clinical decision-making.  
Do not use for individual patient treatment decisions without clinical review.

---

## License

MIT — see LICENSE

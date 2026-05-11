# Real-World Validation of ML Models for HIV Treatment Adherence Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![TRIPOD](https://img.shields.io/badge/Reporting-TRIPOD-orange)](https://www.tripod-statement.org/)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)](app/demo.py)

> **"Real-World Validation of Machine Learning Models for HIV Treatment Adherence Prediction and Care Gap Quantification: A Multi-Country Analysis of 192,732 Clinical Records"**
> Lakshmi Kalyani Chinthala | ORCID: [0009-0009-8736-6673](https://orcid.org/0009-0009-8736-6673)

## Key Results

| Metric | Value |
|--------|-------|
| AUC-ROC (10-fold CV, primary model) | 0.9627 ± 0.0019 |
| AUC-ROC (temporal validation) | 0.772 (95% CI: 0.744–0.802) |
| Sensitivity | 87.3% | Specificity | 95.7% | Brier Score | 0.079 |
| Median diagnosis-to-ART delay | 74 days | Patients delayed >90 days | 47.3% |
| Economic savings (base case) | USD 415/patient |

**Primary model design:** ART interruption excluded (14 features) to maximise prospective deployability. Full 15-feature model provided as secondary analysis.

## Datasets (free, public, no registration)

1. [Quality of Care HIV Dataset](https://www.kaggle.com/datasets/iogbonna/quality-of-care-dataset-for-hiv-clients) — 27,288 HIV+ patients on ART
2. CEPHIA HIV Recency Assay Public Use Dataset (v20210604) — 165,444 HIV+ specimens, 6 countries

Download both to `data/`.

## Repository Structure

```
smartdaas-hiv-validation/
├── src/
│   ├── utils.py                      # Constants, colors, parameters
│   ├── 01_data_preprocessing.py      # Feature engineering + SMOTE
│   ├── 02_model_training_cv.py       # 10-fold CV, all 4 classifiers
│   ├── 03_temporal_validation.py     # Temporal train/test split
│   ├── 04_shap_explainability.py     # SHAP KernelExplainer
│   ├── 05_decision_curve_analysis.py # DCA net benefit
│   ├── 06_subgroup_fairness.py       # Subgroup AUC by demographics
│   ├── 07_economic_modelling.py      # Sensitivity analysis
│   └── 08_figures.py                 # All publication figures (300 DPI)
├── app/
│   └── demo.py                       # Streamlit interactive risk predictor
├── data/                             # Place datasets here (gitignored)
├── figures/                          # Output figures (gitignored)
├── results/                          # Output pickles/CSVs (gitignored)
├── requirements.txt
└── setup.sh
```

## Quick Start

```bash
git clone https://github.com/Kchinthala15/smartdaas-hiv-validation.git
cd smartdaas-hiv-validation
bash setup.sh && source venv/bin/activate

# Run full pipeline
python src/01_data_preprocessing.py
python src/02_model_training_cv.py
python src/03_temporal_validation.py
python src/04_shap_explainability.py
python src/05_decision_curve_analysis.py
python src/06_subgroup_fairness.py
python src/07_economic_modelling.py
python src/08_figures.py

# Launch interactive demo
streamlit run app/demo.py
```

## Reproducibility

- All transforms fitted **only on training folds** (via sklearn Pipeline)
- SMOTE applied **exclusively within training folds**
- Hold-out test set untouched during all training/preprocessing
- Fixed seed: `SEED = 42` in `src/utils.py`
- TRIPOD reporting guidelines followed

## Citation

```bibtex
@article{chinthala2026hiv,
  title  = {Real-World Validation of ML Models for HIV Treatment Adherence Prediction},
  author = {Chinthala, Lakshmi Kalyani},
  journal= {Submitted to npj Digital Medicine},
  year   = {2026},
  url    = {https://github.com/Kchinthala15/smartdaas-hiv-validation}
}
```

## Contact
kchinthala@my.ggu.edu | [ORCID](https://orcid.org/0009-0009-8736-6673)

> ⚠️ Research purposes only. Not validated for clinical use.

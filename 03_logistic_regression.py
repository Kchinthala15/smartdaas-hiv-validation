"""
paper2/src/03_logistic_regression.py
Main multivariable logistic regression with HC3 cluster-robust
standard errors. Reference: tertiary hospital, public ownership,
no NGO/federal/state funding.

Methodological note:
    GEE and mixed-effects logistic regression were considered but
    require ≥20-30 clusters for reliable inference. With 11
    facility-level clusters, HC3 heteroscedasticity-robust SEs
    are the appropriate alternative (Hanley et al., 2003).
"""

import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'facility_primary', 'facility_secondary',
    'type_faith', 'type_private_profit',
    'has_NGO', 'has_federal', 'mixed_funding',
    'sex_female', 'stage_num', 'cd4_std', 'Age'
]

VAR_LABELS = {
    'facility_primary':    'Primary HC vs Tertiary',
    'facility_secondary':  'Secondary HF vs Tertiary',
    'type_faith':          'Faith-Based vs Public',
    'type_private_profit': 'Private Profit vs Public',
    'has_NGO':             'NGO Funding',
    'has_federal':         'Federal Funding',
    'mixed_funding':       'Mixed Funding',
    'sex_female':          'Sex (Female)',
    'stage_num':           'WHO Clinical Stage',
    'cd4_std':             'CD4 at Start (std)',
    'Age':                 'Age (per year)',
}


def main():
    print("=" * 60)
    print("Paper 2 — Step 3: Logistic Regression (HC3 robust SEs)")
    print("=" * 60)

    with open('paper2/results/data_engineered.pkl', 'rb') as f:
        d = pickle.load(f)
    df = d['df']

    # Complete-case dataset
    model_df = df[FEATURES + ['poor_outcome', 'facility_group']].dropna()
    print(f"\nAnalytic sample (complete cases): n={len(model_df):,}")
    print(f"Facility clusters: {model_df['facility_group'].nunique()}")
    print(f"Poor outcome rate: {model_df['poor_outcome'].mean()*100:.1f}%")

    X = sm.add_constant(model_df[FEATURES])
    y = model_df['poor_outcome']

    # Fit with HC3 robust standard errors
    logit = sm.Logit(y, X).fit(disp=0, cov_type='HC3')

    # Extract odds ratios
    params = logit.params
    conf   = logit.conf_int()
    pvals  = logit.pvalues

    OR_df = pd.DataFrame({
        'Variable': [VAR_LABELS.get(v, v) for v in params.index if v != 'const'],
        'OR':       np.exp(params.drop('const')).values,
        'CI_low':   np.exp(conf[0].drop('const')).values,
        'CI_high':  np.exp(conf[1].drop('const')).values,
        'p':        pvals.drop('const').values,
    }).round(4)

    OR_df['sig'] = OR_df['p'].apply(
        lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    )

    print("\nOdds Ratios (HC3 robust SEs):")
    print("-" * 65)
    for _, row in OR_df.iterrows():
        print(f"  {row['Variable']:30s}: OR={row['OR']:.3f} "
              f"[{row['CI_low']:.3f}-{row['CI_high']:.3f}]  "
              f"p={row['p']:.4f} {row['sig']}")

    print(f"\nModel fit:")
    print(f"  AIC: {logit.aic:.1f}")
    print(f"  BIC: {logit.bic:.1f}")
    print(f"  Log-likelihood: {logit.llf:.1f}")
    print(f"  Pseudo R²: {logit.prsquared:.4f}")

    # Save
    OR_df.to_csv('paper2/results/odds_ratios.csv', index=False)
    with open('paper2/results/regression_results.pkl', 'wb') as f:
        pickle.dump({
            'model': logit,
            'OR_df': OR_df,
            'model_df': model_df,
            'X': X, 'y': y,
        }, f)

    print("\nSaved: paper2/results/odds_ratios.csv")
    print("Saved: paper2/results/regression_results.pkl")
    print("Step 3 complete.")


if __name__ == '__main__':
    main()

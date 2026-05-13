"""
paper2/src/02_descriptive_analysis.py
Table 1: Unadjusted outcomes by facility level, ownership, and funding.
Chi-squared tests for all associations.
"""

import pandas as pd
import numpy as np
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("Paper 2 — Step 2: Descriptive Analysis")
    print("=" * 60)

    with open('paper2/results/data_engineered.pkl', 'rb') as f:
        d = pickle.load(f)
    df = d['df']

    outcomes = [
        ('poor_adherence',  'Poor Adherence'),
        ('dead',            'Mortality'),
        ('art_interrupted', 'ART Interruption'),
        ('delayed_ART_90',  'Delayed ART >90d'),
        ('poor_outcome',    'Composite Poor Outcome'),
    ]

    levels = ['Primary health center', 'Secondary health facility', 'Tertiary hospital']

    # ── TABLE 1: Outcomes by facility level ──────────────
    print("\nTable 1: Outcomes by Facility Level")
    print("-" * 70)
    rows = []
    for col, label in outcomes:
        row = {'Outcome': label}
        for level in levels:
            sub = df[df['Health facility level'] == level]
            row[level.split()[0]] = f"{sub[col].mean()*100:.1f}%"
        # Chi-squared test
        ct = pd.crosstab(df['Health facility level'], df[col])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        row['p-value'] = f"{p:.4f}" if p >= 0.001 else "<0.001"
        rows.append(row)
        print(f"  {label:25s}: Primary={row['Primary']:6s} Secondary={row['Secondary']:6s} "
              f"Tertiary={row['Tertiary']:6s} p={row['p-value']}")

    # ── OUTCOMES BY FACILITY TYPE ─────────────────────────
    print("\nOutcomes by Facility Type (composite poor outcome):")
    for ftype in df['FacilityType'].unique():
        sub = df[df['FacilityType'] == ftype]
        print(f"  {ftype:25s}: {sub['poor_outcome'].mean()*100:.1f}% (n={len(sub):,})")

    # ── OUTCOMES BY FUNDING MODEL ─────────────────────────
    print("\nOutcomes by Funding Model (composite poor outcome):")
    funding_groups = {
        'NGO Only':    (df['has_NGO']==1)&(df['has_federal']==0)&(df['has_state']==0),
        'State Only':  (df['has_state']==1)&(df['has_NGO']==0)&(df['has_federal']==0),
        'Federal Only':(df['has_federal']==1)&(df['has_NGO']==0)&(df['has_state']==0),
        'Mixed':        df['mixed_funding']==1,
    }
    for fname, fmask in funding_groups.items():
        sub = df[fmask]
        print(f"  {fname:15s}: {sub['poor_outcome'].mean()*100:.1f}% (n={len(sub):,})")

    # ── DATASET SUMMARY ───────────────────────────────────
    print(f"\nDataset summary:")
    print(f"  Total patients:    {len(df):,}")
    print(f"  Mean age:          {df['Age'].mean():.1f} years")
    print(f"  Female:            {df['sex_female'].mean()*100:.1f}%")
    print(f"  CD4 <200 at start: {(df['Cd4AtStart']<200).mean()*100:.1f}%")
    print(f"  Median ART delay:  {df['days_to_ART'].median():.0f} days")
    print(f"  Delayed >90d:      {df['delayed_ART_90'].mean()*100:.1f}%")

    pd.DataFrame(rows).to_csv('paper2/results/table1.csv', index=False)
    print("\nSaved: paper2/results/table1.csv")
    print("Step 2 complete.")


if __name__ == '__main__':
    main()

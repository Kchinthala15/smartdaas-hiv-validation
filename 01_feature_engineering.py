"""
paper2/src/01_feature_engineering.py
Feature engineering and outcome variable construction
for the facility-level health systems analysis (Paper 2).

Usage:
    python paper2/src/01_feature_engineering.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ── OUTCOME DEFINITIONS ───────────────────────────────────
# Poor adherence: ArvAdherenceLatestLevel == 'Poor'
# Mortality:      PatientDead == 'Yes'
# ART interrupted: ArtInterruption == 'Yes'
# Composite poor outcome: any of the above
# Delayed ART: days from diagnosis to ART start > 90

# ── FACILITY VARIABLES ────────────────────────────────────
# Health facility level: Primary health center / Secondary health facility / Tertiary hospital
# FacilityType: Public / Faith Based / Private for profit / Private not for profit
# FundingSources: combinations of NGO / State Government / Federal Government


def load_data(path: str = 'data/QualityOfCare.xlsx') -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def engineer_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Construct all outcome and predictor variables."""

    # Date parsing
    for col in ['DateOfConfirmedHIV', 'DateArtStarted']:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # Diagnosis-to-ART delay
    df['days_to_ART'] = (
        (df['DateArtStarted'] - df['DateOfConfirmedHIV']).dt.days
    ).clip(0, 3650)
    df['delayed_ART_90'] = (df['days_to_ART'] > 90).astype(int)

    # Binary outcome variables
    df['poor_adherence'] = (
        df['ArvAdherenceLatestLevel'].str.lower().str.strip() == 'poor'
    ).astype(int)

    df['dead'] = (
        df['PatientDead'].str.lower().str.strip() == 'yes'
    ).astype(int)

    df['art_interrupted'] = (
        df['ArtInterruption'].str.lower().str.strip() == 'yes'
    ).astype(int)

    # PRIMARY OUTCOME: composite poor outcome
    df['poor_outcome'] = (
        (df['poor_adherence'] == 1) |
        (df['dead'] == 1) |
        (df['art_interrupted'] == 1)
    ).astype(int)

    # ── FACILITY FEATURES ─────────────────────────────────
    df['has_NGO']     = df['FundingSources'].str.contains('Non-Governmental', na=False).astype(int)
    df['has_federal'] = df['FundingSources'].str.contains('Federal', na=False).astype(int)
    df['has_state']   = df['FundingSources'].str.contains('State', na=False).astype(int)
    df['mixed_funding'] = (
        (df['has_NGO'] + df['has_federal'] + df['has_state']) > 1
    ).astype(int)

    df['facility_primary']   = (df['Health facility level'] == 'Primary health center').astype(int)
    df['facility_secondary'] = (df['Health facility level'] == 'Secondary health facility').astype(int)
    df['type_faith']         = (df['FacilityType'] == 'Faith Based').astype(int)
    df['type_private_profit']= (df['FacilityType'] == 'Private for profit').astype(int)

    # ── PATIENT COVARIATES ────────────────────────────────
    df['sex_female'] = (df['Sex'].str.lower().str.strip() == 'female').astype(int)
    df['stage_num']  = df['ClinicalStageAtStart'].map({'I':1,'II':2,'III':3,'IV':4})
    df['cd4_std']    = (df['Cd4AtStart'] - df['Cd4AtStart'].mean()) / df['Cd4AtStart'].std()

    # Facility group identifier for clustering
    df['facility_group'] = (
        df['Health facility level'].astype(str) + ' | ' +
        df['FacilityType'].astype(str)
    )

    print(f"\nOutcome rates:")
    print(f"  Poor adherence:    {df['poor_adherence'].mean()*100:.1f}%")
    print(f"  Mortality:         {df['dead'].mean()*100:.1f}%")
    print(f"  ART interrupted:   {df['art_interrupted'].mean()*100:.1f}%")
    print(f"  Composite poor:    {df['poor_outcome'].mean()*100:.1f}%")
    print(f"  Delayed ART >90d:  {df['delayed_ART_90'].mean()*100:.1f}%")

    print(f"\nFacility groups: {df['facility_group'].nunique()}")
    print(df['Health facility level'].value_counts().to_string())

    return df


def main():
    print("=" * 60)
    print("Paper 2 — Step 1: Feature Engineering")
    print("=" * 60)

    df = load_data()
    df = engineer_outcomes(df)

    # Save
    with open('paper2/results/data_engineered.pkl', 'wb') as f:
        pickle.dump({'df': df}, f)

    print("\nSaved: paper2/results/data_engineered.pkl")
    print("Step 1 complete.")


if __name__ == '__main__':
    main()

"""
01_data_preprocessing.py
Feature engineering, imputation, and SMOTE class-balancing
for the Quality of Care HIV dataset.

Usage:
    python src/01_data_preprocessing.py \
        --qoc_path data/QualityOfCare.xlsx \
        --cephia_path data/cephia_public_use_dataset_20210604.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from utils import FEATURES, TARGET, SEED

np.random.seed(SEED)


def load_qoc(path: str) -> pd.DataFrame:
    """Load and clean the Quality of Care HIV dataset."""
    df = pd.read_excel(path)
    print(f"Loaded QoC dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def load_cephia(path: str) -> pd.DataFrame:
    """Load and filter the CEPHIA recency assay dataset."""
    df = pd.read_csv(path, low_memory=False)
    df = df[df['hiv_status_at_visit'] == 'P'].copy()
    df = df.dropna(subset=['days_since_eddi'])
    df['recently_infected'] = (df['days_since_eddi'] <= 130).astype(int)
    print(f"Loaded CEPHIA dataset: {len(df):,} HIV+ records")
    print(f"  Recently infected (<=130 days): {df['recently_infected'].sum():,} "
          f"({df['recently_infected'].mean()*100:.1f}%)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all model features from raw QoC variables."""
    # Date parsing
    for col in ['DateOfConfirmedHIV', 'DateArtStarted']:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # Derived features
    df['days_to_ART'] = (
        (df['DateArtStarted'] - df['DateOfConfirmedHIV']).dt.days
    ).clip(0, 3650)

    df['BMI_start'] = (
        df['WeightAtStart'] / ((df['HeightAtStart'] / 100) ** 2)
    ).clip(10, 60)

    df['CD4_improvement'] = df['MostRecentCd4Count'] - df['Cd4AtStart']
    df['weight_change']   = df['WeightAtLastVisit'] - df['WeightAtStart']

    stage_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    df['stage_start_num'] = df['ClinicalStageAtStart'].map(stage_map)
    df['stage_last_num']  = df['ClinicalStageAtLastVisit'].map(stage_map)
    df['stage_worsened']  = (df['stage_last_num'] > df['stage_start_num']).astype(int)

    df['sex_female']      = (df['Sex'].str.lower().str.strip() == 'female').astype(int)
    df['tb_positive']     = df['TbStatusAtLAstVisit'].str.lower().str.contains(
        'positive|active', na=False).astype(int)
    df['opp_infection']   = (
        df['OpportunisticInfectionPresentAtLastVisit'].str.lower().str.strip() == 'yes'
    ).astype(int)
    df['side_effects']    = (
        df['AnySideEffects'].str.lower().str.strip() == 'yes'
    ).astype(int)

    # ART interruption — kept for secondary analysis only
    df['had_interruption'] = (
        df['ArtInterruption'].str.lower().str.strip() == 'yes'
    ).astype(int)

    # Target: binary poor adherence
    df[TARGET] = (
        df['ArvAdherenceLatestLevel'].str.lower().str.strip() == 'poor'
    ).astype(float)

    print(f"\nFeature engineering complete.")
    print(f"  Target prevalence: {df[TARGET].mean()*100:.1f}% poor adherence")
    print(f"  Missing values (top 5):")
    for col, cnt in df[FEATURES].isnull().sum().sort_values(ascending=False).head().items():
        print(f"    {col}: {cnt}")

    return df


def prepare_model_data(df: pd.DataFrame):
    """
    Prepare model-ready data with proper train/test separation.
    
    IMPORTANT — Leakage Prevention:
    - The SimpleImputer here is fitted on the FULL dataset for the purpose
      of saving a reference imputer. In the cross-validation scripts (02_),
      the imputer is RE-FITTED exclusively on each training fold and applied
      to validation/test folds. This is handled via scikit-learn Pipeline.
    - The SMOTE oversampling is applied ONLY within training folds in 02_.
    - The 20% hold-out test set is never touched during imputer fitting in CV.
    
    Returns:
        X_imp   — imputed features (natural distribution, no SMOTE)
        y       — labels (natural distribution)
        X_bal   — SMOTE-balanced features (for CV experiments only)
        y_bal   — SMOTE-balanced labels
        imputer — fitted imputer (for reference/deployment inference)
        df_m    — cleaned model dataframe
    """
    df_m = df[FEATURES + [TARGET]].copy().dropna(subset=[TARGET])
    df_m = df_m[df_m[TARGET].isin([0, 1])]

    print(f"\nModel dataset: {len(df_m):,} patients")
    print(f"  Poor adherence: {int(df_m[TARGET].sum())} ({df_m[TARGET].mean()*100:.1f}%)")

    # NOTE: Imputer fitted here on full dataset for reference storage only.
    # All CV experiments in 02_ refit imputer per training fold via Pipeline.
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(df_m[FEATURES])
    y = df_m[TARGET].values.astype(int)

    # SMOTE — used for CV training folds only (applied in 02_ within Pipeline)
    # Provided here as pre-balanced dataset for convenience
    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_bal, y_bal = sm.fit_resample(X_imp, y)
    print(f"  After SMOTE: {len(X_bal):,} samples | "
          f"{y_bal.mean()*100:.0f}% positive")
    print(f"  Note: SMOTE applied to training folds only in cross-validation (02_model_training_cv.py)")

    return X_imp, y, X_bal, y_bal, imputer, df_m


def main(qoc_path: str, cephia_path: str):
    print("=" * 60)
    print("Step 1: Data Preprocessing")
    print("=" * 60)

    # Load
    df_qoc    = load_qoc(qoc_path)
    df_cephia = load_cephia(cephia_path)

    # Engineer features
    df_qoc = engineer_features(df_qoc)

    # Prepare model data
    X_imp, y, X_bal, y_bal, imputer, df_m = prepare_model_data(df_qoc)

    # Save processed data
    with open('results/preprocessed_data.pkl', 'wb') as f:
        pickle.dump({
            'X_imp': X_imp, 'y': y,
            'X_bal': X_bal, 'y_bal': y_bal,
            'imputer': imputer,
            'df_m': df_m,
            'df_qoc': df_qoc,
            'df_cephia': df_cephia,
            'features': FEATURES,
        }, f)

    print("\nSaved: results/preprocessed_data.pkl")
    print("Step 1 complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qoc_path',    default='data/QualityOfCare.xlsx')
    parser.add_argument('--cephia_path', default='data/cephia_public_use_dataset_20210604.csv')
    args = parser.parse_args()
    main(args.qoc_path, args.cephia_path)

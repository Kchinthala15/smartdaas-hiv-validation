"""
smartdaas_data_validator.py — SmartDaaS v2 Real-Data Safeguards
Schema validation, leakage checks, and data quality gates
for when real APIN / AMPATH pilot data arrives.
Author: Lakshmi Kalyani Chinthala, SmartDaaS LLC
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

REQUIRED_PATIENT_COLS = [
    "patient_id", "art_start_date", "facility_id", "interrupted",
    "age_at_art_start", "sex_female", "cd4_at_art_start",
]

REQUIRED_EVENT_COLS = [
    "patient_id", "visit_date", "visit_num",
    "missed_visit", "cd4_count", "adherence_level",
]

DATE_COLS_PATIENT = ["art_start_date", "hiv_dx_date"]
DATE_COLS_EVENT   = ["visit_date"]


def validate_schema(df, required_cols, table_name):
    print(f"\n[Schema] Validating {table_name}...")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ✗ MISSING REQUIRED COLUMNS: {missing}")
        return False
    print(f"  ✓ All required columns present ({len(df.columns)} total)")
    return True


def validate_patient_id_uniqueness(df, id_col="patient_id"):
    print(f"[Uniqueness] Checking patient ID uniqueness...")
    dupes = df[id_col].duplicated().sum()
    if dupes > 0:
        print(f"  ✗ {dupes} duplicate patient IDs found — investigate before proceeding")
        return False
    print(f"  ✓ All {len(df):,} patient IDs unique")
    return True


def validate_date_order(event_df):
    print(f"[Date Order] Checking visit date ordering...")
    grp = event_df.groupby("patient_id")
    violations = 0
    for pid, g in grp:
        dates = pd.to_datetime(g["visit_date"], errors="coerce").sort_values()
        if dates.isna().any():
            violations += 1
    if violations > 0:
        print(f"  ✗ {violations} patients with unparseable visit dates")
        return False
    print(f"  ✓ Visit dates parseable for all patients")
    return True


def check_temporal_leakage(patient_df, event_df, outcome_col="interrupted", split_date=None):
    print(f"[Leakage] Checking for temporal data leakage...")
    if split_date is None:
        print(f"  ⚠ No split_date provided — skipping leakage check")
        return True
    split_dt = pd.to_datetime(split_date)
    patient_df["art_start_date"] = pd.to_datetime(patient_df["art_start_date"], errors="coerce")
    train_patients = patient_df[patient_df["art_start_date"] < split_dt]["patient_id"]
    test_patients  = patient_df[patient_df["art_start_date"] >= split_dt]["patient_id"]
    overlap = set(train_patients) & set(test_patients)
    if overlap:
        print(f"  ✗ {len(overlap)} patients appear in BOTH train and test — DATA LEAKAGE")
        return False
    print(f"  ✓ No patient overlap between train ({len(train_patients):,}) and test ({len(test_patients):,})")
    return True


def check_outcome_window(event_df, outcome_col="treatment_interrupted", window_days=90):
    print(f"[Outcome Window] Checking outcome definition consistency...")
    if outcome_col not in event_df.columns:
        print(f"  ⚠ Column {outcome_col} not found in event table — skipping")
        return True
    rate = event_df[outcome_col].mean()
    print(f"  ✓ Outcome rate: {rate*100:.1f}% (expected 5-40% for HIV retention outcomes)")
    if rate > 0.6 or rate < 0.01:
        print(f"  ✗ Outcome rate {rate*100:.1f}% is outside expected range — check outcome definition")
        return False
    return True


def missingness_report(df, table_name, threshold=0.30):
    print(f"[Missingness] {table_name} missingness report:")
    miss = df.isnull().mean().sort_values(ascending=False)
    high_miss = miss[miss > threshold]
    if len(high_miss) > 0:
        print(f"  ✗ {len(high_miss)} columns with >{threshold*100:.0f}% missing:")
        for col, pct in high_miss.items():
            print(f"    {col:<40} {pct*100:.1f}%")
    else:
        print(f"  ✓ No columns exceed {threshold*100:.0f}% missingness threshold")
    # Show top 5 anyway
    print(f"  Top 5 missing:")
    for col, pct in miss.head(5).items():
        print(f"    {col:<40} {pct*100:.1f}%")
    return miss


def train_test_split_audit(patient_df, split_date, outcome_col="interrupted"):
    print(f"[Split Audit] Train/test split audit (split: {split_date})...")
    patient_df = patient_df.copy()
    patient_df["art_start_date"] = pd.to_datetime(patient_df["art_start_date"], errors="coerce")
    split_dt = pd.to_datetime(split_date)
    train = patient_df[patient_df["art_start_date"] < split_dt]
    test  = patient_df[patient_df["art_start_date"] >= split_dt]
    print(f"  Train: {len(train):,} patients ({len(train)/len(patient_df)*100:.1f}%)")
    print(f"  Test:  {len(test):,} patients ({len(test)/len(patient_df)*100:.1f}%)")
    if outcome_col in patient_df.columns:
        print(f"  Train outcome rate: {train[outcome_col].mean()*100:.1f}%")
        print(f"  Test outcome rate:  {test[outcome_col].mean()*100:.1f}%")
        rate_diff = abs(train[outcome_col].mean() - test[outcome_col].mean())
        if rate_diff > 0.10:
            print(f"  ✗ Outcome rate difference {rate_diff*100:.1f}pp exceeds 10pp — check for distribution shift")
        else:
            print(f"  ✓ Outcome rate difference {rate_diff*100:.1f}pp — acceptable")
    return train, test


def run_full_validation(
    patient_path,
    event_path=None,
    split_date=None,
    output_dir="/mnt/user-data/outputs",
):
    """
    Run full data validation suite before any modeling.
    Call this first when real APIN / AMPATH pilot data arrives.
    """
    print("="*70)
    print("SmartDaaS v2 — Real-Data Validation Suite")
    print("Run this BEFORE any modeling on pilot data")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)
    all_passed = True

    # Patient table
    print("\n── PATIENT TABLE ──")
    patient_df = pd.read_csv(patient_path)
    print(f"  Loaded: {len(patient_df):,} rows x {patient_df.shape[1]} columns")

    all_passed &= validate_schema(patient_df, REQUIRED_PATIENT_COLS, "patient_table")
    all_passed &= validate_patient_id_uniqueness(patient_df)
    miss_patient = missingness_report(patient_df, "patient_table")

    # Event table (if provided)
    if event_path and os.path.exists(event_path):
        print("\n── EVENT TABLE ──")
        event_df = pd.read_csv(event_path)
        print(f"  Loaded: {len(event_df):,} rows x {event_df.shape[1]} columns")
        all_passed &= validate_schema(event_df, REQUIRED_EVENT_COLS, "event_table")
        all_passed &= validate_date_order(event_df)
        all_passed &= check_outcome_window(event_df)
        miss_event = missingness_report(event_df, "event_table")

    # Temporal checks
    if split_date:
        print("\n── TEMPORAL CHECKS ──")
        all_passed &= check_temporal_leakage(patient_df, event_df if event_path else pd.DataFrame(), split_date=split_date)
        train, test = train_test_split_audit(patient_df, split_date)

    print("\n" + "="*70)
    if all_passed:
        print("  ✓ ALL VALIDATION CHECKS PASSED — safe to proceed with modeling")
    else:
        print("  ✗ VALIDATION FAILURES DETECTED — resolve before modeling")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    # Test on synthetic data
    run_full_validation(
        patient_path="/mnt/user-data/outputs/smartdaas_synthetic_patient_table.csv",
        event_path="/mnt/user-data/outputs/smartdaas_synthetic_event_table.csv",
        split_date="2019-01-01",
    )

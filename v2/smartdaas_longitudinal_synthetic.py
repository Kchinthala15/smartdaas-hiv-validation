"""
smartdaas_longitudinal_synthetic.py
────────────────────────────────────────────────────────────────────────────
SmartDaaS v2 — Longitudinal Architecture Foundation
Synthetic HIV Patient Trajectory Generator

Purpose:
    Generate realistic longitudinal HIV programme data to prototype and
    benchmark next-generation architectures including:
        - Temporal transformers / BEHRT-style sequence models
        - Survival models (time-to-interruption prediction)
        - Multi-task learning (interruption + viral failure + missed visits)
        - Drift detection and dynamic recalibration
        - Facility embeddings and graph intelligence
        - Causal / uplift modeling

    NOTE: Synthetic data only. Not for publication-grade conclusions.
    Designed to enable architecture prototyping before real programme
    data becomes available through APIN / AMPATH pilot studies.

Outputs:
    1. event_table      — patient-event longitudinal table (one row per event)
    2. patient_table    — patient-level derived feature summary
    3. survival_table   — time-to-interruption labels for survival modeling
    4. facility_table   — facility-level metadata
    5. temporal_split   — train/test split by calendar time

Author:  Lakshmi Kalyani Chinthala, SmartDaaS LLC
Contact: lkchinthala@smartdaas.org
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# ── SEED ─────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── SIMULATION PARAMETERS ────────────────────────────────────────────────────
N_PATIENTS         = 5_000          # synthetic cohort size
N_FACILITIES       = 40             # number of simulated facilities
START_DATE         = datetime(2015, 1, 1)
END_DATE           = datetime(2023, 12, 31)
MAX_FOLLOW_UP_DAYS = 1095           # 3 years max follow-up
VISIT_INTERVAL_DAYS = 30            # standard monthly visit schedule
REFILL_INTERVAL_DAYS = 30           # standard monthly refill

# ── COUNTRIES / REGIONS ──────────────────────────────────────────────────────
COUNTRIES = ['Nigeria', 'Kenya', 'Uganda', 'Zambia', 'Tanzania', 'Malawi']
FACILITY_LEVELS = ['Primary', 'Secondary', 'Tertiary']
FACILITY_TYPES  = ['Public', 'Private', 'Faith-based', 'NGO']
PARTNERS        = ['APIN', 'AMPATH', 'FHI360', 'Jhpiego', 'EGPAF', 'ICAP']

# ── CLINICAL DISTRIBUTIONS (calibrated to Nigerian QoC dataset) ──────────────
AGE_MEAN, AGE_STD           = 35.2, 11.0
CD4_START_MEAN, CD4_START_STD = 370.0, 220.0
WEIGHT_MEAN, WEIGHT_STD     = 62.0, 12.0
DAYS_TO_ART_MEDIAN          = 74
FEMALE_PROB                 = 0.614
WHO_STAGE_PROBS             = [0.312, 0.387, 0.241, 0.060]   # I, II, III, IV
INTERRUPTION_BASE_PROB      = 0.004   # baseline monthly interruption risk (~15% annual)
TB_COINFECTION_PROB         = 0.12
OI_PROB                     = 0.08
SIDE_EFFECTS_PROB           = 0.15
REGIMEN_CHANGE_PROB         = 0.05    # monthly probability of regimen change
MISSED_VISIT_BASE_PROB      = 0.12    # baseline monthly missed visit probability
VIRAL_FAILURE_BASE_PROB     = 0.08    # baseline probability if adherence poor


# ════════════════════════════════════════════════════════════════════════════
# 1. FACILITY METADATA TABLE
# ════════════════════════════════════════════════════════════════════════════

def generate_facility_table(n_facilities: int = N_FACILITIES) -> pd.DataFrame:
    """
    Generate facility-level metadata including geographic and structural
    characteristics that influence patient outcomes.
    These will later feed facility embeddings and graph neural layers.
    """
    np.random.seed(SEED)
    records = []
    for fid in range(1, n_facilities + 1):
        country = np.random.choice(COUNTRIES)
        records.append({
            'facility_id'         : f'FAC_{fid:04d}',
            'country'             : country,
            'region'              : f'{country}_Region_{np.random.randint(1, 6)}',
            'lga'                 : f'LGA_{np.random.randint(1, 20):03d}',
            'facility_level'      : np.random.choice(FACILITY_LEVELS, p=[0.5, 0.35, 0.15]),
            'facility_type'       : np.random.choice(FACILITY_TYPES, p=[0.55, 0.15, 0.15, 0.15]),
            'partner'             : np.random.choice(PARTNERS),
            'n_patients_enrolled' : np.random.randint(200, 3000),
            'staff_ratio'         : round(np.random.uniform(0.3, 2.5), 2),
            'distance_km_median'  : round(np.random.exponential(12), 1),
            'supply_chain_score'  : round(np.random.uniform(0.4, 1.0), 2),
            'rural_flag'          : int(np.random.random() < 0.45),
            'security_index'      : round(np.random.uniform(0.3, 1.0), 2),
            'transport_access'    : round(np.random.uniform(0.2, 1.0), 2),
        })
    df = pd.DataFrame(records)
    print(f"[facility_table] Generated {len(df)} facilities across {df['country'].nunique()} countries")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. PATIENT BASELINE TABLE
# ════════════════════════════════════════════════════════════════════════════

def generate_patient_baselines(
    n_patients: int = N_PATIENTS,
    facility_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generate patient-level baseline characteristics at ART initiation.
    Calibrated to Nigerian QoC dataset distributions.
    """
    np.random.seed(SEED)
    facilities = facility_df['facility_id'].tolist() if facility_df is not None else [f'FAC_{i:04d}' for i in range(1, N_FACILITIES + 1)]

    art_start_days = np.random.randint(0, (datetime(2021, 12, 31) - START_DATE).days, n_patients)
    art_start_dates = [START_DATE + timedelta(days=int(d)) for d in art_start_days]

    days_to_art = np.random.exponential(DAYS_TO_ART_MEDIAN, n_patients).clip(0, 1825).astype(int)
    hiv_dx_dates = [art_start_dates[i] - timedelta(days=int(days_to_art[i])) for i in range(n_patients)]

    ages = np.random.normal(AGE_MEAN, AGE_STD, n_patients).clip(15, 75).astype(int)
    sex_female = np.random.binomial(1, FEMALE_PROB, n_patients)
    cd4_start = np.random.normal(CD4_START_MEAN, CD4_START_STD, n_patients).clip(1, 1500).astype(int)
    weight_start = np.random.normal(WEIGHT_MEAN, WEIGHT_STD, n_patients).clip(30, 130)
    height = np.random.normal(163, 9, n_patients).clip(140, 200)
    bmi_start = (weight_start / (height / 100) ** 2).clip(12, 50)
    who_stage = np.random.choice([1, 2, 3, 4], n_patients, p=WHO_STAGE_PROBS)
    tb_baseline = np.random.binomial(1, TB_COINFECTION_PROB, n_patients)
    oi_baseline = np.random.binomial(1, OI_PROB, n_patients)

    # Risk score — determines trajectory (higher = more likely to interrupt)
    risk_latent = (
        (cd4_start < 200).astype(float) * 0.4 +
        (who_stage >= 3).astype(float) * 0.3 +
        (sex_female == 0).astype(float) * 0.1 +
        (ages < 25).astype(float) * 0.2 +
        tb_baseline * 0.1 +
        np.random.normal(0, 0.3, n_patients)
    ).clip(0, 1)

    regimens = np.random.choice(
        ['TDF/3TC/EFV', 'TDF/3TC/DTG', 'AZT/3TC/NVP', 'TDF/FTC/DTG'],
        n_patients, p=[0.35, 0.40, 0.15, 0.10]
    )

    df = pd.DataFrame({
        'patient_id'        : [f'PAT_{i:06d}' for i in range(1, n_patients + 1)],
        'facility_id'       : np.random.choice(facilities, n_patients),
        'hiv_dx_date'       : hiv_dx_dates,
        'art_start_date'    : art_start_dates,
        'days_to_art'       : days_to_art,
        'age_at_art_start'  : ages,
        'sex_female'        : sex_female,
        'cd4_at_art_start'  : cd4_start,
        'weight_at_start'   : weight_start.round(1),
        'height_cm'         : height.round(1),
        'bmi_at_start'      : bmi_start.round(1),
        'who_stage_start'   : who_stage,
        'tb_at_baseline'    : tb_baseline,
        'oi_at_baseline'    : oi_baseline,
        'initial_regimen'   : regimens,
        'risk_latent'       : risk_latent.round(4),
    })

    print(f"[patient_baselines] Generated {len(df):,} patients")
    print(f"  Age: {df['age_at_art_start'].mean():.1f} ± {df['age_at_art_start'].std():.1f}")
    print(f"  Female: {df['sex_female'].mean()*100:.1f}%")
    print(f"  CD4 at start: {df['cd4_at_art_start'].mean():.0f} ± {df['cd4_at_art_start'].std():.0f}")
    print(f"  WHO Stage III/IV: {(df['who_stage_start'] >= 3).mean()*100:.1f}%")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 3. EVENT-LEVEL LONGITUDINAL TABLE
# ════════════════════════════════════════════════════════════════════════════

def generate_event_table(
    patient_df: pd.DataFrame,
    facility_df: pd.DataFrame,
    max_follow_up_days: int = MAX_FOLLOW_UP_DAYS,
) -> pd.DataFrame:
    """
    Simulate realistic HIV patient event timelines including:
    - Clinic visits (scheduled and actual)
    - Refill pickups
    - Lab measurements (CD4, viral load)
    - Regimen changes
    - Side effects
    - TB / OI events
    - Adherence assessments
    - Treatment interruption events

    This is the core longitudinal data structure for:
    - Temporal transformers (sequence of events per patient)
    - Survival models (time to interruption)
    - Multi-task learning (multiple outcomes per visit)
    """
    np.random.seed(SEED)
    facility_lookup = facility_df.set_index('facility_id').to_dict('index') if facility_df is not None else {}

    all_events = []
    interruption_records = []

    for _, pat in patient_df.iterrows():
        pid = pat['patient_id']
        art_start = pat['art_start_date']
        risk = pat['risk_latent']
        current_cd4 = pat['cd4_at_art_start']
        current_weight = pat['weight_at_start']
        current_regimen = pat['initial_regimen']
        who_stage = pat['who_stage_start']
        interrupted = False
        interruption_day = None
        visit_day = 0
        visit_num = 0
        cumulative_missed = 0
        viral_load = np.random.choice([50, 200, 1000, 5000, 50000],
                                       p=[0.70, 0.10, 0.08, 0.07, 0.05])

        while visit_day <= max_follow_up_days and not interrupted:
            visit_num += 1
            visit_date = art_start + timedelta(days=int(visit_day))

            if visit_date > END_DATE:
                break

            # --- Missed visit probability (increases with risk and cumulative misses)
            missed_prob = min(
                MISSED_VISIT_BASE_PROB + risk * 0.25 + cumulative_missed * 0.02,
                0.65
            )
            missed_visit = int(np.random.random() < missed_prob)
            cumulative_missed = cumulative_missed + missed_visit if missed_visit else max(0, cumulative_missed - 1)

            # --- Refill gap (days late for refill pickup)
            refill_gap_days = int(np.random.exponential(3) * (1 + risk) * (1 + missed_visit * 2))

            # --- CD4 trajectory
            cd4_change = np.random.normal(
                15 - risk * 30 - missed_visit * 20,
                40
            )
            current_cd4 = max(1, min(1500, current_cd4 + cd4_change))

            # --- Viral load (measured every ~6 months)
            if visit_num % 6 == 0:
                if missed_visit or risk > 0.5:
                    viral_load = np.random.choice(
                        [50, 1000, 10000, 100000],
                        p=[0.3, 0.25, 0.25, 0.2]
                    )
                else:
                    viral_load = np.random.choice(
                        [50, 200, 1000],
                        p=[0.80, 0.12, 0.08]
                    )
            viral_suppressed = int(viral_load < 1000)

            # --- Weight change
            weight_change = np.random.normal(0.2 - risk * 0.5, 1.5)
            current_weight = max(25, current_weight + weight_change)

            # --- WHO stage worsening
            if np.random.random() < risk * 0.01:
                who_stage = min(4, who_stage + 1)

            # --- Side effects
            side_effects = int(np.random.random() < SIDE_EFFECTS_PROB * (1 + risk))

            # --- TB event
            tb_event = int(np.random.random() < 0.005 + risk * 0.01)

            # --- OI event
            oi_event = int(np.random.random() < 0.003 + risk * 0.008)

            # --- Regimen change
            if np.random.random() < REGIMEN_CHANGE_PROB * (1 + side_effects):
                regimen_options = ['TDF/3TC/EFV', 'TDF/3TC/DTG', 'AZT/3TC/NVP', 'TDF/FTC/DTG']
                regimen_options = [r for r in regimen_options if r != current_regimen]
                current_regimen = np.random.choice(regimen_options)
                regimen_changed = 1
            else:
                regimen_changed = 0

            # --- Adherence level
            if missed_visit or refill_gap_days > 7 or viral_load >= 1000:
                adherence_prob_poor = min(0.6, 0.1 + risk * 0.4 + missed_visit * 0.2)
            else:
                adherence_prob_poor = max(0.01, 0.05 * risk)
            adherence_rand = np.random.random()
            if adherence_rand < adherence_prob_poor:
                adherence = 'Poor'
            elif adherence_rand < adherence_prob_poor + 0.2:
                adherence = 'Fair'
            else:
                adherence = 'Good'

            # --- Treatment interruption
            interrupt_prob = min(
                INTERRUPTION_BASE_PROB +
                risk * 0.015 +
                (adherence == 'Poor') * 0.008 +
                (missed_visit) * 0.005 +
                (viral_load >= 10000) * 0.006,
                0.08
            )
            interrupted = bool(np.random.random() < interrupt_prob)

            # --- Record event
            all_events.append({
                'patient_id'         : pid,
                'facility_id'        : pat['facility_id'],
                'visit_num'          : visit_num,
                'visit_date'         : visit_date.strftime('%Y-%m-%d'),
                'days_since_art'     : visit_day,
                'missed_visit'       : missed_visit,
                'refill_gap_days'    : refill_gap_days,
                'cd4_count'          : round(current_cd4),
                'viral_load'         : viral_load,
                'viral_suppressed'   : viral_suppressed,
                'weight_kg'          : round(current_weight, 1),
                'who_stage'          : who_stage,
                'adherence_level'    : adherence,
                'side_effects'       : side_effects,
                'tb_event'           : tb_event,
                'oi_event'           : oi_event,
                'regimen'            : current_regimen,
                'regimen_changed'    : regimen_changed,
                'cumulative_missed'  : cumulative_missed,
                'treatment_interrupted': int(interrupted),
            })

            if interrupted:
                interruption_day = visit_day
                break

            visit_day += VISIT_INTERVAL_DAYS

        # --- Survival record
        interrupted_flag = 1 if interruption_day is not None else 0
        time_to_event = interruption_day if interruption_day is not None else min(visit_day, max_follow_up_days)
        interruption_records.append({
            'patient_id'        : pid,
            'interrupted'       : interrupted_flag,
            'time_to_event_days': time_to_event,
            'n_visits'          : visit_num,
            'n_missed'          : sum(e['missed_visit'] for e in all_events if e['patient_id'] == pid),
        })

    event_df = pd.DataFrame(all_events)
    survival_df = pd.DataFrame(interruption_records)

    print(f"\n[event_table] Generated {len(event_df):,} events across {event_df['patient_id'].nunique():,} patients")
    print(f"  Total visits: {len(event_df):,}")
    print(f"  Missed visits: {event_df['missed_visit'].mean()*100:.1f}%")
    print(f"  Poor adherence: {(event_df['adherence_level']=='Poor').mean()*100:.1f}%")
    print(f"  Treatment interruptions: {survival_df['interrupted'].mean()*100:.1f}%")
    print(f"  Median time to interruption (interrupted only): {survival_df[survival_df['interrupted']==1]['time_to_event_days'].median():.0f} days")

    return event_df, survival_df


# ════════════════════════════════════════════════════════════════════════════
# 4. PATIENT-LEVEL DERIVED FEATURE TABLE
# ════════════════════════════════════════════════════════════════════════════

def derive_patient_features(
    patient_df: pd.DataFrame,
    event_df: pd.DataFrame,
    survival_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive patient-level summary features from the longitudinal event table.
    These mirror the 15 features in the current SmartDaaS v1 model,
    extended with longitudinal-derived features for v2 architecture.
    """
    grp = event_df.groupby('patient_id')

    derived = pd.DataFrame({
        'patient_id'                : grp['patient_id'].first(),
        # SmartDaaS v1 equivalent features
        'most_recent_cd4'           : grp['cd4_count'].last(),
        'cd4_improvement'           : grp['cd4_count'].last() - grp['cd4_count'].first(),
        'most_recent_weight'        : grp['weight_kg'].last(),
        'weight_change'             : grp['weight_kg'].last() - grp['weight_kg'].first(),
        'who_stage_last'            : grp['who_stage'].last(),
        'stage_worsened'            : (grp['who_stage'].last() > grp['who_stage'].first()).astype(int),
        'had_interruption_history'  : grp['treatment_interrupted'].max(),
        'opp_infection_any'         : grp['oi_event'].max(),
        'side_effects_any'          : grp['side_effects'].max(),
        'tb_any'                    : grp['tb_event'].max(),
        # Extended longitudinal features (v2)
        'n_total_visits'            : grp['visit_num'].count(),
        'n_missed_visits'           : grp['missed_visit'].sum(),
        'missed_visit_rate'         : grp['missed_visit'].mean().round(4),
        'mean_refill_gap_days'      : grp['refill_gap_days'].mean().round(1),
        'max_refill_gap_days'       : grp['refill_gap_days'].max(),
        'n_regimen_changes'         : grp['regimen_changed'].sum(),
        'final_viral_load'          : grp['viral_load'].last(),
        'viral_suppressed_last'     : grp['viral_suppressed'].last(),
        'n_poor_adherence_visits'   : (event_df[event_df['adherence_level']=='Poor'].groupby('patient_id').size()).reindex(grp.groups.keys(), fill_value=0),
        'last_adherence_level'      : grp['adherence_level'].last(),
        'days_follow_up'            : grp['days_since_art'].max(),
        'last_regimen'              : grp['regimen'].last(),
    }).reset_index(drop=True)

    # Merge with baseline and survival
    derived = derived.merge(
        patient_df[['patient_id', 'age_at_art_start', 'sex_female',
                    'cd4_at_art_start', 'weight_at_start', 'bmi_at_start',
                    'who_stage_start', 'days_to_art', 'facility_id',
                    'art_start_date', 'risk_latent']],
        on='patient_id', how='left'
    )
    derived = derived.merge(
        survival_df[['patient_id', 'interrupted', 'time_to_event_days']],
        on='patient_id', how='left'
    )

    # Binary target (matches SmartDaaS v1 framing)
    derived['target_poor_adherence'] = (derived['last_adherence_level'] == 'Poor').astype(int)

    print(f"\n[patient_features] Derived {len(derived):,} patient-level records")
    print(f"  Poor adherence (target): {derived['target_poor_adherence'].mean()*100:.1f}%")
    print(f"  Treatment interrupted: {derived['interrupted'].mean()*100:.1f}%")
    print(f"  Mean follow-up: {derived['days_follow_up'].mean():.0f} days")
    print(f"  Mean missed visit rate: {derived['missed_visit_rate'].mean()*100:.1f}%")

    return derived


# ════════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL TRAIN / TEST SPLIT
# ════════════════════════════════════════════════════════════════════════════

def temporal_split(
    patient_df: pd.DataFrame,
    split_date: str = '2021-01-01',
) -> tuple:
    """
    Split patients into train/test sets by ART start date.
    Mirrors the temporal validation approach in SmartDaaS v1
    (trained on earlier records, tested on later records).
    Ready for use in survival models and transformer training.
    """
    split_dt = pd.to_datetime(split_date)
    patient_df['art_start_date'] = pd.to_datetime(patient_df['art_start_date'])

    train = patient_df[patient_df['art_start_date'] < split_dt].copy()
    test  = patient_df[patient_df['art_start_date'] >= split_dt].copy()

    print(f"\n[temporal_split] Split date: {split_date}")
    print(f"  Train: {len(train):,} patients ({len(train)/len(patient_df)*100:.1f}%)")
    print(f"  Test:  {len(test):,} patients ({len(test)/len(patient_df)*100:.1f}%)")
    print(f"  Train interruption rate: {train['interrupted'].mean()*100:.1f}%")
    print(f"  Test interruption rate:  {test['interrupted'].mean()*100:.1f}%")

    return train, test


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN — GENERATE ALL TABLES
# ════════════════════════════════════════════════════════════════════════════

def generate_all(
    n_patients: int = N_PATIENTS,
    n_facilities: int = N_FACILITIES,
    output_dir: str = '/mnt/user-data/outputs',
    save_csv: bool = True,
) -> dict:
    """
    Generate the complete SmartDaaS longitudinal synthetic dataset.
    Returns all tables as a dictionary and optionally saves to CSV.
    """
    print("=" * 70)
    print("SmartDaaS Longitudinal Synthetic Data Generator")
    print(f"Patients: {n_patients:,}  |  Facilities: {n_facilities}  |  Seed: {SEED}")
    print("=" * 70)

    # Generate
    facility_df  = generate_facility_table(n_facilities)
    patient_df   = generate_patient_baselines(n_patients, facility_df)
    event_df, survival_df = generate_event_table(patient_df, facility_df)
    feature_df   = derive_patient_features(patient_df, event_df, survival_df)
    train_df, test_df = temporal_split(feature_df, split_date='2019-01-01')

    tables = {
        'facility_table'  : facility_df,
        'event_table'     : event_df,
        'survival_table'  : survival_df,
        'patient_table'   : feature_df,
        'train_split'     : train_df,
        'test_split'      : test_df,
    }

    if save_csv:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, df in tables.items():
            path = f"{output_dir}/smartdaas_synthetic_{name}.csv"
            df.to_csv(path, index=False)
            print(f"  Saved: {path}  ({len(df):,} rows)")

    print("\n" + "=" * 70)
    print("Generation complete.")
    print("Next steps:")
    print("  1. Fit Random Survival Forest on survival_table → time-to-event")
    print("  2. Build transformer encoder on event_table → sequence modeling")
    print("  3. Prototype multi-task learning on patient_table")
    print("  4. Implement drift detection on temporal_split")
    print("  5. Build facility embeddings on facility_table + patient_table")
    print("=" * 70)

    return tables


# ── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tables = generate_all(
        n_patients=N_PATIENTS,
        n_facilities=N_FACILITIES,
        output_dir='/mnt/user-data/outputs',
        save_csv=True,
    )

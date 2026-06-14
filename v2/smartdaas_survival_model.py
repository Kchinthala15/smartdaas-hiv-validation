import os
"""
smartdaas_survival_model.py
────────────────────────────────────────────────────────────────────────────
SmartDaaS v2 — Survival Model Module
Time-to-Treatment-Interruption Prediction

Purpose:
    Instead of predicting "high risk / low risk" (binary classification),
    this module predicts WHEN a patient is likely to interrupt treatment.

    Output: "This patient is likely to interrupt within 45 days"
    vs v1:  "This patient is high risk"

    This is more clinically actionable — field teams can prioritise
    patients whose interruption is imminent, not just eventually likely.

Models implemented:
    1. Random Survival Forest (RSF) — primary model, tree-based, handles
       non-linearity, missing data, feature interactions
    2. Kaplan-Meier baseline — population-level survival curve
    3. Risk group stratification — translates survival output into
       operational priority tiers (Urgent / High / Moderate / Low)

Evaluation metrics:
    - Concordance index (C-index) — survival equivalent of AUC
    - Integrated Brier Score (IBS) — calibration over time
    - Time-specific AUC at 90, 180, 365 days
    - Survival curves by risk group

Next steps (after pilot data):
    - DeepSurv (neural survival model)
    - DeepHit (competing risks)
    - Transformer survival architecture

Author:  Lakshmi Kalyani Chinthala, SmartDaaS LLC
Contact: lkchinthala@smartdaas.org
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator

SEED = 42
np.random.seed(SEED)

# ── FEATURE SETS ─────────────────────────────────────────────────────────────
# V1 equivalent features (available from single snapshot)
V1_FEATURES = [
    'age_at_art_start', 'sex_female', 'cd4_at_art_start',
    'most_recent_cd4', 'cd4_improvement', 'who_stage_start',
    'weight_at_start', 'weight_change', 'bmi_at_start',
    'days_to_art', 'had_interruption_history', 'opp_infection_any',
    'side_effects_any', 'tb_any', 'stage_worsened'
]

# V2 extended features (longitudinal — available after visits accumulate)
V2_FEATURES = V1_FEATURES + [
    'n_total_visits', 'n_missed_visits', 'missed_visit_rate',
    'mean_refill_gap_days', 'max_refill_gap_days',
    'n_regimen_changes', 'final_viral_load', 'viral_suppressed_last',
    'n_poor_adherence_visits', 'days_follow_up'
]


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def prepare_survival_data(patient_df: pd.DataFrame, feature_set: list) -> tuple:
    """
    Prepare data for survival analysis.
    Returns X (features), y (structured survival array), feature names.
    """
    df = patient_df.copy()

    # Encode categorical
    for col in ['last_adherence_level', 'last_regimen', 'initial_regimen']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Select available features
    available = [f for f in feature_set if f in df.columns]
    X = df[available].copy()

    # Impute missing
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=available)

    # Survival structured array: (event, time)
    y = np.array(
        [(bool(row['interrupted']), float(row['time_to_event_days']))
         for _, row in df.iterrows()],
        dtype=[('interrupted', bool), ('time_to_event_days', float)]
    )

    print(f"  Features: {len(available)}")
    print(f"  Patients: {len(X_imp):,}")
    print(f"  Events (interruptions): {df['interrupted'].sum():,} ({df['interrupted'].mean()*100:.1f}%)")
    print(f"  Median time to event: {df['time_to_event_days'].median():.0f} days")

    return X_imp, y, available


# ════════════════════════════════════════════════════════════════════════════
# 2. KAPLAN-MEIER BASELINE
# ════════════════════════════════════════════════════════════════════════════

def fit_kaplan_meier(y: np.ndarray) -> tuple:
    """
    Population-level survival curve.
    Shows overall probability of remaining interruption-free over time.
    """
    time, survival_prob = kaplan_meier_estimator(
        y['interrupted'], y['time_to_event_days']
    )
    print(f"\n[Kaplan-Meier] Population survival estimates:")
    for t in [90, 180, 365, 548, 730]:
        idx = np.searchsorted(time, t, side='right') - 1
        if 0 <= idx < len(survival_prob):
            print(f"  At {t:4d} days: {survival_prob[idx]*100:.1f}% interruption-free")
    return time, survival_prob


# ════════════════════════════════════════════════════════════════════════════
# 3. RANDOM SURVIVAL FOREST
# ════════════════════════════════════════════════════════════════════════════

def fit_rsf(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_estimators: int = 200,
) -> tuple:
    """
    Random Survival Forest — primary survival model.
    Handles non-linear relationships, feature interactions,
    and missing data without distributional assumptions.
    """
    print(f"\n[Random Survival Forest]")
    print(f"  Fitting {n_estimators} trees...")

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=SEED,
    )
    rsf.fit(X_train, y_train)

    # C-index
    risk_scores = rsf.predict(X_test)
    c_index = concordance_index_censored(
        y_test['interrupted'], y_test['time_to_event_days'], risk_scores
    )[0]
    print(f"  C-index (test): {c_index:.4f}")

    # Integrated Brier Score
    times = np.percentile(y_test['time_to_event_days'], np.linspace(10, 90, 10))
    times = times[(times > y_train['time_to_event_days'].min()) &
                  (times < y_train['time_to_event_days'].max())]
    try:
        surv_funcs = rsf.predict_survival_function(X_test)
        surv_matrix = np.row_stack([fn(times) for fn in surv_funcs])
        ibs = integrated_brier_score(y_train, y_test, surv_matrix, times)
        print(f"  Integrated Brier Score: {ibs:.4f} (lower = better; 0.25 = uninformative)")
    except Exception:
        ibs = None
        print(f"  Integrated Brier Score: could not compute")

    # Time-specific risk at 90 / 180 / 365 days
    print(f"\n  Time-specific interruption probability (mean across test set):")
    surv_funcs = rsf.predict_survival_function(X_test)
    for t in [90, 180, 365]:
        probs = np.array([max(0, min(1, 1 - fn(t))) for fn in surv_funcs])
        print(f"    At {t:3d} days: {probs.mean()*100:.1f}% mean risk  |  "
              f"top 10%: {np.percentile(probs, 90)*100:.1f}%")

    # Feature importance via risk score correlation (RSF doesn't expose feature_importances_)
    risk_scores_train = rsf.predict(X_train)
    importances = pd.Series(
        [abs(np.corrcoef(X_train[col].fillna(X_train[col].median()), risk_scores_train)[0,1])
         for col in X_train.columns],
        index=X_train.columns
    ).sort_values(ascending=False)
    print(f"\n  Top 10 features by risk-score correlation:")
    for feat, imp in importances.head(10).items():
        print(f"    {feat:<35s} {imp:.4f}")

    return rsf, c_index, ibs, importances


# ════════════════════════════════════════════════════════════════════════════
# 4. RISK GROUP STRATIFICATION
# ════════════════════════════════════════════════════════════════════════════

def stratify_risk_groups(
    rsf,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    prediction_horizon_days: int = 180,
) -> pd.DataFrame:
    """
    Translate continuous survival output into operational priority tiers.
    This is the clinical decision support layer — what field teams see.

    Tiers:
        URGENT   — >50% probability of interruption within horizon
        HIGH     — 25-50%
        MODERATE — 10-25%
        LOW      — <10%
    """
    surv_funcs = rsf.predict_survival_function(X_test)
    interruption_probs = np.array([
        max(0, min(1, 1 - fn(prediction_horizon_days)))
        for fn in surv_funcs
    ])

    tiers = pd.cut(
        interruption_probs,
        bins=[0, 0.10, 0.25, 0.50, 1.01],
        labels=['LOW', 'MODERATE', 'HIGH', 'URGENT'],
        right=False
    )

    result = pd.DataFrame({
        'interruption_prob_180d': interruption_probs.round(4),
        'risk_tier': tiers,
        'actual_interrupted': y_test['interrupted'],
        'actual_time_days': y_test['time_to_event_days'],
    })

    print(f"\n[Risk Group Stratification] Horizon: {prediction_horizon_days} days")
    print(f"  {'Tier':<12} {'N':>6} {'%':>6} {'Actual interrupt rate':>22}")
    print(f"  {'-'*50}")
    for tier in ['URGENT', 'HIGH', 'MODERATE', 'LOW']:
        mask = result['risk_tier'] == tier
        n = mask.sum()
        pct = n / len(result) * 100
        actual_rate = result[mask]['actual_interrupted'].mean() * 100 if n > 0 else 0
        print(f"  {tier:<12} {n:>6} {pct:>5.1f}% {actual_rate:>20.1f}%")

    return result


# ════════════════════════════════════════════════════════════════════════════
# 5. SURVIVAL CURVES PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_survival_curves(
    rsf,
    X_test: pd.DataFrame,
    risk_df: pd.DataFrame,
    km_time: np.ndarray,
    km_survival: np.ndarray,
    output_path: str = f'{output_dir}/smartdaas_survival_curves.png'
) -> None:
    """
    Plot survival curves by risk tier and population KM curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('SmartDaaS v2 — Survival Model: Time-to-Treatment-Interruption',
                 fontsize=13, fontweight='bold', y=1.02)

    # Panel A: Kaplan-Meier population curve
    ax = axes[0]
    ax.step(km_time, km_survival, where='post', color='#0072B2', linewidth=2)
    ax.fill_between(km_time, km_survival, alpha=0.15, color='#0072B2', step='post')
    ax.axvline(x=180, color='#CC79A7', linestyle='--', alpha=0.7, label='180-day horizon')
    ax.axvline(x=365, color='#E69F00', linestyle='--', alpha=0.7, label='365-day horizon')
    ax.set_xlabel('Days since ART initiation')
    ax.set_ylabel('Probability of remaining interruption-free')
    ax.set_title('A. Population Kaplan-Meier Survival Curve')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1095)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Survival curves by risk tier
    ax = axes[1]
    colors = {'URGENT': '#CC0000', 'HIGH': '#E69F00',
              'MODERATE': '#009E73', 'LOW': '#0072B2'}
    times_plot = np.linspace(30, 1000, 100)

    for tier in ['URGENT', 'HIGH', 'MODERATE', 'LOW']:
        mask = (risk_df['risk_tier'] == tier).values
        if mask.sum() < 10:
            continue
        X_tier = X_test[mask]
        surv_funcs = rsf.predict_survival_function(X_tier)
        mean_surv = np.mean(
            np.row_stack([fn(times_plot) for fn in surv_funcs]), axis=0
        )
        n = mask.sum()
        ax.plot(times_plot, mean_surv, color=colors[tier],
                linewidth=2.5, label=f'{tier} (n={n})')

    ax.axvline(x=180, color='grey', linestyle='--', alpha=0.5)
    ax.set_xlabel('Days since ART initiation')
    ax.set_ylabel('Probability of remaining interruption-free')
    ax.set_title('B. Predicted Survival Curves by Risk Tier')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1000)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Plot] Saved: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ════════════════════════════════════════════════════════════════════════════

def run_survival_analysis(
    patient_table_path: str = '/mnt/user-data/outputs/smartdaas_synthetic_patient_table.csv',
    train_split_path: str = '/mnt/user-data/outputs/smartdaas_synthetic_train_split.csv',
    test_split_path: str = '/mnt/user-data/outputs/smartdaas_synthetic_test_split.csv',
):
    print("=" * 70)
    print("SmartDaaS v2 — Survival Model")
    print("Time-to-Treatment-Interruption Prediction")
    print("=" * 70)

    # Load data
    train_df = pd.read_csv(train_split_path)
    test_df  = pd.read_csv(test_split_path)
    print(f"\nLoaded: {len(train_df):,} train  |  {len(test_df):,} test")

    # --- V1 features (snapshot)
    print("\n── V1 Feature Set (15 features — snapshot equivalent) ──")
    X_train_v1, y_train, feats_v1 = prepare_survival_data(train_df, V1_FEATURES)
    X_test_v1,  y_test,  _        = prepare_survival_data(test_df,  V1_FEATURES)

    # --- V2 features (longitudinal)
    print("\n── V2 Feature Set (extended longitudinal features) ──")
    X_train_v2, _, feats_v2 = prepare_survival_data(train_df, V2_FEATURES)
    X_test_v2,  _, _        = prepare_survival_data(test_df,  V2_FEATURES)

    # --- Kaplan-Meier
    print("\n── Population Baseline ──")
    km_time, km_surv = fit_kaplan_meier(y_train)

    # --- RSF V1
    print("\n── Random Survival Forest — V1 features ──")
    rsf_v1, c_v1, ibs_v1, imp_v1 = fit_rsf(X_train_v1, y_train, X_test_v1, y_test)

    # --- RSF V2
    print("\n── Random Survival Forest — V2 features (longitudinal) ──")
    rsf_v2, c_v2, ibs_v2, imp_v2 = fit_rsf(X_train_v2, y_train, X_test_v2, y_test)

    # --- Comparison
    print("\n── Model Comparison ──")
    print(f"  {'Model':<30} {'C-index':>10} {'IBS':>10}")
    print(f"  {'-'*52}")
    print(f"  {'RSF V1 (15 features)':<30} {c_v1:>10.4f} {str(round(ibs_v1,4)) if ibs_v1 else 'N/A':>10}")
    print(f"  {'RSF V2 (longitudinal)':<30} {c_v2:>10.4f} {str(round(ibs_v2,4)) if ibs_v2 else 'N/A':>10}")
    print(f"\n  C-index improvement V1→V2: {(c_v2 - c_v1)*100:+.2f} percentage points")
    print(f"  This quantifies the VALUE of longitudinal data over snapshot data.")

    # --- Risk stratification
    print("\n── Operational Risk Stratification (V2 model) ──")
    risk_df = stratify_risk_groups(rsf_v2, X_test_v2, y_test)

    # --- Plot
    plot_survival_curves(rsf_v2, X_test_v2, risk_df, km_time, km_surv)

    # --- Save risk output
    risk_output_path = f'{output_dir}/smartdaas_survival_risk_output.csv'
    risk_df.to_csv(risk_output_path, index=False)
    print(f"[Output] Risk scores saved: {risk_output_path}")

    # --- Save feature importances
    imp_path = f'{output_dir}/smartdaas_survival_feature_importance.csv'
    imp_v2.reset_index().rename(columns={'index': 'feature', 0: 'importance'}).to_csv(imp_path, index=False)

    print("\n" + "=" * 70)
    print("Survival model complete.")
    print(f"  C-index V2: {c_v2:.4f} (1.0 = perfect, 0.5 = random)")
    print(f"  Key insight: {(c_v2 - c_v1)*100:+.2f}pp gain from longitudinal features")
    print("\nNext steps:")
    print("  → Build DeepSurv neural survival model")
    print("  → Implement competing risks (DeepHit)")
    print("  → Build transformer sequence encoder on event_table")
    print("  → Prototype multi-task learning")
    print("=" * 70)

    return rsf_v2, risk_df, imp_v2


if __name__ == '__main__':
    rsf, risk_df, importances = run_survival_analysis()

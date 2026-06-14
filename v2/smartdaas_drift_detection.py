import os
"""
smartdaas_drift_detection.py — SmartDaaS v2 Drift Detection Module
Population shift monitoring and automatic recalibration triggers
Author: Lakshmi Kalyani Chinthala, SmartDaaS LLC
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

MONITOR_FEATURES = [
    "age_at_art_start", "sex_female", "cd4_at_art_start", "most_recent_cd4",
    "who_stage_start", "days_to_art", "missed_visit_rate",
    "mean_refill_gap_days", "viral_suppressed_last", "n_regimen_changes",
]

DRIFT_THRESHOLDS = {
    "psi"           : 0.2,    # Population Stability Index — >0.2 = significant drift
    "ks_pvalue"     : 0.05,   # KS test p-value — <0.05 = significant drift
    "js_divergence" : 0.1,    # Jensen-Shannon divergence — >0.1 = moderate drift
    "auc_degradation": 0.05,  # AUC drop threshold — >5pp drop triggers recalibration
}


# ════════════════════════════════════════════════════════════════════════════
# 1. SIMULATE TEMPORAL DATA STREAMS
# ════════════════════════════════════════════════════════════════════════════

def simulate_temporal_windows(patient_df, n_windows=6):
    """
    Simulate quarterly data windows to mimic real programme deployment.
    Introduces gradual drift in later windows to test detection.
    """
    df = patient_df.copy()
    df["art_start_date"] = pd.to_datetime(df["art_start_date"])
    df = df.sort_values("art_start_date").reset_index(drop=True)
    
    n = len(df)
    window_size = n // n_windows
    windows = []
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else n
        window = df.iloc[start:end].copy()
        window["window"] = i + 1
        window["quarter"] = f"Q{i+1}"
        
        # Introduce artificial drift in later windows
        # (simulates guideline change, population shift, etc.)
        if i >= 3:
            drift_strength = (i - 2) * 0.3
            drift_mask = np.random.random(len(window)) < 0.3
            window = window.copy()
            window["cd4_at_art_start"] = window["cd4_at_art_start"].astype(float)
            window["days_to_art"] = window["days_to_art"].astype(float)
            window.loc[drift_mask, "cd4_at_art_start"] = (
                window.loc[drift_mask, "cd4_at_art_start"] * (1 + drift_strength * 0.4)
            ).clip(1, 1500)
            window.loc[drift_mask, "days_to_art"] = (
                window.loc[drift_mask, "days_to_art"] * (1 - drift_strength * 0.3)
            ).clip(0, 1000)
            window.loc[drift_mask, "missed_visit_rate"] = (
                window.loc[drift_mask, "missed_visit_rate"] + drift_strength * 0.1
            ).clip(0, 1)
        
        windows.append(window)
        print(f"  Window {i+1} (Q{i+1}): {len(window):,} patients  "
              f"interruption rate: {window['interrupted'].mean()*100:.1f}%"
              + (" [DRIFT INTRODUCED]" if i >= 3 else ""))
    
    return windows


# ════════════════════════════════════════════════════════════════════════════
# 2. DRIFT DETECTION METRICS
# ════════════════════════════════════════════════════════════════════════════

def compute_psi(expected, actual, n_bins=10):
    """
    Population Stability Index (PSI).
    Industry standard for monitoring distribution shift.
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change, monitor
    PSI > 0.2: Significant change, investigate / recalibrate
    """
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf; bins[-1] = np.inf
    
    exp_counts = np.histogram(expected, bins=bins)[0] + 1e-6
    act_counts = np.histogram(actual,   bins=bins)[0] + 1e-6
    
    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()
    
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def compute_js_divergence(expected, actual, n_bins=20):
    """Jensen-Shannon divergence — symmetric measure of distribution difference."""
    bins = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        n_bins + 1
    )
    p = np.histogram(expected, bins=bins, density=True)[0] + 1e-10
    q = np.histogram(actual,   bins=bins, density=True)[0] + 1e-10
    p = p / p.sum(); q = q / q.sum()
    return float(jensenshannon(p, q))


def detect_drift(reference_window, current_window, features=MONITOR_FEATURES):
    """
    Run full drift detection suite on a single feature set.
    Returns per-feature drift scores and overall drift verdict.
    """
    ref_avail = [f for f in features if f in reference_window.columns]
    imp = SimpleImputer(strategy="median")
    ref_data = imp.fit_transform(reference_window[ref_avail].fillna(0))
    cur_data = imp.transform(current_window[ref_avail].fillna(0))
    
    results = []
    for i, feat in enumerate(ref_avail):
        ref_col = ref_data[:, i]
        cur_col = cur_data[:, i]
        psi = compute_psi(ref_col, cur_col)
        ks_stat, ks_pval = stats.ks_2samp(ref_col, cur_col)
        js_div = compute_js_divergence(ref_col, cur_col)
        drift_flag = (
            psi > DRIFT_THRESHOLDS["psi"] or
            ks_pval < DRIFT_THRESHOLDS["ks_pvalue"] or
            js_div > DRIFT_THRESHOLDS["js_divergence"]
        )
        results.append({
            "feature"    : feat,
            "psi"        : round(psi, 4),
            "ks_pvalue"  : round(ks_pval, 4),
            "js_divergence": round(js_div, 4),
            "drift_detected": drift_flag,
            "severity"   : "HIGH" if psi > 0.25 else ("MODERATE" if psi > 0.1 else "LOW"),
        })
    
    drift_df = pd.DataFrame(results)
    n_drifted = drift_df["drift_detected"].sum()
    overall_drift = n_drifted >= 2  # trigger if 2+ features drift
    
    return drift_df, overall_drift, n_drifted


def monitor_model_performance(reference_window, current_window, features):
    """
    Monitor actual model AUC degradation over time.
    Trains a reference model on window 1, evaluates on current window.
    Triggers recalibration if AUC drops more than threshold.
    """
    avail = [f for f in features if f in reference_window.columns]
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_ref = scaler.fit_transform(imp.fit_transform(reference_window[avail].fillna(0)))
    y_ref = reference_window["interrupted"].values
    
    X_cur = scaler.transform(imp.transform(current_window[avail].fillna(0)))
    y_cur = current_window["interrupted"].values
    
    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(X_ref, y_ref)
    
    ref_auc = roc_auc_score(y_ref, model.predict_proba(X_ref)[:, 1])
    cur_auc = roc_auc_score(y_cur, model.predict_proba(X_cur)[:, 1])
    auc_drop = ref_auc - cur_auc
    
    recalibrate = auc_drop > DRIFT_THRESHOLDS["auc_degradation"]
    
    return ref_auc, cur_auc, auc_drop, recalibrate


# ════════════════════════════════════════════════════════════════════════════
# 3. DRIFT MONITORING DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def run_drift_monitoring(windows, reference_idx=0):
    """
    Run drift monitoring across all temporal windows.
    Reference window = deployment baseline (window 1).
    """
    ref_window = windows[reference_idx]
    avail_features = [f for f in MONITOR_FEATURES if f in ref_window.columns]
    
    print(f"\n[Drift Monitor] Reference: Window {reference_idx+1}")
    print(f"{'Window':<10} {'Drifted Features':<20} {'Overall Drift':<16} {'Ref AUC':<10} {'Cur AUC':<10} {'AUC Drop':<12} {'Action'}")
    print("-"*100)
    
    monitoring_log = []
    
    for i, cur_window in enumerate(windows):
        if i == reference_idx:
            continue
        
        drift_df, overall_drift, n_drifted = detect_drift(ref_window, cur_window, avail_features)
        ref_auc, cur_auc, auc_drop, recalibrate = monitor_model_performance(
            ref_window, cur_window, avail_features
        )
        
        action = "RECALIBRATE NOW" if (overall_drift and recalibrate) else                  "MONITOR CLOSELY" if overall_drift else                  "RECALIBRATE" if recalibrate else "OK"
        
        drift_feats = drift_df[drift_df["drift_detected"]]["feature"].tolist()
        
        print(f"W{i+1:<9} {n_drifted}/{len(avail_features)} {'['+ ','.join(drift_feats[:2])+'...'+ ']' if drift_feats else '[none]':<20} "
              f"{'YES' if overall_drift else 'NO':<16} {ref_auc:<10.4f} {cur_auc:<10.4f} "
              f"{auc_drop:+.4f}     {action}")
        
        monitoring_log.append({
            "window"         : f"W{i+1}",
            "n_drifted"      : n_drifted,
            "overall_drift"  : overall_drift,
            "ref_auc"        : ref_auc,
            "cur_auc"        : cur_auc,
            "auc_drop"       : auc_drop,
            "action"         : action,
            "drifted_features": drift_feats,
        })
    
    return pd.DataFrame(monitoring_log)


# ════════════════════════════════════════════════════════════════════════════
# 4. PLOT DRIFT DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def plot_drift_dashboard(log_df, windows, output_path=f"{output_dir}/smartdaas_drift_detection.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SmartDaaS v2 — Drift Detection Dashboard", fontsize=13, fontweight="bold")
    
    quarters = log_df["window"].tolist()
    colors = ["#CC0000" if d else "#009E73" for d in log_df["overall_drift"]]
    
    # Panel A: AUC over time
    ax = axes[0, 0]
    ax.plot(quarters, log_df["ref_auc"], color="#0072B2", linewidth=2, linestyle="--", label="Reference AUC", marker="o")
    ax.plot(quarters, log_df["cur_auc"], color="#CC79A7", linewidth=2, label="Current AUC",   marker="s")
    ax.axhline(log_df["ref_auc"].iloc[0] - DRIFT_THRESHOLDS["auc_degradation"],
               color="red", linestyle=":", alpha=0.7, label="Recalibration threshold")
    ax.set_title("A. Model AUC Over Time"); ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=8); ax.set_ylim(0.5, 1.05)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    
    # Panel B: Number of drifted features
    ax = axes[0, 1]
    bars = ax.bar(quarters, log_df["n_drifted"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(2, color="orange", linestyle="--", alpha=0.7, label="Trigger threshold (2 features)")
    ax.set_title("B. Drifted Features Per Window"); ax.set_ylabel("Number of features with drift")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    
    # Panel C: AUC drop
    ax = axes[1, 0]
    bar_colors = ["#CC0000" if d > DRIFT_THRESHOLDS["auc_degradation"] else "#009E73"
                  for d in log_df["auc_drop"]]
    ax.bar(quarters, log_df["auc_drop"], color=bar_colors, edgecolor="white")
    ax.axhline(DRIFT_THRESHOLDS["auc_degradation"], color="red", linestyle="--",
               alpha=0.7, label=f"Threshold: {DRIFT_THRESHOLDS['auc_degradation']}")
    ax.set_title("C. AUC Degradation from Baseline"); ax.set_ylabel("AUC Drop")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    
    # Panel D: Interruption rate over time
    ax = axes[1, 1]
    int_rates = [w["interrupted"].mean()*100 for w in windows]
    ax.plot(range(1, len(int_rates)+1), int_rates, color="#E69F00", linewidth=2.5, marker="D")
    ax.axhline(int_rates[0], color="grey", linestyle="--", alpha=0.5, label="Baseline rate")
    ax.set_title("D. Interruption Rate Drift Over Time")
    ax.set_ylabel("Interruption Rate (%)"); ax.set_xlabel("Window")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] Drift dashboard saved: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════════════════════════════

def run_drift_detection(patient_path="/mnt/user-data/outputs/smartdaas_synthetic_patient_table.csv"):
    print("="*70)
    print("SmartDaaS v2 — Drift Detection Module")
    print("Population shift monitoring and recalibration triggers")
    print("="*70)
    
    df = pd.read_csv(patient_path)
    print(f"\nLoaded {len(df):,} patients")
    
    print("\n[Simulating temporal windows (quarterly deployment)]")
    windows = simulate_temporal_windows(df, n_windows=6)
    
    log_df = run_drift_monitoring(windows)
    
    print(f"\n[Summary]")
    print(f"  Windows with drift detected: {log_df['overall_drift'].sum()} of {len(log_df)}")
    print(f"  Windows requiring recalibration: {(log_df['action'].str.contains('RECALIBRATE')).sum()} of {len(log_df)}")
    print(f"  Max AUC degradation: {log_df['auc_drop'].max():.4f}")
    print(f"  First drift detected at: {log_df[log_df['overall_drift']==True]['window'].iloc[0] if log_df['overall_drift'].any() else 'None'}")
    
    plot_drift_dashboard(log_df, windows)
    
    log_df.to_csv(f"{output_dir}/smartdaas_drift_log.csv", index=False)
    print(f"[Output] Drift log saved: smartdaas_drift_log.csv")
    
    print("\n" + "="*70)
    print("Drift detection complete.")
    print("SmartDaaS now monitors for:")
    print("  PSI > 0.2       → Population Stability Index (industry standard)")
    print("  KS p < 0.05     → Kolmogorov-Smirnov distribution test")
    print("  JS div > 0.1    → Jensen-Shannon divergence")
    print("  AUC drop > 5pp  → Model performance degradation")
    print("Next: Facility embeddings → Causal/uplift modeling")
    print("="*70)
    
    return log_df, windows

if __name__ == "__main__":
    log_df, windows = run_drift_detection()

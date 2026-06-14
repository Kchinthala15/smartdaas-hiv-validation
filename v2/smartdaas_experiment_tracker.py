"""
smartdaas_experiment_tracker.py — SmartDaaS v2 Experiment Tracker
Lightweight structured logging for reproducibility before MLflow/W&B
Author: Lakshmi Kalyani Chinthala, SmartDaaS LLC
"""
import os
import json
import hashlib
import datetime
import platform
import numpy as np
import warnings
warnings.filterwarnings("ignore")

TRACKER_FILE = "smartdaas_experiment_log.jsonl"


def get_run_id():
    ts = datetime.datetime.utcnow().isoformat()
    return hashlib.md5(ts.encode()).hexdigest()[:8]


def log_experiment(
    module: str,
    config: dict,
    metrics: dict,
    notes: str = "",
    data_source: str = "synthetic",
    output_dir: str = "/mnt/user-data/outputs",
):
    """
    Log a single experiment run to a structured JSONL file.
    One line per run — append-only, human-readable, git-friendly.

    Args:
        module:      Which module was run (e.g. "survival_model_v1")
        config:      Hyperparameters and settings used
        metrics:     Performance metrics (AUC, C-index, IBS, etc.)
        notes:       Free-text notes about this run
        data_source: "synthetic" | "apin_pilot" | "ampath_pilot" | "phia"
        output_dir:  Where to save the log file
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, TRACKER_FILE)

    run = {
        "run_id"      : get_run_id(),
        "timestamp"   : datetime.datetime.utcnow().isoformat(),
        "module"      : module,
        "data_source" : data_source,
        "platform"    : platform.python_version(),
        "config"      : config,
        "metrics"     : metrics,
        "notes"       : notes,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(run) + "\n")

    print(f"[Tracker] Run logged: {run['run_id']} → {log_path}")
    return run["run_id"]


def load_experiments(output_dir: str = "/mnt/user-data/outputs") -> list:
    """Load all logged experiments as a list of dicts."""
    log_path = os.path.join(output_dir, TRACKER_FILE)
    if not os.path.exists(log_path):
        print("[Tracker] No experiment log found.")
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise_experiments(output_dir: str = "/mnt/user-data/outputs"):
    """Print a clean summary table of all logged experiments."""
    runs = load_experiments(output_dir)
    if not runs:
        return

    print("\n" + "="*80)
    print("SmartDaaS v2 — Experiment Log")
    print("="*80)
    print(f"  {'Run ID':<10} {'Timestamp':<22} {'Module':<30} {'Data':<12} {'Key metric'}")
    print(f"  {'-'*76}")

    for r in runs:
        ts = r["timestamp"][:19].replace("T"," ")
        module = r["module"][:28]
        data = r["data_source"][:10]
        # Show first metric as key metric
        metrics = r.get("metrics", {})
        if metrics:
            k, v = next(iter(metrics.items()))
            key_metric = f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        else:
            key_metric = "—"
        print(f"  {r['run_id']:<10} {ts:<22} {module:<30} {data:<12} {key_metric}")

    print(f"\n  Total runs: {len(runs)}")
    print("="*80)


def compare_runs(metric: str = "auc", output_dir: str = "/mnt/user-data/outputs"):
    """Compare runs by a specific metric."""
    runs = load_experiments(output_dir)
    if not runs:
        return

    scored = []
    for r in runs:
        val = r.get("metrics", {}).get(metric)
        if val is not None:
            scored.append((r["run_id"], r["module"], r["data_source"], float(val)))

    if not scored:
        print(f"[Tracker] No runs found with metric: {metric}")
        return

    scored.sort(key=lambda x: x[3], reverse=True)
    print(f"\n[Tracker] Runs ranked by {metric}:")
    print(f"  {'Rank':<6} {'Run ID':<10} {'Module':<30} {'Data':<12} {metric}")
    print(f"  {'-'*64}")
    for i, (rid, mod, data, val) in enumerate(scored, 1):
        print(f"  {i:<6} {rid:<10} {mod[:28]:<30} {data:<12} {val:.4f}")


def demo_logging(output_dir: str = "/mnt/user-data/outputs"):
    """Log example runs from all 7 modules to demonstrate the tracker."""
    print("="*70)
    print("SmartDaaS v2 — Experiment Tracker Demo")
    print("Logging representative runs from all 7 modules...")
    print("="*70)

    # Module 1: Synthetic data
    log_experiment(
        module="synthetic_generator_v1",
        config={"n_patients": 5000, "n_facilities": 40, "seed": 42,
                "max_follow_up_days": 1095, "interruption_base_prob": 0.004},
        metrics={"n_events": 148782, "interruption_rate": 0.330,
                 "missed_visit_rate": 0.198, "median_time_to_event_days": 450},
        notes="Baseline synthetic dataset. Architecture-grade only — not epidemiologically calibrated.",
        data_source="synthetic",
        output_dir=output_dir,
    )

    # Module 2: Survival RSF v1
    log_experiment(
        module="survival_rsf_v1_features",
        config={"model": "RandomSurvivalForest", "n_estimators": 200,
                "features": "v1_snapshot_15", "min_samples_leaf": 5},
        metrics={"c_index": 0.9496, "integrated_brier_score": 0.0473},
        notes="V1 feature set (15 snapshot features). Equivalent to SmartDaaS v1 architecture.",
        data_source="synthetic",
        output_dir=output_dir,
    )

    # Module 2: Survival RSF v2
    log_experiment(
        module="survival_rsf_v2_longitudinal",
        config={"model": "RandomSurvivalForest", "n_estimators": 200,
                "features": "v2_longitudinal_25", "min_samples_leaf": 5},
        metrics={"c_index": 0.9887, "integrated_brier_score": 0.0060,
                 "c_index_improvement_vs_v1": 0.0391},
        notes="V2 longitudinal features. +3.9pp C-index gain quantifies value of sequence data over snapshot.",
        data_source="synthetic",
        output_dir=output_dir,
    )

    # Module 3: Transformer
    log_experiment(
        module="transformer_encoder_behrt",
        config={"d_model": 64, "n_heads": 4, "n_layers": 3, "d_ff": 128,
                "dropout": 0.1, "max_seq_len": 36, "n_epochs": 20,
                "lr": 1e-3, "batch_size": 64, "n_features": 14},
        metrics={"auc_roc": 0.9710, "auc_pr": 0.9573,
                 "embedding_dim": 64, "n_params": 103617},
        notes="BEHRT-style transformer. Explicit attention_mask added for real data safety.",
        data_source="synthetic",
        output_dir=output_dir,
    )

    # Module 4: Multi-task
    log_experiment(
        module="multitask_learning_4heads",
        config={"hidden_dims": [256, 128, 64], "shared_dim": 64,
                "n_tasks": 4, "dropout": 0.2, "n_epochs": 30, "lr": 1e-3},
        metrics={"mean_auc": 0.9319, "auc_interruption": 1.0,
                 "auc_viral_failure": 1.0, "auc_poor_adherence": 0.7287,
                 "auc_high_missed": 0.9989},
        notes="4-task shared encoder. Poor adherence AUC lower — expected, more volatile visit-to-visit.",
        data_source="synthetic",
        output_dir=output_dir,
    )

    # Module 5: Drift detection
    log_experiment(
        module="drift_detection_v1",
        config={"psi_threshold": 0.2, "ks_pvalue_threshold": 0.05,
                "js_divergence_threshold": 0.1, "auc_degradation_threshold": 0.05,
                "n_windows": 6, "monitor_features": 10},
        metrics={"windows_with_drift": 2, "max_auc_degradation": 0.24,
                 "first_drift_window": 5},
        notes="Drift introduced from window 4 onward. Detected at window 5. Thresholds are operational defaults — not clinically calibrated.",
        data_source="synthetic",
        output_dir=output_dir,
    )

    # Module 6: Facility embeddings
    log_experiment(
        module="facility_embeddings_hierarchical",
        config={"facility_emb_dim": 16, "patient_emb_dim": 32,
                "n_facilities": 40, "dropout": 0.2, "n_epochs": 25},
        metrics={"auc_roc": 0.8331, "n_facilities_embedded": 40,
                 "n_countries": 6, "embedding_dim": 16},
        notes="Patient+facility hierarchical model. AUC lower than patient-only — adds different information. Overfitting risk: validate on unseen facilities with real APIN data.",
        data_source="synthetic",
        output_dir=output_dir,
    )

    # Module 7: Causal uplift
    log_experiment(
        module="causal_uplift_tlearner",
        config={"model": "T-Learner", "base_model": "RandomForest",
                "n_estimators": 100, "intervention_effect_size": 0.25},
        metrics={"ite_correlation": 0.73, "uplift_ratio_top20pct": 2.15,
                 "mean_benefit_top20": 0.2185, "mean_benefit_overall": 0.1019},
        notes="T-Learner best performer. Causal Forest DML needs larger sample — will improve with APIN data. benefit = -CATE (sign corrected).",
        data_source="synthetic",
        output_dir=output_dir,
    )

    print("\n[Demo complete] All 7 modules logged.")
    summarise_experiments(output_dir)
    compare_runs("auc_roc", output_dir)


if __name__ == "__main__":
    demo_logging()

import os
"""
smartdaas_causal_uplift.py — SmartDaaS v2 Causal / Uplift Modeling Module
"Who benefits MOST from intervention?" — not just "who is at risk?"
Author: Lakshmi Kalyani Chinthala, SmartDaaS LLC
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    print("[WARNING] econml not installed. Causal Forest DML unavailable.")
    print("         Install with: pip install econml")
    print("         Falling back to S-Learner and T-Learner only.")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

FEATURES = [
    "age_at_art_start", "sex_female", "cd4_at_art_start", "most_recent_cd4",
    "cd4_improvement", "who_stage_start", "weight_at_start", "days_to_art",
    "missed_visit_rate", "mean_refill_gap_days", "n_regimen_changes",
    "viral_suppressed_last", "n_poor_adherence_visits", "had_interruption_history",
    "opp_infection_any", "side_effects_any", "tb_any", "stage_worsened",
]


# ════════════════════════════════════════════════════════════════════════════
# 1. SIMULATE TREATMENT ASSIGNMENT AND OUTCOMES
# ════════════════════════════════════════════════════════════════════════════

def simulate_intervention(patient_df, intervention_effect_size=0.25):
    """
    Simulate a real-world scenario where ~40% of high-risk patients
    received an enhanced retention intervention (phone call, home visit,
    peer support) and the rest received standard care.

    In real data: treatment = whether patient received intervention
    In synthetic: we simulate it with known effect sizes so we can
    validate the causal model's recovery of true treatment effects.

    Key insight: treatment effect is HETEROGENEOUS —
    some patients benefit a lot, some barely at all.
    High-risk patients benefit more. Low-risk patients benefit less.
    This is what uplift modeling captures.
    """
    df = patient_df.copy()
    n = len(df)

    # Treatment assignment (observational — not random)
    # High-risk patients more likely to receive intervention
    # (mimics real programme targeting)
    treatment_prob = 0.2 + df["risk_latent"] * 0.4
    treatment_prob = treatment_prob.clip(0.1, 0.7)
    df["treatment"] = np.random.binomial(1, treatment_prob, n)

    # Potential outcomes (counterfactual framework)
    # True individual treatment effect (ITE) — heterogeneous
    # High risk patients benefit more from intervention
    true_ite = (
        intervention_effect_size * df["risk_latent"] +
        0.05 * (df["cd4_at_art_start"] < 200).astype(float) +
        0.03 * df["had_interruption_history"] +
        np.random.normal(0, 0.02, n)
    ).clip(0, 0.6)

    df["true_ite"] = true_ite  # ground truth for validation

    # Observed outcome: interruption probability reduced by ITE if treated
    base_prob = (
        0.1 + df["risk_latent"] * 0.4 +
        np.random.normal(0, 0.05, n)
    ).clip(0.02, 0.85)

    outcome_prob = np.where(
        df["treatment"] == 1,
        (base_prob - true_ite * df["treatment"]).clip(0.01, 0.95),
        base_prob
    )
    df["outcome"] = np.random.binomial(1, outcome_prob, n)  # 1 = interrupted

    print(f"[Simulation] Treatment assignment:")
    print(f"  Treated:   {df['treatment'].sum():,} ({df['treatment'].mean()*100:.1f}%)")
    print(f"  Untreated: {(df['treatment']==0).sum():,} ({(df['treatment']==0).mean()*100:.1f}%)")
    print(f"  Interruption rate (treated):   {df[df['treatment']==1]['outcome'].mean()*100:.1f}%")
    print(f"  Interruption rate (untreated): {df[df['treatment']==0]['outcome'].mean()*100:.1f}%")
    print(f"  Mean true ITE: {true_ite.mean():.4f} (absolute risk reduction)")
    print(f"  ITE range: [{true_ite.min():.4f}, {true_ite.max():.4f}]")

    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. S-LEARNER UPLIFT MODEL (BASELINE)
# ════════════════════════════════════════════════════════════════════════════

def fit_s_learner(X_train, T_train, Y_train, X_test):
    """
    S-Learner: Single model approach.
    Train one model with treatment as a feature.
    Estimate ITE by predicting with T=1 and T=0, taking the difference.
    Simple but can suffer from regularisation bias.
    """
    X_with_T_train = np.column_stack([X_train, T_train])
    X_test_t1 = np.column_stack([X_test, np.ones(len(X_test))])
    X_test_t0 = np.column_stack([X_test, np.zeros(len(X_test))])

    model = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_with_T_train, Y_train)

    prob_t1 = model.predict_proba(X_test_t1)[:, 1]
    prob_t0 = model.predict_proba(X_test_t0)[:, 1]
    uplift = prob_t0 - prob_t1  # positive = intervention reduces interruption risk (benefit)

    return uplift, model


# ════════════════════════════════════════════════════════════════════════════
# 3. T-LEARNER UPLIFT MODEL
# ════════════════════════════════════════════════════════════════════════════

def fit_t_learner(X_train, T_train, Y_train, X_test):
    """
    T-Learner: Two separate models — one for treated, one for control.
    Estimate ITE as difference in predictions.
    Better than S-Learner for heterogeneous effects.
    """
    mask_t = T_train == 1
    mask_c = T_train == 0

    model_t = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model_c = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)

    model_t.fit(X_train[mask_t], Y_train[mask_t])
    model_c.fit(X_train[mask_c], Y_train[mask_c])

    prob_t1 = model_t.predict_proba(X_test)[:, 1]
    prob_t0 = model_c.predict_proba(X_test)[:, 1]
    uplift = prob_t0 - prob_t1

    return uplift, model_t, model_c


# ════════════════════════════════════════════════════════════════════════════
# 4. CAUSAL FOREST (ECONML) — PRIMARY MODEL
# ════════════════════════════════════════════════════════════════════════════

def fit_causal_forest(X_train, T_train, Y_train, X_test):
    """
    Causal Forest DML (Double Machine Learning).
    Most rigorous causal estimator — handles confounding,
    estimates heterogeneous treatment effects (CATE),
    and provides confidence intervals.

    This is the frontier of causal ML in healthcare.
    Answers: "What is the INDIVIDUAL causal effect of intervention
    on this specific patient's probability of interruption?"
    """
    if not ECONML_AVAILABLE:
        print("  [SKIP] econml not available — skipping Causal Forest")
        n = len(X_test)
        return np.zeros(n), np.zeros(n), np.zeros(n), None

    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        model_t=RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        n_estimators=200,
        discrete_treatment=True,
        random_state=SEED,
        n_jobs=-1,
    )
    cf.fit(Y_train.astype(float), T_train, X=X_train)

    # NOTE: outcome is 1=interruption (bad). Treatment reduces risk.
    # CATE = E[Y(1) - Y(0)] = treated_risk - untreated_risk → negative if helpful.
    # benefit = -cate (positive = good, use for prioritization)
    cate_raw = cf.effect(X_test)
    cate = -cate_raw  # flip sign: positive benefit = intervention helps
    cate_lb_raw, cate_ub_raw = cf.effect_interval(X_test, alpha=0.05)
    cate_lb = -cate_ub_raw  # flip bounds too
    cate_ub = -cate_lb_raw

    return cate, cate_lb, cate_ub, cf


# ════════════════════════════════════════════════════════════════════════════
# 5. UPLIFT EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_uplift(uplift_scores, true_ite, T_test, Y_test, model_name):
    """
    Evaluate uplift model using:
    1. Correlation with true ITE (ground truth — only available in simulation)
    2. Qini coefficient (real-world uplift metric)
    3. Uplift at top-K (how much benefit if we intervene on top K% of patients)
    """
    # Correlation with true ITE
    corr = np.corrcoef(uplift_scores, true_ite)[0, 1]

    # Uplift at top 20% and top 40%
    n = len(uplift_scores)
    top20_idx = np.argsort(uplift_scores)[::-1][:int(n*0.20)]
    top40_idx = np.argsort(uplift_scores)[::-1][:int(n*0.40)]

    # Among top-K predicted to benefit most — what's the actual benefit?
    treated_rate_top20 = T_test[top20_idx].mean()
    outcome_top20 = Y_test[top20_idx].mean()
    mean_ite_top20 = true_ite[top20_idx].mean()

    print(f"  {model_name}:")
    print(f"    Correlation with true ITE: {corr:.4f}")
    print(f"    Mean predicted uplift (top 20%): {uplift_scores[top20_idx].mean():.4f}")
    print(f"    Mean TRUE uplift (top 20%):      {mean_ite_top20:.4f}")
    print(f"    Mean TRUE uplift (top 40%):      {true_ite[top40_idx].mean():.4f}")
    print(f"    Mean TRUE uplift (all):          {true_ite.mean():.4f}")

    return corr, mean_ite_top20


# ════════════════════════════════════════════════════════════════════════════
# 6. INTERVENTION ALLOCATION TABLE
# ════════════════════════════════════════════════════════════════════════════

def build_intervention_table(X_test_df, uplift_causal, uplift_tlearner, risk_scores, y_test):
    """
    The operational output — for each patient, show:
    - Their interruption risk (from standard risk model)
    - Their predicted benefit from intervention (uplift)
    - Their recommended priority tier
    
    This is what field teams actually use.
    SmartDaaS becomes: "intervene here, not there"
    """
    df = X_test_df.copy()
    df["interruption_risk"]    = risk_scores
    df["uplift_causal"]        = uplift_causal
    df["uplift_tlearner"]      = uplift_tlearner
    df["actual_interrupted"]   = y_test

    # Priority tiers based on BOTH risk AND uplift
    def assign_tier(row):
        risk = row["interruption_risk"]
        uplift = row["uplift_causal"]
        if risk > 0.6 and uplift > 0.15:
            return "URGENT — High risk, high benefit"
        elif risk > 0.4 and uplift > 0.10:
            return "HIGH — Moderate risk, meaningful benefit"
        elif risk > 0.6 and uplift <= 0.10:
            return "MONITOR — High risk, low expected benefit"
        elif uplift > 0.20:
            return "OPPORTUNITY — Lower risk, high intervention benefit"
        else:
            return "STANDARD CARE"

    df["intervention_tier"] = df.apply(assign_tier, axis=1)

    print(f"\n[Intervention Allocation Table]")
    print(f"  {'Tier':<45} {'N':>6} {'%':>6} {'Actual interrupt rate':>22}")
    print(f"  {'-'*82}")
    for tier in df["intervention_tier"].unique():
        mask = df["intervention_tier"] == tier
        n = mask.sum()
        pct = n/len(df)*100
        actual = df[mask]["actual_interrupted"].mean()*100
        print(f"  {tier:<45} {n:>6} {pct:>5.1f}% {actual:>20.1f}%")

    return df


# ════════════════════════════════════════════════════════════════════════════
# 7. PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_uplift(uplift_s, uplift_t, cate, true_ite, risk_scores,
    output_path=f"{output_dir}/smartdaas_causal_uplift.png"):

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("SmartDaaS v2 — Causal / Uplift Modeling\n'Who benefits MOST from intervention?'",
                 fontsize=13, fontweight="bold")

    # Panel A: True ITE distribution
    ax = axes[0,0]
    ax.hist(true_ite, bins=40, color="#0072B2", alpha=0.7, edgecolor="white")
    ax.axvline(true_ite.mean(), color="#CC79A7", linewidth=2, linestyle="--",
               label=f"Mean ITE: {true_ite.mean():.3f}")
    ax.set_xlabel("True Individual Treatment Effect (absolute risk reduction)")
    ax.set_ylabel("Number of patients")
    ax.set_title("A. True ITE Distribution (ground truth)")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel B: Causal Forest CATE vs True ITE
    ax = axes[0,1]
    ax.scatter(true_ite, cate, alpha=0.3, s=8, color="#009E73")
    lims = [min(true_ite.min(), cate.min()), max(true_ite.max(), cate.max())]
    ax.plot(lims, lims, "r--", alpha=0.7, linewidth=1.5, label="Perfect recovery")
    corr = np.corrcoef(true_ite, cate)[0,1]
    ax.set_xlabel("True ITE"); ax.set_ylabel("Predicted CATE (Causal Forest)")
    ax.set_title(f"B. Causal Forest vs True ITE (r={corr:.3f})")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel C: Uplift vs Risk — the key strategic quadrant
    ax = axes[1,0]
    sc = ax.scatter(risk_scores, cate, c=true_ite, cmap="RdYlGn",
                    alpha=0.5, s=10, vmin=0, vmax=0.4)
    plt.colorbar(sc, ax=ax, label="True ITE")
    ax.axvline(0.4, color="red",   linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(0.1, color="blue",  linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Interruption Risk Score")
    ax.set_ylabel("Predicted Treatment Benefit (CATE)")
    ax.set_title("C. Risk vs Benefit — Strategic Quadrant\n(top-right = intervene urgently)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.65, 0.25, "URGENT\nIntervene", fontsize=8, color="darkred",
            ha="center", style="italic")
    ax.text(0.15, 0.25, "OPPORTUNITY\nLow risk,\nhigh benefit", fontsize=7,
            color="darkblue", ha="center", style="italic")

    # Panel D: Cumulative uplift curves (Qini-style)
    ax = axes[1,1]
    n = len(true_ite)
    random_line = np.linspace(0, true_ite.sum(), n)
    for scores, label, color in [
        (cate,     "Causal Forest", "#009E73"),
        (uplift_t, "T-Learner",     "#0072B2"),
        (uplift_s, "S-Learner",     "#CC79A7"),
    ]:
        order = np.argsort(scores)[::-1]
        cumulative = true_ite[order].cumsum()
        ax.plot(range(n), cumulative, color=color, linewidth=2, label=label)
    ax.plot(range(n), random_line, color="grey", linestyle="--", linewidth=1.5, label="Random")
    ax.set_xlabel("Patients targeted (ranked by predicted benefit)")
    ax.set_ylabel("Cumulative true benefit (risk reduction)")
    ax.set_title("D. Cumulative Uplift Curves\n(above random = model adds value)")
    ax.legend(fontsize=9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] Saved: {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ════════════════════════════════════════════════════════════════════════════

def run_causal_uplift(patient_path="/mnt/user-data/outputs/smartdaas_synthetic_patient_table.csv", output_dir="/mnt/user-data/outputs"):
    print("="*70)
    print("SmartDaaS v2 — Causal / Uplift Modeling Module")
    print("'Who benefits MOST from intervention?' — not just 'who is at risk?'")
    print("="*70)

    df = pd.read_csv(patient_path)

    # Simulate intervention
    print("\n[Step 1] Simulating intervention assignment and outcomes...")
    df_sim = simulate_intervention(df)

    # Prepare features
    avail = [f for f in FEATURES if f in df_sim.columns]
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imp.fit_transform(df_sim[avail].fillna(0))).astype(np.float32)
    T = df_sim["treatment"].values.astype(np.float32)
    Y = df_sim["outcome"].values.astype(np.float32)
    true_ite = df_sim["true_ite"].values

    # Split
    idx = np.arange(len(Y))
    idx_tr, idx_val = train_test_split(idx, test_size=0.3, random_state=SEED,
                                        stratify=T.astype(int))
    X_tr, X_val = X[idx_tr], X[idx_val]
    T_tr, T_val = T[idx_tr], T[idx_val]
    Y_tr, Y_val = Y[idx_tr], Y[idx_val]
    true_ite_val = true_ite[idx_val]

    # Risk scores (standard model — for comparison)
    risk_model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    risk_model.fit(X_tr, Y_tr)
    risk_scores = risk_model.predict_proba(X_val)[:,1]

    # Uplift models
    print("\n[Step 2] Fitting uplift models...")
    print("  S-Learner...")
    uplift_s, _ = fit_s_learner(X_tr, T_tr, Y_tr, X_val)

    print("  T-Learner...")
    uplift_t, _, _ = fit_t_learner(X_tr, T_tr, Y_tr, X_val)

    print("  Causal Forest DML...")
    cate, cate_lb, cate_ub, cf = fit_causal_forest(X_tr, T_tr, Y_tr, X_val)

    # Evaluate
    print("\n[Step 3] Evaluation vs ground truth ITE:")
    corr_s, ite_top20_s = evaluate_uplift(uplift_s, true_ite_val, T_val, Y_val, "S-Learner")
    corr_t, ite_top20_t = evaluate_uplift(uplift_t, true_ite_val, T_val, Y_val, "T-Learner")
    corr_cf, ite_top20_cf = evaluate_uplift(cate,     true_ite_val, T_val, Y_val, "Causal Forest")

    print(f"\n  Model comparison:")
    print(f"  {'Model':<20} {'ITE Correlation':<20} {'Benefit Top 20%'}")
    print(f"  {'-'*55}")
    print(f"  {'S-Learner':<20} {corr_s:<20.4f} {ite_top20_s:.4f}")
    print(f"  {'T-Learner':<20} {corr_t:<20.4f} {ite_top20_t:.4f}")
    print(f"  {'Causal Forest':<20} {corr_cf:<20.4f} {ite_top20_cf:.4f}")

    # Intervention allocation
    print("\n[Step 4] Building intervention allocation table...")
    X_val_df = pd.DataFrame(X_val, columns=[f"feat_{i}" for i in range(X_val.shape[1])])
    intervention_df = build_intervention_table(X_val_df, cate, uplift_t, risk_scores, Y_val)

    # Plot
    plot_uplift(uplift_s, uplift_t, cate, true_ite_val, risk_scores)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    intervention_df.to_csv(f"{output_dir}/smartdaas_intervention_allocation.csv", index=False)
    print(f"\n[Output] Intervention allocation saved")

    print("\n" + "="*70)
    print("Causal / Uplift modeling complete.")
    print(f"  Causal Forest ITE correlation: {corr_cf:.4f}")
    print(f"  Benefit captured in top 20%:   {ite_top20_cf:.4f} vs {true_ite_val.mean():.4f} overall")
    print(f"  Uplift ratio:                  {ite_top20_cf/true_ite_val.mean():.2f}x")
    print("\nSmartDaaS v2 architecture COMPLETE:")
    print("  ✓ Synthetic longitudinal data generator")
    print("  ✓ Random Survival Forest (time-to-event)")
    print("  ✓ Transformer sequence encoder (BEHRT-style)")
    print("  ✓ Multi-task learning (4 simultaneous outcomes)")
    print("  ✓ Drift detection (PSI + KS + JS + AUC monitoring)")
    print("  ✓ Facility embeddings (hierarchical programme intelligence)")
    print("  ✓ Causal / uplift modeling (intervention optimization)")
    print("\nReady for APIN pilot data.")
    print("="*70)

    return intervention_df, cate, corr_cf

if __name__ == "__main__":
    run_causal_uplift()

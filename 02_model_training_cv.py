"""
02_model_training_cv.py
10-fold stratified cross-validation for all four classifiers.
Primary model: 14 features (ART interruption EXCLUDED).
Secondary model: 15 features (ART interruption INCLUDED) — for comparison.

Outputs:
    results/cv_results.pkl       — CV metrics for all models
    results/model_performance.csv — Summary table
"""

import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              brier_score_loss)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from utils import FEATURES, SEED

np.random.seed(SEED)


def build_models():
    """Build all four classifier pipelines."""
    return {
        'Logistic Regression': Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=0.5, random_state=SEED))
        ]),
        'Random Forest': Pipeline([
            ('sc', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=8,
                min_samples_leaf=10, n_jobs=-1, random_state=SEED))
        ]),
        'Gradient Boosting': Pipeline([
            ('sc', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05,
                max_depth=4, subsample=0.8, random_state=SEED))
        ]),
        'Neural Network': Pipeline([
            ('sc', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(64, 32), activation='relu',
                alpha=0.001, max_iter=300,
                early_stopping=True, random_state=SEED))
        ]),
    }


def run_cv(X, y, models, cv_folds=10):
    """Run stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    scoring = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
    results = {}

    for name, model in models.items():
        cv = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        results[name] = {
            'Accuracy':  (cv['test_accuracy'].mean(),  cv['test_accuracy'].std()),
            'AUC-ROC':   (cv['test_roc_auc'].mean(),   cv['test_roc_auc'].std()),
            'F1-Score':  (cv['test_f1'].mean(),         cv['test_f1'].std()),
            'Precision': (cv['test_precision'].mean(),  cv['test_precision'].std()),
            'Recall':    (cv['test_recall'].mean(),     cv['test_recall'].std()),
        }
        r = results[name]
        print(f"  {name:22s}: "
              f"AUC={r['AUC-ROC'][0]:.4f}±{r['AUC-ROC'][1]:.4f}  "
              f"Acc={r['Accuracy'][0]:.4f}  "
              f"F1={r['F1-Score'][0]:.4f}")
    return results


def holdout_evaluation(X, y, best_model_name, models):
    """Evaluate best model on 20% stratified hold-out test set."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = models[best_model_name]
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_prob  = model.predict_proba(X_te)[:, 1]

    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    auc  = roc_auc_score(y_te, y_prob)
    brier = brier_score_loss(y_te, y_prob)

    # Bootstrap CIs
    boot_aucs, boot_sens, boot_specs = [], [], []
    for _ in range(1000):
        idx = np.random.choice(len(y_te), len(y_te), replace=True)
        yt_b, yp_b, ybin_b = y_te[idx], y_prob[idx], y_pred[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt_b, yp_b))
        _tp = ((ybin_b==1)&(yt_b==1)).sum()
        _fn = ((ybin_b==0)&(yt_b==1)).sum()
        _tn = ((ybin_b==0)&(yt_b==0)).sum()
        _fp = ((ybin_b==1)&(yt_b==0)).sum()
        if (_tp+_fn) > 0: boot_sens.append(_tp/(_tp+_fn))
        if (_tn+_fp) > 0: boot_specs.append(_tn/(_tn+_fp))

    holdout = {
        'auc': auc, 'sensitivity': sens, 'specificity': spec,
        'brier': brier, 'cm': cm,
        'auc_ci':  (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5)),
        'sens_ci': (np.percentile(boot_sens, 2.5), np.percentile(boot_sens, 97.5)),
        'spec_ci': (np.percentile(boot_specs, 2.5), np.percentile(boot_specs, 97.5)),
        'y_te': y_te, 'y_prob': y_prob,
        'X_te': X_te, 'y_tr': y_tr, 'X_tr': X_tr,
    }

    print(f"\n  Hold-out ({best_model_name}):")
    print(f"    AUC  : {auc:.4f} (95%CI: {holdout['auc_ci'][0]:.4f}-{holdout['auc_ci'][1]:.4f})")
    print(f"    Sens : {sens:.4f} (95%CI: {holdout['sens_ci'][0]:.4f}-{holdout['sens_ci'][1]:.4f})")
    print(f"    Spec : {spec:.4f} (95%CI: {holdout['spec_ci'][0]:.4f}-{holdout['spec_ci'][1]:.4f})")
    print(f"    Brier: {brier:.4f}")

    return holdout, model


def delong_pairwise(results, best_name):
    """Pairwise AUC significance tests (DeLong approximation)."""
    print("\n  Pairwise AUC z-tests (DeLong):")
    best_auc, best_std = results[best_name]['AUC-ROC']
    for name, r in results.items():
        if name == best_name:
            continue
        auc, std = r['AUC-ROC']
        se = np.sqrt(best_std**2 + std**2)
        z  = (best_auc - auc) / se if se > 0 else 0
        p  = 2 * (1 - stats.norm.cdf(abs(z)))
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"    {best_name} vs {name:22s}: z={z:.2f}, p={p:.4f} {sig}")


def main():
    print("=" * 60)
    print("Step 2: Model Training & Cross-Validation")
    print("=" * 60)

    with open('results/preprocessed_data.pkl', 'rb') as f:
        d = pickle.load(f)

    X_bal = d['X_bal']
    y_bal = d['y_bal']

    models = build_models()

    # ── PRIMARY MODEL (14 features, no ART interruption) ──
    print("\n[PRIMARY MODEL — 14 features, ART interruption excluded]")
    results_primary = run_cv(X_bal, y_bal, models)

    best_name = max(results_primary, key=lambda m: results_primary[m]['AUC-ROC'][0])
    print(f"\n  Best model: {best_name}")
    delong_pairwise(results_primary, best_name)
    holdout_primary, best_model = holdout_evaluation(X_bal, y_bal, best_name, models)

    # ── SECONDARY MODEL (15 features, with ART interruption) ──
    print("\n[SECONDARY MODEL — 15 features, ART interruption included]")
    features_full = FEATURES + ['had_interruption']
    from sklearn.impute import SimpleImputer
    from imblearn.over_sampling import SMOTE
    df_m = d['df_m'].copy()
    if 'had_interruption' in df_m.columns:
        X_full_raw = d['imputer'].fit_transform(df_m[features_full])
        y_full = df_m['target'].values.astype(int)
        sm = SMOTE(random_state=SEED, k_neighbors=5)
        X_full_bal, y_full_bal = sm.fit_resample(X_full_raw, y_full)
        results_secondary = run_cv(X_full_bal, y_full_bal, models)
        print("  (Full model with ART interruption — presented as secondary analysis)")
    else:
        results_secondary = None
        print("  (had_interruption not available — skipping secondary)")

    # ── SAVE ──────────────────────────────────────────────
    save = {
        'results_primary':   results_primary,
        'results_secondary': results_secondary,
        'best_model_name':   best_name,
        'holdout':           holdout_primary,
        'best_model':        best_model,
    }
    with open('results/cv_results.pkl', 'wb') as f:
        pickle.dump(save, f)

    # Summary CSV
    rows = []
    for name, r in results_primary.items():
        rows.append({
            'Model': name,
            'Accuracy':  f"{r['Accuracy'][0]:.4f}±{r['Accuracy'][1]:.4f}",
            'AUC-ROC':   f"{r['AUC-ROC'][0]:.4f}±{r['AUC-ROC'][1]:.4f}",
            'F1-Score':  f"{r['F1-Score'][0]:.4f}±{r['F1-Score'][1]:.4f}",
            'Precision': f"{r['Precision'][0]:.4f}±{r['Precision'][1]:.4f}",
            'Recall':    f"{r['Recall'][0]:.4f}±{r['Recall'][1]:.4f}",
        })
    pd.DataFrame(rows).to_csv('results/model_performance.csv', index=False)

    print("\nSaved: results/cv_results.pkl")
    print("Saved: results/model_performance.csv")
    print("Step 2 complete.")


if __name__ == '__main__':
    main()

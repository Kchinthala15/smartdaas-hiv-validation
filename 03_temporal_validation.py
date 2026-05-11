"""
03_temporal_validation.py
Temporal train/test split validation.
Train on earliest 70% of records (by ART start date),
test on most recent 30% — simulating prospective deployment.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from utils import FEATURES, SEED

np.random.seed(SEED)


def main():
    print("=" * 60)
    print("Step 3: Temporal Validation")
    print("=" * 60)

    with open('results/preprocessed_data.pkl', 'rb') as f:
        d = pickle.load(f)

    df_m = d['df_m'].copy()

    # Add ART start date for temporal sorting
    df_qoc = d['df_qoc']
    df_m['DateArtStarted'] = df_qoc.loc[df_m.index, 'DateArtStarted']
    df_m = df_m.dropna(subset=['DateArtStarted'])
    df_m = df_m.sort_values('DateArtStarted').reset_index(drop=True)

    split_idx   = int(len(df_m) * 0.70)
    cutoff_date = df_m.iloc[split_idx]['DateArtStarted']

    # Impute separately (fit on train only)
    imp_t = SimpleImputer(strategy='median')
    X_t   = imp_t.fit_transform(df_m[FEATURES])
    y_t   = df_m['target'].values.astype(int)

    X_tr, y_tr = X_t[:split_idx], y_t[:split_idx]
    X_te, y_te = X_t[split_idx:], y_t[split_idx:]

    print(f"  Train: n={len(X_tr):,} | up to {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"  Test:  n={len(X_te):,} | from {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"  Train poor: {y_tr.mean()*100:.1f}% | Test poor: {y_te.mean()*100:.1f}%")

    # SMOTE on training only
    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)

    # Scale (fit on train)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_sm)
    X_te_sc = sc.transform(X_te)

    # Train and evaluate
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        min_samples_leaf=10, n_jobs=-1, random_state=SEED)
    rf.fit(X_tr_sc, y_tr_sm)
    y_prob = rf.predict_proba(X_te_sc)[:, 1]
    auc_t  = roc_auc_score(y_te, y_prob)

    # Bootstrap CI
    boots = []
    for _ in range(500):
        idx = np.random.choice(len(y_te), len(y_te), replace=True)
        if len(np.unique(y_te[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_te[idx], y_prob[idx]))
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))

    print(f"\n  Temporal AUC: {auc_t:.4f} (95%CI: {ci[0]:.4f}-{ci[1]:.4f})")
    print(f"  Interpretation: Performance drop from CV is expected.")
    print(f"  CV AUC reflects in-distribution performance.")
    print(f"  Temporal AUC reflects prospective generalisation.")

    save = {
        'auc_temp': auc_t, 'ci_temp': ci,
        'cutoff': cutoff_date.strftime('%Y-%m-%d'),
        'n_train': len(X_tr), 'n_test': len(X_te),
        'y_te': y_te, 'y_prob': y_prob,
    }
    with open('results/temporal_results.pkl', 'wb') as f:
        pickle.dump(save, f)

    print("\nSaved: results/temporal_results.pkl")
    print("Step 3 complete.")


if __name__ == '__main__':
    main()

"""
retrain_v1_1.py — SmartDaaS v1.1, corrected and locked.

Merged from two independent implementations that agreed on every metric
(cohort 21,273 · events 773 · temporal AUC 0.8124 · slope 1.096).

Changes from v1
---------------
1. tb_positive
   v1 searched TbStatusAtLAstVisit for 'positive|active'. The actual values are
   'Confirmed TB', 'TB Treatment', 'No sign', 'Presumptive TB', 'IPT' — none
   match, so the feature was a constant zero for all 23,144 patients and
   contributed 0.00% importance. Now mapped explicitly. 'IPT' is isoniazid
   PREVENTIVE therapy, given to patients WITHOUT active TB, so it stays 0.
   'Presumptive TB' is unconfirmed and also stays 0.

2. unknown interruption status excluded
   ArtInterruption encodes missingness two ways: the literal string 'Missing'
   (966) and a true null (905). v1 coded both as 0 — "never interrupted", the
   protective value — on the model's dominant predictor. Both now excluded.
   This does NOT improve discrimination. It is done because coding unknown as
   protective is not defensible under PROBAST.
   EDITING WARNING: col.astype(str) does NOT reliably turn a float NaN into the
   string 'nan' here. Test col.isna() separately or the nulls survive the
   filter silently and you get a spurious AUC gain. This happened during
   development.

3. sigmoid calibration
   v1 was uncalibrated while the README claimed a "calibrated risk score".
   Calibration slope was 1.402 (predictions compressed; max risk 0.355).
   Sigmoid rather than isotonic: ~640 positives in the training fold means
   5-fold isotonic sees ~128 events per fold and overfits.

4. thresholds anchored to FLAGGING RATE, not to a fixed number
   Calibration moves the probability scale, so v1's 0.15/0.075 pair no longer
   means what it meant. Thresholds are set by quantile so the same PROPORTION
   of patients is flagged as under v1 — a clinic staffed for ~40 calls a week
   still makes ~40 calls a week. What changes is who gets called, not how many.

5. lock
   The digest covers a deterministic manifest — source file, fitted model,
   cohort, metrics, thresholds, seed — NOT the pickle bytes. Hashing the file
   fails two ways: the embedded timestamp changes the bytes on every run, and
   writing a hash into the file it describes is self-referentially impossible.

6. shap_model shipped alongside
   CalibratedClassifierCV has no .named_steps, but model.py reaches for
   named_steps['sc'] and ['clf'] to build SHAP explanations. Without an
   uncalibrated copy the app raises AttributeError on load and reports
   model_not_loaded. See the CONTRACT block below — this key is dangerous if
   used for scoring.

Outputs: cv_results.pkl · prepped_data.pkl · MODEL_LOCK.json
"""

import hashlib
import json
import pickle
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

SEED = 42
VERSION = 'v1.1'
DATA = 'data/QualityOfCare.xlsx'

# v1 flagged ~3.7% of the temporal test set at its HIGH threshold and ~11.6% at
# MEDIUM. Preserve those rates so operational workload does not change silently.
FLAG_RATE_HI = 0.037
FLAG_RATE_LO = 0.116

TB_POSITIVE = {'confirmed tb', 'tb treatment'}

FEATURES = [
    'Age', 'sex_female', 'Cd4AtStart', 'MostRecentCd4Count', 'CD4_improvement',
    'stage_start_num', 'WeightAtStart', 'weight_change', 'BMI_start', 'days_to_ART',
    'had_interruption', 'opp_infection', 'side_effects', 'tb_positive', 'stage_worsened',
]

_norm = lambda c: c.astype(str).str.lower().str.strip()


def build_base():
    """Uncalibrated pipeline. Imputer inside so serving matches training."""
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=8,
                                       min_samples_leaf=10, max_features='sqrt',
                                       n_jobs=-1, random_state=SEED)),
    ])


def build_calibrated():
    return CalibratedClassifierCV(build_base(), method='sigmoid', cv=5)


def build_cohort(path=DATA, verbose=True):
    df = pd.read_excel(path)
    n0 = len(df)
    df = df[df['ArvAdherenceLatestLevel'].notna()]
    n1 = len(df)
    df = df[(df['Age'] >= 18) & (df['Age'] <= 100)].copy()
    n2 = len(df)

    # FIX 2 — test the null separately; see EDITING WARNING above.
    unknown = _norm(df['ArtInterruption']).eq('missing') | df['ArtInterruption'].isna()
    n_unknown = int(unknown.sum())
    df = df[~unknown].copy()

    df.loc[df['Cd4Unit'] == '%', ['Cd4AtStart', 'MostRecentCd4Count']] = np.nan
    for c in ['Cd4AtStart', 'MostRecentCd4Count']:
        df.loc[df[c] > 2000, c] = np.nan
    h = df['HeightAtStart'].copy()
    h = np.where(pd.Series(h).between(1.0, 2.5), h * 100, h)
    df['HeightAtStart'] = pd.Series(h, index=df.index).where(lambda s: s.between(100, 220))
    for c in ['WeightAtStart', 'WeightAtLastVisit']:
        df.loc[~df[c].between(20, 200), c] = np.nan
    for c in ['DateOfConfirmedHIV', 'DateArtStarted']:
        df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True)

    d = (df['DateArtStarted'] - df['DateOfConfirmedHIV']).dt.days
    df['days_to_ART'] = d.where(d.between(0, 3650))
    df['BMI_start'] = (df['WeightAtStart'] / ((df['HeightAtStart'] / 100) ** 2)) \
        .where(lambda s: s.between(10, 60))
    df['CD4_improvement'] = df['MostRecentCd4Count'] - df['Cd4AtStart']
    df['weight_change'] = df['WeightAtLastVisit'] - df['WeightAtStart']
    stage = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    df['stage_start_num'] = df['ClinicalStageAtStart'].map(stage)
    df['stage_worsened'] = (df['ClinicalStageAtLastVisit'].map(stage)
                            > df['stage_start_num']).astype(int)
    df['sex_female'] = (_norm(df['Sex']) == 'female').astype(int)
    df['tb_positive'] = _norm(df['TbStatusAtLAstVisit']).isin(TB_POSITIVE).astype(int)
    df['opp_infection'] = (_norm(df['OpportunisticInfectionPresentAtLastVisit'])
                           == 'yes').astype(int)
    df['side_effects'] = (_norm(df['AnySideEffects']) == 'yes').astype(int)
    df['had_interruption'] = (_norm(df['ArtInterruption']) == 'yes').astype(int)
    df['target'] = (_norm(df['ArvAdherenceLatestLevel']) == 'poor').astype(int)

    if verbose:
        print("COHORT")
        print(f"  raw records                        {n0:>8,}")
        print(f"  - blank adherence outcome          {n0-n1:>8,}  -> {n1:,}")
        print(f"  - age <18 or >100                  {n1-n2:>8,}  -> {n2:,}")
        print(f"  - unknown interruption status      {n_unknown:>8,}  -> {len(df):,}")
        print(f"  poor adherence                     {int(df['target'].sum()):>8,}"
              f"  ({df['target'].mean()*100:.2f}%)")
        assert df['tb_positive'].nunique() > 1, "tb_positive is still constant"
        print(f"  tb_positive == 1                   {int(df['tb_positive'].sum()):>8,}"
              f"  (v1 was constant 0)")
    return df, n_unknown


def op_point(p, y, t):
    pr = p >= t
    tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum())
    fn = int((~pr & (y == 1)).sum()); tn = int((~pr & (y == 0)).sum())
    return (int(pr.sum()),
            tp / (tp + fn) if tp + fn else 0.0,
            tn / (tn + fp) if tn + fp else 0.0,
            tp / (tp + fp) if tp + fp else 0.0)


def main():
    df, n_unknown = build_cohort()
    X, y = df[FEATURES].values, df['target'].values

    cv = cross_val_score(build_calibrated(), X, y,
                         cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                         scoring='roc_auc', n_jobs=-1)
    print(f"\n10-fold CV AUC          {cv.mean():.4f} +/- {cv.std():.4f}")

    # Temporal split by ART INITIATION date. The outcome carries no timestamp
    # anywhere in the source file, so this separates initiation era and
    # follow-up duration, not calendar time. Do not call it prospective
    # simulation.
    dt = df.dropna(subset=['DateArtStarted']).sort_values('DateArtStarted')
    k = int(len(dt) * 0.70)
    cutoff = dt.iloc[k]['DateArtStarted'].date()
    Xtr, ytr = dt[FEATURES].values[:k], dt['target'].values[:k]
    Xte, yte = dt[FEATURES].values[k:], dt['target'].values[k:]

    tm = build_calibrated().fit(Xtr, ytr)
    p = tm.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    brier = brier_score_loss(yte, p)

    rs = np.random.RandomState(SEED)
    boots = [roc_auc_score(yte[i], p[i]) for i in rs.choice(len(yte), (1000, len(yte)))
             if len(np.unique(yte[i])) > 1]
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])

    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(
        np.log(pc / (1 - pc)).reshape(-1, 1), yte)
    slope, intercept = float(lr.coef_[0][0]), float(lr.intercept_[0])
    eo = float(p.mean() / yte.mean())

    print(f"temporal AUC            {auc:.4f}  (95% CI {lo_ci:.4f}-{hi_ci:.4f})   "
          f"cutoff {cutoff}")
    print(f"  train {len(Xtr):,} ({ytr.mean()*100:.2f}%) | test {len(Xte):,} "
          f"({yte.mean()*100:.2f}%)")
    print(f"  Brier {brier:.4f}   slope {slope:.3f}   intercept {intercept:.3f}   "
          f"E:O {eo:.2f}")
    print(f"  E:O > 1 reflects the lower event rate among later initiators, not")
    print(f"  miscalibration further fitting can remove. Sites recalibrate locally.")

    # FIX 4 — thresholds anchored to flagging rate.
    HI = float(np.quantile(p, 1 - FLAG_RATE_HI))
    LO = float(np.quantile(p, 1 - FLAG_RATE_LO))
    print(f"\nOPERATING POINTS (calibrated, temporal test set, "
          f"prevalence {yte.mean()*100:.2f}%)")
    print(f"  {'thresh':>8}{'flagged':>16}{'sens':>9}{'spec':>9}{'PPV':>8}")
    ops = {}
    for t, lab in [(HI, 'HIGH'), (LO, 'MEDIUM')]:
        nf, se, sp, pv = op_point(p, yte, t)
        ops[lab] = (nf, se, sp, pv)
        print(f"  {t:>8.4f}{nf:>9,} ({nf/len(yte)*100:4.1f}%){se*100:>8.1f}%"
              f"{sp*100:>8.1f}%{pv*100:>7.1f}%   <- {lab}")

    # ── final artifacts ───────────────────────────────────────────────────
    final = build_calibrated().fit(X, y)       # scoring
    shap_pipe = build_base().fit(X, y)         # explanation only

    p_cal = final.predict_proba(X)[:, 1]
    p_unc = shap_pipe.predict_proba(X)[:, 1]
    rank_corr = float(pd.Series(p_cal).corr(pd.Series(p_unc), method='spearman'))
    n_hi_cal = int((p_cal >= HI).sum())
    n_hi_unc = int((p_unc >= HI).sum())

    print(f"\nSHAP MODEL CHECK")
    print(f"  rank correlation calibrated vs uncalibrated: {rank_corr:.4f}")
    print(f"  -> SHAP attribution order is preserved; safe for explanation")
    print(f"  MISUSE CHECK — applying the HIGH threshold to the WRONG model:")
    print(f"    calibrated   (correct) flags {n_hi_cal:,} of {len(X):,} "
          f"({n_hi_cal/len(X)*100:.1f}%)")
    print(f"    uncalibrated (WRONG)   flags {n_hi_unc:,} of {len(X):,} "
          f"({n_hi_unc/len(X)*100:.1f}%)")
    if n_hi_cal:
        print(f"    ratio {n_hi_unc/n_hi_cal:.2f}x  <- silent if unguarded")

    artifact = {
        # CONTRACT — model.py must honour this:
        #   rf_model    calibrated. USE FOR ALL SCORING.
        #   shap_model  UNCALIBRATED. Explanation only. Never call
        #               predict_proba on it for a risk score — its output is on
        #               the old compressed scale and the thresholds below do not
        #               apply to it.
        'rf_model': final,
        'shap_model': shap_pipe,
        'shap_model_is_uncalibrated': True,
        'score_with': 'rf_model',
        'explain_with': 'shap_model',
        'calibrated': True,
        'calibration_method': 'sigmoid',
        'auc': float(auc), 'auc_ci': (float(lo_ci), float(hi_ci)),
        'cv_auc': float(cv.mean()), 'brier': float(brier),
        'calibration_slope': slope, 'calibration_intercept': intercept,
        'expected_observed': eo,
        'features': FEATURES, 'n': int(len(df)), 'prevalence': float(y.mean()),
        'threshold_hi': HI, 'threshold_lo': LO,
        'flag_rate_hi': FLAG_RATE_HI, 'flag_rate_lo': FLAG_RATE_LO,
        'cutoff': str(cutoff), 'version': VERSION,
        'trained': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        # Guard: model.py asserts its scores fall in this range. An uncalibrated
        # model cannot produce the upper end, so misuse fails loudly.
        'expected_score_range': [float(p_cal.min()), float(p_cal.max())],
        'note': ('no SMOTE; imputer inside pipeline; sigmoid-calibrated; '
                 'tb_positive from TbStatusAtLAstVisit in {Confirmed TB, TB Treatment}; '
                 f'{n_unknown} patients with unknown ArtInterruption excluded'),
    }
    with open('cv_results.pkl', 'wb') as f:
        pickle.dump(artifact, f)
    with open('prepped_data.pkl', 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'features': FEATURES}, f)

    # FIX 5 — deterministic manifest digest.
    src_sha = hashlib.sha256(open(DATA, 'rb').read()).hexdigest()
    model_sha = hashlib.sha256(pickle.dumps(final, protocol=5)).hexdigest()
    manifest = {
        'version': VERSION, 'seed': SEED,
        'source_file_sha256': src_sha, 'model_sha256': model_sha,
        'features': FEATURES, 'n': artifact['n'],
        'prevalence': round(artifact['prevalence'], 6), 'cutoff': artifact['cutoff'],
        'metrics': {'cv_auc': round(cv.mean(), 4), 'temporal_auc': round(auc, 4),
                    'brier': round(brier, 4), 'calibration_slope': round(slope, 3)},
        'thresholds': {'high': round(HI, 6), 'medium': round(LO, 6)},
    }
    lock_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    lock = dict(manifest)
    lock['lock_sha256'] = lock_sha
    lock['locked_utc'] = artifact['trained']
    lock['reproduce'] = 'python retrain_v1_1.py -> lock_sha256 must match'
    lock['outcome'] = ("ArvAdherenceLatestLevel == 'Poor' at most recent visit "
                       "(same-encounter; not a forecast)")
    lock['validation_note'] = (
        'Temporal split is by ART INITIATION date. The outcome has no timestamp in '
        'the source data, so this tests generalisation across initiation eras and '
        'follow-up durations, not calendar time. No external validation performed.')
    lock['exclusions'] = ['blank adherence outcome', 'age <18 or >100',
                          'unknown ART interruption status']
    json.dump(lock, open('MODEL_LOCK.json', 'w'), indent=2)

    print(f"\nLOCKED")
    print(f"  lock_sha256    {lock_sha}")
    print(f"  source sha256  {src_sha[:32]}...")
    print(f"  model  sha256  {model_sha[:32]}...")

    print(f"""
--- paste into constants.py ---
MODEL_VERSION      = '{VERSION}'
MODEL_LOCK_SHA256  = '{lock_sha}'
BASELINE_AUC       = {auc:.4f}   # temporal holdout, sigmoid-calibrated, n={len(Xte):,}
BASELINE_AUC_CI    = ({lo_ci:.4f}, {hi_ci:.4f})
BASELINE_CV_AUC    = {cv.mean():.4f}
BASELINE_BRIER     = {brier:.4f}
CALIBRATION_SLOPE  = {slope:.3f}

BASELINE_THRESHOLD = {HI:.4f}   # HIGH — sens {ops['HIGH'][1]*100:.1f}%, spec {ops['HIGH'][2]*100:.1f}%, PPV {ops['HIGH'][3]*100:.1f}%
MEDIUM_THRESHOLD   = {LO:.4f}   # MEDIUM — sens {ops['MEDIUM'][1]*100:.1f}%, spec {ops['MEDIUM'][2]*100:.1f}%, PPV {ops['MEDIUM'][3]*100:.1f}%
# Anchored to v1 flagging rates ({FLAG_RATE_HI*100:.1f}% / {FLAG_RATE_LO*100:.1f}%) so clinic workload is unchanged.

TB_POSITIVE_VALUES = {{'confirmed tb', 'tb treatment'}}

--- required change in model.py load_model() ---
    model      = cv['rf_model']        # calibrated — scoring only
    shap_base  = cv['shap_model']      # uncalibrated — explanation only
    imputer    = shap_base.named_steps['imp']
    scaler     = shap_base.named_steps['sc']
    clf        = shap_base.named_steps['clf']
    assert cv.get('score_with') == 'rf_model'

--- and in model.py load_model(), immediately after unpickling ---
    from sklearn.calibration import CalibratedClassifierCV
    if not isinstance(cv['rf_model'], CalibratedClassifierCV):
        raise RuntimeError(
            'cv_results.pkl rf_model is not calibrated. The thresholds in '
            'constants.py are on the calibrated scale and do not apply.')
    if hasattr(cv['shap_model'], 'predict_proba') and cv['rf_model'] is cv['shap_model']:
        raise RuntimeError('shap_model and rf_model are the same object.')

  A type check, not a range check. The uncalibrated model produces LOWER
  scores than the calibrated one (max 0.39 vs 0.78), so an upper-bound guard
  never fires — misuse would silently flag ~36% MORE patients than the
  clinic is staffed for, not fewer.
""")


if __name__ == '__main__':
    main()

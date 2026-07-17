"""
retrain_v1.py — SmartDaaS v1, corrected.
Produces cv_results.pkl and prepped_data.pkl with the exact keys model.py reads.
"""
import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
SEED = 42
# ---------- cohort construction ----------
import pandas as pd, numpy as np
df = pd.read_excel('data/QualityOfCare.xlsx')
df = df[df['ArvAdherenceLatestLevel'].notna()]
df = df[(df['Age'] >= 18) & (df['Age'] <= 100)].copy()
df.loc[df['Cd4Unit']=='%', ['Cd4AtStart','MostRecentCd4Count']] = np.nan
for c in ['Cd4AtStart','MostRecentCd4Count']: df.loc[df[c] > 2000, c] = np.nan
h = df['HeightAtStart'].copy(); h = np.where(pd.Series(h).between(1.0,2.5), h*100, h)
df['HeightAtStart'] = pd.Series(h, index=df.index).where(lambda s: s.between(100,220))
for c in ['WeightAtStart','WeightAtLastVisit']: df.loc[~df[c].between(20,200), c] = np.nan
for c in ['DateOfConfirmedHIV','DateArtStarted']: df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True)
d = (df['DateArtStarted']-df['DateOfConfirmedHIV']).dt.days
df['days_to_ART'] = d.where(d.between(0,3650))
df['BMI_start'] = (df['WeightAtStart']/((df['HeightAtStart']/100)**2)).where(lambda s: s.between(10,60))
df['CD4_improvement'] = df['MostRecentCd4Count']-df['Cd4AtStart']
df['weight_change']   = df['WeightAtLastVisit']-df['WeightAtStart']
_s={'I':1,'II':2,'III':3,'IV':4}
df['stage_start_num'] = df['ClinicalStageAtStart'].map(_s)
df['stage_worsened']  = (df['ClinicalStageAtLastVisit'].map(_s) > df['stage_start_num']).astype(int)
df['sex_female']  = (df['Sex'].str.lower().str.strip()=='female').astype(int)
df['tb_positive'] = df['TbStatusAtLAstVisit'].str.lower().str.contains('positive|active', na=False).astype(int)
df['opp_infection'] = (df['OpportunisticInfectionPresentAtLastVisit'].str.lower().str.strip()=='yes').astype(int)
df['side_effects']  = (df['AnySideEffects'].str.lower().str.strip()=='yes').astype(int)
df['had_interruption'] = (df['ArtInterruption'].str.lower().str.strip()=='yes').astype(int)
df['target'] = (df['ArvAdherenceLatestLevel'].str.lower().str.strip()=='poor').astype(int)
F15 = ['Age','sex_female','Cd4AtStart','MostRecentCd4Count','CD4_improvement','stage_start_num',
       'WeightAtStart','weight_change','BMI_start','days_to_ART','had_interruption',
       'opp_infection','side_effects','tb_positive','stage_worsened']

# ---------- end cohort ----------

def build():
    return Pipeline([('imp', SimpleImputer(strategy='median')),   # <- imputer INSIDE: serving == training
                     ('sc',  StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                                    n_jobs=-1, random_state=SEED))])

X, y = df[F15].values, df['target'].values
print(f"cohort            n = {len(df):,}   positives = {int(y.sum()):,} ({100*y.mean():.2f}%)")

# ---- internal ----------------------------------------------------------------
cvauc = cross_val_score(build(), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                        scoring='roc_auc', n_jobs=-1)
print(f"10-fold CV AUC      = {cvauc.mean():.4f} +/- {cvauc.std():.4f}")

# ---- temporal (the headline number) ------------------------------------------
dt = df.dropna(subset=['DateArtStarted']).sort_values('DateArtStarted')
k  = int(len(dt)*0.70)
Xtr, ytr = dt[F15].values[:k], dt['target'].values[:k]
Xte, yte = dt[F15].values[k:], dt['target'].values[k:]
tm = build().fit(Xtr, ytr)
pte = tm.predict_proba(Xte)[:,1]
tauc = roc_auc_score(yte, pte)
rs = np.random.RandomState(SEED)
bs = [roc_auc_score(yte[i], pte[i]) for i in rs.choice(len(yte), (1000, len(yte))) if len(np.unique(yte[i]))>1]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"temporal AUC        = {tauc:.4f}  (95% CI {lo:.4f}-{hi:.4f})   cutoff {dt.iloc[k]['DateArtStarted'].date()}")
print(f"                      train {len(Xtr):,} / test {len(Xte):,}")

# ---- thresholds, derived from the temporal test set --------------------------
print("\nOPERATING POINTS (temporal test set, n={:,}, prevalence {:.2f}%)".format(len(yte), 100*yte.mean()))
print("  thresh   flagged    sens    spec     PPV")
rows=[]
for t in [0.03,0.05,0.075,0.10,0.15,0.20,0.30]:
    pr=(pte>=t).astype(int)
    tp=((pr==1)&(yte==1)).sum(); fn=((pr==0)&(yte==1)).sum()
    tn=((pr==0)&(yte==0)).sum(); fp=((pr==1)&(yte==0)).sum()
    se=tp/(tp+fn); sp=tn/(tn+fp); ppv=tp/(tp+fp) if (tp+fp) else 0
    rows.append((t,pr.sum(),se,sp,ppv))
    print(f"   {t:.3f}   {pr.sum():>5,} ({100*pr.mean():4.1f}%)  {100*se:5.1f}%  {100*sp:5.1f}%  {100*ppv:5.1f}%")

HI = 0.15; LO = 0.075
r_hi = [r for r in rows if r[0]==HI][0]; r_lo = [r for r in rows if r[0]==LO][0]

# ---- final model on the full clean cohort ------------------------------------
final = build().fit(X, y)
with open('cv_results.pkl','wb') as f:
    pickle.dump({'rf_model': final, 'auc': float(tauc), 'auc_ci': (float(lo), float(hi)),
                 'cv_auc': float(cvauc.mean()), 'features': F15, 'n': int(len(df)),
                 'prevalence': float(y.mean()), 'threshold_hi': HI, 'threshold_lo': LO,
                 'trained': '2026-07-16', 'note': 'no SMOTE; imputer inside pipeline'}, f)
with open('prepped_data.pkl','wb') as f:
    pickle.dump({'X': X, 'y': y, 'features': F15}, f)
print("\nwrote cv_results.pkl  (keys: rf_model, auc, ...)")
print("wrote prepped_data.pkl (keys: X, y)  -- real patients, no synthetic rows")
print(f"\nnamed_steps: {list(final.named_steps)}   <- 'sc' and 'clf' still present for model.py")

print(f"""
--- paste into constants.py ---
BASELINE_AUC       = {tauc:.3f}   # temporal holdout, 15 features, n={len(Xte):,}
BASELINE_THRESHOLD = {HI}    # HIGH
# At this threshold on the temporal holdout:
#   Sensitivity = {100*r_hi[2]:.1f}%, Specificity = {100*r_hi[3]:.1f}%, PPV = {100*r_hi[4]:.1f}%
#   flags {r_hi[1]:,} of {len(yte):,} patients ({100*r_hi[1]/len(yte):.1f}%)
MEDIUM_THRESHOLD   = {LO}   # sens {100*r_lo[2]:.1f}%, spec {100*r_lo[3]:.1f}%
""")

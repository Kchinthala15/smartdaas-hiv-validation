"""
06_subgroup_fairness.py
AUC-ROC by sex, age group, CD4 strata, and WHO clinical stage.
"""
import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from utils import FEATURES, SEED
np.random.seed(SEED)

def main():
    print("="*60); print("Step 6: Subgroup Fairness Analysis"); print("="*60)
    with open('results/preprocessed_data.pkl','rb') as f: d=pickle.load(f)
    with open('results/cv_results.pkl','rb') as f: r=pickle.load(f)

    df_m = d['df_m'].copy()
    best = r['best_model']
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy='median')
    X = imp.fit_transform(df_m[FEATURES])
    sc = StandardScaler(); X_sc = sc.fit_transform(X)
    df_m['y_prob'] = best.named_steps['clf'].predict_proba(
        best.named_steps['sc'].transform(X))[:,1]
    df_m['target'] = d['y']

    subgroups = {
        'Sex':       {'Female':df_m['sex_female']==1,'Male':df_m['sex_female']==0},
        'Age':       {'<30':df_m['Age']<30,'30-49':(df_m['Age']>=30)&(df_m['Age']<50),'>=50':df_m['Age']>=50},
        'CD4':       {'<200':df_m['Cd4AtStart']<200,'200-350':(df_m['Cd4AtStart']>=200)&(df_m['Cd4AtStart']<350),'>=350':df_m['Cd4AtStart']>=350},
        'WHO Stage': {'I-II':df_m['stage_start_num']<=2,'III-IV':df_m['stage_start_num']>2},
    }

    sg_res = {}
    for gname, subs in subgroups.items():
        sg_res[gname] = {}
        for label, mask in subs.items():
            sg = df_m[mask]
            if sg['target'].sum() < 10: continue
            auc = roc_auc_score(sg['target'], sg['y_prob'])
            sg_res[gname][label] = {'n':len(sg),'auc':float(auc),'prev':float(sg['target'].mean()*100)}
            print(f"  {gname}|{label:10s}: n={len(sg):,} AUC={auc:.4f} poor={sg['target'].mean()*100:.1f}%")

    aucs_all = [v['auc'] for g in sg_res.values() for v in g.values()]
    print(f"\n  Max AUC difference: {max(aucs_all)-min(aucs_all):.4f}")

    with open('results/subgroup_results.pkl','wb') as f:
        pickle.dump(sg_res, f)
    print("Saved: results/subgroup_results.pkl\nStep 6 complete.")

if __name__ == '__main__': main()

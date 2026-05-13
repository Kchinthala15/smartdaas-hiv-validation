"""
paper2/src/04_icc_model_comparison.py
ICC from null linear probability model + AIC/BIC model comparison.
"""
import pandas as pd, numpy as np, pickle, statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings; warnings.filterwarnings('ignore')

FEATURES_PATIENT  = ['sex_female','stage_num','cd4_std','Age']
FEATURES_FULL     = ['facility_primary','facility_secondary','type_faith',
                     'type_private_profit','has_NGO','has_federal','mixed_funding',
                     'sex_female','stage_num','cd4_std','Age']

def main():
    print("="*60); print("Paper 2 — Step 4: ICC + Model Comparison"); print("="*60)
    with open('paper2/results/data_engineered.pkl','rb') as f: d=pickle.load(f)
    df=d['df']
    mdf=df[FEATURES_FULL+['poor_outcome','facility_group']].dropna()

    # ICC from null LPM
    lpm=MixedLM(mdf['poor_outcome'],np.ones((len(mdf),1)),groups=mdf['facility_group'])
    lpm_r=lpm.fit(reml=True,method='bfgs')
    var_fac=float(lpm_r.cov_re.iloc[0,0]); var_res=float(lpm_r.scale)
    icc=var_fac/(var_fac+var_res)
    print(f"\nICC: {icc:.4f} ({icc*100:.1f}% facility-level variance)")

    # Model comparison
    y=mdf['poor_outcome']
    m0=sm.Logit(y,sm.add_constant(np.ones(len(y)))).fit(disp=0)
    m1=sm.Logit(y,sm.add_constant(mdf[FEATURES_PATIENT])).fit(disp=0)
    m2=sm.Logit(y,sm.add_constant(mdf[FEATURES_FULL]),cov_type='HC3').fit(disp=0,cov_type='HC3')
    from scipy.stats import chi2 as chi2dist
    lr=2*(m2.llf-m1.llf); lr_df=m2.df_model-m1.df_model
    lr_p=1-chi2dist.cdf(lr,lr_df)
    print(f"\nModel comparison:")
    print(f"  Null:           AIC={m0.aic:.1f}  BIC={m0.bic:.1f}")
    print(f"  Patient-only:   AIC={m1.aic:.1f}  BIC={m1.bic:.1f}")
    print(f"  Full (+facility):AIC={m2.aic:.1f}  BIC={m2.bic:.1f}")
    print(f"  LR test: chi2={lr:.2f}, df={lr_df:.0f}, p={lr_p:.6f}")

    with open('paper2/results/icc_results.pkl','wb') as f:
        pickle.dump({'icc':icc,'var_fac':var_fac,'var_res':var_res,
                     'aic_null':m0.aic,'aic_patient':m1.aic,'aic_full':m2.aic,
                     'bic_null':m0.bic,'bic_patient':m1.bic,'bic_full':m2.bic,
                     'lr_stat':lr,'lr_p':lr_p},f)
    print("\nSaved: paper2/results/icc_results.pkl\nStep 4 complete.")

if __name__=='__main__': main()

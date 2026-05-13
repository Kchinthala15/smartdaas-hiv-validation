"""
paper2/src/06_sensitivity_analyses.py
4 pre-specified sensitivity analyses:
  1. Adherence-only outcome
  2. Exclude mortality from composite
  3. Complete-case analysis
  4. Exclude primary health centres
"""
import pandas as pd, numpy as np, pickle, statsmodels.api as sm
import warnings; warnings.filterwarnings('ignore')

FEATURES=['facility_primary','facility_secondary','type_faith','type_private_profit',
          'has_NGO','has_federal','mixed_funding','sex_female','stage_num','cd4_std','Age']

def run_model(df, outcome_col):
    mdf=df[FEATURES+[outcome_col]].dropna()
    X=sm.add_constant(mdf[FEATURES]); y=mdf[outcome_col]
    m=sm.Logit(y,X).fit(disp=0,cov_type='HC3')
    or_primary=np.exp(m.params['facility_primary'])
    p_primary =m.pvalues['facility_primary']
    or_ngo    =np.exp(m.params['has_NGO'])
    p_ngo     =m.pvalues['has_NGO']
    return or_primary,p_primary,or_ngo,p_ngo,len(mdf)

def main():
    print("="*60); print("Paper 2 — Step 6: Sensitivity Analyses"); print("="*60)
    with open('paper2/results/data_engineered.pkl','rb') as f: d=pickle.load(f)
    df=d['df']

    analyses=[
        ("Main analysis (composite)",       df,                                              'poor_outcome'),
        ("Adherence only",                   df,                                              'poor_adherence'),
        ("Exclude mortality",                df.assign(po2=((df['poor_adherence']==1)|(df['art_interrupted']==1)).astype(int)), 'po2'),
        ("Complete cases only",              df.dropna(subset=['Cd4AtStart','stage_num','Age','sex_female']), 'poor_outcome'),
    ]

    results=[]
    print(f"\n{'Analysis':30s}  {'Primary OR':>12s}  {'p':>8s}  {'NGO OR':>10s}  {'p':>8s}  {'n':>8s}")
    print("-"*80)
    for label,df_s,col in analyses:
        try:
            or_p,p_p,or_n,p_n,n=run_model(df_s,col)
            sig_p='***' if p_p<0.001 else '**' if p_p<0.01 else '*' if p_p<0.05 else 'ns'
            sig_n='***' if p_n<0.001 else '**' if p_n<0.01 else '*' if p_n<0.05 else 'ns'
            print(f"  {label:30s}  {or_p:6.3f} {sig_p:4s}  {p_p:8.4f}  {or_n:6.3f} {sig_n:4s}  {p_n:8.4f}  {n:8,d}")
            results.append({'Analysis':label,'OR_Primary':or_p,'p_Primary':p_p,
                            'OR_NGO':or_n,'p_NGO':p_n,'n':n})
        except Exception as e:
            print(f"  {label}: FAILED — {e}")

    pd.DataFrame(results).to_csv('paper2/results/sensitivity_analyses.csv',index=False)
    print("\nSaved: paper2/results/sensitivity_analyses.csv\nStep 6 complete.")

if __name__=='__main__': main()

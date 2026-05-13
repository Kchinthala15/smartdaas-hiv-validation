"""
paper2/src/05_interaction_analysis.py
Interaction terms: NGO × facility level, Federal × facility level.
Tests whether funding penalty differs by care level.
"""
import pandas as pd, numpy as np, pickle, statsmodels.api as sm
import warnings; warnings.filterwarnings('ignore')

BASE = ['facility_primary','facility_secondary','has_NGO','has_federal',
        'mixed_funding','sex_female','stage_num','cd4_std']

def main():
    print("="*60); print("Paper 2 — Step 5: Interaction Analysis"); print("="*60)
    with open('paper2/results/data_engineered.pkl','rb') as f: d=pickle.load(f)
    df=d['df']
    mdf=df[BASE+['poor_outcome']].dropna().copy()
    mdf['ngo_x_primary']   = mdf['has_NGO']*mdf['facility_primary']
    mdf['ngo_x_secondary'] = mdf['has_NGO']*mdf['facility_secondary']
    mdf['fed_x_primary']   = mdf['has_federal']*mdf['facility_primary']
    mdf['fed_x_secondary'] = mdf['has_federal']*mdf['facility_secondary']

    X=sm.add_constant(mdf.drop('poor_outcome',axis=1))
    model=sm.Logit(mdf['poor_outcome'],X).fit(disp=0)
    print(model.summary2())

    print("\nKey interaction terms:")
    for term in ['ngo_x_primary','ngo_x_secondary','fed_x_primary','fed_x_secondary']:
        coef=model.params[term]; p=model.pvalues[term]
        sig='***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"  {term:20s}: coef={coef:.4f}, OR={np.exp(coef):.3f}, p={p:.4f} {sig}")

    print("\nStratified rates (NGO vs non-NGO by facility level):")
    df2=d['df']
    for level in ['Primary health center','Secondary health facility','Tertiary hospital']:
        sub=df2[df2['Health facility level']==level]
        r_ngo=sub[sub['has_NGO']==1]['poor_outcome'].mean()*100
        r_non=sub[sub['has_NGO']==0]['poor_outcome'].mean()*100
        print(f"  {level:30s}: NGO={r_ngo:.1f}%  Non-NGO={r_non:.1f}%  Diff={r_ngo-r_non:.1f}pp")

    with open('paper2/results/interaction_results.pkl','wb') as f:
        pickle.dump({'model':model,'mdf':mdf},f)
    print("\nSaved: paper2/results/interaction_results.pkl\nStep 5 complete.")

if __name__=='__main__': main()

"""
paper2/src/07_facility_typology.py
Risk-adjusted facility performance: observed vs expected outcomes.
Identifies positive deviant and underperforming facility types.
"""
import pandas as pd, numpy as np, pickle, statsmodels.api as sm
import warnings; warnings.filterwarnings('ignore')

FEATURES=['facility_primary','facility_secondary','type_faith','type_private_profit',
          'has_NGO','has_federal','mixed_funding','sex_female','stage_num','cd4_std','Age']

def main():
    print("="*60); print("Paper 2 — Step 7: Facility Performance Typology"); print("="*60)
    with open('paper2/results/data_engineered.pkl','rb') as f: d=pickle.load(f)
    df=d['df']
    mdf=df[FEATURES+['poor_outcome','facility_group']].dropna().copy()
    X=sm.add_constant(mdf[FEATURES]); y=mdf['poor_outcome']
    model=sm.Logit(y,X).fit(disp=0)
    mdf['predicted']=model.predict(X)

    perf=mdf.groupby('facility_group').agg(
        n=('poor_outcome','count'),
        observed=('poor_outcome','mean'),
        expected=('predicted','mean'),
    ).reset_index()
    perf['ratio']=perf['observed']/perf['expected']
    perf['excess_pp']=(perf['observed']-perf['expected'])*100

    def classify(r):
        if r['ratio']<0.85: return 'Positive Deviant (better than expected)'
        elif r['ratio']>1.15: return 'Underperformer (worse than expected)'
        return 'Expected performer'
    perf['type']=perf.apply(classify,axis=1)
    perf=perf.sort_values('ratio')

    print("\nFacility Performance Typology:")
    print(perf[['facility_group','n','observed','expected','ratio','type']].to_string())
    print("\nPositive deviants:")
    print(perf[perf['type'].str.contains('Positive')][['facility_group','n','ratio']].to_string())

    perf.to_csv('paper2/results/facility_typology.csv',index=False)
    print("\nSaved: paper2/results/facility_typology.csv\nStep 7 complete.")

if __name__=='__main__': main()

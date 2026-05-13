"""
paper2/src/09_economic_implications.py
Estimates excess poor outcomes and avoidable costs attributable
to sub-tertiary facility care.

Assumptions (clearly stated, based on published literature):
  - USD 1,850 per poor outcome (retreatment + monitoring + second-line ART)
  - Reference: PEPFAR unit cost estimates for treatment interruption management
  - Counterfactual: all facilities achieve tertiary-equivalent outcome rates

Note: These are indicative estimates for a Nigerian programme context.
Direct extrapolation to other settings requires caution.
"""
import pandas as pd, numpy as np, pickle, json
import warnings; warnings.filterwarnings('ignore')

COST_PER_POOR_OUTCOME = 1850  # USD

def main():
    print("="*60); print("Paper 2 — Step 9: Economic Implications"); print("="*60)
    with open('paper2/results/data_engineered.pkl','rb') as f: d=pickle.load(f)
    df=d['df']

    levels=['Primary health center','Secondary health facility','Tertiary hospital']
    rates={l: df[df['Health facility level']==l]['poor_outcome'].mean() for l in levels}
    ns   ={l: (df['Health facility level']==l).sum() for l in levels}

    ref_rate=rates['Tertiary hospital']
    print(f"\nOutcome rates:")
    for l in levels:
        print(f"  {l:35s}: {rates[l]*100:.1f}% (n={ns[l]:,})")

    print(f"\nCounterfactual (all achieve tertiary rate = {ref_rate*100:.1f}%):")
    total_excess=0
    for l in ['Primary health center','Secondary health facility']:
        excess=(rates[l]-ref_rate)*ns[l]
        total_excess+=excess
        print(f"  Excess at {l.split()[0]:10s}: {excess:.0f} poor outcomes")

    total_cost=total_excess*COST_PER_POOR_OUTCOME
    print(f"\nTotal excess poor outcomes: {total_excess:.0f}")
    print(f"Estimated avoidable cost:   USD {total_cost:,.0f}")
    print(f"Per 10,000 ART patients:    USD {total_cost/len(df)*10000:,.0f}")
    print(f"\nNote: Indicative estimates for Nigerian programme context only.")
    print(f"USD {COST_PER_POOR_OUTCOME:,} per poor outcome assumption — see paper for justification.")

    results={'rates':{l:float(r) for l,r in rates.items()},
             'ns':{l:int(n) for l,n in ns.items()},
             'excess_primary':float((rates['Primary health center']-ref_rate)*ns['Primary health center']),
             'excess_secondary':float((rates['Secondary health facility']-ref_rate)*ns['Secondary health facility']),
             'total_excess':float(total_excess),
             'total_cost_usd':float(total_cost),
             'cost_per_poor_outcome':COST_PER_POOR_OUTCOME}
    with open('paper2/results/economic_results.json','w') as f:
        json.dump(results,f,indent=2)
    print("\nSaved: paper2/results/economic_results.json\nStep 9 complete.")

if __name__=='__main__': main()

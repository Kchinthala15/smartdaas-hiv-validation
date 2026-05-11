"""
07_economic_modelling.py
One-way sensitivity analysis of per-patient cost savings.
Parameters anchored to peer-reviewed literature.
"""
import pickle, numpy as np, pandas as pd
from utils import ECON_PARAMS

def compute_savings(params):
    delay_saving = (params['trad_delay_days'] - params['ai_delay_days']) * \
                   params['daily_tx_prob'] * params['secondary_infection_cost']
    art_saving   = params['art_avoidance_pct'] * params['art_annual_cost']
    return delay_saving + art_saving

def main():
    print("="*60); print("Step 7: Economic Modelling"); print("="*60)

    results = {}
    for scenario, params in ECON_PARAMS.items():
        savings = compute_savings(params)
        results[scenario] = {
            'per_patient': round(savings, 2),
            'total_5k':    round(savings * 5000, 0),
            'art_saving':  round(params['art_avoidance_pct']*params['art_annual_cost'], 2),
            'delay_saving':round((params['trad_delay_days']-params['ai_delay_days'])*
                                 params['daily_tx_prob']*params['secondary_infection_cost'], 2),
        }
        print(f"  {scenario:12s}: ${savings:.2f}/patient | ${savings*5000:,.0f} at 5,000 patients")

    # One-way sensitivity (tornado)
    base = ECON_PARAMS['base']
    base_savings = compute_savings(base)
    tornado = {}
    for param_name, (low_val, high_val) in [
        ('art_annual_cost',     (900, 1500)),
        ('art_avoidance_pct',   (0.18, 0.35)),
        ('trad_delay_days',     (45, 90)),
        ('ai_delay_days',       (14, 5)),
        ('secondary_infection_cost', (800, 1600)),
    ]:
        low_params  = {**base, param_name: low_val}
        high_params = {**base, param_name: high_val}
        tornado[param_name] = {
            'low':  compute_savings(low_params),
            'high': compute_savings(high_params),
            'base': base_savings,
        }

    with open('results/economic_results.pkl','wb') as f:
        pickle.dump({'scenarios':results,'tornado':tornado,'base_savings':base_savings}, f)

    pd.DataFrame([
        {'Scenario': s, **v} for s,v in results.items()
    ]).to_csv('results/economic_summary.csv', index=False)

    print(f"\n  Base case: ${base_savings:.2f}/patient")
    print("Saved: results/economic_results.pkl")
    print("Saved: results/economic_summary.csv")
    print("Step 7 complete.")

if __name__ == '__main__': main()

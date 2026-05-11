"""
04_shap_explainability.py — SHAP KernelExplainer analysis
"""
import pickle, numpy as np, warnings
warnings.filterwarnings('ignore')
import shap
from utils import FEATURES_DISPLAY, SEED
np.random.seed(SEED)

def main():
    print("="*60); print("Step 4: SHAP Explainability"); print("="*60)
    with open('results/preprocessed_data.pkl','rb') as f: d=pickle.load(f)
    with open('results/cv_results.pkl','rb') as f: r=pickle.load(f)

    X_bal = d['X_bal']
    best_model = r['best_model']
    rf = best_model.named_steps['clf']
    sc = best_model.named_steps['sc']
    X_sc = sc.transform(X_bal)

    bg_idx = np.random.choice(len(X_sc), 100, replace=False)
    ex_idx = np.random.choice(len(X_sc), 800, replace=False)
    X_bg, X_ex = X_sc[bg_idx], X_sc[ex_idx]

    print("  Computing SHAP values (KernelExplainer, n=800)...")
    def model_pred(x): return rf.predict_proba(x)[:,1]
    explainer = shap.KernelExplainer(model_pred, X_bg)
    sv = explainer.shap_values(X_ex, nsamples=50, silent=True)
    mean_shap = np.abs(sv).mean(axis=0)

    print("  Top 5 features by mean |SHAP|:")
    for i in np.argsort(mean_shap)[::-1][:5]:
        print(f"    {FEATURES_DISPLAY[i]:25s}: {mean_shap[i]:.4f}")

    with open('results/shap_results.pkl','wb') as f:
        pickle.dump({'sv':sv,'X_ex':X_ex,'mean_shap':mean_shap,
                     'feat_display':FEATURES_DISPLAY}, f)
    print("Saved: results/shap_results.pkl\nStep 4 complete.")

if __name__ == '__main__': main()

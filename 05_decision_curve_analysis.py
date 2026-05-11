"""
05_decision_curve_analysis.py
Decision curve analysis — net benefit across threshold probabilities.
Compares Random Forest vs. treat-all and treat-none strategies.
"""
import pickle, numpy as np, warnings
warnings.filterwarnings('ignore')
from utils import SEED
np.random.seed(SEED)

def compute_dca(y_true, y_prob, thresholds):
    n = len(y_true)
    nb_model, nb_all = [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tp = ((yp==1)&(y_true==1)).sum()
        fp = ((yp==1)&(y_true==0)).sum()
        nb_model.append(tp/n - fp/n * (t/(1-t)))
        nb_all.append(max(0, y_true.mean() - (1-y_true.mean())*(t/(1-t))))
    return np.array(nb_model), np.array(nb_all)

def main():
    print("="*60); print("Step 5: Decision Curve Analysis"); print("="*60)
    with open('results/cv_results.pkl','rb') as f: r=pickle.load(f)
    h = r['holdout']
    thresholds = np.linspace(0.01, 0.50, 100)
    nb_model, nb_all = compute_dca(h['y_te'], h['y_prob'], thresholds)
    max_nb_threshold = thresholds[np.argmax(nb_model)]
    print(f"  Max net benefit: {max(nb_model):.4f} at threshold {max_nb_threshold:.3f}")
    print(f"  Model outperforms treat-all at {(nb_model > nb_all).sum()} of {len(thresholds)} thresholds")
    with open('results/dca_results.pkl','wb') as f:
        pickle.dump({'thresholds':thresholds,'nb_model':nb_model,
                     'nb_all':nb_all,'prev':float(h['y_te'].mean())}, f)
    print("Saved: results/dca_results.pkl\nStep 5 complete.")

if __name__ == '__main__': main()

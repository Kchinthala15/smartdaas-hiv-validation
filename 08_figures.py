"""
08_figures.py
Generate all publication figures at 300 DPI.
Colorblind-safe palette. Liberation Sans font (Helvetica-equivalent).
"""
import pickle, numpy as np, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from utils import COLORS as C, set_style, panel_label, FEATURES_DISPLAY
import os
os.makedirs('figures', exist_ok=True)

set_style()

def load_all():
    out = {}
    for name in ['preprocessed_data','cv_results','temporal_results',
                 'shap_results','dca_results','subgroup_results','economic_results']:
        try:
            with open(f'results/{name}.pkl','rb') as f:
                out[name] = pickle.load(f)
        except FileNotFoundError:
            print(f"  Warning: {name}.pkl not found — run earlier scripts first")
    return out

def fig1_model_comparison(results_primary):
    metrics = ['AUC-ROC','Accuracy','F1-Score','Recall']
    mnames  = list(results_primary.keys())
    short   = ['LR','RF','GB','NN']
    colors  = [C['lgrey'],C['blue'],C['sky'],C['grey']]
    fig, axes = plt.subplots(1,4,figsize=(7.2,2.5)); fig.subplots_adjust(wspace=0.55)
    for i,metric in enumerate(metrics):
        ax = axes[i]
        means = [results_primary[m][metric][0] for m in mnames]
        stds  = [results_primary[m][metric][1] for m in mnames]
        ax.bar(short, means, color=colors, width=0.58,
               yerr=stds, capsize=2.5,
               error_kw={'elinewidth':0.8,'ecolor':'#444'},
               edgecolor='white', linewidth=0.3)
        ax.set_title(metric, fontsize=9, fontweight='bold', pad=3)
        ymin = max(0.62, min(means)-0.06)
        ax.set_ylim(ymin, min(1.04, max(means)+0.07))
        ax.tick_params(axis='x', length=0, labelsize=8)
        best = np.argmax(means)
        ax.text(best, means[best]+stds[best]+0.007,
                f'{means[best]:.3f}', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color=C['blue'])
        if i==0: ax.set_ylabel('Score', fontsize=9)
        panel_label(ax, chr(97+i), x=-0.22)
    plt.savefig('figures/fig1_model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print("  fig1 saved")

def fig2_roc_calibration(holdout):
    fig, axes = plt.subplots(1,2,figsize=(6.5,3.0)); fig.subplots_adjust(wspace=0.45)
    ax = axes[0]
    fpr,tpr,_ = roc_curve(holdout['y_te'], holdout['y_prob'])
    auc = roc_auc_score(holdout['y_te'], holdout['y_prob'])
    ax.fill_between(fpr, tpr, alpha=0.10, color=C['blue'])
    ax.plot(fpr, tpr, color=C['blue'], lw=1.8,
            label=f'RF AUC={auc:.4f}\n95%CI: {holdout["auc_ci"][0]:.3f}-{holdout["auc_ci"][1]:.3f}')
    ax.plot([0,1],[0,1],'--',color=C['lgrey'],lw=1.0,label='Reference')
    ax.set_xlabel('1-Specificity'); ax.set_ylabel('Sensitivity')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend(loc='lower right', fontsize=7.5)
    ax.set_xlim(-0.01,1.01); ax.set_ylim(-0.01,1.01); ax.set_aspect('equal')
    panel_label(ax,'a')
    ax2 = axes[1]
    pt, pp = calibration_curve(holdout['y_te'], holdout['y_prob'], n_bins=10)
    brier = brier_score_loss(holdout['y_te'], holdout['y_prob'])
    ax2.plot([0,1],[0,1],'--',color=C['lgrey'],lw=1.0,label='Perfect')
    ax2.plot(pp, pt, 's-', color=C['blue'], lw=1.8, ms=4,
             label=f'RF Brier={brier:.4f}')
    ax2.set_xlabel('Mean Predicted Probability'); ax2.set_ylabel('Observed Fraction')
    ax2.set_title('Calibration Curve', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=7.5)
    ax2.set_xlim(-0.01,1.01); ax2.set_ylim(-0.01,1.01); ax2.set_aspect('equal')
    panel_label(ax2,'b')
    plt.savefig('figures/fig2_roc_calibration.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print("  fig2 saved")

def fig_dca(dca):
    fig, ax = plt.subplots(figsize=(5.5,4.0))
    ax.plot(dca['thresholds'], dca['nb_model'], color=C['blue'], lw=2, label='Random Forest')
    ax.plot(dca['thresholds'], np.maximum(dca['nb_all'],0), color=C['orange'],
            lw=1.8, linestyle='--', label='Treat all')
    ax.axhline(0, color=C['lgrey'], lw=1.5, linestyle=':', label='Treat none')
    ax.fill_between(dca['thresholds'], dca['nb_model'],
                    np.maximum(np.maximum(dca['nb_all'],0),0),
                    where=(dca['nb_model']>np.maximum(np.maximum(dca['nb_all'],0),0)),
                    alpha=0.12, color=C['blue'])
    ax.set_xlabel('Threshold Probability'); ax.set_ylabel('Net Benefit')
    ax.set_title('Decision Curve Analysis', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0,0.50); ax.set_ylim(-0.05,0.60)
    panel_label(ax,'a',x=-0.14)
    plt.savefig('figures/fig_dca.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(); print("  fig_dca saved")

def main():
    print("="*60); print("Step 8: Generating Figures"); print("="*60)
    data = load_all()

    if 'cv_results' in data:
        fig1_model_comparison(data['cv_results']['results_primary'])
        fig2_roc_calibration(data['cv_results']['holdout'])

    if 'dca_results' in data:
        fig_dca(data['dca_results'])

    print("\nAll figures saved to figures/")
    print("Step 8 complete.")

if __name__ == '__main__': main()

"""
utils.py — Shared constants, helpers, and style settings.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── RANDOM SEED ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── FEATURE DEFINITIONS ───────────────────────────────────
FEATURES = [
    'Age', 'sex_female', 'Cd4AtStart', 'MostRecentCd4Count',
    'CD4_improvement', 'stage_start_num', 'WeightAtStart',
    'weight_change', 'BMI_start', 'days_to_ART',
    'opp_infection', 'side_effects', 'tb_positive', 'stage_worsened'
]

FEATURES_DISPLAY = [
    'Age', 'Sex (Female)', 'CD4 at ART Start', 'Recent CD4 Count',
    'CD4 Improvement', 'WHO Stage at Start', 'Weight at Start',
    'Weight Change', 'BMI at Start', 'Days to ART',
    'Opp. Infection', 'Side Effects', 'TB Positive', 'Stage Worsened'
]

# NOTE: ART interruption (had_interruption) is intentionally EXCLUDED
# from the primary model to maximise prospective deployability and
# avoid potential temporal confounding. It is included in the
# secondary (full 15-feature) analysis only.

TARGET = 'target'  # Binary: 1 = poor adherence, 0 = non-poor

# ── COLORBLIND-SAFE PALETTE (npj Digital Medicine standard) ──
COLORS = {
    'blue':    '#0072B2',
    'magenta': '#CC79A7',
    'orange':  '#E69F00',
    'sky':     '#56B4E9',
    'green':   '#009E73',
    'grey':    '#555555',
    'lgrey':   '#BBBBBB',
    'dkgrey':  '#222222',
}

# ── FIGURE STYLE ─────────────────────────────────────────
FONT = 'Liberation Sans'

def set_style():
    plt.rcParams.update({
        'font.family': FONT,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.titleweight': 'bold',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
    })

def panel_label(ax, label, x=-0.18, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')

# ── ECONOMIC MODEL PARAMETERS ────────────────────────────
ECON_PARAMS = {
    'base': {
        'art_annual_cost': 1200,       # USD/patient/year (UNAIDS 2023)
        'art_avoidance_pct': 0.28,     # Barnighausen et al. 2012
        'trad_delay_days': 74,         # Observed median in dataset
        'ai_delay_days': 8,            # SmartDaaS target
        'daily_tx_prob': 0.001,        # Granich et al. 2009
        'secondary_infection_cost': 1200,
    },
    'pessimistic': {
        'art_annual_cost': 900,
        'art_avoidance_pct': 0.18,
        'trad_delay_days': 45,
        'ai_delay_days': 14,
        'daily_tx_prob': 0.001,
        'secondary_infection_cost': 800,
    },
    'optimistic': {
        'art_annual_cost': 1500,
        'art_avoidance_pct': 0.35,
        'trad_delay_days': 90,
        'ai_delay_days': 5,
        'daily_tx_prob': 0.001,
        'secondary_infection_cost': 1600,
    }
}

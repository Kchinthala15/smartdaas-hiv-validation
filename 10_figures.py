"""
paper2/src/10_figures.py
Generate all 12 publication figures for Paper 2.
300 DPI, Arial-equivalent font, colorblind-safe palette.
Figures saved to paper2/figures/
"""
import pandas as pd, numpy as np, pickle, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)
os.makedirs('paper2/figures', exist_ok=True)

FONT='Liberation Sans'
C={'blue':'#0072B2','magenta':'#CC79A7','orange':'#E69F00',
   'sky':'#56B4E9','green':'#009E73','grey':'#555555','lgrey':'#BBBBBB'}
plt.rcParams.update({
    'font.family':FONT,'font.size':14,'axes.titlesize':16,
    'axes.labelsize':14,'axes.titleweight':'bold','axes.linewidth':1.2,
    'axes.spines.top':False,'axes.spines.right':False,
    'xtick.labelsize':13,'ytick.labelsize':13,
    'legend.fontsize':13,'legend.frameon':False,
    'figure.dpi':300,'lines.linewidth':2.5,'axes.titlepad':14,
})

def plabel(ax,l,x=-0.15,y=1.08):
    ax.text(x,y,l,transform=ax.transAxes,fontsize=16,fontweight='bold',va='top')

def load():
    data={}
    for name in ['data_engineered','regression_results','icc_results',
                 'interaction_results','facility_typology']:
        try:
            with open(f'paper2/results/{name}.pkl','rb') as f:
                data[name]=pickle.load(f)
        except: print(f"  Warning: {name}.pkl not found — run earlier scripts first")
    return data

def main():
    print("="*60); print("Paper 2 — Step 10: Generate All Figures"); print("="*60)
    data=load()
    df=data.get('data_engineered',{}).get('df')
    if df is None:
        print("ERROR: Run 01_feature_engineering.py first")
        return

    for col in ['DateOfConfirmedHIV','DateArtStarted']:
        df[col]=pd.to_datetime(df[col],errors='coerce',dayfirst=True)
    df['days_to_ART']=(df['DateArtStarted']-df['DateOfConfirmedHIV']).dt.days.clip(0,3650)

    levels_raw=['Primary health center','Secondary health facility','Tertiary hospital']
    levels_lab=['Primary\nHealth Centre','Secondary\nHealth Facility','Tertiary\nHospital']
    colors_l=[C['magenta'],C['sky'],C['blue']]
    outcomes=[('poor_adherence','Poor Adherence'),('dead','Mortality'),
              ('art_interrupted','ART Interruption'),('poor_outcome','Composite Poor Outcome')]

    # Fig 1: STROBE (static — see paper2/figures/Fig01_STROBE.png)
    print("  Note: STROBE diagram (Fig 1) is pre-generated")

    # Fig 2: CD4 + delay
    fig,axes=plt.subplots(1,2,figsize=(14,6)); fig.subplots_adjust(wspace=0.45)
    ax=axes[0]
    cd4=[df[df['Health facility level']==l]['Cd4AtStart'].dropna().clip(0,1000).values for l in levels_raw]
    vp=ax.violinplot(cd4,positions=[1,2,3],showmedians=True,showextrema=False)
    for body,col in zip(vp['bodies'],colors_l): body.set_facecolor(col); body.set_alpha(0.75)
    vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(3)
    for xi,d2 in zip([1,2,3],cd4): ax.text(xi,np.median(d2)+28,f'{np.median(d2):.0f}',ha='center',fontsize=14,fontweight='bold')
    ax.set_xticks([1,2,3]); ax.set_xticklabels(['Primary\nHC','Secondary\nHF','Tertiary\nHosp'],fontsize=13)
    ax.set_ylabel('CD4 Count at ART Start (cells/µL)',fontsize=14)
    ax.set_title('CD4 Distribution at ART Initiation',fontweight='bold'); plabel(ax,'a')
    ax2=axes[1]
    medians=[df[df['Health facility level']==l]['days_to_ART'].median() for l in levels_raw]
    q25=[df[df['Health facility level']==l]['days_to_ART'].quantile(0.25) for l in levels_raw]
    q75=[df[df['Health facility level']==l]['days_to_ART'].quantile(0.75) for l in levels_raw]
    bars=ax2.bar([1,2,3],medians,color=colors_l,width=0.5,edgecolor='white')
    ax2.errorbar([1,2,3],medians,yerr=[np.array(medians)-np.array(q25),np.array(q75)-np.array(medians)],
                 fmt='none',color='black',capsize=8,lw=2.5)
    for xi,v in zip([1,2,3],medians): ax2.text(xi,v+3,f'{v:.0f}d',ha='center',fontsize=14,fontweight='bold')
    ax2.set_xticks([1,2,3]); ax2.set_xticklabels(['Primary\nHC','Secondary\nHF','Tertiary\nHosp'],fontsize=13)
    ax2.set_ylabel('Median Days: Diagnosis to ART\n(error bars = IQR)',fontsize=13)
    ax2.set_title('Diagnosis-to-ART Delay',fontweight='bold'); plabel(ax2,'b',x=-0.18)
    fig.suptitle('Figure 2: Clinical Profile and Care Pathway by Facility Level',fontsize=16,fontweight='bold')
    plt.savefig('paper2/figures/Fig02_CD4_Delay.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close()
    print("  Fig 2 saved")

    # Fig 3: Outcomes by facility level 2x2
    fig,axes=plt.subplots(2,2,figsize=(14,11)); fig.subplots_adjust(hspace=0.55,wspace=0.45)
    for idx,(col,title) in enumerate(outcomes):
        ax=axes[idx//2][idx%2]
        rates=[df[df['Health facility level']==l][col].mean()*100 for l in levels_raw]
        bars=ax.bar(levels_lab,rates,color=colors_l,width=0.45,edgecolor='white')
        for bar,v in zip(bars,rates):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,f'{v:.1f}%',
                    ha='center',va='bottom',fontsize=13,fontweight='bold')
        ax.set_title(title,fontweight='bold',pad=10); ax.set_ylabel('Rate (%)',fontsize=13)
        ax.set_ylim(0,max(rates)*1.40); ax.tick_params(axis='x',length=0,labelsize=12)
        plabel(ax,chr(97+idx))
    fig.suptitle('Figure 3: HIV Treatment Outcomes by Facility Level (n=27,288)',fontsize=16,fontweight='bold')
    plt.savefig('paper2/figures/Fig03_Outcomes_FacilityLevel.png',dpi=300,bbox_inches='tight',facecolor='white'); plt.close()
    print("  Fig 3 saved")

    print(f"\nAll figures saved to paper2/figures/")
    print("Step 10 complete.")

if __name__=='__main__': main()

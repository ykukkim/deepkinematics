"""Create compact manuscript summaries of spatial Bland-Altman results."""

import numpy as np, pandas as pd, sys
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import linregress
from helpers import icc_a1
HERE = Path(__file__).resolve().parent
PLOT_DIR = HERE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
G=pd.read_csv(HERE.parent / "DK04_Gait_Parameter" / "shared_spatial.csv")
MODELS=["BILSTM","ATT","DIFF"]; NAME={"BILSTM":"BiLSTM","ATT":"ATT","DIFF":"DIFF"}; COL={"BILSTM":"#1f77b4","ATT":"#ff7f0e","DIFF":"#2ca02c"}
PAR=[("stepLength","Step length"),("strideLength","Stride length"),("stepWidth","Step width")]
rng=np.random.RandomState(0)
def beta_ci(d,Bn=500):
    subs=d.subject.unique(); vals=[]
    for _ in range(Bn):
        pick=rng.choice(subs,len(subs),replace=True); dd=pd.concat([d[d.subject==p] for p in pick])
        v=linregress(dd.omc.values,dd.pred.values).slope
        if np.isfinite(v): vals.append(v)
    return np.nanpercentile(vals,2.5),np.nanpercentile(vals,97.5)
fig,ax=plt.subplots(3,3,figsize=(11,9))
for ri,(param,plab) in enumerate(PAR):
    for ci,m in enumerate(MODELS):
        a=ax[ri,ci]; d=G[(G.parameter==param)&(G.model==m)].dropna(subset=['omc','pred'])
        o=d.omc.values; p=d.pred.values; mean=(o+p)/2; diff=p-o
        bias=diff.mean(); sd=diff.std(ddof=1)
        a.scatter(mean,diff,s=5,alpha=0.10,color=COL[m])
        lr=linregress(mean,diff); xs=np.array([mean.min(),mean.max()])
        a.plot(xs,lr.intercept+lr.slope*xs,color='k',lw=1.6)
        a.axhline(bias,color=COL[m],lw=1.2); a.axhline(bias+1.96*sd,color='0.5',ls='--',lw=0.9); a.axhline(bias-1.96*sd,color='0.5',ls='--',lw=0.9)
        a.axhline(0,color='k',lw=0.5,alpha=0.4)
        beta=linregress(o,p).slope; blo,bhi=beta_ci(d)
        a.text(0.03,0.03,f"bias {bias:+.1f}, LoA ±{1.96*sd:.1f}\nβ {beta:.2f} [{blo:.2f},{bhi:.2f}]",transform=a.transAxes,fontsize=7.5,va='bottom',
               bbox=dict(boxstyle='round',fc='white',ec='0.7',alpha=0.9))
        if ri==0: a.set_title(NAME[m],fontsize=12)
        if ci==0: a.set_ylabel(f"{plab}\nrecon − OMC (cm)",fontsize=10)
        if ri==2: a.set_xlabel("mean of methods (cm)")
fig.suptitle("Spatial agreement — reconstruction vs OMC at shared events (Bland–Altman; black line = diff-vs-mean fit)\nβ = slope of recon on OMC (1 = observed range preserved, <1 = compressed)",fontsize=11,y=0.997)
fig.tight_layout(rect=[0,0,1,0.965])
fig.savefig(PLOT_DIR / "main_spatial3.pdf",bbox_inches='tight'); fig.savefig(PLOT_DIR / "main_spatial3.png",dpi=125,bbox_inches='tight')
print("saved 3-row BA figure")

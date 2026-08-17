"""Generate the combined kinematic and spatial manuscript result figures."""

import numpy as np, pandas as pd, sys
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, resample_poly, correlate
from scipy.stats import skew, pearsonr
from helpers import cycles_from_signal as segment, cmc_waveform, icc_a1
HERE = Path(__file__).resolve().parent
GAIT_DIR = HERE.parent / "DK04_Gait_Parameter"
DATA_DIR = GAIT_DIR / "data"
PLOT_DIR = HERE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
D=np.load(DATA_DIR / "_omc_gyro_extract.npz")
pos=np.load(DATA_DIR / "posfeet.npz")
subs=["04","09","14","24","51","54","67","81"]; MODELS=["BILSTM","ATT","DIFF"]; JOINTS=["Hip","Knee","Ankle"]
NAME={"BILSTM":"BiLSTM","ATT":"ATT","DIFF":"DIFF"}; COL={"BILSTM":"#1f77b4","ATT":"#ff7f0e","DIFF":"#2ca02c"}
SF=200.0; b,a=butter(4,12/(SF/2),'low')
flex={m:np.load(DATA_DIR / f"flex_{m}.npz") for m in MODELS}
def gyro_ic(sig):
    g=filtfilt(b,a,sig)
    if skew(g)<0: g=-g
    gn=g/np.abs(g).max(); sw,_=find_peaks(gn,height=0.4,distance=int(0.7*SF)); ic=[]
    for k,p in enumerate(sw):
        end=sw[k+1] if k+1<len(sw) else min(len(gn),p+int(1.3*SF)); seg=gn[p:end]
        if len(seg)<int(0.3*SF): continue
        neg=p+int(np.argmin(seg)); w=gn[neg:min(end,neg+int(0.45*SF))]
        if len(w)<3: continue
        ic.append(neg+int(np.argmax(w)))
    return np.array(sorted(set(ic)),int)
def offset(s):
    nat=resample_poly(D[f'{s}|LAJC'][:,1],1,4); gt=pos[f'{s}|gt|LAJC'][:,1]
    a1=(nat-nat.mean())/(nat.std()+1e-9); a2=(gt-gt.mean())/(gt.std()+1e-9)
    c=correlate(a1,a2,mode='full'); return int(np.argmax(np.abs(c))-(len(a2)-1))
def map_ev(ev,lag,n): pf=np.round(ev/4.0).astype(int)-lag; return pf[(pf>=0)&(pf<n)]

# subject-mean curves
EV={}
for s in subs:
    lag=offset(s); n=len(pos[f'{s}|gt|LAJC'])
    EV[s]={'L':map_ev(gyro_ic(D[f'{s}|gL']),lag,n),'R':map_ev(gyro_ic(D[f'{s}|gR']),lag,n)}
curves={}  # curves[(m,j)] -> (gt_curves nx101, hat_curves)
for m in MODELS:
    F=flex[m]
    for j in JOINTS:
        gc=[]; hc=[]
        for s in subs:
            g_c=[]; h_c=[]
            for side in ['L','R']:
                ev=EV[s][side]
                cg=segment(F[f"SonE_{s}|{side}|{j}|gt"],ev+1); ch=segment(F[f"SonE_{s}|{side}|{j}|hat"],ev+1)
                nmin=min(len(cg),len(ch))
                if nmin>0: g_c.append(cg[:nmin]); h_c.append(ch[:nmin])
            if g_c: gc.append(np.vstack(g_c).mean(0)); hc.append(np.vstack(h_c).mean(0))
        curves[(m,j)]=(np.array(gc),np.array(hc))

# ===== FIG 1: kinematic waveforms (3 joints x 3 models) =====
x=np.linspace(0,100,101)
fig,axes=plt.subplots(3,3,figsize=(11,9),sharex=True)
for ci,m in enumerate(MODELS):
    for ri,j in enumerate(JOINTS):
        ax=axes[ri,ci]; gc,hc=curves[(m,j)]
        gm,gs=gc.mean(0),gc.std(0); hm,hs=hc.mean(0),hc.std(0)
        ax.fill_between(x,gm-gs,gm+gs,color='k',alpha=0.12)
        ax.plot(x,gm,'k',lw=2,label='OMC')
        ax.fill_between(x,hm-hs,hm+hs,color=COL[m],alpha=0.15)
        ax.plot(x,hm,color=COL[m],lw=2,label=NAME[m])
        rmse=np.mean([np.sqrt(np.mean((gc[i]-hc[i])**2)) for i in range(len(gc))])
        cmc=cmc_waveform(gc,hc)
        ax.text(0.03,0.97,f"RMSE {rmse:.1f}°\nCMC {cmc:.2f}",transform=ax.transAxes,va='top',fontsize=8.5,
                bbox=dict(boxstyle='round',fc='white',ec='0.7',alpha=0.85))
        if ri==0: ax.set_title(NAME[m],fontsize=12)
        if ci==0: ax.set_ylabel(f"{j}\nangle (°)",fontsize=10)
        if ri==2: ax.set_xlabel("Gait cycle (%)")
        ax.legend(fontsize=7,loc='lower right')
fig.suptitle("Joint kinematic waveforms — OMC vs reconstruction (mean ± SD across subjects, shared gyro events)",fontsize=12,y=0.995)
fig.tight_layout(rect=[0,0,1,0.98])
fig.savefig(PLOT_DIR / "main_kinematics.pdf",bbox_inches='tight'); fig.savefig(PLOT_DIR / "main_kinematics.png",dpi=130,bbox_inches='tight')

# ===== FIG 2: spatial Bland-Altman (3 params x 3 models) =====
spat=pd.read_csv(GAIT_DIR / "shared_spatial.csv")
PAR=[("stepLength","Step length"),("strideLength","Stride length"),("stepWidth","Step width")]
fig2,ax2=plt.subplots(3,3,figsize=(11,9))
for ri,(param,plab) in enumerate(PAR):
    for ci,m in enumerate(MODELS):
        ax=ax2[ri,ci]; d=spat[(spat.parameter==param)&(spat.model==m)].dropna(subset=['omc','pred'])
        mean=(d.omc.values+d.pred.values)/2; diff=d.pred.values-d.omc.values
        bias=diff.mean(); sd=diff.std(ddof=1)
        ax.scatter(mean,diff,s=5,alpha=0.12,color=COL[m])
        ax.axhline(bias,color=COL[m],lw=1.5); ax.axhline(bias+1.96*sd,color='0.5',ls='--',lw=1); ax.axhline(bias-1.96*sd,color='0.5',ls='--',lw=1)
        ax.axhline(0,color='k',lw=0.6,alpha=0.5)
        ax.text(0.03,0.03,f"bias {bias:+.1f}\nLoA ±{1.96*sd:.1f}",transform=ax.transAxes,fontsize=8,va='bottom',
                bbox=dict(boxstyle='round',fc='white',ec='0.7',alpha=0.85))
        if ri==0: ax.set_title(NAME[m],fontsize=12)
        if ci==0: ax.set_ylabel(f"{plab}\nrecon − OMC (cm)",fontsize=10)
        if ri==2: ax.set_xlabel("mean of methods (cm)")
fig2.suptitle("Spatial parameter agreement — reconstruction vs OMC at shared gyro events (Bland–Altman)",fontsize=12,y=0.995)
fig2.tight_layout(rect=[0,0,1,0.98])
fig2.savefig(PLOT_DIR / "main_spatial.pdf",bbox_inches='tight'); fig2.savefig(PLOT_DIR / "main_spatial.png",dpi=130,bbox_inches='tight')
print("main figures saved")

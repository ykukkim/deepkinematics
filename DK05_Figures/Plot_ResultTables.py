"""Render compact table panels from the manuscript result summaries."""

import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
GAIT_DIR = HERE.parent / "DK04_Gait_Parameter"
PLOT_DIR = HERE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
S=pd.read_csv(GAIT_DIR / "spatial_table.csv"); K=pd.read_csv(GAIT_DIR / "shared_kin.csv")
NAME={"BILSTM":"BiLSTM","ATT":"ATT","DIFF":"DIFF"}
fig=plt.figure(figsize=(12,9))
def add_table(ax,title,col,rows,note=None):
    ax.axis('off'); ax.set_title(title,fontsize=12,loc='left',weight='bold',pad=6)
    t=ax.table(cellText=rows,colLabels=col,loc='upper center',cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1,1.45)
    for (r,c),cell in t.get_celld().items():
        if r==0: cell.set_facecolor('#40466e'); cell.set_text_props(color='w',weight='bold')
        elif r%2==0: cell.set_facecolor('#f2f2f5')
    if note: ax.text(0.5,-0.02,note,transform=ax.transAxes,fontsize=8,ha='center',style='italic',va='top')

# Spatial (main) with beta
ax1=fig.add_axes([0.03,0.5,0.94,0.44])
rows=[]
for p in ["Step length","Stride length","Step width","Stride width"]:
    for m in ["BiLSTM","ATT","DIFF"]:
        d=S[(S.parameter==p)&(S.model==m)].iloc[0]
        comp="*" if d.bhi<1 else ""
        rows.append([p if m=="BiLSTM" else "",m,f"{d.bias:+.1f}",f"±{d.loa:.1f}",f"{d.mae:.1f}",f"{d.icc:.2f}",f"{d.beta:.2f} [{d.blo:.2f},{d.bhi:.2f}]{comp}"])
add_table(ax1,"Table A  Spatial parameters — reconstruction vs OMC at shared events (cm)",
          ["Parameter","Model","Bias","95% LoA","MAE","ICC(A,1)","β (recon~OMC)"],rows,
          "β = slope of reconstructed on OMC values: 1.0 preserves the true spread, <1 compresses it (range flattened). * clustered 95% CI < 1.")

# Kinematics (main)
ax2=fig.add_axes([0.03,0.03,0.94,0.4])
rows2=[]
for j in ["Hip","Knee","Ankle"]:
    for m in ["BILSTM","ATT","DIFF"]:
        d=K[(K.model==m)&(K.joint==j)].iloc[0]
        rows2.append([j if m=="BILSTM" else "",NAME[m],f"{d.rmse}",f"{int(d.romGT)} / {int(d.romHat)}",f"{d.r}",f"{d.cmc}"])
add_table(ax2,"Table B  Joint kinematics — segmented at shared events vs OMC",
          ["Joint","Model","RMSE (°)","ROM OMC / recon (°)","Pearson r","CMC"],rows2)
fig.savefig(PLOT_DIR / "main_tables.pdf",bbox_inches='tight'); fig.savefig(PLOT_DIR / "main_tables.png",dpi=125,bbox_inches='tight')
print("main tables saved")

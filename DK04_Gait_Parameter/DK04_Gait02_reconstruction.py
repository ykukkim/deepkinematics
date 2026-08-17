#!/usr/bin/env python3
"""
compute_reconstruction.py  — Reconstruction-quality metrics from model outputs -> CSV.

Reads per-subject model output npz (keys: joint_rot_hat, joint_rot_gt  as (T,J,3,3);
optionally pose_hat, pose_gt as (T,J,3)) and computes, per subject and per
body region (full / upper / lower):
  MPJAE  = mean per-joint geodesic rotation error (deg)   [joint_rot]
  MPJPE  = mean per-joint position error (mm)              [pose, if present]
Writes a tidy CSV: model, subject, group(young/older), region, MPJAE_deg, MPJPE_mm.

Convention-free, fully auditable — no .mat, no hidden pipeline.
"""
import sys, os, glob, argparse, numpy as np

FK_JOINTS = ['Hips','Spine','Spine1','Spine2','Spine3','Neck','Neck1','Head','Head_end0',
 'RightShoulder','RightArm','RightForeArm','RightHand','RightHand_end0','RightHand_end1',
 'LeftShoulder','LeftArm','LeftForeArm','LeftHand','LeftHand_end0','LeftHand_end1',
 'RightUpLeg','RightLeg','RightFoot','RightToeBase','RightToeBase_end0',
 'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase','LeftToeBase_end0']
END = lambda n: n.endswith(('_end0','_end1'))
LOWER = lambda n: any(k in n for k in ('UpLeg','Leg','Foot','Toe')) and not END(n)
UPPER = lambda n: any(k in n for k in ('Shoulder','Arm','Hand','Neck','Head','Spine')) and not END(n)
REGIONS = {
  'full':  [i for i,n in enumerate(FK_JOINTS) if not END(n)],
  'upper': [i for i,n in enumerate(FK_JOINTS) if UPPER(n)],
  'lower': [i for i,n in enumerate(FK_JOINTS) if LOWER(n)],
}
def group_of(subj):   # young = ID 1-50, older = 51-87 (config convention)
    num = int(''.join(c for c in subj if c.isdigit()))
    return 'young' if num <= 50 else 'older'

def geodesic_deg(A, B):
    rel = np.matmul(np.swapaxes(A,-1,-2), B)
    tr = rel[...,0,0]+rel[...,1,1]+rel[...,2,2]
    return np.degrees(np.arccos(np.clip((tr-1)/2, -1.0, 1.0)))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="LABEL=glob, e.g. ATT='/tmp/att_out_*.npz'")
    ap.add_argument("--out", default="reconstruction_metrics.csv")
    ap.add_argument("--perjoint-out", default="reconstruction_perjoint.csv")
    args = ap.parse_args()
    rows, prows = [], []
    for entry in args.inputs:
        label, pattern = entry.split("=", 1)
        for f in sorted(glob.glob(pattern)):
            subj = "".join(c for c in os.path.basename(f) if c.isdigit())
            subj = f"SonE_{subj[-2:]}" if subj else os.path.basename(f)
            d = np.load(f)
            hat = d["joint_rot_hat"]; gt = d["joint_rot_gt"].reshape(hat.shape)
            err = geodesic_deg(hat, gt).mean(0)   # (J,) per-joint mean over time
            for r, idx in REGIONS.items():
                row = {"model": label, "subject": subj, "group": group_of(subj),
                       "region": r, "MPJAE_deg": round(float(err[list(idx)].mean()), 3)}
                if "pose_hat" in d.files and "pose_gt" in d.files:
                    ph = d["pose_hat"]; pg = d["pose_gt"].reshape(ph.shape)
                    pe = np.linalg.norm(ph-pg, axis=-1).mean(0)   # (J,) mm (assume mm)
                    row["MPJPE_mm"] = round(float(pe[list(idx)].mean()), 3)
                rows.append(row)
            for j,n in enumerate(FK_JOINTS):
                if not END(n):
                    prows.append({"model":label,"subject":subj,"joint":n,
                                  "MPJAE_deg":round(float(err[j]),3)})
    import csv
    with open(args.out,"w",newline="") as fo:
        w=csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(args.perjoint_out,"w",newline="") as fo:
        w=csv.DictWriter(fo, fieldnames=list(prows[0].keys())); w.writeheader(); w.writerows(prows)
    # console summary: model x region mean over subjects (and by group)
    import collections
    agg=collections.defaultdict(list)
    for r in rows: agg[(r["model"],r["region"],"All")].append(r["MPJAE_deg"]); agg[(r["model"],r["region"],r["group"])].append(r["MPJAE_deg"])
    print(f"{'model':7}{'region':7}{'group':7}{'MPJAE_deg':>10}")
    for k in sorted(agg):
        print(f"{k[0]:7}{k[1]:7}{k[2]:7}{np.mean(agg[k]):10.2f}")
    print(f"\nwrote {args.out} , {args.perjoint_out}")

if __name__=="__main__": main()

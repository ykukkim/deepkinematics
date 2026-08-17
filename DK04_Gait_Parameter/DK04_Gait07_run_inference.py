#!/usr/bin/env python3
"""
01_run_inference.py — Run a trained DeepKinematics model on the held-out subjects
and save its predictions to .npz (no .mat). One npz per (model, subject) with:
    joint_rot_hat (T,J,3,3), joint_rot_gt (T,J,3,3),
    pose_hat (T,J,3), pose_gt (T,J,3)          [if the model returns pose]
These npz files feed 02_/03_/04_ (reconstruction, kinematics, spatio-temporal).

Runs in YOUR training environment (needs torch + the repo's DK00_Utils on PYTHONPATH).
CPU is fine. BiLSTM/BiRNN must be run here too — its 466 MB checkpoint is why this
step lives on your machine rather than in the cloud sandbox.

Usage:
    python 01_run_inference.py \
        --code /path/to/dl_humanmotion-main \
        --model-dir /path/to/Models/.../ATT-...-FK-acc_gyro \
        --data-dir  /path/to/00_Data/Data \
        --subjects 04 09 14 24 51 54 67 81 --trial Norm_Post \
        --out predictions/ATT
Model dir must contain config.json + test_model.pth (the repo convention).
"""
import argparse, os, sys, types
import numpy as np, torch
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", required=True, help="path to dl_humanmotion-main")
    ap.add_argument("--model-dir", required=True, help="dir with config.json + test_model.pth")
    ap.add_argument("--data-dir", required=True, help="C.DATA_DIR (folder of SonE_XX)")
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--trial", default="Norm_Post")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv-events", default=None,
                    help="optional CSV to also dump the trial's gait events used")
    args = ap.parse_args()

    sys.path.insert(0, args.code)
    # aitviewer is only needed for the interactive viewer; stub it for headless runs
    for m in ["aitviewer","aitviewer.renderables","aitviewer.renderables.spheres",
              "aitviewer.renderables.skeletons","aitviewer.renderables.lines",
              "aitviewer.utils","aitviewer.utils.so3","aitviewer.viewer"]:
        sys.modules.setdefault(m, types.ModuleType(m))

    from DK00_Utils.DK00_UT00_config import CONSTANTS as C, Configuration
    C.DATA_DIR = args.data_dir
    C.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from DK00_Utils.DK00_UT05_helpersEval import load_model, load_eval_data, window_generator

    mc = Configuration.from_json(os.path.join(args.model_dir, "config.json"))
    md = {"config": mc, "checkpoint": os.path.join(args.model_dir, "test_model.pth"),
          "model_dir": args.model_dir}
    net, vrn, mc, _, pre = load_model(md); net.eval()
    if vrn: vrn.eval()
    os.makedirs(args.out, exist_ok=True)

    for subj in args.subjects:
        cfg = types.SimpleNamespace(partition="test_specific", trial=args.trial,
                                    subject=[int(subj)], n_samples=8)
        loader = load_eval_data(cfg, str(subj).zfill(2), mc)
        with torch.no_grad():
            for batch in loader:
                b = pre(batch); out = defaultdict(list); gt = defaultdict(list)
                for chunk in window_generator(b, window_size=mc.window_size):
                    bg = chunk.to_gpu()
                    vo = vrn(bg) if vrn else None
                    mo = net(bg, vo) if mc.m_type in ('rnn','vrnrnn','att','vrnatt','diff','vrndiff') else net(bg)
                    for k in ('joint_rot_hat','pose_hat'):
                        if k in mo: out[k].append(mo[k])
                    gt['joint_rot_gt'].append(bg.joint_rotations)
                    gt['pose_gt'].append(bg.pose)
                save = {}
                jrh = torch.cat(out['joint_rot_hat'],1).squeeze().cpu().numpy()
                save['joint_rot_hat'] = jrh
                save['joint_rot_gt']  = torch.cat(gt['joint_rot_gt'],1).squeeze().cpu().numpy()
                if out.get('pose_hat'):
                    save['pose_hat'] = torch.cat(out['pose_hat'],1).squeeze().cpu().numpy()
                    save['pose_gt']  = torch.cat(gt['pose_gt'],1).squeeze().cpu().numpy()
                fn = os.path.join(args.out, f"pred_{subj}.npz")
                np.savez_compressed(fn, **save)
                print(f"  {subj}: saved {fn}  joint_rot_hat {jrh.shape}")
                break

if __name__ == "__main__":
    main()

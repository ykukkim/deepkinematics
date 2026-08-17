#!/usr/bin/env python3
"""
DK04_Gait08_joint_kinematics.py

Compare lower-limb JOINT KINEMATICS (hip / knee / ankle flexion-extension) of the
deep-learning reconstruction against the Vicon ground truth, over the gait cycle,
for one or several models.

This is the Python port + upgrade of the MATLAB DK04_Gait04_Jointplot_Concatenated.m.
The MATLAB script only drew mean+-SD curves. This script reproduces the SAME
joint-angle definition and additionally computes everything a method-comparison
reviewer (IEEE TIM) expects for a "Reconstruction Quality" section:

  - Per-gait-cycle waveforms (0-100 %) for Vicon and prediction, per side/joint.
  - Waveform error: RMSE, MAE, peak error, range-of-motion (ROM) error, computed
    per subject then summarised as mean +- SD across subjects (subject = unit of
    observation, the statistically correct level).
  - Waveform similarity: Pearson r and CMC (coefficient of multiple correlation).
  - Agreement of discrete features (peak flexion, ROM) across subjects:
    ICC(A,1) = ICC(2,1) and Lin's CCC with 95 % CIs.
  - SPM1D paired t-test (Vicon vs prediction) on the subject-mean curves, with the
    supra-threshold clusters (the % of the gait cycle where the two differ
    significantly). Falls back gracefully if spm1d is not installed.
  - Publication figures: one mean+-SD panel per model (joints x sides) with SPM
    significance shading, plus a combined all-models overlay per joint.
  - Tidy CSVs: a per-(model, side, joint) summary and a long per-cycle table.

Joint-angle definition (identical to DK04_Gait04_Jointplot_Concatenated.m)
-------------------------------------------------------------------------
For a joint rotation matrix R (BVH local rotation, as stored in
GaitSummary.<subj>.<walk>_joint_rot_{gt,hat}.Joint.<JOINT>):
    eul = rotm2eul(R, 'XYZ')                 # intrinsic X-Y-Z, radians
    flexion_deg = -rad2deg(eul[0])           # first (X) angle, sign-flipped
The knee is a clean 1-DOF signal (the Y and Z Euler angles are ~0), confirming
these are anatomical joint angles rather than segment orientations.

Reading the v7.3 .mat needs h5py (`pip install h5py`) OR mat73. If you already
extracted the angle waveforms to .npz with DK04_Gait07_extract_joint_angles.py
(keys "<subj>|<JOINT>|<gt|hat>" -> (T,3) Euler-deg arrays, plus "<subj>|HSleft" /
"<subj>|HSright" event indices), pass --from-npz and the script reads those
directly, needing only numpy/scipy/pandas/matplotlib.

Two-step pipeline:
    DK04_Gait07_extract_joint_angles.py   # big mats -> small angles_<model>.npz
    DK04_Gait08_joint_kinematics.py       # this file: metrics + curves + SPM1D

Usage
-----
  # from the raw GaitSummary mats (one or more models):
  python DK04_Gait08_joint_kinematics.py \
      BiLSTM=GaitSummary_DL.mat ATT=GaitSummary_DL_ATT.mat \
      BIRNN=GaitSummary_DL_BIRNN.mat Diff=GaitSummary_DL_Diff.mat \
      --walk Norm_Post --outdir joint_kinematics_out

  # from pre-extracted angle npz files:
  python DK04_Gait08_joint_kinematics.py BiLSTM=angles_BiLSTM.npz ... --from-npz

  # validate the statistics on synthetic data, no data needed:
  python DK04_Gait08_joint_kinematics.py --selftest

Author: built for Yong Kuk Kim, 2026.
"""

import argparse
import os
import sys
import warnings

import numpy as np


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Joints compared, with the mat "Joint" field names and readable labels.
JOINTS = [
    ("Hip",   {"L": "LHJC", "R": "RHJC"}),
    ("Knee",  {"L": "LKJC", "R": "RKJC"}),
    ("Ankle", {"L": "LAJC", "R": "RAJC"}),
]
SIDES = ["L", "R"]
SIDE_LABEL = {"L": "Left", "R": "Right"}
# Sagittal-plane motion name per joint (hip/knee flex-ext; ankle dorsi/plantar).
MOTION_AXIS = {"Hip": "Flex.–Ext. (°)",
               "Knee": "Flex.–Ext. (°)",
               "Ankle": "Dorsi.–Plantar. (°)"}
N_POINTS = 101          # gait-cycle normalisation grid (0..100 %)
MIN_CYCLE_LEN = 10      # ignore segments shorter than this many frames
MIN_STRIDE_FRAMES = 40  # min plausible stride at 50 Hz (~0.8 s); peak-spacing floor
FLEX_AXIS = 0           # Euler axis used as flexion (X); matches MATLAB eul(:,1)
FLEX_SIGN = -1.0        # MATLAB uses -rad2deg(eul(:,1))


# ----------------------------------------------------------------------------
# Joint-angle maths
# ----------------------------------------------------------------------------

def rotm2eul_xyz(R):
    """MATLAB rotm2eul(R,'XYZ') for a stack of matrices R of shape (...,3,3).

    Intrinsic X-Y-Z. Returns (...,3) angles in radians:
        [ atan2(-R[1,2], R[2,2]),  asin(R[0,2]),  atan2(-R[0,1], R[0,0]) ].
    """
    R = np.asarray(R, dtype=float)
    a = np.arctan2(-R[..., 1, 2], R[..., 2, 2])
    b = np.arcsin(np.clip(R[..., 0, 2], -1.0, 1.0))
    c = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    return np.stack([a, b, c], axis=-1)


def flexion_from_h5_rot(h5_rot):
    """Flexion angle (deg) time-series from an h5py joint-rotation dataset.

    The dataset is stored (3, 3, T) by h5py, which is the transpose of MATLAB's
    R(:,:,t); MATLAB R = h5_rot[:,:,t].T. We therefore build Rm[t] = h5_rot[:,:,t].T
    and apply the same rotm2eul + sign the MATLAB pipeline used.
    """
    arr = np.asarray(h5_rot)                       # (3,3,T)
    Rm = np.transpose(arr, (2, 1, 0))              # (T,3,3), Rm[t] = arr[:,:,t].T
    eul = rotm2eul_xyz(Rm)                          # (T,3) rad
    return FLEX_SIGN * np.rad2deg(eul[:, FLEX_AXIS])


def flexion_from_euler_deg(euler_deg):
    """Flexion (deg) from a pre-extracted (T,3) Euler-in-degrees array."""
    euler_deg = np.asarray(euler_deg, dtype=float)
    return FLEX_SIGN * euler_deg[:, FLEX_AXIS]


# ----------------------------------------------------------------------------
# Gait-cycle segmentation
# ----------------------------------------------------------------------------

def cycles_from_signal(signal, events_1based):
    """Split a 1-D signal into gait cycles and time-normalise each to N_POINTS.

    events_1based : heel-strike frame indices (MATLAB 1-based). A cycle spans
    consecutive events. Returns (n_cycles, N_POINTS) or an empty (0, N_POINTS).
    """
    signal = np.asarray(signal, dtype=float)
    ev = np.asarray(events_1based, dtype=int).ravel() - 1        # -> 0-based
    ev = ev[(ev >= 0) & (ev < len(signal))]
    xn = np.linspace(0.0, 1.0, N_POINTS)
    out = []
    for i in range(len(ev) - 1):
        seg = signal[ev[i]:ev[i + 1] + 1]
        if len(seg) > MIN_CYCLE_LEN:
            xp = np.linspace(0.0, 1.0, len(seg))
            out.append(np.interp(xn, xp, seg))
    if not out:
        return np.empty((0, N_POINTS))
    return np.vstack(out)


# Mat "Joint" field used as the per-side segmentation reference (the knee).
KNEE_NAME = {"L": "LKJC", "R": "RKJC"}


def detect_ic_kinematic(knee_flexion):
    """Detect initial-contact (heel-strike) frames from the knee flexion signal.

    The stored foot-gyroscope events are not reliably synchronised to the
    reconstructed joint arrays, so cycles are instead defined from the knee
    kinematics: each swing-flexion peak is located, and initial contact is taken
    as the terminal-swing extension (the flexion minimum immediately after the
    peak). Returns (ic_frames_0based, stride_period_P).
    """
    from scipy.signal import find_peaks
    knee = np.asarray(knee_flexion, float)
    lo, hi = np.percentile(knee, 5), np.percentile(knee, 95)
    span = hi - lo
    if span <= 1e-6:
        return np.array([], dtype=int), 60
    # Swing-flexion peaks: tall (> loading-response bump) and >= MIN_STRIDE apart,
    # so exactly one per stride. Stride period P = their median spacing (robust;
    # avoids the within-stride double-bump that fools an autocorrelation estimate).
    peaks, _ = find_peaks(knee, height=lo + 0.55 * span,
                          distance=MIN_STRIDE_FRAMES, prominence=0.30 * span)
    if len(peaks) < 3:
        return np.array([], dtype=int), 60
    P = int(np.median(np.diff(peaks)))
    # Initial contact = terminal-swing extension: the flexion minimum after each peak.
    ic = []
    for p in peaks:
        w = knee[p:p + int(0.6 * P)]
        if len(w) > 8:
            ic.append(p + int(np.argmin(w)))
    return np.array(sorted(set(ic)), dtype=int), P


def cycles_from_ic(signal, ic, P):
    """Time-normalise cycles cut at kinematically-detected IC frames (0-based).

    Cycles whose length falls outside [0.6, 1.5] x stride period are discarded.
    """
    signal = np.asarray(signal, float)
    xn = np.linspace(0.0, 1.0, N_POINTS)
    out = []
    for i in range(len(ic) - 1):
        L = ic[i + 1] - ic[i]
        if 0.6 * P < L < 1.5 * P:
            seg = signal[ic[i]:ic[i + 1]]
            xp = np.linspace(0.0, 1.0, len(seg))
            out.append(np.interp(xn, xp, seg))
    if not out:
        return np.empty((0, N_POINTS))
    return np.vstack(out)


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def icc_a1(y):
    """ICC(A,1) = ICC(2,1): two-way random, absolute agreement, single measures.

    y : (n, k) array, n targets x k raters. Returns (icc, ci_low, ci_high).
    (McGraw & Wong 1996; validated against Shrout & Fleiss 1979 in --selftest.)
    """
    from scipy.stats import f as f_dist
    y = np.asarray(y, dtype=float)
    n, k = y.shape
    grand = y.mean()
    row = y.mean(axis=1)
    col = y.mean(axis=0)
    SSR = k * np.sum((row - grand) ** 2)
    SSC = n * np.sum((col - grand) ** 2)
    SST = np.sum((y - grand) ** 2)
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    if denom == 0:
        return np.nan, np.nan, np.nan
    icc = (MSR - MSE) / denom
    try:
        a = (k * icc) / (n * (1 - icc))
        b = 1 + (k * icc * (n - 1)) / (n * (1 - icc))
        v = (a * MSC + b * MSE) ** 2 / (
            (a * MSC) ** 2 / (k - 1) + (b * MSE) ** 2 / ((n - 1) * (k - 1)))
        FL = f_dist.ppf(0.975, n - 1, v)
        FU = f_dist.ppf(0.975, v, n - 1)
        low = (n * (MSR - FL * MSE)) / (FL * (k * MSC + (k * n - k - n) * MSE) + n * MSR)
        high = (n * (FU * MSR - MSE)) / (k * MSC + (k * n - k - n) * MSE + n * FU * MSR)
    except (ZeroDivisionError, FloatingPointError, ValueError):
        low, high = np.nan, np.nan
    return icc, low, high


def lins_ccc(x, y):
    """Lin's Concordance Correlation Coefficient with 95 % CI (Fisher-z)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    mx, my = x.mean(), y.mean()
    sx2 = np.mean((x - mx) ** 2)
    sy2 = np.mean((y - my) ** 2)
    sxy = np.mean((x - mx) * (y - my))
    denom = sx2 + sy2 + (mx - my) ** 2
    if denom == 0:
        return np.nan, np.nan, np.nan
    ccc = 2 * sxy / denom
    try:
        sx, sy = np.sqrt(sx2), np.sqrt(sy2)
        r = sxy / (sx * sy)
        u = (mx - my) / np.sqrt(sx * sy)
        ccc2 = ccc ** 2
        var_z = ((1 - r ** 2) * ccc2 / ((1 - ccc2) * r ** 2)
                 + 2 * ccc ** 3 * (1 - ccc) * u ** 2 / (r * (1 - ccc2) ** 2)
                 - ccc ** 4 * u ** 4 / (2 * r ** 2 * (1 - ccc2) ** 2)) / (n - 2)
        se = np.sqrt(var_z)
        z = np.arctanh(ccc)
        low, high = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    except (ZeroDivisionError, FloatingPointError, ValueError):
        low, high = np.nan, np.nan
    return ccc, low, high


def cmc_waveform(curves_a, curves_b):
    """Coefficient of Multiple Correlation between GT and predicted waveforms.

    curves_a, curves_b : (n_subjects, N_POINTS). CMC (Kadaba et al. 1989)
    measures how closely two methods reproduce the SAME waveform. It is computed
    WITHIN each subject (between the two methods) and then averaged across
    subjects, so between-subject gait differences do not contaminate it. Value in
    [0,1], 1 = identical curves. Returns nan if variance is degenerate.
    """
    a = np.asarray(curves_a, float)
    b = np.asarray(curves_b, float)
    n, T = a.shape
    J = 2                                        # two "methods": GT and prediction
    vals = []
    for i in range(n):
        stack = np.stack([a[i], b[i]], axis=0)   # (2, T)
        frame_mean = stack.mean(axis=0)          # (T,)  mean of the two methods
        grand = stack.mean()
        num = np.sum((stack - frame_mean[None, :]) ** 2) / (T * (J - 1))
        den = np.sum((stack - grand) ** 2) / (T * J - 1)
        if den <= 0:
            continue
        val = 1.0 - num / den
        vals.append(np.sqrt(val) if val > 0 else 0.0)
    return float(np.mean(vals)) if vals else np.nan


def spm_paired(curves_gt, curves_hat, alpha=0.05):
    """SPM1D paired t-test between paired subject-mean curves.

    curves_* : (n_subjects, N_POINTS). Returns a dict with the critical
    threshold, the fraction of the cycle above threshold, and cluster extents.
    Degrades to a per-frame paired t-test with a note if spm1d is unavailable.
    """
    gt = np.asarray(curves_gt, float)
    hat = np.asarray(curves_hat, float)
    n = gt.shape[0]
    result = {"n": n, "method": None, "zstar": np.nan,
              "percent_sig": np.nan, "clusters": []}
    if n < 3:
        result["method"] = "insufficient-n"
        return result
    try:
        import spm1d
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = spm1d.stats.ttest_paired(hat, gt)
            ti = t.inference(alpha, two_tailed=True, interp=True)
        result["method"] = "spm1d"
        result["zstar"] = float(ti.zstar)
        x = np.linspace(0, 100, gt.shape[1])
        clusters = []
        for c in ti.clusters:
            lo, hi = c.endpoints
            clusters.append((float(lo), float(hi)))
        result["clusters"] = clusters
        supra = np.abs(ti.z) >= ti.zstar
        result["percent_sig"] = 100.0 * np.mean(supra)
        result["_z"] = np.asarray(ti.z)
        result["_x"] = x
        return result
    except Exception as e:
        # fallback: uncorrected per-frame paired t (flagged, not for inference)
        from scipy.stats import t as tdist
        d = hat - gt
        sd = d.std(axis=0, ddof=1)
        sd[sd == 0] = np.nan
        tval = d.mean(axis=0) / (sd / np.sqrt(n))
        tcrit = tdist.ppf(1 - alpha / 2, n - 1)
        supra = np.abs(tval) >= tcrit
        result["method"] = f"per-frame-fallback ({type(e).__name__})"
        result["zstar"] = float(tcrit)
        result["percent_sig"] = 100.0 * np.nanmean(supra)
        result["_z"] = tval
        result["_x"] = np.linspace(0, 100, gt.shape[1])
        return result


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_from_mat(path, walk):
    """Load joint flexion waveforms + gait events from a GaitSummary v7.3 mat.

    Returns {subject: {"L": {joint: {"gt":(T,), "hat":(T,)}}, "R": {...},
                        "HSleft": arr, "HSright": arr}}.
    """
    import h5py
    subjects = {}
    with h5py.File(path, "r") as f:
        gs = f["GaitSummary"]
        for s in gs.keys():
            if s.startswith("#"):
                continue
            tr = gs[s]
            gtk, hatk = f"{walk}_joint_rot_gt", f"{walk}_joint_rot_hat"
            rrk = f"{walk}_root_rec_gt"
            if gtk not in tr or hatk not in tr or rrk not in tr:
                continue
            gj, hj = tr[gtk]["Joint"], tr[hatk]["Joint"]
            entry = {"L": {}, "R": {}}
            ok = True
            for _, names in JOINTS:
                for side in SIDES:
                    jn = names[side]
                    if jn not in gj or jn not in hj:
                        ok = False
                        continue
                    entry[side][jn] = {
                        "gt": flexion_from_h5_rot(gj[jn]),
                        "hat": flexion_from_h5_rot(hj[jn]),
                    }
            ge = tr[rrk]["GaitEvents"]
            entry["HSleft"] = np.asarray(ge["HSleftlocs"]).ravel().astype(int)
            entry["HSright"] = np.asarray(ge["HSrightlocs"]).ravel().astype(int)
            if ok:
                subjects[s] = entry
    return subjects


def load_from_npz(path):
    """Load from an angle-npz (keys '<subj>|<JOINT>|<gt|hat>' -> (T,3) Euler deg,
    plus '<subj>|HSleft' / '<subj>|HSright'). Mirrors load_from_mat's output."""
    d = np.load(path)
    subs = sorted({k.split("|")[0] for k in d.files})
    out = {}
    for s in subs:
        entry = {"L": {}, "R": {}}
        for _, names in JOINTS:
            for side in SIDES:
                jn = names[side]
                gk, hk = f"{s}|{jn}|gt", f"{s}|{jn}|hat"
                if gk in d.files and hk in d.files:
                    entry[side][jn] = {
                        "gt": flexion_from_euler_deg(d[gk]),
                        "hat": flexion_from_euler_deg(d[hk]),
                    }
        hl, hr = f"{s}|HSleft", f"{s}|HSright"
        if hl not in d.files or hr not in d.files:
            continue
        entry["HSleft"] = np.asarray(d[hl]).ravel().astype(int)
        entry["HSright"] = np.asarray(d[hr]).ravel().astype(int)
        out[s] = entry
    return out


# ----------------------------------------------------------------------------
# Per-subject waveform assembly
# ----------------------------------------------------------------------------

def subject_mean_curves(subjects, segmentation="kinematic"):
    """For each side/joint, build subject-mean GT and HAT curves.

    segmentation:
      "kinematic" (default) — cut cycles at knee-derived initial contact
        (robust; the stored foot-gyro events are not reliably synchronised to
        the reconstructed joint arrays). The SAME IC frames (from the GT knee)
        segment both GT and prediction for every joint of that side.
      "events" — cut at the stored foot-gyro heel-strike events (legacy;
        kept for comparison/reproducibility).

    Returns curves[(side, joint_label)] = dict with:
        gt   : (n_subj, N_POINTS)   subject-mean GT curves
        hat  : (n_subj, N_POINTS)   subject-mean HAT curves
        subj : list of subject ids kept
        n_cycles : total cycles pooled across subjects
        gt_cyc, hat_cyc : pooled per-cycle arrays (for optional per-cycle stats)
    """
    curves = {}
    for label, names in JOINTS:
        for side in SIDES:
            jn = names[side]
            g_subj, h_subj, kept = [], [], []
            g_all, h_all = [], []
            ev_key = "HSleft" if side == "L" else "HSright"
            knee_name = KNEE_NAME[side]
            for s, entry in subjects.items():
                if jn not in entry[side]:
                    continue
                if segmentation == "kinematic":
                    if knee_name not in entry[side]:
                        continue
                    ic, P = detect_ic_kinematic(entry[side][knee_name]["gt"])
                    if len(ic) < 3:
                        continue
                    gc = cycles_from_ic(entry[side][jn]["gt"], ic, P)
                    hc = cycles_from_ic(entry[side][jn]["hat"], ic, P)
                else:
                    if ev_key not in entry:
                        continue
                    ev = entry[ev_key]
                    gc = cycles_from_signal(entry[side][jn]["gt"], ev)
                    hc = cycles_from_signal(entry[side][jn]["hat"], ev)
                m = min(len(gc), len(hc))
                if m < 1:
                    continue
                gc, hc = gc[:m], hc[:m]
                g_subj.append(gc.mean(axis=0))
                h_subj.append(hc.mean(axis=0))
                g_all.append(gc)
                h_all.append(hc)
                kept.append(s)
            if not kept:
                continue
            curves[(side, label)] = {
                "gt": np.vstack(g_subj),
                "hat": np.vstack(h_subj),
                "subj": kept,
                "n_cycles": int(sum(len(x) for x in g_all)),
                "gt_cyc": np.vstack(g_all),
                "hat_cyc": np.vstack(h_all),
            }
    return curves


# ----------------------------------------------------------------------------
# Metrics per (side, joint)
# ----------------------------------------------------------------------------

def metrics_for(curves_entry):
    gt = curves_entry["gt"]                 # (n_subj, N_POINTS)
    hat = curves_entry["hat"]
    n = gt.shape[0]

    # per-subject waveform errors, then mean +- SD across subjects
    rmse_i = np.sqrt(np.mean((gt - hat) ** 2, axis=1))
    mae_i = np.mean(np.abs(gt - hat), axis=1)
    peak_gt = gt.max(axis=1); peak_hat = hat.max(axis=1)
    rom_gt = gt.max(axis=1) - gt.min(axis=1)
    rom_hat = hat.max(axis=1) - hat.min(axis=1)
    peak_err_i = np.abs(peak_hat - peak_gt)
    rom_err_i = np.abs(rom_hat - rom_gt)
    r_i = np.array([np.corrcoef(gt[i], hat[i])[0, 1] for i in range(n)])

    icc_peak = icc_a1(np.column_stack([peak_gt, peak_hat])) if n >= 3 else (np.nan,)*3
    ccc_peak = lins_ccc(peak_gt, peak_hat)
    icc_rom = icc_a1(np.column_stack([rom_gt, rom_hat])) if n >= 3 else (np.nan,)*3
    ccc_rom = lins_ccc(rom_gt, rom_hat)
    cmc = cmc_waveform(gt, hat)
    spm = spm_paired(gt, hat)

    return {
        "n_subjects": n,
        "n_cycles": curves_entry["n_cycles"],
        "rmse_deg": rmse_i.mean(), "rmse_sd": rmse_i.std(ddof=1) if n > 1 else np.nan,
        "mae_deg": mae_i.mean(), "mae_sd": mae_i.std(ddof=1) if n > 1 else np.nan,
        "peak_gt_deg": peak_gt.mean(), "peak_hat_deg": peak_hat.mean(),
        "peak_err_deg": peak_err_i.mean(),
        "rom_gt_deg": rom_gt.mean(), "rom_hat_deg": rom_hat.mean(),
        "rom_err_deg": rom_err_i.mean(),
        "pearson_r": np.nanmean(r_i), "cmc": cmc,
        "icc_peak": icc_peak[0], "icc_peak_lo": icc_peak[1], "icc_peak_hi": icc_peak[2],
        "ccc_peak": ccc_peak[0],
        "icc_rom": icc_rom[0], "ccc_rom": ccc_rom[0],
        "spm_method": spm["method"], "spm_zstar": spm["zstar"],
        "spm_percent_sig": spm["percent_sig"],
        "spm_clusters": ";".join(f"{lo:.0f}-{hi:.0f}%" for lo, hi in spm["clusters"]),
        "_spm": spm,
    }


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def _meanSD(ax, curves, color, label):
    x = np.linspace(0, 100, curves.shape[1])
    m = curves.mean(axis=0)
    sd = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(m)
    ax.plot(x, m, color=color, lw=2, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.25, linewidth=0)


def plot_model_panel(model, curves, metrics, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt_col, hat_col = (0.0, 0.0, 0.6), (1.0, 0.4, 0.3)
    fig, axes = plt.subplots(len(SIDES), len(JOINTS), figsize=(12, 7),
                             sharex=True)
    for r, side in enumerate(SIDES):
        for c, (label, _) in enumerate(JOINTS):
            ax = axes[r, c]
            key = (side, label)
            if key not in curves:
                ax.set_visible(False); continue
            ce = curves[key]; mt = metrics[key]
            _meanSD(ax, ce["gt"], gt_col, "Vicon")
            _meanSD(ax, ce["hat"], hat_col, model)
            # SPM significance shading
            spm = mt["_spm"]
            for lo, hi in spm.get("clusters", []):
                ax.axvspan(lo, hi, color="grey", alpha=0.18, lw=0)
            ax.set_title(f"{SIDE_LABEL[side]} {label}", fontsize=11)
            ax.grid(alpha=0.3)
            txt = (f"RMSE {mt['rmse_deg']:.1f}°  r {mt['pearson_r']:.2f}\n"
                   f"sig {mt['spm_percent_sig']:.0f}% GC")
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7.5,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.7, lw=0.4))
            ax.set_ylabel(MOTION_AXIS[label])
            if r == len(SIDES) - 1:
                ax.set_xlabel("Gait cycle (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Joint kinematics: {model} vs Vicon  "
                 f"(mean±SD, grey = SPM p<0.05)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_all_models(all_curves, outpath, side="L"):
    """Overlay every model's prediction against the shared Vicon GT, per joint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model_colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    fig, axes = plt.subplots(1, len(JOINTS), figsize=(13, 4), sharex=True)
    x = np.linspace(0, 100, N_POINTS)
    for c, (label, _) in enumerate(JOINTS):
        ax = axes[c]
        key = (side, label)
        # GT from the first model that has it (identical across models)
        gt_plotted = False
        for mi, (model, curves) in enumerate(all_curves.items()):
            if key not in curves:
                continue
            if not gt_plotted:
                _meanSD(ax, curves[key]["gt"], (0, 0, 0), "Vicon")
                gt_plotted = True
            m = curves[key]["hat"].mean(axis=0)
            ax.plot(x, m, color=model_colors[mi % len(model_colors)],
                    lw=1.6, label=model)
        ax.set_title(f"{SIDE_LABEL[side]} {label}", fontsize=11)
        ax.set_xlabel("Gait cycle (%)")
        ax.grid(alpha=0.3)
        ax.set_ylabel(MOTION_AXIS[label])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"All models vs Vicon — {SIDE_LABEL[side]} side "
                 f"(subject-mean curves)", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run(models, walk, outdir, from_npz=False, segmentation="kinematic"):
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    fig_dir = os.path.join(outdir, "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"Segmentation: {segmentation}"
          + ("  (knee-derived initial contact)" if segmentation == "kinematic"
             else "  (stored foot-gyro events)"))

    summary_rows, long_rows = [], []
    all_curves = {}

    for model, path in models:
        print(f"\n[{model}] loading {path} ...")
        subjects = load_from_npz(path) if from_npz else load_from_mat(path, walk)
        print(f"  {len(subjects)} subjects: {list(subjects.keys())}")
        curves = subject_mean_curves(subjects, segmentation=segmentation)
        metrics = {k: metrics_for(v) for k, v in curves.items()}
        all_curves[model] = curves

        for (side, label), mt in metrics.items():
            row = {"model": model, "side": SIDE_LABEL[side], "joint": label}
            row.update({k: v for k, v in mt.items() if not k.startswith("_")})
            summary_rows.append(row)
            ce = curves[(side, label)]
            for i, s in enumerate(ce["subj"]):
                long_rows.append({
                    "model": model, "side": SIDE_LABEL[side], "joint": label,
                    "subject": s,
                    "peak_gt": ce["gt"][i].max(), "peak_hat": ce["hat"][i].max(),
                    "rom_gt": np.ptp(ce["gt"][i]), "rom_hat": np.ptp(ce["hat"][i]),
                    "rmse": np.sqrt(np.mean((ce["gt"][i] - ce["hat"][i]) ** 2)),
                })
        panel = os.path.join(fig_dir, f"JointKinematics_{model}.png")
        plot_model_panel(model, curves, metrics, panel)
        print(f"  wrote {panel}")

    for side in SIDES:
        plot_all_models(all_curves, os.path.join(
            fig_dir, f"AllModels_vs_Vicon_{SIDE_LABEL[side]}.png"), side=side)

    summary = pd.DataFrame(summary_rows)
    long_df = pd.DataFrame(long_rows)
    cols = ["model", "side", "joint", "n_subjects", "n_cycles",
            "rmse_deg", "rmse_sd", "mae_deg", "mae_sd",
            "peak_gt_deg", "peak_hat_deg", "peak_err_deg",
            "rom_gt_deg", "rom_hat_deg", "rom_err_deg",
            "pearson_r", "cmc", "icc_peak", "icc_peak_lo", "icc_peak_hi",
            "ccc_peak", "icc_rom", "ccc_rom",
            "spm_method", "spm_zstar", "spm_percent_sig", "spm_clusters"]
    summary = summary[[c for c in cols if c in summary.columns]]
    sp = os.path.join(outdir, "joint_kinematics_summary.csv")
    lp = os.path.join(outdir, "joint_kinematics_perSubject.csv")
    summary.to_csv(sp, index=False)
    long_df.to_csv(lp, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:.2f}"):
        show = summary[["model", "side", "joint", "n_subjects", "rmse_deg",
                        "mae_deg", "peak_err_deg", "rom_err_deg", "pearson_r",
                        "cmc", "icc_peak", "spm_percent_sig"]]
        print("\n=== Joint-kinematics agreement (degrees) ===")
        print(show.to_string(index=False))
    print(f"\nWrote:\n  {sp}\n  {lp}\n  {fig_dir}/")
    return summary, long_df


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def selftest():
    # 1) ICC against the Shrout & Fleiss (1979) benchmark (ICC(2,1)=0.2898)
    data = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                     [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    icc, lo, hi = icc_a1(data)
    print(f"ICC(A,1) = {icc:.4f} (expected 0.2898)  95% CI [{lo:.3f},{hi:.3f}]")
    assert abs(icc - 0.2898) < 1e-3

    # 2) rotm2eul round-trip: build R from known XYZ angles, recover them
    from scipy.spatial.transform import Rotation as Rot
    ang = np.deg2rad([12.0, -7.0, 25.0])
    R = Rot.from_euler("XYZ", ang).as_matrix()
    rec = rotm2eul_xyz(R)
    print(f"rotm2eul round-trip max err = {np.max(np.abs(rec - ang)):.2e} rad")
    assert np.max(np.abs(rec - ang)) < 1e-9

    # 3) SPM / waveform recovery on synthetic curves with a known constant offset
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, N_POINTS)
    base = 30 * np.sin(2 * np.pi * x)            # a fake flexion curve
    n = 8
    gt = base[None, :] + rng.normal(0, 1.5, (n, N_POINTS))
    hat = gt + 5.0 + rng.normal(0, 1.0, (n, N_POINTS))   # +5 deg bias everywhere
    ce = {"gt": gt, "hat": hat, "n_cycles": n * 10,
          "gt_cyc": gt, "hat_cyc": hat, "subj": list(range(n))}
    mt = metrics_for(ce)
    print(f"Recovered RMSE = {mt['rmse_deg']:.2f} deg (expected ~5.2)")
    print(f"Pearson r = {mt['pearson_r']:.3f}  CMC = {mt['cmc']:.3f}")
    print(f"SPM method = {mt['spm_method']}  sig = {mt['spm_percent_sig']:.0f}% GC "
          f"(expected ~100% for a constant 5 deg offset)")
    assert 4.5 < mt["rmse_deg"] < 6.0
    assert mt["spm_percent_sig"] > 80
    print("Self-test PASSED.")


def parse_model_arg(arg, index):
    if "=" in arg and not os.path.exists(arg.split("=", 1)[0]):
        label, path = arg.split("=", 1)
        return label.strip(), os.path.expanduser(path.strip())
    path = os.path.expanduser(arg)
    base = os.path.splitext(os.path.basename(path))[0]
    for prefix in ("GaitSummary_DL_", "GaitSummary_", "angles_"):
        if base.startswith(prefix) and len(base) > len(prefix):
            return base[len(prefix):], path
    return f"model{index + 1}", path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", nargs="*",
                    help="LABEL=path entries (mat or npz). e.g. BiLSTM=GaitSummary_DL.mat")
    ap.add_argument("--walk", default="Norm_Post", help="walk condition (default Norm_Post)")
    ap.add_argument("--outdir", default="joint_kinematics_out", help="output directory")
    ap.add_argument("--from-npz", action="store_true",
                    help="inputs are pre-extracted angle .npz files, not raw mats")
    ap.add_argument("--segmentation", choices=["kinematic", "events"],
                    default="kinematic",
                    help="gait-cycle segmentation: 'kinematic' (default, knee-derived "
                         "initial contact) or 'events' (stored foot-gyro heel strikes)")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the statistics on synthetic data, then exit")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if not args.mat:
        ap.error("provide one or more LABEL=path inputs (or use --selftest)")
    models = [parse_model_arg(a, i) for i, a in enumerate(args.mat)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        run(models, args.walk, args.outdir, from_npz=args.from_npz,
            segmentation=args.segmentation)


if __name__ == "__main__":
    main()

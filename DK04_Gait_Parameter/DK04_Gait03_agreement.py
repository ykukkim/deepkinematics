#!/usr/bin/env python3
"""
DK04_Gait03_agreement.py

Python port + upgrade of DK04_Gait03_BlandAltmanplot.m

The MATLAB script only draws bias/LoA. This script reproduces the same
Vicon-vs-predicted comparison from GaitSummary_DL.mat and additionally computes
everything a method-comparison reviewer (TIM) expects:

  - Bland-Altman bias and 95% limits of agreement (naive, per-stride)
  - Repeated-measures-corrected LoA (Bland & Altman 1999) for clustered strides
  - Per-subject-mean LoA (fully independent sensitivity analysis)
  - ICC(A,1) = ICC(2,1): two-way, absolute-agreement, single measures, with 95% CI
  - Lin's Concordance Correlation Coefficient (CCC) with 95% CI
  - Per-stride MAE and RMSE
  - Bland-Altman plots (one PNG per parameter)
  - A tidy summary CSV: rows = variant x parameter, plus a long paired CSV

Data structure consumed (exactly as DK04_Gait03_BlandAltmanplot.m accesses it):
  GaitSummary.<subject>.<walk>_root_rec_gt .GaitParameters.<param>   (Vicon, root added)
  GaitSummary.<subject>.<walk>_root_rec_hat.GaitParameters.<param>   (predicted, root added)
  GaitSummary.<subject>.<walk>_pose_gt     .GaitParameters.<param>   (Vicon, no root)
  GaitSummary.<subject>.<walk>_pose_hat    .GaitParameters.<param>   (predicted, no root)
  param in {stepLengthL/R, strideLengthL/R, stepWidthL/R, strideWidthL/R}

Usage:
  python DK04_Gait03_agreement.py GaitSummary_DL.mat
  python DK04_Gait03_agreement.py GaitSummary_DL.mat --walk Norm_Post --outdir agreement_out
  python DK04_Gait03_agreement.py --selftest      # validate stats, no data needed

Reading the v7.3 .mat needs h5py (`pip install h5py`) OR mat73. Everything else
uses only numpy / scipy / pandas / matplotlib.

Author: ported for Yong Kuk Kim, 2026.
"""

import argparse
import os
import sys
import warnings

import numpy as np


# ----------------------------------------------------------------------------
# Statistics (no pingouin dependency; formulas implemented directly)
# ----------------------------------------------------------------------------

def icc_a1(y):
    """ICC(A,1) = ICC(2,1): two-way random, absolute agreement, single measures.

    y : (n, k) array, n targets (rows) x k raters/methods (columns).
    Returns (icc, ci_low, ci_high) with a 95% CI (McGraw & Wong 1996).
    """
    from scipy.stats import f as f_dist

    y = np.asarray(y, dtype=float)
    n, k = y.shape
    grand = y.mean()
    row_means = y.mean(axis=1)
    col_means = y.mean(axis=0)

    SSR = k * np.sum((row_means - grand) ** 2)          # between rows (targets)
    SSC = n * np.sum((col_means - grand) ** 2)          # between cols (methods)
    SST = np.sum((y - grand) ** 2)
    SSE = SST - SSR - SSC

    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    denom = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    if denom == 0:
        return np.nan, np.nan, np.nan
    icc = (MSR - MSE) / denom

    # 95% CI (McGraw & Wong 1996, Table 7, ICC(A,1))
    alpha = 0.05
    try:
        a = (k * icc) / (n * (1 - icc))
        b = 1 + (k * icc * (n - 1)) / (n * (1 - icc))
        v_num = (a * MSC + b * MSE) ** 2
        v_den = (a * MSC) ** 2 / (k - 1) + (b * MSE) ** 2 / ((n - 1) * (k - 1))
        v = v_num / v_den
        FL = f_dist.ppf(1 - alpha / 2, n - 1, v)
        FU = f_dist.ppf(1 - alpha / 2, v, n - 1)
        low = (n * (MSR - FL * MSE)) / (FL * (k * MSC + (k * n - k - n) * MSE) + n * MSR)
        high = (n * (FU * MSR - MSE)) / (k * MSC + (k * n - k - n) * MSE + n * FU * MSR)
    except (ZeroDivisionError, FloatingPointError, ValueError):
        low, high = np.nan, np.nan
    return icc, low, high


def lins_ccc(x, y):
    """Lin's Concordance Correlation Coefficient with 95% CI (Fisher-z, Lin 1989/2000).

    x, y : paired 1-D arrays. Returns (ccc, ci_low, ci_high).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    mx, my = x.mean(), y.mean()
    # population (1/n) moments, as in Lin's definition
    sx2 = np.mean((x - mx) ** 2)
    sy2 = np.mean((y - my) ** 2)
    sxy = np.mean((x - mx) * (y - my))
    denom = sx2 + sy2 + (mx - my) ** 2
    if denom == 0:
        return np.nan, np.nan, np.nan
    ccc = 2 * sxy / denom

    # Asymptotic CI via Fisher's z-transform (Lin 1989, corrected 2000)
    try:
        sx, sy = np.sqrt(sx2), np.sqrt(sy2)
        r = sxy / (sx * sy)               # Pearson
        u = (mx - my) / np.sqrt(sx * sy)
        ccc2 = ccc ** 2
        var_z = (
            (1 - r ** 2) * ccc2 / ((1 - ccc2) * r ** 2)
            + 2 * ccc ** 3 * (1 - ccc) * u ** 2 / (r * (1 - ccc2) ** 2)
            - ccc ** 4 * u ** 4 / (2 * r ** 2 * (1 - ccc2) ** 2)
        ) / (n - 2)
        se_z = np.sqrt(var_z)
        z = np.arctanh(ccc)
        low = np.tanh(z - 1.96 * se_z)
        high = np.tanh(z + 1.96 * se_z)
    except (ZeroDivisionError, FloatingPointError, ValueError):
        low, high = np.nan, np.nan
    return ccc, low, high


def repeated_measures_loa_sd(diff, subject_ids):
    """SD of differences corrected for within-subject clustering (Bland & Altman 1999).

    One-way ANOVA of the differences by subject:
        var(d) = MSW + (MSB - MSW) / n0
    Returns sqrt(var(d)); falls back to plain SD if only one subject.
    """
    diff = np.asarray(diff, dtype=float)
    subject_ids = np.asarray(subject_ids)
    subjects = np.unique(subject_ids)
    m = len(subjects)
    N = len(diff)
    if m < 2:
        return np.std(diff, ddof=1)
    grand = diff.mean()
    SSB = 0.0
    SSW = 0.0
    ni = []
    for s in subjects:
        d = diff[subject_ids == s]
        ni.append(len(d))
        SSB += len(d) * (d.mean() - grand) ** 2
        SSW += np.sum((d - d.mean()) ** 2)
    ni = np.array(ni, dtype=float)
    MSB = SSB / (m - 1)
    MSW = SSW / (N - m) if (N - m) > 0 else 0.0
    n0 = (N - np.sum(ni ** 2) / N) / (m - 1)
    var_d = MSW + (MSB - MSW) / n0 if n0 > 0 else np.var(diff, ddof=1)
    var_d = max(var_d, 0.0)
    return np.sqrt(var_d)


def agreement_stats(gt, pred, subject_ids):
    """Full agreement panel for one (variant, parameter) set of paired values."""
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    subject_ids = np.asarray(subject_ids)

    diff = pred - gt                       # predicted - reference
    mean = (pred + gt) / 2.0
    bias = diff.mean()
    sd = diff.std(ddof=1)

    icc, icc_lo, icc_hi = icc_a1(np.column_stack([gt, pred]))
    ccc, ccc_lo, ccc_hi = lins_ccc(gt, pred)
    sd_rm = repeated_measures_loa_sd(diff, subject_ids)

    # per-subject-mean sensitivity analysis (fully independent)
    subs = np.unique(subject_ids)
    gt_sm = np.array([gt[subject_ids == s].mean() for s in subs])
    pr_sm = np.array([pred[subject_ids == s].mean() for s in subs])
    d_sm = pr_sm - gt_sm
    bias_sm = d_sm.mean()
    sd_sm = d_sm.std(ddof=1) if len(subs) > 1 else np.nan
    if len(subs) >= 3:
        icc_sm, icc_sm_lo, icc_sm_hi = icc_a1(np.column_stack([gt_sm, pr_sm]))
    else:
        icc_sm = icc_sm_lo = icc_sm_hi = np.nan

    return {
        "n_pairs": len(gt),
        "n_subjects": len(subs),
        "bias": bias,
        "sd_diff": sd,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "loa_lower_rm": bias - 1.96 * sd_rm,
        "loa_upper_rm": bias + 1.96 * sd_rm,
        "icc_a1": icc,
        "icc_ci_low": icc_lo,
        "icc_ci_high": icc_hi,
        "ccc": ccc,
        "ccc_ci_low": ccc_lo,
        "ccc_ci_high": ccc_hi,
        "mae": np.mean(np.abs(diff)),
        "rmse": np.sqrt(np.mean(diff ** 2)),
        "bias_subjmean": bias_sm,
        "loa_lower_subjmean": bias_sm - 1.96 * sd_sm,
        "loa_upper_subjmean": bias_sm + 1.96 * sd_sm,
        "icc_a1_subjmean": icc_sm,
        "icc_subjmean_ci_low": icc_sm_lo,
        "icc_subjmean_ci_high": icc_sm_hi,
        # kept for plotting
        "_mean": mean,
        "_diff": diff,
    }


# ----------------------------------------------------------------------------
# v7.3 .mat reading
# ----------------------------------------------------------------------------

def _h5_vector(node):
    arr = np.array(node).ravel()
    return arr.astype(float)


def load_gaitsummary(path):
    """Return nested dict: {subject: {trial: {param: 1d-array}}} from GaitSummary_DL.mat.

    Tries h5py (v7.3), then mat73, then scipy.io.loadmat (<v7.3).
    """
    # --- v7.3 via h5py ---
    try:
        import h5py
    except ImportError:
        h5py = None

    if h5py is not None:
        try:
            with h5py.File(path, "r") as f:
                top = [k for k in f.keys() if not k.startswith("#")]
                if "GaitSummary" in top:
                    root = f["GaitSummary"]
                elif len(top) == 1:
                    root = f[top[0]]
                else:
                    raise KeyError(f"Cannot find GaitSummary; top-level keys={top}")
                out = {}
                for subj in root.keys():
                    subj_grp = root[subj]
                    if not isinstance(subj_grp, h5py.Group):
                        continue
                    out[subj] = {}
                    for trial in subj_grp.keys():
                        tgrp = subj_grp[trial]
                        if not isinstance(tgrp, h5py.Group) or "GaitParameters" not in tgrp:
                            continue
                        gp = tgrp["GaitParameters"]
                        params = {}
                        for p in gp.keys():
                            try:
                                params[p] = _h5_vector(gp[p])
                            except Exception:
                                pass
                        out[subj][trial] = params
                return out
        except OSError:
            pass  # not an HDF5 file -> fall through to older readers

    # --- mat73 ---
    try:
        import mat73
        d = mat73.loadmat(path)
        return _from_plain_dict(d["GaitSummary"])
    except Exception:
        pass

    # --- <v7.3 via scipy ---
    from scipy.io import loadmat
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    return _from_matobj(d["GaitSummary"])


def _from_plain_dict(gs):
    out = {}
    for subj, trials in gs.items():
        out[subj] = {}
        for trial, node in trials.items():
            gp = node.get("GaitParameters", {}) if isinstance(node, dict) else {}
            out[subj][trial] = {k: np.asarray(v, float).ravel() for k, v in gp.items()}
    return out


def _from_matobj(gs):
    out = {}
    for subj in gs._fieldnames:
        subj_obj = getattr(gs, subj)
        out[subj] = {}
        for trial in subj_obj._fieldnames:
            tobj = getattr(subj_obj, trial)
            if not hasattr(tobj, "_fieldnames") or "GaitParameters" not in tobj._fieldnames:
                continue
            gp = getattr(tobj, "GaitParameters")
            out[subj][trial] = {p: np.asarray(getattr(gp, p), float).ravel()
                                for p in gp._fieldnames}
    return out


def inspect_mat(path, walk=None, max_subjects=None):
    """Print a readable tree of GaitSummary_DL.mat: subjects -> trials -> params.

    Shows n strides and min/median/max per parameter, so you can eyeball the
    structure and spot corrupt values (e.g. SonE_54's ~626 cm left length).
    """
    gs = load_gaitsummary(path)
    print(f"\nFile: {path}")
    print(f"Subjects ({len(gs)}): {list(gs.keys())}\n")
    subjects = list(gs.keys())
    if max_subjects:
        subjects = subjects[:max_subjects]
    for subj in subjects:
        trials = gs[subj]
        print(f"== {subj} ==  ({len(trials)} trials)")
        for trial in sorted(trials.keys()):
            if walk and not trial.startswith(walk):
                continue
            params = trials[trial]
            print(f"  {trial}")
            for p in GAIT_PARAMS:
                if p not in params:
                    continue
                v = np.asarray(params[p], float)
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    print(f"      {p:16s} (empty)")
                    continue
                flag = ""
                if _param_family(p) == "length" and (v.max() > 200 or v.min() < 0):
                    flag = "  <-- out-of-range length!"
                elif _param_family(p) == "width" and (v.max() > 40 or v.min() < -40):
                    flag = "  <-- out-of-range width!"
                print(f"      {p:16s} n={len(v):4d}  "
                      f"min={v.min():8.1f}  median={np.median(v):8.1f}  "
                      f"max={v.max():8.1f}{flag}")
        print()


def mad_report(path, walk="Norm_Post", nmad=2.0, target="diff",
               variant="root_rec", physio_bounds=False, max_length=None,
               outcsv=None):
    """Per-participant 2-MAD removal counts, one row per subject.

    For each subject and parameter, the 2-MAD bounds are computed from THAT
    subject's own strides (median +/- nmad*1.4826*MAD), then strides outside are
    counted. target selects the series the rule is applied to:
        "diff" : predicted - reference (Bland-Altman difference; default)
        "ref"  : the Vicon/reference values (physiological cleaning of GT)
        "pred" : the predicted values
    Cells show removed/total. Optionally applies physio bounds first.
    """
    import pandas as pd
    gs = load_gaitsummary(path)
    gt_tpl, hat_tpl = VARIANTS[variant]
    gt_trial, hat_trial = gt_tpl.format(walk=walk), hat_tpl.format(walk=walk)

    rows = []
    for subj, trials in gs.items():
        if gt_trial not in trials or hat_trial not in trials:
            continue
        row = {"subject": subj}
        tot_n = tot_rem = 0
        for p in GAIT_PARAMS:
            if p not in trials[gt_trial] or p not in trials[hat_trial]:
                row[p] = "-"
                continue
            a = np.asarray(trials[gt_trial][p], float)
            b = np.asarray(trials[hat_trial][p], float)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if physio_bounds and len(a):
                a, b, _, _ = apply_physio_bounds(a, b, np.zeros(len(a)), p, max_length)
            series = {"diff": b - a, "ref": a, "pred": b}[target]
            if len(series) < 2:
                row[p] = f"0/{len(series)}"
                tot_n += len(series)
                continue
            med = np.median(series)
            mad = np.median(np.abs(series - med))
            scaled = 1.4826 * mad
            if scaled == 0:                       # all identical -> nothing to trim
                rem = 0
            else:
                lo, hi = med - nmad * scaled, med + nmad * scaled
                rem = int(np.sum((series < lo) | (series > hi)))
            row[p] = f"{rem}/{len(series)}"
            tot_n += len(series)
            tot_rem += rem
        pct = 100.0 * tot_rem / tot_n if tot_n else 0.0
        row["TOTAL"] = f"{tot_rem}/{tot_n} ({pct:.1f}%)"
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n=== Per-participant {nmad}-MAD removal on '{target}' "
          f"({variant}, {walk}{', physio-filtered' if physio_bounds else ''}) ===")
    print(f"    (cells = removed/total strides; MAD computed within each subject)\n")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(df.to_string(index=False))
    grand_rem = sum(int(r["TOTAL"].split("/")[0]) for r in rows)
    grand_n = sum(int(r["TOTAL"].split("/")[1].split(" ")[0]) for r in rows)
    print(f"\n  GRAND TOTAL: {grand_rem}/{grand_n} strides removed "
          f"({100.0 * grand_rem / grand_n if grand_n else 0:.1f}%)")
    if outcsv:
        df.to_csv(outcsv, index=False)
        print(f"  wrote {outcsv}")
    return df


# ----------------------------------------------------------------------------
# Pairing (mirrors DK04_Gait03_BlandAltmanplot.m)
# ----------------------------------------------------------------------------

GAIT_PARAMS = ["stepLengthL", "stepLengthR", "strideLengthL", "strideLengthR",
               "stepWidthL", "stepWidthR", "strideWidthL", "strideWidthR"]

VARIANTS = {
    "root_rec": ("{walk}_root_rec_gt", "{walk}_root_rec_hat"),  # root added (primary)
    "pose":     ("{walk}_pose_gt",     "{walk}_pose_hat"),      # no root
}


def build_pairs(gs, walk, variant, exclude=None):
    """Collect long-format paired rows for one variant across all subjects/params.

    Returns dict param -> (gt_array, pred_array, subject_id_array), plus a
    list of (subject, param, n_gt, n_pred) mismatch warnings.
    Subjects in `exclude` are skipped entirely.
    """
    exclude = set(exclude or [])
    gt_tpl, hat_tpl = VARIANTS[variant]
    gt_trial = gt_tpl.format(walk=walk)
    hat_trial = hat_tpl.format(walk=walk)

    collected = {p: ([], [], []) for p in GAIT_PARAMS}
    mismatches = []

    for subj, trials in gs.items():
        if subj in exclude:
            continue
        if gt_trial not in trials or hat_trial not in trials:
            continue
        gp_gt = trials[gt_trial]
        gp_hat = trials[hat_trial]
        for p in GAIT_PARAMS:
            if p not in gp_gt or p not in gp_hat:
                continue
            a = np.asarray(gp_gt[p], float)
            b = np.asarray(gp_hat[p], float)
            n = min(len(a), len(b))
            if len(a) != len(b):
                mismatches.append((subj, p, len(a), len(b)))
            a, b = a[:n], b[:n]
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            g, pr, sid = collected[p]
            g.extend(a.tolist())
            pr.extend(b.tolist())
            sid.extend([subj] * len(a))

    return collected, mismatches


def subject_diagnostics(gs, walk):
    """Flag subjects whose reference values are implausibly off the cohort.

    For each parameter, compute every subject's median reference (Vicon) value,
    then report subjects whose median is >3x or <1/3 of the cohort median. A
    clean ~10x/100x ratio points to a unit/scale error (fixable); an erratic one
    points to a corrupt trial (drop). Returns the set of flagged subjects.
    """
    gt_trial = VARIANTS["root_rec"][0].format(walk=walk)
    # gather per-subject median per parameter
    per_subj = {}
    for subj, trials in gs.items():
        if gt_trial not in trials:
            continue
        gp = trials[gt_trial]
        per_subj[subj] = {p: np.nanmedian(gp[p]) for p in GAIT_PARAMS
                          if p in gp and len(gp[p])}
    flagged = {}
    for p in GAIT_PARAMS:
        vals = {s: d[p] for s, d in per_subj.items() if p in d and np.isfinite(d[p])}
        if len(vals) < 3:
            continue
        cohort = np.median(list(vals.values()))
        if cohort == 0:
            continue
        for s, v in vals.items():
            ratio = v / cohort
            if ratio > 3 or ratio < 1 / 3:
                flagged.setdefault(s, []).append((p, v, cohort, ratio))
    if flagged:
        print("\n  [diagnostic] subjects with implausible reference values "
              "(median vs cohort median):")
        for s, items in flagged.items():
            worst = max(items, key=lambda t: abs(np.log(t[3])))
            p, v, cohort, ratio = worst
            print(f"    {s}: {p} median={v:.1f} vs cohort={cohort:.1f} "
                  f"({ratio:.1f}x)  -> {'likely unit/scale error' if _near_pow10(ratio) else 'erratic; consider dropping'}")
        print("    Use --exclude <subject> to drop, or fix the unit upstream.\n")
    return set(flagged)


def _near_pow10(ratio):
    """True if ratio is close to a clean power of ten (10x, 100x, 0.1x, ...)."""
    if ratio <= 0:
        return False
    log = np.log10(ratio)
    return abs(log - round(log)) < 0.15 and round(log) != 0


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

# Physiological plausibility bounds (cm), applied to BOTH reference and prediction.
# A pair is dropped if either value falls outside these bounds. Catches the
# SonE_54 left-length leak (~626 cm strides) while leaving all normal gait intact.
PHYSIO_BOUNDS = {
    "length": (0.0, 200.0),   # step / stride length
    "width": (-40.0, 40.0),   # step / stride width
}


def _param_family(param):
    return "width" if "Width" in param else "length"


def apply_physio_bounds(gt, pred, sid, param, max_length=None):
    """Drop pairs where reference OR prediction is physiologically impossible."""
    gt = np.asarray(gt, float)
    pred = np.asarray(pred, float)
    sid = np.asarray(sid)
    lo, hi = PHYSIO_BOUNDS[_param_family(param)]
    if max_length is not None and _param_family(param) == "length":
        hi = max_length
    keep = (gt >= lo) & (gt <= hi) & (pred >= lo) & (pred <= hi)
    return gt[keep], pred[keep], sid[keep], int((~keep).sum())


def drop_outliers(gt, pred, sid, method="sd", nsd=3.0, nmad=2.0):
    """Remove outlier pairs based on the predicted-reference difference.

    method="sd"  : drop pairs whose difference is outside bias +/- nsd*SD
    method="mad" : drop pairs outside median +/- nmad*1.4826*MAD (robust; this is
                   the trim the MATLAB Bland-Altman used, 2-MAD by default)
    method="iqr" : drop pairs whose difference is outside Q1-1.5*IQR .. Q3+1.5*IQR
    Returns (gt, pred, sid, n_removed). Filtering is on the pooled differences
    for one variant x parameter, so the rule is uniform and reportable.
    """
    gt = np.asarray(gt, float)
    pred = np.asarray(pred, float)
    sid = np.asarray(sid)
    diff = pred - gt
    if method == "iqr":
        q1, q3 = np.percentile(diff, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    elif method == "mad":
        med = np.median(diff)
        mad = np.median(np.abs(diff - med))
        scaled = 1.4826 * mad          # ~ SD for normal data
        lo, hi = med - nmad * scaled, med + nmad * scaled
    else:  # sd
        b, s = diff.mean(), diff.std(ddof=1)
        lo, hi = b - nsd * s, b + nsd * s
    keep = (diff >= lo) & (diff <= hi)
    return gt[keep], pred[keep], sid[keep], int((~keep).sum())


def drop_outliers_by_subject(gt, pred, sid, method="mad", nsd=3.0, nmad=2.0):
    """Apply the outlier rule WITHIN each subject, then recombine.

    This trims each participant against their own median/spread, so it removes
    genuine within-subject artefacts without deleting normal between-subject
    variation the way a pooled trim does. Returns (gt, pred, sid, n_removed).
    """
    gt = np.asarray(gt, float)
    pred = np.asarray(pred, float)
    sid = np.asarray(sid)
    keep_g, keep_p, keep_s = [], [], []
    n_removed = 0
    for s in np.unique(sid):
        m = sid == s
        a, b = gt[m], pred[m]
        if m.sum() >= 3:
            a2, b2, _, nr = drop_outliers(a, b, sid[m], method=method,
                                          nsd=nsd, nmad=nmad)
            n_removed += nr
        else:
            a2, b2 = a, b
        keep_g.append(a2)
        keep_p.append(b2)
        keep_s.append(np.full(len(a2), s))
    return (np.concatenate(keep_g), np.concatenate(keep_p),
            np.concatenate(keep_s), n_removed)


def bland_altman_plot(mean, diff, stats, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bias = stats["bias"]
    lo, hi = stats["loa_lower"], stats["loa_upper"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(mean, diff, s=14, alpha=0.5, edgecolor="none")
    ax.axhline(bias, color="k", lw=1.4, label=f"bias {bias:.2f}")
    ax.axhline(hi, color="r", ls="--", lw=1.1, label=f"+1.96 SD {hi:.2f}")
    ax.axhline(lo, color="r", ls="--", lw=1.1, label=f"-1.96 SD {lo:.2f}")
    ax.set_xlabel("Mean of Vicon and predicted")
    ax.set_ylabel("Predicted - Vicon")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    txt = (f"ICC(A,1)={stats['icc_a1']:.3f}  CCC={stats['ccc']:.3f}\n"
           f"MAE={stats['mae']:.2f}  RMSE={stats['rmse']:.2f}  "
           f"n={stats['n_pairs']} ({stats['n_subjects']} subj)")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=7,
            va="bottom", ha="left", bbox=dict(boxstyle="round", fc="w", alpha=0.7))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run(mat_path, walk, outdir, model=None, exclude=None,
        physio_bounds=False, max_length=None,
        remove_outliers=False, outlier_method="sd", nsd=3.0, nmad=2.0,
        outlier_scope="pooled"):
    import pandas as pd

    exclude = set(exclude or [])
    tag = f"[{model}] " if model else ""
    os.makedirs(outdir, exist_ok=True)
    fig_dir = os.path.join(outdir, "Figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"{tag}Loading {mat_path} ...")
    gs = load_gaitsummary(mat_path)
    print(f"  {len(gs)} subjects: {list(gs.keys())[:6]}"
          f"{' ...' if len(gs) > 6 else ''}")

    subject_diagnostics(gs, walk)
    if exclude:
        print(f"  excluding subjects: {sorted(exclude)}")

    summary_rows = []
    long_rows = []

    for variant in VARIANTS:
        collected, mismatches = build_pairs(gs, walk, variant, exclude=exclude)
        if mismatches:
            print(f"  [warn] {variant}: {len(mismatches)} subject/param GT-vs-pred "
                  f"length mismatches (truncated to min). "
                  f"e.g. {mismatches[:3]}")
            print("        -> if counts differ a lot, nearest-event matching is "
                  "needed before pairing.")
        for p in GAIT_PARAMS:
            g, pr, sid = collected[p]
            if len(g) < 3:
                continue
            n_before = len(g)
            n_physio = 0
            n_outlier = 0
            if physio_bounds:
                g, pr, sid, n_physio = apply_physio_bounds(g, pr, sid, p, max_length)
                if n_physio:
                    print(f"  [physio] {variant}/{p}: removed {n_physio} of "
                          f"{n_before} strides outside plausible range")
            if remove_outliers and len(g) >= 3:
                if outlier_scope == "subject":
                    g, pr, sid, n_outlier = drop_outliers_by_subject(
                        g, pr, sid, method=outlier_method, nsd=nsd, nmad=nmad)
                else:
                    g, pr, sid, n_outlier = drop_outliers(
                        g, pr, sid, method=outlier_method, nsd=nsd, nmad=nmad)
                if n_outlier:
                    detail = {"sd": f", {nsd} SD", "mad": f", {nmad} MAD"}.get(outlier_method, "")
                    print(f"  [outlier] {variant}/{p}: removed {n_outlier} strides "
                          f"({outlier_scope} {outlier_method}{detail})")
            if len(g) < 3:
                continue
            n_removed = n_physio + n_outlier
            st = agreement_stats(g, pr, sid)
            row = {"model": model, "variant": variant, "parameter": p,
                   "n_removed": n_removed, "n_physio": n_physio,
                   "n_outlier": n_outlier}
            row.update({k: v for k, v in st.items() if not k.startswith("_")})
            summary_rows.append(row)
            for gg, pp, ss in zip(g, pr, sid):
                long_rows.append({"model": model, "variant": variant, "parameter": p,
                                  "subject": ss, "gt": gg, "pred": pp})
            if variant == "root_rec":
                bland_altman_plot(
                    st["_mean"], st["_diff"], st,
                    f"{walk} - {p} (root_rec)",
                    os.path.join(fig_dir, f"BA_{walk}_{p}_root_rec.png"))

    summary = pd.DataFrame(summary_rows)
    long_df = pd.DataFrame(long_rows)

    cols = ["model", "variant", "parameter", "n_pairs", "n_removed",
            "n_physio", "n_outlier", "n_subjects",
            "bias", "loa_lower", "loa_upper", "loa_lower_rm", "loa_upper_rm",
            "icc_a1", "icc_ci_low", "icc_ci_high",
            "ccc", "ccc_ci_low", "ccc_ci_high", "mae", "rmse",
            "bias_subjmean", "loa_lower_subjmean", "loa_upper_subjmean",
            "icc_a1_subjmean", "icc_subjmean_ci_low", "icc_subjmean_ci_high"]
    summary = summary[[c for c in cols if c in summary.columns]]

    summary_path = os.path.join(outdir, "agreement_summary.csv")
    long_path = os.path.join(outdir, "paired_long.csv")
    summary.to_csv(summary_path, index=False)
    long_df.to_csv(long_path, index=False)

    # readable console table
    with pd.option_context("display.width", 200,
                           "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.3f}"):
        show = summary[["model", "variant", "parameter", "n_pairs", "n_subjects",
                        "bias", "loa_lower", "loa_upper",
                        "icc_a1", "icc_ci_low", "icc_ci_high",
                        "ccc", "mae", "rmse"]]
        print("\n=== Agreement summary (bias/LoA in the parameter's own units) ===")
        print(show.to_string(index=False))

    print(f"\nWrote:\n  {summary_path}\n  {long_path}\n  {fig_dir}/BA_*.png")
    return summary, long_df


# ----------------------------------------------------------------------------
# Self-test (no data needed): validates ICC against the Shrout-Fleiss benchmark
# ----------------------------------------------------------------------------

def selftest():
    # Shrout & Fleiss (1979) classic example, n=6 targets x k=4 judges.
    # Published ICC(2,1) [= ICC(A,1)] = 0.2898.
    data = np.array([
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ], dtype=float)
    icc, lo, hi = icc_a1(data)
    print(f"ICC(A,1) = {icc:.4f}  (expected 0.2898)  95% CI [{lo:.3f}, {hi:.3f}]")
    assert abs(icc - 0.2898) < 1e-3, "ICC formula mismatch!"

    # Injected-bias recovery + CCC sanity
    rng = np.random.default_rng(0)
    x = rng.normal(50, 8, 400)
    y = x + 0.6 + rng.normal(0, 2, 400)          # +0.6 bias
    st = agreement_stats(x, y, np.repeat(np.arange(20), 20))
    print(f"Recovered bias = {st['bias']:.3f}  (injected 0.600)")
    ccc, clo, chi = lins_ccc(x, y)
    print(f"CCC = {ccc:.3f}  95% CI [{clo:.3f}, {chi:.3f}]")
    assert abs(st["bias"] - 0.6) < 0.3
    assert 0.8 < ccc < 1.0
    print("Self-test PASSED.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", nargs="*",
                    help="one or more GaitSummary_DL.mat paths. Label a model with "
                         "LABEL=path (e.g. BiLSTM='/a/GaitSummary_DL.mat'); "
                         "unlabeled paths get a name inferred from the folder.")
    ap.add_argument("--walk", default="Norm_Post", help="walk condition (default Norm_Post)")
    ap.add_argument("--outdir", default="agreement_out", help="output directory")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="SUBJECT",
                    help="subject id(s) to exclude entirely, e.g. --exclude SonE_54")
    ap.add_argument("--physio-bounds", action="store_true",
                    help="drop strides physiologically impossible (length outside "
                         "0-200 cm, width outside -40..40 cm) on ref OR prediction; "
                         "removes the SonE_54 left-length leak while keeping n=8")
    ap.add_argument("--max-length", type=float, default=None,
                    help="override the upper length bound in cm (default 200)")
    ap.add_argument("--remove-outliers", action="store_true",
                    help="drop outlier strides by predicted-reference difference")
    ap.add_argument("--outlier-method", choices=["sd", "mad", "iqr"], default="mad",
                    help="outlier rule: 2-MAD (default, matches MATLAB), N*SD, or 1.5*IQR")
    ap.add_argument("--outlier-scope", choices=["pooled", "subject"], default="pooled",
                    help="compute the outlier rule pooled across subjects (default) "
                         "or WITHIN each participant (recommended; avoids deleting "
                         "normal between-subject variation)")
    ap.add_argument("--nsd", type=float, default=3.0,
                    help="N for the SD rule (default 3.0)")
    ap.add_argument("--nmad", type=float, default=2.0,
                    help="N for the MAD rule (default 2.0, as in the MATLAB pipeline)")
    ap.add_argument("--inspect", action="store_true",
                    help="print the structure/contents of the .mat file(s) and exit")
    ap.add_argument("--mad-report", action="store_true",
                    help="print per-participant N-MAD removal counts and exit")
    ap.add_argument("--mad-target", choices=["diff", "ref", "pred"], default="diff",
                    help="series the per-participant MAD is applied to "
                         "(diff=pred-ref default, ref=Vicon values, pred=predicted)")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the statistics on known data, then exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.inspect:
        if not args.mat:
            ap.error("provide at least one GaitSummary_DL.mat to inspect")
        for arg in args.mat:
            _, path = parse_model_arg(arg, 0)
            inspect_mat(path, walk=None)
        return

    if args.mad_report:
        if not args.mat:
            ap.error("provide at least one GaitSummary_DL.mat for the MAD report")
        os.makedirs(args.outdir, exist_ok=True)
        for arg in args.mat:
            label, path = parse_model_arg(arg, 0)
            outcsv = os.path.join(args.outdir, f"mad_report_{label}.csv")
            mad_report(path, walk=args.walk, nmad=args.nmad, target=args.mad_target,
                       physio_bounds=args.physio_bounds, max_length=args.max_length,
                       outcsv=outcsv)
        return
    if not args.mat:
        ap.error("provide one or more GaitSummary_DL.mat paths  (or use --selftest)")

    models = [parse_model_arg(a, i) for i, a in enumerate(args.mat)]

    import pandas as pd
    all_summaries = []
    all_longs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for label, path in models:
            model_outdir = os.path.join(args.outdir, label) if len(models) > 1 else args.outdir
            summ, long_df = run(path, args.walk, model_outdir, model=label,
                                exclude=args.exclude,
                                physio_bounds=args.physio_bounds, max_length=args.max_length,
                                remove_outliers=args.remove_outliers,
                                outlier_method=args.outlier_method,
                                nsd=args.nsd, nmad=args.nmad,
                                outlier_scope=args.outlier_scope)
            all_summaries.append(summ)
            all_longs.append(long_df)

    if len(all_summaries) > 1:
        os.makedirs(args.outdir, exist_ok=True)
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = os.path.join(args.outdir, "agreement_summary_all_models.csv")
        combined.to_csv(combined_path, index=False)
        # combined per-stride long table (filtering already applied) with model column
        combined_long = pd.concat(all_longs, ignore_index=True)
        combined_long_path = os.path.join(args.outdir, "paired_long_all_models.csv")
        combined_long.to_csv(combined_long_path, index=False)
        print(f"\n=== Combined across {len(all_summaries)} models ===")
        print(f"Wrote {combined_path}")
        print(f"Wrote {combined_long_path}")


def parse_model_arg(arg, index):
    """Parse a 'LABEL=path' or bare 'path' CLI entry into (label, path)."""
    if "=" in arg and not os.path.exists(arg):
        label, path = arg.split("=", 1)
        return label.strip(), os.path.expanduser(path.strip())
    path = os.path.expanduser(arg)
    parts = os.path.normpath(path).split(os.sep)
    # 1) model encoded in the filename, e.g. GaitSummary_DL_ATT.mat -> ATT
    base = os.path.splitext(os.path.basename(path))[0]
    for prefix in ("GaitSummary_DL_", "GaitSummary_"):
        if base.startswith(prefix) and len(base) > len(prefix):
            return base[len(prefix):], path
    # 2) a 'Full_*' folder in the path, e.g. Full_DIFF -> DIFF
    for comp in parts:
        if comp.startswith("Full_"):
            return comp.replace("Full_", ""), path
    # 3) fall back to the parent folder name
    parent = parts[-2] if len(parts) >= 2 else f"model{index + 1}"
    return parent, path


if __name__ == "__main__":
    main()

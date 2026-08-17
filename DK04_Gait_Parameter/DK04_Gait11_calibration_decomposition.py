#!/usr/bin/env python3
"""
DK04_Gait11_calibration_decomposition.py

Between- vs within-participant calibration of the spatial gait parameters
(reconstruction vs OMC at the shared foot-gyroscope initial contacts).

Why this script exists
----------------------
The pooled per-stride ICC and pooled calibration slope reported so far mix two
different questions:

  (1) BETWEEN participants — does the reconstruction distinguish people with
      shorter vs longer average steps?
  (2) WITHIN participants  — does it follow stride-to-stride deviations around
      each person's own average?

This script separates them. For participant i and stride j, with OMC value
x_ij and reconstructed value y_ij:

  Between-participant calibration (n = 8 participant means):
      ybar_i = alpha_b + beta_between * xbar_i + e_i
      -> bias, MAE, beta_between [participant-bootstrap 95% CI],
         ICC(A,1) on the means (DESCRIPTIVE ONLY, n = 8).

  Within-participant calibration (participant means removed):
      (y_ij - ybar_i) = beta_within * (x_ij - xbar_i) + e_ij
      -> beta_within [participant-cluster bootstrap 95% CI] and the RMSE of the
         centred errors  (y_ij - ybar_i) - (x_ij - xbar_i), i.e. how accurately
         stride-to-stride variation is recovered.

  IMPORTANT: mean-centring forces the average within-participant error to zero,
  so "no bias within participants" is guaranteed mathematically and is NOT a
  finding. The defensible statement is: participant-specific offsets were
  present (between-level bias), and stride-to-stride deviations were / were not
  preserved depending on beta_within and the centred-error RMSE.

  Side-centred robustness level: step parameters alternate between left and
  right contacts. Centring by participant leaves stable left/right asymmetry
  inside the "within" signal. The script therefore also reports beta_within
  after centring by participant x side (beta_within_side), which isolates
  within-side step-to-step fluctuation. Comparing the two slopes distinguishes
  preservation of asymmetry from preservation of fluctuations about each
  side's habitual value.

  Absolute interchangeability is summarised separately by a repeated-measures
  Bland-Altman analysis: limits of agreement use the total SD of the
  differences from a one-way variance-components decomposition
  (between-participant + within-participant components; Bland & Altman 2007),
  NOT the naive SD of the pooled differences.

Confidence intervals resample PARTICIPANTS (cluster bootstrap), never
individual strides.

Data and provenance
-------------------
Reads the same inputs as the manuscript spatial pipeline and rebuilds the
identical per-stride table (same gyro event detector, same event->frame
mapping, same step/stride definitions):

  data/_omc_gyro_extract.npz   raw foot gyro ({s}|gL/gR, 200 Hz) + native joint centres
  data/posfeet.npz             pelvis-centred OMC and reconstructed positions (50 Hz)

The default samples ankle-centre positions at the initial-contact frame
(`--window 0`), matching the spatial definition in the manuscript. A non-zero
window is retained only as an explicit sensitivity option.

Outputs (in --outdir, default ./calibration_out)
------------------------------------------------
  calibration_summary.csv          one row per model x parameter, all measures
  calibration_perparticipant.csv   per-participant means, slopes, centred SDs
  perstride_pairs.csv              audit copy of the per-stride (omc, pred) pairs
  calibration_table.tex            LaTeX (booktabs) rows ready for the manuscript
  calibration_<parameter>.pdf/png  paired between/within calibration figure (main text)
  rmBA_appendix.pdf/png            3x3 Bland-Altman panels with repeated-measures LoA (appendix)

Usage
-----
  python DK04_Gait11_calibration_decomposition.py                 # instantaneous contact frame
  python DK04_Gait11_calibration_decomposition.py --window 3      # optional +/-3-frame average
  python DK04_Gait11_calibration_decomposition.py --selftest      # validate statistics, no data needed

Requires: numpy, scipy, pandas, matplotlib.
Author: built for Yong Kuk Kim, 2026.
"""

import argparse
import os
import sys
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Configuration (identical conventions to the manuscript spatial pipeline)
# ---------------------------------------------------------------------------
SUBS = ["04", "09", "14", "24", "51", "54", "67", "81"]
MODELS = ["BILSTM", "ATT", "DIFF"]
NAME = {"BILSTM": "BiLSTM", "ATT": "ATT", "DIFF": "DIFF"}
COL = {"BILSTM": "#1f77b4", "ATT": "#ff7f0e", "DIFF": "#2ca02c"}
PARAMS = ["stepLength", "strideLength", "stepWidth"]
PLAB = {"stepLength": "Step length", "strideLength": "Stride length",
        "stepWidth": "Step width"}
SF = 200.0     # native gyro / OMC rate
SF50 = 50.0    # reconstruction rate


def grp(s):
    return "young" if int(s) <= 50 else "older"


# ---------------------------------------------------------------------------
# Per-cycle table builder used by the manuscript spatial analysis
# ---------------------------------------------------------------------------

def build_perstride(data_dir, window):
    """Rebuild the per-stride (omc, pred) pairs at the shared gyro events.

    Identical to the pipeline that produced the manuscript agreement table:
    same detector, same 200->50 Hz mapping, same step/stride definitions.
    `window` = frames averaged on each side of the contact frame (3 -> t-3..t+3,
    +/-60 ms; 0 -> instantaneous).
    """
    import pandas as pd
    from scipy.signal import butter, filtfilt, find_peaks, resample_poly, correlate
    from scipy.stats import skew

    D = np.load(os.path.join(data_dir, "_omc_gyro_extract.npz"))
    pos = np.load(os.path.join(data_dir, "posfeet.npz"))
    b, a = butter(4, 12 / (SF / 2), "low")

    def gyro_ic(sig):
        g = filtfilt(b, a, sig)
        if skew(g) < 0:
            g = -g
        gn = g / np.abs(g).max()
        sw, _ = find_peaks(gn, height=0.4, distance=int(0.7 * SF))
        ic = []
        for k, p in enumerate(sw):
            end = sw[k + 1] if k + 1 < len(sw) else min(len(gn), p + int(1.3 * SF))
            seg = gn[p:end]
            if len(seg) < int(0.3 * SF):
                continue
            neg = p + int(np.argmin(seg))
            w = gn[neg:min(end, neg + int(0.45 * SF))]
            if len(w) < 3:
                continue
            ic.append(neg + int(np.argmax(w)))
        return np.array(sorted(set(ic)), int)

    def offset(s):
        nat = resample_poly(D[f"{s}|LAJC"][:, 1], 1, 4)
        gt = pos[f"{s}|gt|LAJC"][:, 1]
        a1 = (nat - nat.mean()) / (nat.std() + 1e-9)
        a2 = (gt - gt.mean()) / (gt.std() + 1e-9)
        c = correlate(a1, a2, mode="full")
        return int(np.argmax(np.abs(c)) - (len(a2) - 1))

    def map_ev(ev_native, lag, n):
        pf = np.round(ev_native / 4.0).astype(int) - lag
        return pf[(pf >= 0) & (pf < n)]

    def sep(P, t):
        if window <= 0:
            return P[t]
        lo = max(0, t - window)
        hi = min(len(P), t + window + 1)
        return P[lo:hi].mean(0)

    def steps(subj, posvar, hl, hr):
        LA = pos[f"{subj}|{posvar}|LAJC"]
        RA = pos[f"{subj}|{posvar}|RAJC"]
        rows = []
        for t in hr:
            la, ra = sep(LA, t), sep(RA, t)
            rows.append(["R", int(t), 0.1 * abs(ra[1] - la[1]), 0.1 * abs(ra[0] - la[0])])
        for t in hl:
            la, ra = sep(LA, t), sep(RA, t)
            rows.append(["L", int(t), 0.1 * abs(la[1] - ra[1]), 0.1 * abs(la[0] - ra[0])])
        rows.sort(key=lambda r: r[1])
        return rows

    def strides_idx(rows):
        st = []
        for i in range(1, len(rows)):
            if rows[i][0] != rows[i - 1][0]:
                st.append((rows[i - 1][2] + rows[i][2],   # stride length
                           rows[i][0]))                   # ending-contact side
        return st

    out = []
    n_events = {}
    for s in SUBS:
        lag = offset(s)
        n = len(pos[f"{s}|gt|LAJC"])
        gl = map_ev(gyro_ic(D[f"{s}|gL"]), lag, n)
        gr = map_ev(gyro_ic(D[f"{s}|gR"]), lag, n)
        n_events[s] = (len(gl), len(gr))
        omc = steps(s, "gt", gl, gr)
        omcS = strides_idx(omc)
        for m in MODELS:
            mod = steps(s, m + "|hat", gl, gr)
            modS = strides_idx(mod)
            for i in range(min(len(omc), len(mod))):
                out.append(dict(subject=s, group=grp(s), model=m, side=omc[i][0],
                                parameter="stepLength", omc=omc[i][2], pred=mod[i][2]))
                out.append(dict(subject=s, group=grp(s), model=m, side=omc[i][0],
                                parameter="stepWidth", omc=omc[i][3], pred=mod[i][3]))
            for k in range(min(len(omcS), len(modS))):
                out.append(dict(subject=s, group=grp(s), model=m, side=omcS[k][1],
                                parameter="strideLength", omc=omcS[k][0], pred=modS[k][0]))
    print("shared gyro events per subject (L,R):", n_events)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def icc_a1(y):
    """ICC(A,1) = ICC(2,1), two-way random, absolute agreement, single measures.
    y: (n, 2). Returns (icc, lo, hi). McGraw & Wong 1996; validated against the
    Shrout & Fleiss (1979) benchmark in --selftest."""
    from scipy.stats import f as f_dist
    y = np.asarray(y, float)
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
        a_ = (k * icc) / (n * (1 - icc))
        b_ = 1 + (k * icc * (n - 1)) / (n * (1 - icc))
        v = (a_ * MSC + b_ * MSE) ** 2 / (
            (a_ * MSC) ** 2 / (k - 1) + (b_ * MSE) ** 2 / ((n - 1) * (k - 1)))
        FL = f_dist.ppf(0.975, n - 1, v)
        FU = f_dist.ppf(0.975, v, n - 1)
        lo = (n * (MSR - FL * MSE)) / (FL * (k * MSC + (k * n - k - n) * MSE) + n * MSR)
        hi = (n * (FU * MSR - MSE)) / (k * MSC + (k * n - k - n) * MSE + n * FU * MSR)
    except (ZeroDivisionError, FloatingPointError, ValueError):
        lo, hi = np.nan, np.nan
    return icc, lo, hi


def rm_loa(diff, subj):
    """Repeated-measures 95% limits of agreement (Bland & Altman 2007).

    One-way variance components on the differences with participant as the
    random factor:  total SD^2 = sigma_between^2 + sigma_within^2, where
    sigma_between^2 = (MSB - MSW) / n0 and n0 is the unbalanced-design average
    cluster size. Returns dict(bias, sd_total, lo, hi, sigma_b, sigma_w).
    """
    diff = np.asarray(diff, float)
    subj = np.asarray(subj)
    ids = np.unique(subj)
    a = len(ids)
    N = len(diff)
    dbar = diff.mean()
    if a < 2 or N - a < 1:
        sd = diff.std(ddof=1)
        return dict(bias=dbar, sd_total=sd, lo=dbar - 1.96 * sd, hi=dbar + 1.96 * sd,
                    sigma_b=np.nan, sigma_w=np.nan)
    ni = np.array([np.sum(subj == i) for i in ids], float)
    di = np.array([diff[subj == i].mean() for i in ids])
    MSB = np.sum(ni * (di - dbar) ** 2) / (a - 1)
    MSW = sum(((diff[subj == i] - di[k]) ** 2).sum() for k, i in enumerate(ids)) / (N - a)
    n0 = (N - (ni ** 2).sum() / N) / (a - 1)
    sigma_b2 = max(0.0, (MSB - MSW) / n0)
    sd_total = np.sqrt(sigma_b2 + MSW)
    return dict(bias=dbar, sd_total=sd_total,
                lo=dbar - 1.96 * sd_total, hi=dbar + 1.96 * sd_total,
                sigma_b=np.sqrt(sigma_b2), sigma_w=np.sqrt(MSW))


def _ols_slope(x, y):
    """OLS slope (with intercept) of y on x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    vx = np.var(x)
    if vx <= 0:
        return np.nan
    return np.cov(x, y, bias=True)[0, 1] / vx


def _centred(d, by=("subject",)):
    """Mean-centred columns xc, yc; centring groups given by `by`
    (("subject",) = participant level; ("subject","side") = participant x side)."""
    keys = list(by)
    xm = d.groupby(keys)["omc"].transform("mean")
    ym = d.groupby(keys)["pred"].transform("mean")
    return (d["omc"] - xm).values, (d["pred"] - ym).values


def _beta_within(xc, yc):
    """Fixed-effects (through-origin on centred data) slope."""
    sxx = np.sum(xc * xc)
    if sxx <= 0:
        return np.nan
    return float(np.sum(xc * yc) / sxx)


def cluster_boot(d, stat_fn, B, rng):
    """Percentile 95% CI of stat_fn over participant-cluster bootstrap resamples.

    stat_fn receives a dataframe (participants resampled with replacement;
    centring, means, etc. are recomputed inside stat_fn per resample)."""
    import pandas as pd
    ids = d["subject"].unique()
    groups = {s: d[d["subject"] == s] for s in ids}
    vals = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids), replace=True)
        dd = pd.concat([groups[p].assign(subject=f"{p}#{k}")   # unique id per draw
                        for k, p in enumerate(pick)], ignore_index=True)
        v = stat_fn(dd)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def analyse(d, B=2000, seed=0):
    """All measures for one (model, parameter) long dataframe d[subject, omc, pred]."""
    import pandas as pd
    rng = np.random.RandomState(seed)
    res = {}

    # ---- pooled / absolute agreement -------------------------------------
    diff = (d["pred"] - d["omc"]).values
    res["n_strides"] = len(d)
    res["n_participants"] = d["subject"].nunique()
    res["bias"] = float(diff.mean())
    res["mae"] = float(np.abs(diff).mean())
    res["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    # Participant-level MAE gives each independent participant equal weight.
    # Its sample SD (ddof=1) is suitable for reporting mean +/- SD across the
    # held-out participants; it is distinct from the pooled per-stride MAE.
    participant_mae = d.assign(abs_error=np.abs(diff)).groupby("subject")["abs_error"].mean()
    res["mae_participant_mean"] = float(participant_mae.mean())
    res["mae_participant_sd"] = float(participant_mae.std(ddof=1))
    L = rm_loa(diff, d["subject"].values)
    res["rm_loa_lo"], res["rm_loa_hi"] = L["lo"], L["hi"]
    res["rm_sd_total"], res["rm_sigma_b"], res["rm_sigma_w"] = (
        L["sd_total"], L["sigma_b"], L["sigma_w"])
    res["naive_loa_halfwidth"] = 1.96 * float(diff.std(ddof=1))  # for comparison only
    # Pooled cycle-level ICC is reported with a participant-clustered bootstrap
    # CI for comparison with conventional agreement studies. It combines
    # between-participant, between-foot, and within-foot sources of variation;
    # the decomposed slopes below are therefore required for interpretation.
    res["pooled_icc_ref"] = icc_a1(np.c_[d["omc"].values, d["pred"].values])[0]

    def _icc_pooled(dd):
        return icc_a1(np.c_[dd["omc"].values, dd["pred"].values])[0]

    rng_icc = np.random.RandomState(seed + 7919)
    res["pooled_icc_lo"], res["pooled_icc_hi"] = cluster_boot(
        d, _icc_pooled, B, rng_icc)
    res["pooled_beta_ref"] = _ols_slope(d["omc"].values, d["pred"].values)

    # ---- between-participant level ---------------------------------------
    m = d.groupby("subject")[["omc", "pred"]].mean()
    res["bias_between"] = float((m["pred"] - m["omc"]).mean())
    res["mae_between"] = float((m["pred"] - m["omc"]).abs().mean())
    res["beta_between"] = _ols_slope(m["omc"].values, m["pred"].values)

    def _bt(dd):
        mm = dd.groupby("subject")[["omc", "pred"]].mean()
        return _ols_slope(mm["omc"].values, mm["pred"].values)
    res["beta_between_lo"], res["beta_between_hi"] = cluster_boot(d, _bt, B, rng)

    icc_bt = icc_a1(np.c_[m["omc"].values, m["pred"].values])
    res["icc_between_desc"] = icc_bt[0]        # descriptive only, n = 8
    res["icc_between_desc_lo"], res["icc_between_desc_hi"] = icc_bt[1], icc_bt[2]

    # ---- within-participant level ----------------------------------------
    xc, yc = _centred(d)
    res["beta_within"] = _beta_within(xc, yc)

    def _wi(dd):
        xcc, ycc = _centred(dd)
        return _beta_within(xcc, ycc)
    res["beta_within_lo"], res["beta_within_hi"] = cluster_boot(d, _wi, B, rng)

    cent_err = yc - xc
    res["centred_bias_check"] = float(cent_err.mean())   # == 0 by construction
    res["centred_rmse"] = float(np.sqrt(np.mean(cent_err ** 2)))

    def _ce(dd):
        xcc, ycc = _centred(dd)
        return float(np.sqrt(np.mean((ycc - xcc) ** 2)))
    res["centred_rmse_lo"], res["centred_rmse_hi"] = cluster_boot(d, _ce, B, rng)

    # within-participant SD of OMC (context: the variation to be recovered)
    res["sd_within_omc"] = float(np.sqrt(np.mean(xc ** 2)))
    res["sd_within_pred"] = float(np.sqrt(np.mean(yc ** 2)))

    # ---- within-participant, SIDE-centred (robustness) -------------------
    # Removes each participant's left and right means separately. This isolates
    # within-side step-to-step fluctuations from stable left/right asymmetry.
    if "side" in d.columns:
        xs, ys = _centred(d, by=("subject", "side"))
        res["beta_within_side"] = _beta_within(xs, ys)

        def _wis(dd):
            xcc, ycc = _centred(dd, by=("subject", "side"))
            return _beta_within(xcc, ycc)
        res["beta_within_side_lo"], res["beta_within_side_hi"] = cluster_boot(d, _wis, B, rng)

        res["centred_rmse_side"] = float(np.sqrt(np.mean((ys - xs) ** 2)))

        def _ces(dd):
            xcc, ycc = _centred(dd, by=("subject", "side"))
            return float(np.sqrt(np.mean((ycc - xcc) ** 2)))
        res["centred_rmse_side_lo"], res["centred_rmse_side_hi"] = cluster_boot(d, _ces, B, rng)
        res["sd_within_omc_side"] = float(np.sqrt(np.mean(xs ** 2)))
    return res


def per_participant_table(d_mp):
    """Per-participant descriptive rows for one (model, parameter)."""
    rows = []
    for s, ds in d_mp.groupby("subject"):
        x = ds["omc"].values
        y = ds["pred"].values
        xc, yc = x - x.mean(), y - y.mean()
        rows.append(dict(subject=s, group=grp(s), n_strides=len(ds),
                         omc_mean=x.mean(), pred_mean=y.mean(),
                         offset=y.mean() - x.mean(),
                         mae=float(np.abs(y - x).mean()),
                         sd_omc_within=x.std(ddof=1), sd_pred_within=y.std(ddof=1),
                         beta_within_i=_beta_within(xc, yc),
                         centred_rmse_i=float(np.sqrt(np.mean((yc - xc) ** 2)))))
    return rows


def leave_one_participant_out(d):
    """Calibration slopes after omitting each participant in turn."""
    rows = []
    for omitted in sorted(d["subject"].unique()):
        dd = d[d["subject"] != omitted]
        means = dd.groupby("subject")[["omc", "pred"]].mean()
        xc, yc = _centred(dd)
        row = {
            "omitted_subject": omitted,
            "beta_between": _ols_slope(means["omc"].values, means["pred"].values),
            "beta_within": _beta_within(xc, yc),
        }
        if "side" in dd.columns:
            xs, ys = _centred(dd, by=("subject", "side"))
            row["beta_within_side"] = _beta_within(xs, ys)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_calibration(d, S, param, outdir):
    """Main-text replacement figure: between (top) and within (bottom) x models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(11.5, 7.2))
    for ci, mdl in enumerate(MODELS):
        dd = d[(d["model"] == mdl) & (d["parameter"] == param)]
        r = S[(mdl, param)]
        m = dd.groupby("subject")[["omc", "pred"]].mean().reset_index()

        # --- between: participant means -------------------------------------------
        a1 = ax[0, ci]
        lim = [min(m["omc"].min(), m["pred"].min()), max(m["omc"].max(), m["pred"].max())]
        pad = 0.08 * (lim[1] - lim[0] + 1e-9)
        lim = [lim[0] - pad, lim[1] + pad]
        a1.plot(lim, lim, "k--", lw=0.9, alpha=0.6, label="identity")
        for _, row in m.iterrows():
            mk = "o" if grp(row["subject"]) == "young" else "s"
            a1.scatter(row["omc"], row["pred"], s=46, marker=mk,
                       facecolor=COL[mdl], edgecolor="k", linewidth=0.5, zorder=3)
        bb = r["beta_between"]
        if np.isfinite(bb):
            x0 = np.array(lim)
            ym = m["pred"].mean() + bb * (x0 - m["omc"].mean())
            a1.plot(x0, ym, color=COL[mdl], lw=1.8)
        a1.set_xlim(lim), a1.set_ylim(lim)
        a1.set_title(NAME[mdl], fontsize=12)
        a1.text(0.03, 0.97,
                f"$\\beta_B$ {bb:.2f} [{r['beta_between_lo']:.2f},{r['beta_between_hi']:.2f}]",
                transform=a1.transAxes, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        if ci == 0:
            a1.set_ylabel("Reconstruction,\nparticipant mean (cm)", fontsize=10)
        a1.set_xlabel("OMC, participant mean (cm)", fontsize=9)

        # --- within: centred strides (coloured by contact side) --------------------
        from matplotlib.colors import to_rgb
        a2 = ax[1, ci]
        base = np.array(to_rgb(COL[mdl]))
        light = tuple(0.45 * base + 0.55 * np.array([1, 1, 1]))
        xm = dd.groupby("subject")["omc"].transform("mean")
        ym2 = dd.groupby("subject")["pred"].transform("mean")
        xc = (dd["omc"] - xm).values
        yc = (dd["pred"] - ym2).values
        sides = dd["side"].values
        s95 = np.nanpercentile(np.abs(np.r_[xc, yc]), 99)
        lim2 = [-1.05 * s95, 1.05 * s95]
        a2.plot(lim2, lim2, "k--", lw=0.9, alpha=0.6)
        for sd_, col_, lab_ in (("L", tuple(base), "left"), ("R", light, "right")):
            msk = sides == sd_
            a2.scatter(xc[msk], yc[msk], s=5, alpha=0.12, color=col_,
                       rasterized=True, label=f"{lab_} contacts")
        bw, bws = r["beta_within"], r["beta_within_side"]
        x0 = np.array(lim2)
        if np.isfinite(bw):
            a2.plot(x0, bw * x0, color="k", lw=1.6)
        if np.isfinite(bws):
            a2.plot(x0, bws * x0, color="k", lw=1.4, ls=":")
        a2.set_xlim(lim2), a2.set_ylim(lim2)
        a2.text(0.03, 0.97,
                (f"$\\beta_W$ {bw:.2f} [{r['beta_within_lo']:.2f},{r['beta_within_hi']:.2f}] (solid)\n"
                 f"$\\beta_{{WS}}$ {bws:.2f} "
                 f"[{r['beta_within_side_lo']:.2f},{r['beta_within_side_hi']:.2f}] (dotted)"),
                transform=a2.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        if ci == 0:
            leg = a2.legend(loc="lower right", fontsize=7, framealpha=0.9,
                            handletextpad=0.2, borderpad=0.3)
            for h in leg.legend_handles:
                h.set_alpha(0.9)
        if ci == 0:
            a2.set_ylabel("Reconstruction,\nparticipant-centred (cm)", fontsize=10)
        a2.set_xlabel("OMC, participant-centred (cm)", fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"calibration_{param}.{ext}"),
                    dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def fig_rm_bland_altman(d, S, outdir):
    """Appendix: Bland-Altman panels with repeated-measures LoA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ["stepLength", "strideLength", "stepWidth"]
    fig, ax = plt.subplots(3, 3, figsize=(11, 9))
    for ri, param in enumerate(rows):
        for ci, mdl in enumerate(MODELS):
            a = ax[ri, ci]
            dd = d[(d["model"] == mdl) & (d["parameter"] == param)]
            r = S[(mdl, param)]
            o = dd["omc"].values
            p = dd["pred"].values
            mean = (o + p) / 2
            diff = p - o
            a.scatter(mean, diff, s=5, alpha=0.10, color=COL[mdl], rasterized=True)
            a.axhline(r["bias"], color=COL[mdl], lw=1.2)
            a.axhline(r["rm_loa_lo"], color="0.4", ls="--", lw=0.9)
            a.axhline(r["rm_loa_hi"], color="0.4", ls="--", lw=0.9)
            a.axhline(0, color="k", lw=0.5, alpha=0.4)
            a.text(0.03, 0.03,
                   (f"bias {r['bias']:+.1f}\nRM-LoA [{r['rm_loa_lo']:+.1f},{r['rm_loa_hi']:+.1f}]\n"
                    f"$\\sigma_b$ {r['rm_sigma_b']:.1f}, $\\sigma_w$ {r['rm_sigma_w']:.1f}"),
                   transform=a.transAxes, fontsize=7.5, va="bottom",
                   bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
            if ri == 0:
                a.set_title(NAME[mdl], fontsize=12)
            if ci == 0:
                a.set_ylabel(f"{PLAB[param]}\nrecon − OMC (cm)", fontsize=10)
            if ri == 2:
                a.set_xlabel("mean of methods (cm)")
    fig.suptitle("Bland–Altman agreement with repeated-measures 95% limits of agreement\n"
                 "(variance components: between-participant $\\sigma_b$ + within-participant $\\sigma_w$; "
                 "Bland & Altman 2007)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"rmBA_appendix.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def write_tex(S, outdir, window):
    win = ("positions averaged over $\\pm$%d frames ($\\pm$%d\\,ms) around the "
           "contact frame" % (window, round(window / SF50 * 1000))
           ) if window > 0 else "positions sampled at the contact frame"
    show = ["stepLength", "strideLength", "stepWidth"]
    # OMC within-participant SDs are model-independent -> report in the caption
    wsd = " , ".join(f"{PLAB[p].lower()} {S[(MODELS[0], p)]['sd_within_omc']:.1f}"
                     f" ({S[(MODELS[0], p)]['sd_within_omc_side']:.1f})\\,cm"
                     for p in show)
    lines = [
        "% Auto-generated by DK04_Gait11_calibration_decomposition.py",
        "% Between- vs within-participant calibration; " + win + ".",
        "\\begin{table*}[!t]", "\\centering",
        "\\caption{Between- and within-participant agreement between reconstructed and "
        "OMC spatial gait parameters at the shared gyroscope events (held-out test set, "
        "$n=8$; sides pooled). Bias, MAE, limits of agreement (LoA), and centred RMSE are "
        "in centimetres. LoA are repeated-measures limits from a one-way variance-components "
        "decomposition of the per-stride differences. $\\beta_{\\mathrm{between}}$ is the "
        "slope of the participant-mean reconstruction on the participant-mean OMC value "
        "($n=8$ means); $\\beta_{\\mathrm{within}}$ is the slope of the mean-centred "
        "reconstructed strides on the mean-centred OMC strides after removing each "
        "participant's mean; $\\beta_{\\mathrm{within}}^{\\mathrm{side}}$ additionally "
        "removes each side's mean within each participant, so left--right asymmetry cannot "
        "contribute. Brackets are participant-bootstrap 95\\% CIs. Centred RMSE is the "
        "root-mean-square of the participant-centred (side-centred) errors; the "
        "within-participant bias is zero by construction and is therefore not reported. "
        "The OMC within-participant (within-side) SDs, i.e.\\ the stride-to-stride "
        "variation to be recovered, were " + wsd + ".}",
        "\\label{tab:calibration_decomposition}", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4.0pt}", "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{llccccccc}", "\\toprule",
        "\\textbf{Model} & \\textbf{Parameter} & \\textbf{Bias} & \\textbf{95\\% RM-LoA} & "
        "\\textbf{MAE} & \\textbf{$\\beta_{\\mathrm{between}}$ [95\\% CI]} & "
        "\\textbf{$\\beta_{\\mathrm{within}}$ [95\\% CI]} & "
        "\\textbf{$\\beta_{\\mathrm{within}}^{\\mathrm{side}}$ [95\\% CI]} & "
        "\\textbf{Centred RMSE} \\\\", "\\midrule",
    ]
    for mi, mdl in enumerate(MODELS):
        for pi, param in enumerate(show):
            r = S[(mdl, param)]
            first = f"\\multirow{{{len(show)}}}{{*}}{{{NAME[mdl]}}}" if pi == 0 else ""
            lines.append(
                f"{first} & {PLAB[param]} & ${r['bias']:+.1f}$ & "
                f"$[{r['rm_loa_lo']:+.1f},\\,{r['rm_loa_hi']:+.1f}]$ & ${r['mae']:.1f}$ & "
                f"${r['beta_between']:.2f}\\,[{r['beta_between_lo']:.2f},{r['beta_between_hi']:.2f}]$ & "
                f"${r['beta_within']:.2f}\\,[{r['beta_within_lo']:.2f},{r['beta_within_hi']:.2f}]$ & "
                f"${r['beta_within_side']:.2f}\\,[{r['beta_within_side_lo']:.2f},{r['beta_within_side_hi']:.2f}]$ & "
                f"${r['centred_rmse']:.1f}$ (${r['centred_rmse_side']:.1f}$) \\\\")
        if mi < len(MODELS) - 1:
            lines.append("\\cmidrule(l){2-9}")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    path = os.path.join(outdir, "calibration_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest():
    import pandas as pd
    print("Self-test 1: ICC(A,1) vs Shrout & Fleiss (1979) benchmark")
    data = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                     [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    icc4, _, _ = icc_a1(data)
    print(f"  4-rater ICC(A,1) = {icc4:.4f} (expected 0.2898)")
    assert abs(icc4 - 0.2898) < 1e-3

    print("Self-test 2: between/within slope recovery on synthetic data")
    rng = np.random.RandomState(1)
    rows = []
    for i in range(8):
        xbar = rng.uniform(50, 80)
        ybar = 10 + 0.4 * xbar                    # true beta_between = 0.4
        u = rng.normal(0, 3.0, 300)               # within-participant variation
        x = xbar + u
        y = ybar + 0.9 * u + rng.normal(0, 0.5, 300)   # true beta_within = 0.9
        for j, (xx, yy) in enumerate(zip(x, y)):
            rows.append(dict(subject=f"{i + 1:02d}", side="LR"[j % 2], omc=xx, pred=yy))
    d = pd.DataFrame(rows)
    r = analyse(d, B=300, seed=0)
    print(f"  beta_between = {r['beta_between']:.3f} (true 0.4)   "
          f"beta_within = {r['beta_within']:.3f} (true 0.9)   "
          f"beta_within_side = {r['beta_within_side']:.3f} (true 0.9)")
    print(f"  centred RMSE = {r['centred_rmse']:.3f} (expected ~0.58)   "
          f"centred bias = {r['centred_bias_check']:.2e} (0 by construction)")
    assert abs(r["beta_between"] - 0.4) < 0.1
    assert abs(r["beta_within"] - 0.9) < 0.02
    assert abs(r["beta_within_side"] - 0.9) < 0.02
    assert abs(r["centred_rmse"] - np.sqrt((0.1 * 3.0) ** 2 + 0.5 ** 2)) < 0.05
    assert abs(r["centred_bias_check"]) < 1e-9

    print("Self-test 2b: side-centring isolates asymmetry from stride variability")
    # OMC has a +/-1.5 cm L/R asymmetry; the model EXAGGERATES it 3x but tracks
    # genuine stride fluctuation with slope 0.9. Participant-only centring must
    # inflate beta_within (expected ~2.35 here); side-centring must recover 0.9.
    rng = np.random.RandomState(3)
    rows = []
    for i in range(8):
        xbar = rng.uniform(50, 80)
        for j in range(300):
            side = "LR"[j % 2]
            s_x = 1.5 if side == "L" else -1.5
            u = rng.normal(0, 1.0)
            rows.append(dict(subject=f"{i + 1:02d}", side=side,
                             omc=xbar + s_x + u,
                             pred=xbar + 3.0 * s_x + 0.9 * u + rng.normal(0, 0.3)))
    r = analyse(pd.DataFrame(rows), B=200, seed=0)
    exp_inflated = (3.0 * 1.5 ** 2 + 0.9 * 1.0) / (1.5 ** 2 + 1.0)
    print(f"  beta_within (participant-centred) = {r['beta_within']:.2f} "
          f"(expected ~{exp_inflated:.2f}, inflated by asymmetry)")
    print(f"  beta_within_side (side-centred)   = {r['beta_within_side']:.3f} (true 0.9)")
    assert abs(r["beta_within"] - exp_inflated) < 0.1
    assert abs(r["beta_within_side"] - 0.9) < 0.03

    print("Self-test 3: repeated-measures LoA variance components")
    rng = np.random.RandomState(2)
    subj = np.repeat(np.arange(200), 30)
    dvals = np.repeat(rng.normal(1.0, 2.0, 200), 30) + rng.normal(0, 1.0, 200 * 30)
    L = rm_loa(dvals, subj)
    print(f"  sd_total = {L['sd_total']:.3f} (expected ~{np.sqrt(5):.3f}); "
          f"sigma_b = {L['sigma_b']:.2f} (2.0), sigma_w = {L['sigma_w']:.2f} (1.0)")
    assert abs(L["sd_total"] - np.sqrt(5)) < 0.15
    assert abs(L["sigma_b"] - 2.0) < 0.25 and abs(L["sigma_w"] - 1.0) < 0.05
    print("Self-test PASSED.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data",
                    help="folder with _omc_gyro_extract.npz and posfeet.npz")
    ap.add_argument("--csv", default=None,
                    help="optional pre-built per-stride CSV (subject,model,parameter,omc,pred); "
                         "skips the npz rebuild")
    ap.add_argument("--outdir", default="./calibration_out")
    ap.add_argument("--window", type=int, default=0,
                    help="frames averaged each side of the contact frame "
                         "(0 = instantaneous; 3 = +/-60 ms sensitivity analysis)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    import pandas as pd
    os.makedirs(args.outdir, exist_ok=True)

    if args.csv:
        d = pd.read_csv(args.csv, dtype={"subject": str})
        print(f"Loaded {len(d)} per-stride rows from {args.csv}")
    else:
        d = build_perstride(args.data_dir, args.window)
        print(f"Built {len(d)} per-stride rows "
              f"(window = +/-{args.window} frames = +/-{round(args.window / SF50 * 1000)} ms)")
    d = d.dropna(subset=["omc", "pred"]).reset_index(drop=True)
    d.to_csv(os.path.join(args.outdir, "perstride_pairs.csv"), index=False)

    S = {}
    pp_rows = []
    loo_rows = []
    for mdl in MODELS:
        for param in PARAMS:
            dd = d[(d["model"] == mdl) & (d["parameter"] == param)]
            if not len(dd):
                continue
            S[(mdl, param)] = analyse(dd, B=args.boot, seed=args.seed)
            for row in per_participant_table(dd):
                pp_rows.append(dict(model=NAME[mdl], parameter=param, **row))
            current_loo = leave_one_participant_out(dd)
            for row in current_loo:
                loo_rows.append(dict(model=NAME[mdl], parameter=param, **row))
            for key in ("beta_between", "beta_within", "beta_within_side"):
                vals = np.asarray([row[key] for row in current_loo], float)
                S[(mdl, param)][f"{key}_loo_min"] = float(np.nanmin(vals))
                S[(mdl, param)][f"{key}_loo_max"] = float(np.nanmax(vals))

    # ---- console report ---------------------------------------------------
    print("\nNOTE: within-participant bias is zero BY CONSTRUCTION (mean-centring); "
          "it is reported only as a numerical check, never as a finding.\n")
    hdr = (f"{'param':<13}{'model':<8}{'bias':>7}{'MAE':>6}   {'RM-LoA':<16}"
           f"{'b_betw [CI]':<19}{'b_with [CI]':<19}{'b_with,side [CI]':<20}"
           f"{'cRMSE':>6}{'cRMSE_s':>8}{'wSD':>5}{'wSD_s':>6}")
    for param in PARAMS:
        print(f"\n=== {PLAB[param]} ===")
        print(hdr)
        for mdl in MODELS:
            r = S.get((mdl, param))
            if r is None:
                continue
            print(f"{param:<13}{NAME[mdl]:<8}{r['bias']:>+7.1f}{r['mae']:>6.1f}   "
                  f"[{r['rm_loa_lo']:+.1f},{r['rm_loa_hi']:+.1f}]   "
                  f"{r['beta_between']:.2f} [{r['beta_between_lo']:.2f},{r['beta_between_hi']:.2f}]  "
                  f"{r['beta_within']:.2f} [{r['beta_within_lo']:.2f},{r['beta_within_hi']:.2f}]  "
                  f"{r['beta_within_side']:.2f} [{r['beta_within_side_lo']:.2f},{r['beta_within_side_hi']:.2f}]  "
                  f"{r['centred_rmse']:>6.1f}{r['centred_rmse_side']:>8.1f}"
                  f"{r['sd_within_omc']:>5.1f}{r['sd_within_omc_side']:>6.1f}")

    # ---- outputs ----------------------------------------------------------
    summary = pd.DataFrame([dict(model=NAME[m], parameter=p, **S[(m, p)])
                            for (m, p) in S])
    sp = os.path.join(args.outdir, "calibration_summary.csv")
    summary.to_csv(sp, index=False)
    pd.DataFrame(pp_rows).to_csv(
        os.path.join(args.outdir, "calibration_perparticipant.csv"), index=False)
    pd.DataFrame(loo_rows).to_csv(
        os.path.join(args.outdir, "calibration_leave_one_out.csv"), index=False)
    tex = write_tex(S, args.outdir, args.window)

    for param in ["stepLength", "strideLength", "stepWidth"]:
        fig_calibration(d, S, param, args.outdir)
    fig_rm_bland_altman(d, S, args.outdir)

    print(f"\nWrote:\n  {sp}\n  calibration_perparticipant.csv\n"
          f"  calibration_leave_one_out.csv\n  perstride_pairs.csv\n"
          f"  {tex}\n  calibration_<param>.pdf/png (main-text candidates)\n"
          f"  rmBA_appendix.pdf/png (appendix)")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()

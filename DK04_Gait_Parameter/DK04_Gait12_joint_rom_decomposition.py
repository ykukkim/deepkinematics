#!/usr/bin/env python3
"""Cycle-to-cycle joint-ROM agreement for the held-out gait dataset.

The script uses the same foot-gyroscope initial contacts and 200-to-50-Hz
alignment as DK04_Gait11_calibration_decomposition.py. For each consecutive
pair of contacts from the same foot, it calculates sagittal hip, knee, and
ankle range of motion (ROM) in the OMC and reconstructed signals.

Two levels of calibration are reported:
  beta_B  : slope across participant-mean ROM values;
  beta_WS : slope after removing each participant-by-foot mean, which measures
            preservation of cycle-to-cycle ROM changes within the same foot.

Participant-clustered percentile bootstrap confidence intervals preserve the
independence of the held-out participants.
"""

import argparse
import os

import numpy as np


SUBJECTS = ["04", "09", "14", "24", "51", "54", "67", "81"]
MODELS = ["BILSTM", "ATT", "DIFF"]
MODEL_LABEL = {"BILSTM": "BiLSTM", "ATT": "ATT", "DIFF": "DIFF"}
JOINTS = ["Hip", "Knee", "Ankle"]
SIDES = ["L", "R"]
SF_NATIVE = 200.0
SF_RECON = 50.0


def _ols_slope(x, y):
    """OLS slope of y on x with an intercept."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    vx = np.var(x)
    if vx <= 0:
        return np.nan
    return float(np.cov(x, y, bias=True)[0, 1] / vx)


def _icc_a1(values):
    """Two-way random, absolute-agreement, single-measure ICC(A,1)."""
    values = np.asarray(values, float)
    n, k = values.shape
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_columns = n * np.sum((column_means - grand) ** 2)
    ss_total = np.sum((values - grand) ** 2)
    ss_error = ss_total - ss_rows - ss_columns
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = (
        ms_rows + (k - 1) * ms_error + (k / n) * (ms_columns - ms_error)
    )
    if denominator == 0:
        return np.nan
    return float((ms_rows - ms_error) / denominator)


def _repeated_measures_loa(difference, subjects):
    """Repeated-measures 95% LoA using a one-way variance decomposition."""
    difference = np.asarray(difference, float)
    subjects = np.asarray(subjects)
    ids = np.unique(subjects)
    n_participants = len(ids)
    n_observations = len(difference)
    bias = float(difference.mean())
    if n_participants < 2 or n_observations - n_participants < 1:
        standard_deviation = float(difference.std(ddof=1))
        return bias - 1.96 * standard_deviation, bias + 1.96 * standard_deviation

    cluster_sizes = np.array([np.sum(subjects == participant) for participant in ids], float)
    cluster_means = np.array([difference[subjects == participant].mean() for participant in ids])
    ms_between = np.sum(cluster_sizes * (cluster_means - bias) ** 2) / (n_participants - 1)
    ms_within = sum(
        ((difference[subjects == participant] - cluster_means[index]) ** 2).sum()
        for index, participant in enumerate(ids)
    ) / (n_observations - n_participants)
    effective_cluster_size = (
        n_observations - np.sum(cluster_sizes ** 2) / n_observations
    ) / (n_participants - 1)
    between_variance = max(0.0, (ms_between - ms_within) / effective_cluster_size)
    total_standard_deviation = np.sqrt(between_variance + ms_within)
    return bias - 1.96 * total_standard_deviation, bias + 1.96 * total_standard_deviation


def _centred(d, by):
    keys = list(by)
    xc = d["omc"] - d.groupby(keys)["omc"].transform("mean")
    yc = d["pred"] - d.groupby(keys)["pred"].transform("mean")
    return xc.to_numpy(float), yc.to_numpy(float)


def _through_origin_slope(x, y):
    denominator = float(np.sum(x * x))
    if denominator <= 0:
        return np.nan
    return float(np.sum(x * y) / denominator)


def _cluster_bootstrap(d, statistic, n_boot, rng):
    """Percentile CI after resampling complete participant clusters."""
    import pandas as pd

    ids = d["subject"].unique()
    groups = {s: d[d["subject"] == s] for s in ids}
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(ids, len(ids), replace=True)
        dd = pd.concat(
            [groups[s].assign(subject=f"{s}#{i}") for i, s in enumerate(sampled)],
            ignore_index=True,
        )
        value = statistic(dd)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def _gyro_events(data_dir):
    """Return aligned 50-Hz initial contacts using the Gait11 detector."""
    from scipy.signal import butter, correlate, filtfilt, find_peaks, resample_poly
    from scipy.stats import skew

    source = np.load(os.path.join(data_dir, "_omc_gyro_extract.npz"))
    positions = np.load(os.path.join(data_dir, "posfeet.npz"))
    b, a = butter(4, 12 / (SF_NATIVE / 2), "low")

    def detect(signal):
        g = filtfilt(b, a, signal)
        if skew(g) < 0:
            g = -g
        scale = np.max(np.abs(g))
        if scale <= 0:
            return np.array([], dtype=int)
        gn = g / scale
        swing, _ = find_peaks(
            gn, height=0.4, distance=int(0.7 * SF_NATIVE)
        )
        contacts = []
        for i, peak in enumerate(swing):
            end = (
                swing[i + 1]
                if i + 1 < len(swing)
                else min(len(gn), peak + int(1.3 * SF_NATIVE))
            )
            segment = gn[peak:end]
            if len(segment) < int(0.3 * SF_NATIVE):
                continue
            minimum = peak + int(np.argmin(segment))
            window = gn[minimum : min(end, minimum + int(0.45 * SF_NATIVE))]
            if len(window) < 3:
                continue
            contacts.append(minimum + int(np.argmax(window)))
        return np.asarray(sorted(set(contacts)), dtype=int)

    def alignment_offset(subject):
        native = resample_poly(source[f"{subject}|LAJC"][:, 1], 1, 4)
        target = positions[f"{subject}|gt|LAJC"][:, 1]
        native = (native - native.mean()) / (native.std() + 1e-9)
        target = (target - target.mean()) / (target.std() + 1e-9)
        correlation = correlate(native, target, mode="full")
        return int(np.argmax(np.abs(correlation)) - (len(target) - 1))

    events = {}
    for subject in SUBJECTS:
        n_frames = len(positions[f"{subject}|gt|LAJC"])
        lag = alignment_offset(subject)
        for side in SIDES:
            native = detect(source[f"{subject}|g{side}"])
            mapped = np.round(native / (SF_NATIVE / SF_RECON)).astype(int) - lag
            mapped = np.unique(mapped[(mapped >= 0) & (mapped < n_frames)])
            events[(subject, side)] = mapped
    return events


def _cycle_pairs(signal_omc, signal_pred, events):
    """Calculate OMC and predicted ROM for plausible same-foot gait cycles."""
    events = np.asarray(events, dtype=int)
    intervals = np.diff(events)
    if len(intervals) == 0:
        return []
    period = float(np.median(intervals))
    rows = []
    for cycle, (start, stop) in enumerate(zip(events[:-1], events[1:])):
        length = stop - start
        if not (max(20, 0.6 * period) <= length <= min(100, 1.5 * period)):
            continue
        # The stop frame begins the next cycle and is excluded.
        omc = np.asarray(signal_omc[start:stop], float)
        pred = np.asarray(signal_pred[start:stop], float)
        if len(omc) < 2 or not (np.all(np.isfinite(omc)) and np.all(np.isfinite(pred))):
            continue
        rows.append(
            {
                "cycle": cycle,
                "start_frame": int(start),
                "stop_frame": int(stop),
                "cycle_frames": int(length),
                "omc": float(np.ptp(omc)),
                "pred": float(np.ptp(pred)),
            }
        )
    return rows


def _normalised_cycle_matrices(signal_omc, signal_pred, events, n_points=101):
    """Return paired, time-normalised cycles using the same plausibility rule."""
    events = np.asarray(events, dtype=int)
    intervals = np.diff(events)
    if len(intervals) == 0:
        empty = np.empty((0, n_points))
        return empty, empty
    period = float(np.median(intervals))
    grid = np.linspace(0.0, 1.0, n_points)
    omc_cycles, pred_cycles = [], []
    for start, stop in zip(events[:-1], events[1:]):
        length = stop - start
        if not (max(20, 0.6 * period) <= length <= min(100, 1.5 * period)):
            continue
        omc = np.asarray(signal_omc[start:stop], float)
        pred = np.asarray(signal_pred[start:stop], float)
        if len(omc) < 2 or not (np.all(np.isfinite(omc)) and np.all(np.isfinite(pred))):
            continue
        source_grid = np.linspace(0.0, 1.0, len(omc))
        omc_cycles.append(np.interp(grid, source_grid, omc))
        pred_cycles.append(np.interp(grid, source_grid, pred))
    if not omc_cycles:
        empty = np.empty((0, n_points))
        return empty, empty
    return np.vstack(omc_cycles), np.vstack(pred_cycles)


def build_cycle_table(data_dir):
    import pandas as pd

    contacts = _gyro_events(data_dir)
    arrays = {
        model: np.load(os.path.join(data_dir, f"flex_{model}.npz"))
        for model in MODELS
    }
    rows = []
    for model in MODELS:
        data = arrays[model]
        for subject in SUBJECTS:
            for side in SIDES:
                events = contacts[(subject, side)]
                for joint in JOINTS:
                    prefix = f"SonE_{subject}|{side}|{joint}"
                    pairs = _cycle_pairs(data[f"{prefix}|gt"], data[f"{prefix}|hat"], events)
                    for pair in pairs:
                        pair.update(
                            model=MODEL_LABEL[model],
                            subject=subject,
                            side=side,
                            joint=joint,
                        )
                        rows.append(pair)
    return pd.DataFrame(rows), contacts


def waveform_variability_summary(data_dir, contacts, n_boot, seed):
    """Agreement of cycle-specific waveform deviations from each foot's mean.

    Each time-normalised cycle is centred by the participant-by-foot mean curve.
    The pooled slope therefore asks whether a cycle-specific departure at a given
    gait-cycle phase is reproduced by the reconstruction. Bootstrap resampling
    uses participant-level sufficient statistics and does not treat frames or
    cycles as independent observations.
    """
    import pandas as pd

    arrays = {
        model: np.load(os.path.join(data_dir, f"flex_{model}.npz"))
        for model in MODELS
    }
    contribution_rows = []
    for model in MODELS:
        data = arrays[model]
        for subject in SUBJECTS:
            for joint in JOINTS:
                sxx = syy = sxy = sse = 0.0
                n_values = n_cycles = 0
                for side in SIDES:
                    prefix = f"SonE_{subject}|{side}|{joint}"
                    omc, pred = _normalised_cycle_matrices(
                        data[f"{prefix}|gt"],
                        data[f"{prefix}|hat"],
                        contacts[(subject, side)],
                    )
                    if len(omc) == 0:
                        continue
                    x = omc - omc.mean(axis=0, keepdims=True)
                    y = pred - pred.mean(axis=0, keepdims=True)
                    sxx += float(np.sum(x * x))
                    syy += float(np.sum(y * y))
                    sxy += float(np.sum(x * y))
                    sse += float(np.sum((y - x) ** 2))
                    n_values += x.size
                    n_cycles += len(x)
                contribution_rows.append(
                    dict(
                        model=MODEL_LABEL[model], subject=subject, joint=joint,
                        n_cycles=n_cycles, n_values=n_values,
                        sxx=sxx, syy=syy, sxy=sxy, sse=sse,
                    )
                )
    contributions = pd.DataFrame(contribution_rows)

    def metrics(d):
        sums = d[["n_cycles", "n_values", "sxx", "syy", "sxy", "sse"]].sum()
        n = float(sums["n_values"])
        sxx, syy, sxy = float(sums["sxx"]), float(sums["syy"]), float(sums["sxy"])
        return {
            "n_cycles": int(sums["n_cycles"]),
            "beta_waveform_within_foot": sxy / sxx if sxx > 0 else np.nan,
            "r_waveform_within_foot": sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else np.nan,
            "omc_waveform_within_sd": np.sqrt(sxx / n),
            "pred_waveform_within_sd": np.sqrt(syy / n),
            "waveform_centred_rmse": np.sqrt(float(sums["sse"]) / n),
        }

    rng = np.random.RandomState(seed)
    output = []
    for model in ["BiLSTM", "ATT", "DIFF"]:
        for joint in JOINTS:
            d = contributions[(contributions["model"] == model) & (contributions["joint"] == joint)]
            result = metrics(d)
            sampled_metrics = []
            ids = d["subject"].to_numpy()
            by_subject = {s: d[d["subject"] == s] for s in ids}
            for _ in range(n_boot):
                sampled = rng.choice(ids, len(ids), replace=True)
                dd = pd.concat([by_subject[s] for s in sampled], ignore_index=True)
                sampled_metrics.append(metrics(dd))
            for key in ["beta_waveform_within_foot", "r_waveform_within_foot", "waveform_centred_rmse"]:
                values = [entry[key] for entry in sampled_metrics if np.isfinite(entry[key])]
                result[f"{key}_lo"], result[f"{key}_hi"] = [
                    float(v) for v in np.percentile(values, [2.5, 97.5])
                ]
            result.update(model=model, joint=joint)
            output.append(result)
    return pd.DataFrame(output), contributions


def participant_mean_waveform_analysis(data_dir, contacts):
    """Participant-mean waveform agreement using the shared gyro contacts.

    Cycles from the left and right feet are combined within each participant
    before calculating that participant's mean waveform.  The eight paired
    participant means are then used for RMSE, Pearson correlation, CMC, and
    paired SPM1D, matching the evaluation described in the manuscript.
    """
    import pandas as pd
    from DK04_Gait_Parameter.DK04_Gait08_joint_kinematics import (
        cmc_waveform,
        spm_paired,
    )

    arrays = {
        model: np.load(os.path.join(data_dir, f"flex_{model}.npz"))
        for model in MODELS
    }
    rows = []
    curves = {}
    for model in MODELS:
        data = arrays[model]
        for joint in JOINTS:
            omc_participants, pred_participants = [], []
            n_cycles = 0
            for subject in SUBJECTS:
                omc_cycles, pred_cycles = [], []
                for side in SIDES:
                    prefix = f"SonE_{subject}|{side}|{joint}"
                    omc, pred = _normalised_cycle_matrices(
                        data[f"{prefix}|gt"],
                        data[f"{prefix}|hat"],
                        contacts[(subject, side)],
                    )
                    if len(omc):
                        omc_cycles.append(omc)
                        pred_cycles.append(pred)
                        n_cycles += len(omc)
                if not omc_cycles:
                    continue
                omc_participants.append(np.vstack(omc_cycles).mean(axis=0))
                pred_participants.append(np.vstack(pred_cycles).mean(axis=0))

            omc = np.vstack(omc_participants)
            pred = np.vstack(pred_participants)
            rmse_by_participant = np.sqrt(np.mean((pred - omc) ** 2, axis=1))
            correlation_by_participant = np.array(
                [np.corrcoef(omc[i], pred[i])[0, 1] for i in range(len(omc))]
            )
            spm = spm_paired(omc, pred)
            label = MODEL_LABEL[model]
            rows.append(
                {
                    "model": label,
                    "joint": joint,
                    "n_participants": len(omc),
                    "n_cycles": n_cycles,
                    "rmse_deg": float(rmse_by_participant.mean()),
                    "rmse_sd_deg": float(rmse_by_participant.std(ddof=1)),
                    "omc_rom_deg": float(np.ptp(omc, axis=1).mean()),
                    "pred_rom_deg": float(np.ptp(pred, axis=1).mean()),
                    "pearson_r": float(np.nanmean(correlation_by_participant)),
                    "cmc": cmc_waveform(omc, pred),
                    "spm_method": spm["method"],
                    "spm_percent_sig": spm["percent_sig"],
                    "spm_clusters": ";".join(
                        f"{low:.1f}-{high:.1f}%" for low, high in spm["clusters"]
                    ),
                }
            )
            curves[(label, joint)] = {
                "omc": omc,
                "pred": pred,
                "spm": spm,
            }
    return pd.DataFrame(rows), curves


def plot_participant_mean_waveforms(curves, outdir):
    """Plot participant-mean waveforms and paired SPM1D regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    colours = {"BiLSTM": "#0072B2", "ATT": "#009E73", "DIFF": "#D55E00"}
    ylabels = {
        "Hip": "Hip flexion--extension (deg)",
        "Knee": "Knee flexion--extension (deg)",
        "Ankle": "Ankle dorsiflexion--plantarflexion (deg)",
    }
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 8.4), sharex=True)
    x = np.linspace(0.0, 100.0, 101)
    for row, joint in enumerate(JOINTS):
        for column, model in enumerate(["BiLSTM", "ATT", "DIFF"]):
            ax = axes[row, column]
            entry = curves[(model, joint)]
            for values, colour, label in [
                (entry["omc"], "black", "OMC"),
                (entry["pred"], colours[model], "Reconstruction"),
            ]:
                mean = values.mean(axis=0)
                sd = values.std(axis=0, ddof=1)
                ax.plot(x, mean, color=colour, linewidth=1.8, label=label)
                ax.fill_between(x, mean - sd, mean + sd, color=colour,
                                alpha=0.16, linewidth=0)
            for low, high in entry["spm"].get("clusters", []):
                ax.axvspan(low, high, color="#CC79A7", alpha=0.20, linewidth=0)
            if row == 0:
                ax.set_title(model)
            if column == 0:
                ax.set_ylabel(ylabels[joint])
                ax.yaxis.set_label_coords(-0.14, 0.5)
            if row == len(JOINTS) - 1:
                ax.set_xlabel("Gait cycle (%)")
            ax.set_xlim(0, 100)
            ax.grid(alpha=0.20, linewidth=0.5)

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.8, label="Vicon (OMC)"),
        Line2D([0], [0], color=colours["BiLSTM"], linewidth=1.8, label="BiLSTM"),
        Line2D([0], [0], color=colours["ATT"], linewidth=1.8, label="ATT"),
        Line2D([0], [0], color=colours["DIFF"], linewidth=1.8, label="DIFF"),
        Patch(facecolor="#CC79A7", alpha=0.20, edgecolor="none",
              label="SPM1D p<0.05"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.5,
        handlelength=2.0,
    )
    fig.tight_layout(rect=[0, 0.065, 1, 1], h_pad=2.0)
    pdf_path = os.path.join(outdir, "joint_kinematic_waveforms.pdf")
    png_path = os.path.join(outdir, "joint_kinematic_waveforms.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def analyse(d, n_boot, seed):
    rng = np.random.RandomState(seed)
    difference = (d["pred"] - d["omc"]).to_numpy(float)
    participant_mae = (
        d.assign(abs_error=np.abs(difference))
        .groupby("subject")["abs_error"]
        .mean()
    )

    participant_means = d.groupby("subject")[["omc", "pred"]].mean()
    beta_between = _ols_slope(participant_means["omc"], participant_means["pred"])

    def between_stat(dd):
        means = dd.groupby("subject")[["omc", "pred"]].mean()
        return _ols_slope(means["omc"], means["pred"])

    beta_between_ci = _cluster_bootstrap(d, between_stat, n_boot, rng)

    x_within, y_within = _centred(d, by=("subject", "side"))
    beta_within = _through_origin_slope(x_within, y_within)

    def within_stat(dd):
        x, y = _centred(dd, by=("subject", "side"))
        return _through_origin_slope(x, y)

    beta_within_ci = _cluster_bootstrap(d, within_stat, n_boot, rng)

    centred_error = y_within - x_within
    centred_rmse = float(np.sqrt(np.mean(centred_error**2)))

    def rmse_stat(dd):
        x, y = _centred(dd, by=("subject", "side"))
        return float(np.sqrt(np.mean((y - x) ** 2)))

    centred_rmse_ci = _cluster_bootstrap(d, rmse_stat, n_boot, rng)

    rmse = float(np.sqrt(np.mean(difference**2)))
    loa_low, loa_high = _repeated_measures_loa(difference, d["subject"].to_numpy())
    pooled_icc = _icc_a1(np.c_[d["omc"].to_numpy(float), d["pred"].to_numpy(float)])

    def icc_stat(dd):
        return _icc_a1(np.c_[dd["omc"].to_numpy(float), dd["pred"].to_numpy(float)])

    icc_rng = np.random.RandomState(seed + 7919)
    pooled_icc_ci = _cluster_bootstrap(d, icc_stat, n_boot, icc_rng)

    # Root-mean-square within-foot SD, matching the denominator underlying beta_WS.
    omc_within_sd = float(np.sqrt(np.mean(x_within**2)))
    pred_within_sd = float(np.sqrt(np.mean(y_within**2)))

    return {
        "n_cycles": len(d),
        "n_participants": d["subject"].nunique(),
        "omc_mean_rom": float(d["omc"].mean()),
        "pred_mean_rom": float(d["pred"].mean()),
        "bias": float(difference.mean()),
        "mae": float(np.abs(difference).mean()),
        "rmse": rmse,
        "rm_loa_low": loa_low,
        "rm_loa_high": loa_high,
        "pooled_icc_a1": pooled_icc,
        "pooled_icc_a1_lo": pooled_icc_ci[0],
        "pooled_icc_a1_hi": pooled_icc_ci[1],
        "participant_mean_mae": float(participant_mae.mean()),
        "participant_sd_mae": float(participant_mae.std(ddof=1)),
        "beta_between": beta_between,
        "beta_between_lo": beta_between_ci[0],
        "beta_between_hi": beta_between_ci[1],
        "beta_within_foot": beta_within,
        "beta_within_foot_lo": beta_within_ci[0],
        "beta_within_foot_hi": beta_within_ci[1],
        "omc_within_foot_sd": omc_within_sd,
        "pred_within_foot_sd": pred_within_sd,
        "centred_rmse": centred_rmse,
        "centred_rmse_lo": centred_rmse_ci[0],
        "centred_rmse_hi": centred_rmse_ci[1],
    }


def summarise(cycles, n_boot, seed):
    import pandas as pd

    rows = []
    for model in ["BiLSTM", "ATT", "DIFF"]:
        for joint in JOINTS:
            d = cycles[(cycles["model"] == model) & (cycles["joint"] == joint)]
            result = analyse(d, n_boot=n_boot, seed=seed)
            result.update(model=model, joint=joint)
            rows.append(result)
    return pd.DataFrame(rows)


def leave_one_participant_out(cycles):
    import pandas as pd

    rows = []
    for model in ["BiLSTM", "ATT", "DIFF"]:
        for joint in JOINTS:
            d = cycles[(cycles["model"] == model) & (cycles["joint"] == joint)]
            for omitted in SUBJECTS:
                dd = d[d["subject"] != omitted]
                means = dd.groupby("subject")[["omc", "pred"]].mean()
                x, y = _centred(dd, by=("subject", "side"))
                rows.append(
                    {
                        "model": model,
                        "joint": joint,
                        "omitted_subject": omitted,
                        "beta_between": _ols_slope(means["omc"], means["pred"]),
                        "beta_within_foot": _through_origin_slope(x, y),
                    }
                )
    return pd.DataFrame(rows)


def selftest():
    import pandas as pd

    benchmark = np.array(
        [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
         [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
        float,
    )
    assert np.isclose(_icc_a1(benchmark), 0.2898, atol=5e-5)

    rows = []
    for subject, mean in [("A", 10.0), ("B", 20.0), ("C", 30.0)]:
        for side in SIDES:
            for deviation in [-2.0, -1.0, 1.0, 2.0]:
                omc = mean + (1.0 if side == "R" else -1.0) + deviation
                pred = 5.0 + 0.5 * mean + (3.0 if side == "R" else -3.0) + 0.25 * deviation
                rows.append({"subject": subject, "side": side, "omc": omc, "pred": pred})
    d = pd.DataFrame(rows)
    means = d.groupby("subject")[["omc", "pred"]].mean()
    assert np.isclose(_ols_slope(means["omc"], means["pred"]), 0.5)
    x, y = _centred(d, by=("subject", "side"))
    assert np.isclose(_through_origin_slope(x, y), 0.25)
    print("Self-test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "joint_rom_out"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    os.makedirs(args.outdir, exist_ok=True)
    cycles, contacts = build_cycle_table(args.data_dir)
    summary = summarise(cycles, n_boot=args.bootstrap, seed=args.seed)
    robustness = leave_one_participant_out(cycles)
    waveform_summary, waveform_contributions = waveform_variability_summary(
        args.data_dir, contacts, n_boot=args.bootstrap, seed=args.seed
    )
    participant_waveform_summary, participant_waveform_curves = (
        participant_mean_waveform_analysis(args.data_dir, contacts)
    )

    cycles.to_csv(os.path.join(args.outdir, "joint_rom_cycles.csv"), index=False)
    summary.to_csv(os.path.join(args.outdir, "joint_rom_summary.csv"), index=False)
    robustness.to_csv(os.path.join(args.outdir, "joint_rom_leave_one_out.csv"), index=False)
    waveform_summary.to_csv(
        os.path.join(args.outdir, "joint_waveform_variability_summary.csv"), index=False
    )
    waveform_contributions.to_csv(
        os.path.join(args.outdir, "joint_waveform_variability_by_participant.csv"), index=False
    )
    participant_waveform_summary.to_csv(
        os.path.join(args.outdir, "joint_waveform_summary.csv"), index=False
    )
    np.savez_compressed(
        os.path.join(args.outdir, "joint_waveform_participant_curves.npz"),
        **{
            f"{model}|{joint}|{method}": values[method]
            for (model, joint), values in participant_waveform_curves.items()
            for method in ("omc", "pred")
        },
    )
    waveform_figure = plot_participant_mean_waveforms(
        participant_waveform_curves, args.outdir
    )

    counts = {
        subject: {side: len(contacts[(subject, side)]) for side in SIDES}
        for subject in SUBJECTS
    }
    print("Mapped gyro contacts (L/R):", counts)
    columns = [
        "model", "joint", "n_cycles", "omc_mean_rom", "pred_mean_rom",
        "bias", "mae", "rmse", "rm_loa_low", "rm_loa_high",
        "pooled_icc_a1", "pooled_icc_a1_lo", "pooled_icc_a1_hi",
        "participant_mean_mae", "participant_sd_mae",
        "beta_between", "beta_between_lo", "beta_between_hi",
        "beta_within_foot", "beta_within_foot_lo", "beta_within_foot_hi",
        "omc_within_foot_sd", "pred_within_foot_sd", "centred_rmse",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    waveform_columns = [
        "model", "joint", "n_cycles", "beta_waveform_within_foot",
        "beta_waveform_within_foot_lo", "beta_waveform_within_foot_hi",
        "r_waveform_within_foot", "omc_waveform_within_sd",
        "pred_waveform_within_sd", "waveform_centred_rmse",
    ]
    print("\nCycle-to-cycle waveform deviations:")
    print(waveform_summary[waveform_columns].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))
    print("\nParticipant-mean waveform agreement:")
    print(participant_waveform_summary.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))
    print("Waveform figure:", waveform_figure)
    print(f"\nSaved outputs to {args.outdir}")


if __name__ == "__main__":
    main()

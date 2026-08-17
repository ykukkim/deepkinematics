"""Separate between- and within-participant spatial calibration.

Input
-----
shared_spatial.csv
    One row per shared gyroscope event and model, produced by the spatial
    agreement analysis.

Outputs
-------
within_between_agreement.csv
    Overall error and between/within calibration slopes with participant-
    clustered bootstrap confidence intervals.
within_between_subject_means.csv
    Participant means used for the between-participant analysis.
../../../../06_IEEE_TIM/01_Figures/04_spatial_agreement_between_within.pdf
    Paired between/within calibration plots used in the manuscript.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "shared_spatial.csv"
SUMMARY_OUT = HERE / "within_between_agreement.csv"
MEANS_OUT = HERE / "within_between_subject_means.csv"
FIGURE_OUT = HERE.parents[2] / "06_IEEE_TIM" / "01_Figures" / "04_spatial_agreement_between_within.pdf"

MODELS = ["BILSTM", "ATT", "DIFF"]
MODEL_LABELS = {"BILSTM": "BiLSTM", "ATT": "ATT", "DIFF": "DIFF"}
PARAMETERS = ["stepLength", "strideLength", "stepWidth"]
PARAMETER_LABELS = {
    "stepLength": "Step length",
    "strideLength": "Stride length",
    "stepWidth": "Step width",
}
N_BOOT = 10_000
SEED = 20260811


def slope_with_intercept(x, y):
    """OLS slope with a fitted intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    den = np.sum((x - x.mean()) ** 2)
    if len(x) < 2 or den <= np.finfo(float).eps:
        return np.nan
    return np.sum((x - x.mean()) * (y - y.mean())) / den


def slope_through_origin(x, y):
    """OLS slope through the origin, used after participant centring."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    den = np.sum(x**2)
    if len(x) < 2 or den <= np.finfo(float).eps:
        return np.nan
    return np.sum(x * y) / den


def centred_rows(d):
    """Add participant-centred OMC and reconstruction columns."""
    out = d.copy()
    out["omc_centred"] = out["omc"] - out.groupby("subject")["omc"].transform("mean")
    out["pred_centred"] = out["pred"] - out.groupby("subject")["pred"].transform("mean")
    return out


def bootstrap_slopes(d, n_boot=N_BOOT, seed=SEED):
    """Bootstrap participants, retaining all observations within each cluster."""
    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(d["subject"].unique()))
    means = d.groupby("subject", as_index=False)[["omc", "pred"]].mean()
    centred = centred_rows(d)
    by_subject = {s: centred.loc[centred.subject == s] for s in subjects}
    mean_by_subject = means.set_index("subject")
    between = []
    within = []

    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        mb = mean_by_subject.loc[sampled]
        b = slope_with_intercept(mb["omc"].to_numpy(), mb["pred"].to_numpy())
        if np.isfinite(b):
            between.append(b)

        wb = pd.concat([by_subject[s] for s in sampled], ignore_index=True)
        w = slope_through_origin(wb["omc_centred"], wb["pred_centred"])
        if np.isfinite(w):
            within.append(w)

    return (
        np.percentile(between, [2.5, 97.5]),
        np.percentile(within, [2.5, 97.5]),
        len(between),
        len(within),
    )


def repeated_measures_loa(diff, subject):
    """Equal-participant bias and random-effects limits of agreement."""
    z = pd.DataFrame({"diff": np.asarray(diff), "subject": np.asarray(subject)})
    grouped = z.groupby("subject")["diff"]
    means = grouped.mean()
    counts = grouped.size().astype(float)
    grand = z["diff"].mean()
    n_groups = len(means)
    n_total = int(counts.sum())
    ss_between = float(np.sum(counts * (means - np.average(means, weights=counts)) ** 2))
    ss_within = float(grouped.apply(lambda x: np.sum((x - x.mean()) ** 2)).sum())
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups)
    n0 = (n_total - float(np.sum(counts**2)) / n_total) / (n_groups - 1)
    var_between = max((ms_between - ms_within) / n0, 0.0)
    sd_total = np.sqrt(var_between + ms_within)
    return grand, grand - 1.96 * sd_total, grand + 1.96 * sd_total


def analyse_group(d, model, parameter, n_boot=N_BOOT, seed=SEED):
    """Compute absolute, between-participant, and within-participant agreement."""
    d = d.loc[(d.model == model) & (d.parameter == parameter)].dropna(
        subset=["subject", "omc", "pred"]
    )
    means = d.groupby("subject", as_index=False)[["omc", "pred"]].mean()
    centred = centred_rows(d)
    diff = d["pred"] - d["omc"]
    centred_diff = centred["pred_centred"] - centred["omc_centred"]
    beta_between = slope_with_intercept(means["omc"], means["pred"])
    beta_within = slope_through_origin(centred["omc_centred"], centred["pred_centred"])
    ci_between, ci_within, n_between_boot, n_within_boot = bootstrap_slopes(
        d, n_boot=n_boot, seed=seed
    )
    rm_bias, rm_loa_low, rm_loa_high = repeated_measures_loa(diff, d["subject"])

    return {
        "model": MODEL_LABELS[model],
        "parameter": parameter,
        "n_subjects": d["subject"].nunique(),
        "n_observations": len(d),
        "bias": diff.mean(),
        "mae": diff.abs().mean(),
        "rm_bias": rm_bias,
        "rm_loa_low": rm_loa_low,
        "rm_loa_high": rm_loa_high,
        "beta_between": beta_between,
        "beta_between_ci_low": ci_between[0],
        "beta_between_ci_high": ci_between[1],
        "beta_within": beta_within,
        "beta_within_ci_low": ci_within[0],
        "beta_within_ci_high": ci_within[1],
        "within_mae": centred_diff.abs().mean(),
        "within_rmse": np.sqrt(np.mean(centred_diff**2)),
        "n_between_boot": n_between_boot,
        "n_within_boot": n_within_boot,
    }


def identity_limits(x, y, pad=0.06):
    """Return shared finite plotting limits with a fractional margin."""
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    span = max(hi - lo, 1.0)
    return lo - pad * span, hi + pad * span


def make_figure(data, summary, figure_out=FIGURE_OUT):
    """Draw paired between/within calibration panels and save PDF plus PNG."""
    colours = {"BILSTM": "#0072B2", "ATT": "#D55E00", "DIFF": "#009E73"}
    fig, axes = plt.subplots(3, 6, figsize=(13.8, 7.2), constrained_layout=True)
    rng = np.random.default_rng(SEED)

    for row, parameter in enumerate(PARAMETERS):
        for model_index, model in enumerate(MODELS):
            d = data.loc[(data.model == model) & (data.parameter == parameter)].dropna()
            means = d.groupby("subject", as_index=False)[["omc", "pred"]].mean()
            centred = centred_rows(d)
            result = summary.loc[
                (summary.model == MODEL_LABELS[model]) & (summary.parameter == parameter)
            ].iloc[0]

            ax_b = axes[row, model_index * 2]
            ax_w = axes[row, model_index * 2 + 1]
            colour = colours[model]

            ax_b.scatter(means.omc, means.pred, s=22, color=colour, edgecolor="white", linewidth=0.4)
            lo, hi = identity_limits(means.omc, means.pred)
            ax_b.plot([lo, hi], [lo, hi], color="0.55", linestyle="--", linewidth=0.8)
            intercept = means.pred.mean() - result.beta_between * means.omc.mean()
            ax_b.plot([lo, hi], intercept + result.beta_between * np.array([lo, hi]), color=colour, linewidth=1.1)
            ax_b.set_xlim(lo, hi)
            ax_b.set_ylim(lo, hi)
            ax_b.text(
                0.04,
                0.96,
                f"$\\beta_B$={result.beta_between:.2f}\n[{result.beta_between_ci_low:.2f}, {result.beta_between_ci_high:.2f}]",
                transform=ax_b.transAxes,
                va="top",
                fontsize=7,
            )

            draw = centred
            if len(draw) > 700:
                draw = draw.iloc[rng.choice(len(draw), size=700, replace=False)]
            ax_w.scatter(
                draw.omc_centred,
                draw.pred_centred,
                s=4,
                color=colour,
                alpha=0.14,
                linewidth=0,
                rasterized=True,
            )
            lo, hi = identity_limits(centred.omc_centred, centred.pred_centred)
            ax_w.plot([lo, hi], [lo, hi], color="0.55", linestyle="--", linewidth=0.8)
            ax_w.plot([lo, hi], result.beta_within * np.array([lo, hi]), color=colour, linewidth=1.1)
            ax_w.set_xlim(lo, hi)
            ax_w.set_ylim(lo, hi)
            ax_w.text(
                0.04,
                0.96,
                f"$\\beta_W$={result.beta_within:.2f}\n[{result.beta_within_ci_low:.2f}, {result.beta_within_ci_high:.2f}]",
                transform=ax_w.transAxes,
                va="top",
                fontsize=7,
            )

            if row == 0:
                ax_b.set_title(f"{MODEL_LABELS[model]}\nbetween", fontsize=9)
                ax_w.set_title(f"{MODEL_LABELS[model]}\nwithin", fontsize=9)
            if model_index == 0:
                ax_b.set_ylabel(f"{PARAMETER_LABELS[parameter]}\nreconstructed (cm)", fontsize=8)
            if row == 2:
                ax_b.set_xlabel("OMC mean (cm)", fontsize=8)
                ax_w.set_xlabel("OMC deviation (cm)", fontsize=8)
            ax_b.tick_params(labelsize=7)
            ax_w.tick_params(labelsize=7)
            ax_b.spines[["top", "right"]].set_visible(False)
            ax_w.spines[["top", "right"]].set_visible(False)

    figure_out = Path(figure_out)
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out, bbox_inches="tight")
    fig.savefig(figure_out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the spatial calibration analysis from a tidy per-event CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--means-out", type=Path, default=MEANS_OUT)
    parser.add_argument("--figure-out", type=Path, default=FIGURE_OUT)
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data = data.loc[data.parameter.isin(PARAMETERS) & data.model.isin(MODELS)].copy()

    rows = [
        analyse_group(data, model, parameter, n_boot=args.bootstrap, seed=args.seed)
        for model in MODELS
        for parameter in PARAMETERS
    ]
    summary = pd.DataFrame(rows)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)

    means = (
        data.groupby(["model", "parameter", "subject"], as_index=False)[["omc", "pred"]]
        .mean()
        .assign(model=lambda x: x.model.map(MODEL_LABELS))
    )
    args.means_out.parent.mkdir(parents=True, exist_ok=True)
    means.to_csv(args.means_out, index=False)
    make_figure(data, summary, args.figure_out)

    columns = [
        "model",
        "parameter",
        "bias",
        "mae",
        "beta_between",
        "beta_between_ci_low",
        "beta_between_ci_high",
        "beta_within",
        "beta_within_ci_low",
        "beta_within_ci_high",
        "within_rmse",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

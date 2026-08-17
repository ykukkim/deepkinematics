"""Create the combined joint-position reconstruction figure."""

import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "plots"
filename = "pose_rotation_errors_by_region_and_age.pdf"
os.makedirs(output_dir, exist_ok=True)

models = ["BiLSTM", "ATT", "DIFF"]
groups = ['All', 'Young', 'Old']
regions = ["Full Body", "Upper Body", "Lower Body"]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

# WIDTH OF MODEL GROUP
width = 0.2
x = np.arange(len(groups))

# === ENTER YOUR DATA (unchanged) ===

pose_means = np.array([
    [[4.66, 4.78, 4.43],
     [4.52, 4.45, 4.68],
     [4.80, 5.11, 4.18]],

    [[5.67, 5.63, 6.31],
     [5.64, 5.12, 6.75],
     [5.69, 5.60, 5.87]],

    [[5.07, 4.99, 5.24],
     [4.83, 4.64, 5.20],
     [5.32, 5.34, 5.29]]
])

pose_stds = np.array([
    [[2.14, 2.09, 2.25],
     [2.26, 2.19, 2.41],
     [2.03, 1.99, 2.10]],

    [[3.02, 2.55, 4.03],
     [3.22, 2.61, 4.49],
     [2.83, 2.48, 3.56]],

    [[2.53, 2.13, 3.33],
     [2.57, 2.15, 3.40],
     [2.49, 2.11, 3.27]]
])

rot_means = np.array([
    [[11.32, 11.30, 11.76],
     [9.84, 7.71, 14.31],
     [12.81, 14.53, 9.22]],

    [[12.26, 12.02, 13.20],
     [10.64, 8.33, 15.51],
     [13.88, 15.31, 10.89]],

    [[10.33, 10.94, 9.45],
     [7.93, 7.20, 9.46],
     [12.73, 14.29, 9.43]]
])

rot_stds = np.array([
    [[2.89, 2.43, 3.85],
     [3.06, 2.55, 4.14],
     [2.72, 2.31, 3.57]],

    [[3.55, 3.00, 4.69],
     [3.68, 3.05, 5.00],
     [3.41, 2.96, 4.37]],

    [[3.23, 2.79, 4.15],
     [3.37, 2.86, 4.43],
     [3.08, 2.71, 3.87]]
])

# --------------------------
# Plotting
# --------------------------
x = np.arange(len(groups))   # positions for age groups
width = 0.22                 # bar width
filename = "pose_rotation_errors_by_region_and_age.pdf"

fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.reshape(3, 2)

for r_idx, region in enumerate(regions):

    # Column 0: Pose error
    ax_pose = axs[r_idx, 0]
    for j, model in enumerate(models):
        vals = pose_means[j, :, r_idx]
        errs = pose_stds[j, :, r_idx]
        ax_pose.bar(
            x + j * width - width,
            vals,
            width,
            yerr=errs,
            color=colors[j],
            label=model if r_idx == 0 else "",
            error_kw=dict(elinewidth=0.7, capsize=3)
        )
    # ground truth reference (zero error)
    ax_pose.hlines(0, -0.5, len(groups) - 0.5, linestyles="dashed", colors="black", linewidth=1.0)

    ax_pose.set_title(region, fontsize=13)
    ax_pose.set_ylabel("Pose Error (cm)", fontsize=11)
    ax_pose.set_xticks(x)
    ax_pose.set_xticklabels(groups, fontsize=10)
    ax_pose.tick_params(axis="y", labelsize=10)
    ax_pose.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Column 1: Joint rotation error
    ax_rot = axs[r_idx, 1]
    for j, model in enumerate(models):
        vals = rot_means[j, :, r_idx]
        errs = rot_stds[j, :, r_idx]
        ax_rot.bar(
            x + j * width - width,
            vals,
            width,
            yerr=errs,
            color=colors[j],
            label=model if r_idx == 0 else "",
            error_kw=dict(elinewidth=0.7, capsize=3)
        )
    ax_rot.hlines(0, -0.5, len(groups) - 0.5, linestyles="dashed", colors="black", linewidth=1.0)

    ax_rot.set_title(region, fontsize=13)
    ax_rot.set_ylabel("Joint Rotation Error (°)", fontsize=11)
    ax_rot.set_xticks(x)
    ax_rot.set_xticklabels(groups, fontsize=10)
    ax_rot.tick_params(axis="y", labelsize=10)
    ax_rot.grid(True, axis="y", linestyle="--", alpha=0.5)

# Legend from first row
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.0)
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(output_dir, filename), format="pdf")
plt.close()

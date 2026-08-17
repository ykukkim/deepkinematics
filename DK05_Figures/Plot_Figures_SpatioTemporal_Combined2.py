"""Create the alternate combined spatiotemporal comparison figure."""

import matplotlib.pyplot as plt
import numpy as np
import os

# Directory to save plot
output_dir = "plots"
filename = "spatiotemporal_group_comparison_coloured_GT_lines2.pdf"
os.makedirs(output_dir, exist_ok=True)

# Model and group setup
models = ['BiLSTM', 'ATT', 'DIFF']
groups = ['All', 'Young', 'Old']
parameters = [
    "R_Step_Length", "L_Step_Length",
    "R_Stride_Length", "L_Stride_Length",
    "R_Step_Width", "L_Step_Width",
    "R_Stride_Width", "L_Stride_Width"
]

colors = ['#1f77b4', '#2ca02c', '#ff7f0e']       # BiLSTM, ATT, DIFF
gt_colors = ['black', 'darkred', 'darkgreen']    # GT All, GT Young, GT Old
width = 0.2
x = np.arange(len(groups))

# Means and standard deviations for each model, parameter, and group
means = np.array([
    [[52.5, 40.85, 52.79], [53.53, 39.62, 55.93], [51.46, 42.09, 49.66]],
    [[52.89, 39.08, 52.44], [53.65, 37.53, 54.64], [52.13, 40.64, 50.24]],
    [[102.36, 75.26, 101.02], [104.44, 72.56, 106.46], [100.27, 77.96, 95.58]],
    [[102.90, 75.70, 101.58], [104.73, 72.79, 106.74], [101.06, 78.62, 96.42]],
    [[8.40, 9.56, 10.43], [8.07, 8.98, 10.59], [8.74, 10.14, 10.28]],
    [[8.27, 9.50, 10.29], [7.97, 9.19, 10.38], [8.57, 9.81, 10.20]],
    [[1.70, 2.91, 2.57], [1.48, 2.90, 2.42], [1.92, 2.92, 2.72]],
    [[1.81, 3.11, 2.49], [1.68, 2.75, 2.37], [1.94, 3.46, 2.62]]
])

stds = np.array([
    [[10.2, 9.94, 11.61], [9.37, 8.98, 10.38], [11.02, 10.89, 12.83]],
    [[8.95, 9.45, 10.52], [8.30, 8.56, 9.62], [9.59, 10.34, 11.42]],
    [[10.30, 10.26, 11.68], [8.53, 9.00, 10.07], [12.06, 11.52, 13.28]],
    [[16.43, 14.91, 18.24], [9.87, 9.44, 11.41], [22.98, 20.38, 25.07]],
    [[4.57, 5.20, 5.16], [3.97, 4.91, 4.92], [5.17, 5.49, 5.40]],
    [[4.23, 5.06, 4.84], [4.21, 4.87, 4.88], [4.25, 5.25, 4.80]],
    [[1.43, 2.25, 2.03], [1.26, 2.15, 1.91], [1.59, 2.36, 2.14]],
    [[1.68, 2.38, 1.99], [1.61, 2.13, 1.89], [1.61, 2.13, 1.89]]
])

gt_values = np.array([
    [55.17, 58.14, 52.2],
    [54.31, 56.96, 51.67],
    [106.72, 112.20, 101.22],
    [107.16, 112.51, 101.99],
    [8.15, 8.24, 6.98],
    [7.62, 8.18, 6.56],
    [1.87, 1.61, 2.19],
    [1.90, 1.59, 2.34]
])

# Plot setup: 4 rows (metrics), 2 columns (right / left)
fig, axs = plt.subplots(4, 2, figsize=(14, 12))
axs = axs.flatten()

for i, param in enumerate(parameters):

    ax = axs[i]

    # Base name and side
    base_name = param.replace("R_", "").replace("L_", "")
    side = "Right" if param.startswith("R_") else "Left"
    title = f"{base_name} ({side})"

    # Bars for each model
    for j, model in enumerate(models):
        vals = means[i, :, j]
        errs = stds[i, :, j]
        ax.bar(
            x + j * width - width,
            vals,
            width,
            yerr=errs,
            color=colors[j],
            label=models[j] if i == 0 else "",
            error_kw=dict(elinewidth=0.7, capsize=3)
        )

    # Ground truth dashed lines per group
    for xi in range(len(groups)):
        ax.hlines(
            gt_values[i, xi],
            xi - 0.4,
            xi + 0.4,
            colors=gt_colors[xi],
            linestyles='dashed',
            linewidth=1.2
        )

    # Titles and axes
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("cm", fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Updated metric-specific y-limits
    if "Step_Length" in param:
        ax.set_ylim(0, 80)
    elif "Stride_Width" in param:
        ax.set_ylim(0, 8)
    elif "Stride_Length" in param:
        ax.set_ylim(0, 140)
    elif "Width" in base_name:
        ax.set_ylim(0, 15)

# Legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels + ['GT All', 'GT Young', 'GT Old'],
    loc='lower center',
    ncol=6,
    frameon=False,
    fontsize=12
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(output_dir, filename), format='pdf')
plt.close()

"""Plot participant-level spatiotemporal gait-parameter comparisons."""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data setup
models = ['BiLSTM', 'DIFF', 'ATT']
regions = ['Full Body', 'Upper Body', 'Lower Body']

# Directory to save plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Updated group data
parameters = [
    "R_Step_Length", "L_Step_Length",
    "R_Stride_Length", "L_Stride_Length",
    "R_Step_Width", "L_Step_Width",
    "R_Stride_Width", "L_Stride_Width"
]
# Spatiotemporal
gt_values = [54.68, 68.98, 106.72, 107.16, 8.15, 7.62, 1.87, 1.90]
birnn_values = [52.99, 66.91, 102.36, 102.90, 8.40, 8.27, 1.70, 1.81]
att_values   = [41.29, 50.57, 75.26, 75.70, 9.56, 9.50, 2.91, 3.11]
diff_values  = [53.35, 67.64, 101.02, 101.58, 10.43, 10.29, 2.57, 2.49]

# Aesthetic settings
vivid_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
width = 0.25
x = np.arange(len(regions))

# Function to save spatiotemporal plots
def save_spatiotemporal_plot(param_group, title, filename):
    param_indices = [parameters.index(p) for p in param_group]
    x = np.arange(len(param_group))
    birnn_vals = [birnn_values[i] for i in param_indices]
    diff_vals = [diff_values[i] for i in param_indices]
    att_vals = [att_values[i] for i in param_indices]
    gt_vals = [gt_values[i] for i in param_indices]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, birnn_vals, width, label='BiLSTM', color=vivid_colors[0])
    ax.bar(x, diff_vals, width, label='DIFF', color=vivid_colors[1])
    ax.bar(x + width, att_vals, width, label='ATT', color=vivid_colors[2])
    for i, gt in enumerate(gt_vals):
        ax.hlines(gt, x[i] - 0.3, x[i] + 0.3, colors='black', linestyles='dashed', linewidth=1)

    ax.set_ylabel('Value (cm)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(param_group)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Save each spatiotemporal figure
save_spatiotemporal_plot(["R_Step_Length", "L_Step_Length"], "Step Length Comparison", "step_length.pdf")
save_spatiotemporal_plot(["R_Stride_Length", "L_Stride_Length"], "Stride Length Comparison", "stride_length.pdf")
save_spatiotemporal_plot(["R_Step_Width", "L_Step_Width"], "Step Width Comparison", "step_width.pdf")
save_spatiotemporal_plot(["R_Stride_Width", "L_Stride_Width"], "Stride Width Comparison", "stride_width.pdf")

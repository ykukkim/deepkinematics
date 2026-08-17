"""
Python port of DK04_Gait06_FindingBestModel.m

Original MATLAB header (preserved for reference, copy-pasted from the .m
file across the Gait05/Gait06 scripts -- it describes the Excel-writing
scripts, not this one specifically):
  This code writes the mean and standard deviation of all gait parameters
  and MoS in one Excel file --> left and right are separated
  created by Paciane, Nov. 2022 / Adapted by Alex, Oct. 2023
  Gait00 - Gait04 needs to be run beforehand

What this script actually does: reads the Summary_Report.xlsx that
DK04_Gait02_Euclidean_Distance.py writes into each "...-acc_gyro/Results/"
folder (sheet 1 = Pose_error, sheet 2 = Joint Error), averages the
per-marker mean columns across all markers, and reports which folder
(i.e. which trained-model config) has the lowest overall error.

Dependencies: Functions_py/bland_altman_plots.py -> find_folders_ending_with_string
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from Functions_py.bland_altman_plots import find_folders_ending_with_string


def main():
    """Report the model folder with the lowest mean pose/joint error."""
    project_root = Path(__file__).resolve().parents[3]
    default_models = os.environ.get(
        'DEEPKINEMATICS_MODEL_DIR', str(project_root / '00_Results')
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', default=default_models,
                        help='root folder containing model-group directories')
    parser.add_argument('--group', default='Full_DIFF')
    parser.add_argument('--folder-suffix', default='-acc_gyro')
    args = parser.parse_args()

    base_directory = os.path.join(args.models_root, args.group)
    matching_folders = find_folders_ending_with_string(
        base_directory, args.folder_suffix
    )

    all_results = []  # list of dicts: {folder, pose_error, joint_error}

    for folder in matching_folders:
        test_path2 = os.path.join(folder, 'Results')
        xlsx_files = sorted(glob.glob(os.path.join(test_path2, '*.xlsx')))

        if not xlsx_files:
            print(f'Warning: No Excel file found in {test_path2}')
            continue

        # NOTE: the original MATLAB used `xlsx_files.folder`/`.name` without
        # indexing, which only works cleanly for a single match -- so this
        # assumes (as the original implicitly did) exactly one xlsx per
        # Results folder. If more than one exists, only the first is used;
        # flagging this instead of silently picking one with no comment.
        if len(xlsx_files) > 1:
            print(f'Warning: multiple xlsx files in {test_path2}, using the first: {xlsx_files[0]}')
        xlsx_full_path = xlsx_files[0]

        pose_error = pd.read_excel(xlsx_full_path, sheet_name=0)
        joint_error = pd.read_excel(xlsx_full_path, sheet_name=1)

        all_results.append({
            'folder': os.path.basename(os.path.normpath(folder)),
            'pose_error': pose_error,
            'joint_error': joint_error,
        })

    if not all_results:
        print('No results found.')
        return None

    folder_avg = np.full(len(all_results), np.nan)

    for i, res in enumerate(all_results):
        # columns 2:end in MATLAB (1-based) == all columns after the first
        # (Subject) column here.
        pose_vals = res['pose_error'].iloc[:, 1:].to_numpy(dtype=float)
        joint_vals = res['joint_error'].iloc[:, 1:].to_numpy(dtype=float)

        # MATLAB's mean(X, 'omitnan') on a matrix averages down each
        # column (dim 1), giving a per-column mean row vector; the
        # (unused-downstream) std() call in the original is omitted here.
        pose_col_means = np.nanmean(pose_vals, axis=0)
        joint_col_means = np.nanmean(joint_vals, axis=0)

        # Folder-level average = mean over ALL markers (pose + joint
        # combined), matching `mean([total_subject_mean_pose,
        # total_subject_mean_jot], 'omitnan')` on the concatenated row
        # vectors.
        folder_avg[i] = np.nanmean(np.concatenate([pose_col_means, joint_col_means]))

    best_idx = int(np.nanargmin(folder_avg))
    best_folder = all_results[best_idx]['folder']

    print(f'Best performing folder: {best_folder}')
    return best_folder


if __name__ == '__main__':
    main()

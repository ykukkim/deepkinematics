"""
Python port of DK04_Gait06_FindingBestModel_Euclidean.m

Reads the Gait_Parameters_Mean_R_L.xlsx that DK04_Gait05_Mean_into_Excel_R_L.py
writes into each "...-acc_gyro/Results/Gait/" folder (sheet 1 = Mean Values,
sheet 2 = Error-Percentage), averages the Error-Percentage columns per
subject and then across subjects, and reports which folder (trained-model
config) has the lowest overall error.

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
    """Report the model folder with the lowest mean gait-parameter error."""
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

    all_results = []  # list of dicts: {folder, data1 (Mean Values), data2 (Error-Percentage)}

    for folder in matching_folders:
        test_path2 = os.path.join(folder, 'Results', 'Gait')
        xlsx_files = sorted(glob.glob(os.path.join(test_path2, '*.xlsx')))

        if not xlsx_files:
            print(f'Warning: No Excel file found in {test_path2}')
            continue

        if len(xlsx_files) > 1:
            print(f'Warning: multiple xlsx files in {test_path2}, using the first: {xlsx_files[0]}')
        xlsx_full_path = xlsx_files[0]

        data1 = pd.read_excel(xlsx_full_path, sheet_name=0)  # Mean Values
        data2 = pd.read_excel(xlsx_full_path, sheet_name=1)  # Error-Percentage

        all_results.append({
            'folder': os.path.basename(os.path.normpath(folder)),
            'data1': data1,
            'data2': data2,
        })

    if not all_results:
        print('No results found.')
        return None

    folder_avg = np.full(len(all_results), np.nan)

    for i, res in enumerate(all_results):
        table = res['data2']  # Error-Percentage sheet
        subject_col = table.iloc[:, 0]
        unique_subjects = subject_col.unique()

        subject_means = []
        for subj in unique_subjects:
            rows = table[subject_col == subj]
            # columns 3:end in MATLAB (1-based) == everything after the
            # first two columns (Subject, Trial) here.
            numeric_data = rows.iloc[:, 2:].to_numpy(dtype=float)
            # MATLAB's mean(numeric_data) with >1 row averages down each
            # column (dim 1) -> one row per subject.
            subject_means.append(np.nanmean(numeric_data, axis=0) if numeric_data.shape[0] > 1
                                  else np.mean(numeric_data, axis=0))

        subject_means = np.vstack(subject_means)
        total_subject_mean = np.nanmean(subject_means, axis=0)  # per-column mean across subjects
        folder_avg[i] = np.nanmean(total_subject_mean)  # scalar mean over all columns

    best_idx = int(np.nanargmin(folder_avg))
    best_folder = all_results[best_idx]['folder']

    print(f'Best performing folder: {best_folder}')
    return best_folder


if __name__ == '__main__':
    main()

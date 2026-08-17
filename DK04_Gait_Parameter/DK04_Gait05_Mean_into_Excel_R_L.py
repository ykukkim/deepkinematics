"""
Python port of DK04_Gait05_Mean_into_Excel_R_L.m

Original MATLAB header (preserved for reference):
  This code writes the mean and standard deviation of all gait parameters
  and MoS in one Excel file --> left and right are separated
  created by Paciane, Nov. 2022
  Adapted by Alex, Oct. 2023
  mean, std, median is calculated and stored in mat file in Gait02
  Gait00 - Gait04 needs to be run beforehand

Dependencies:
  Functions_py/bland_altman_plots.py -> find_folders_ending_with_string
  DK04_Gait03_agreement.py           -> load_gaitsummary (reused rather than
                                         reimplemented -- same v7.3-mat / mat73
                                         / scipy.io.loadmat fallback chain,
                                         already returns exactly the
                                         {subject: {trial: {param: array}}}
                                         shape this script needs)
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from Functions_py.bland_altman_plots import find_folders_ending_with_string
from DK04_Gait03_agreement import load_gaitsummary


COLUMN_NAMES = [
    'Subject', 'Trial', 'Group',
    'R_Mean_Step_Width', 'L_Mean_Step_Width', 'R_STD_Step_Width', 'L_STD_Step_Width',
    'R_Mean_Stride_Width', 'L_Mean_Stride_Width', 'R_STD_Stride_Width', 'L_STD_Stride_Width',
    'R_Mean_Step_Length', 'L_Mean_Step_Length', 'R_STD_Step_Length', 'L_STD_Step_Length',
    'R_Mean_Stride_Length', 'L_Mean_Stride_Length', 'R_STD_Stride_Length', 'L_STD_Stride_Length',
]

COLUMN_NAMES_ERROR = [
    'Subject', 'Trial',
    'R_Mean_Step_Width (%)', 'L_Mean_Step_Width (%)', 'R_STD_Step_Width (%)', 'L_STD_Step_Width (%)',
    'R_Mean_Stride_Width (%)', 'L_Mean_Stride_Width (%)', 'R_STD_Stride_Width (%)', 'L_STD_Stride_Width (%)',
    'R_Mean_Step_Length (%)', 'L_Mean_Step_Length (%)', 'R_STD_Step_Length (%)', 'L_STD_Step_Length (%)',
    'R_Mean_Stride_Length (%)', 'L_Mean_Stride_Length (%)', 'R_STD_Stride_Length (%)', 'L_STD_Stride_Length (%)',
]

GAIT_PARAM_FIELDS = [
    'stepWidthR', 'stepWidthL', 'strideWidthR', 'strideWidthL',
    'stepLengthR', 'stepLengthL', 'strideLengthR', 'strideLengthL',
]


def _matlab_stat(vals):
    """mean and sample std, matching MATLAB's mean()/std() defaults (std
    uses ddof=1, i.e. N-1; MATLAB defines std of a single-element vector as
    0, not NaN)."""
    if vals is None or len(vals) == 0:
        return np.nan, np.nan
    arr = np.asarray(vals, dtype=float)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def main():
    """Create one left/right gait-summary workbook per matching model folder."""
    project_root = Path(__file__).resolve().parents[3]
    default_models = os.environ.get(
        'DEEPKINEMATICS_MODEL_DIR', str(project_root / '00_Results')
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', default=default_models,
                        help='root folder containing model-group directories')
    parser.add_argument('--group', default='Full_DIFF',
                        help='model group below --models-root (default: Full_DIFF)')
    parser.add_argument('--folder-suffix', default='-acc_gyro',
                        help='only process model folders ending with this suffix')
    args = parser.parse_args()

    base_directory = os.path.join(args.models_root, args.group)
    matching_folders = find_folders_ending_with_string(
        base_directory, args.folder_suffix
    )

    for folder in matching_folders:
        test_path2 = os.path.join(folder, 'Results')
        mat_files = sorted(glob.glob(os.path.join(test_path2, 'GaitSummary_DL.mat')))

        all_mean_rows = []

        for mat_path in mat_files:
            gait_summary = load_gaitsummary(mat_path)

            for part, trials in gait_summary.items():
                # Exclude joint_rot trials, matching the original's
                # `mask = ~contains(fnames, 'joint_rot')`.
                trial_names = [t for t in trials.keys() if 'joint_rot' not in t]

                for trial in trial_names:
                    gp = trials[trial]
                    stats = {f: _matlab_stat(gp.get(f)) for f in GAIT_PARAM_FIELDS}

                    group_label = 'Test'  # the original overwrites its own Group var with this literal
                    all_mean_rows.append([
                        part, trial, group_label,
                        stats['stepWidthR'][0], stats['stepWidthL'][0], stats['stepWidthR'][1], stats['stepWidthL'][1],
                        stats['strideWidthR'][0], stats['strideWidthL'][0], stats['strideWidthR'][1], stats['strideWidthL'][1],
                        stats['stepLengthR'][0], stats['stepLengthL'][0], stats['stepLengthR'][1], stats['stepLengthL'][1],
                        stats['strideLengthR'][0], stats['strideLengthL'][0], stats['strideLengthR'][1], stats['strideLengthL'][1],
                    ])

        if not all_mean_rows:
            continue

        mean_table = pd.DataFrame(all_mean_rows, columns=COLUMN_NAMES)

        # ---- Calculate error with respect to ground truth -------------
        error_rows = []
        for subject in mean_table['Subject'].unique():
            sub_table = mean_table[mean_table['Subject'] == subject].reset_index(drop=True)

            # NOTE: faithfully reproduces the original's positional
            # assumption. The original indexes meanArray rows 1..4 (MATLAB,
            # 1-based) BY POSITION, not by trial name, to pair up
            # "with root" vs "without root" gt/prediction rows. That only
            # gives the intended pairing if this subject's trials appear
            # in a specific, consistent 4-row order -- which in turn
            # depends on the .mat file's own field (insertion) order being
            # preserved all the way through `load_gaitsummary`'s dict
            # iteration. Not verified against real data here -- flagging
            # rather than silently reordering or "fixing" it.
            trial_first = str(sub_table.loc[0, 'Trial'])
            trial_name = trial_first.split('_pose')[0]
            mean_array = sub_table.iloc[:, 3:].to_numpy(dtype=float)  # numeric columns only

            if mean_array.shape[0] < 4:
                # Not enough trial rows for this subject to compute both
                # error pairs -- the original .m would error out (index
                # out of bounds) in this situation; skipping here instead
                # of crashing the whole folder's output.
                continue

            error_val = np.abs((mean_array[3, :] - mean_array[2, :]) / mean_array[2, :]) * 100
            error_rows.append([subject, f'{trial_name}_root_rec', *error_val])

            error_val_root = np.abs((mean_array[1, :] - mean_array[0, :]) / mean_array[0, :]) * 100
            # NOTE: the original's own label for this second row is
            # '_root_rec_X' (not, say, '_no_root') -- preserved verbatim,
            # it reads like a naming slip in the source but I have no basis
            # to rename it.
            error_rows.append([subject, f'{trial_name}_root_rec_X', *error_val_root])

        error_table = pd.DataFrame(error_rows, columns=COLUMN_NAMES_ERROR)

        dest_path = os.path.join(test_path2, 'Gait')
        os.makedirs(dest_path, exist_ok=True)
        full_path = os.path.join(dest_path, 'Gait_Parameters_Mean_R_L.xlsx')
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            mean_table.to_excel(writer, sheet_name='Mean Values', index=False)
            error_table.to_excel(writer, sheet_name='Error-Percentage', index=False)

        print(f'Written: {full_path}')


if __name__ == '__main__':
    main()

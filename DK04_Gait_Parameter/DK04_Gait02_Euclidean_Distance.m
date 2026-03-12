%% Gait Analysis
% Extracts all the relevant spatio-temporal gait parameters
% 1. It fills the gap by interpolating the missing datasets, but this
% process should be performed in VICON, such as gap filling.
% 1. It detects the IC and TO using FVA from O'connor et al 2007
% 1. It detects the IC and TO using the wavelet transform on the gyroscope
% from the feet IMUs
% 2. Spatio-temporal parameters are calculated from these events
%   i Midstance & Midswing
%   ii.Duration
%       i   duration Gait cycle
%       ii  duration Stance Phase
%       iii duration Swing Phase
%       iv  duration of double limb support
%   iii. Stride and Step lengths
%   iv. Stepwidth
%   v. Phase Coordination Index
% Yong Kuk Kim
% Adapted by Michelle Gwerder, June 2022
% Adjusted to predicted data by Alexander Häfliger, October 2023

clear all; close all; clc;

%% Define the path and group to analyze
addpath(genpath('Functions'));
TestPath1 = '/Users/yonkim/DataForWork/DeepKinematics/Models_trained';

% Data Group
ifgroup = 0;

switch ifgroup
    case 0
        Group = 'Full_DIFF';
    case 1
        Group = 'PartlyGeneral';
end

%% Loop through all the participants and load the data
baseDirectory = [TestPath1,filesep,Group];
matchingFolders = findFoldersEndingWithString(baseDirectory, '-acc_gyro');

errorLog = [];

for folder_idx = 1:length(matchingFolders)

    folder_in = dir(matchingFolders{folder_idx});
    folder_in(strncmp({folder_in.name}, '.', 1)) = [];
    subjectNames = folder_in(strncmp({folder_in.name}, 'SonE', 4));
    destPath = [matchingFolders{folder_idx},filesep,'Results',filesep,'Summary_Report.xlsx'];

    pose_data = [];
    joint_data = [];
    subject_ids = {};

    for sub_indx = 1:length(subjectNames)
        part = subjectNames(sub_indx).name;
        TestPath2 = [matchingFolders{folder_idx},filesep,subjectNames(sub_indx).name,filesep];
        xlsx_files = dir(fullfile(TestPath2, '*.xlsx'));

        subject_mean_pose = [];
        subject_std_pose = [];
        subject_mean_joint = [];
        subject_std_joint = [];

        for trial = 1:length(xlsx_files)
            try
                xlsx_full_path = fullfile(xlsx_files(trial).folder, xlsx_files(trial).name);
                pos_error = readtable(xlsx_full_path, 'Sheet', 1, 'VariableNamingRule', 'preserve');
                pos_error.Mean = pos_error.Mean * 100;
                pos_error.Std = pos_error.Std * 100;
                joint_error = readtable(xlsx_full_path, 'Sheet', 2, 'VariableNamingRule', 'preserve');
                marker_params = joint_error.Label;
                for m = 1:length(marker_params)
                    marker = marker_params{m};
                    idx = strcmpi(pos_error{:,1}, marker);

                    if any(idx)
                        subject_mean_pose(m, trial) = pos_error{idx, 2};
                        subject_std_pose(m, trial)  = pos_error{idx, 3};
                        subject_mean_joint(m, trial) = joint_error{idx, 2};
                        subject_std_joint(m, trial)  = joint_error{idx, 3};
                    else
                        subject_mean_pose(m, trial) = NaN;
                        subject_std_pose(m, trial)  = NaN;
                        subject_mean_joint(m, trial) = NaN;
                        subject_std_joint(m, trial)  = NaN;
                    end
                end

            catch ME
                errorLog = LogsSONEMS(errorLog, part, ME);
                disp(['Pipeline failed: ', part]);
                continue
            end
        end

        % Average across trials for the subject
        mean_pose_vals  = mean(subject_mean_pose, 2, 'omitnan')';
        std_pose_vals   = mean(subject_std_pose, 2, 'omitnan')';
        mean_joint_vals = mean(subject_mean_joint, 2, 'omitnan')';
        std_joint_vals  = mean(subject_std_joint, 2, 'omitnan')';

        % Store
        pose_data(end+1, :)  = [mean_pose_vals, std_pose_vals];
        joint_data(end+1, :) = [mean_joint_vals, std_joint_vals];
        subject_ids{end+1} = part;
    end

    % Create tables
    pose_table = array2table(pose_data, ...
        'VariableNames', [strcat(marker_params, '_mean'), strcat(marker_params, '_std')]);
    joint_table = array2table(joint_data, ...
        'VariableNames', [strcat(marker_params, '_mean'), strcat(marker_params, '_std')]);

    % Add subject ID column
    pose_table = addvars(pose_table, subject_ids', 'Before', 1, 'NewVariableNames', 'Subject');
    joint_table = addvars(joint_table, subject_ids', 'Before', 1, 'NewVariableNames', 'Subject');

    % Write to Excel (one sheet per folder)
    writetable(pose_table, destPath, 'Sheet', 'Pose_error');
    writetable(joint_table, destPath, 'Sheet', 'Joint Error');

    disp(['✅ Written: ', folder_idx]);
end
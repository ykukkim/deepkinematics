%% This code writes the mean and standard deviation of all gait parameters and Mos in one Excel file
% --> left and right are seperated
% created by Paciane, Nov. 2022
% Adapted by Alex, Oct. 2023
% mean, std, median is calculated and stored in mat file in Gait02
% Gait00 - Gait04 needs to be run beforehand

clc;close all;clear all;

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

baseDirectory = [TestPath1,filesep,Group];
matchingFolders = findFoldersEndingWithString(baseDirectory, '-acc_gyro');
folders_name = dir(baseDirectory);
folders_name(strncmp({folders_name.name}, '.', 1)) = [];

%% Load the Dat
all_results = struct('folder', [], 'data1', [], 'data2', []);

for folders_idx = 1:length(matchingFolders)

    TestPath2 = fullfile(matchingFolders{folders_idx}, 'Results');

    xlsx_files = dir(fullfile(TestPath2, '*.xlsx'));
    if isempty(xlsx_files)
        warning('No Excel file found in %s', TestPath2);
        continue
    end

    % Load Excel sheets
    xlsx_full_path = fullfile(xlsx_files.folder, xlsx_files.name);
    data1 = readtable(xlsx_full_path, 'Sheet', 1,'VariableNamingRule', 'preserve');
    data2 = readtable(xlsx_full_path, 'Sheet', 2,'VariableNamingRule', 'preserve');

    % Save to structured array
    all_results(folders_idx).folder = folders_name(folders_idx).name;
    all_results(folders_idx).data1 = data1;
    all_results(folders_idx).data2 = data2;

end

% Preallocate array to store mean values per folder
num_folders = length(all_results);
folder_avg = zeros(num_folders, 1);

for i = 1:num_folders
    Pose_error = all_results(i).data1;
    Jot_error = all_results(i).data2;

    total_subject_mean_pose = mean(Pose_error{:,2:end}, 'omitnan');
    total_subject_std_pose = std(Pose_error{:,2:end}, 'omitnan');

    total_subject_mean_jot = mean(Jot_error{:,2:end}, 'omitnan');
    total_subject_std_jot = std(Jot_error{:,2:end}, 'omitnan');

    % Folder-level mean = mean over subjects
    folder_avg(i) = mean([total_subject_mean_pose, total_subject_mean_jot], 'omitnan');  % mean over all markers
end

% Find best folder (lowest average)
[~, best_idx] = min(folder_avg);
best_folder = all_results(best_idx).folder;

fprintf('Best performing folder: %s\n', best_folder);

%BiRNNVRN-2x256x128-2x1024-pol1-jrotl10-fl10-phl1-cl10-cosine-adamw-lr0.005-wd0.1-ns6-bs64-spl512-FK-acc_gyro
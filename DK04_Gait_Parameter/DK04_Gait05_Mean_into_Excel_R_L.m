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

% columnNames = {"Subject" 'Trial' 'Group' 'R_Mean_Step_Width' 'L_Mean_Step_Width' 'R_STD_Step_Width' 'L_STD_Step_Width' ...
%     'R_Mean_Step_Length' 'L_Mean_Step_Length' 'R_STD_Step_Length' 'L_STD_Step_Length'  'R_Mean_Stride_Length' 'L_Mean_Stride_Length'  ...
%     'R_STD_Stride_Length' 'L_STD_Stride_Length'  'R_Mean_Stride_Time' 'L_Mean_Stride_Time'  'R_STD_Stride_Time' 'L_STD_Stride_Time' ...
%     'R_Mean_Double_Limb_Support_Time'  'L_Mean_Double_Limb_Support_Time'  'R_STD_Double_Limb_Support_Time'  'L_STD_Double_Limb_Support_Time'};
% 
% columnNames = {'Subject' 'Trial' 'Group' ...
%     'R_Mean_Step_Width' 'L_Mean_Step_Width' 'R_STD_Step_Width' 'L_STD_Step_Width' ...
%     'R_Mean_Stride_Width' 'L_Mean_Stride_Width' 'R_STD_Stride_Width' 'L_STD_Stride_Width' ...
%     'R_Mean_Step_Length' 'L_Mean_Step_Length' 'R_STD_Step_Length' 'L_STD_Step_Length' ...
%     'R_Mean_Stride_Length' 'L_Mean_Stride_Length' 'R_STD_Stride_Length' 'L_STD_Stride_Length'...
%     'R_Mean_Stride_Time' 'L_Mean_Stride_Time'  'R_STD_Stride_Time' 'L_STD_Stride_Time' ...
%     'R_Mean_Double_Limb_Support_Time'  'L_Mean_Double_Limb_Support_Time'  'R_STD_Double_Limb_Support_Time'  'L_STD_Double_Limb_Support_Time'};

% 2. Spatio-temporal parameters are calculated from these events
%   i Midstance & Midswing
%   ii.Duration
%       i   duration Gait cycle
%       ii  duration Stance Phase
%       iii duration Swing Phase
%       iv  duration of double limb support
%   iii. Stride and Step lengths
%   iv. Stepwidth

columnNames = {'Subject' 'Trial' 'Group' ...
    'R_Mean_Step_Width' 'L_Mean_Step_Width' 'R_STD_Step_Width' 'L_STD_Step_Width' ...
    'R_Mean_Stride_Width' 'L_Mean_Stride_Width' 'R_STD_Stride_Width' 'L_STD_Stride_Width' ...
    'R_Mean_Step_Length' 'L_Mean_Step_Length' 'R_STD_Step_Length' 'L_STD_Step_Length' ...
    'R_Mean_Stride_Length' 'L_Mean_Stride_Length' 'R_STD_Stride_Length' 'L_STD_Stride_Length'};


columnNames_error = {'Subject' 'Trial' 'R_Mean_Step_Width (%)' 'L_Mean_Step_Width (%)' 'R_STD_Step_Width (%)' 'L_STD_Step_Width (%)' ...
    'R_Mean_Stride_Width (%)' 'L_Mean_Stride_Width (%)' 'R_STD_Stride_Width (%)' 'L_STD_Stride_Width (%)' ...
    'R_Mean_Step_Length (%)' 'L_Mean_Step_Length (%)' 'R_STD_Step_Length (%)' 'L_STD_Step_Length (%)' ...
    'R_Mean_Stride_Length (%)' 'L_Mean_Stride_Length (%)' 'R_STD_Stride_Length (%)' 'L_STD_Stride_Length (%)'};

baseDirectory = [TestPath1,filesep,Group];
matchingFolders = findFoldersEndingWithString(baseDirectory, '-acc_gyro');


%% Load the Dat
% Load data from each folder
for folders_name = 1:length(matchingFolders)

    TestPath2 = fullfile(matchingFolders{folders_name}, 'Results');
    mat_files = dir(fullfile(TestPath2, 'GaitSummary_DL.mat'));

    All_mean = columnNames;

    % Load each .mat file
    for k = 1:length(mat_files)
        load(fullfile(mat_files(k).folder, mat_files(k).name));

        % Get subject names
        subjectNames = fieldnames(GaitSummary);

        % Loop through each subject
        for subjectloop = 1:length(subjectNames)

            part = subjectNames{subjectloop};
            fnames = fieldnames(GaitSummary.(part));
            mask = ~contains(fnames, 'joint_rot');
            fnames = fnames(mask);
            % Loop through each trial
            for i = 1:length(fnames)
                trial = fnames{i};

                % Calculate the mean and standard deviation of gait parameters
                mean_stepwR = mean(GaitSummary.(part).(trial).GaitParameters.stepWidthR);
                mean_stepwL = mean(GaitSummary.(part).(trial).GaitParameters.stepWidthL);
                std_stepwR = std(GaitSummary.(part).(trial).GaitParameters.stepWidthR);
                std_stepwL = std(GaitSummary.(part).(trial).GaitParameters.stepWidthL);

                mean_stridewR = mean(GaitSummary.(part).(trial).GaitParameters.strideWidthR);
                mean_stridewL = mean(GaitSummary.(part).(trial).GaitParameters.strideWidthL);
                std_stridewR = std(GaitSummary.(part).(trial).GaitParameters.strideWidthR);
                std_stridewL = std(GaitSummary.(part).(trial).GaitParameters.strideWidthL);

                mean_steplR = mean(GaitSummary.(part).(trial).GaitParameters.stepLengthR);
                mean_steplL = mean(GaitSummary.(part).(trial).GaitParameters.stepLengthL);
                std_steplR = std(GaitSummary.(part).(trial).GaitParameters.stepLengthR);
                std_steplL = std(GaitSummary.(part).(trial).GaitParameters.stepLengthL);

                mean_stridelR = mean(GaitSummary.(part).(trial).GaitParameters.strideLengthR);
                mean_stridelL = mean(GaitSummary.(part).(trial).GaitParameters.strideLengthL);
                std_stridelR = std(GaitSummary.(part).(trial).GaitParameters.strideLengthR);
                std_stridelL = std(GaitSummary.(part).(trial).GaitParameters.strideLengthL);
                %
                %                 mean_stR = mean(GaitSummary.(part).(trial).GaitParameters.durationGaitCycleR);
                %                 mean_stL = mean(GaitSummary.(part).(trial).GaitParameters.durationGaitCycleL);
                %                 std_stR = std(GaitSummary.(part).(trial).GaitParameters.durationGaitCycleR);
                %                 std_stL = std(GaitSummary.(part).(trial).GaitParameters.durationGaitCycleL);
                %
                %                 mean_dlR = mean(GaitSummary.(part).(trial).GaitParameters.dlsR);
                %                 mean_dlL = mean(GaitSummary.(part).(trial).GaitParameters.dlsL);
                %                 std_dlR = std(GaitSummary.(part).(trial).GaitParameters.dlsR);
                %                 std_dlL = std(GaitSummary.(part).(trial).GaitParameters.dlsL);

                % Append the data to All_mean
                Group = 'Test';
                tmp_all_raw = {part, trial, Group, ...
                    mean_stepwR, mean_stepwL, std_stepwR, std_stepwL, ...
                    mean_stridewR, mean_stridewL, std_stridewR, std_stridewL, ...
                    mean_steplR, mean_steplL, std_steplR, std_steplL, ...
                    mean_stridelR, mean_stridelL, std_stridelR, std_stridelL};

                %                 tmp_all_raw = {part, trial, Group, mean_stepwR, mean_stepwL, std_stepwR, std_stepwL, mean_steplR, mean_steplL, std_steplR, std_steplL, ...
                %                     mean_stridelR, mean_stridelL, std_stridelR, std_stridelL, mean_stR, mean_stL, std_stR, std_stL, mean_dlR, mean_dlL, std_dlR, std_dlL};

                All_mean = [All_mean; tmp_all_raw]; %#ok<AGROW>
            end
        end
    end
    %% Calculate Error with respect to ground truth

    meanTable = cell2table(All_mean(2:end,:), "VariableNames", string(columnNames));
    subjectNames = unique(meanTable.Subject);
    errorValues = {};

    for s = 1:size(subjectNames)

        % Find all rows of subject and create temporary table and array
        idx_subject = find(strcmp(subjectNames(s), meanTable.Subject));
        meanTableTmp = meanTable(idx_subject,:);
        trial_name = extractBefore(meanTableTmp.Trial{1},'_pose');
        meanArray = table2array(meanTable(idx_subject, 4:end));

        % Calculate error with respect to ground truth with root added
        errorValTmp = abs((meanArray(4,:) - meanArray(3,:)) ./ meanArray(3,:)) * 100;
        errorValTmp = [{subjectNames{s}}, {strcat(trial_name, '_root_rec')}, num2cell(errorValTmp)];

        % Calculate error with respect to ground truth without the root
        % added
        errorValTmp_root = abs((meanArray(2,:) - meanArray(1,:)) ./ meanArray(1,:)) * 100;
        errorValTmp_root = [{subjectNames{s}}, {strcat(trial_name, '_root_rec_X')}, num2cell(errorValTmp_root)];

        % Append both error values to errorValues
        errorValues = cat(1, errorValues, errorValTmp, errorValTmp_root);
    end

    errorTable = cat(1, columnNames_error, errorValues);


    %% Save to Excel File

    destPath = [TestPath2, filesep, 'Gait'];

    if ~exist(destPath,'dir')
        mkdir(destPath)
    end

    % Define sheet names

    FullPathName = fullfile(destPath, 'Gait_Parameters_Mean_R_L.xlsx');
    writecell(All_mean, FullPathName, 'Sheet', 'Mean Values');
    writecell(errorTable, FullPathName, 'Sheet', 'Error-Percentage');

    clearvars -except TestPath1 baseDirectory matchingFolders columnNames columnNames_error
end


%% This code creates Bland-Altman plots to compare gait parameters between Vicon and predicted data
% created by Alexander on 26.06.2023

% Description:
% This script processes gait data to create Bland-Altman plots for comparing
% gait parameters obtained from Vicon (ground truth) with predicted data.
% The steps include:
% 1. Adding necessary functions to the path.
% 2. Defining the test path and data group.
% 3. Identifying folders containing the required data.
% 4. Loading the data and extracting relevant gait parameters.
% 5. Creating Bland-Altman plots for each participant and gait parameter.
% 6. Saving the plots to a specified directory.

clc;close all;clear all;

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
results_for_excel =[];

%% Load the Data
walks = {'Norm_Post'};%, 'Norm_Post', 'White', 'Pink'};
gait_params = {'stepLengthL', 'stepLengthR', 'strideLengthL', 'strideLengthR', 'stepWidthL', 'stepWidthR', 'strideWidthL', 'strideWidthR'};

%% Loop trough walks and gait parameters

for folders_name = 1:length(matchingFolders)

    folder_in = dir(matchingFolders{folders_name});
    folder_in(strncmp({folder_in.name}, '.', 1)) = [];

    TestPath2 = [matchingFolders{folders_name},filesep,'Results',filesep];
    mat_files = dir([TestPath2, filesep 'GaitSummary_DL.mat']);
    load(fullfile([mat_files.folder,filesep,mat_files.name]));

    destPath = [TestPath2,filesep,'Figures'];

    % Ensure TestPath2 exists, create if it doesn't
    if ~exist(destPath, 'dir')
        mkdir(destPath);
    end

    subjectNames = fieldnames(GaitSummary);

    for sub_indx = 1:length(subjectNames)
        part = subjectNames{sub_indx};

        for walk = 1:length(walks)
            try

                for param = 1:length(gait_params)

                    % Define Data
                    GT_root = GaitSummary.(part).(strcat(walks{walk},'_root_rec_gt')).GaitParameters.(gait_params{param}); % Ground Truth (Vicon)
                    predict_root = GaitSummary.(part).(strcat(walks{walk},'_root_rec_hat')).GaitParameters.(gait_params{param}); % Prediction with added Vicon Root

                    GT_woroot = GaitSummary.(part).(strcat(walks{walk},'_pose_gt')).GaitParameters.(gait_params{param}); % Ground Truth (Vicon)
                    predict_woroot = GaitSummary.(part).(strcat(walks{walk},'_pose_hat')).GaitParameters.(gait_params{param}); % Prediction without any added Root

                    % Create a figure for Bland-Altmann Plots
                    figure;

                    % Bland-Altman Plot for original root
                    BAP_or = BlandAltmanPlot(GT_root, predict_root);

                    % Set title
                    title([part ': ' walks{walk} ' - ' gait_params{param}], 'Interpreter', 'none');

                    % Set axis labels
                    ylabel('Bias (cm)');                % Y-axis: error in cm
                    xlabel('Mean of GT and Prediction');

                    % Sanitize title for file name
                    sanitized_title_temp = regexprep([part ': ' walks{walk} ' - ' gait_params{param}], '[:*\?<>|]', '_');

                    % Save the figure
                    saveas(gcf, fullfile(destPath, [sanitized_title_temp, '_combined.png']));
                    close(gcf);
                end
            catch ME
                disp(['Pipeline failed  ', [part,'_', walks{walk}]]);
                continue
            end
        end
    end
end

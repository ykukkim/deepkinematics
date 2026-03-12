%% This code creates Bland-Altmann plots to compare gait parameters between from Vicon and predicted Data
% created by Alexander on 26.06.2023

clc;close all;clear all;

addpath(genpath('.\Functions\'));
TestPath1 = 'D:\04_DeepKinematics\Models_trained';

% Data Group
ifgroup = 0;

switch ifgroup
    case 0
        Group = 'All';
    case 1
        Group = 'PartlyGeneral';
end

%% Loop through all the participants and load the data
baseDirectory = [TestPath1,filesep,Group];
matchingFolders = findFoldersEndingWithString(baseDirectory, '-rot_mat');
errorLog = [];

%% Load the Data
walks = {'Norm_Pre'};%, 'Norm_Post', 'White', 'Pink'};
gait_params = {'stepLengthL', 'stepLengthR', 'stepWidthL', 'stepWidthR', 'strideLengthL', 'strideLengthR'};

%% Loop trough walks and gait parameters

for folders_name =  1:length(matchingFolders)

    folder_in = dir(matchingFolders{folders_name});
    folder_in(strncmp({folder_in.name}, '.', 1)) = [];

    TestPath2 = [matchingFolders{folders_name},filesep,'Results',filesep];
    mat_files = dir([TestPath2, filesep '*.mat']);
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
            for param = 1:length(gait_params)

                % Define Data
                GT = GaitSummary.(part).(strcat(walks{walk},'_gt')).GaitParameters.(gait_params{param}); % Ground Truth (Vicon)
                %             predict_or = GaitSummary.(part).(strcat(walks{walk},'_original_root')).GaitParameters.(gait_params{param}); % Prediction with added Vicon Root
                %             predict_r = data.GaitSummary.(strcat(walks{walk},'_root')).GaitParameters.(gait_params{param}); % Prediction with predicted Root
                predict = GaitSummary.(part).(walks{walk}).GaitParameters.(gait_params{param}); % Prediction without any added Root

                % Create Bland-Altmann Plots
                %             figure;
                %             BAP_or = BlandAltmanPlot(GT,predict_or);
                %             title([baseFileName ': ' walks{walk} ' - Original Root - ' gait_params{param}], 'Interpreter', 'none');
                %             figure;
                %             BAP_r = BlandAltmanPlot(GT,predict_r);
                %             title([baseFileName ': ' walks{walk} ' - Root - ' gait_params{param}], 'Interpreter', 'none');
                figure;
                BAP = BlandAltmanPlot(GT,predict);
                title_temp = [part ': ' walks{walk} ' - ' gait_params{param}];
                title(title_temp, 'Interpreter', 'none');

                % Sanitize title_temp for file name
                sanitized_title_temp = regexprep(title_temp, '[:*\?<>|]', '_');

                % Save the plot to the specified path
                saveas(gcf, fullfile(destPath, [sanitized_title_temp, '.png']));
                close(gcf);  % Close the figure to save memory

                % Store results for Excel
                results_for_excel = [results_for_excel; {part, walks{walk}, gait_params{param}, mean(GT), std(GT), mean(predict), std(predict)}];
            end
        end
    end
end


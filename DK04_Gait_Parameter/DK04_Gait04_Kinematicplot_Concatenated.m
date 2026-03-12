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

clc; close all; clear all;

%% Define the path and group to analyze
addpath(genpath('Functions'));
TestPath1 = '/Users/yonkim/DataForWork/DeepKinematics/Models_trained';

% Data Group
ifgroup = 0;

switch ifgroup
    case 0
        Group = 'Full_BIRNN';
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
marker_params = {'Hip','Neck','Head', ...
    'LSHO', 'LARM', 'LFAM', 'LHND',...
    'RSHO','RARM', 'RFAM', 'RHND',...
    'LHJC', 'LKJC','LAJC','LTO3'...
    'RHJC','RKJC', 'RAJC', 'RTO3'};

nAxes = 3; % X, Y, Z
axes_labels = {'X-axis', 'Y-axis', 'Z-axis'};
nPoints = 100;
all_subjects_data = struct();

%% Loop trough walks and gait parameters
for folders_name =  1:length(matchingFolders)

    folder_in = dir(matchingFolders{folders_name});
    folder_in(strncmp({folder_in.name}, '.', 1)) = [];

    TestPath2 = [matchingFolders{folders_name},filesep,'Results',filesep];
    mat_files = dir([TestPath2, filesep 'GaitSummary_DL.mat']);
    load(fullfile([mat_files.folder,filesep,mat_files.name]));

    destPath = [TestPath2,filesep,'Figures',filesep,'Kinematic'];
    if ~exist(destPath, 'dir')
        mkdir(destPath);
    end

    subjectNames = fieldnames(GaitSummary);

    try
        for sub_indx = 1:length(subjectNames)
            part = subjectNames{sub_indx};
            for walk = 1:length(walks)
                % === Loop over sides ===
                for side = ["R", "L"]
                    isLeft = side == "L";
                    if isLeft
                        side_label = 'Left';
                        gaitEvents_GT = GaitSummary.(part).(strcat(walks{walk},'_root_rec_gt')).GaitEvents.HSleftlocs;
                        gaitEvents_PRED = GaitSummary.(part).(strcat(walks{walk},'_root_rec_hat')).GaitEvents.HSleftlocs;
                        marker_params = {'LSHO', 'LARM', 'LFAM', 'LHND','LHJC', 'LKJC','LAJC','LTO3'};
                    else
                        side_label = 'Right';
                        gaitEvents_GT = GaitSummary.(part).(strcat(walks{walk},'_root_rec_gt')).GaitEvents.HSrightlocs;
                        gaitEvents_PRED = GaitSummary.(part).(strcat(walks{walk},'_root_rec_hat')).GaitEvents.HSrightlocs;
                        marker_params = {'RSHO', 'RARM', 'RFAM', 'RHND','RHJC', 'RKJC','RAJC','RTO3'};

                    end

                    for param_idx = 1:length(marker_params)

                        param = marker_params{param_idx};


                        % Define Data -- Without Root added
                        GT_kinematic_woroot = GaitSummary.(part).(strcat(walks{walk},'_pose_gt')).KinematicData.Marker.(param); % Ground Truth (Vicon)
                        predict_kinematic_woroot = GaitSummary.(part).(strcat(walks{walk},'_pose_hat')).KinematicData.Marker.(param); % Prediction without any added Root

                        % Define Data -- Root added
                        GT_kinematic_root = GaitSummary.(part).(strcat(walks{walk},'_root_rec_gt')).KinematicData.Marker.(param); % Prediction with predicted Root
                        predict_kinematic_root = GaitSummary.(part).(strcat(walks{walk},'_root_rec_hat')).KinematicData.Marker.(param); % Prediction without any added Root

                        % --- Ground Truth: Normalise all cycles ---
                        GT_cycles_x = []; GT_cycles_y = []; GT_cycles_z = [];

                        for i = 1:(length(gaitEvents_GT)-1)
                            cycle = double(GT_kinematic_root(gaitEvents_GT(i):gaitEvents_GT(i+1), :));
                            cycle_resampled = resample(cycle, nPoints, size(cycle, 1));
                            GT_cycles_x(:,i) = cycle_resampled(:,1); % average across dimensions if necessary
                            GT_cycles_y(:,i) = cycle_resampled(:,2); % average across dimensions if necessary
                            GT_cycles_z(:,i) = cycle_resampled(:,3); % average across dimensions if necessary
                        end

                        % --- Prediction with root: Normalise all cycles ---
                        predict_cycles_x = []; predict_cycles_y = []; predict_cycles_z = [];
                        for i = 1:(length(gaitEvents_PRED)-1)
                            cycle = double(predict_kinematic_root(gaitEvents_PRED(i):gaitEvents_PRED(i+1), :));
                            cycle_resampled = resample(cycle, nPoints, size(cycle, 1));
                            predict_cycles_x(:,i) = cycle_resampled(:,1); % average across dimensions if necessary
                            predict_cycles_y(:, i) = cycle_resampled(:,2); % average across dimensions if necessary
                            predict_cycles_z(:, i) = cycle_resampled(:,3); % average across dimensions if necessary                        end
                        end

                        %% Store to structured array
                        sideTag = char(side_label);  % 'Left' or 'Right'

                        all_subjects_data.(sideTag).(part).(param).GT.x = GT_cycles_x;
                        all_subjects_data.(sideTag).(part).(param).GT.y = GT_cycles_y;
                        all_subjects_data.(sideTag).(part).(param).GT.z = GT_cycles_z;

                        all_subjects_data.(sideTag).(part).(param).PRED.x = predict_cycles_x;
                        all_subjects_data.(sideTag).(part).(param).PRED.y = predict_cycles_y;
                        all_subjects_data.(sideTag).(part).(param).PRED.z = predict_cycles_z;

                    end
                end

                %% Plotting

                if isLeft
                    side_label = 'Left';
                    marker_params_interested = {'LKJC','LAJC','LTO3'};
                    marker_Names = {'Knee','Ankle','Toe'};
                else
                    side_label = 'Right';
                    marker_params_interested = {'RKJC','RAJC','RTO3'};
                    marker_Names = {'Knee','Ankle','Toe'};
                end

                % Create figure
                fig = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
                tiledlayout(length(marker_params_interested), nAxes, 'TileSpacing', 'compact', 'Padding', 'compact');
                x = linspace(0, 100, nPoints);
                axis_name = {'x','y','z'};
                for param_idx = 1:length(marker_params_interested)

                    param = marker_params_interested{param_idx};

                    for axis = 1:nAxes
                        nexttile;
                        hold on;

                        GT_values = all_subjects_data.(side_label).(part).(param).GT.(axis_name{axis});  % [nPoints x nSubjects]
                        Pred_values = all_subjects_data.(side_label).(part).(param).PRED.(axis_name{axis});  % [nPoints x nSubjects]

                        spm1d.plot.plot_meanSD(GT_values,'color', [0, 0, 0.6]);
                        spm1d.plot.plot_meanSD(Pred_values,'color', [1,0.4,0.3]);

                        if param_idx == 1
                            if axis == 1
                                title('Medio-Lateral');
                            elseif axis == 2
                                title('Anterior-Posterior');
                            else
                                title('Vertical');
                            end
                        end
                        ylabel([marker_Names{param_idx},' (mm)'], 'Interpreter', 'none');

                        if param_idx == length(marker_params)
                            xlabel('Gait Cycle (%)');
                        end

                        grid on;
                        hold off;
                    end
                end

                % Shared legend
                lgd = legend({'GT Mean','GT Std','Pred Mean','Pred Std'}, ...
                    'Location', 'southoutside', 'Orientation', 'horizontal');
                lgd.Layout.Tile = 'south';
                % Save figure
                fig = gcf;
                set(fig, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
                filename = fullfile(destPath, [(part),'-',(side_label),'-',(strcat(walks{walk})),'-Joint Positions.png']);
                exportgraphics(fig, filename, 'Resolution', 300);
                filename = fullfile(destPath, [(part),'-',(side_label),'-',(strcat(walks{walk})),'-Joint Positions.pdf']);
                exportgraphics(fig, filename, 'Resolution', 300);
                close(fig);
            end
        end
        save([TestPath2,filesep,'Allsubjects_pos.mat'],"all_subjects_data",'-mat');
    catch ME
        disp(['Pipeline failed  ', [part,'_', walks{walk}]]);
        continue
    end
end
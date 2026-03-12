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
%
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

nPoints = 100;
all_subjects_data = struct();

%% Loop through all participants and trials
for folders_name = 1:length(matchingFolders)

    folder_in = dir(matchingFolders{folders_name});
    folder_in(strncmp({folder_in.name}, '.', 1)) = [];

    TestPath2 = [matchingFolders{folders_name}, filesep, 'Results', filesep];
    mat_files = dir([TestPath2, 'GaitSummary_DL.mat']);
    load(fullfile(mat_files.folder, mat_files.name));

    destPath = [TestPath2, 'Figures', filesep, 'Joints'];
    if ~exist(destPath, 'dir')
        mkdir(destPath);
    end

    subjectNames = fieldnames(GaitSummary);

    try
        for sub_indx = 1:length(subjectNames)
            part = subjectNames{sub_indx};
            for walk = 1:length(walks)
                %% Plotting
                fig = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
                tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
                x = linspace(0, 100, nPoints);
                % === Loop over sides ===
                for side = ["R", "L"]
                    isLeft = side == "L";
                    %% Joint Labels
                    if isLeft
                        hip = 'LHJC'; knee = 'LKJC'; ankle = 'LAJC';
                        side_label = 'Left';
                        gaitEvents_GT = GaitSummary.(part).(strcat(walks{walk},'_root_rec_gt')).GaitEvents.HSleftlocs;
                        gaitEvents_GT = gaitEvents_GT(gaitEvents_GT< 11250);
                    else
                        hip = 'RHJC'; knee = 'RKJC'; ankle = 'RAJC';
                        side_label = 'Right';
                        gaitEvents_GT = GaitSummary.(part).(strcat(walks{walk},'_root_rec_gt')).GaitEvents.HSrightlocs;
                        gaitEvents_GT = gaitEvents_GT(gaitEvents_GT< 11250);
                    end

                    %% GT
                    GT_Hip = permute(GaitSummary.(part).(strcat(walks{walk},'_joint_rot_gt')).Joint.(hip), [2 3 1]);
                    GT_Knee = permute(GaitSummary.(part).(strcat(walks{walk},'_joint_rot_gt')).Joint.(knee), [2 3 1]);
                    GT_Ankle  = permute(GaitSummary.(part).(strcat(walks{walk},'_joint_rot_gt')).Joint.(ankle),  [2 3 1]);

                    GT_Hip = GT_Hip(:,:,3250:end);
                    GT_Knee = GT_Knee(:,:,3250:end);
                    GT_Ankle = GT_Ankle(:,:,3250:end);


                    T = size(GT_Knee, 3);
                    GT_hip_eul = zeros(T, 3);
                    GT_knee_eul = zeros(T, 3);
                    GT_ankle_eul = zeros(T, 3);

                    for t = 1:T
                        GT_hip_eul(t,:)  = rotm2eul(GT_Hip(:, :, t), 'XYZ');
                        GT_knee_eul(t,:)  = rotm2eul(GT_Knee(:, :, t), 'XYZ');
                        GT_ankle_eul(t,:) = rotm2eul(GT_Ankle(:, :, t), 'XYZ');
                    end

                    GT_hip_flex  = -rad2deg(GT_hip_eul(:,1));
                    GT_knee_flex  = -rad2deg(GT_knee_eul(:,1));
                    GT_ankle_flex =  -rad2deg(GT_ankle_eul(:,1));

                    %% Predictionthigh
                    PRED_Hip = permute(GaitSummary.(part).(strcat(walks{walk},'_joint_rot_hat')).Joint.(hip), [2 3 1]);
                    PRED_Knee = permute(GaitSummary.(part).(strcat(walks{walk},'_joint_rot_hat')).Joint.(knee), [2 3 1]);
                    PRED_Ankle  = permute(GaitSummary.(part).(strcat(walks{walk},'_joint_rot_hat')).Joint.(ankle),  [2 3 1]);
                   
                    PRED_Hip = PRED_Hip(:,:,3250:end);
                    PRED_Knee = PRED_Knee(:,:,3250:end);
                    PRED_Ankle = PRED_Ankle(:,:,3250:end);
                    
                    
                    T = size(PRED_Knee, 3);
                    PRED_hip_eul = zeros(T, 3);
                    PRED_knee_eul = zeros(T, 3);
                    PRED_ankle_eul = zeros(T, 3);

                    for t = 1:T
                        PRED_hip_eul(t,:)  = rotm2eul(PRED_Hip(:, :, t), 'XYZ');
                        PRED_knee_eul(t,:)  = rotm2eul(PRED_Knee(:, :, t), 'XYZ');
                        PRED_ankle_eul(t,:) = rotm2eul(PRED_Ankle(:, :, t), 'XYZ');
                    end

                    PRED_hip_flex  = -rad2deg(PRED_hip_eul(:,1));
                    PRED_knee_flex  = -rad2deg(PRED_knee_eul(:,1));
                    PRED_ankle_flex =  -rad2deg(PRED_ankle_eul(:,1));

                    %% Segment and resample gait cycles
                    GT_cycles_Hip = []; GT_cycles_Knee = []; GT_cycles_Ankle = [];
                    PRED_cycles_Hip = []; PRED_cycles_Knee = []; PRED_cycles_Ankle = [];

                    for i = 1:(length(gaitEvents_GT)-1)
                        seg = gaitEvents_GT(i):gaitEvents_GT(i+1);
                        if length(seg) > 10
                            GT_cycles_Hip(:, i)  = resample(GT_hip_flex(seg), nPoints, length(seg));
                            GT_cycles_Knee(:, i)  = resample(GT_knee_flex(seg), nPoints, length(seg));
                            GT_cycles_Ankle(:, i) = resample(GT_ankle_flex(seg), nPoints, length(seg));
                            PRED_cycles_Hip(:, i)  = resample(PRED_hip_flex(seg), nPoints, length(seg));
                            PRED_cycles_Knee(:,i)  = resample(PRED_knee_flex(seg), nPoints, length(seg));
                            PRED_cycles_Ankle(:,i) = resample(PRED_ankle_flex(seg), nPoints, length(seg));
                        end
                    end


                    %% Store in structured array
                    sideTag = char(side_label);  % 'Left' or 'Right'
                    all_subjects_data.(sideTag).(part).Hip.GT  = GT_cycles_Hip;
                    all_subjects_data.(sideTag).(part).Hip.PRED = PRED_cycles_Hip;

                    all_subjects_data.(sideTag).(part).Knee.GT   = GT_cycles_Knee;
                    all_subjects_data.(sideTag).(part).Knee.PRED = PRED_cycles_Knee;

                    all_subjects_data.(sideTag).(part).Ankle.GT  = GT_cycles_Ankle;
                    all_subjects_data.(sideTag).(part).Ankle.PRED = PRED_cycles_Ankle;

                    joints = fieldnames(all_subjects_data.(side_label).(part));

                    for i = 1 : length(joints)
                        nexttile;
                        hold on;

                        GT_values = all_subjects_data.(side_label).(part).(joints{i}).GT;
                        Pred_values = all_subjects_data.(side_label).(part).(joints{i}).PRED;

                        spm1d.plot.plot_meanSD(GT_values, 'color', [0, 0, 0.6]);
                        spm1d.plot.plot_meanSD(Pred_values, 'color', [1, 0.4, 0.3]);

                        title(joints{i}, 'Interpreter', 'none');  % dynamic title based on axis label
                        ylabel('Flexion/Extension (deg)', 'Interpreter', 'none');  % joint name per row
                        xlabel('Gait Cycle (%)');

                        grid on;
                        hold off;
                    end
                end

                % Add side labels (Left and Right) using annotation
                annotation('textbox', [0.006 0.95 0.1 0.05], 'String', 'Right Side', ...
                    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
                annotation('textbox', [0.006 0.46 0.1 0.05], 'String', 'Left Side', ...
                    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');

                % Shared legend
                lgd = legend({'GT Mean','GT Std','Pred Mean','Pred Std'}, ...
                    'Location', 'southoutside', 'Orientation', 'horizontal');
                lgd.Layout.Tile = 'south';

                % Save figure
                fig = gcf;
                set(fig, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
                filename = fullfile(destPath, [(part),'-',(side_label),'-',(strcat(walks{walk})),'-Flexion Angles.png']);
                exportgraphics(fig, filename, 'Resolution', 300);
                filename = fullfile(destPath, [(part),'-',(side_label),'-',(strcat(walks{walk})),'-Flexion Angles.pdf']);
                exportgraphics(fig, filename, 'Resolution', 300);
                close(fig);
            end
        end
        save([TestPath2,filesep,'Allsubjects_rot_corrected.mat'],"all_subjects_data",'-mat');
    catch ME
        disp(['Pipeline failed for ', part, ' - ', walks{walk}]);
        continue
    end
end

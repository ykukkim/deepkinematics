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
        Group = 'Manuscript';
    case 1
        Group = 'PartlyGeneral';
end

%% Loop through all the participants and load the data
baseDirectory = [TestPath1,filesep,Group];
matchingFolders = findFoldersEndingWithString(baseDirectory, '-acc_gyro');
temp_folder = matchingFolders;

%% Load the Data
model_names = {'ATT','BIRNN','DIFF'};

data_ATT = load(fullfile([temp_folder{1}, filesep, 'Results', filesep],'Allsubjects_rot.mat'));
data_BIRNN = load(fullfile([temp_folder{2}, filesep, 'Results', filesep],'Allsubjects_rot.mat'));
data_DIFF = load(fullfile([temp_folder{3}, filesep, 'Results', filesep],'Allsubjects_rot.mat'));

nPoints = 100;

fig = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
x = linspace(0, 100, nPoints);

for side = ["R", "L"]

    isLeft = side == "L";
    if isLeft
        side_label = 'Left';
    else
        side_label = 'Right';
    end

    joints = fieldnames(data_ATT.all_subjects_data.(side_label));

    for i = 1 : length(joints)
        nexttile;
        hold on;

        % === Load Data ===
        GT_values         = data_ATT.all_subjects_data.(side_label).(joints{i}).GT.MeanFlex;
        Pred_values_att   = data_ATT.all_subjects_data.(side_label).(joints{i}).PRED.MeanFlex;
        Pred_values_birnn = data_BIRNN.all_subjects_data.(side_label).(joints{i}).PRED.MeanFlex;
        Pred_values_diff  = data_DIFF.all_subjects_data.(side_label).(joints{i}).PRED.MeanFlex;

        % === Plot Mean ± SD Curves ===
        spm1d.plot.plot_meanSD(GT_values,         'color', [0, 0, 0],        'linewidth', 3);  % Ground Truth (black)
        hold on;
        spm1d.plot.plot_meanSD(Pred_values_att,   'color', [0.85, 0.33, 0.10], 'linewidth', 3);  % ATT (orange)
        spm1d.plot.plot_meanSD(Pred_values_birnn, 'color', [0.47, 0.67, 0.19], 'linewidth', 3);  % BIRNN (green)
        spm1d.plot.plot_meanSD(Pred_values_diff,  'color', [0.49, 0.18, 0.56], 'linewidth', 3);  % Diffusion (purple)

        % === Compute Error Metrics for ATT ===
        abs_error_att  = abs(GT_values - Pred_values_att);           % [100 x 8]
        sq_error_att   = (GT_values - Pred_values_att).^2;           % [100 x 8]

        mae_per_subject_att  = mean(abs_error_att, 1);               % [1 x 8]
        rmse_per_subject_att = sqrt(mean(sq_error_att, 1));          % [1 x 8]

        mean_mae_att   = mean(mae_per_subject_att);                  % scalar
        std_mae_att    = std(mae_per_subject_att);                   % scalar

        mean_rmse_att  = mean(rmse_per_subject_att);                 % scalar
        std_rmse_att   = std(rmse_per_subject_att);                  % scalar

        % === Compute Error Metrics for birnn ===
        abs_error_birnn  = abs(GT_values - Pred_values_birnn);           % [100 x 8]
        sq_error_birnn   = (GT_values - Pred_values_att).^2;           % [100 x 8]

        mae_per_subject_birnn = mean(abs_error_birnn, 1);               % [1 x 8]
        rmse_per_subject_birnn= sqrt(mean(sq_error_birnn, 1));          % [1 x 8]

        mean_mae_birnn   = mean(mae_per_subject_birnn);                  % scalar
        std_mae_birnn    = std(mae_per_subject_birnn);                   % scalar

        mean_rmse_birnn  = mean(rmse_per_subject_birnn);                 % scalar
        std_rmse_birnn  = std(rmse_per_subject_birnn);                  % scalar

        % === Compute Error Metrics for diff ===
        abs_error_diff  = abs(GT_values - Pred_values_diff);           % [100 x 8]
        sq_error_diff   = (GT_values - Pred_values_att).^2;           % [100 x 8]

        mae_per_subject_diff = mean(abs_error_diff, 1);               % [1 x 8]
        rmse_per_subject_diff= sqrt(mean(sq_error_diff, 1));          % [1 x 8]

        mean_mae_diff   = mean(mae_per_subject_diff);                  % scalar
        std_mae_diff    = std(mae_per_subject_diff);                   % scalar

        mean_rmse_diff  = mean(rmse_per_subject_diff);                 % scalar
        std_rmse_diff  = std(rmse_per_subject_diff);                  % scalar

        title(joints{i}, 'Interpreter', 'none');  % dynamic title based on axis label
        ylabel('Flexion/Extension (deg)', 'Interpreter', 'none');  % joint name per row
        xlabel('Gait Cycle (%)');

        % === Annotate Error Summary ===
        x_lim = xlim;
        y_lim = ylim;
        x_pos = x_lim(2) - 40;
        y_pos = y_lim(2) - 5;

        error_summary_att = {
            sprintf('\\bfATT: \\rmMAE:  %.2f ± %.2f deg',  mean_mae_att,  std_mae_att)
            sprintf('\\bfBIRNN \\rmMAE:  %.2f ± %.2f deg',  mean_mae_birnn,  std_mae_birnn)
            sprintf('\\bfDIFF \\rmMAE:  %.2f ± %.2f deg',  mean_mae_diff,  std_mae_diff)
            };

        %                 x_lim = xlim;
        %         y_lim = ylim;
        %         x_pos = x_lim(2) - 70;
        %         y_pos = y_lim(2) - 5;
        %
        %         error_summary_att = {
        %             sprintf('\\bfATT:   \\rmMAE = %.2f ± %.2f deg, RMSE = %.2f ± %.2f deg',  mean_mae_att,  std_mae_att,  mean_rmse_att,  std_rmse_att)
        %             sprintf('\\bfBIRNN: \\rmMAE = %.2f ± %.2f deg, RMSE = %.2f ± %.2f deg',  mean_mae_birnn, std_mae_birnn, mean_rmse_birnn, std_rmse_birnn)
        %             sprintf('\\bfDIFF:  \\rmMAE = %.2f ± %.2f deg, RMSE = %.2f ± %.2f deg',  mean_mae_diff,  std_mae_diff,  mean_rmse_diff,  std_rmse_diff)
        %         };


        text(x_pos, y_pos, error_summary_att, ...
            'VerticalAlignment', 'top', ...
            'FontSize', 9, ...
            'FontName', 'Helvetica', ...
            'Interpreter', 'tex');


        grid on;
        hold off;
    end
end

% Add side labels (Left and Right) using annotation
annotation('textbox', [0.006, 0.95,  0.1 0.05], 'String', 'Right Side', ...
    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');
annotation('textbox', [0.006 0.46 0.1 0.05], 'String', 'Left Side', ...
    'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'left');

% Shared legend
lgd = legend({'GT Mean','GT Std',...
    'ATT Mean','ATT Std',...
    'BIRNN Mean', 'BIRNN Std', ...
    'DIFF Mean', 'DIFF Std'}, ...
    'Location', 'southwest', 'Orientation', 'horizontal');
lgd.Layout.Tile = 'south';

% Save figure
fig = gcf;
set(fig, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
destPath = [baseDirectory,filesep,'Results',filesep,'Figures', filesep, 'Kinematic'];
if ~exist(destPath, 'dir')
    mkdir(destPath);
end
filename = fullfile(destPath, 'Final_fig.pdf');
exportgraphics(fig, filename, 'Resolution', 300);
close(fig);

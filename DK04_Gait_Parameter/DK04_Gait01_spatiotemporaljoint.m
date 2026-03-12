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

%% Load the Data
for folder_idx =  1:length(matchingFolders)

    folder_in = dir(matchingFolders{folder_idx});
    folder_in(strncmp({folder_in.name}, '.', 1)) = [];
    subjectNames = folder_in(strncmp({folder_in.name}, 'SonE', 4));
    destPath = [matchingFolders{folder_idx},filesep,'Results',filesep];

    for sub_indx = 1:length(subjectNames)
        part = subjectNames(sub_indx).name;

        TestPath2 = [matchingFolders{folder_idx},filesep,subjectNames(sub_indx).name,filesep];
        mat_files = dir(fullfile(TestPath2,'*.mat'));

        nooffiles = length(mat_files);

        if ~exist(destPath,'dir')
            mkdir(destPath)
        end

        %% Load all files in directory
        for trial = 1:nooffiles %7
            try
                filename = mat_files(trial).name(1:end-4);
                Name = part;

%                 if contains(filename, ["Norm_Pre", "Norm_Post", "White", "Pink"]) && (contains(filename, "pose") || contains(filename, "root"))
                if contains(filename, "Norm_Post") && (contains(filename, "pose") || contains(filename, "root"))

                    data_1 = load([mat_files(trial).folder,filesep,mat_files(trial).name]);

                    VD = data_1.VD;
                    VD.SF = double(VD.SF);
                    Test = filename;

                    if isfield(data_1, 'VD')
                        [interpolatedVD,ctrialsfewEvents,ctrialsfewEventsnames] = FillGaps(VD, filename, Name);
                        [GaitEvents,error] = StepDetection_DL(interpolatedVD,VD, Name, Test);
                        [GaitEvents, GaitParameters] = GaitWork_Treadmill(interpolatedVD, GaitEvents, VD,filename);

                        GaitSummary.(Name).(filename).GaitEvents = GaitEvents;
                        GaitSummary.(Name).(filename).KinematicData = VD;
                        GaitSummary.(Name).(filename).GaitParameters = GaitParameters;
                    end
                elseif contains(filename, ["Norm_Pre", "Norm_Post", "White", "Pink"]) && (contains(filename, "joint_rot"))
                        data_1 = load([mat_files(trial).folder,filesep,mat_files(trial).name]);
                        GaitSummary.(Name).(filename).Joint = data_1.VD.JointsRot;
                end

            catch ME
                errorLog = LogsSONEMS(errorLog,part,ME);
                disp(['Pipeline failed  ', [part,'_', filename]]);
                continue
            end
        end
    end
    %% Save Data
    disp('--Saving results:');
    disp('Directory:');
    datasave= fullfile(destPath,'GaitSummary_DL');
    disp(datasave);
    save(datasave,"GaitSummary",'-v7.3');
    disp('Saving done');
    disp('Pipeline done');

    clearvars -except baseDirectory matchingFolders Group errorLog destPath
end
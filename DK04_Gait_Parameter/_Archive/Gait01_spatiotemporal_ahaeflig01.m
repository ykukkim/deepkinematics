%% Gait Analysis
% Extracts all the relevant spatio-temporal gait parameters
% 1. It fills the gap by interpolating the missing datasets, but this
% process should be performed in VICON, such as gap filling.
% 1. It detects the IC and TO using FVA from O'connor et al 2007
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

clear all;
close all;
clc;


%% Define the path and group to analyze

% TestPath1 = 'P:\Projects\NCM\NCM_SonEMS\02_Data';
TestPath1 =  'C:\Users\alexa\Documents\ETH\Master\Masterthesis\MatLab\'; 
pathCompSep = '\';

ifgroup = input('Please choose group you want to analyze? \n Enter 0 for YOUNG or 1 for ELDERLY \n');
switch ifgroup
    case 0
        Group = 'Healthy_young';
    case 1
        Group = 'Healthy_older';
end

TestPath2 = [TestPath1,pathCompSep,Group];

%Add functions
addpath(genpath('\\hest.nas.ethz.ch\green_groups_lmb_public\Projects\NCM\NCM_SonEMS\project_only\01_Codes\Vicon\Gait\Functions'))

%% Loop through all the participants and load the data
% Get folder names of participant
listL1 = dir(TestPath2);
listL1(strncmp({listL1.name}, '.', 1)) = [];
dirFlagsL1 = [listL1.isdir];
listL1 = listL1(dirFlagsL1);


addpath(genpath(TestPath2));
currentfolder = pwd;
addpath(genpath(currentfolder));
subjectNames = listL1;
subjectNames(strncmp({subjectNames.name}, '.', 1)) = [];


%% Load the Data
for subjectloop = 3:4 %[1, 3:6, 8, 9, 11:13] %1 %1:length(listL1) [1, 3:6, 8, 9, 11:

    procVic = ['Processed', pathCompSep, 'Vicon', pathCompSep];
    TestPath3 = [TestPath2,pathCompSep,listL1(subjectloop).name,pathCompSep,procVic,'matfiles'];

    listL3 = dir(TestPath3);
    listL3(strncmp({listL3.name}, '.', 1)) = [];
    dirFlagsL3 = [listL3.isdir];

    TestPath4 = [TestPath3,pathCompSep,listL3.name];
    dirFiles = dir(fullfile(TestPath3,'*.mat'));

    destPath = [TestPath2,pathCompSep,listL1(subjectloop).name,pathCompSep,procVic,'GaitSummary'];

    part = subjectNames(subjectloop).name;
    addpath(genpath(TestPath3))

    if ~exist(destPath,'dir')
        mkdir(destPath)
    end

    trialnames = dir([TestPath3,pathCompSep, '*.mat']); % Specify filename based on the format /*participant*session*trial*.mat
    trialnames(strncmp({trialnames.name}, '.', 1)) = [];
    nooffiles = length(trialnames);

    %% Load all files in directory
    for trial = 1:nooffiles %7
        filename = trialnames(trial).name(1:end-4);
        Name = part;
        try
            if filename == "Norm_Pre" || filename == "Norm_Post"
                   % || filename == "Pink" || filename =="White"
                %                     || filename == "Decline" || filename == "Incline"
                data_1 = load(filename);
                VD = data_1.VD;
                Test = filename;
                %         Test = filename(9:end);
                %         No = filename(end);

                if isfield(data_1, 'VD')
                    [interpolatedVD,ctrialsfewEvents,ctrialsfewEventsnames] = FillGaps(VD, filename, Name);
                    [GaitEvents,error] = StepDetection(interpolatedVD,VD, Name,Test);
                    [GaitEvents, GaitParameters, GaitCycles,GaitPathDescription] = GaitWork_Treadmill(interpolatedVD, GaitEvents, VD,filename);

                    %% Re-work on the structure. Should not use eval.....
                    comd = [Name, '.', Test, '.GaitPathDescription = GaitPathDescription;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.GaitCycles = GaitCycles;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.GaitEvents = GaitEvents;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.GaitParameters = GaitParameters;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.KinematicData = VD;'];
                    eval(comd);

                    comd = [Name,'.',Test, '= orderfields(', Name, '.', Test,');'];
                    eval(comd);

                elseif ctrialsfewEvents>0
                    disp({ctrialsfewEventsnames});
                    % continue;
                else
                    fprintf('%s has not been correctly converted to the MAT format',filename);
                    % continue;
                end

            elseif filename == "White_1" || filename == "White_2" || filename == "White_3"...
                    || filename == "White_4" || filename =="White_5" || filename == "White_6"...
                    || filename == "Pink_1" || filename == "Pink_2"
                data_1 = load(filename);
                VD = data_1.VD;
                Test = filename;
                %         Test = filename(9:end);
                %         No = filename(end);

                if isfield(data_1, 'VD')

                    [interpolatedVD,ctrialsfewEvents,ctrialsfewEventsnames] = FillGaps(VD, filename, Name);
                    [GaitEvents,error] = StepDetection_noise(interpolatedVD,VD, Name,Test);
                    [GaitEvents, GaitParameters, GaitCycles,GaitPathDescription] = GaitWork_Treadmill(interpolatedVD, GaitEvents, VD,filename);

                    %% Re-work on the structure. Should not use eval.....
                    comd = [Name, '.', Test, '.GaitPathDescription = GaitPathDescription;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.GaitCycles = GaitCycles;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.GaitEvents = GaitEvents;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.GaitParameters = GaitParameters;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.KinematicData = VD;'];
                    eval(comd);

                    comd = [Name,'.',Test, '= orderfields(', Name, '.', Test,');'];
                    eval(comd);

                elseif ctrialsfewEvents>0
                    disp({ctrialsfewEventsnames});
                    % continue;
                else
                    fprintf('%s has not been correctly converted to the MAT format',filename);
                    % continue;
                end

            elseif filename == "Incline" || filename == "Decline"
                data_1 = load(filename);
                VD = data_1.VD;
                Test = filename;
                %         Test = filename(9:end);
                %         No = filename(end);

                if isfield(data_1, 'VD')
                    [interpolatedVD,ctrialsfewEvents,ctrialsfewEventsnames] = FillGaps(VD, filename, Name);
                    [GaitEvents,error] = StepDetection_angle(interpolatedVD,VD, Name,Test);
                    [GaitEvents, GaitParameters, GaitCycles,GaitPathDescription] = GaitWork_Treadmill(interpolatedVD, GaitEvents, VD,filename);

                    %% Re-work on the structure. Should not use eval.....
                    comd = [Name, '.', Test, '.GaitPathDescription = GaitPathDescription;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.GaitCycles = GaitCycles;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.GaitEvents = GaitEvents;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.GaitParameters = GaitParameters;'];
                    eval(comd);

                    comd = [Name,'.',Test,'.KinematicData = VD;'];
                    eval(comd);

                    comd = [Name,'.',Test, '= orderfields(', Name, '.', Test,');'];
                    eval(comd);

                elseif ctrialsfewEvents>0
                    disp({ctrialsfewEventsnames});
                    % continue;
                else
                    fprintf('%s has not been correctly converted to the MAT format',filename);
                    % continue;
                end

            else
                fprintf("%s Test skipped\n",filename)
                % continue;
            end
        catch
            fprintf('error %d \n',trial);
        end
    end

    datasave= fullfile(destPath,Name);
    save(datasave,Name,'-v7.3');
    clearvars -except TestPath1 TestPath2 pathCompSep currentfolder foldername ProcessedData_idx Group ifgroup listL1 subjectNames Name subjectloop dirFlagsL1
end

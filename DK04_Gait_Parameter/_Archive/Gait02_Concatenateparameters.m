%% Concatenates relevant spatio-temporal gait parameters
% This scripts concatenates relevant gait parameters into one.
% Then box plots
clear; close; clc;

%% Setting Directory for data
ifMac = input('Please choose computer is being used for analysis? \n Enter 0 for Mac or 1 for Non Mac or Junk\n');

switch ifMac
    
    case 0
        %         TestPath1   = '/Volumes/Macintosh HD - Data/HRX/ProcessedData/';
        TestPath1 = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_STM/NCM_HRX_Walking/Data/Processed';
        pathCompSep = '/';
        
    case 1
        TestPath1    = 'D:\HRX\'; % change to the relevant Windows path.
        pathCompSep  = '\';
end

%% Setting Directory for data
addpath(genpath(TestPath1));
currentfolder = pwd;
addpath(genpath(currentfolder));
foldername = dir(TestPath1);
foldername(strncmp({foldername.name}, '.', 1)) = [];

ProcessedData_idx = find(strcmp({foldername.name},'Processed'));
subjectsNames = dir([foldername(ProcessedData_idx).folder,pathCompSep,foldername(ProcessedData_idx).name]);
subjectsNames(strncmp({subjectsNames.name}, '.', 1)) = [];

destPath = [TestPath1,'Results',pathCompSep,'Gait'];

if ~exist(destPath, 'dir')
    mkdir(destPath)
end

%% Load the Data
for subjectloop = 1:length(subjectsNames)
    
    temp_dir = dir([subjectsNames(subjectloop).folder,pathCompSep,subjectsNames(subjectloop).name,pathCompSep]);
    temp_dir(strncmp({temp_dir.name}, '.', 1)) = [];
    
    Vicon_idx = find(strcmp({temp_dir.name},'Vicon'));
    temp_dir_vicon = dir([temp_dir(Vicon_idx).folder,pathCompSep,temp_dir(Vicon_idx).name]);
    temp_dir_vicon(strncmp({temp_dir_vicon.name}, '.', 1)) = [];
    
    DIR_HRXdata = [temp_dir_vicon(1).folder,pathCompSep,'GaitSummary'];
    addpath(genpath(DIR_HRXdata))
    trialnames = dir([DIR_HRXdata,pathCompSep, '*.mat']); % Specify filename based on the format /*participant*session*trial*.mat
    trialnames(strncmp({trialnames.name}, '.', 1)) = [];
    
    Sub=load([DIR_HRXdata,pathCompSep,trialnames.name]);
    filename = trialnames(1).name(1:end-4);  % remove the file extensions
    Name = filename(1:7);
    condition = fieldnames(Sub.(Name));
    for trial = 1:length(fieldnames(Sub.(Name)))
        
        stridelength.(Name).(condition{trial}).left.length = Sub.(Name).(condition{trial}).GaitParameters.strideLengthL;
        stridelength.(Name).(condition{trial}).left.mean   = mean(Sub.(Name).(condition{trial}).GaitParameters.strideLengthL);
        stridelength.(Name).(condition{trial}).left.std    = std(Sub.(Name).(condition{trial}).GaitParameters.strideLengthL);
        stridelength.(Name).(condition{trial}).left.median = median(Sub.(Name).(condition{trial}).GaitParameters.strideLengthL);
        
        stridelength.(Name).(condition{trial}).right.length = Sub.(Name).(condition{trial}).GaitParameters.strideLengthR;
        stridelength.(Name).(condition{trial}).right.mean   = mean(Sub.(Name).(condition{trial}).GaitParameters.strideLengthR);
        stridelength.(Name).(condition{trial}).right.std    = std(Sub.(Name).(condition{trial}).GaitParameters.strideLengthR);
        stridelength.(Name).(condition{trial}).right.median = median(Sub.(Name).(condition{trial}).GaitParameters.strideLengthR);
        
        steplength.(Name).(condition{trial}).left.length = Sub.(Name).(condition{trial}).GaitParameters.stepLengthL;
        steplength.(Name).(condition{trial}).left.mean   = mean(Sub.(Name).(condition{trial}).GaitParameters.stepLengthL);
        steplength.(Name).(condition{trial}).left.std    = std(Sub.(Name).(condition{trial}).GaitParameters.stepLengthL);
        steplength.(Name).(condition{trial}).left.median = median(Sub.(Name).(condition{trial}).GaitParameters.stepLengthL);
        
        steplength.(Name).(condition{trial}).right.length = Sub.(Name).(condition{trial}).GaitParameters.stepLengthR;
        steplength.(Name).(condition{trial}).right.mean   = mean(Sub.(Name).(condition{trial}).GaitParameters.stepLengthR);
        steplength.(Name).(condition{trial}).right.std    = std(Sub.(Name).(condition{trial}).GaitParameters.stepLengthR);
        steplength.(Name).(condition{trial}).right.median = median(Sub.(Name).(condition{trial}).GaitParameters.stepLengthR);
        
        stepwidth.(Name).(condition{trial}).left.length = Sub.(Name).(condition{trial}).GaitParameters.stepWidthL;
        stepwidth.(Name).(condition{trial}).left.mean   = mean(Sub.(Name).(condition{trial}).GaitParameters.stepWidthL);
        stepwidth.(Name).(condition{trial}).left.std    = std(Sub.(Name).(condition{trial}).GaitParameters.stepWidthL);
        stepwidth.(Name).(condition{trial}).left.median = median(Sub.(Name).(condition{trial}).GaitParameters.stepWidthL);
        
        stepwidth.(Name).(condition{trial}).right.length = Sub.(Name).(condition{trial}).GaitParameters.stepWidthR;
        stepwidth.(Name).(condition{trial}).right.mean   = mean(Sub.(Name).(condition{trial}).GaitParameters.stepWidthR);
        stepwidth.(Name).(condition{trial}).right.std    = std(Sub.(Name).(condition{trial}).GaitParameters.stepWidthR);
        stepwidth.(Name).(condition{trial}).right.median = median(Sub.(Name).(condition{trial}).GaitParameters.stepWidthR);
        
        cadence.(Name).(condition{trial}).right.length = Sub.(Name).(condition{trial}).GaitParameters.Cadence;
        cadence.(Name).(condition{trial}).right.mean   = mean(Sub.(Name).(condition{trial}).GaitParameters.Cadence);
        cadence.(Name).(condition{trial}).right.std    = std(Sub.(Name).(condition{trial}).GaitParameters.Cadence);
        cadence.(Name).(condition{trial}).right.median = median(Sub.(Name).(condition{trial}).GaitParameters.Cadence);
    end
end


%% check results
fnames = fieldnames(stridelength);

for i = 1:length(fnames)
    condition = fieldnames(stridelength.(fnames{i}));
    for j = 1:length(condition)
        results.stridelength.mean.(string(condition{j}))(i,1) = stridelength.(fnames{i}).(condition{j}).left.mean;
        results.stridelength.mean.(string(condition{j}))(i,2) = stridelength.(fnames{i}).(condition{j}).right.mean;
        results.stridelength.std.(string(condition{j}))(i,1) = stridelength.(fnames{i}).(condition{j}).left.std;
        results.stridelength.std.(string(condition{j}))(i,2) = stridelength.(fnames{i}).(condition{j}).right.std;
        results.steplength.mean.(string(condition{j}))(i,1) = steplength.(fnames{i}).(condition{j}).left.mean;
        results.steplength.mean.(string(condition{j}))(i,2) = steplength.(fnames{i}).(condition{j}).right.mean;
        results.steplength.std.(string(condition{j}))(i,1) = steplength.(fnames{i}).(condition{j}).left.std;
        results.steplength.std.(string(condition{j}))(i,2) = steplength.(fnames{i}).(condition{j}).right.std;
        results.stepwidth.mean.(string(condition{j}))(i,1) = stepwidth.(fnames{i}).(condition{j}).left.mean;
        results.stepwidth.mean.(string(condition{j}))(i,2) = stepwidth.(fnames{i}).(condition{j}).right.mean;
        results.stepwidth.std.(string(condition{j}))(i,1)  = stepwidth.(fnames{i}).(condition{j}).left.std;
        results.stepwidth.std.(string(condition{j}))(i,2)  = stepwidth.(fnames{i}).(condition{j}).right.std;
        results.cadence.mean.(string(condition{j}))(i,1) = cadence.(fnames{i}).(condition{j}).right.mean;
        
    end
end

datasave= fullfile(destPath,'results');
save(datasave,'results','-v7.3');

%% Box Plot
stridelength_boxplot = [results.stridelength.mean.har40_0(:,1) results.stridelength.mean.har20_0(:,1) results.stridelength.mean.normg_0(:,1) results.stridelength.mean.wei20_0(:,1) results.stridelength.mean.wei40_0(:,1)];
stepwidth_boxplot    = [results.stepwidth.mean.har40_0(:,1) results.stepwidth.mean.har20_0(:,1) results.stepwidth.mean.normg_0(:,1) results.stepwidth.mean.wei20_0(:,1) results.stepwidth.mean.wei40_0(:,1)];
steplength_boxplot   = [results.steplength.mean.har40_0(:,1) results.steplength.mean.har20_0(:,1) results.steplength.mean.normg_0(:,1) results.steplength.mean.wei20_0(:,1) results.steplength.mean.wei40_0(:,1)];
cadence_boxplot      = [results.cadence.mean.har40_0(:,1) results.cadence.mean.har20_0(:,1) results.cadence.mean.normg_0(:,1) results.cadence.mean.wei20_0(:,1) results.cadence.mean.wei40_0(:,1)];

figure; boxplot(stridelength_boxplot,'Notch','off','Labels',{'Har40','Har20','Normg','Wei20','Wei40'});
xlabel('Weight Conditions'); ylabel('Stride length (cm)')
title('Stride Length');

figure; boxplot(stepwidth_boxplot,'Notch','off','Labels',{'Har40','Har20','Normg','Wei20','Wei40'});
xlabel('Weight Conditions'); ylabel('Step Width (cm)')
title('Stepwidth');

figure; boxplot(steplength_boxplot,'Notch','off','Labels',{'Har40','Har20','Normg','Wei20','Wei40'});
xlabel('Weight Conditions');ylabel('Step length (cm)')
title('Step Length');

figure; boxplot(cadence_boxplot,'Notch','off','Labels',{'Har40','Har20','Normg','Wei20','Wei40'});
xlabel('Weight Conditions');ylabel('Cadence (n/min)')
title('Cadence');

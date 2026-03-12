%%Open MAT files from Vicon and get gait variables.

%Matfiles need to be in the new structure
%all matfiles in the DIR will be loaded and ordered
%Capitals are important in naming the files

clc; 
clear all;
close all;

%add path to directory

ifMac = input('Please choose computer is being used for analysis? \n Enter 0 for Mac or 1 for Non Mac or Junk\n');

switch ifMac
    case 0 
        DIR_main = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData';
        DIR_gt = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData/Data';
        DIR_opt = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData/Output_FAT';
        DIR_anthro = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData/anthroDATA';
        DIR_jAngles = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/Angles_Navrag/Bewegungsanalyse_v2.00';
        pathCompSep = '/';
    case 1
        DIR_main = '\\Projects\NCM\NCM_EXP\NCM_FAT\for Navrag\GaiT\analyseData';
        DIR_gt='\\Projects\NCM\NCM_EXP\NCM_FAT\for Navrag\GaiT\analyseData\Data';
        DIR_opt='\\Projects\NCM\NCM_EXP\NCM_FAT\for Navrag\GaiT\analyseData\Output_FAT';
        DIR_anthro = '\\Projects\NCM\NCM_EXP\NCM_FAT\for Navrag\Gait\analyseData\anthroDATA';
        DIR_jAngles = '\\Projects\NCM\NCM_EXP\NCM_FAT\Angles_Navrag\Bewegungsanalyse_v2.00';
        pathCompSep = '\';
end

addpath(genpath(DIR_gt));
addpath(genpath(DIR_opt));
addpath(genpath(DIR_anthro));
addpath(genpath(DIR_jAngles));
currentfolder = pwd;
addpath(genpath(currentfolder));


%% run program

GaitSummary_FAT = struct();
vis=false; %show pictures
name_study = 'FAT';

partFolders = dir(DIR_gt);
partFolders = partFolders([partFolders.isdir]); % removes the .DS_Store value
partFolders(strncmp({partFolders.name}, '.', 1)) = []; % removes the . and .. entries
partFoldernames = {partFolders.name};
for parti = 1:length(partFoldernames)
    
    if parti == 4 || parti == 5
        continue;
    else
name_participant=char(partFoldernames(parti)); %name of participant to show figures


DIR_subject = [DIR_gt, pathCompSep, name_participant];

%% make new structure
% do this if I add a new participant --> participants data are included in
% the new GaitVariability.mat

% if new
%     %delete GaitVariability if this file exists 
%     if exist('GaitVariability.mat','file');
%         delete('GaitVariability.mat')
%     end
%     
%     if exist('normVD.mat','file');
%         delete('normVD.mat')
%     end

	%make GaitVariability file and normalizes videodata
    [Gait_Summary]=Open_Gait_allfiles(DIR_subject, DIR_anthro, name_study, pathCompSep, parti, DIR_main);
    
    %map all gait data with a substructure Total
    Gait_Summary=ListData(Gait_Summary);
    
    % Just added left and right sided cycles. Need to be rechecked for
    % variabilility calculations. Also check the Rotated components
    % something is funny there.
            
% end

%% graph spatial data
%1=Pre-fatigue 2=Post-fatigue
%Fill in participant name to show data
%'Run' vs. 'Squat'

% graph_data(Gait_Summary, name_participant, 'Run', 1, vis );
% graph_data(Gait_Summary, name_participant, 'Run', 2, vis );
% 
% graph_data(Gait_Summary, name_participant, 'Squat', 1, vis );
% graph_data(Gait_Summary, name_participant, 'Squat', 2, vis );


%% make CV struct
% returns a struct CV with all variability data for the Gait parameters

Gait_Summary = gaitVariability(Gait_Summary);

%Gait_Summary = gaitCycleRotations(Gait_Summary);
Gait_Summary = gaitCycleVariability(Gait_Summary);
%% graph variable data
%vis=true;
%CV_Graph(Gait_Summary,name_participant,vis);

%% save all data in GaitVariability
% display(GaitVariability);
% save GaitVariability GaitVariability
% save VideoData VideoData

 comd = ['GaitSummary_FAT.', name_participant '= Gait_Summary.', name_participant, ';'];
 eval(comd);
    end

end
 outputfilename = [DIR_opt, pathCompSep, 'FATStudyGaitOutput.mat'];
 save(outputfilename, 'GaitSummary_FAT');
 
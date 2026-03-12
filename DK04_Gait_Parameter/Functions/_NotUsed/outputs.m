

clc;
%clear all ;
close all;

%add path to directory

ifMac = input('Please choose computer is being used for analysis? \n Enter 0 for Mac or 1 for Non Mac or Junk\n');

switch ifMac
    case 0
        DIR_gt = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData/Data';
        DIR_opt = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData/Output_FAT';
        DIR_an = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData';
        DIR_ancc = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData/GaitCodes_FAT/MinBoundSuite/MinBoundSuite';
        pathCompSep = '/';
    case 1
        
        DIR_gt='Z:\Projects\NCM\NCM_EXP\NCM_FAT\for Navrag\Gait\analyseData\Data';
        DIR_opt='Z:\Projects\NCM\NCM_EXP\NCM_FAT\for Navrag\Gait\analyseData\Output_FAT';
        DIR_opt='Z:\Projects\NCM\NCM_EXP\NCM_FAT\for Navrag\Gait\analyseData';
        DIR_ancc = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData/GaitCodes_FAT/MinBoundSuite/MinBoundSuite';
        pathCompSep = '\';
end

addpath(DIR_gt);
addpath(DIR_opt);
addpath(DIR_an);
addpath(DIR_ancc);
currentfolder = pwd;
addpath(currentfolder);

%load('GaitSummary_FATStudy_v01.mat')

vis=false; %show pictures
name_study = 'FAT';

colorset = {'ob', 'or', 'oc', 'om'};
colorsetcc = {'b', 'r', 'c', 'm'};
colorsetid = 0;
figureGaitFAT = figure;
ax = gca;
set(ax, 'FontSize', 14, 'FontName', 'Calibri');

figureGaitFAT1 = figure;
ax = gca;
set(ax, 'FontSize', 14, 'FontName', 'Calibri');

figureGaitFAT3D = figure;
ax = gca;
set(ax, 'FontSize', 14, 'FontName', 'Calibri');

figureGaitFATwl = figure;
ax = gca;
set(ax, 'FontSize', 14, 'FontName', 'Calibri');

partFolders = dir(DIR_gt);
partFoldernames = {partFolders.name};
for parti = 13%:length(partFoldernames)
    
    parti=char(partFoldernames(parti)); %name of participant to show figures
    
    %DIR_subject = [DIR_gt, pathCompSep, name_participant];
    
    if strcmp(name_study, 'FAT')
        typ_Exercise = {'Run', 'Squat'};
        time_Fatigue = {'Pre_gt', 'Post_gt'};
    end
    
    for typE = 1%1:size(typ_Exercise, 2)
        
        %DIR_data_tE = [DIR, pathCompSep, typ_Exercise{typE}];
        ex = typ_Exercise{typE};
        for tiFat = 1:size(time_Fatigue,2)
            %DIR_data = [DIR_data_tE, pathCompSep, time_Fatigue{tiFat}];
            colorsetid = colorsetid + 1;
            tf = time_Fatigue{tiFat};
            
            Stepwidth = GaitSummary_FAT.(parti).(ex).(tf).Total.GaitParameters.stepWidthR;
            Stridelength = GaitSummary_FAT.(parti).(ex).(tf).Total.GaitParameters.strideLengthR;
            Stridetime = GaitSummary_FAT.(parti).(ex).(tf).Total.GaitParameters.durationGaitCycleR;
            MTC = GaitSummary_FAT.(parti).(ex).(tf).Total.GaitParameters.MTCright;
            
            set(0, 'CurrentFigure', figureGaitFAT);
            hold on;
            noofplotValues = min(size(Stridelength, 2), size(Stridetime, 2));
            plot(Stridelength(1:noofplotValues), Stridetime(1:noofplotValues), colorset{colorsetid});
            xlabel('Stride length [cms]');
            ylabel('Stride time [s]');
                        xlim([125 145]);
                        ylim([0.95 1.10]);
            %             hold on;
            %             [center_lt, radius_lt] = ...
            %                 minboundcircle(Stridelength(1:noofplotValues), Stridetime(1:noofplotValues));
            %             viscircles(center_lt, radius_lt);
            
            set(0, 'CurrentFigure', figureGaitFAT1);
            hold on;
            noofplotValues = min(size(Stridelength, 2), size(Stridetime, 2));
            plot(Stridelength(1:noofplotValues), Stridetime(1:noofplotValues), colorset{colorsetid});
            xlabel('Stride length [cms]');
            ylabel('Stride time [s]');
                        xlim([125 145]);
                        ylim([0.95 1.10]);
            k = convhull(Stridelength(1:noofplotValues), Stridetime(1:noofplotValues));
            hold on;
            plot(Stridelength(k), Stridetime(k));
            
            set(0, 'CurrentFigure', figureGaitFAT3D);
            hold on;
            noofplotValues3d = min(noofplotValues, size(Stepwidth, 2));
            plot3(Stridelength(1:noofplotValues3d), Stepwidth(1:noofplotValues3d), Stridetime(1:noofplotValues3d), colorset{colorsetid});
            %plot(Stridelength(1:noofplotValues3d), Stepwidth(1:noofplotValues3d), colorset{colorsetid});
            ylabel('Step width [cms]');
            xlabel('Stride length [cms]');
            zlabel('Stride time [s]');
            %xlim([125 145]);
            %ylim([0.95 1.15]);
            
            
            set(0, 'CurrentFigure', figureGaitFATwl);
            hold on;
            
            %plot3(Stridelength(1:noofplotValues3d), MTC(1:noofplotValues3d)./10, Stridetime(1:noofplotValues3d), colorset{colorsetid});
            plot(Stridelength(1:noofplotValues3d), Stepwidth(1:noofplotValues3d), colorset{colorsetid});
            ylabel('Step width [cms]');
            xlabel('Stride length [cms]');
            %zlabel('Stride time [s]');
            xlim([125 145]);
            ylim([0 10]);
            
        end
    end
end
set(0, 'CurrentFigure', figureGaitFAT);
h_legend = legend('Pre fatigue', 'Post fatigue', 'Location', 'northeast');
figurefilename_tiff = [DIR_opt, pathCompSep, 'lengthvstime.tiff'];
print(figureGaitFAT, '-dtiff', '-r600', figurefilename_tiff);

set(0, 'CurrentFigure', figureGaitFAT1);
h_legend = legend('Pre fatigue', 'Pre-fatigue cloud', 'Post fatigue', 'Post fatigue cloud', 'Location', 'northeast');
figurefilename1_tiff = [DIR_opt, pathCompSep, 'lengthvstime1.tiff'];
print(figureGaitFAT1, '-dtiff', '-r600', figurefilename1_tiff);

set(0, 'CurrentFigure', figureGaitFATwl);
h_legend = legend('Pre fatigue', 'Post fatigue', 'Location', 'northeast');
figurefilenamelw_tiff = [DIR_opt, pathCompSep, 'lengthvswidth.tiff'];
print(figureGaitFATwl, '-dtiff', '-r600', figurefilenamelw_tiff);

set(0, 'CurrentFigure', figureGaitFAT3D);
h_legend1 = legend('Pre fatigue', 'Post fatigue', 'Location', 'northeast');




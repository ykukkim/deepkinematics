% % % This script cuts White and Pink SonEMS data in its subparts % % %
% % % Michelle Gwerder, September 2022

clear all; close all; clc;


%% Define the path and group to analyze

% TestPath1 = 'P:\Projects\NCM\NCM_SonEMS\02_Data';
TestPath1 = 'C:\Users\schauba\Documents';
% TestPath1 = 'P:\Projects\NCM\NCM_SonEMS\project_only\02_Data';
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
addpath(genpath('P:\Projects\NCM\NCM_SonEMS\project_only\01_Codes\Vicon\Gait\Functions'))

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


endingString = '-rot_mat';

matchingFolders = findFoldersEndingWithString(baseDirectory, endingString);


%% Load the Data
for subjectloop = 16%[8:9,11:14]%2:length(listL1)

    procVic = 'Processed\Vicon\';
    TestPath3 = [TestPath2,pathCompSep,listL1(subjectloop).name,pathCompSep,procVic,'matfiles'];
    addpath(genpath(TestPath3))

    destPath = [TestPath2,pathCompSep,listL1(subjectloop).name,pathCompSep,procVic,'matfiles'];
    
    % White
    data_1 = load('White');
    VD = data_1.VD;

    marker=fieldnames(VD.Marker);
    
    % Comment Anna: Maybe take some seconds out for each trial so they all
    % have the same length? Would also be better for small errors when
    % switching noise etc

    VD_1=VD;
    for i=1:length(marker)
        VD_1.Marker.(marker{i}) = [];
        VD_1.Marker.(marker{i})=VD.Marker.(marker{i})(5000:28999,:);
    end

    VD_2=VD;
    for i=1:length(marker)
        VD_2.Marker.(marker{i}) = [];
        VD_2.Marker.(marker{i})=VD.Marker.(marker{i})(29000:52999,:);
    end

    VD_3=VD;
    for i=1:length(marker)
        VD_3.Marker.(marker{i}) = [];
        VD_3.Marker.(marker{i})=VD.Marker.(marker{i})(53000:76999,:);
    end

    VD_4=VD;
    for i=1:length(marker)
        VD_4.Marker.(marker{i}) = [];
        VD_4.Marker.(marker{i})=VD.Marker.(marker{i})(77000:100999,:);
    end

    VD_5=VD;
    for i=1:length(marker)
        VD_5.Marker.(marker{i}) = [];
        VD_5.Marker.(marker{i})=VD.Marker.(marker{i})(101000:124999,:);
    end

    VD_6=VD;
    for i=1:length(marker)
        VD_6.Marker.(marker{i}) = [];
        VD_6.Marker.(marker{i})=VD.Marker.(marker{i})(125000:148863,:);
    end
    
    % Sorting trials, added by AS, 14.9.22
    noise_ord = zeros(6,1);
    for noise = 1:6
        % enter gain or db in order they were played for this participant
        inputname = str2double(input(['Noise played on place ' num2str(noise) ':  '],'s'));
        noise_ord(noise) = inputname;
    end

    VD_sort = [VD_1 VD_2 VD_3 VD_4 VD_5 VD_6];
    for k = 1:length(VD_sort)
        for j = 1:length(VD_sort)-1
            if noise_ord(j)<noise_ord(j+1)
                temp1 = noise_ord(j);
                noise_ord(j) = noise_ord(j+1);
                noise_ord(j+1) = temp1;
                temp2 = VD_sort(j);
                VD_sort(j) = VD_sort(j+1);
                VD_sort(j+1) = temp2;
            end
        end
    end

    % Save trials: White_1 to White_6 are now in order: -10 to -40 gain

%     VD=VD_1;
    VD=VD_sort(1);
    datasave= fullfile(destPath,'White_1');
    save(datasave,"VD",'-v7.3');

%     VD=VD_2;
    VD=VD_sort(2);
    datasave= fullfile(destPath,'White_2');
    save(datasave,"VD",'-v7.3');

%     VD=VD_3;
    VD=VD_sort(3);
    datasave= fullfile(destPath,'White_3');
    save(datasave,"VD",'-v7.3');

%     VD=VD_4;
    VD=VD_sort(4);
    datasave= fullfile(destPath,'White_4');
    save(datasave,"VD",'-v7.3');

%     VD=VD_5;
    VD=VD_sort(5);
    datasave= fullfile(destPath,'White_5');
    save(datasave,"VD",'-v7.3');

%     VD=VD_6;
    VD=VD_sort(6);
    datasave= fullfile(destPath,'White_6');
    save(datasave,"VD",'-v7.3');


    % Pink
    data_1 = load('Pink');
    VD = data_1.VD;

    marker=fieldnames(VD.Marker);

    VD_1=VD;
    for i=1:length(marker)
        VD_1.Marker.(marker{i}) = [];
        VD_1.Marker.(marker{i})=VD.Marker.(marker{i})(5000:28999,:);
    end

    VD_2=VD;
    for i=1:length(marker)
        VD_2.Marker.(marker{i}) = [];
        VD_2.Marker.(marker{i})=VD.Marker.(marker{i})(29000:52999,:);
    end

    % Sorting, added by AS, 14.9.22
    noise_ord = zeros(2,1);
    for noise = 1:2
        % enter gain or db in order they were played for this participant
        inputname = str2double(input(['Noise played on place ' num2str(noise) ':  '],'s'));
        noise_ord(noise) = inputname;
    end
    
    VD_sort = [VD_1 VD_2];
    if noise_ord(1)<noise_ord(2)
        temp1 = noise_ord(1);
        noise_ord(1) = noise_ord(2);
        noise_ord(2) = temp1;
        temp2 = VD_sort(1);
        VD_sort(1) = VD_sort(2);
        VD_sort(2) = temp2;
    end

% Saving trials: Pink_1 is -10, Pink_2 is -37 gain

%    VD=VD_1;
    VD=VD_sort(1);
    datasave= fullfile(destPath,'Pink_1');
    save(datasave,"VD",'-v7.3');

%    VD=VD_2;
    VD=VD_sort(2);
    datasave= fullfile(destPath,'Pink_2');
    save(datasave,"VD",'-v7.3');

end
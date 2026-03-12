




standfolderPath = [DIR, pathCompSep, 'Stand_'];
% cd(standfolderPath);
% spot = dir('*.mat');
% names = spot.name;
% k = 1; % there is only 1 file for normal stance.
%[analogdata,coord,analfreq,videofreq,parameter,group,fail,event] = new_old_converter_V2(spot,k);
createsubject_forFAT;

% cd ..;

standFolder = ['Stand_'];
if exist(standFolder, 'dir') ~= 7
    mkdir(standFolder);
end

subjectfolderPath = [DIR, pathCompSep, 'Static_'];
currentFolder = pwd;
filetoMove = [currentFolder, '/subject*.mat'];
movefile(filetoMove,subjectfolderPath);
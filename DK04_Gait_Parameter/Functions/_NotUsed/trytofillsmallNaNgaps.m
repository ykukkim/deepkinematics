clc
clear all
close all

%% load the file

kinematicData = load('GAAL_93_Post_gt_03.mat');
kinematicDataVar = fieldnames(kinematicData);

datatoFill = kinematicData.(genvarname(kinematicDataVar{2}));
markernames = fieldnames(datatoFill);


% [datatoFillIDs, interpolatedData] = interpolateNaN(datatoFill, markernames);

for i = 1:length(markernames)-1
    if (strcmp(markernames{i}, 'RTFM') ~=0)
    [timeSeriesx, startidx, endidx] = interpolateNaN(datatoFill.(genvarname(markernames{i})).Values.x_coord);
    interpolatedData.VD.(genvarname(markernames{i})).Values.x_coord = timeSeriesx;
    isnanIDarray = [startidx endidx];
    datatoFillIDs.isnanIDs.(genvarname(markernames{i})) = isnanIDarray;
    end
end
%markernames = ;
%for
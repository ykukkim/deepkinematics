function [GaitEvents , error] = StepDetection_noise(interpolatedVD, VD, Name,Test)
error=false; % if for any reason the trial was invalid, a true value is assigned to this variable
GaitEvents=struct;
ctrialsfewEventsnames  = {};
cStepSkippedByFVAnames  = {};
ctrialsfewEvents      = 0;
cStepSkippedByFVA     = 0;

%% fva, find HS and TO
% Input parameters for the algorithm determined through approximate
% walking speed of the participant as well as via experience.
toprdfac  = 0.8;  % Factor for discarding TO more than 1 step per 60 - 70 frames
hsprdfac  = 0.08; % Factor for discarding HS
zrangefac = 0.4; % Heel marker z coordinate by the HS event should be in the range [min(z), min(z) + zrangefac*(max(z) - min(z))]
% zrangefac = 0.15
filterOrder = 4; % seems to provide the best results
cutfreqFilter = 25; % Not much happening in walking task above 25 Hz
try
    % adapted since all trials last for 2min
    [HSleftlocs, TOleftlocs,zfootVelLeft] = fva(double(interpolatedVD.LHEE.Values.z_coord),double(interpolatedVD.LTO3.Values.z_coord), ...
        VD.SF, toprdfac, hsprdfac, zrangefac, filterOrder, cutfreqFilter);

    [HSrightlocs, TOrightlocs,zfootVelRight] = fva(double(interpolatedVD.RHEE.Values.z_coord), double(interpolatedVD.RTO3.Values.z_coord), ...
        VD.SF, toprdfac, hsprdfac, zrangefac, filterOrder, cutfreqFilter);

    HSleft = interpolatedVD.LHEE.Values.z_coord(HSleftlocs);
    TOleft = interpolatedVD.LTO3.Values.z_coord(TOleftlocs);
    HSright = interpolatedVD.RHEE.Values.z_coord(HSrightlocs);
    TOright = interpolatedVD.RTO3.Values.z_coord(TOrightlocs);
catch
    fprintf("Data is broken in %s %s\n", Name,Test);
end
%% check data on few events

fewEvents = length(HSleftlocs) < 2 || length(TOleftlocs) < 1 || length(HSrightlocs) < 2 || length(TOrightlocs) < 1;

if fewEvents
    ctrialsfewEvents = ctrialsfewEvents + 1;
    ctrialsfewEventsnames(ctrialsfewEvents, 1:2) = {Name, VD.Marker};
    error=true;
    return;
end

%% Format the data. Start with first HS, end with HS or TO
[HSleft, HSleftlocs, TOleft, TOleftlocs, wrongRecogL] = formatdata(HSleft, HSleftlocs, TOleft, TOleftlocs);
[HSright, HSrightlocs, TOright, TOrightlocs, wrongRecogR] = formatdata(HSright, HSrightlocs, TOright, TOrightlocs);

if wrongRecogL || wrongRecogR
    cStepSkippedByFVA = cStepSkippedByFVA + 1;
    cStepSkippedByFVAnames(cStepSkippedByFVA, 1:2) = {Name, VD.Marker};
    error=true;
    return;
end

%% Check if any steps skipped (HS not detected) by FVA

l = zeros(1, length(HSleftlocs));
r = zeros(1, length(HSrightlocs));
l(1:end) = 'L';     % left
r(1:end) = 'R';     % right
hs_matrix = [HSleftlocs HSrightlocs; l r];
hs_matrix = sortrows(hs_matrix', 1);
HS_HH_TT = find((diff(hs_matrix(:,2))==0)==1);
HS_HH_TT_size = length(HS_HH_TT);

ltoe = zeros(1, length(TOleftlocs));
rtoe = zeros(1, length(TOrightlocs));
ltoe(1:end) = 'L';     % left
rtoe(1:end) = 'R';     % right
to_matrix = [TOleftlocs TOrightlocs; ltoe rtoe];

to_matrix = sortrows(to_matrix', 1);
TO_HH_TT = find((diff(to_matrix(:,2))==0)==1);
TO_HH_TT_size = length(TO_HH_TT);

for a = HS_HH_TT_size(1,1):-1:1
    hs_matrix(HS_HH_TT(a,:),:) = [];
end

for a = TO_HH_TT_size(1,1):-1:1
    to_matrix(TO_HH_TT(a,:),:) = [];
end

for j = 1:size(hs_matrix, 1)-1
    stepMissed = isequal(hs_matrix(j, 2), 'L') && isequal(hs_matrix(j+1, 2), 'L') || isequal(hs_matrix(j, 2), 'R') && isequal(hs_matrix(j+1, 2), 'R');
    if stepMissed
        break;
    end
end

if stepMissed
    cStepSkippedByFVA = cStepSkippedByFVA + 1;
    cStepSkippedByFVAnames(cStepSkippedByFVA, 1:2) = {Name, VD.Marker};
    disp('stepMissed, end loop')
    error=true;
    return;
end

fewEvents = length(HSleft) < 2 || length(TOleft) < 1 || length(HSright) < 2 || length(TOright) < 1;
if fewEvents
    ctrialsfewEvents = ctrialsfewEvents + 1;
    ctrialsfewEventsnames(ctrialsfewEvents, 1:2) = {Name, VD.Marker};
    error=true;
    return;
end

hs_leftlocs = hs_matrix(:,2) == 'L';
hs_rightlocs = hs_matrix(:,2) == 'R';
HSrightlocs = hs_matrix(hs_rightlocs,1)';
HSleftlocs = hs_matrix(hs_leftlocs,1)';

to_leftlocs = to_matrix(:,2) == 'L';
to_rightlocs = to_matrix(:,2) == 'R';
TOrightlocs = to_matrix(to_rightlocs,1)';
TOleftlocs= to_matrix(to_leftlocs,1)';

HSleft = interpolatedVD.LHEE.Values.z_coord(HSleftlocs);
TOleft = interpolatedVD.LTO3.Values.z_coord(TOleftlocs);
HSright = interpolatedVD.RHEE.Values.z_coord(HSrightlocs);
TOright = interpolatedVD.RTO3.Values.z_coord(TOrightlocs);
HSleft = HSleft';
TOleft = TOleft';
HSright = HSright';
TOright = TOright';
hleft = zeros(1, length(HSleftlocs));

tleft = zeros(1, length(TOleftlocs));
hleft(1:end) = 'H' ;   % hs
tleft(1:end) = 'T'  ;  % to
Left_hs_to_matrix = [HSleft TOleft; HSleftlocs TOleftlocs; hleft tleft];
Left_hs_to_matrix = sortrows(Left_hs_to_matrix', 2);

hright = zeros(1, length(HSrightlocs));
tright = zeros(1, length(TOrightlocs));
hright(1:end) = 'H' ;   % hs
tright(1:end) = 'T'  ;  % to
Right_hs_to_matrix = [HSright TOright; HSrightlocs TOrightlocs; hright tright];
Right_hs_to_matrix = sortrows(Right_hs_to_matrix', 2);
%% create structure

GaitEvents.HSleft=HSleft;
GaitEvents.HSleftlocs=HSleftlocs;
GaitEvents.HSright=HSright;
GaitEvents.HSrightlocs=HSrightlocs;
GaitEvents.TOleft=TOleft;
GaitEvents.TOleftlocs=TOleftlocs;
GaitEvents.TOright=TOright;
GaitEvents.TOrightlocs=TOrightlocs;
GaitEvents.zfootVelRight= zfootVelRight;
GaitEvents.zfootVelLeft= zfootVelLeft;
GaitEvents.Left_matrix = Left_hs_to_matrix;
GaitEvents.Right_matrix = Right_hs_to_matrix;

%% Number of Gait events

nHSleft = length(HSleft);
nTOleft = length(TOleft);
nHSright = length(HSright);
nTOright = length(TOright);
GaitEvents.nHSleft=nHSleft;
GaitEvents.nHSright=nHSright;
GaitEvents.nTOleft=nTOleft;
GaitEvents.nTOright=nTOright;

end


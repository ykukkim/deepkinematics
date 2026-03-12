%% Extraction of Spatio-temproal gait parameters
% 1. Midstance & Midswing
% 2. Duration
%   i   duration Gait cycle
%   ii  duration Stance Phase
%   iii duration Swing Phase
%   iv  duration of double limb support
% 3. Stride and Step lengths
% 4. Stepwidth
% 5. Phase Coordination Index
    
function [GaitEvents, GaitParameters] = GaitWork_Treadmill(interpolatedVD, GaitEvents,VD, ~,~)

%% Adding velocity of treadmill to markers in y-direction
% For treadmill measurement, velocity is added onto markers
% Need set the window with a positive gradient of the slope to get the
% treadmil velocity.
% If it is over-gorund please comment this one out.
Markers = {'RTO3'; 'LTO3'; 'RAJC'; 'LAJC'};
sf=VD.SF;
[VD_tmvy] = f_approxVelocity_treadmill(VD.Marker,Markers,sf);

%% Reallocation of Variables
HSleftlocs=GaitEvents.HSleftlocs;
HSrightlocs=GaitEvents.HSrightlocs;
TOleftlocs=GaitEvents.TOleftlocs;
TOrightlocs=GaitEvents.TOrightlocs;

%% Stride and Step lengths
[strideLengthL, strideLengthR, strideWidthL, strideWidthR, stepLengthL, stepLengthR, ...
    durationStepL, durationStepR, Cadence] = ...
    strLength_Treadmill(VD_tmvy.LAJC(:,1),VD_tmvy.LAJC(:,2),VD_tmvy.RAJC(:,1),VD_tmvy.RAJC(:,2), HSleftlocs, HSrightlocs, VD.SF);

GaitParameters.strideLengthL=strideLengthL;
GaitParameters.strideLengthR=strideLengthR;
GaitParameters.strideWidthL=strideWidthL;
GaitParameters.strideWidthR=strideWidthR;
GaitParameters.stepLengthL=stepLengthL;
GaitParameters.stepLengthR=stepLengthR;
GaitParameters.stepTimeL = durationStepL;
GaitParameters.stepTimeR = durationStepR;
GaitParameters.Cadence = Cadence;


%% Stepwidth
[stepWidthL, stepWidthR] = ...
    stepwidth(VD.Marker.LAJC(:,1), ...
    VD.Marker.LAJC(:,2), ...
    VD.Marker.RAJC(:,1), ...
    VD.Marker.RAJC(:,2), ...
    HSleftlocs, HSrightlocs);

GaitParameters.stepWidthL = stepWidthL;
GaitParameters.stepWidthR = stepWidthR;

%% Stepwidth
[stepWidthL_T, stepWidthR_T] = ...
    stepwidth_Treadmill(VD.Marker.LAJC(:,1), ...
    VD.Marker.LAJC(:,2), ...
    VD.Marker.RAJC(:,1), ...
    VD.Marker.RAJC(:,2), ...
    HSleftlocs, HSrightlocs,...
    TOleftlocs, TOrightlocs);

GaitParameters.stepWidthL_T = stepWidthL_T;
GaitParameters.stepWidthR_T = stepWidthR_T;

end






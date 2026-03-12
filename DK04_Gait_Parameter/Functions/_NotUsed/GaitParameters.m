function [GaitEvents, GaitParameters, GaitCycles] = Gaitwork(interpolatedVD, GaitEvents,VD, ~,~)

%% Reallocation of Variables
% Name = filename(1:3);
% No = filename(5:7);
% Test = filename(9:12);

HSleftlocs=GaitEvents.HSleftlocs;
HSleft=GaitEvents.HSleft;
nHSleft=GaitEvents.nHSleft;
HSrightlocs=GaitEvents.HSrightlocs;
HSright=GaitEvents.HSright;
nHSright=GaitEvents.nHSright;
TOleftlocs=GaitEvents.TOleftlocs;
TOleft=GaitEvents.TOleft;
TOrightlocs=GaitEvents.TOrightlocs;
TOright=GaitEvents.TOright;
zfootVelRight = GaitEvents.zfootVelRight;
zfootVelLeft = GaitEvents.zfootVelLeft;
vis=false; %visualisation of figures

%% findmidstance

[LmidStlocs, RmidStlocs] = findmidstance2(...
    interpolatedVD.LLMA.Values.x_coord, interpolatedVD.LLMA.Values.y_coord,...
    interpolatedVD.RLMA.Values.x_coord, interpolatedVD.RLMA.Values.y_coord,...
    HSleftlocs, HSrightlocs, TOleftlocs, TOrightlocs,...
    zfootVelRight, zfootVelLeft);

LmidSwlocs = RmidStlocs;
RmidSwlocs = LmidStlocs;

GaitParameters.LmidStlocs=LmidStlocs;
GaitParameters.RmidStlocs=RmidStlocs;
GaitParameters.LmidSwlocs=LmidSwlocs;
GaitParameters.RmidSwlocs=RmidSwlocs;
GaitEvents = orderfields(GaitEvents);

%% Time of occurance of mid-stance and mid-swing events
%(stance/swing phase)

[midStOccTimeL, midStOccTimeR, midSwOccTimeL, midSwOccTimeR]= midtimes(...
    LmidStlocs, RmidStlocs, HSleftlocs, HSrightlocs, TOleftlocs, TOrightlocs,...
    LmidSwlocs, RmidSwlocs);

GaitParameters.midStOccTimeL=midStOccTimeL;
GaitParameters.midStOccTimeR=midStOccTimeR;
GaitParameters.midSwOccTimeL=midSwOccTimeL;
GaitParameters.midSwOccTimeR=midSwOccTimeR;

%% Duration of gait cycle events(s)
%duration Gait cycle
%duration Stance Phase
%duration Swing Phase
%duration of double limb support

[durationGaitCycleL, durationGaitCycleR, durationStancePhL, ...
    durationStancePhR, durationSwingPhL, durationSwingPhR, ...
    dlsL, dlsR, ndlsL, ndlsR] = durationPhase(HSleft, HSright, ...
    TOleft, TOright, HSleftlocs,HSrightlocs, TOleftlocs, TOrightlocs,...
    VD.SF);

GaitParameters.durationGaitCycleL=durationGaitCycleL;
GaitParameters.durationGaitCycleR=durationGaitCycleR;
GaitParameters.durationStancePhL=durationStancePhL;
GaitParameters.durationStancePhR=durationStancePhR;
GaitParameters.durationSwingPhL=durationSwingPhL;
GaitParameters.durationSwingPhR=durationSwingPhR;
GaitParameters.dlsL=dlsL;
GaitParameters.dlsR=dlsR;
GaitParameters.ndlsL=ndlsL;
GaitParameters.ndlsR=ndlsR;

%% Stride and Step lengths

[strideLengthL, strideLengthR, stepLengthL, stepLengthR, ...
    hs_matrix_new, durationStepL, durationStepR, Cadence] = ...
    strLength(interpolatedVD.LHEE.Values.x_coord, ...
    interpolatedVD.LHEE.Values.y_coord, ...
    interpolatedVD.RHEE.Values.x_coord, ...
    interpolatedVD.RHEE.Values.y_coord,...
    HSleft,HSright, HSleftlocs, HSrightlocs, VD.SF );

GaitParameters.strideLengthL=strideLengthL;
GaitParameters.strideLengthR=strideLengthR;
GaitParameters.stepLengthL=stepLengthL;
GaitParameters.stepLengthR=stepLengthR;
GaitParameters.stepTimeL = durationStepL;
GaitParameters.stepTimeL = durationStepR;
GaitParameters.Cadence = Cadence;


%% Stepwidth
[stepWidthL, stepWidthR] = ...
    stepwidth(interpolatedVD.LHEE.Values.x_coord, ...
    interpolatedVD.LHEE.Values.y_coord, ...
    interpolatedVD.RHEE.Values.x_coord, ...
    interpolatedVD.RHEE.Values.y_coord, ...
    hs_matrix_new);

GaitParameters.stepWidthL = stepWidthL;
GaitParameters.stepWidthR = stepWidthR;

%% Get mtc: Minimum toe clearance from the ground
%Data from left and right side are in one file

[MTCleft, MTCleftlocs]=get_mtc(interpolatedVD,TOleftlocs,HSleftlocs,'LTO3');
[MTCright, MTCrightlocs]=get_mtc(interpolatedVD,TOrightlocs,HSrightlocs,'RTO3');

GaitParameters.MTCleftlocs=MTCleftlocs;
GaitParameters.MTCrightlocs=MTCrightlocs;
GaitParameters.MTCleft=MTCleft;
GaitParameters.MTCright=MTCright;

%% Gait Asymmetry

[GA_shortvslong, GA_leftvsright] = gaitasymmetry(durationSwingPhL, durationSwingPhR);

GaitParameters.GaitAsymmetry_ShortvsLong = GA_shortvslong;
GaitParameters.GaitAsymmetry_LeftvsRight = GA_leftvsright;

%% Phase Coordination Index

[PCI_shortvslong, PCI_leftvsright] = phasecoordindex(durationSwingPhL, durationSwingPhR, ...
    durationGaitCycleL, durationGaitCycleR, durationStepL, durationStepR);

GaitParameters.PhaseCoordinationIndex_ShortvsLong = PCI_shortvslong;
GaitParameters.PhaseCoordinationIndex_LeftvsRight = PCI_leftvsright;

GaitParameters = orderfields(GaitParameters);
%% Normalize step data in % Heelstrikes as GaitCycles

% Linear trajectories or marker positions
[GaitCycle_Left_mediolateral, GaitCycle_Right_mediolateral] ...
    = normStep(interpolatedVD, nHSleft, nHSright, ...
    HSleftlocs, HSrightlocs, 'x_coord');
[GaitCycle_Left_anteroposterior, GaitCycle_Right_anteroposterior] ...
    = normStep(interpolatedVD, nHSleft, nHSright, ...
    HSleftlocs, HSrightlocs, 'y_coord');
[GaitCycle_Left_longitudinal, GaitCycle_Right_longitudinal] ...
    = normStep(interpolatedVD, nHSleft, nHSright, ...
    HSleftlocs, HSrightlocs, 'z_coord');

GaitCycles.OrigInterpolated.Left.mediolateral = GaitCycle_Left_mediolateral;
GaitCycles.OrigInterpolated.Left.anteroposterior = GaitCycle_Left_anteroposterior;
GaitCycles.OrigInterpolated.Left.longitudinal = GaitCycle_Left_longitudinal;

GaitCycles.OrigInterpolated.Right.mediolateral = GaitCycle_Right_mediolateral;
GaitCycles.OrigInterpolated.Right.anteroposterior = GaitCycle_Right_anteroposterior;
GaitCycles.OrigInterpolated.Right.longitudinal = GaitCycle_Right_longitudinal;


%% Transformation to the components here on GaitCycles data - do not need for treadmil
% Rotating the linear trajectories for walking in both directions

% [GaitCycle_Left_rotated_ml, GaitCycle_Left_rotated_ap, GaitCycle_Left_rotated_ln] ...
%     = applyrotationMatrix(GaitCycle_Left_mediolateral, ...
%     GaitCycle_Left_anteroposterior, GaitCycle_Left_longitudinal, ...
%     walkingDirection.walkingDirectionString);

% [GaitCycle_Right_rotated_ml, GaitCycle_Right_rotated_ap, GaitCycle_Right_rotated_ln] ...
%     = applyrotationMatrix(GaitCycle_Right_mediolateral, ...
%     GaitCycle_Right_anteroposterior, GaitCycle_Right_longitudinal, ...
%     walkingDirection.walkingDirectionString);

% GaitCycles.Transformed.Left.mediolateral = GaitCycle_Left_rotated_ml;
% GaitCycles.Transformed.Left.anteroposterior = GaitCycle_Left_rotated_ap;
% GaitCycles.Transformed.Left.longitudinal = GaitCycle_Left_rotated_ln;

% GaitCycles.Transformed.Right.mediolateral = GaitCycle_Right_rotated_ml;
% GaitCycles.Transformed.Right.anteroposterior = GaitCycle_Right_rotated_ap;
% GaitCycles.Transformed.Right.longitudinal = GaitCycle_Right_rotated_ln;

%% Calculating dimensionless velocity
% [walkingSpeed, dimlessSpeed] = dimensionlessVelocity(interpolatedVD,HSleftlocs, HSrightlocs, VD.SF);
%
% GaitParameters.walkingSpeed = walkingSpeed;
% GaitParameters.dimensionlessSpeed = dimlessSpeed;

end




function [strideLengthL, strideLengthR, strideWidthL, strideWidthR, ...
    stepLengthL, stepLengthR, ...
    stepTimeL, stepTimeR, cadence] = ...
    strLength_Treadmill(x_LHEE,y_LHEE,x_RHEE,y_RHEE,...
    HSleftlocs,HSrightlocs,...
    samplingRate)

%% finds stride length
nHSleft = length(HSleftlocs);
nHSright = length(HSrightlocs);

%% xy coordinates from heel markers.
y_LHEE = y_LHEE - mean(y_LHEE);
y_RHEE = y_RHEE - mean(y_RHEE);

xy_LHEE=[x_LHEE , y_LHEE];
xy_RHEE=[x_RHEE , y_RHEE];

%% (projected onto transverse plane) (cm)
strideLengthL = nan(1, nHSleft-1);
for j=1:nHSleft-1
    strideLengthL(j) = 1/10 * norm(xy_LHEE(HSleftlocs(j+1),1:2) - xy_LHEE(HSleftlocs(j), 1:2));
end

strideLengthR = nan(1, nHSright-1);
for j=1:nHSright-1
    strideLengthR(j) = 1/10 * norm(xy_RHEE(HSrightlocs(j+1), 1:2) - xy_RHEE(HSrightlocs(j), 1:2));
end

%% (projected onto transverse plane) (cm)
strideWidthL = nan(1, nHSleft-1);
for j=1:nHSleft-1
    strideWidthL(j) = 1/10 * abs(xy_LHEE(HSleftlocs(j+1),1) - xy_LHEE(HSleftlocs(j), 1));
end

strideWidthR = nan(1, nHSright-1);
for j=1:nHSright-1
    strideWidthR(j) = 1/10 * abs(xy_RHEE(HSrightlocs(j+1), 1) - xy_RHEE(HSrightlocs(j), 1));
end

%% step length (projected onto transverse plane)
% (cm)
% Step duration (ms), Cadence (steps/s)
stepLengthL = nan(1, nHSleft);
stepLengthR = nan(1, nHSright);
stepTimeL = nan(1, nHSleft);
stepTimeR = nan(1, nHSright);

for j=1:min(nHSleft, nHSright)
    if HSleftlocs(j) <  HSrightlocs(j)
        stepLengthR(j) = 1/10*norm(abs(xy_RHEE(HSrightlocs(j), 1:2) ...
            - xy_LHEE(HSleftlocs(j), 1:2)));
        stepTimeR(j) = (HSrightlocs(j) - HSleftlocs(j))/samplingRate;
    end
end

for j=1:min(nHSleft, nHSright)-1
    if HSrightlocs(j) <  HSleftlocs(j+1)
        stepLengthL(j) = 1/10*norm(abs(xy_LHEE(HSleftlocs(j+1), 1:2) ...
            - xy_RHEE(HSrightlocs(j), 1:2)));
        stepTimeL(j) = (HSleftlocs(j+1) - HSrightlocs(j))/samplingRate;
    end
end

cadence = 60*(length(stepTimeL)+length(stepTimeR))...
    / ((HSrightlocs(end)-HSleftlocs(1))/samplingRate);

stepLengthL = stepLengthL(not(isnan(stepLengthL)));
stepLengthR = stepLengthR(not(isnan(stepLengthR)));
stepTimeL = stepTimeL(not(isnan(stepTimeL)));
stepTimeR = stepTimeR(not(isnan(stepTimeR)));
% stepLengthL = rmoutliers(stepLengthL);
% stepLengthR = rmoutliers(stepLengthR);
% stepTimeL = rmoutliers(stepTimeL);
% stepTimeR = rmoutliers(stepTimeR);

return;
end

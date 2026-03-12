function [walkingSpeed, dimSpeed] = dimensionlessVelocity_Treadmill(interpolatedVD, HSleftlocs, HSrightlocs, SF)

%% calculate dimensionless velocity
timeleft = (HSleftlocs(end) - HSleftlocs(1))/SF;
timeright = (HSrightlocs(end) - HSrightlocs(1))/SF;
% markerforVelocity = {'SACR'}; %MSTC is missed in many trials

Y_SACR = interpolatedVD.SACR(:,2);
X_SACR = interpolatedVD.SACR(:,1);

ydistleft = abs(Y_SACR(HSleftlocs(end)) ...
    - Y_SACR(HSleftlocs(1)));
ydistright = abs(Y_SACR(HSrightlocs(end)) ...
    - Y_SACR(HSrightlocs(1)));

% x axis is added by Mostafa:
xdistleft = abs(X_SACR(HSleftlocs(end)) ...
    - X_SACR(HSleftlocs(1)));
xdistright = abs(X_SACR(HSrightlocs(end)) ...
    - X_SACR(HSrightlocs(1)));

walkingSpeedleft = sqrt(ydistleft^2+xdistleft^2)/timeleft;
walkingSpeedright = sqrt(ydistright^2+xdistright^2)/timeright;

walkingSpeed = 0.1*(walkingSpeedleft + walkingSpeedright)/2; % Dimension: cm/sec

accGra=9.807; %acceleration of gravity
meanlegLength = 1000;

dimSpeed = walkingSpeed/sqrt(accGra*meanlegLength/1000);


return
end


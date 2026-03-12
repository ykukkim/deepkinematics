function [walkingSpeed, dimSpeed] = dimensionlessVelocity(interpolatedVD, HSleftlocs, HSrightlocs, SF)

%% calculate dimensionless velocity

timeleft = (HSleftlocs(end) - HSleftlocs(1))/SF;
timeright = (HSrightlocs(end) - HSrightlocs(1))/SF;
%markerforVelocity = 'MSTC';
markerforVelocity = 'SACR'; %MSTC is missed in many trials


%noofmarkers = fieldnames(interpolatedVD);

%for i = 1:length(noofmarkers)
if isfield(interpolatedVD, (genvarname('SACR'))) % to check if the marker exist in the data
    if all(interpolatedVD.(genvarname('SACR')).Values.y_coord(HSleftlocs(1:end)))% to check
        %if the marker was visible between the first and the last HS
        markerforVelocity='SACR';
    end
end

ydistleft = abs(interpolatedVD.(genvarname(markerforVelocity)).Values.y_coord(HSleftlocs(end)) ...
    - interpolatedVD.(genvarname(markerforVelocity)).Values.y_coord(HSleftlocs(1)));
ydistright = abs(interpolatedVD.(genvarname(markerforVelocity)).Values.y_coord(HSrightlocs(end)) ...
    - interpolatedVD.(genvarname(markerforVelocity)).Values.y_coord(HSrightlocs(1)));

% x axis is added by Mostafa:
xdistleft = abs(interpolatedVD.(genvarname(markerforVelocity)).Values.x_coord(HSleftlocs(end)) ...
    - interpolatedVD.(genvarname(markerforVelocity)).Values.x_coord(HSleftlocs(1)));
xdistright = abs(interpolatedVD.(genvarname(markerforVelocity)).Values.x_coord(HSrightlocs(end)) ...
    - interpolatedVD.(genvarname(markerforVelocity)).Values.x_coord(HSrightlocs(1)));

walkingSpeedleft = sqrt(ydistleft^2+xdistleft^2)/timeleft;
walkingSpeedright = sqrt(ydistright^2+xdistright^2)/timeright;

walkingSpeed = 0.1*(walkingSpeedleft + walkingSpeedright)/2; %Dimention: cm/sec


accGra=9.807; %acceleration of gravity
meanlegLength = 100;

dimSpeed = walkingSpeed/sqrt(accGra*meanlegLength/1000);


return
end


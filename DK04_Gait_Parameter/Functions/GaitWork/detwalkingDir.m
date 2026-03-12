%% Co-ordinates for laboratory
% laboratory x corresponds to mediolateral direction
% laboratory y corresponds to anteroposterior direction


function [walkingDirection] = detwalkingDir(interpolatedVD)

markerlist = {'RASI', 'LASI'}; % RASI or RTMS, LASI or LTMS must exist

for i = 1:length(markerlist)
    if (strcmp(markerlist{i}, 'RASI') ~= 0)
        if isfield(interpolatedVD,(markerlist{i}))
            asisRx = interpolatedVD.(genvarname(markerlist{i})).Values.x_coord;
            asisRy = interpolatedVD.(genvarname(markerlist{i})).Values.y_coord;

        else
            asisRx = interpolatedVD.RTMS.Values.x_coord;
            asisRy = interpolatedVD.RTMS.Values.y_coord;
        end
    elseif (strcmp(markerlist{i}, 'LASI') ~= 0)
        if isfield(interpolatedVD,(markerlist{i}))
            asisLx = interpolatedVD.(genvarname(markerlist{i})).Values.x_coord;
            asisLy = interpolatedVD.(genvarname(markerlist{i})).Values.y_coord;
        else
            asisRx = interpolatedVD.LTMS.Values.x_coord;
            asisRy = interpolatedVD.LTMS.Values.y_coord;
        end
    end
end

walkingDirection.(genvarname('walkingDirec')) = unwrap(atan2((asisRy-asisLy),(asisRx-asisLx)));
walkingDirection.(genvarname('meanwalkingAngle')) = mean(rad2deg(walkingDirection.walkingDirec));
%% N/A for Treadmill.
% if mean(rad2deg(walkingDirection.walkingDirec)) > 170  && mean(rad2deg(walkingDirection.walkingDirec)) < -10
%     walkingDirection.(genvarname('walkingDirectionString')) = 'Straight';
% else
%     walkingDirection.(genvarname('walkingDirectionString')) = 'Inclined_forward';
% end


return
end

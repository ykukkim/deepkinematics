function [VD]=f_approxVelocity_treadmill(Data,Markers,SF)
% Speed of treadmill
Ymid_right = Data.RTO3(10000:12000,2);
Ymid_left  = Data.LTO3(10000:12000,2);

%Calculates velocity per track
mpd = 0.8 * SF;
% [val_max_right, loc_max_right] = findpeaks(Ymid_right, 'minpeakdistance', mpd);
% [val_min_right, loc_min_right] = findpeaks(-Ymid_right,'minpeakdistance', mpd);
% [val_max_left,  loc_max_left]  = findpeaks(Ymid_left,  'minpeakdistance', mpd);
% [val_min_left,  loc_min_left]  = findpeaks(-Ymid_left, 'minpeakdistance', mpd);
[~, loc_max_right] = findpeaks(Ymid_right, 'minpeakdistance', mpd);
[~, loc_min_right] = findpeaks(-Ymid_right,'minpeakdistance', mpd);
[~,  loc_max_left]  = findpeaks(Ymid_left,  'minpeakdistance', mpd);
[~,  loc_min_left]  = findpeaks(-Ymid_left, 'minpeakdistance', mpd);


start_min_point = find(loc_min_right> loc_max_right(1));
loc_min_right = loc_min_right(start_min_point(1):end);
start_min_point = find(loc_min_left> loc_max_left(1));
loc_min_left = loc_min_left(start_min_point(1):end);

min_right = min(length(loc_max_right),length(loc_min_right));
min_left = min(length(loc_max_left),length(loc_min_left));


for i=1:min(min_right,min_left)-1
    idx     = find(loc_max_right > loc_min_right(i));
    speed_r_Y(i) = ((Ymid_right(floor(loc_max_right(idx(1))-0.05*SF)) - (Ymid_right(floor(loc_min_right(i)+0.05*SF))))...
        /(floor(loc_max_right(idx(1))-0.05*SF)-floor(loc_min_right(i)+0.05*SF)))*SF;
    idx     = find(loc_max_left > loc_min_left(i));
    speed_l_Y(i) = ((Ymid_left(floor(loc_max_left(idx(1))-0.05*SF)) - (Ymid_left(floor(loc_min_left(i)+0.05*SF))))...
        /(floor(loc_max_left(idx(1))-0.05*SF)-floor(loc_min_left(i)+0.05*SF)))*SF;
end


% for i=1:min(min_right,min_left)-1
%     idx     = find(loc_max_right > loc_min_right(i));
%     speed_r_Y(i) = ((Ymid_right(floor(loc_max_right(idx(1)))) - (Ymid_right(floor(loc_min_right(i)))))...
%         /(floor(loc_max_right(idx(1)))-floor(loc_min_right(i))))*SF;
%     idx     = find(loc_max_left > loc_min_left(i));
%     speed_l_Y(i) = ((Ymid_left(floor(loc_max_left(idx(1)))) - (Ymid_left(floor(loc_min_left(i)))))...
%         /(floor(loc_max_left(idx(1)))-floor(loc_min_left(i))))*SF;
% end


speed_l_Y = rmoutliers(speed_l_Y);
speed_r_Y = rmoutliers(speed_r_Y);

tm_v_Y    = (nanmean(speed_l_Y) + nanmean(speed_r_Y))/2;

% Iterate over each marker
for h = 1:length(Markers)
    
    Markername = Markers{h};  % Get marker name
    Marker = Data.(Markername);  % Get marker data
    
    % Preallocate the modified Y values
    Marker_v = zeros(length(Marker), 1);
    
    % Calculate mm/frame for the Y direction
    mmpf_Y = (1 / SF) * tm_v_Y;
    
    % Apply speed to the Y-axis for each frame
    Marker_v = Marker(:, 2) + ((mmpf_Y * (1:length(Marker)))');
    
    % Store the modified marker data in VD
    VD.(Markername) = [Marker(:, 1), Marker_v, Marker(:, 3)];
end
end

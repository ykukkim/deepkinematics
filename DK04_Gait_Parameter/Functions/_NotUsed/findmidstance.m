function [LmidStlocs, RmidStlocs] = findmidstance(...
    LMMAdata, RMMAdata, ...
    HSleftlocs, HSrightlocs, TOleftlocs, TOrightlocs, ...
    samplingRate, pathtovisu, i, participant, vis)
% Finds the mid-stance events in the stance phase for left and right feet
% using the kinematic definition of the mid-stance event.

n = size(LMMAdata, 1);
d = nan(n, 1);
for k = 1:n
    d(k) = norm(LMMAdata(k, 1:2) - RMMAdata(k, 1:2));
end

mpd = floor(0.3 * samplingRate);
% [midStCand, midStCandlocs] = findpeaks(-d, 'minpeakdistance', mpd);
[midStCand, midStCandlocs] = findlocalmax(d, round(mpd/2), 'min');

if vis
    figure('Visible','off');
    plot(d);
    hold on;
    plot(midStCandlocs, d(midStCandlocs), 'or');

    grid on;
    title('Distance between left and right heels');

    xlabel('Time (s)');
    ylabel('Distance (mm)');

    print('-djpeg', [pathtovisu '\mids\' num2str(i) '_' ...
            participant '_lrDist.jpg']);
end
    
    

n = min(length(HSleftlocs), length(TOleftlocs));
Lset = [];
for k = 1:n
    Lset = union(Lset, HSleftlocs(k):TOleftlocs(k));
end

n = min(length(HSrightlocs), length(TOrightlocs));
Rset = [];
for k = 1:n
    Rset = union(Rset, HSrightlocs(k):TOrightlocs(k));
end

LmidStlocs = midStCandlocs(ismember(midStCandlocs, Lset));
RmidStlocs = midStCandlocs(ismember(midStCandlocs, Rset));

return;
end
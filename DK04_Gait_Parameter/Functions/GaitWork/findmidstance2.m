function [LmidStlocs, RmidStlocs, d] = findmidstance2(...
    LMMA_x_data, LMMA_y_data, ...
    RMMA_x_data, RMMA_y_data, ...
    HSleftlocs, HSrightlocs, TOleftlocs, TOrightlocs, ...
    samplingRate, vis)
% Findsmid-stance events in the stance phase for left and right feet
% using the kinematic definition of the mid-stance event.

n = size(LMMA_x_data, 1);
d = nan(n, 1);
xy_L=[LMMA_x_data , LMMA_y_data];
xy_R=[RMMA_x_data , RMMA_y_data];
for k = 1000:n-1000
    d(k) = norm(xy_L(k, 1:2) - xy_R(k, 1:2)); %stepwidth
end

mpd = floor(0.3 * samplingRate);
% [midStCand, midStCandlocs] = findpeaks(-d, 'minpeakdistance', mpd);
[midStCand, midStCandlocs] = findlocalmax(d, round(mpd/2), 'min');

if vis
    figure;
    plot(d);
    hold on;
    plot(midStCandlocs, d(midStCandlocs), 'or')
    legend('distance l&r','min distance');

    grid on;
    title('Distance between left and right heels');

    xlabel('Time (s)');
    ylabel('Distance (mm)');
end

mean_d=mean(d); %mean stepwidth

Lset = [];

n = min(length(HSleftlocs), length(TOleftlocs));
for k = 1:n

    if HSleftlocs(k) > TOleftlocs(k) && length(HSleftlocs)<length(TOleftlocs)
        Lset=union(Lset, HSleftlocs(k):TOleftlocs(k+1));

    elseif HSleftlocs(k) > TOleftlocs(k) && length(HSleftlocs)>= length(TOleftlocs)

        if k<length(TOleftlocs)
            Lset=union(Lset, HSleftlocs(k):TOleftlocs(k+1));
        end

    elseif TOleftlocs(k) > HSleftlocs(k)

        Lset = union(Lset, HSleftlocs(k):TOleftlocs(k));
    end
end

n = min(length(HSrightlocs), length(TOrightlocs));
Rset = [];
for k = 1:n
    if HSrightlocs(k)>TOrightlocs(k)&& length(HSrightlocs)<length(TOrightlocs)
        Rset = union(Rset, HSrightlocs(k):TOrightlocs(k+1));
    elseif HSrightlocs(k)>TOrightlocs(k)&& length(HSrightlocs)>= length(TOrightlocs)
        if k<length(TOrightlocs)
            Rset = union(Rset, HSrightlocs(k):TOrightlocs(k+1));
        end
    else
        Rset = union(Rset, HSrightlocs(k):TOrightlocs(k));
    end
end

LmidStlocs = midStCandlocs(ismember(midStCandlocs, Lset));
RmidStlocs = midStCandlocs(ismember(midStCandlocs, Rset));

return;
end

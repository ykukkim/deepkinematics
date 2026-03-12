function [GaitEvents] = EventDetection_legs_acc_DL(data,IMU,traj)

% [SamplingRate,error] = getSamplingRate(IMU);
% if ~isempty(error); display(error); return; end
% SamplingRate = IMU.SF;
SamplingRate = 200;

% abs_acc = vecnorm(data, 2, 2);
abs_acc = data(:,1) + data(:,2) + data(:,3);
%% Filtering
freqtofilter = 10;

[b_lowpass, a_lowpass] = butter(2, freqtofilter/(SamplingRate/2), 'low');
lowfilt_data = filtfilt(b_lowpass, a_lowpass, data);

lowfilt_abs_acc_1 = lowfilt_data(:,1) + lowfilt_data(:,2) + lowfilt_data(:,3);

lowfilt_abs_acc = filtfilt(b_lowpass, a_lowpass, abs_acc);
lowfilt_abs_acc = permute(lowfilt_abs_acc,[2 1]);

% figure;
% subplot(2,1,1)
% plot(lowfilt_data(3000:4000,3))
% subplot(2,1,2)
% plot(lowfilt_abs_acc(3000:4000))

%% Removing offset and trending.
detrended_data = lowfilt_data - mean(lowfilt_data);
detrended_data = detrend(detrended_data);
% 
[~,HSLocs_z] = findpeaks(lowfilt_data(:,3), 'MinPeakDistance', 0.9*SamplingRate);
[~,HSLocs] = findpeaks(lowfilt_abs_acc, 'MinPeakDistance',0.9*SamplingRate);

% GaitEvents.HSLocs_x = HSLocs_x;
% GaitEvents.TOLocs_x = TOLocs_x;
% GaitEvents.HSLocs_y = HSLocs_y;
GaitEvents.HSLocs = HSLocs;
% GaitEvents.TOLocs_y = TOLocs_y;
GaitEvents.HSLocs_z = HSLocs_z;
% GaitEvents.TOLocs_z = TOLocs_z;

% figure
% plot(traj(:,2))
% hold on
% GaitEvents.HSLocs_z = round(GaitEvents.HSLocs_z/4);
% plot(GaitEvents.HSLocs_z, traj(GaitEvents.HSLocs_z,2), 'or')

%% precise search for the HS
Data = lowfilt_data;
bound = 20; %/(200/IMU.SF);      % Upper or lower limit for area search
for i = 1:length(GaitEvents.HSLocs) % for every detected HS
    help = GaitEvents.HSLocs_z(i);

%     if i == 1 && help - bound < 1
%         % Check if area of first HS has values below zero
%         area = (1):(help + bound);
    if i == length(GaitEvents.HSLocs_z) && GaitEvents.HSLocs_z(i) > length(Data)+bound
        % Check if area of last HS has bigger values than length of Data
        area = help:(length(Data));
    else
        area = help:help+bound;
    end
    
    hs = Data(area,3); % column 3 --> Z- Axis (Vertical Axis)
    hs_n = find(Data(area,3) == min(hs)); %searching for a lower value than the cwt- calculated result

    if hs_n > 1
        GaitEvents.HSLocs_z(i) = GaitEvents.HSLocs_z(i) + hs_n-1;

%     elseif hs_n < bound+1
% 
%         GaitEvents.HSLocs_z(i) = GaitEvents.HSLocs_z(i) - abs((bound+1 - hs_n));
    end
end

figure
plot(lowfilt_data(:,3))
hold on
plot(GaitEvents.HSLocs_z, lowfilt_data(GaitEvents.HSLocs_z,3), 'or')

HSLocs_z = round(GaitEvents.HSLocs_z/4);
HSLocs = round(GaitEvents.HSLocs/4);

GaitEvents.HSLocs_z = round(GaitEvents.HSLocs_z/4);
GaitEvents.HSLocs = round(GaitEvents.HSLocs/4);
% figure;
% plot(traj(:,3))
% hold on
% plot(HSLocs_z, traj(HSLocs_z,3), 'or')

end



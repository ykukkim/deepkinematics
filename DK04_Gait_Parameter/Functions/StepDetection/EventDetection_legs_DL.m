function [GaitEvents] = EventDetection_legs_DL(gyro, acc, IMU) % IMU.leg.synced.trim, IMU_sync

SamplingRate = 50;

%% %% Removing offset, trending, Filtering
gyro = gyro - mean(gyro);
[b_lowpass, a_lowpass] = butter(4, 10/(SamplingRate/2), 'low');
lowfilt_data = filtfilt(b_lowpass, a_lowpass, gyro(:,2));
lowfilt_data_acc = filtfilt(b_lowpass, a_lowpass, acc(:,3));

%% Wavelet scale
% Frequency analysis of the signal
% domfreq = domifreq(lowfilt_data, SamplingRate); % stride frequency
detrended_data = detrend(lowfilt_data(1250:end,:));
[~,~,funfreq_step] = thd(detrended_data,SamplingRate,20);
wavelet_scale = (centfrq('gaus1').*SamplingRate)./(funfreq_step(2));
wavelet_scale = round(wavelet_scale/2);

%% Integration
% The high peaks in the signal are mid swings
% Identifying the starting point as the first mid swing (first template)
% Identifying their window of occurence : within 30% of peak on both sides
% Cumulative integration of the detrended data
integrated_data = cumtrapz(detrended_data);

% Calculating the derivative using continuous wavelet transform
derivative_data = derivative_cwt(integrated_data, 'gaus1', wavelet_scale, 1/SamplingRate, 0);

% Determine the threshold window for mid-swing detection
Maxvalue = max(derivative_data(5000:10000)) - mean(abs(derivative_data(5000:10000)));
Minvalue = min(derivative_data(5000:10000)) - mean(abs(derivative_data(5000:10000)));
Window_midswing = [Maxvalue - (0.4 * Maxvalue), Maxvalue + (0.4 * Maxvalue)];
Window_midswing_TO = [Minvalue - (0.4 * Minvalue), Minvalue + (0.4 * Minvalue)];

% Finding midswings using the determined threshold
[~, peakLocs] = findpeaks(derivative_data, 'MinPeakHeight', Window_midswing(1), 'MinPeakDistance', 0.5 * SamplingRate);


try
    %% Identifying Gait Events Between Mid Swings
    for hh = 1:(length(peakLocs) - 1)

        % Isolate data between consecutive mid swings
        clippeddata = derivative_data(peakLocs(hh):peakLocs(hh + 1));
        clippeddata_acc = lowfilt_data_acc(peakLocs(hh):peakLocs(hh + 1));

        % Detecting gait events (HS, TO) in the clipped data
        [EventVals, EventLocs_HS] = findpeaks(-clippeddata, 'MinPeakDistance', 0.5 * SamplingRate);
        [~, EventLocs_TO] = findpeaks(-clippeddata,'MinPeakHeight', -Window_midswing_TO(1)*0.5,'MinPeakDistance', 0.5 * SamplingRate);
        [~, EventLocs_accHS] = findpeaks(clippeddata_acc, 'MinPeakDistance', 0.5 * SamplingRate);

        % Identifying Heel Strike (HS)
        if EventLocs_HS(1) > EventLocs_accHS(1)*6
            GaitEvents.HSLocs(hh) = EventLocs_accHS(1) + peakLocs(hh);
            heel_temp(hh) = EventLocs_accHS(1);
        else
            GaitEvents.HSLocs(hh) =  EventLocs_HS(1) + peakLocs(hh);
            heel_temp(hh) = EventLocs_accHS(1);
        end

        % Handle multiple potential TO events`
        if isempty(EventLocs_TO)
            if length(EventLocs_HS) >= 2
                max_loc = find(EventVals == max(EventVals(:)));
                if max_loc == length(EventLocs_HS)
                    GaitEvents.TOLocs(hh) = EventLocs_HS(end) + peakLocs(hh);
                else
                    GaitEvents.TOLocs(hh) = EventLocs_HS(max_loc) + peakLocs(hh);
                end
                toe_temp(hh) = GaitEvents.TOLocs(hh) - peakLocs(hh);

            elseif length(EventLocs_HS) == 1 || EventLocs_HS(1) > mean(heel_temp) * 1.5
                GaitEvents.HSLocs(hh) = floor(mean(heel_temp)) + peakLocs(hh);
                GaitEvents.TOLocs(hh) = floor(mean(toe_temp)) + GaitEvents.HSLocs(hh);
                toe_temp(hh) = GaitEvents.TOLocs(hh) - GaitEvents.HSLocs(hh);
            end

        elseif ~isempty(EventLocs_TO)
            % IdentifyingToe Off (TO) events
            GaitEvents.TOLocs(hh) = max(EventLocs_TO) + peakLocs(hh);
            toe_temp(hh) = max(EventLocs_TO);
        end

        % Detecting Foot Flat (FF) events
        if toe_temp(hh)> heel_temp(hh)
            [~, FootFlatLocs] = findpeaks(clippeddata(heel_temp(hh):toe_temp(hh)));
        else
            [~, FootFlatLocs] = findpeaks(clippeddata(heel_temp(hh):end));
        end

        if ~isempty(FootFlatLocs)
            GaitEvents.FFLocs(hh) =  FootFlatLocs(1) +  GaitEvents.HSLocs(hh);
        else
            start_index = max(1, floor(mean(heel_temp)));  % Ensure start index is at least 1
            end_index = min(length(clippeddata), floor(median(heel_temp) + median(toe_temp)));  % Ensure end index doesn't exceed array length
            indx_FF = find(clippeddata(start_index:end_index) <= 0);
            GaitEvents.FFLocs(hh) = GaitEvents.HSLocs(hh) + floor(median(indx_FF));
        end
        % % Recording Mid Swing (MS) locations
        % GaitEvents.MSLocs(hh) = peakLocs(hh);
    end

    %% precise search for the HS
    for i = 1:length(GaitEvents.HSLocs)
        initial_temp = GaitEvents.HSLocs(i);
        area = (initial_temp-4):(initial_temp+4);
        area(area<=0) = [];
        hs = derivative_data(area); % Y- Axis (Sagittal Plane)
        hs_n = find(derivative_data(area) == min(hs)); %searching for a lower value than the cwt- calculated result

        if hs_n > 5
            GaitEvents.HSLocs(i) = GaitEvents.HSLocs(i) + abs((5 - hs_n));
            GaitEvents.FFLocs(i) = GaitEvents.FFLocs(i) + abs((21 - hs_n));

        elseif hs_n < 5
            GaitEvents.HSLocs(i) = GaitEvents.HSLocs(i) - abs((5 - hs_n));
            GaitEvents.FFLocs(i) = GaitEvents.FFLocs(i) - abs((5 - hs_n));

        end
    end

    %% precise search for the TO
    for i = 1:length(GaitEvents.TOLocs)
        initial_temp = GaitEvents.TOLocs(i);
        area = (initial_temp-4):(initial_temp+20);

        % Ensure 'area' does not include negative indices
        area(area < 1) = [];

        % Adjust 'area' to not exceed the length of 'derivative_data'
        if max(area) > length(derivative_data)
            area = area(area <= length(derivative_data));
        end

        to = derivative_data(area);
        to_n = find(derivative_data(area) == min(to), 1, 'first'); % Ensuring a single index is returned

        if to_n > 5
            GaitEvents.TOLocs(i) = GaitEvents.TOLocs(i) + (5 - to_n);
        elseif to_n < 5
            GaitEvents.TOLocs(i) = GaitEvents.TOLocs(i) - (5 - to_n);
        end
    end

    [New_HS,New_FF,New_TO] = Sorting_HSFFTO(GaitEvents.HSLocs,GaitEvents.FFLocs,GaitEvents.TOLocs);

    GaitEvents.TOLocs = New_TO;
    GaitEvents.HSLocs = New_HS;
    GaitEvents.FFLocs = New_FF;


    %     GaitEvents.TOLocs = floor(New_TO/(SamplingRate/IMU.SF));
    %     GaitEvents.HSLocs = floor(New_HS/(SamplingRate/IMU.SF));
    %     GaitEvents.FFLocs = floor(New_FF/(SamplingRate/IMU.SF));
    %     figure
    %     plot(derivative_data);
    %     hold on
    %     plot(GaitEvents.TOLocs, derivative_data(GaitEvents.TOLocs), 'or')
    %     plot(GaitEvents.HSLocs, derivative_data(GaitEvents.HSLocs), '*g')

catch ME
    fprintf('Cannot find event at index %d of Trial %d\n',hh);
end






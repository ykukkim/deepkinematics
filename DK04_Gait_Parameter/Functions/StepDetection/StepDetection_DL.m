function [GaitEvents, error] = StepDetection_DL(interpolatedVD, VD, Name, Test)
    error = false; % if for any reason the trial was invalid, a true value is assigned to this variable
    GaitEvents = struct;

    %% Input parameters for the algorithm
    toprdfac = 0.8;  % Factor for discarding TO more than 1 step per 60 - 70 frames
    hsprdfac = 0.08; % Factor for discarding HS
    zrangefac = 0.4; % Heel marker z coordinate by the HS event should be in the range [min(z), min(z) + zrangefac*(max(z) - min(z))]
    filterOrder = 4; % seems to provide the best results
    cutfreqFilter = 25; % Not much happening in walking task above 25 Hz
    smpf = 200 / VD.SF; % Sampling Factor

    %% Set start and end frames based on test type and data length
    [sf_gyro, ef_gyro, sf_fva, ef_fva] = setFrameIndices(VD, Test, smpf);

    %% Detect gait events using gyroscope data
    GaitEventsLeft = EventDetection_legs_DL(VD.IMU.ankle_L.gyro(sf_gyro:ef_gyro,:), VD.IMU.ankle_L.acc(sf_gyro:ef_gyro,:), VD);
    GaitEventsRight = EventDetection_legs_DL(VD.IMU.ankle_R.gyro(sf_gyro:ef_gyro,:), VD.IMU.ankle_R.acc(sf_gyro:ef_gyro,:), VD);

    %% Adjust for the HS/TO detection offset if necessary
    if sf_gyro ~= 1
        HSleftlocs = GaitEventsLeft.HSLocs + sf_fva;
        HSrightlocs = GaitEventsRight.HSLocs + sf_fva;
        TOleftlocs = GaitEventsLeft.TOLocs + sf_fva;
        TOrightlocs = GaitEventsRight.TOLocs + sf_fva;
    else
        HSleftlocs = GaitEventsLeft.HSLocs + 1250;
        HSrightlocs = GaitEventsRight.HSLocs + 1250;
        TOleftlocs = GaitEventsLeft.TOLocs + 1250;
        TOrightlocs = GaitEventsRight.TOLocs + 1250;
    end

    %% Retrieve and sort gait events
    [Sorted_HSleftlocs, Sorted_TOrightlocs, Sorted_HSrightlocs, Sorted_TOleftlocs] = Sorting_Steps(HSleftlocs, TOrightlocs, HSrightlocs, TOleftlocs);

    %     figure; subplot(2,1,1)
    %     plot(VD.IMU.ankle_L.gyro(:,2));
    %     hold on
    %     plot(Sorted_HSleftlocs, VD.IMU.ankle_L.gyro(Sorted_HSleftlocs,2), 'or');
    %     plot(Sorted_TOleftlocs, VD.IMU.ankle_L.gyro(Sorted_TOleftlocs,2), '*g');
    %     subplot(2,1,2)
    %     plot(VD.IMU.ankle_R.gyro(:,2));
    %     hold on
    %     plot(Sorted_HSrightlocs, VD.IMU.ankle_R.gyro(Sorted_HSrightlocs,2), 'or');
    %     plot(Sorted_TOrightlocs, VD.IMU.ankle_R.gyro(Sorted_TOrightlocs,2), '*g');
    %
    %     figure; subplot(2,1,1)
    %     plot(VD.IMU.ankle_L.gyro(:,2));
    %     hold on
    %     plot(HSleftlocs, VD.IMU.ankle_L.gyro(HSleftlocs,2), 'or');
    %     plot(TOleftlocs, VD.IMU.ankle_L.gyro(TOleftlocs,2), '*g');
    %     subplot(2,1,2)
    %     plot(VD.IMU.ankle_R.gyro(:,2));
    %     hold on
    %     plot(HSrightlocs, VD.IMU.ankle_R.gyro(HSrightlocs,2), 'or');
    %     plot(TOrightlocs, VD.IMU.ankle_R.gyro(TOrightlocs,2), '*g');

    %% Populate GaitEvents structure
    GaitEvents.HSleftlocs = Sorted_HSleftlocs;
    GaitEvents.HSrightlocs = Sorted_HSrightlocs;
    GaitEvents.TOleftlocs = Sorted_TOleftlocs;
    GaitEvents.TOrightlocs = Sorted_TOrightlocs;

    %% Number of gait events
    GaitEvents.nHSleft = length(Sorted_HSleftlocs);
    GaitEvents.nHSright = length(Sorted_HSrightlocs);
    GaitEvents.nTOleft = length(Sorted_TOleftlocs);
    GaitEvents.nTOright = length(Sorted_TOrightlocs);
end


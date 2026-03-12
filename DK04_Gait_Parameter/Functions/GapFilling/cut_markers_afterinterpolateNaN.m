%% Cut all data, that no NAN exists in heel and toe marker

function [interpolatedfilledVD, cNaNmarkerTrials, cNaNmarkerTrialsnames] = ...
    cut_markers_afterinterpolateNaN(interpolatedVD, participant, filename, ...
    cNaNmarkerTrials, cNaNmarkerTrialsnames)



matrix = [];
% make a matrix of the important markers, include NAN values
markerlist = generateMarkerList(interpolatedVD);

for i = 1:length(markerlist)
    if isfield(interpolatedVD, (markerlist{i}))
        for dirtn = 1:3
            if dirtn == 1
                timeSeries = interpolatedVD.(markerlist{i}).Values.x_coord;
            elseif dirtn == 2
                timeSeries = interpolatedVD.(markerlist{i}).Values.y_coord;
            elseif dirtn == 3
                timeSeries = interpolatedVD.(markerlist{i}).Values.z_coord;
            end
            
            matrix=[matrix timeSeries];
        end
    end
end

%% Extract data. Only not NaN values are of interest
% Data relevant for following computations is the data of the largest
% time intervall of the data without NaN values for the relevant
% markers

dataRel=not(isnan(matrix));
assert(not(isempty(markerlist)));
%     eval(['d = size(' markersRelListFull{1} ');']);
%     m = length(markersRelListFull);
%     dataRel = nan(d .* [1 m]);
%     for jj = 1:m
%         eval(['dataRel(:, (jj-1)*d(2)+1:jj*d(2)) = ' ...
%             markersRelListFull{jj} ';']);
%     end
% %     dataRel = [LHEEL RHEEL LTOE RTOE];
%


dataRel1=(not(isnan(matrix)));
start=[];
stop=[];
%     cNaNmarkerTrials={};
%     cNaNmarkerTrialsnames   = {};

for k=1:size(matrix, 2)
    dataRel=dataRel1(:,k);
    % If in every frame some of the four markers (2x heel + 2x toe) is nan
    if all(not(dataRel))
        % Set all the markers to NaN
        %         for jj = 1:length(markersListFull)
        %             eval([markersListFull{jj} ...
        %                 ' = nan(size(' markersListFull{jj} '));']);
        %         end
        
        cNaNmarkerTrials = cNaNmarkerTrials + 1;
        cNaNmarkerTrialsnames(cNaNmarkerTrials, 1:2) = ...
            {participant,  filename}; 
        continue;
    else % Find the largest intervall where all the relevant markers
        % are valid in every frame
        
        fstindx = find(dataRel, 1, 'first');
        lstindx = find(dataRel, 1, 'last');
        
        indxofnans = find(not(dataRel));
        indxofnans = indxofnans(fstindx <= indxofnans ...
            & indxofnans <= lstindx);
        
        indxofnans = [fstindx-1; indxofnans; lstindx+1];
        l = indxofnans(2:end) - indxofnans(1:end-1);
        
        % Returns the largest intervall without gaps and its starting
        % position index
        [tmp, indx] = max(l);
        startcut = indxofnans(indx)     + 1;
        finshcut = indxofnans(indx+1)   - 1;
        
        if finshcut - startcut < 100
            cNaNmarkerTrials = cNaNmarkerTrials + 1;
            cNaNmarkerTrialsnames(cNaNmarkerTrials, 1:2) = ...
                {participant, filename};
            continue;
        end
        
        start=[start; startcut];
        stop=[stop;finshcut];
        % Cut all the data from to the intervall [startcut, finshcut]
        %cutVars = union(markersListFull,{'frames', 'data.trajectories.samples'});
        %         for jj = 1:length(cutVars)
        %             eval([cutVars{jj} ' = ' cutVars{jj} '(startcut:finshcut, :);']);
        %         end
    end
end

startcut=max(start);
finshcut=min(stop);

% Cut all the data from to the intervall [startcut, finshcut]
cutVars=fieldnames(interpolatedVD);

% Parameters
SamplingRate = 50;     % Hz
CutOffFreq = 6;        % Hz (adjust as needed for gait data)
FilterOrder = 4;       % 4th-order Butterworth

% Normalised cut-off frequency
Wn = CutOffFreq / (SamplingRate / 2);
[b, a] = butter(FilterOrder, Wn, 'low');

% Loop through markers and filter each coordinate
for j = 1:length(cutVars) % only cut for the markers
    if isfield(interpolatedVD, markerlist{j})
        marker = char(cutVars(j));

        % Cut the data
        x = double(interpolatedVD.(marker).Values.x_coord(startcut:finshcut));
        y = double(interpolatedVD.(marker).Values.y_coord(startcut:finshcut));
        z = double(interpolatedVD.(marker).Values.z_coord(startcut:finshcut));

        % Remove mean (to eliminate DC bias)
        x = x - mean(x, 'omitnan');
        y = y - mean(y, 'omitnan');
        z = z - mean(z, 'omitnan');

%         % Remove mean (to eliminate DC bias)
%         x = detrend(x);
%         y = detrend(y);
%         z = detrend(z);

        % Apply zero-phase Butterworth filtering
        x_filt = filtfilt(b, a, x);
        y_filt = filtfilt(b, a, y);
        z_filt = filtfilt(b, a, z);

        % Store filtered data
        interpolatedfilledVD.(marker).Values.x_coord = x_filt;
        interpolatedfilledVD.(marker).Values.y_coord = y_filt;
        interpolatedfilledVD.(marker).Values.z_coord = z_filt;
    end
end

return
end



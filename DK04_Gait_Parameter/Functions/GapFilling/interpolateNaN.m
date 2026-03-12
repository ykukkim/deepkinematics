function [interpolatedVD, fileidtoignore, fileswithmissingMarkers, nCriticalMarkers] = ...
    interpolateNaN(VD, fileName,Name)

markerlist = generateMarkerList(VD.Marker); % look for available markers

fileidtoignore = [];
fileswithmissingMarkers = [];
nCriticalMarkers = length(markerlist);
for i = 1:nCriticalMarkers
    if isfield(VD.Marker, (markerlist{i}))
        interpolatedVD.(markerlist{i}).Values.x_coord = [];
        interpolatedVD.(markerlist{i}).Values.y_coord = [];
        interpolatedVD.(markerlist{i}).Values.z_coord = [];
        for dirtn = 1:3
            if dirtn == 1
                timeSeriesx = VD.Marker.(markerlist{i})(:,1);
            elseif dirtn == 2
                timeSeriesx = VD.Marker.(markerlist{i})(:,2);
            elseif dirtn == 3
                timeSeriesx = VD.Marker.(markerlist{i})(:,3);
            end
            
            lengthofSeries = length(timeSeriesx);
            oddorEven = rem(lengthofSeries, 2);
            if oddorEven ~=0
                splitHalf = (lengthofSeries + 1)/2;
            else
                splitHalf = lengthofSeries/2;
            end
            
            timeseriessplit_first = timeSeriesx(1:splitHalf);
            timeseriessplit_last = timeSeriesx((splitHalf + 1):end);
            timeseriessplit_last_01 = flipud(timeseriessplit_last);  % flip the last
            %half of the array
            %             disp(i);
            %             disp(dirtn);
            [startidx, timeseries_firstHalf] = donotExtrapolate(timeseriessplit_first);
            
            [endidx_flp, timeseries_secondHalf] = donotExtrapolate(timeseriessplit_last_01);
            
            endidx = lengthofSeries - endidx_flp + 1;
            
            timeseries_secondHalf_01 = flipud(timeseries_secondHalf); % flip the last
            % half back
            
            timeSeriesx_01 = [timeseries_firstHalf; timeseries_secondHalf_01];
            
            nanxnew = isnan(timeSeriesx_01);
            nonNanloc = find(~(nanxnew > 0));
            
            bigGapSizeT = 0.2;
            bigGapSize = bigGapSizeT * VD.SF;
            
            nooflargeholes = length(find((diff([0; nonNanloc; numel(nanxnew) + 1]) - 1)>bigGapSize));
            % maybe i dont need to run it over and over again... try leaving all the
            % large gaps at once!!!!
            if nooflargeholes > 0
                fileidtoignore = fileName;
                interpolatedVD.(genvarname(markerlist{i})).Values.x_coord = VD.Marker.(genvarname(markerlist{i})).Values.x_coord;
                interpolatedVD.(genvarname(markerlist{i})).Values.y_coord = VD.Marker.(genvarname(markerlist{i})).Values.y_coord;
                interpolatedVD.(genvarname(markerlist{i})).Values.z_coord = VD.Marker.(genvarname(markerlist{i})).Values.z_coord;

                return;
                
                %             elseif noofsmallholes > 0
                %                     noofstartpoints = length(nanxnew)*ones(noofsmallholes+1,1);
                %                     noofendpoints = zeros(noofsmallholes+1, 1);
                %                     nanxnewMult = [];
                %                     for nooftimestofill=1:noofsmallholes
                %                         eval(['startpointbigNan_', num2str(nooftimestofill), '= nonNanloc(noofsmallholes(', num2str(nooftimestofill), ')-1);']);
                %                         eval(['endpointbigNan_', num2str(nooftimestofill), '= nonNanloc(noofsmallholes(', num2str(nooftimestofill), '));']);
                %                         eval(['noofstartpoints(', num2str(nooftimestofill),') = startpointbigNan_', num2str(nooftimestofill),';']);
                %                         eval(['noofendpoints(', num2str(nooftimestofill),'+1) = endpointbigNan_', num2str(nooftimestofill),';']);
                %                         smallNans = (noofstartpoints(nooftimestofill)+1:noofendpoints(nooftimestofill+1)-1);
                %                         nanRem = logical(zeros(length(smallNans),1));
                %                         nanxnewMult = [nanxnewMult; nanxnew((noofendpoints(nooftimestofill))+1:noofstartpoints(nooftimestofill)); nanRem];
                %                         clear nanRem largeNans
                %                     end
                %                     nanxnewMult = [nanxnewMult; nanxnew(noofendpoints(end):noofstartpoints(end))];
                %                     nanxnewMult = logical(nanxnewMult);
                %                     t = 1:numel(timeSeriesx_01);
                %                     timeSeriesx_01(nanxnewMult) = interp1(t(~nanxnewMult), timeSeriesx_01(~nanxnewMult), t(nanxnewMult));
                %                     timeSeriesx_fnal = [timeSeriesx(1:startidx);timeSeriesx_01; timeSeriesx(endidx:end)];
                %                 end
                
            else
                t = 1:numel(timeSeriesx_01);
                timeSeriesx_01(nanxnew) = interp1(t(~nanxnew), timeSeriesx_01(~nanxnew), t(nanxnew));
                
                if (startidx == 1 && endidx == lengthofSeries)
                    timeSeriesx_fnal = timeSeriesx_01;
                elseif startidx == 1
                    timeSeriesx_fnal = [timeSeriesx_01; timeSeriesx(endidx+1:end)];
                elseif endidx == lengthofSeries
                    timeSeriesx_fnal = [timeSeriesx(1:startidx-1);timeSeriesx_01];
                else
                    timeSeriesx_fnal = [timeSeriesx(1:startidx-1);timeSeriesx_01;timeSeriesx(endidx+1:end)];
                end
            end
            if ~isempty(fileidtoignore)
                fileidtoignore = fileName;
                return;
            else
                if dirtn == 1
                    interpolatedVD.(genvarname(markerlist{i})).Values.x_coord = timeSeriesx_fnal;
                elseif dirtn == 2
                    interpolatedVD.(genvarname(markerlist{i})).Values.y_coord = timeSeriesx_fnal;
                elseif dirtn == 3
                    interpolatedVD.(genvarname(markerlist{i})).Values.z_coord = timeSeriesx_fnal;
                end
            end
        end
    else
        fileswithmissingMarkers = fileName;
    end
end

return;



function [nanidx, timeseries] = donotExtrapolate(timeSeriesx)

% donotExtrapolate function removes the non-existing (NaN values)
% trajectory at the beginning. The assumption is to remove the NaN values
% from the trajectory up to 100 samples or 1 sec of data.

nanx = isnan(timeSeriesx);
summ_first10 = sum(isnan(timeSeriesx(1:10)));     % grab first 10% samples

if ~isempty(find(nanx, 1))             %Check if the first element is NaN
    
    % check if there are no Nan values
    if ((find(nanx, 1) == 1) || (summ_first10 > 6))
        % NaN values in more
        % than 5% of sampling
        % rate
        firstnonzeroEle = find(nanx, 100);            % 100% sampling rate
        firstnonzeroEleDiff = diff(firstnonzeroEle);
        startidxfinder = find(firstnonzeroEleDiff > 5);
        if (length(firstnonzeroEleDiff) == 149 && isempty(startidxfinder)) % >1.5 SR
            % 1 less than sampling R
            error('Error. \nThere is more than One sec of Nan values in the beginning. \nPlease check!');
        else
            if isempty(startidxfinder)
                nanidx = firstnonzeroEle(end)+1;
            elseif firstnonzeroEle(startidxfinder(1)) == 1
                nanidx = firstnonzeroEle(end)+1;
            else
                nanidx = firstnonzeroEle(startidxfinder(1)-1)+1;
            end
        end
    else
        nanidx = 1;
    end
else
    nanidx = 1;
end
timeseries = timeSeriesx(nanidx:end);
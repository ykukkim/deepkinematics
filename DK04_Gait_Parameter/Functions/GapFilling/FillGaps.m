%% interpolation function for small gaps of upto 20% sampling rate.
% If the file still has NaN values, the file will be skipped.

function [interpolatedVD, cNaNmarkerTrials,cNaNmarkerTrialsnames,error] = FillGaps(VD, filename, Name)
%% Write down names of the skipped files
cNaNmarkerTrialsnames   = {};
cNaNmarkerTrials      = 0;
error=false;

[interpolatedVD, fileidlargeGap, fileidtoignore, nCriticalMarkers] = interpolateNaN(VD, filename, Name);
if ~(isempty(fileidlargeGap))
    warning('the file in process contains large labelling gaps if these gaps are critical, the file will be removed from analysis');
end
if ~(isempty(fileidtoignore)) || (size(fieldnames(interpolatedVD), 1) < nCriticalMarkers)
    return
else
    
    participant = Name;
    
    %% make sure L/RTHL and L/RTFb don't contain NAN values, cut all NaN values off
    [interpolatedVD, cNaNmarkerTrials, cNaNmarkerTrialsnames] = ...
        cut_markers_afterinterpolateNaN(interpolatedVD, participant, filename,cNaNmarkerTrials, cNaNmarkerTrialsnames);
    interpolatedVD.SF = VD.SF;
    
    %% skip file if NAN exists
    if max(max(strcmp(filename,cNaNmarkerTrialsnames)))
        error=true;
        return
    end
end


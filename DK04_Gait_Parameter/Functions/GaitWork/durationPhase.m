function [durationGaitCycleL, durationGaitCycleR, durationStancePhL, durationStancePhR,...
    durationSwingPhL, durationSwingPhR, dlsL, dlsR, ndlsL, ndlsR]=durationPhase(HSleftlocs,HSrightlocs, TOleftlocs, TOrightlocs,...
    samplingRate)
%calculates durationGaitCycle, durationStancePh, durationSwingPh and dls

%set parameters
nHSleft = length(HSleftlocs);
nTOleft = length(TOleftlocs);
nHSright = length(HSrightlocs);
nTOright = length(TOrightlocs);

%% Duration of gait cycle (s)
    durationGaitCycleL = nan(1, nHSleft-1);
    for j=1:nHSleft-1
        durationGaitCycleL(j) = (HSleftlocs(j+1) - HSleftlocs(j)) / samplingRate;
    end
    
    durationGaitCycleR = nan(1, nHSright-1);
    for j=1:nHSright-1
        durationGaitCycleR(j) = (HSrightlocs(j+1) - HSrightlocs(j)) / samplingRate;
    end
    
%% Duration of the stance phase (s)
    durationStancePhL = nan(1, min(nHSleft, nTOleft));
    for j=1:min(nHSleft, nTOleft)
        durationStancePhL(j) = (TOleftlocs(j) - HSleftlocs(j))/samplingRate;
    end
    durationStancePhR = nan(1, min(nHSright, nTOright));
    for j=1:min(nHSright, nTOright)
        durationStancePhR(j) = (TOrightlocs(j) - HSrightlocs(j))/samplingRate;
    end
    
    
%% Duration of the swing phase (s)
    durationSwingPhL = nan(1, min(nHSleft-1, nTOleft));
    for j=1:min(nHSleft-1, nTOleft)
        durationSwingPhL(j) = ...
            (HSleftlocs(j+1) - TOleftlocs(j))/samplingRate;
    end
    durationSwingPhR = nan(1, min(nHSright-1, nTOright));
    for j=1:min(nHSright-1, nTOright)
        durationSwingPhR(j) = ...
            (HSrightlocs(j+1) - TOrightlocs(j))/samplingRate;
    end

    
%% Duration of double limb support (s)
    dlsL = [];
    for j=1:min(nHSleft, nTOleft)
        indx = find(HSleftlocs(j) <= HSrightlocs & HSrightlocs <= TOleftlocs(j)); %#ok<EFIND>
        if length(indx) == 1
            value = (TOleftlocs(j) - HSrightlocs(indx)) / samplingRate;
            dlsL = [dlsL value]; %#ok<AGROW>
        end
    end
    if isempty(dlsL), dlsL = NaN; end
    dlsR = [];
    for j=1:min(nHSright, nTOright)
        indx = find(HSrightlocs(j) <= HSleftlocs & HSleftlocs <= TOrightlocs(j)); %#ok<EFIND>
        if length(indx) == 1
            value = (TOrightlocs(j) - HSleftlocs(indx)) / samplingRate;
            dlsR = [dlsR value]; %#ok<AGROW>
        end
    end
    if isempty(dlsR), dlsR = NaN; end
    
%% normalized double limb support (%)
    ndlsL=[];
    ndlsR=[];
    for k=1:length(dlsL)
        valL=100*(dlsL(k)/durationGaitCycleL(k));
        ndlsL=[ndlsL valL];
    end
    
    for m=1:length(dlsR) 
        valR=100*(dlsR(m)/durationGaitCycleR(m));
        ndlsR=[ndlsR valR];
    end

return
end
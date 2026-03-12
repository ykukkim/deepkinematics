function[midStOccTimeL, midStOccTimeR, midSwOccTimeL, midSwOccTimeR]= midtimes(...
    LmidStlocs, RmidStlocs, HSleftlocs, HSrightlocs, TOleftlocs, TOrightlocs,...
    LmidSwlocs, RmidSwlocs)
    
% Time of occurance of mid-stance and mid-swing events
    % (% stance/swing phase)
    n = length(LmidStlocs);
    midStOccTimeL = nan(1, n);
    for j=1:n
        lstHS = max(HSleftlocs(HSleftlocs <= LmidStlocs(j)));
        nxtTO = min(TOleftlocs(TOleftlocs >= LmidStlocs(j)));
        
        if isempty(lstHS) || isempty(nxtTO)
            midStOccTimeL(j) = NaN; continue;
        end
        midStOccTimeL(j) = 100*(LmidStlocs(j) - lstHS) / (nxtTO - lstHS);
    end
    n = length(RmidStlocs);
    midStOccTimeR = nan(1, n);
     for j=1:n
        lstHS = max(HSrightlocs(HSrightlocs <= RmidStlocs(j)));
        nxtTO = min(TOrightlocs(TOrightlocs >= RmidStlocs(j)));
        
        if isempty(lstHS) || isempty(nxtTO)
            midStOccTimeR(j) = NaN; continue;
        end
        midStOccTimeR(j) = 100*(RmidStlocs(j) - lstHS) / (nxtTO - lstHS);
    end
    n = length(LmidSwlocs);
    midSwOccTimeL = nan(1, n);
    for j=1:n
        lstTO = max(TOleftlocs(TOleftlocs <= LmidSwlocs(j)));
        nxtHS = min(HSleftlocs(HSleftlocs >= LmidSwlocs(j)));
        
        if isempty(lstTO) || isempty(nxtHS)
            midSwOccTimeL(j) = NaN; continue;
        end
        midSwOccTimeL(j) = 100*(LmidSwlocs(j) - lstTO) / (nxtHS - lstTO);
    end
    n = length(RmidSwlocs);
    midSwOccTimeR = nan(1, n);
    for j=1:n
        lstTO = max(TOrightlocs(TOrightlocs <= RmidSwlocs(j)));
        nxtHS = min(HSrightlocs(HSrightlocs >= RmidSwlocs(j)));
        
        if isempty(lstTO) || isempty(nxtHS)
            midSwOccTimeR(j) = NaN; continue;
        end
        midSwOccTimeR(j) = 100*(RmidSwlocs(j) - lstTO) / (nxtHS - lstTO);
    end
    midStOccTimeL = midStOccTimeL(not(isnan(midStOccTimeL)));
    midStOccTimeR = midStOccTimeR(not(isnan(midStOccTimeR)));
    midSwOccTimeL = midSwOccTimeL(not(isnan(midSwOccTimeL)));
    midSwOccTimeR = midSwOccTimeR(not(isnan(midSwOccTimeR)));
    if isempty(midStOccTimeL), midStOccTimeL = NaN; end
    if isempty(midStOccTimeR), midStOccTimeR = NaN; end
    if isempty(midSwOccTimeL), midSwOccTimeL = NaN; end
    if isempty(midSwOccTimeR), midSwOccTimeR = NaN; end
return
end 
function [stepLngthL, stepLngthR, cadence, hs_matrix] = ...
    steplength(heelLx, heelLy, heelRx, heelRy, SF)

LHEEL = [heelLx heelLy];
RHEEL = [heelRx heelRy];

 % step length (projected onto transverse plane)
    % (cm)
    % Step duration (ms), Cadence (steps/s)
    stepLngthL = nan(1, nHSleft + nHSright);
    stepLngthR = nan(1, nHSleft + nHSright);
    l = zeros(1, nHSleft);
    r = zeros(1, nHSright);
    l(1:end) = 'L';     % left
    r(1:end) = 'R';     % right
    hs_matrix = [HSleft HSright; HSleftlocs HSrightlocs; l r];
    hs_matrix = sortrows(hs_matrix', 2);
    
    minIndx = 1;
    start = 0;
    while ~start
        startL = ...
            hs_matrix(minIndx, 3) == 'L' && hs_matrix(minIndx+1, 3) == 'R';
        startR = ...
            hs_matrix(minIndx, 3) == 'R' && hs_matrix(minIndx+1, 3) == 'L';
        start = startL || startR;
        if start, break; end
        
        minIndx = minIndx + 1;
        
        if minIndx == size(hs_matrix, 1), break; end
    end
    if startL
        fst = 'L'; snd = 'R';
    elseif startR
        fst = 'R'; snd = 'L';
    end

    maxIndx = minIndx;
    htrepeatable = start;
    while htrepeatable
        if mod(maxIndx-minIndx+1, 2) == 1
            htrepeatable = hs_matrix(maxIndx, 3) == fst ...
                && hs_matrix(maxIndx+1, 3) == snd;
        else
            htrepeatable = hs_matrix(maxIndx, 3) == snd ...
                && hs_matrix(maxIndx+1, 3) == fst;
        end
        
        maxIndx = maxIndx + 1;
        
        if maxIndx == size(hs_matrix, 1), break; end
    end
    hs_matrix = hs_matrix(minIndx:maxIndx, :);
    
    time = (hs_matrix(end, 2) - hs_matrix(1, 2)) / SF;
    cadence = (size(hs_matrix, 1) - 1) / time;
    
    j = 1;
    while j < size(hs_matrix, 1)
        if hs_matrix(j, 3) == 'L'
            stepLngthR(j) = 1/10*norm(RHEEL(hs_matrix(j+1, 2), 1:2) ...
                - LHEEL(hs_matrix(j, 2), 1:2));
        elseif hs_matrix(j, 3) == 'R'
            stepLngthL(j) = 1/10*norm(LHEEL(hs_matrix(j+1, 2), 1:2) ...
                - RHEEL(hs_matrix(j, 2), 1:2));
        end
        
        j = j + 1;
    end
    stepLngthL = stepLngthL(not(isnan(stepLngthL)));
    stepLngthR = stepLngthR(not(isnan(stepLngthR)));
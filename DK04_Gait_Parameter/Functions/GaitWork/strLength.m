function [strideLengthL, strideLengthR, stepLengthL, stepLengthR, hs_matrix, ...
    stepTimeL, stepTimeR, cadence] = ...
    strLength(x_LHEE,y_LHEE, x_RHEE, y_RHEE,...
    HSleft,HSright,...
    HSleftlocs,HSrightlocs,...
    samplingRate)

%% finds stride length

nHSleft = length(HSleft);
nHSright = length(HSright);

%% xy coordinates from heel markers
xy_LHEE=[x_LHEE , y_LHEE];
xy_RHEE=[x_RHEE , y_RHEE];

%% (projected onto transverse plane) (cm)
strideLengthL = nan(1, nHSleft-1);
for j=1:nHSleft-1
    strideLengthL(j) = 1/10 * norm(xy_LHEE(HSleftlocs(j+1),1:2) - xy_LHEE(HSleftlocs(j), 1:2));
end

strideLengthR = nan(1, nHSright-1);
for j=1:nHSright-1
    strideLengthR(j) = 1/10 * norm(xy_RHEE(HSrightlocs(j+1), 1:2) - xy_RHEE(HSrightlocs(j), 1:2));
end

%% step length (projected onto transverse plane)
% (cm)
% Step duration (ms), Cadence (steps/s)
stepLengthL = nan(1, nHSleft + nHSright);
stepLengthR = nan(1, nHSleft + nHSright);
stepTimeL = nan(1, nHSleft + nHSright);
stepTimeR = nan(1, nHSleft + nHSright);

l = zeros(1, nHSleft);
r = zeros(1, nHSright);
l(1:end) = 'L';     % left
r(1:end) = 'R';     % right
hs_matrix = [HSleft HSright; HSleftlocs HSrightlocs; l r];
hs_matrix = sortrows(hs_matrix', 2);
HS_HH_TT = find((diff(hs_matrix(:,3))==0)==1);
HS_HH_TT_size = length(HS_HH_TT);

for a = HS_HH_TT_size(1,1):-1:1
    hs_matrix(HS_HH_TT(a,:),:) = [];
end

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

steptime = (hs_matrix(end, 2) - hs_matrix(1, 2)) / samplingRate;
cadence = 60*(size(hs_matrix, 1) - 1) / steptime;

j = 1;
while j < size(hs_matrix, 1)
    if hs_matrix(j, 3) == 'L'
        stepLengthR(j) = 1/10*norm(xy_RHEE(hs_matrix(j+1, 2), 1:2) ...
            - xy_LHEE(hs_matrix(j, 2), 1:2));
        stepTimeR(j) = (hs_matrix(j+1, 2) - hs_matrix(j, 2))/samplingRate;
    elseif hs_matrix(j, 3) == 'R'
        stepLengthL(j) = 1/10*norm(xy_LHEE(hs_matrix(j+1, 2), 1:2) ...
            - xy_RHEE(hs_matrix(j, 2), 1:2));
        stepTimeL(j) = (hs_matrix(j+1, 2) - hs_matrix(j, 2))/samplingRate;
    end
    
    j = j + 1;
end
stepLengthL = stepLengthL(not(isnan(stepLengthL)));
stepLengthR = stepLengthR(not(isnan(stepLengthR)));
stepTimeL = stepTimeL(not(isnan(stepTimeL)));
stepTimeR = stepTimeR(not(isnan(stepTimeR)));

return;
end
function [wbL, wbR] = stepwidth(x_LHEE, y_LHEE, x_RHEE, y_RHEE, HSleftlocs, HSrightlocs)

nHSleft = length(HSleftlocs);
nHSright = length(HSrightlocs);

% Center the y-coordinates around their mean
y_LHEE = y_LHEE - mean(y_LHEE);
y_RHEE = y_RHEE - mean(y_RHEE);

% Combine x and y coordinates into one array for each heel
xy_LHEE = [x_LHEE, y_LHEE];
xy_RHEE = [x_RHEE, y_RHEE];

% Initialize counters for left and right walking base
cl = 1;
cr = 1;

% Initialize walking base arrays
wbL = NaN(1, nHSleft - 1);
wbR = NaN(1, nHSright - 1);

% Calculate left walking base
for j = 1:min(nHSleft, nHSright) - 1
    if HSleftlocs(j) < HSrightlocs(j)
        % Vector between two consecutive left heel strikes
        vec = xy_LHEE(HSleftlocs(j + 1), :) - xy_LHEE(HSleftlocs(j), :);
        vec = vec / norm(vec); % Normalize to get unit vector

        % Vector connecting current left heel strike to right heel strike
        w = xy_LHEE(HSleftlocs(j), :) - xy_RHEE(HSrightlocs(j), :);

        % Cross product to find the perpendicular distance
        d_vec = cross([vec, 0], [w, 0]);
        d = norm(d_vec); % Magnitude of the perpendicular vector

        % Store the calculated distance
        wbL(cl) = d;
        cl = cl + 1;
    end
end

% Calculate right walking base
for j = 1:min(nHSleft, nHSright) - 1
    if HSrightlocs(j) < HSleftlocs(j + 1)
        % Vector between two consecutive right heel strikes
        vec = xy_RHEE(HSrightlocs(j + 1), :) - xy_RHEE(HSrightlocs(j), :);
        vec = vec / norm(vec); % Normalize to get unit vector

        % Vector connecting right and left heel strikes
        w = xy_RHEE(HSrightlocs(j), :) - xy_LHEE(HSleftlocs(j), :);

        % Cross product to find the perpendicular distance
        d_vec = cross([vec, 0], [w, 0]);
        d = norm(d_vec); % Magnitude of the perpendicular vector

        % Store the calculated distance
        wbR(cr) = d;
        cr = cr + 1;
    end
end

% Convert from mm to cm
wbL = wbL * 0.1;
wbR = wbR * 0.1;

% Remove NaN values
wbL = wbL(~isnan(wbL));
wbR = wbR(~isnan(wbR));

% If no values are calculated, set to NaN
if isempty(wbL), wbL = NaN; end
if isempty(wbR), wbR = NaN; end

end

%                 y
%                  ↑
%                 |
% L2 •            |
%        |        | 
%        |        |
% vec (L2 - L1)   |     
%        |        |
%        | - - - -• R1
%        |  width |
%        |        |
%        |        |
%        |        |
% L1 •----•-------•

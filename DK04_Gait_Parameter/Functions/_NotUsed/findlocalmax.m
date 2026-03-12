function [pks, locs] = findlocalmax(signal, neighbourhoodrad, option)
%FINDLOCALMAX computes the local extrema (maxima or minima).
%   [PKS, LOCS] = FINDLOCALMAX(SIGNAL, NEIGHBOURHOODRAD, OPTION) computes
%   the local extrema with values PKS and locations LOCS for the vector
%   SIGNAL by looking at the neighbourhood of each point point in
%   the signal. The neighbourhood of point n is defined as the interval
%   [n-NEIGHBOURHOODRAD; n+NEIGHBOURHOODRAD].
%   No point from the first or last neighbourhoodrad-many frames can be an
%   extrema. Parameter OPTION specifies whether to find maxima or minima
%   and takes the values 'min' or 'max'.
%
% Written on 2011-07-18
% Last modified on 2011-07-18

% find local maxima or minima?
switch option
    case 'min'
        signal = -signal;
        sign = -1;
    case 'max'
        sign = 1;
    otherwise
        error('MATLAB:findlocalmax:wrongOption', ...
            'Option must have a value ''min'' or ''max''');
end

r = neighbourhoodrad;
d = length(signal);
islocalmaximum = false(1, d);
% for the neighbourhood of each frame
for n = r+1:d-r
    islocalmaximum(n) = all(signal([n-r:n-1 n+1:n+r]) < signal(n));
end

locs = find(islocalmaximum);
pks = signal(locs);

% sign correction
pks = sign * pks;

return;
end
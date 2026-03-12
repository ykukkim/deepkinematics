function [h] = plot_errorcloud_seg(y, ye, muscle_name,varargin)
%__________________________________________________________________________
% Copyright (C) 2016 Todd Pataky
% $Id: plot_errorcloud.m 1 2016-01-04 16:07 todd $

%parse inputs
parser         = inputParser;
addOptional(parser, 'facecolor', 'k');
addOptional(parser, 'facealpha', 0.5, @(x) isscalar(x) && (x>=0) && (x<=1) );
parser.parse(varargin{:});
facecolor      = parser.Results.facecolor;
facealpha      = parser.Results.facealpha;

y = y(~isnan(y));
ye =ye(~isnan(ye));
if muscle_name == 'SOL'
    t  = 45/(length(y)-1);
    x  = 5:t:50;
elseif muscle_name == 'TIA'
    t  = 15/(length(y)-1);
    x  = 0:t:15;
elseif muscle_name == 'GAL'
    t  = 40/(length(y)-1);
    x  = 20:t:60;
    
elseif muscle_name == 'GAM'
    t  = 40/(length(y)-1);
    x  = 20:t:60;
    
elseif muscle_name == 'VAM'
    t  = 15/(length(y)-1);
    x  = 0:t:15;
    
elseif muscle_name == 'VAL'
    t  = 25/(length(y)-1);
    x  = 0:t:25;
    
elseif muscle_name == 'BIF'
    t  = 40/(length(y)-1);
    x  = 60:t:100;
end

[y0,y1]   = deal(y+ye, y-ye);
[x,y0,y1] = deal( [x(1) x x(end)], [y0(1) y0 y0(end)], [y1(1) y1 y1(end)]);
[x1,y1]   = deal(fliplr(x), fliplr(y1));
[X,Y]     = deal([x x1], [y0 y1]);
h         = patch(X, Y, 0.3*[1,1,1]);
set(h, 'FaceColor',facecolor, 'FaceAlpha',facealpha, 'EdgeColor','None')

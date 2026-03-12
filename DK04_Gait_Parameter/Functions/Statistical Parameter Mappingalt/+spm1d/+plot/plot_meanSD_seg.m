function [h0,h1] = plot_meanSD_seg(Y,muscle_name, varargin)
%__________________________________________________________________________
% Copyright (C) 2016 Todd Pataky
% $Id: plot_meanSD.m 1 2016-01-04 16:07 todd $


%parse inputs
parser = inputParser;
addOptional(parser, 'color', 'k');
addOptional(parser, 'linewidth', 2, @(x) isscalar(x));
addOptional(parser, 'facealpha', 0.25, @(x) isscalar && (x>=0) && (x<=1) );
parser.parse(varargin{:});
color          = parser.Results.color;
linewidth      = parser.Results.linewidth;
facealpha      = parser.Results.facealpha;

Y = Y';
[y,ys]    = deal(nanmean(Y,1), nanstd(Y,1));
y = y(~isnan(y));
ys =ys(~isnan(ys));

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

h0        = plot(x, y, 'color',color, 'linewidth',linewidth);
h1        = spm1d.plot.plot_errorcloud_seg(y, ys, muscle_name, 'facecolor',color, 'facealpha',facealpha);

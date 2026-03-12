function [h0,h1] = plot_meanSD_saliency(Y, varargin)
%__________________________________________________________________________
% Copyright (C) 2016 Todd Pataky
% $Id: plot_meanSD.m 1 2016-01-04 16:07 todd $


%parse inputs
parser = inputParser;
addOptional(parser, 'color', 'k');
addOptional(parser, 'linewidth', 2, @(x) isscalar(x));
addOptional(parser, 'facealpha', 0.2, @(x) isscalar && (x>=0) && (x<=1) );
parser.parse(varargin{:});
color          = parser.Results.color;
linewidth      = parser.Results.linewidth;
facealpha      = parser.Results.facealpha;

Y = Y';
[y,ys]    = deal(nanmean(Y,1), nanstd(Y,1)); 
y = y(~isnan(y));
ys =ys(~isnan(ys));
% t = /(length(y)-1);
x         = 1:1:18;
h0        = plot(x, y,'color',color, 'linewidth',linewidth);
h1        = spm1d.plot.plot_errorcloud_saliency(y, ys, 'facecolor',color, 'facealpha',facealpha);

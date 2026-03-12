%% List of marker
% This is the list of markers are used in laboratory for Movement
function [markerlist] = generateMarkerList(VD)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    fulllist = {'RTO1', 'RTO3', 'RTO5', 'RHEE', ... % right foot
        'LTO1', 'LTO3', 'LTO5', 'LHEE', ... % left foot
        'RMMA', 'RLMA', 'RTIB', 'RTTT', 'RTMT', 'RTLF', ... % shank right
        'LMMA', 'LLMA', 'LTIB', 'LTTT', 'LTMT', 'LTLF', ... % shank left
        'RMCO', 'RLCO', 'RTFR', 'RTLL', 'RTLH', ... % thigh right
        'LMCO', 'LLCO', 'LTFR', 'LTLL', 'LTLH', ... % thigh right
        'RASI', 'RTMS', 'RPSI', 'SACR', 'LASI', 'LTMS', 'LPSI', ... %pelvis
        'RWRA', 'RWUL', 'RFRA', 'RFUL', ... % forearm right
        'LWRA', 'LWUL', 'LFRA', 'LFUL', ... % forearm left
        'RMEC', 'RLEC', 'RHLT', 'RHVT', ... % humerus right
        'LMEC', 'LLEC', 'LHLT', 'LHVT', ... % humerus left
        'RSHO', 'LSHO', 'CVC7', 'MSTC', ... % shoulder
        'RFHD', 'RBHD', 'LFHD', 'LBHD' % head
        };

    fulllist_predicted = {'Head', 'Left_Shoulder', 'Right_Shoulder', 'LHJC', 'RHJC',...
        'LKJC', 'RKJC', 'LAJC', 'RAJC', 'LTO3', 'RTO3'}; %,'LHEE','RHEE'};
    
    markerlist = '';
    for marker = fieldnames(VD)'
       markerlist =  [markerlist, {(marker{1})}];
    end
    markerlist = intersect(fulllist_predicted, markerlist);
    
end


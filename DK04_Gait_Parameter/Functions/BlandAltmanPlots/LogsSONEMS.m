function [logs] = LogsSONEMS(logs,subject,ME)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if isempty(logs)
    logs(1).Participant = subject;
    idx = 1;
else
    idx = size(logs,2);
    idx = max(idx) + 1; % NEED FIXING
    logs(idx).Participant = subject;
end

if exist('ME','var')
    error_fcn  = char;
    error_line = [];
    error_msg  = char;
    for i=1:size(ME.stack,1)
        error_fcn  = [ME.stack(i).name,   ' >> ' error_fcn];
        error_line = [ME.stack(i).line error_line, ];
        error_msg  = [ME.message, ' >> ' error_msg];
    end

    logs(idx).Note1 = "Failed";
    logs(idx).ErrorFunction = error_fcn;
    logs(idx).line    = error_line;
    logs(idx).Message = error_msg;
    logs(idx).LastUpdated = datestr(datetime);
else
    logs(idx).Note1 = "Successful";
    logs(idx).LastUpdated = datestr(datetime);
    logs(idx).ErrorFunction = "";
    logs(idx).Note2 = "";
end

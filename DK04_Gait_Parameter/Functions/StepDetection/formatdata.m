function [hs_out, hslocs_out, to_out, tolocs_out,wrongRecog] = formatdata(hs, hslocs, to, tolocs)

%% always start with HS
wrongRecog = false;
hs_out      = [];
hslocs_out  = [];
to_out      = [];
tolocs_out  = [];

n = min([length(hs) length(to)]);
hs(end+1:n) = nan;
hslocs(end+1:n) = nan;
to(end+1:n) = nan;
tolocs(end+1:n) = nan;

%% vectors must be row-wise
if not(size(hs, 2) >= size(hs, 1))
    hs = hs';
end

if not(size(to, 2) >= size(to, 1))
    to = to';
end

h = zeros(1, length(hs));
t = zeros(1, length(to));
h(1:end) = 'H' ;   % hs
t(1:end) = 'T'  ;  % to
hs_to_matrix = [hs to; hslocs tolocs; h t];
hs_to_matrix = sortrows(hs_to_matrix', 2);
hs_to_matrix = hs_to_matrix(not(isnan(hs_to_matrix(:,1))),:);
HH_TT = find((diff(hs_to_matrix(:,3))==0)==1);
HH_TT_size = size(HH_TT(:,1));

%% Removing consecutive HH or TT
for a = HH_TT_size(1,1):-1:1
    hs_to_matrix(HH_TT(a,:),:) = [];
end

%% start with HS
startHeel = hs_to_matrix(1, 3) == 'H' && hs_to_matrix(2, 3) == 'T';
while not(startHeel)
    hs_to_matrix = hs_to_matrix(2:end, :);
    
    wrongRecog = not(startHeel) && size(hs_to_matrix, 1) < 2;
    if wrongRecog, return; end
    
    startHeel = hs_to_matrix(1, 3) == 'H' && hs_to_matrix(2, 3) == 'T';
end

%% end with HS
endHeel = hs_to_matrix(end, 3) == 'H' && hs_to_matrix(end-1, 3) == 'T';
while not(endHeel)
    hs_to_matrix = hs_to_matrix(1:end-1, :);
    
    wrongRecog = not(endHeel) && size(hs_to_matrix, 1) < 2;
    if wrongRecog, return; end
    
    endHeel = hs_to_matrix(end, 3) == 'H' && hs_to_matrix(end-1, 3) == 'T';
end

maxIndx = 1;
htrepeatable = 1;

while htrepeatable
    if mod(maxIndx, 2) == 1
        htrepeatable = hs_to_matrix(maxIndx, 3) == 'H' && hs_to_matrix(maxIndx+1, 3) == 'T';
    else
        htrepeatable = hs_to_matrix(maxIndx, 3) == 'T' && hs_to_matrix(maxIndx+1, 3) == 'H';
    end
    maxIndx = maxIndx + 1;
    if maxIndx == size(hs_to_matrix, 1), break; end
end

hs_to_matrix      = hs_to_matrix(1:maxIndx, :);
values = hs_to_matrix(:, 1);
loctns = hs_to_matrix(:, 2);
hs_out          = transpose(values(hs_to_matrix(:, 3) == 'H'));
hslocs_out      = transpose(loctns(hs_to_matrix(:, 3) == 'H'));
to_out          = transpose(values(hs_to_matrix(:, 3) == 'T'));
tolocs_out      = transpose(loctns(hs_to_matrix(:, 3) == 'T'));

return;
end
% function [hs_out, hslocs_out, to_out, tolocs_out, wrongRecog] = ...
%     formatdata(hs, hslocs, to, tolocs)
% 
% %always start with HS 
% 
% wrongRecog = false;
% hs_out      = [];
% hslocs_out  = [];
% to_out      = [];
% tolocs_out  = [];
% 
% % n = min([length(hs) length(to)]);
% % hs = hs(1:n);
% % hslocs = hslocs(1:n);
% % to = to(1:n);
% % tolocs = tolocs(1:n);
% n = max([length(hs) length(to)]);
% hs(end+1:n) = nan;
% hslocs(end+1:n) = nan;
% to(end+1:n) = nan;
% tolocs(end+1:n) = nan;
% 
% % vectors must be row-wise
% if not(size(hs, 2) >= size(hs, 1))
%     hs = hs';
% end
% 
% if not(size(to, 2) >= size(to, 1))
%     to = to';
% %     hslocs = hslocs';
% %     tolocs = tolocs';
% end
% 
% h = zeros(1, length(hs));
% t = zeros(1, length(to));
% h(1:end) = 'H' ;    % hs
% t(1:end) = 'T'  ;  % to
% hs_to_matrix = [hs to; hslocs tolocs; h t];
% hs_to_matrix = sortrows(hs_to_matrix', 2);
% % 
% hs_to_matrix = hs_to_matrix(not(isnan(hs_to_matrix(:,1))),:);
% 
% %start with HS
% startHeel = hs_to_matrix(1, 3) == 'H' && hs_to_matrix(2, 3) == 'T';
% while not(startHeel)
%     hs_to_matrix = hs_to_matrix(2:end, :);
%     
%     wrongRecog = not(startHeel) && size(hs_to_matrix, 1) < 2;
%     if wrongRecog, return; end
%     
%     startHeel = hs_to_matrix(1, 3) == 'H' && hs_to_matrix(2, 3) == 'T';
% end
% 
% %end with HS
% endHeel = hs_to_matrix(end, 3) == 'H' && hs_to_matrix(end-1, 3) == 'T';
% while not(endHeel)
%     hs_to_matrix = hs_to_matrix(1:end-1, :);
%     
%     wrongRecog = not(endHeel) && size(hs_to_matrix, 1) < 2;
%     if wrongRecog, return; end
%     
%     endHeel = hs_to_matrix(end, 3) == 'H' && hs_to_matrix(end-1, 3) == 'T';
% end
% 
% maxIndx = 1;
% htrepeatable = 1;
% while htrepeatable
%     if mod(maxIndx, 2) == 1
%         htrepeatable = hs_to_matrix(maxIndx, 3) == 'H' && hs_to_matrix(maxIndx+1, 3) == 'T';
%     else
%         htrepeatable = hs_to_matrix(maxIndx, 3) == 'T' && hs_to_matrix(maxIndx+1, 3) == 'H';
%     end
%     maxIndx = maxIndx + 1;
%     if maxIndx == size(hs_to_matrix, 1), break; end
% end
% 
% hs_to_matrix      = hs_to_matrix(1:maxIndx, :);
% values = hs_to_matrix(:, 1);
% loctns = hs_to_matrix(:, 2);
% hs_out          = transpose(values(hs_to_matrix(:, 3) == 'H'));
% hslocs_out      = transpose(loctns(hs_to_matrix(:, 3) == 'H'));
% to_out          = transpose(values(hs_to_matrix(:, 3) == 'T'));
% tolocs_out      = transpose(loctns(hs_to_matrix(:, 3) == 'T'));
% 
% 
% % % TODO include L = left; R = right column. Starting with HS divide events
% % into two-blocks. If LLL or RRR, then skip the trial
% 
% % Possible that some was not recognized by FVA, those trials skipped with a
% % phase more then 1 sec
% % if any(diff(hs_to_matrix(:, 2)) > 1.5*samplingRate)
% %     hs_out = [];
% %     to_out = [];
% %     hslocs_out = [];
% %     tolocs_out = [];
% %     return;
% % end
% 
% 
% return;
% end
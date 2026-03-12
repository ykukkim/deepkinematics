function [wbL, wbR] = stepwidth_Treadmill(heelLx, heelLy, heelRx, heelRy, hsleft,hsright,toleft,toright)
%% function to calculate step width on a treadmill! 
%  in order to recieve a better walking direction vector, vec is calculated
%  from heel-strike and the follwoing toe-off.
%  wbL and wbR are identical, just for reasons of laziness still both
%  included

LHEEL = [heelLx heelLy];
RHEEL = [heelRx heelRy];

minSteps = min([numel(hsleft),numel(hsright),numel(toleft),numel(toright)]);

wbL = nan(1, minSteps);
wbR = nan(1, minSteps);

if hsleft(1) < toleft(1)
    vec = LHEEL(hsleft(1),1:2)-LHEEL(toleft(1),1:2);
else
    vec = LHEEL(hsleft(1),1:2)-LHEEL(toleft(2),1:2);
end
vec = vec / norm(vec); %assuming that vec will always be same direction, because of the static treadmill 

for j = 1:minSteps-1
 
w = LHEEL(hsleft(j),1:2) - RHEEL(hsright(j),1:2);
d_vec = cross([vec 0],[w 0]);
d = norm(d_vec);
wbL(j) = d;

w = RHEEL(hsright(j),1:2) - LHEEL(hsleft(j),1:2);
d_vec = cross([vec 0],[w 0]);
d = norm(d_vec);
wbR(j) = d;

end 
    
%     % Walking base (WB) (in the transverse plane)
%     j = 1;
%     cl = 1;
%     cr = 1;
%     n = ceil(size(hs_matrix, 1)-2 / 2);
%     wbL = nan(1, n);
%     wbR = nan(1, n);
%     while j <= size(hs_matrix, 1)-2
%         if hs_matrix(j, 3) == 'L'
%             vec = LHEEL(hs_matrix(j+2, 2), 1:2) - LHEEL(hs_matrix(j, 2), 1:2);
%             vec = vec / norm(vec); % Unit vector connecting 
%                                    % the two consecutive heels
%             w = LHEEL(hs_matrix(j, 2), 1:2) - RHEEL(hs_matrix(j+1, 2), 1:2);
%                                    % Vector connecting the Left heel
%                                    % contact and the right heel contact
%                                    % points
%             d_vec = cross([vec 0], [w 0]); % the cross product here gives 
%                                    % the area of the base (a unit vector) 
%                                    % and the height. 
%             d = norm(d_vec);
%             
%             vec = sign(vec(2))*vec;
%             d = sign(dot(vec,w))*d;
%             
%                        
%             wbR(1, cr) = d;
%             cr = cr + 1;
%         elseif hs_matrix(j, 3) == 'R'
%             vec = RHEEL(hs_matrix(j+2, 2), 1:2) - RHEEL(hs_matrix(j, 2), 1:2);
%             vec = vec / norm(vec);
%             w = RHEEL(hs_matrix(j, 2), 1:2) - LHEEL(hs_matrix(j+1, 2), 1:2);
%             d_vec = cross([vec 0], [w 0]);
%             d = norm(d_vec);
%             
%             vec = sign(vec(2))*vec;
%             d = sign(dot(vec,w))*d;            
%                         
%             wbL(1, cl) = d;
%             cl = cl + 1;
%         end
%         j = j + 1;
%     end

    % Convert to cm
    wbL = 10^(-1) * wbL;
    wbR = 10^(-1) * wbR;
    
    wbL = wbL(not(isnan(wbL)));
    wbR = wbR(not(isnan(wbR)));
    if isempty(wbL), wbL = NaN; end
    if isempty(wbR), wbR = NaN; end
end

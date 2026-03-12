function [normVDleft, normVDright] = normStep(VD, nHSleft, ...
    nHSright, HSleftlocs, HSrightlocs, direction) 
%Temporal normalization of data from HS to HS 
%set each step from 0 - 100%
% starting from HS and going to HS as 1 complete cycle

% 
% 
ex = exist('normVD.mat','file');
if ex
    load('normVD.mat')
end

%markerlist
Markerlist=char(fieldnames(VD));
LMarkerlist=length(Markerlist)-1;
        
%time for interpolation
t = (0:1:100).';
        
%LTHL marker
%nHSleft=trial.nHSleft
        for i=1:LMarkerlist
            % strcmp(s1,s2) compares two strings for equality
            % Markerlist(i,1:2) distinguishes between right and left leg
            % SF = sampling frequency = 100Hz 
            if not(strcmp(Markerlist(i,1:2), 'SF'))
                
            %if Markerlist(i)=='L' || Markerlist(i)=='S' || Markerlist(i)=='M'  %left and middle side
                for l=1:nHSleft-1;
                    
                    start=HSleftlocs(l);
                    stop=HSleftlocs(l+1);
                    
                    comd = ['StepValue=VD.', Markerlist(i,:), '.Values.', direction, '(start:stop);'];
                    eval(comd);
                    StepPoints=1:length(StepValue);
                    %StepPoints=start:stop;
                                                          
                    samples=stop-start;
                    NewData=resample(double(StepValue),double(101),double(samples+1),0);
                    NewData=smooth(NewData);
                    
%                     if (stop-start) < length(t);
%                         NewData=interp1(StepPoints,StepValue,t,'linear','extrap');
%                     else
%                         NewData=interp1(StepPoints,StepValue,t,'linear');
%                     end
                                       
                    comd = ['normVDleft.', Markerlist(i,:),'(:,l)=NewData;'];
                    eval(comd);
                end
                
            %elseif Markerlist(i)=='R' %only right side 
                for r=1:nHSright-1;
                    start=HSrightlocs(r);
                    stop=HSrightlocs(r+1);
                    
                    comd = ['StepValue=VD.', Markerlist(i,:), '.Values.', direction, '(start:stop);'];
                    eval(comd); %read VD from all markers in Markerlist
                    StepPoints=1:length(StepValue);
                    
                    samples=stop-start;
                    NewData=resample(double(StepValue),double(101),double(samples+1),0);
                    NewData=smooth(NewData);
                    
%                     if (stop-start) < length(t);
%                         NewData=interp1(StepPoints,StepValue,t,'linear','extrap');
%                     else
%                         NewData=interp1(StepPoints,StepValue,t,'linear');
%                     end
                              
                    comd = ['normVDright.', Markerlist(i,:),'(:,r)=NewData;'];
                    eval(comd);
                end
            %end
            end
        end
            
return
end
            
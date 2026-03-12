function [normVDleft, normVDright] = normstep_angles(angles, nHSleft, ...
    nHSright, HSleftlocs, HSrightlocs, direct)
%Temporal normalization of data from HS to HS
%set each step from 0 - 100%
% starting from HS and going to HS as 1 complete cycle

%
%
ex = exist('normVD.mat','file');
if ex
    load('normVD.mat')
end
njointAnglesEv = {'ankle_L', 'knee_L', 'hip_L', 'pelvis_L', 'trunk_L', ...
    'ankle_R', 'knee_R', 'hip_R', 'pelvis_R', 'trunk_R'};
if direct == 1
    direction = {'flex', 'flex', 'flex', 'ap', 'ap',...
        'flex', 'flex', 'flex', 'ap', 'ap'};
elseif direct == 2
    direction = {'eversion', 'add', 'add', 'side', 'side',...
        'eversion', 'add', 'add', 'side', 'side'};
elseif direct == 3
    direction = {'rotlong', 'rotlong', 'rotlong', 'rot', 'rot',...
        'rotlong', 'rotlong', 'rotlong', 'rot', 'rot'};
end


%markerlist
Markerlist=char(fieldnames(angles));

%time for interpolation
t = (0:1:100).';
anglesCounter = 0;
%LTHL marker
%nHSleft=trial.nHSleft
for i=1:size(njointAnglesEv,2)
    % strcmp(s1,s2) compares two strings for equality
    % Markerlist(i,1:2) distinguishes between right and left leg
    % SF = sampling frequency = 100Hz
    if isfield(angles, (genvarname(njointAnglesEv{i})))
        %anglesCounter = anglesCounter + 1;
        
        
        %if Markerlist(i)=='L' || Markerlist(i)=='S' || Markerlist(i)=='M'  %left and middle side
        for l=1:nHSleft-1;
            
            start=HSleftlocs(l);
            stop=HSleftlocs(l+1);
            
            if (strcmp(njointAnglesEv{i}, 'trunk_L')) ...
                    || (strcmp(njointAnglesEv{i}, 'trunk_R'))
                comd = ['StepValue=angles.', njointAnglesEv{i}, '.pelvis_trunk.', direction{i}, '(start:stop);'];
            elseif (strcmp(njointAnglesEv{i}, 'arm_L')) ...
                    || (strcmp(njointAnglesEv{i}, 'arm_R'))
                comd = ['StepValue=angles.', njointAnglesEv{i}, '.shoulder.', direction{i}, '(start:stop);'];
            else
                comd = ['StepValue=angles.', njointAnglesEv{i}, '.', direction{i}, '(start:stop);'];
            end
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
            
            comd = ['normVDleft.', njointAnglesEv{i},'(:,l)=NewData;'];
            eval(comd);
        end
        
        %elseif Markerlist(i)=='R' %only right side
        for r=1:nHSright-1;
            start=HSrightlocs(r);
            stop=HSrightlocs(r+1);
            
            if (strcmp(njointAnglesEv{i}, 'trunk_L')) ...
                    || (strcmp(njointAnglesEv{i}, 'trunk_R'))
                comd = ['StepValue=angles.', njointAnglesEv{i}, '.pelvis_trunk.', direction{i}, '(start:stop);'];
            else
                comd = ['StepValue=angles.', njointAnglesEv{i}, '.', direction{i}, '(start:stop);'];
            end
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
            
            comd = ['normVDright.', njointAnglesEv{i},'(:,r)=NewData;'];
            eval(comd);
        end
        %end
    end
end

return
end
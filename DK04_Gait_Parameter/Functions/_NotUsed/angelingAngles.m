function [kinematik] = angelingAngles(subjectFile, filename, newVD)

% load([dirdata, '/', filename]);
% VD = [];
% VD = newVD;
% GE = newGE;
%
% cd(dirdata);
% spot = dir('*.mat');
% names = spot.name;

% k = filename(end-5:end-4);
% k = str2num(k);
% [analogdata,coord,analfreq,videofreq,parameter,group,fail,event] = new_old_converter_V21(spot,k, newVD, newGE);

%analfreq = 2000;
%if exist(subject, 'var') ~= 1
% subjectFolder = '/Volumes/green_groups_lmb_public/Projects/NCM/NCM_EXP/NCM_FAT/for Navrag/Gait/analyseData/Data/BUJU_94/subject';
% addpath(genpath(subjectFolder));
% filetocopy = 'subject_BUJU_94_01.mat';
% sourceFold = [subjectFolder, '/', filetocopy];
% destFold = [dirdata, '/', filetocopy];
% copyfile(sourceFold, destFold);

LoadedData = load(subjectFile);
subject = LoadedData.subject;
%end

answer = [0, 0, 1, 0];
if answer(1)
    Drehung = 'G';
    shift1=0;
end
if answer(3)
    Drehung = 'U';
    shift1=0;
end
if answer(1) && answer(3)
    Drehung= 'G';
    shift1=1;
end

markers = fieldnames(newVD);

GaitEvents.THT1.L = 1;
GaitEvents.THT1.R = 1;
GaitEvents.THT2.L = 1;
GaitEvents.THT2.R = size(newVD.(genvarname(markers{1})).Values.x_coord,1);
GaitEvents.TTO1.L = size(newVD.(genvarname(markers{1})).Values.x_coord,1);
GaitEvents.TTO1.R = size(newVD.(genvarname(markers{1})).Values.x_coord,1);
GaitEvents.THT3.L = NaN;% 23.3.15 (Michi Angst) ?nderung damit mehr als nur ein Schritt ausgewertet werden kann
GaitEvents.THT3.R = NaN;
GaitEvents.TTO2.L = NaN;
GaitEvents.TTO2.R = NaN;

noofStrides=1;

[Referenz_stat,Referenz,Joints]=LadeReferenzSysteme_Navi(subject,Drehung);

% Extrahiere die Segmentmarker aus dem Gang-file:
Segm = segmentextractor_stride_Navi_ReducedFAT_forFAT(newVD); % create the Segm struct

% f?hre die Berechnungen durch:
kinematik = []; % inizialisiere die Werte
negated   = 0;  % inizialisiere die Werte
stopper=0;
fail = 0;

%strideSide ={'Right stride','Left stride'};
Side = {'R', 'L'};

System = 'neu';

% for strS=1%:length(strideSide)
%     if strcmp(strideSide{strS}, 'Right stride')
%         noofStrides = nHSright - 1;
%     elseif strcmp(strideSide{strS}, 'Left stride')
%         noofStrides = nHSleft - 1;
%     end
for p=1:noofStrides % 23.3.15 (Michi Angst) damit jeder Schritt ausgewertet wird
    for c = 1:length(Side)
        % to be taken from GaitCycles structure.
        % to be run for left vs. the right cycles.
        % waitbar aktualisieren:
        %steps=1/5;
        %             waitbar(ACounter*Anteil,h,{['bearbeite: "' Files{durchlauf} '"'],...
        %                 ['verarbeitete Daten: ' num2str(round(ACounter*Anteil*100)) '%']})
        %             ACounter=ACounter+steps;
        
        % Kontrolliere die Gateevents:
        kinematik.gaitcycle.general = answer(2); % =0 falls nur standphase, =1 wenn stand- und schwungphase
        %% probably dont need this segment??
        %             if strcmp(Info.GaitType{durchlauf},'_gt_')==1 || ...
        %                     strcmp(Info.GaitType{durchlauf},'_ru_')==1 ||...
        %                     strcmp(Info.GaitType{durchlauf},'_md_dn_')==1 ||...
        %                     strcmp(Info.GaitType{durchlauf},'_md_up_')==1
        %                 [fail,kinematik,TempSnipedSegm,Tempnframes,Tempstandphase,Range] = ...
        %                     EventChecker_Navi_FAT(kinematik,Segm,GaitEvents,Files{durchlauf},Side{c},p);  % 23.3.15 (Michi Angst), variable p hinzugef?gt und Event_Checker ge?ndert
        %             else
        [fail,kinematik,TempSnipedSegm,Tempnframes,Tempstandphase,Range] = ...
            EventChecker_Navi(kinematik,Segm,GaitEvents,filename,Side{c},p);
        %             end
        %%
        %         if strcmp(strideSide{strS}, 'Right stride')
        %             start=HSrightlocs(p);
        %             stop=HSrightlocs(p+1);
        %         elseif strcmp(strideSide{strS}, 'Left stride')
        %             start=HSleftlocs(p);
        %             stop=HSleftlocs(p+1);
        %         end
        %         Range = [start stop];
        %         TempSnipedSegm.(genvarname(strideSide{strS})).(['Stride' num2str(p)]).(Side{c}).foot = ...
        %             Segm.(['foot_' Side{c}])(start:stop);
        %         %         SnipedSegm(p).(Side{c}).foot_R = ...
        %         %             Segm.(['foot_R'])(start:stop);
        %         %     SnipedSegm(i).(Side).rearfoot = ...
        %         %         Segm.(['rearfoot_' Side])(StartFrame(i):EndFrame(i));
        %         TempSnipedSegm.(genvarname(strideSide{strS})).(['Stride' num2str(p)]).(Side{c}).shank =...
        %             Segm.(['shank_' Side{c}])(start:stop);
        %         %         SnipedSegm(p).(Side{c}).shank_R =...
        %         %             Segm.(['shank_R'])(start:stop);
        %         TempSnipedSegm.(genvarname(strideSide{strS})).(['Stride' num2str(p)]).(Side{c}).thigh = ...
        %             Segm.(['thigh_' Side{c}])(start:stop);
        %         %         SnipedSegm(p).(Side{c}).thigh_R = ...
        %         %             Segm.(['thigh_R'])(start:stop);
        %         TempSnipedSegm.(genvarname(strideSide{strS})).(['Stride' num2str(p)]).(Side{c}).pelvis = ...
        %             Segm.pelvis(start:stop);
        %
        %         Tempnframes.(genvarname(strideSide{strS})).(['Stride' num2str(p)]).(Side{c}) = stop - start + 1;
        
        
        % starte die Berechnung:
        if fail==0
            SnipedSegm = [];
            nframes = [];
            standphase = [];
            SnipedSegm.(Side{c}) = TempSnipedSegm(p).(Side{c}); % 23.3.15 (Michi Angst), TempSnipedSegm ist jetzt abh?nging von p (Schrittvariable)
            nframes.(Side{c}) = Tempnframes(p).(Side{c});
            standphase.(Side{c}) = Tempstandphase(p).(Side{c});
            %
            %             % Bestimmung der Bewegungsrichtung (f?r Fall "Treppe ab"), f?r den Fall Kniebeuge wird dies ?bersprungen und die Werte Check fest gesetzt:
            %             if strcmp(Info.GaitType{durchlauf},'_uncut_') || strcmp(Info.GaitType{durchlauf},'_cut_') || strcmp(Info.GaitType{durchlauf},'_stand_')  %Einschub von Renate 31.7.08
            %                 fail2 = 0;
            %                 Check = [1,2];
            %             else
            [Referenz_stat,Referenz,Joints,SnipedSegm,negated,Check,fail2] = ...
                TreppeAbEinschub_Navi(subject,Referenz_stat,Referenz,Joints,SnipedSegm,Side{c},negated);              % TreppeABEinschub, welches auch Richtung des Standingtrials betrachtet ersetzt 13.6.12
            %             end
            %% (Start falls fail2=0)
            if fail2 == 0
                kinematik.gaitcycle.(Side{c}) = 1;
                % bestimme die tolerablen Marker:
                tolerable = BestimmeTolerableMarker_NaviReduced(Referenz);
                
                % HJC in Thighcluster einsetzen:
                HJC_pelvis.(Side{c}) = Joints.(['hip_' Side{c}]).HJC_pelvis;
                HJC_lab.(Side{c}) = HJCtransformation_NR(Referenz.pelvis, SnipedSegm.(Side{c}).pelvis,...
                    tolerable.pelvis,HJC_pelvis.(Side{c}),Side{c});
                for counter = 1:nframes.(Side{c}) % start here!!!
                    a = SnipedSegm.(Side{c}).thigh(counter).data;
                    b = HJC_lab.(Side{c})(counter).data;
                    av = SnipedSegm.(Side{c}).thigh(counter).valid;
                    bv = HJC_lab.(Side{c})(counter).valid;
                    SnipedSegm.(Side{c}).thigh(counter).data = [a b];
                    SnipedSegm.(Side{c}).thigh(counter).valid = [av bv];
                end
                
                % Berechne Relativbewebung aller Frames:
                Tempresult = BerechneRelativBewegungAllFrames_Navi(Referenz,SnipedSegm,tolerable,Side{c},Drehung);
                result.(Side{c}) = Tempresult.(Side{c});
                
                % Beschreiben der Gelenksbewegungen mittels klinischer
                % Terminologie:
                %                     kinematik = ForefootRearfoot(kinematik,Referenz,Referenz_stat,SnipedSegm,result,Side{c},System);
                kinematik = AncleRotations_Navi(kinematik,Referenz,Referenz_stat,Joints,SnipedSegm,result,Side{c},System);
                kinematik = KneeRotations(kinematik,result,Referenz,Joints,SnipedSegm,Side{c});
                kinematik = HipRotations(kinematik,result,Referenz,Referenz_stat,SnipedSegm,Joints,Side{c});
                kinematik = PelvisRotations(kinematik,result,Referenz,SnipedSegm,Side{c});
                
                %Flo; August 2012
                % Beschreiben der Gelenksbewegungen mittels klinischer
                % Terminologie (Grood and Santey):
                kinematik = GroodSuntayHip(kinematik,...
                    result,Side{c},subject);
                kinematik = GroodSuntayKnee(kinematik,...
                    result,Side{c},subject);
                kinematik = GroodSuntayAnkle_Navi(kinematik,...
                    result,Side{c},subject);
                %                     kinematik = GroodSuntayFoot(kinematik,...
                %                         result,Side{c},subject);
                %                 kinematik = GroodSuntayPelvis(kinematik,...
                %                     result,Side{c},subject);
                
                % Unterscheidung Auswertung Trunk, Back und / oder Arm
                if (strcmp(Drehung,'G') && durchlauf==1 && c==1) || (strcmp(Drehung,'G') && durchlauf==1 && c==2 && stopper==1)
                    uiwait(msgbox('F?r die Trunk und Back Auswertung ist keine gedrehte Version vorhanden. Output dieser Segmente wird NaN gesetzt!!!'))
                end
                % Trunk:
                if isfield(subject.Model.Segments,'trunk') && strcmp(Drehung,'U');
                    kinematik = trunk_implement_NR(subject,kinematik,Range,newVD,result.(Side{c}),Check,Side{c});
                    % Flo, August 2012
                    % Beschreiben der Gelenksbewegungen mittels klinischer
                    % Terminologie (Grood and Santey): In
                    % trunk_implement_NR integriert!
                else
                    if c==2
                        kinematik.trunk_L.trunk_raum.ap = NaN;
                        kinematik.trunk_L.trunk_raum.side = NaN;
                        kinematik.trunk_L.trunk_raum.rot = NaN;
                        kinematik.trunk_L.pelvis_trunk.ap = NaN;
                        kinematik.trunk_L.pelvis_trunk.side = NaN;
                        kinematik.trunk_L.pelvis_trunk.rot = NaN;
                    elseif c==1
                        kinematik.trunk_R.trunk_raum.ap = NaN;
                        kinematik.trunk_R.trunk_raum.side = NaN;
                        kinematik.trunk_R.trunk_raum.rot = NaN;
                        kinematik.trunk_R.pelvis_trunk.ap = NaN;
                        kinematik.trunk_R.pelvis_trunk.side = NaN;
                        kinematik.trunk_R.pelvis_trunk.rot = NaN;
                    end
                end
                
                % Back:
                if isfield(subject.Model.Segments,'back') && strcmp(Drehung,'U')
                    if ~((strcmp(Info.GaitType{durchlauf},'_uncut_') || strcmp(Info.GaitType{durchlauf},'_cut_')) && c==2)
                        % Flo, August 2012
                        % Beschreiben der Gelenksbewegungen mittels klinischer
                        % Terminologie (Grood and Santey): In
                        % back_implement_gelenkskinematik integriert!
                        [kinematik,result.(Side{c})]=back_implement_gelenkskinematik(subject,kinematik,result.(Side{c}),Range,Check,coord,group,parameter,Side{c});
                        [kinematik]=spine_implement_gelenkskinematik(kinematik,Range,coord,group,parameter,Side{c});
                    end
                else
                    % 22.08.13, Oli und Flo: Ganzer Abschnitt ge?ndert (Funktion erstellt und Spine-Felder hinzugef?gt)
                    if c==1
                        [kinematik]=set_NaN_Back_Spine(kinematik,Side{c});
                    elseif c==2
                        [kinematik]=set_NaN_Back_Spine(kinematik,Side{c});
                    end
                end
                
                % Arm:
                
                if (answer(4)==1) && ((durchlauf==1 && c==1) ||...
                        (durchlauf==1 && c==2 && stopper==1))  % Arm_GUI ?nderung 20.07.2014
                    %                     kinematik = arm_implement(subject,SubParFileName,kinematik,Range,coord,group, parameter,result.(Side{c}),Check,Side{c});
                    kinematik = arm_implement2...
                        (subject,SubParFileName,kinematik,...
                        Range,coord,group, parameter,result...
                        .(Side{c}),Check,Side{c});    % 25.11.13 (David Burkhardt): Aufgrund von Julia Lindorfer und Florian Schellenberg ge?ndertes Programm.
                else
                    if c==2
                        kinematik.arm_L.shoulder.ant = NaN;     %%% Turgut G?lay 27.10.2010
                        kinematik.arm_L.shoulder.abd = NaN;
                        kinematik.arm_L.shoulder.rot = NaN;
                        kinematik.arm_L.elbow.flex = NaN;
                        kinematik.arm_L.elbow.NaN = NaN;
                        kinematik.arm_L.elbow.NaN2 = NaN;
                    elseif c==1
                        kinematik.arm_R.shoulder.ant = NaN;
                        kinematik.arm_R.shoulder.abd = NaN;
                        kinematik.arm_R.shoulder.rot = NaN;
                        kinematik.arm_R.elbow.flex = NaN;
                        kinematik.arm_R.elbow.NaN = NaN;
                        kinematik.arm_R.elbow.NaN2 = NaN;
                    end
                end
            else
                kinematik.gaitcycle.(Side{c}) = 0;
                kinematik = FillWithNaNs_Navi(kinematik,Side{c});
            end
            
        else
            kinematik = FillWithNaNs_Navi(kinematik,Side{c});
        end
        if kinematik.gaitcycle.(Side{c})==1;
            kinematik.Translation.pelvis = result.(Side{c}).pelvis; %R und L sind gleich
            kinematik.Translation.pelvis_gedreht = result.(Side{c})...
                .pelvis_gedreht;
            kinematik.back= kinematik.(['back_' Side{c}]);
            kinematik.spine= kinematik.(['spine_' Side{c}]);
            kinematik=rmfield(kinematik,['back_' Side{c}]);
            kinematik=rmfield(kinematik,['spine_' Side{c}]);
        end
        %             if strcmp(Info.GaitType{durchlauf},'_uncut_') ||...
        %                     strcmp(Info.GaitType{durchlauf},'_cut_')
        %                 kinematik.back= kinematik.(['back_' Side{c}]);
        %                 kinematik.spine= kinematik.(['spine_' Side{c}]);
        %                 kinematik=rmfield(kinematik,['back_' Side{c}]);
        %                 kinematik=rmfield(kinematik,['spine_' Side{c}]);
        %             end
    end
    %     end
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %% Shiftberechnung gedreht vs. ungedreht %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % if shift1
    %     Drehung='U';
    %     kinematik_all=shiftcalculation(Files, Info, kinematik_all, LoadedData, Drehung, Side, answer);
    %     for a=1:length(kinematik_all)
    %         kinematik_all(a).referenz='GU';
    %     end
    % end
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %% Kontrolle der Berechungen: %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % % Stelle die berechneten Daten zur Kontrolle dar:
    % kinematik_all = PlotterGUINR_Navi(kinematik_all, Info, shift1);
    % if isempty(kinematik_all)
    %     disp(' ');disp('Programm abgebrochen!');disp(' ')
    %     return
    % end
    % Info.ProcessedFiles = {kinematik_all.name};
    
    return
end
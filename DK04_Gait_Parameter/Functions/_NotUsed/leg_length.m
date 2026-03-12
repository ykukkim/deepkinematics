function GaitSummary=leg_length(DIR, GaitSummary, participant)
%calculate leg length

%calculated for 2 segment centers
%thigh and shank centers to knee, hip and ankle joint
%Add these values to calculate the leg length

%load right directory
% list = dir(DIR);
% names={list.name};
% 
% for i = 4:length(names) % 4 for Mac and 4 for Windows
%     fileName=char(names(i));
%     
%     display(fileName);
%     display({'file',i-2, 'out of' ,length(names)-2});
%     load([DIR_data, '/', fileName]);
%     load(fileName);
    
    
    side=['L';'R'];
    for i=1:length(side);
        %calculate leg length
        ankle=char(['ankle_' side(i)]);
        knee=char(['knee_' side(i)]);
        hip=char(['hip_' side(i)]);
        
        AS=subject.Model.Joint.(genvarname(ankle)).SegmentSystem.AJC_shank; %ankle to shank centre
        SK=subject.Model.Joint.(genvarname(knee)).SegmentSystem.KJC_shank; %Shank centre to knee
        KT=subject.Model.Joint.(genvarname(knee)).SegmentSystem.KJC_thigh; %knee to thigh centre
        TH=subject.Model.Joint.(genvarname(hip)).SegmentSystem.HJC_thigh; %thigh centre to hip
        
        d1=sqrt(AS(1)^2+AS(2)^2+AS(3)^2) ;
        d2=sqrt(SK(1)^2+SK(2)^2+SK(3)^2) ;
        d3=sqrt(KT(1)^2+KT(2)^2+KT(3)^2) ;
        d4=sqrt(TH(1)^2+TH(2)^2+TH(3)^2) ;
        
        d=d1+d2+d3+d4;
        
        GaitSummary.(genvarname(P)).Personalien.LegLength.(genvarname(char(side(i))))=d;
    end
    
    %mean of the leg length
    L=GaitSummary.(genvarname(P)).Personalien.LegLength.L;
    R=GaitSummary.(genvarname(P)).Personalien.LegLength.R;
    M=(L+R)/2;
    GaitSummary.(genvarname(P)).Personalien.LegLength.mean=M;
    
    %calculate dimensionless velocity
    
    g=9.807; %acceleration of gravity
    
    statename=fieldnames(GaitSummary.(genvarname(P)));
    for j=1:length(statename);
        S=char(statename(j));
        
        %if statename is pre or post_gt
        pre=strcmp(S,'Pre_gt');
        post=strcmp(S,'Post_gt');
        if or(pre, post)
            
            
            GtNo=fieldnames(GaitSummary.(genvarname((P))).(genvarname((S))));
            l_gt=length(GtNo);
            for k=1:l_gt;
                
                %                 L=GaitVariability.(genvarname(P)).Personalien.LegLength.L;
                %                 R=GaitVariability.(genvarname(P)).Personalien.LegLength.R;
                %                 M=(L+R)/2
                %                 GaitVariability.(genvarname(P)).Personalien.LegLength.mean=M;
                
                V=GaitSummary.(genvarname(P)).(genvarname(S)).(genvarname(char(GtNo(k)))).velocity;
                
                VdimmLess=V/sqrt(g*M/1000);
                
                
                GaitSummary.(genvarname(P)).(genvarname(S)).(genvarname(char(GtNo(k)))).V_dimmLess=VdimmLess;
            end
        end
    end
    return
end


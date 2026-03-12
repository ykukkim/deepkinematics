function GaitSummary=ListData(GaitSummary)
%%list all gait data in Total

%P=Participant
%S=State (post/pre)

partname=fieldnames(GaitSummary);
for i=1:length(partname);
    P=char(partname(i));
    exercisename=fieldnames(GaitSummary.(genvarname(P)));
    for j=1:length(exercisename);
        E=char(exercisename(j));
        statename = fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)));
        for statek = 1:length(statename)
            S = char(statename(statek));
            %if statename is pre or post_gt
            pre=strcmp(S,'Pre_gt');
            post=strcmp(S,'Post_gt');
            if or(pre, post)
                
                GtNo=fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)));
                Gaitparams = char(fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1))))));
                params = strcmp(Gaitparams, 'GaitParameters');
                cycles = strcmp(Gaitparams, 'GaitCycles');
                if params
                    ParameterData=fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1))))).(genvarname(Gaitparams));
                    
                    % extend structure with Total map and add data from all the walking patterns
                    l_gt=length(GtNo);
                    l_d=length(ParameterData);
                    
                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters=[];
                    
                    for k = 1:l_d
                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.(genvarname(char(ParameterData(k))))=[];
                        for l= 1:l_gt
                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParamters.(genvarname(char(ParameterData(k))))=...
                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.(genvarname(char(ParameterData(k))))...
                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(l)))).(genvarname(char(ParameterData(k))))];
                        end
                        
                    end
                end
                l_nm = length(CycleData_Mediolateral);
                ml = 'mediolateral';
                ap = 'anteroposterior';
                lg = 'longitudinal';
                
                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles=[];
                if cycles
                    CycleData_Mediolateral = fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1))))).(genvarname(Gaitparams)).(genvarname(ml));
                    for markerk = 1:l_nm
                        
                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles(genvarname(ml)).(genvarname(char(CycleData(markerk))))=[];
                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles(genvarname(ap)).(genvarname(char(CycleData(markerk))))=[];
                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles(genvarname(lg)).(genvarname(char(CycleData(markerk))))=[];
                        for cyclel = 1:l_gt
                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(ml)).(genvarname(char(CycleData(markerk))))=...
                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(ml)).(genvarname(char(CycleData(markerk))))...
                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(GaitParams)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))];
                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(ap)).(genvarname(char(CycleData(markerk))))=...
                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(ap)).(genvarname(char(CycleData(markerk))))...
                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(GaitParams)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))];
                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(lg)).(genvarname(char(CycleData(markerk))))=...
                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(lg)).(genvarname(char(CycleData(markerk))))...
                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(GaitParams)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))];
                        end
                    end
                end
            end
        end
    end
end

%     % save structure
%     save GaitVariability.mat GaitVariability
return
end



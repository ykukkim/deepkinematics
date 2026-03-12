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
                Gaitparams = fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1)))));
                for gaitp = 1:length(Gaitparams)
                    Gp = char(Gaitparams(gaitp));
                    events = strcmp(Gp, 'GaitEvents');
                    params = strcmp(Gp, 'GaitParameters');
                    cycles = strcmp(Gp, 'GaitCycles');
                    l_gt=length(GtNo);
                    
                    if events
                        EventData = fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1)))).(genvarname(Gp)));
                        l_e = length(EventData);
                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp))=[];
                        for eventk = 1:l_e
                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(char(EventData(eventk))))=[];
                            for eventl = 1:l_gt
                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(char(EventData(eventk))))=...
                                    [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(char(EventData(eventk))))...
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(eventl)))).(genvarname(Gp)).(genvarname(char(EventData(eventk))))];
                            end
                        end
                    end
                    
                    if params
                        ParameterData=fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1)))).(genvarname(Gp)));
                        
                        % extend structure with Total map and add data from all the walking patterns
                        
                        l_d=length(ParameterData);
                        
                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp))=[];
                        
                        for k = 1:l_d
                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(char(ParameterData(k))))=[];
                            for l= 1:l_gt
                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(char(ParameterData(k))))=...
                                    [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(char(ParameterData(k))))...
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(l)))).(genvarname(Gp)).(genvarname(char(ParameterData(k))))];
                            end
                            
                        end
                    end
                    
                    if cycles
                        ml = 'mediolateral';
                        ap = 'anteroposterior';
                        lg = 'longitudinal';
                        sg = 'sagittal';
                        fr = 'frontal';
                        tr = 'transverse';
                        datatype = fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1)))).(genvarname(Gp)));
                        for datat = 1:length(datatype)
                            dt = char(datatype(datat));
                            orig = strcmp(dt, 'OrigInterpolated');
                            trans = strcmp(dt, 'Transformed');
                            if orig
                                le = 'Left';
                                ri = 'Right';
                                CycleData = fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(ml)));
                                
                                l_nm = length(CycleData);
                                
                                %GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).incForward=[];
                                for markerk = 1:l_nm
                                    
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).incForward.(genvarname(ml)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).incForward.(genvarname(ap)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).incForward.(genvarname(lg)).(genvarname(char(CycleData(markerk))))=[];
                                    
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).straightReturn.(genvarname(ml)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).straightReturn.(genvarname(ap)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).straightReturn.(genvarname(lg)).(genvarname(char(CycleData(markerk))))=[];
                                    
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).incForward.(genvarname(ml)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).incForward.(genvarname(ap)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).incForward.(genvarname(lg)).(genvarname(char(CycleData(markerk))))=[];
                                    
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).straightReturn.(genvarname(ml)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).straightReturn.(genvarname(ap)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).straightReturn.(genvarname(lg)).(genvarname(char(CycleData(markerk))))=[];
                                    for cyclel = 1:l_gt
                                        direction = GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).GaitPathDescription.walkingDirection.walkingDirectionString;
                                        
                                        forward = strcmp(direction, 'Inclined_forward');
                                        straightreturn = strcmp(direction, 'Straight_return');
                                        if forward
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).incForward.(genvarname(ml)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).incForward.(genvarname(ml)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))];
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).incForward.(genvarname(ap)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).incForward.(genvarname(ap)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))];
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).incForward.(genvarname(lg)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).incForward.(genvarname(lg)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))];
                                            
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).incForward.(genvarname(ml)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).incForward.(genvarname(ml)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))];
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).incForward.(genvarname(ap)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).incForward.(genvarname(ap)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))];
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).incForward.(genvarname(lg)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).incForward.(genvarname(lg)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))];
                                        elseif straightreturn
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).straightReturn.(genvarname(ml)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).straightReturn.(genvarname(ml)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))];
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).straightReturn.(genvarname(ap)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).straightReturn.(genvarname(ap)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))];
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).straightReturn.(genvarname(lg)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).straightReturn.(genvarname(lg)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))];
                                            
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).straightReturn.(genvarname(ml)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).straightReturn.(genvarname(ml)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))];
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).straightReturn.(genvarname(ap)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).straightReturn.(genvarname(ap)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))];
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).straightReturn.(genvarname(lg)).(genvarname(char(CycleData(markerk))))=...
                                                [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).straightReturn.(genvarname(lg)).(genvarname(char(CycleData(markerk))))...
                                                GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))];
                                        end
                                    end
                                end
                            elseif trans
                                CycleData = fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(ml)));
                                
                                l_nm = length(CycleData);
                                
                                %GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(ri)).(genvarname(dt))=[];
                                for markerk = 1:l_nm
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))=[];
                                    
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))=[];
                                    
                                    for cyclel = 1:l_gt
                                        
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))];
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))];
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))];
                                        
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(ml)).(genvarname(char(CycleData(markerk))))];
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(ap)).(genvarname(char(CycleData(markerk))))];
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(lg)).(genvarname(char(CycleData(markerk))))];                                        
                                        
                                    end
                                end
                                AngleData = fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(1)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(sg)));
                                l_ad = length(AngleData);
                                
                                for jointk = 1:l_ad
                                    
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(sg)).(genvarname(char(AngleData(jointk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(fr)).(genvarname(char(AngleData(jointk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(tr)).(genvarname(char(AngleData(jointk))))=[];
                                    
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(sg)).(genvarname(char(AngleData(jointk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(fr)).(genvarname(char(AngleData(jointk))))=[];
                                    GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(tr)).(genvarname(char(AngleData(jointk))))=[];
                                    
                                    for cyclel = 1:l_gt
                                        
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(sg)).(genvarname(char(AngleData(jointk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(sg)).(genvarname(char(AngleData(jointk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(sg)).(genvarname(char(AngleData(jointk))))];
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(fr)).(genvarname(char(AngleData(jointk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(fr)).(genvarname(char(AngleData(jointk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(fr)).(genvarname(char(AngleData(jointk))))];
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(tr)).(genvarname(char(AngleData(jointk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(le)).(genvarname(tr)).(genvarname(char(AngleData(jointk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(le)).(genvarname(tr)).(genvarname(char(AngleData(jointk))))];
                                        
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(sg)).(genvarname(char(AngleData(jointk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(sg)).(genvarname(char(AngleData(jointk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(sg)).(genvarname(char(AngleData(jointk))))];
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(fr)).(genvarname(char(AngleData(jointk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(fr)).(genvarname(char(AngleData(jointk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(fr)).(genvarname(char(AngleData(jointk))))];
                                        GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(tr)).(genvarname(char(AngleData(jointk))))=...
                                            [GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitCycles.(genvarname(dt)).(genvarname(ri)).(genvarname(tr)).(genvarname(char(AngleData(jointk))))...
                                            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).(genvarname(char(GtNo(cyclel)))).(genvarname(Gp)).(genvarname(dt)).(genvarname(ri)).(genvarname(tr)).(genvarname(char(AngleData(jointk))))];
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

return
end

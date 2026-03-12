function GaitSummary=gaitVariability(GaitSummary)

%calculate gait variability using covariation.mat
%insert spatial parameters

partname=fieldnames(GaitSummary);
for i=1:length(partname);
    P=char(partname(i));
    exname = fieldnames(GaitSummary.(genvarname(P)));
    for ex = 1:length(exname)
        E = char(exname(ex));
        statename=fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)));
        for j=1:length(statename);
            S=char(statename(j));
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.durationGaitCycleL=covariation(GaitSummary, P, E, S,...
                'durationGaitCycleL');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.durationGaitCycleR=covariation(GaitSummary, P, E, S,...
                'durationGaitCycleR');
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.stridelengthL=covariation(GaitSummary, P, E, S,...
                'strideLengthL');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.stridelengthR=covariation(GaitSummary, P, E, S,...
                'strideLengthR');
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.durationStancePhL=covariation(GaitSummary, P, E, S,...
                'durationStancePhL');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.durationStancePhR=covariation(GaitSummary, P, E, S,...
                'durationStancePhR');
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.durationSwingPhL=covariation(GaitSummary, P, E, S,...
                'durationSwingPhL');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.durationSwingPhR=covariation(GaitSummary, P, E, S,...
                'durationSwingPhR');
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.ndlsL=covariation(GaitSummary, P, E, S,...
                'ndlsL');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.ndlsR=covariation(GaitSummary, P, E, S,...
                'ndlsR');
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.dlsL=covariation(GaitSummary, P, E, S,...
                'dlsL');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.dlsR=covariation(GaitSummary, P, E, S,...
                'dlsR');
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.GaitAsymmetry_ShortvsLong=covariation(GaitSummary, P, E, S,...
                'GaitAsymmetry_ShortvsLong');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.PhaseCoordinationIndex_ShortvsLong=covariation(GaitSummary, P, E, S,...
                'PhaseCoordinationIndex_ShortvsLong');
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.MTCL=covariation(GaitSummary, P, E, S,...
                'MTCleft');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.MTCR=covariation(GaitSummary, P, E, S,...
                'MTCright');
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.stepWidthL=covariation(GaitSummary, P, E, S,...
                'stepWidthL');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.stepWidthR=covariation(GaitSummary, P, E, S,...
                'stepWidthR');
            
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.stepLengthL=covariation(GaitSummary, P, E, S,...
                'stepLengthL');
            GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.stepLengthR=covariation(GaitSummary, P, E, S,...
                'stepLengthR');
            
        end
    end
end


return
end

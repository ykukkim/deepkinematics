function CV=gaitVariability(GaitVariability)

%calculate gait variability using covariance.mat
%insert spatial parameters

partname=fieldnames(GaitVariability);
for i=1:length(partname);
    P=char(partname(i));
    statename=fieldnames(GaitVariability.(genvarname(P)));
    for j=1:length(statename);
        S=char(statename(j));

        CV.(genvarname(P)).(genvarname(S)).strideL=covariance(GaitVariability, P, S,...
            'durationGaitCycleL');
        CV.(genvarname(P)).(genvarname(S)).strideR=covariance(GaitVariability, P, S,...
            'durationGaitCycleR');
        
        CV.(genvarname(P)).(genvarname(S)).stridelengthL=covariance(GaitVariability, P, S,...
            'strideLengthL');
        CV.(genvarname(P)).(genvarname(S)).stridelengthR=covariance(GaitVariability, P, S,...
            'strideLengthR');
        
        CV.(genvarname(P)).(genvarname(S)).stanceL=covariance(GaitVariability, P, S,...
            'durationStancePhL');
        CV.(genvarname(P)).(genvarname(S)).stanceR=covariance(GaitVariability, P, S,...
            'durationStancePhR');

        CV.(genvarname(P)).(genvarname(S)).swingL=covariance(GaitVariability, P, S,...
            'durationSwingPhL');
        CV.(genvarname(P)).(genvarname(S)).swingR=covariance(GaitVariability, P, S,...
            'durationSwingPhR');
        
        CV.(genvarname(P)).(genvarname(S)).ndlsL=covariance(GaitVariability, P, S,...
            'ndlsL');
        CV.(genvarname(P)).(genvarname(S)).ndlsR=covariance(GaitVariability, P, S,...
            'ndlsR');
        
        CV.(genvarname(P)).(genvarname(S)).MFCL=covariance(GaitVariability, P, S,...
            'MFCleft');
        CV.(genvarname(P)).(genvarname(S)).MFCR=covariance(GaitVariability, P, S,...
            'MFCright');
        
    end
end

        
return
end

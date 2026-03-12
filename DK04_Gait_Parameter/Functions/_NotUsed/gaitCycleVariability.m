function [ structure ] = gaitCycleVariability( structure )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%direction = 
partname=fieldnames(structure);
for i=1:length(partname);
    P=char(partname(i));
    exname = fieldnames(structure.(genvarname(P)));
    for ex = 1:length(exname)
        E = char(exname(ex));
        statename=fieldnames(structure.(genvarname(P)).(genvarname(E)));
        for j=1:length(statename);
            S=char(statename(j));
            
            % rotate the inclined walks
            %% Change it to trajectories out of the Total set...
            structure.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitCyles.mediolateral = cyclesCovariation(structure, P, E, S, 'mediolateral');
            structure.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitCyles.anteroposterior = cyclesCovariation(structure, P, E, S, 'anteroposterior');
            structure.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitCyles.longitudinal = cyclesCovariation(structure, P, E, S, 'longitudinal');
            structure.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitCyles.sagittal = cyclesCovariation(structure, P, E, S, 'sagittal');
            structure.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitCyles.frontal = cyclesCovariation(structure, P, E, S, 'frontal');
            structure.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitCyles.transverse = cyclesCovariation(structure, P, E, S, 'transverse');
        end
    end
end
end



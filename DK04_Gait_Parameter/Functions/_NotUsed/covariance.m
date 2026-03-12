function CV_final=covariation(structure, Name, state, variable)
% calculate CV 

pre=strcmp(state(1:6),'Pre_gt');
post=strcmp(state,'Post_gt');
CV=[];
SDl=[];
mSDl=[];


if pre;
    Spatial=structure.(genvarname(Name)).(genvarname(state)).Total;
    Var=Spatial.(genvarname(variable))';
    for i=10:length(Var)
        m=mean(Var(1:i));
        SD=std(Var(1:i));
        SDl=[SDl SD];
        CVn=SD/m;
        CV=[CV CVn];
    end
    mSDl=mean(SDl);

elseif post;
        Spatial=structure.(genvarname(Name)).(genvarname(state)).Total;
    Var=Spatial.(genvarname(variable))';
    for i=10:length(Var)
        m=mean(Var(1:i));
        SD=std(Var(1:i));
        SDl=[SDl SD];
        CVn=SD/m;
        CV=[CV CVn];
%     events=fieldnames(structure.(genvarname(Name)).(genvarname(state)));
%     for k=1:length(events)-1 %exclude total
%             Spatial=structure.(genvarname(Name)).(genvarname(state)).(genvarname(char(events(k))));
%             Var=Spatial.(genvarname(variable))';
%             m=mean(Var);
%             SD=std(Var);
%             CVn=SD/m;
%             CV=[CV CVn];
    end
    mSDl=mean(SDl);

% else
%     'error: something is wrong with the states'
end

% comd=[Name,'.CV.',state,'.',variable,'=CV;'];
% eval(comd);
% comd=[Name,'.SD.',state,'.',variable,'=SD;'];
% eval(comd);

CV_final.CV=CV;
CV_final.SD=SDl;
CV_final.meanSD=mSDl;

end 
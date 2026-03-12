function CV_Graph(GaitSummary,Part,vis)
%creates figure for all CV variable

if vis

P=Part;

exercisename = fieldnames(GaitSummary.(genvarname(P)));
statename=fieldnames(GaitSummary.(genvarname(P)).(genvarname(char(exercisename(1)))));
subplot=1;
NVar=fieldnames(GaitSummary.(genvarname(P)).(genvarname(char(exercisename(1)))).(genvarname(char((statename(1))))).Variability.GaitParameters);
%S=Pre/Post
f=1;
for i=1:2:length(NVar);
    for exer = 1:length(exercisename);
        E = char(exercisename(exer));
    for j=1:length(statename);
        S=char(statename(j));
        var=fieldnames(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters);    
        if strcmp(S, 'Pre_gt')
            var1=char(var(i));
            VarL_pre=GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.(genvarname(var1)).CV;
            PreTL=1:length(VarL_pre);
            
            var2=char(var(i+1));
            VarR_pre=GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.(genvarname(var2)).CV;
            PreTR=1:length(VarR_pre);
            
        elseif strcmp(S,'Post_gt')
            var1=char(var(i));
            VarL_post=GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.(genvarname(var1)).CV;
            PostTL=1:length(VarL_post);
            
            var2=char(var(i+1));
            VarR_post=GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Variability.GaitParameters.(genvarname(var2)).CV;
            PostTR=1:length(VarR_post);
        end       
    end
    end
    
    %plot
    figure
    plot(PreTL,VarL_pre,'b',PreTR,VarR_pre,'b:',...
    PostTL,VarL_post,'r',PostTR,VarR_post,'r:');

    var=char(var(i));
    var(1:(length(var)-1));
    legend('Pre-Left','Pre-Right','Post-Left','Post-Right');
    xlabel('Number of steps') ;
    ylabel('CV in (%)');
    T=[P(1:4) '-' P(6:7) ' ' var(1:(length(var)-1))];
    title(T);
    f=f+1;
end

end

end
            
%         
% VarL_pre=CV.(genvarname(P)).Pre_gt.(genvarname(VarL));
% VarL_post=CV.(genvarname(P)).Post_gt.(genvarname(VarL));
% PostTL=1:length(VarL_post);
% PreTL=1:length(VarL_pre);
% 
% VarR_pre=CV.(genvarname(P)).Pre_gt.(genvarname(VarR));
% VarR_post=CV.(genvarname(P)).Post_gt.(genvarname(VarR));
% PostTR=1:length(VarR_post);
% PreTR=1:length(VarR_pre);

% figure
% plot(PreTL,VarL_pre,'b',PreTR,VarR_pre,'b:',...
%     PostTL,VarL_post,'r',PostTR,VarR_post,'r:');
% legend('Pre-Left','Pre-Right','Post-Left','Post-Right');
% xlabel('Number of steps') ;
% ylabel('CV in (%)');
% title(P VarL(1:(length(VarL)-1)));


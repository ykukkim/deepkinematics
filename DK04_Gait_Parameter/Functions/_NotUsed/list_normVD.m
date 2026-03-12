%calculates SD and lists all normStep data 

%% make an empty folder for all markers for Total data and SD of all the data
P= fieldnames(normVD);
for i=1:length(P)
    
    comd = ['S=fieldnames(normVD.', char(P(i)), ');'];
    eval(comd);
    
    for ii=1:length(S)
        
        comd = ['Trial=fieldnames(normVD.' ,char(P(i)), '.' ,char(S(ii)), ');'];
        eval(comd);
        
        for iii= 1: length(Trial)
            
            comd = ['Marker=fieldnames(normVD.', char(P(i)) ,'.' ,char(S(ii)) ,'.' ,char(Trial(iii)) ,');'];
            eval(comd);
            
            
            for iv=1:length(Marker);
                
                %makes an emty list for every marker
                comd = ['normVD.' ,char(P(i)), '.' ,char(S(ii)) ,'.Total.', char(Marker(iv)), '= [];'];
                eval(comd);
                comd = ['normVD.' ,char(P(i)), '.' ,char(S(ii)) ,'.SD.', char(Marker(iv)), '= [];'];
                eval(comd);
                
            end
        end
    end
end

%% paste all correct data in the right folder
P= fieldnames(normVD);
for i=1:length(P)
    
    comd = ['S=fieldnames(normVD.', char(P(i)), ');'];
    eval(comd);
    
    for ii=1:length(S)
        
        comd = ['Trial=fieldnames(normVD.' ,char(P(i)), '.' ,char(S(ii)), ');'];
        eval(comd);
        
        for iii= 1: length(Trial)-2
            
            comd = ['Marker=fieldnames(normVD.', char(P(i)) ,'.' ,char(S(ii)) ,'.' ,char(Trial(iii)) ,');'];
            eval(comd);
           
            
            for iv=1:length(Marker);
                %list all trials
                comd = ['normVD.' ,char(P(i)), '.', char(S(ii)), '.Total.', char(Marker(iv)), '= [normVD.' ,char(P(i)), '.',...
                    char(S(ii)), '.' 'Total', '.', char(Marker(iv)), ' '...
                    'normVD.', char(P(i)), '.', char(S(ii)), '.', char(Trial(iii)), '.' char(Marker(iv)), '];'];
                eval(comd);
            end
        end
    end
end

%% Calculate standard deviation

P= fieldnames(normVD);
for i=1:length(P)
    
    comd = ['S=fieldnames(normVD.', char(P(i)), ');'];
    eval(comd);
    
    for ii=1:length(S)
        
        comd = ['Trial=fieldnames(normVD.' ,char(P(i)), '.' ,char(S(ii)), ');'];
        eval(comd);
        
        %for iii= 1: length(Trial)-2
            
            comd = ['Marker=fieldnames(normVD.', char(P(i)) ,'.' ,char(S(ii)) ,'.Total);'];
            eval(comd);
           
            
            for iv=1:length(Marker);
                
                %calculate SD and mean of all the SD for every marker. 
                StDsec=[];
                
                for k=1:100;
                    comd=['stdsec=std(normVD.' ,char(P(i)), '.', char(S(ii)), '.Total.', char(Marker(iv)),'(k,:));'];
                    eval(comd);
                    StDsec=[StDsec stdsec];
                end
                
                StDe=mean(StDsec);
                
                
                comd = ['normVD.' ,char(P(i)), '.', char(S(ii)), '.SD.', char(Marker(iv)), '= StDe;'];
                eval(comd);
            end
            
            comd = ['normVD.' ,char(P(i)), '.', char(S(ii)), '.SD=' ...
                'orderfields(normVD.' ,char(P(i)), '.', char(S(ii)), '.SD);'];
            eval(comd);
       % end
    end
end

save normVD normVD

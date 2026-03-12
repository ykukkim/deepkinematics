%speed_position
% speed against position plot

marker='MSBP';

names=fieldnames(VideoData);
for i=1:length(names) %participants
    P=char(names(i));
    
    comd = ['states=fieldnames(VideoData.',P,');'];
    eval(comd);
    
    for k=1:length(states) %states, pre and post
        S=char(states(k));
        
        comd = ['No=fieldnames(VideoData.',P,'.',S,');'];
        eval(comd);
        
        for l=1:length(No) %walking tracks
            file=char(No(l));
            
            %check if file exists in GaitVariability
            comd = ['exfield=fieldnames(GaitVariability.',P,'.',S,');'];
            eval(comd);
            ex=max(strcmp(file, exfield));
             if not(ex)
                i=i+1;
                continue
             end
    
            comd = ['zmarker=VideoData.',P,'.',S,'.',file,'.',marker,'.Values.z_coord;'];
            eval(comd);
            
            %cut marker with left hs locations
            HSlocs=GaitVariability.(genvarname(P)).(genvarname(S)).(genvarname(char(file))).HSleftlocs;
                    
            zfootVel = ... % Compute velocity of the marker
                diff(zmarker)' ./ transpose(diff(1:length(zmarker)));
                        
            c={'r','g'};

            %plot only from HS to HS
            figure(4); subplot(2,4,i);
            hold on
            D=zmarker(HSlocs(1)-1:HSlocs(end)-1);
            V=zfootVel(HSlocs(1): HSlocs(end));
            plot(D,V,char(c(k)))
            title(['velocity-position plot of midfoot' ' ' P(1:4) '-' P(6:7)])
            xlabel('position (mm)');
            ylabel('velocitiy');
            
        end 
    end
end



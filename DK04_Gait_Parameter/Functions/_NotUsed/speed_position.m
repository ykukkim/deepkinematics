function speed_position(GaitVariability, VideoData, vis)
% speed against position plot

%DIR_gt='R:\Projects\NCM\NCM_EXP\NCM_FAT\Gait\Matfiles_newstructure';
%open NAME.STATE.ALL

% open correct file for locations of the markers in DIR
% open correct path in gaitvariability 

if vis %only run function if vis is true
    
% %define state 
% if state == 1
%     S='Pre_gt';
%     T=['velocity-position plot of midfoot' ' ' P(1:4) '-' P(6:7) ' ' S(1:3) '-' S(5:6)]; %title of plot
%     c='g' %color of plot
% elseif state == 2
%     S='Post_gt' ;
%     T=['velocity-position plot of midfoot' ' ' P(1:4) '-' P(6:7) ' ' S(1:4) '-' S(6:7)];
%     c='r'
% end 

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

            heel={'LTHL' 'RTHL'};
            toe={'LTFB' 'RTFB'};
    
    
            for j=1:2;
            %get heel and toe positions 
            comd = ['zheel=VideoData.',P,'.',S,'.',file,'.',char(heel(j)),'.Values.z_coord;'];
            eval(comd);
            comd = ['ztoe=VideoData.',P,'.',S,'.',file,'.',char(toe(j)),'.Values.z_coord;'];
            eval(comd);

       
            if j==1
                HSlocs=GaitVariability.(genvarname(P)).(genvarname(S)).(genvarname(char(file))).HSleftlocs;
            elseif j==2
                HSlocs=GaitVariability.(genvarname(P)).(genvarname(S)).(genvarname(char(file))).HSrightlocs;
            end

            % Estimated center of the foot
            zCoordfootcentre = 1/2 * (zheel + ztoe);
            zfootVel = ... % Compute velocity of the foot
                diff(zCoordfootcentre) ./ transpose(diff(1:length(zCoordfootcentre)));
            
            c={'r','g'};

            %plot only from HS to HS
            figure(4); subplot(2,4,i);
            hold on
            D=zCoordfootcentre(HSlocs(1)-1:HSlocs(end)-1);
            V=zfootVel(HSlocs(1): HSlocs(end));
            plot(D,V,char(c(k)))
            title(['velocity-position plot of midfoot' ' ' P(1:4) '-' P(6:7)])
            xlabel('position (mm)');
            ylabel('velocitiy');
            end
        end 
    end
end
end
return
end

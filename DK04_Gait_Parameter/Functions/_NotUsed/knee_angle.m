%plot position shank against postion thigh

names=fieldnames(VideoData);
figure();
for i=1:length(names)
    name=char(names(i));
    
    comd = ['states=fieldnames(VideoData.',name,');'];
    eval(comd)
    
    for k=1:length(states)
        state=char(states(k));
        
        comd = ['tracks=fieldnames(VideoData.',name,'.',state,');'];
        eval(comd)
        
        shank=[];
        thigh=[];
        for l=1:length(tracks)
            track=char(tracks(l));
            
            comd = ['shank=[shank VideoData.',name,'.',state,'.',track,'.LSLL.Values.z_coord];'];
            eval(comd);
            
            comd = ['thigh=[thigh VideoData.',name,'.',state,'.',track,'.LSUL.Values.z_coord];'];
            eval(comd);
        end
        
        hold on 
        c={'r','b'};
        subplot(2,4,i)
        plot(shank,thigh,char(c(k)));
        ylabel('thigh position [mm]')
        xlabel('shank position [mm]')
        title([name(1:4),'-',name(6:7)])
    end
end

        
        
        
            

% shank=VideoData.FEMA_88.Post_gt.x01.LSLL.Values.z_coord;
% thigh=VideoData.FEMA_88.Post_gt.x01.LSUL.Values.z_coord;
% 
% figure;
% plot(shank,thigh)
% ylabel('thigh')
% xlabel('shank')
% 
% figure
% hold on 
% plot(shank,'r')
% plot(thigh,'b')
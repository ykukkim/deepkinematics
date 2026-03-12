%check HS and TO
close all

name='HARA_92';
state='Post_gt';

comd = ['No=fieldnames(VideoData.',name,'.',state,');'];
eval(comd);

for i=1:length(No); 
    %left--------------------------------------------------------------------------
    Track=char(No(i));
    
    %HS
    comd = ['VDleftHS=VideoData.',name,'.',state,'.',Track,'.LTHL.Values.z_coord;'];
    eval(comd);
    comd = ['HSleftlocs=GaitVariability.',name,'.',state,'.',Track,'.HSleftlocs;'];
    eval(comd);
    
    %TO
    comd = ['VDleftTO=VideoData.',name,'.',state,'.',Track,'.LTFB.Values.z_coord;'];
    eval(comd);
    comd = ['TOleftlocs=GaitVariability.',name,'.',state,'.',Track,'.TOleftlocs;'];
    eval(comd);
           
    figure;
    hold on;
    plot(VDleftHS,'b');
    plot(VDleftTO,'g');
    plot(HSleftlocs, VDleftHS(HSleftlocs), 'or')
    plot(TOleftlocs, VDleftTO(TOleftlocs), 'sr')
    legend('VDleftHeel','VDleftToe','HS','TO');

    grid on;
    title(['leftfoot LTHL and LTFB marker ',name,' ',state,' ',Track]);

    xlabel('Time (s)');
    ylabel('Distance (mm)');
    
    %right------------------------------------------------------------------------------
    %HS
    comd = ['VDrightHS=VideoData.',name,'.',state,'.',Track,'.RTHL.Values.z_coord;'];
    eval(comd);
    comd = ['HSrightlocs=GaitVariability.',name,'.',state,'.',Track,'.HSrightlocs;'];
    eval(comd);
    
    %TO
    comd = ['VDrightTO=VideoData.',name,'.',state,'.',Track,'.RTFB.Values.z_coord;'];
    eval(comd);
    comd = ['TOrightlocs=GaitVariability.',name,'.',state,'.',Track,'.TOrightlocs;'];
    eval(comd);
    
    figure;
    hold on;
    plot(VDrightHS,'b');
    plot(VDrightTO,'g');
    plot(HSrightlocs, VDrightHS(HSrightlocs), 'or')
    plot(TOrightlocs, VDrightTO(TOrightlocs), 'sr')
    legend('VDrightHeel','VDrightToe','HS','TO');

    grid on;
    title(['rightfoot RTHL and RTFB marker',name,' ',state,' ',Track]);

    xlabel('Time (s)');
    ylabel('Distance (mm)');
    
end
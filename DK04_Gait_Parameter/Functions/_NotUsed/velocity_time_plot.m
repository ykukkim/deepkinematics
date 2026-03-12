%plot velocity against stride time variability

names=fieldnames(GaitVariability);

for j=1:length(names);
    name=char(names(j));

    comd = ['track_pre=fieldnames(GaitVariability.', name, '.Pre_gt);'];
    eval(comd);

    %pre
    for i=1:length(track_pre)-1

        comd = ['VelPre=GaitVariability.', name, '.Pre_gt.',char(track_pre(i)), '.velocity;'];
        eval(comd);
        comd = ['NVelPre=GaitVariability.', name, '.Pre_gt.',char(track_pre(i)), '.V_dimmLess;'];
        eval(comd);

        comd = ['StrideTimePreL=GaitVariability.', name, '.Pre_gt.' ,char(track_pre(i)), '.durationGaitCycleL;'];
        eval(comd);
        comd = ['StrideTimePreR=GaitVariability.', name, '.Pre_gt.' ,char(track_pre(i)), '.durationGaitCycleR;'];
        eval(comd);
        
        %make lists of VelPre with same length as StrideTimePreL
        VL=zeros(length(StrideTimePreL),1);
        for x=1:length(StrideTimePreL)
            VL(x)=VelPre;
        end

        VR=zeros(length(StrideTimePreR),1);
        for x=1:length(StrideTimePreR)
            VR(x)=VelPre;
        end

        figure(7)
        hold on
        scatter(VL,StrideTimePreL,'c');
        scatter(VR,StrideTimePreR,'b');
        
        %same for normalized velocity
        nVL=zeros(length(StrideTimePreL),1);
        for x=1:length(StrideTimePreL)
            nVL(x)=NVelPre;
        end

        nVR=zeros(length(StrideTimePreR),1);
        for x=1:length(StrideTimePreR)
            nVR(x)=NVelPre;
        end
        
        figure(8)
        hold on
        scatter(nVL,StrideTimePreL,'c');
        scatter(nVR,StrideTimePreR,'b');

    end

    comd = ['track_post=fieldnames(GaitVariability.', name, '.Post_gt);'];
    eval(comd);

    
    %post
    for k=1:length(track_post)-1
  
        comd = ['VelPost=GaitVariability.', name, '.Post_gt.', char(track_post(k)), '.velocity;'];
        eval(comd);
        comd = ['NVelPost=GaitVariability.', name, '.Post_gt.' ,char(track_post(k)), '.V_dimmLess;'];
        eval(comd);

        comd = ['StrideTimePostL=GaitVariability.', name, '.Post_gt.' ,char(track_post(k)), '.durationGaitCycleL;'];
        eval(comd);
        comd = ['StrideTimePostR=GaitVariability.', name, '.Post_gt.' ,char(track_post(k)), '.durationGaitCycleR;'];
        eval(comd);

        VL=zeros(length(StrideTimePostL),1);
        for x=1:length(StrideTimePostL)
            VL(x)=VelPost;
        end

        VR=zeros(length(StrideTimePostR),1);
        for x=1:length(StrideTimePostR)
            VR(x)=VelPost;
        end

        figure(7)
        hold on
        scatter(VL,StrideTimePostL,'r');
        scatter(VR,StrideTimePostR,'m');
        legend('PreL','PreR','PostL','PostR');
        xlabel('velocity')
        ylabel('stide duration')
        title('Velocity / Stride duration plot')
        
        %same for normalized velocity
        nVL=zeros(length(StrideTimePostL),1);
        for x=1:length(StrideTimePostL)
            nVL(x)=NVelPost;
        end

        nVR=zeros(length(StrideTimePostR),1);
        for x=1:length(StrideTimePostR)
            nVR(x)=NVelPost;
        end
        
        figure(8)
        hold on
        scatter(nVL,StrideTimePostL,'r');
        scatter(nVR,StrideTimePostR,'m');
        legend('PreL','PreR','PostL','PostR');
        xlabel('normalized velocity')
        ylabel('stide duration')
        title('Velocity / Stride duration plot')


    end

end
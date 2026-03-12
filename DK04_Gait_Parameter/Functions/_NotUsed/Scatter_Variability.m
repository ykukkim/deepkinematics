%variability vs velocity
%Plot variability or SD of every participant in one graph against velocity

%velocity should be the mean velocity of each participant 


names=fieldnames(GaitVariability);
variable='stance'; %stride swing stance
value='CV'; %CV SD meanSD


%pre
for j=1:length(names);
    name=char(names(j));
    
    %velocity
    comd = ['V=mean(GaitVariability.', name, '.Pre_gt.Total.velocity);'];
    eval(comd);
    
    comd = ['prcvStrideL=CV.', name, '.Pre_gt.' ,variable, 'L.',value,';'];
    eval(comd);
    comd = ['prcvStrideR=CV.', name, '.Pre_gt.' ,variable, 'R.',value,';'];
    eval(comd);
    
    VL=zeros(length(prcvStrideL),1);
    for x=1:length(prcvStrideL)
        VL(x)=V;
    end

    VR=zeros(length(prcvStrideR),1);
    for x=1:length(prcvStrideR)
        VR(x)=V;
    end
    
    %plot
    figure(9)
    subplot(1,2,1);
    hold on 
    scatter(VL,prcvStrideL,'b')
    scatter(VR,prcvStrideR,'c')
    
    %normalized velocity
    comd = ['nV=mean(GaitVariability.', name, '.Pre_gt.Total.V_dimmLess);'];
    eval(comd);
    
    nVL=zeros(length(prcvStrideL),1);
    for x=1:length(prcvStrideL)
        nVL(x)=nV;
    end

    nVR=zeros(length(prcvStrideR),1);
    for x=1:length(prcvStrideR)
        nVR(x)=nV;
    end
    
    %plot
    figure(9)
    subplot(1,2,2);
    hold on 
    scatter(nVL,prcvStrideL,'b')
    scatter(nVR,prcvStrideR,'c')
end

%post
for j=1:length(names);
    name=char(names(j));
    
    %velocity
    comd = ['V=mean(GaitVariability.', name, '.Post_gt.Total.velocity);'];
    eval(comd);
  
    comd = ['cvStrideL=CV.', name, '.Post_gt.' ,variable, 'L.',value,';'];
    eval(comd);   
    comd = ['cvStrideR=CV.', name, '.Post_gt.' ,variable, 'R.',value,';'];
    eval(comd);
    
    VL=zeros(length(cvStrideL),1);
    for x=1:length(cvStrideL)
        VL(x)=V;
    end

    VR=zeros(length(cvStrideR),1);
    for x=1:length(cvStrideR)
        VR(x)=V;
    end
    
    figure(9)
    subplot(1,2,1);
    hold on 
    scatter(VL,cvStrideL,'r')
    scatter(VR,cvStrideR,'m')
    %legend('PreL','PreR','PostL','PostR','Orientation','horizontal');
    xlabel('velocity')
    ylabel([value,' ' ,variable, ' duration'])
	title(['Velocity / ',value, ' ', variable,' duration plot'])

    
    %normalized velocity
    comd = ['nV=mean(GaitVariability.', name, '.Post_gt.Total.V_dimmLess);'];
    eval(comd);
    
    nVL=zeros(length(cvStrideL),1);
    for x=1:length(cvStrideL)
        nVL(x)=nV;
    end

    nVR=zeros(length(cvStrideR),1);
    for x=1:length(cvStrideR)
        nVR(x)=nV;
    end
    
    %plot
    figure(9)
    subplot(1,2,2)
    hold on 
    scatter(nVL,cvStrideL,'r')
    scatter(nVR,cvStrideR,'m')
	%legend('PreL','PreR','PostL','PostR','Orientation','horizontal');
	xlabel('normalized velocity')
    ylabel([value,' ',variable,' duration'])
	title(['Normalized Velocity / ',value,' ',variable ,' duration plot'])
end
legend('PreL','PreR','PostL','PostR','Orientation','horizontal');
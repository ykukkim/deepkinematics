%scatter variability with CV per trial

clear('newCV');
names=fieldnames(GaitVariability);
Vpre=[];
Vpost=[];
nVpre=[];
nVpost=[];
SDpre=[];
SDpost=[];
CVpre=[];
CVpost=[];

for i=1:length(names)
    name=char(names(i));
    
    comd = ['states=fieldnames(GaitVariability.',name,');'];
    eval(comd)
    
    for k=1:length(states)-1
        state=char(states(k));
        
        comd = ['tracks=fieldnames(GaitVariability.',name,'.',state,');'];
        eval(comd)
        
        count=1;
        for l=1:2:length(tracks)-1
            track=char(tracks(l));
            track2=char(tracks(l+1));
            
            comd = ['V=[GaitVariability.',name,'.',state,'.',track,'.velocity ',...
                'GaitVariability.',name,'.',state,'.',track2,'.velocity];'];
            eval(comd);
            comd = ['nV=[GaitVariability.',name,'.',state,'.',track,'.V_dimmLess ',...
                'GaitVariability.',name,'.',state,'.',track2,'.V_dimmLess];'];
            eval(comd);
            comd = ['GaitL=[GaitVariability.',name,'.',state,'.',track,'.durationGaitCycleL ',...
                'GaitVariability.',name,'.',state,'.',track2,'.durationGaitCycleL];'];
            eval(comd);
            comd = ['GaitR=[GaitVariability.',name,'.',state,'.',track,'.durationGaitCycleR ',...
                'GaitVariability.',name,'.',state,'.',track2,'.durationGaitCycleR];'];
            eval(comd);
            
            %left
            mL=mean(GaitL);
            SDL=std(GaitL);
            comd = ['newCV.',name,'.',state,'.CV(count,1)=SDL/mL;'];
            eval(comd);
            comd = ['newCV.',name,'.',state,'.SD(count,1)=SDR;'];
            eval(comd);
            
            %right
            mR=mean(GaitR);
            SDR=std(GaitR);
            comd = ['newCV.',name,'.',state,'.CV(count,2)=SDR/mR;'];
            eval(comd);
            comd = ['newCV.',name,'.',state,'.SD(count,2)=SDR;'];
            eval(comd);
            
            %velocity
            comd = ['newCV.',name,'.',state,'.vel(count,1)=mean(V);'];
            eval(comd);
            comd = ['newCV.',name,'.',state,'.vel(count,2)=mean(nV);'];
            eval(comd);
            
            count=count+1;
                        
        end
    end
    
    comd = ['Vpre=[Vpre ;newCV.',name,'.Pre_gt.vel];'];
    eval(comd);
    comd = ['Vpost=[Vpost ;newCV.',name,'.Post_gt.vel];'];
    eval(comd);
    comd = ['SDpre=[SDpre ;newCV.',name,'.Pre_gt.SD];'];
    eval(comd);
    comd = ['SDpost=[SDpost ;newCV.',name,'.Post_gt.SD];'];
    eval(comd);
    comd = ['CVpre=[CVpre ;newCV.',name,'.Pre_gt.CV];'];
    eval(comd);
    comd = ['CVpost=[CVpost ;newCV.',name,'.Post_gt.CV];'];
    eval(comd);
end
        
newCV.total.Pre_gt.V=Vpre;
newCV.total.Post_gt.V=Vpost;
newCV.total.Pre_gt.SD=SDpre;
newCV.total.Post_gt.SD=SDpost;
newCV.total.Pre_gt.CV=CVpre;
newCV.total.Post_gt.CV=CVpost;

figure()

subplot(2,2,1)
hold on 
scatter(newCV.total.Pre_gt.V(:,1), newCV.total.Pre_gt.SD(:,1),'c')
scatter(newCV.total.Pre_gt.V(:,1), newCV.total.Pre_gt.SD(:,2),'b')
scatter(newCV.total.Post_gt.V(:,1), newCV.total.Post_gt.SD(:,1),'r')
scatter(newCV.total.Post_gt.V(:,1), newCV.total.Post_gt.SD(:,2),'m')
title('V/SD')
legend('PreL','PreR','PostL','PostR');
xlabel('Velocity (m/s) per vicon track')
ylabel('SD per vicon track')

subplot(2,2,2)
hold on 
scatter(newCV.total.Pre_gt.V(:,2), newCV.total.Pre_gt.SD(:,1),'c')
scatter(newCV.total.Pre_gt.V(:,2), newCV.total.Pre_gt.SD(:,2),'b')
scatter(newCV.total.Post_gt.V(:,2), newCV.total.Post_gt.SD(:,1),'r')
scatter(newCV.total.Post_gt.V(:,2), newCV.total.Post_gt.SD(:,2),'m')
title('nV/SD')
legend('PreL','PreR','PostL','PostR');
xlabel('normalized Velocity (-) per vicon track')
ylabel('SD per vicon track')

subplot(2,2,3)
hold on 
scatter(newCV.total.Pre_gt.V(:,1), newCV.total.Pre_gt.CV(:,1),'c')
scatter(newCV.total.Pre_gt.V(:,1), newCV.total.Pre_gt.CV(:,2),'b')
scatter(newCV.total.Post_gt.V(:,1), newCV.total.Post_gt.CV(:,1),'r')
scatter(newCV.total.Post_gt.V(:,1), newCV.total.Post_gt.CV(:,2),'m')
title('V/CV')
legend('PreL','PreR','PostL','PostR');
xlabel('Velocity (m/s) per vicon track')
ylabel('CV per vicon track')

subplot(2,2,4)
hold on 
scatter(newCV.total.Pre_gt.V(:,2), newCV.total.Pre_gt.CV(:,1),'c')
scatter(newCV.total.Pre_gt.V(:,2), newCV.total.Pre_gt.CV(:,2),'b')
scatter(newCV.total.Post_gt.V(:,2), newCV.total.Post_gt.CV(:,1),'r')
scatter(newCV.total.Post_gt.V(:,2), newCV.total.Post_gt.CV(:,2),'m')
title('nV/CV')
legend('PreL','PreR','PostL','PostR');
xlabel('normalized Velocity (-) per vicon track')
ylabel('CV per vicon track')

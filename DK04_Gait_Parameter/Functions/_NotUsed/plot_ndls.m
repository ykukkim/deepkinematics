%plot normalized double support time

names=fieldnames(GaitVariability);
figure();
suptitle('normalized double limb support Left')

%Left
for i=1:length(names)
    name=char(names(i));
    
    comd = ['states=fieldnames(GaitVariability.',name,');'];
    eval(comd)
    
    for k=1:length(states)-1
        state=char(states(k));
        
        comd = ['dataL=GaitVariability.',name,'.',state,'.Total.ndlsL;'];
        eval(comd);
        
        c={'r','b'};
        subplot(2,4,i)
        hold on 
        plot(dataL,char(c(k)));
        title([name(1:4),'-',name(6:7)])
        xlabel('number of steps')
        ylabel('ndlsL (% of gaitcycle)')
                        
    end
    
    legend('post','pre')
end

figure();
suptitle('normalized double limb support Right')
%Right
for i=1:length(names)
    name=char(names(i));
    
    comd = ['states=fieldnames(GaitVariability.',name,');'];
    eval(comd)
    
    for k=1:length(states)-1
        state=char(states(k));
        
        comd = ['dataR=GaitVariability.',name,'.',state,'.Total.ndlsR;'];
        eval(comd);
        
        c={'r','b'};
        subplot(2,4,i)
        hold on 
        plot(dataR,char(c(k)));
        title([name(1:4),'-',name(6:7)])
        xlabel('number of steps')
        ylabel('ndlsR (% of gaitcycle)')
    end
    
    legend('post','pre')
end
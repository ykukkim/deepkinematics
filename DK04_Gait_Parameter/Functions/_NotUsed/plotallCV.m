%plot all CV 

names=fieldnames(CV);
var='MFC'; %swing stance stride ndls MFC
value='CV' ; %CV SD meanSD


%define matrix assumed no dataset has more then 40 values
comd = ['pre',var,'R=nan(40,7);','pre',var,'L=nan(40,7);','post',var,'L=nan(40,7);','post',var,'R=nan(40,7);'];
eval(comd);

figure();

for i=1:length(names) %for every participant
    P=char(names(i));
    
    comd = ['pre',var,'R(1:(length(CV.',P,'.Pre_gt.',var,'R.',value,')),i)=CV.',P,'.Pre_gt.',var,'R.',value,';'];
    eval(comd)
    
    comd = ['pre',var,'L(1:(length(CV.',P,'.Pre_gt.',var,'L.',value,')),i)=CV.',P,'.Pre_gt.',var,'L.',value,';'];
    eval(comd)
    
    comd = ['post',var,'L(1:(length(CV.',P,'.Post_gt.',var,'L.',value,')),i)=CV.',P,'.Post_gt.',var,'L.',value,';'];
    eval(comd)
    
    comd = ['post',var,'R(1:(length(CV.',P,'.Post_gt.',var,'R.',value,')),i)=CV.',P,'.Post_gt.',var,'R.',value,';'];
    eval(comd)
    
    %plot 6 plots for the CV for every participant
    subplot(2,4,i) 
    hold on 
    plot(eval(['pre',var,'R(:,i)']),'b');
    plot(eval(['pre',var,'L(:,i)']),'b');
    plot(eval(['post',var,'R(:,i)']),'r');
    plot(eval(['post',var,'L(:,i)']),'r');
    title([P(1:4),'-',P(6:7),' ',value,' ', var]);
    ylabel(['',value,'', var, '(%)']);
    xlabel('numbers of steps');
    legend('preR','preL','postR','postL');
    
end 

%plot all participants in one figure 
figure 
hold on 
plot(eval(['pre',var,'R']),'b');
plot(eval(['pre',var,'L']),'b');
plot(eval(['post',var,'R']),'r');
plot(eval(['post',var,'L']),'r');
title([value,' ', var]);
ylabel([value,'', var, '(%)']);
xlabel(['numbers of steps']);
legend('preR','preL','postR','postL');


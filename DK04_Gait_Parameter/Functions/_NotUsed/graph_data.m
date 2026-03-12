function graph_data(GaitSummary, Name, exercise, state, vis )
%plot spatial parameters

if vis;

P=Name;
E = exercise;

%HS and TO Pre and Post
figure(1)
subplot(2,2,1)
plot(GaitSummary.(genvarname(P)).(genvarname(E)).Pre_gt.Total.GaitEvents.HSleft,'r'); hold on ; plot(GaitSummary.(genvarname(P)).(genvarname(E)).Pre_gt.Total.GaitEvents.HSright,'b')
title('Pre-gt HS')
legend('left','right')
subplot(2,2,2)
plot(GaitSummary.(genvarname(P)).(genvarname(E)).Pre_gt.Total.GaitEvents.TOleft,'r'); hold on ; plot(GaitSummary.(genvarname(P)).(genvarname(E)).Pre_gt.Total.GaitEvents.TOright,'b')
title('Pre-gt TO')
legend('left','right')
subplot(2,2,3)
plot(GaitSummary.(genvarname(P)).(genvarname(E)).Post_gt.Total.GaitEvents.HSleft,'r'); hold on ; plot(GaitSummary.(genvarname(P)).(genvarname(E)).Post_gt.Total.GaitEvents.HSright,'b')
title('Post-gt HS')
legend('left','right')
subplot(2,2,4)
plot(GaitSummary.(genvarname(P)).(genvarname(E)).Post_gt.Total.GaitEvents.TOleft,'r'); hold on ; plot(GaitSummary.(genvarname(P)).(genvarname(E)).Post_gt.Total.GaitEvents.TOright,'b')
title('Post-gt TO')
legend('left','right')

T=[P(1:4) '-' P(6:7) ': Exercise - ' E 'check HS and TO'];
suptitle(T)

%all other parameters
if state == 1
    S='Pre_gt';
    T=['spatial parameters :' P(1:4) '-' P(6:7) ': Exercise - ' E ';' S(1:3) '-' S(5:6)];
elseif state == 2
    S='Post_gt' ;
    T=['spatial parameters :' P(1:4) '-' P(6:7) ': Exercise - ' E ';' S(1:4) '-' S(6:7)];
end 

figure()

subplot(3,3,1)
hold on
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.midSwOccTimeR,'b')
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.midSwOccTimeL,'r')
title('midSwOccTime')
legend('R','L')

subplot(3,3,2)
hold on
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.midStOccTimeR,'b')
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.midStOccTimeL,'r')
title('midStOccTime')
legend('R','L')

subplot(3,3,3)
hold on 
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.durationGaitCycleR,'b')
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.durationGaitCycleL,'r')
title('durationGaitCycle')
legend('R','L')

subplot(3,3,4)
hold on 
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.durationStancePhR,'b')
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.durationStancePhL,'r')
title('durationStancePh')
legend('R','L')

subplot(3,3,5)
hold on 
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.durationSwingPhR,'b')
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.durationSwingPhL,'r')
title('durationSwingPh')
legend('R','L')

subplot(3,3,6)
hold on 
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.strideLengthR,'b')
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.strideLengthL,'r')
title('strideLength')
legend('R','L')

subplot(3,3,7)
hold on 
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.stepLengthR,'b')
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.stepLengthL,'r')
title('stepLength')
legend('R','L')

subplot(3,3,8)
hold on 
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.dlsR,'b')
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.dlsL,'r')
legend('dlsR','dlsL')
title('DST')
ylabel('time [s]')
xlabel('number of steps')

subplot(3,3,9)
hold on 
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.MTCright,'b')
plot(GaitSummary.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.MTCleft,'r')
legend('MTCright','MTCleft')
title('MTC')
ylabel('position [mm]')
xlabel('number of steps')

% subplot(3,3,9)
% hold on 
% plot(GaitVariability.(genvarname(P)).(genvarname(E)).(genvarname(S)).Total.GaitParameters.mean_stepWidth)
% title('mean stepWidth')
% xlabel('length [mm?]')
% ylabel('number of tracks')


suptitle(T)

%HS and TO Pre and Post
% figure()
% subplot(2,2,1)
% plot(GaitSummary.(genvarname(P)).(genvarname(E)).Pre_gt.Total.GaitCycles.mediolateral.LTHL, 'r'); hold on ; plot(GaitSummary.(genvarname(P)).(genvarname(E)).Pre_gt.Total.GaitCycles.mediolateral.RTHL, 'b')
% title('Pre-gt heelmarker mediolateral direction')
% legend('left','right')
% subplot(2,2,2)
% plot(GaitSummary.(genvarname(P)).(genvarname(E)).Pre_gt.Total.GaitCycles.mediolateral.LTTO, 'r'); hold on ; plot(GaitSummary.(genvarname(P)).(genvarname(E)).Pre_gt.Total.GaitCycles.mediolateral.RTTO, 'b')
% title('Pre-gt TO')
% legend('left','right')
% subplot(2,2,3)
% plot(GaitSummary.(genvarname(P)).(genvarname(E)).Post_gt.Total.GaitCycles.mediolateral.LTHL, 'r'); hold on ; plot(GaitSummary.(genvarname(P)).(genvarname(E)).Post_gt.Total.GaitCycles.mediolateral.RTHL, 'b')
% title('Post-gt HS')
% legend('left','right')
% subplot(2,2,4)
% plot(GaitSummary.(genvarname(P)).(genvarname(E)).Post_gt.Total.GaitCycles.mediolateral.LTTO, 'r'); hold on ; plot(GaitSummary.(genvarname(P)).(genvarname(E)).Post_gt.Total.GaitCycles.mediolateral.LTTO, 'b')
% title('Post-gt TO')
% legend('left','right')
% 
% T=[P(1:4) '-' P(6:7) ': Exercise - ' E ';' 'Gait Cycles'];
% suptitle(T)

end

return
end
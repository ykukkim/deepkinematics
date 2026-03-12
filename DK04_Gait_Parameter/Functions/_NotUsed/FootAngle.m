function [HSangle,FF] = FootAngle(HEE,TO3,HSlocs,SamplingRate)
%UNTITLED2 Summary of this function goes here - [AngularVel, HSangle] = FoofAngle(HEE,TO3,HS)
%   Detailed explanation goes here
n=length(HEE.z_coord);
TETA=zeros(n,1);
for i=1:n
    %v1=[TO3.x_coord(i-1)-HEE.x_coord(i-1),TO3.y_coord(i-1)-HEE.y_coord(i-1),TO3.z_coord(i-1)-HEE.z_coord(i-1)];
    v1=[0 0 -1];
    v2=[TO3.x_coord(i)-HEE.x_coord(i),TO3.y_coord(i)-HEE.y_coord(i),TO3.z_coord(i)-HEE.z_coord(i)];
    v2=v2/norm(v2);
    TETA(i)=acosd(dot(v1,v2));
%     if dTETA(i)>90
%         dTETA(i)=dTETA(i)-180;
%     end
    
end
% now detecting FF moments
% Calculate filtered data
[b3_5, a3_5] = butter(4, 3.5/(SamplingRate/2), 'low');
[b28, a28] = butter(4, 28/(SamplingRate/2), 'low');

TETA= filtfilt(b28, a28, TETA);
gyro = [diff(TETA); 0];
gyro3_5Hz = filter(b3_5, a3_5, gyro);
gyro_dt = [diff(gyro3_5Hz); 0];

foundLocalMax = false;
nHC = 0;
nFF = 0; % 


for nsamples = 2:n
            
    % Identify the local maximum in the gyro curve (-> swing phase)
    if ~foundLocalMax&&(gyro_dt(nsamples-1) > 0) && (gyro_dt(nsamples) <= 0)
            %Local maximum
            if (gyro3_5Hz(nsamples) > 1)
                %This is the local maximum of the swing phase
                foundLocalMax = true;
                lastMax = nsamples; % Max the swing phase for HS identification
            end
    end
        
    %Then identify HC then FF   !!!!! an "or" is added to the next line for
                                    %test
    if  foundLocalMax && ((gyro_dt(nsamples-1) > 0) && (gyro_dt(nsamples) <= 0)&& (abs(gyro3_5Hz(nsamples))<0.9))
        %Search minimum after the local maximum in the "normal"
        %28Hz-Gyro-Kurve (--> HC)
        tempmin = gyro(lastMax,1);

        nHC = nHC+1;
        HC(nHC) = lastMax;
        for i=lastMax:nsamples-1

            if gyro(i,1) < tempmin
                tempmin = gyro(i,1);
                HC(nHC) = i;             
            end
        end
        
        %Now FF
        tempmax = gyro(HC(nHC),1);
        nFF = nFF+1;
        FF(nFF) = HC(nHC);
        for i=HC(nHC): nsamples
            % finds the maximum that correspond to FF
            if (gyro(i,1) > tempmax)
                tempmax = gyro(i,1);
                FF(nFF) = i;
            end
        end
        
        foundLocalMax=false; %!!!test
       
        
       
    end 
end

origin=mean(TETA(FF));
TETA=TETA-origin;

HSangle=TETA(HSlocs);

%To check if a stride is missed in detecting the FF and if so filling it
%with NaN
while FF(1)<HSlocs(1)
    FF(1)=[];
end

if FF(1)-HSlocs(1)>200
    FF=[NaN,FF];
end


end


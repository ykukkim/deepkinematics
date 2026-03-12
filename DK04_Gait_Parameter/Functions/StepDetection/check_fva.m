% until fva line 108
plot(zheel, 'b')
hold on
plot(ztoe, 'r')
hold on
plot(zCoordfootcentre, 'g')
hold on
plot(zfootVel, 'black')

% until fva line 112 (all)/ 114 (only peaks that are >1)
figure
plot(zfootVel)
hold on
plot(TOlocs,TOpks, 'ob')

% until fva line 117 (all)/
figure
plot(-zfootVel, 'g')
hold on
plot(HSlocs,HSpks, 'og')

% until fva line 120 (only peaks that are >1)
figure
plot(zfootVel, 'g')
hold on
plot(HSlocs,HSpks, 'og')

% until fva line 127: 
figure
plot(zheel, 'r')
hold on
plot(HSlocs, HSpks, 'or')

% until fva line 174: to check whether all the HS and TO have been
% recognized
figure
plot(zfootVel, 'black')
hold on
plot(HSlocs, HSpks, 'or')
hold on
plot(TOlocs, TOpks, 'sr')


function [DST,DST_HrTl,DST_HlTr]=Get_dst(HSleftlocs, HSrightlocs,...
    TOleftlocs, TOrightlocs,...
    SF);

%calculates double support time (DST)
%defined as period between HS of one foot and TO of the ohter foot. 
%programm only works if started with a HS --> earlier defined
i=1;

%% DST right Heel to left Toe
TL=TOleftlocs;
HR=HSrightlocs;

j=1;
start=j;
while TL(i)>HR(j);
    j=j+1;
    start=j-1;
end

HR=HR(start:end);
L1=min(length(HR),length(TL));
HR=HR(1:L1);
TL=TL(1:L1);

%substract the HS from TO to get the datapoints for DST
DST_HrTl=TL-HR;
DST_HrTl=DST_HrTl/SF;

%% DST left Heel to right Toe
TR=TOrightlocs;
HL=HSleftlocs;
j=1;
start=j;
while TR(i)>HL(j);
    j=j+1;
    start=j-1;
end

HL=HL(start:end);
L2=min(length(HL),length(TR));
HL=HL(1:L2);
TR=TR(1:L2);

%substract the HS from TO to get the datapoints for DST
DST_HlTr=TR-HL;
DST_HlTr=DST_HlTr/SF;


%% DST both feet
listHS=sort([HSleftlocs,HSrightlocs]);
listTO=sort([TOleftlocs,TOrightlocs]);

%make sure there are equaly amount of TO and HS, starting with HS

k=1;
start=k;
while listTO(i)>listHS(k);
    k=k+1;
    start=k-1;
end

listHS=listHS(start:end);
L=min(length(listHS),length(listTO));
listHS=listHS(1:L);
listTO=listTO(1:L);

%substract the HS from TO to get the datapoints for DST
DST=listTO-listHS;
DST=DST/SF;

return
end

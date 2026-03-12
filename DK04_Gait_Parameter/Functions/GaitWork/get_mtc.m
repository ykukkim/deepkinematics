function [MTC, MTClocs]=get_mtc(VD,TOlocs,HSlocs,toemarker)
%MFC foot ground clearance


comd = ['ToeMarker=VD.',toemarker,'.Values.z_coord;'];
eval(comd);

%cut signal from TO till HS and calculate minumum
MTClocs=[];
MTC=[];
temp_length =  min(length(TOlocs), length(HSlocs));
for i=1:temp_length-2
    start=TOlocs(i);
    stop=HSlocs(i+1);
        
    marker=(ToeMarker(start:stop));
%     figure()
%     plot(marker);
    
    [val,loc]=findpeaks(double(-marker));
    [val,x]=max(val);
    loc=loc(x);
   
    loc=loc+start-1;
    
    value=-val;
    
    MTC=[MTC value];
    MTClocs=[MTClocs loc];
    
end

MG=5;
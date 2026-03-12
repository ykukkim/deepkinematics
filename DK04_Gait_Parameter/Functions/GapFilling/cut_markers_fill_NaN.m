% filename = uigetfile('Z:\Projects\NCM\NCM_EXP\NCM_FAT\Running_Vicon\Codes');
% markerlist={'LTHL','RTHL','LTFB','RTFB'};
%     matrix=[VD.LTHL.Values.z_coord, VD.RTHL.Values.z_coord,...
%         VD.LTFB.Values.z_coord,VD.RTFB.Values.z_coord ];
%     
% A = table(VD.LTHL.Values.z_coord,VD.RTHL.Values.z_coord,...
%     VD.LTFB.Values.z_coord,VD.RTFB.Values.z_coord);
% TF = ismissing(A);
% 
% ind = findall(A, 'NaN');

% create an example array
A = randi(100,10,5);
A(A < 5) = NaN;
B = A;

% to find the positions of the values next to the NaNs
ind=find(isnan(B));
otherInd=setdiff(1:numel(B(:)),ind);
for i=1:size(ind,1)
   temp=otherInd(otherInd<ind(i));
   prevInd(i,1)=temp(end);
   temp=otherInd(otherInd>ind(i));
   nextInd(i,1)=temp(1);
end

allInd(1:2:2*length(prevInd))=prevInd;
allInd(2:2:2*length(prevInd))=nextInd;
fun=@(block_struct) mean(block_struct.data);
prevNextNums=B(allInd);
B
newB(ind) = blockproc(prevNextNums,[1 2],fun);

for j=1:length(B)
    isnan(B)= newB(ind);
end


% import matlab.io.hdfeos.*
% gfid = gd.open('grid.hdf');
% gridID = gd.attach(gfid,'PolarGrid');
% fillvalue = gd.getFillValue(gridID,'ice_temp');
% gd.detach(gridID);
% gd.close(gfid);
% fillvalue = getFillValue(gridID,fieldname)
% 
% 
% import matlab.io.hdfeos.*
% gfid = gd.open('grid.hdf');
% gridID = gd.attach(gfid,'PolarGrid');
% fillvalue = gd.getFillValue(gridID,'ice_temp');
% gd.detach(gridID);
% gd.close(gfid);
% fillvalue = getFillValue(gridID,fieldname)
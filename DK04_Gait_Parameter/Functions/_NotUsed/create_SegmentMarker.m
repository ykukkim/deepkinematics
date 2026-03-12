function VD=create_SegmentMarker(VD)

%segments to be created
MSBP={'SACR','RPSI','LPSI'}; %MarkerSegmentBackPelvis
MSFP={'RTMS','LTMS','RASI','LASI'}; %MarkerSegmentFrontPelvis
MSCP = {'RTMS','LTMS','RASI','LASI', 'RPSI','LPSI'}; % MarkerSegmentCenterPelvis


NewSegments={'MSBP', 'MSFP', 'MSCP'};

for i=1:length(NewSegments);
    marker = eval(char(NewSegments(i)));
    x=[]; y=[]; z=[];
    for j=1:length(marker)
        %for every marker look at the length en save all x y and z values
        %and take the middle value of those to calculate the centre of the
        %marker. 
        
        %check if maker exists
        if not(isfield(VD,char(marker(j))));
            continue
        else     
        
        comd = ['x=[x VD.', char(marker(j)), '.Values.x_coord];'];
        eval(comd);
        comd = ['y=[y VD.', char(marker(j)), '.Values.y_coord];'];
        eval(comd);
        comd = ['z=[z VD.', char(marker(j)), '.Values.z_coord];'];
        eval(comd);
        end
    end
    
    
    for k=1:length(x);
        comd = ['VD.', char(NewSegments(i)), '.Values.x_coord(k)=mean(x(k,:));'];
        eval(comd);
        comd = ['VD.', char(NewSegments(i)), '.Values.y_coord(k)=mean(y(k,:));'];
        eval(comd);
        comd = ['VD.', char(NewSegments(i)), '.Values.z_coord(k)=mean(z(k,:));'];
        eval(comd);
    end
        
end

VD=orderfields(VD);
return
end


        
% figure();
% hold on
% plot3(VD.RTMS.Values.x_coord,VD.RTMS.Values.y_coord, VD.RTMS.Values.z_coord,'b')
% plot3(VD.LTMS.Values.x_coord,VD.LTMS.Values.y_coord, VD.LTMS.Values.z_coord,'b')
% plot3(VD.RTAS.Values.x_coord,VD.RTAS.Values.y_coord, VD.RTAS.Values.z_coord,'b')
% plot3(VD.LTAS.Values.x_coord,VD.LTAS.Values.y_coord, VD.LTAS.Values.z_coord,'b')
% plot3(NewData.MSFP.x_coord,NewData.MSFP.y_coord,NewData.MSFP.z_coord,'r')
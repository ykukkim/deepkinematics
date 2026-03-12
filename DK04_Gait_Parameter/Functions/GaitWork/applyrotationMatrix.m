function [rot_Datax, rot_Datay, rot_Dataz] = applyrotationMatrix(datax, ...
    datay, dataz, direcString)

Mag = 1;
nomrkrs = fieldnames(datax);
for i = 1:length(nomrkrs)
    mrkn = char(nomrkrs(i));
    unrotDatax = datax.(genvarname(mrkn));
    unrotDatay = datay.(genvarname(mrkn));
    unrotDataz = dataz.(genvarname(mrkn));
    
    
    
    for j = 1:size(unrotDatax, 2)
        samplesforStride = (1:1:size(unrotDatax, 1))'; % changed!!
        fittoRotatex = polyfit(samplesforStride, unrotDatax(:,j), 1);
        fitdatax = fittoRotatex(1)*samplesforStride + fittoRotatex(2);
        
        fittoRotatey = polyfit(samplesforStride, unrotDatay(:,j), 1);
        fitdatay = fittoRotatey(1)*samplesforStride + fittoRotatey(2);
        
        angVecx = fitdatax(end) - fitdatax(1);
        angVecy = fitdatay(end) - fitdatay(1);
        
        angVec = [angVecx; angVecy; 0];
        %angVec1 = [angVecx angVecy 0];
        %axesVec = [0 1 0];
        angleFrontalplaneVertax(1,j) = subspace(angVec, [0;1;0]);
        
        %angleFrontalplaneVertax(1,j) = atan2(norm(cross(angVec1, axesVec)), dot(angVec1, axesVec));
        
        datatoRotate = [unrotDatax(:,j) unrotDatay(:,j) unrotDataz(:,j)].';
        if strcmp(direcString, 'Straight_return')
            if (fitdatax(end) <= fitdatax(1) && fitdatay(end) < fitdatay(1))
                angleFrontalplaneVertax(1,j) = angleFrontalplaneVertax(1,j);
                Mag = -1;
            elseif (fitdatax(end) > fitdatax(1) && fitdatay(end) < fitdatay(1))
                angleFrontalplaneVertax(1,j) = pi - angleFrontalplaneVertax(1,j);
                Mag = 1;
            end
        end
        
        rotationMatrixVertax = ...
            [cos(angleFrontalplaneVertax(1,j)) -sin(angleFrontalplaneVertax(1,j)) 0; ...
            sin(angleFrontalplaneVertax(1,j)) cos(angleFrontalplaneVertax(1,j)) 0; ...
            0   0   1];
        
        rotdataMatrixVertax = rotationMatrixVertax * datatoRotate;
        rotdataMatrixVertax = Mag * rotdataMatrixVertax;
        
        rot_Datax.(genvarname(mrkn))(:,j) = rotdataMatrixVertax(1,:) - mean(rotdataMatrixVertax(1,:));
        rot_Datay.(genvarname(mrkn))(:,j) = rotdataMatrixVertax(2,:) - mean(rotdataMatrixVertax(2,:));
        
        if rotdataMatrixVertax(3,1) < 0
            Mag = -1;
        else Mag = 1;
        end
        % the longitudinal direction is always positive...
        rot_Dataz.(genvarname(mrkn))(:,j) = Mag * rotdataMatrixVertax(3,:);
        %         end
    end
end
return;
end



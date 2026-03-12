function [sf_gyro, ef_gyro, sf_fva, ef_fva] = setFrameIndices(VD, Test, smpf)
    if contains(Test, "Norm_Pre") || contains(Test, "Norm_Post")
        if length(VD.IMU.ankle_L.acc) < 20000
            sf_gyro = 1; ef_gyro = length(VD.IMU.ankle_L.acc);
        elseif length(VD.IMU.ankle_L.acc) < 72000
            sf_gyro = 5000; ef_gyro = length(VD.IMU.ankle_L.acc);
        else
            sf_gyro = 5000; ef_gyro = 72000;
        end
    elseif contains(Test, "White")
        if length(VD.IMU.ankle_L.acc) < 20000
            sf_gyro = 1; ef_gyro = length(VD.IMU.ankle_L.acc);
        elseif length(VD.IMU.ankle_L.acc) < 144000
            sf_gyro = 5000; ef_gyro = length(VD.IMU.ankle_L.acc);
        else
            sf_gyro = 5000; ef_gyro = 144000;
        end
    elseif contains(Test, "Pink")
        if length(VD.IMU.ankle_L.acc) < 20000
            sf_gyro = 1; ef_gyro = length(VD.IMU.ankle_L.acc);
        elseif length(VD.IMU.ankle_L.acc) < 46000
            sf_gyro = 5000; ef_gyro = length(VD.IMU.ankle_L.acc);
        else
            sf_gyro = 5000; ef_gyro = 46000;
        end
    elseif contains(Test, "Incline") || contains(Test, "Decline")
        if length(VD.IMU.ankle_L.acc) < 5000
            sf_gyro = 1; ef_gyro = length(VD.IMU.ankle_L.acc);
        elseif length(VD.IMU.ankle_L.acc) < 29000
            sf_gyro = 5000; ef_gyro = length(VD.IMU.ankle_L.acc);
        else
            sf_gyro = 5000; ef_gyro = 29000;
        end
    else
        sf_gyro = 1; ef_gyro = length(VD.IMU.ankle_L.acc);
    end
    sf_fva = sf_gyro / smpf;
    ef_fva = ef_gyro / smpf;
end

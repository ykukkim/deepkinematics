""" 
This script imports Inertial Measurement Unit (IMU) data (acc, gyro, mag, quat) 
stored in Matlab-structs and positions, local rotations and skeletons stored in 
bvh files into numpy arrays, synchronize IMU and bvh data and saves them to the 
disk as npz-files.
"""

import os
from DK00_Utils import DK00_UT00_config as config_v

from DK01_DC00_conversion_functions import IMUFile, C3dFile, BvhFile
from DK01_DC00_conversion_functions import synchronize_vicon_to_imu

# Path to directory with c3d/imu/bvh files of the subjects
if __name__ == "__main__":
    for subject in [74]:
        if subject in [5,38,39,52,55,57,58,60,68]:
            # These subjects are skipped due to bad walking, missing VICON markers
            # or missing IMU synchronization      
            continue

        # Convert subject numbert to string
        subject_nr = str(subject).zfill(2)
        print('Subject: ', subject_nr)

        # Create path to subject folder
        path = config_v.dir_path + f'SonE_{subject_nr}/'

        # Initialize IMU file
        imu_file = IMUFile(path, subject_nr)

        # Create forward kinematics (DK03_FK_00_functions) folder if it does not exist already
        if not os.path.isdir(path + f'/fk/'):
                os.mkdir(path + '/fk/')

        for file in config_v.trials:

            # Intialize c3d to synchronize bvh with IMU
            c3d_file = C3dFile(path, file, subject_nr)

            # Determine starting frames
            # Vicon/IMU have a sampling frequency of 200Hz
            # BVH has a sampling frequency of 100Hz
            # Therefore, start frame of BVH is half of Vicon
            sf_vicon, sf_imu = synchronize_vicon_to_imu(c3d_file.n_frames, imu_file.n_frames[file])
            sf_bvh = int(sf_vicon/2)

            # Initialize BVH file
            bvh_file = BvhFile(path, subject_nr, file)
        
            # If skeleton does not exist yet, save it to disk
            if not os.path.isfile(path + f'/fk/SonE_{subject_nr}_skeleton.npz'):
                bvh_file.save_skeleton()

            # Extract joint positions and local rotations from bvh file
            bvh_file.extract_motion()

            # Synchronize motion with IMU data
            bvh_file.synchronize_motion(sf_bvh)

            # Save joint positions and local rotations to disk
            bvh_file.save_to_disk()

            # Extract data from IMU struct and concatenate sensors
            imu_data = imu_file.concatenate_data(file, sf_imu, [])

            # Trim IMU data to 2 * size of bvh (different Sampling Frequency)
            imu_data = imu_file.trim_data(imu_data, bvh_file.nframes * 2)
            
            # Save trimmed IMU to disk
            imu_file.save_to_disk(file, 'fk/', **imu_data)

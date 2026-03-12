""" 
This script imports Inertial Measurement Unit (IMU) data (acc, gyro, mag, quat) 
stored in Matlab-structs and body markers from Vicon stored in c3d files into 
numpy arrays, synchronize IMU and Vicon data and saves them to the disk as npz-files.
"""
import os
from DK00_Utils import DK00_UT00_config as config_v

from DK01_DC00_conversion_functions import C3dFile, IMUFile, synchronize_vicon_to_imu
from DK01_Data_Conversion.DK00_DC00_JC_step_detection import contact_phase_detection

if __name__ == "__main__":
    for subject in [44]:
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
        if not os.path.isdir(path):
                os.mkdir(path)

        for file in config_v.trials:

            # Initalize C3d file (one file per trial)
            c3d_file = C3dFile(path, file, subject_nr)

            # Calculate segment orientation
            c3d_file.compute_segment_orientation()

            # Compare nr of frames for IMU and c3d and set start frame
            sf_vicon, sf_imu = synchronize_vicon_to_imu(c3d_file.n_frames, imu_file.n_frames[file])

            # Adjust start frame for c3d
            c3d_file.adjust_start_frame(sf_vicon)

            # Check c3d markers for NaNs and delete the respective frames
            corrupted_frames = c3d_file.check_data_for_NaN()

            # Determine the contact phase for the walking trials. 
            # If there is an error in the step detection, check the Excel
            # "Neural_Nets_Comparison" for a possible solution
            contact_phase = {}
            if file in ['Norm_Pre', 'Norm_Post', 'White', 'Pink']:
                # Step detection for left foot
                left_contact = contact_phase_detection(c3d_file.joint_centers[:, 15, ...],
                                                       c3d_file.joint_centers[:, 13, ...], sf_vicon)
                # Step detection for right foot
                right_contact = contact_phase_detection(c3d_file.joint_centers[:, 16, ...],
                                                        c3d_file.joint_centers[:, 14, ...], sf_vicon)
                contact_phase['lcp'] = left_contact
                contact_phase['rcp'] = right_contact

            # Save joint center, segment orientation & for walking trials contact phase
            c3d_file.save_to_disk(**contact_phase)

            # Read and concatenate acc, gyro, mag and quats from matlab struct
            # and delete corrupted frames
            output = imu_file.concatenate_data(file, sf_imu, corrupted_frames)

            # Save the concatenated IMU data to disk
            imu_file.save_to_disk(file, **output)

    print()
    print('All files have been converted!')

import torch
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Or 'Qt5Agg'
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from DK00_Utils.DK00_UT00_config import CONSTANTS as C

from scipy.signal import find_peaks
from DK00_Utils import DK00_UT00_config as config_v
from DK00_Utils.DK00_UT00_utilsVisualisation import BvhVisualization
from DK00_Utils.DK00_UT00_utils import load_vicon_data, load_IMU_quat_and_transform_to_rot_matrix, \
    calculate_offset_and_compute_alignment, compute_relative_angle_for_skeleton

if __name__ == "__main__":

    for subject in [4]:#range(0,76):
        if subject in [5,38,39,52,55,57,58,60,68]:
            # These subjects are skipped due to bad walking, missing VICON markers
            # or missing IMU synchronization
            continue

        subject_nr = str(subject).zfill(2)
        print('Subject: ', subject_nr)

        # for file in config_v.trials:
        file = 'Norm_Pre'
        path_IMU_JC = f'{config_v.dir_path}/SonE_{subject_nr}/{file}_{subject_nr}_imu.npz'
        path_Vicon = f'{config_v.dir_path}/SonE_{subject_nr}/{file}_{subject_nr}_vicon.npz'
        path_IMU_fk = f'{config_v.dir_path}/SonE_{subject_nr}/fk/{file}_{subject_nr}_imu.npz'

        bvh = BvhVisualization(subject_nr, file, config_v.dir_path)
        data_vicon_jc, data_vicon_ori = load_vicon_data(path_Vicon)
        ori_imu = load_IMU_quat_and_transform_to_rot_matrix(path_IMU_JC)
        relative_phase = bvh._load_local_rotations()
        convention = C.JOINT_EULER_CONVENTIONS.get('LeftLeg', 'xyz')
        r = R.from_matrix(relative_phase[:,27,:,:])
        ankle_eul_angles = -r.as_euler(convention, degrees=True)  # shape: (f, 3)

        # === First figure: joint ankle of left ankle ===
        fig1 = plt.figure()
        plt.plot(ankle_eul_angles[:, 0],label='left', color='blue')  # First plot
        plt.plot(ankle_eul_angles[:, 1],label='left', color='red')  # First plot
        plt.plot(ankle_eul_angles[:, 2],label='left', color='green')  # First plot
        plt.title('Left Ankle Relative Phase')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # # === Calculate angular deviation ===
        # """ Sensor Drift without any movement: Difference of angular error in comparison to the first frame"""
        # ori_imu_offset = calculate_offset_and_compute_alignment(ori_imu, data_vicon_ori)
        # angle_error = np.zeros((data_vicon_ori.shape[0], 6))
        #
        # for i in range(data_vicon_ori.shape[1]):
        #     diff = np.linalg.inv(data_vicon_ori[:,i,...]) @ ori_imu_offset[:,i,...]
        #     angle_error[:,i] = np.linalg.norm(R.as_rotvec(R.from_matrix(diff)), axis = 1)
        #
        # angle_error = np.rad2deg(angle_error)
        #
        # sf = 0
        # # ef = angle_error.shape[0]
        # ef = data_vicon_jc.shape[0]
        # ef = relative_phase.shape[0]
        # peaks_left_ankle, _ = find_peaks(angle_error[sf:ef,0], height=63)
        # peaks_right_ankle, _ = find_peaks(angle_error[sf:ef,1], height=30)
        #
        # font = {
        #     'family': 'serif',  # Use 'serif', 'sans-serif', 'monospace', or a specific font name
        #     'weight': 'normal',
        #     'size': 22
        # }
        #
        # plt.rc('font', **font)
        #
        # print(angle_error[:,0])
        # print('Number of peaks:', len(peaks_left_ankle))
        #
        # # === Plot ankle error only ===
        # fig2 = plt.figure()
        # plt.subplot(211)
        # plt.plot(angle_error[sf:ef, 0])
        # plt.title('Left Ankle Angular Deviation [°]',fontsize=10)
        # plt.grid(True)
        #
        # plt.subplot(212)
        # plt.plot(angle_error[sf:ef, 1])
        # plt.title('Right Ankle Angular Deviation [°]',fontsize=10)
        # plt.grid(True)
        #
        # # plt.ion()  # Turns on interactive mode
        # plt.show()
        #
        # # === Full body angular error ===
        # fig3 = plt.figure()
        # titles = ['Right Ankle','Left Ankle','Left Hand', 'Right Hand', 'Head', 'Trunk']
        #
        # for i in range(6):
        #     plt.subplot(6, 1, i+1)
        #     plt.plot(angle_error[sf:ef, i])
        #     plt.title(titles[i])
        #     plt.grid(True)
        #
        # fig3.text(0.04, 0.5, 'Angular Deviation [°]', va='center', rotation='vertical', fontsize=12)
        # plt.xlabel('Frames')
        # plt.tight_layout(rect=[0.05, 0.05, 1, 0.97])
        # plt.show(block=False)
        # plt.pause(0.1)
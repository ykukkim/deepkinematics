import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from DK00_Utils import DK00_UT00_config as config_v
import scipy.io as sio



def load_IMU_quat_data(path):
    data = np.load(path)
    quat_data = data['quat']
    return quat_data

def average_rotation(rot1, rot2):
    """
    Compute the rotation needed to align `rot1` and `rot2` as an average along the first dimension.
    :param rot1: A tensor of shape (N, 3, 3).
    :param rot2: A tensor of shape (N, 3, 3).
    :return: The average rotation R such that rot1 * R ~ rot2. The shape of R is (3, 3).
    """
    rot1, rot2 = torch.from_numpy(rot1), torch.from_numpy(rot2)
    n = rot1.shape[0]
    m = torch.matmul(rot1.transpose(1, 2), rot2).sum(dim=0) / n

    u, _, v = torch.svd(m)

    # Ensure det(R) == 1 by fixing the sign of the last singular value.
    z = torch.eye(3, device=rot1.device, dtype=rot1.dtype)
    z[-1, -1] *= torch.sign(torch.det(torch.matmul(u, v.transpose(0, 1))))

    # Construct R.
    r = torch.matmul(u, torch.matmul(z, v.transpose(0, 1)))
    r = np.array(r)
    return r

def compute_alignment(ori_imu, offset):
    """Compute alignment between Vicon and IMU orientation"""
    print('ori_imu', ori_imu.shape)
    print('offset', offset.shape)
    ori = np.matmul(ori_imu, offset)
    print('ori-alignment', ori.shape)
    return ori

def transform_quat_to_rot_matrix(quat_data, rotation_matrix):
    # quat_data = load_IMU_quat_data(path_IMU)
    quat_data = quat_data[:,:,[1,2,3,0]] # adjust order of quaternions to scipy module (x,y,z,w)

    ankle_L = R.from_quat(quat_data[:,0,:])
    ankle_L = np.matmul(ankle_L.as_matrix(),rotation_matrix)
    ankle_R = R.from_quat(quat_data[:,1,:])
    ankle_R = np.matmul(ankle_R.as_matrix(), rotation_matrix)

    arm_L = R.from_quat(quat_data[:,2,:])
    arm_L = np.matmul(arm_L.as_matrix(), rotation_matrix)
    arm_R = R.from_quat(quat_data[:,3,:])
    arm_R = np.matmul(arm_R.as_matrix(), rotation_matrix)

    head = R.from_quat(quat_data[:,4,:])
    head = np.matmul(head.as_matrix(), rotation_matrix)
    trunk = R.from_quat(quat_data[:,5,:])
    trunk = np.matmul(trunk.as_matrix(), rotation_matrix)

    ori = np.stack((ankle_L, ankle_R, arm_L, arm_R, head, trunk), axis = 1)
    print('IMU orientation shape:', ori.shape)

    return ori

if __name__ == "__main__":

    for subject in [44]:#range(0,76):
        if subject in [5,38,39,52,55,57,58,60,68]:
            # These subjects are skipped due to bad walking, missing VICON markers
            # or missing IMU synchronization
            continue

        subject_nr = str(subject).zfill(2)
        print('Subject: ', subject_nr)

        for file in config_v.trials:
            path_imu_files = f'{config_v.dir_path}/SonE_{subject_nr}/fk/{file}_{subject_nr}_imu.npz'
            path_vicon_files = f'{config_v.dir_path}/SonE_{subject_nr}/{file}_{subject_nr}_vicon.npz'

        # data_vicon_jc, data_vicon_ori = load_vicon_data(path_Vicon)

        IMU_data_ankle_L = sio.loadmat(path_imu_files + '\\ankle_L.mat')['quat'][:1000000,...]
        IMU_data_ankle_R = sio.loadmat(path_imu_files + '\\ankle_R.mat')['quat'][:1000000,...]
        IMU_data_arm_L = sio.loadmat(path_imu_files + '\\arm_L.mat')['quat'][:1000000,...]
        IMU_data_arm_R = sio.loadmat(path_imu_files + '\\arm_R.mat')['quat'][:1000000,...]
        IMU_data_head = sio.loadmat(path_imu_files + '\\head.mat')['quat'][:1000000,...]
        IMU_data_trunk = sio.loadmat(path_imu_files + '\\trunk.mat')['quat'][:1000000,...]

        # print(IMU_data_ankle_L.shape)
        # print(IMU_data_ankle_R.shape)
        # print(IMU_data_arm_L.shape)
        # print(IMU_data_arm_R.shape)
        # print(IMU_data_head.shape)
        # print(IMU_data_trunk.shape)

        IMU_data_ori = np.stack((IMU_data_ankle_L, IMU_data_ankle_R, IMU_data_arm_L, IMU_data_arm_R, IMU_data_head, IMU_data_trunk), axis = 1)
        print('IMU orientation shape:', IMU_data_ori.shape)

        unit_matrix = np.array([[1,0,0], [0,1,0], [0,0,1]])
        IMU_data_rot_mat = transform_quat_to_rot_matrix(IMU_data_ori, unit_matrix)

        # offset = np.zeros((6,3,3))
        # for i in range(6):
        #     offset[i,...] = average_rotation(ori_imu[:10,i,:], data_vicon_ori[:10,i,...])

        # print('Offset shape:', offset.shape)
        # ori_imu_offset = compute_alignment(ori_imu, offset)

        """ Sensor Drift without any movement: Difference of angular error in comparison to the first frame"""
        angle_error = np.zeros((IMU_data_rot_mat.shape[0], 6))
        for i in range(IMU_data_rot_mat.shape[1]):
            diff = np.linalg.inv(IMU_data_rot_mat[0,i,...]) @ IMU_data_rot_mat[:,i,...]
            angle_error[:,i] = np.linalg.norm(R.as_rotvec(R.from_matrix(diff)), axis = 1)


        angle_error = np.rad2deg(angle_error)
        print(angle_error[:,0])

        sf = 0
        # ef = angle_error.shape[0]
        ef = 100000

        # peaks_left_ankle, _ = find_peaks(angle_error[sf:ef,0], height=63)
        # peaks_right_ankle, _ = find_peaks(angle_error[sf:ef,1], height=30)


        # print(angle_error[:,0])
        # print('Number of peaks:', len(peaks_left_ankle))


        plt.figure('Norm Pre 24')
        plt.subplot(611)
        plt.plot(angle_error[sf:ef, 0])
        # plt.plot(peaks_left_ankle, angle_error[peaks_left_ankle,0], "x")
        plt.title('left_ankle')

        plt.subplot(612)
        plt.plot(angle_error[sf:ef, 1])
        # plt.plot(peaks_right_ankle, angle_error[peaks_right_ankle,1], "x")
        plt.title('right_ankle')

        plt.subplot(613)
        plt.plot(angle_error[sf:ef, 2])
        plt.title('left_hand')

        plt.subplot(614)
        plt.plot(angle_error[sf:ef, 3])
        plt.title('right_hand')

        plt.subplot(615)
        plt.plot(angle_error[sf:ef, 4])
        plt.title('head')

        plt.subplot(616)
        plt.plot(angle_error[sf:ef, 5])
        plt.title('trunk')

        plt.subplots_adjust(hspace=0.6, wspace=0.35)
        plt.show()

        """
        
        cords_IMU = np.stack((np.zeros((IMU_data_rot_mat.shape[0],3)), np.ones((IMU_data_rot_mat.shape[0],3))), axis = 1)
        print(cords_IMU.shape)
        
        
        rigid_body = RigidBodies(cords_IMU[sf:ef,:,:], IMU_data_rot_mat[sf:ef,0:2,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
        rigid_body.rotation = rigid_body.rotation
        
        # rigid_body_without_rot = RigidBodies(cords_IMU[start_frame_vicon:,:,:], ori_imu[start_frame_imu:,:,:], position=np.array([2.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
        # rigid_body_without_rot.rotation = rigid_body_without_rot.rotation
        
        v = Viewer()
        v.scene.add(rigid_body)
        v.run()
        
        """
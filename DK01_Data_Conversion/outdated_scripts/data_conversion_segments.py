from pathlib import Path

import numpy as np
import torch
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.spheres import Spheres
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer
from ezc3d import c3d
from scipy.spatial.transform import Rotation as R

Vicon_labels_orientation = ['LFOO', 'LFOA', 'LFOL', 'LFOP', 'RFOO', 'RFOA', 'RFOL', 'RFOP', 'LLoArmO', 'LLoArmA', 'LLoArmL', 'LLoArmP', 
'RLoArmO', 'RLoArmA', 'RLoArmL', 'RLoArmP', 'HeadO', 'HeadA', 'HeadL', 'HeadP', 'PELO', 'PELA', 'PELL', 'PELP']

Vicon_labels_joint_centers = ['Head_CtoHThorax_score', 'LThorax_LUpArm_score', 'RThorax_RUpArm_score',
'LUpArm_LLoArm_score', 'RUpArm_RLoArm_score', 'LWRA', 'RWRA', 
'LHJC', 'RHJC', 'LKJC', 'RKJC', 'LAJC', 'RAJC', 'LTO3', 'RTO3', 'LHEE', 'RHEE']


def Vicon_import_data_from_c3d(path):
    c = c3d(path)
    data_labels = list(c['parameters']['POINT']['LABELS']['value'])
    # print(data_labels)
    return c, data_labels

def Vicon_get_coords_of_jc(data, index_jc):
    data = data['data']['points'][:3, index_jc, :]/1000 
    data = data.transpose()
    return data

def Vicon_compute_orientation_of_segments(data):
     
    for i in range(6):
        x = (data[:,4*i+1,:] - data[:,4*i,:]) / np.linalg.norm(data[:,4*i+1,:] - data[:,4*i,:], axis = -1, keepdims= True)
        y = (data[:,4*i+2,:] - data[:,4*i,:]) / np.linalg.norm(data[:,4*i+2,:] - data[:,4*i,:], axis = -1, keepdims= True)
        z = (data[:,4*i+3,:] - data[:,4*i,:]) / np.linalg.norm(data[:,4*i+3,:] - data[:,4*i,:], axis = -1, keepdims= True)

        ori = np.transpose(np.stack((x,y,z), axis = 1), axes = (0,2,1))

        if i == 0:
            segments = ori[:,np.newaxis,...]
        else:
            segments = np.concatenate((segments, ori[:,np.newaxis,...]), axis = 1)

    return segments

def Vicon_get_index_of_labels(data_labels, labels):
    index = []
    for label in labels:
        if label in data_labels:
            index.append(data_labels.index(label))
       
    """ add control that all 24 orientations are imported!!! """
    if len(labels) == 17:
        assert len(index) == len(labels), "Joint center markers are missing!"
        # print('Joint center markers are missing!')
    
    elif len(labels) == 24:
        assert len(index) == len(labels), "Orientation markers are missing!"

    elif len(labels) == 7:
        assert len(index) == len(labels), "Euler angle markers are missing!"

    return index    


def load_IMU_quat_and_transform_to_rot_matrix(path):
    """" Transform quatornions to rotation matrices. """
    data = np.load(path)
    quat_data = data['quat']
    quat_data = quat_data[:,:,[1,2,3,0]] # adjust order of quaternions to scipy module (x,y,z,w)

    ankle_L = R.from_quat(quat_data[:,0,:]).as_matrix()
    ankle_R = R.from_quat(quat_data[:,1,:]).as_matrix()
    arm_L = R.from_quat(quat_data[:,2,:]).as_matrix()
    arm_R = R.from_quat(quat_data[:,3,:]).as_matrix()
    head = R.from_quat(quat_data[:,4,:]).as_matrix()
    trunk = R.from_quat(quat_data[:,5,:]).as_matrix()

    ori = np.stack((ankle_L, ankle_R, arm_L, arm_R, head, trunk), axis = 1)
    print('IMU orientation:', ori.shape)
    return ori

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

def calculate_offset_and_compute_alignment(IMU_ori, Vicon_ori):
    offset = np.zeros((6,3,3))
    for i in range(IMU_ori.shape[1]):
        offset[i,...] = average_rotation(IMU_ori[1000:2000,i,:], Vicon_ori[1000:2000,i,:])
    
    IMU_ori_offset = np.matmul(IMU_ori, offset)
    return IMU_ori_offset

participant_number = '23'
dir = 'Norm_Pre'
path = r'D:\ZM_Data\SonE_{}'.format(participant_number)
Vicon_data, Vicon_data_labels = Vicon_import_data_from_c3d(path + '\Vicon\\' + dir + '.c3d')
Vicon_index_jc = Vicon_get_index_of_labels(Vicon_data_labels, Vicon_labels_joint_centers)
Vicon_index_ori = Vicon_get_index_of_labels(Vicon_data_labels, Vicon_labels_orientation)

Vicon_data_joint_centers = Vicon_get_coords_of_jc(Vicon_data, Vicon_index_jc)
Vicon_data_orientation = Vicon_get_coords_of_jc(Vicon_data, Vicon_index_ori)
Vicon_segment_orientations = Vicon_compute_orientation_of_segments(Vicon_data_orientation)

path_imu = Path('D:/ZM_Data/SonE_{}/{}_{}_imu.npz'.format(participant_number, dir, participant_number))
ori_imu = load_IMU_quat_and_transform_to_rot_matrix(path_imu)



""" Rotations """
""" Feet """
Vicon_segment_orientations[:,0:2,...] = np.matmul(Vicon_segment_orientations[:,0:2,...], R.from_euler('y', -90, degrees = True).as_matrix())
Vicon_segment_orientations[:,0:2,...] = np.matmul(Vicon_segment_orientations[:,0:2,...], R.from_euler('z', 180, degrees = True).as_matrix())

""" Arms """
Vicon_segment_orientations[:,2:4,...] = np.matmul(Vicon_segment_orientations[:,2:4,...], R.from_euler('z', -90, degrees = True).as_matrix())
""" Left Arm """
Vicon_segment_orientations[:,2:3,...] = np.matmul(Vicon_segment_orientations[:,2:3,...], R.from_euler('x', -180, degrees = True).as_matrix())

""" Head """
Vicon_segment_orientations[:,4,...] = np.matmul(Vicon_segment_orientations[:,4,...], R.from_euler('y', 90, degrees = True).as_matrix())

""" Trunk """
Vicon_segment_orientations[:,5,...] = np.matmul(Vicon_segment_orientations[:,5,...], R.from_euler('x', 180, degrees = True).as_matrix())
Vicon_segment_orientations[:,5,...] = np.matmul(Vicon_segment_orientations[:,5,...], R.from_euler('y', 90, degrees = True).as_matrix())


# ori_imu[:,0,...] = np.matmul(R.from_euler('Z', 180, degrees = True).as_matrix(), ori_imu[:,0,...])
# ori_imu[:,2,...] = np.matmul(ori_imu[:,2,...], R.from_euler('Z', 180, degrees = True).as_matrix())

# ori_imu[:,:,...] = calculate_offset_and_compute_alignment(ori_imu[:,:,...], Vicon_segment_orientations[:48002,:,...])


predicted = Spheres(Vicon_data_joint_centers, color = (2,0,0,1), radius = 0.02, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
predicted.rotation = predicted.rotation

# ground_truth = Lines(person[:,:,:], mode = "lines", color = (1,0,0,1), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
# ground_truth.rotation = ground_truth.rotation

rigid_body = RigidBodies(Vicon_data_joint_centers[:,[15,16,5,6,0,7],:], Vicon_segment_orientations[:,:,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
rigid_body.rotation = rigid_body.rotation

rigid_body_imu = RigidBodies(Vicon_data_joint_centers[:ori_imu.shape[0],[15,16,5,6,0,7],:], ori_imu[:,:,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
rigid_body_imu.rotation = rigid_body_imu.rotation


v = Viewer()
v.scene.add(rigid_body, predicted, rigid_body_imu)
v.run()
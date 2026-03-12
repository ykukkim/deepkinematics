import re
import torch
import random
import numpy as np

from scipy.spatial.transform import Rotation as R
from DK00_Utils.DK00_UT00_config import CONSTANTS as C

#####################
def set_random_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        torch.backends.cudnn.deterministic = True

def clean_string(s):
    """
    Remove special characters from a string and replace spaces with underscores.
    """
    # Remove special characters
    cleaned = re.sub(r'[^\w\s-]', '', s)
    # Replace spaces with underscores
    return cleaned.replace(' ', '_')

def load_vicon_data(path):
    """ Load joint center and segment orientation from npz files """
    data = np.load(path)
    data_jc = data['jc']
    data_ori = data['ori']
        
    print('Vicon joint center:', data_jc.shape)
    print('Vicon segment orientation:', data_ori.shape)

    return data_jc, data_ori #, ori_IMU

def load_IMU_quat_and_transform_to_rot_matrix(path):
    """" Transform quaternions to rotation matrices. """
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

def get_coords_for_lines(data_vicon):
    """ Lines to represent stickfigure in AIT viewer """
    # Prepare neck and hip keypoints for stickfigure visualization
    neck = np.mean(np.array([data_vicon[:,1,:], data_vicon[:,2,:]]), axis=0)
    hip_avg = np.mean(np.array([data_vicon[:,7,:], data_vicon[:,8,:]]), axis=0)
    
    """ Lines: mode = 'lines'
        Start and End coordinate of a line are saved to the person array.
        Order of the keypoints can be found in the data_conversion_Vicon.py file. 
    """    
    stick_figure = np.stack((data_vicon[:,0,:], neck[:,:], # head
    data_vicon[:,1,:], data_vicon[:,2,:], # shoulder
    data_vicon[:,1,:], data_vicon[:,3,:], # left upper arm
    data_vicon[:,3,:], data_vicon[:,5,:], # left lower arm
    data_vicon[:,2,:], data_vicon[:,4,:], # right upper arm
    data_vicon[:,4,:], data_vicon[:,6,:], # right lower arm
    neck, hip_avg,  # core
    data_vicon[:,7,:], data_vicon[:,8,:], # hips
    data_vicon[:,7,:], data_vicon[:,9,:], # left upper leg 
    data_vicon[:,9,:], data_vicon[:,11,:], # left lower leg
    data_vicon[:,11,:], data_vicon[:,13,:], # left foot
    data_vicon[:,8,:], data_vicon[:,10,:],  # right upper leg
    data_vicon[:,10,:], data_vicon[:,12,:], # right lower leg
    data_vicon[:,12,:], data_vicon[:,14,:] # right foot
    ), axis = 1)
          
    return stick_figure

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
    ori = np.matmul(ori_imu, offset)
    return ori
    
def calculate_offset_and_compute_alignment(IMU_ori, Vicon_ori):
    """ Compute the rotation needed to align `IMU_ori` and `Vicon_ori` and align the orientations """
    offset = np.zeros((6,3,3))
    # Compute rotational offset for all six IMU sensor
    for i in range(IMU_ori.shape[1]):
        offset[i,...] = average_rotation(IMU_ori[:10,i,:], Vicon_ori[:10,i,:])
    # Align IMU_ori to Vicon_ori for the entire sequence
    IMU_ori_offset = np.matmul(IMU_ori, offset)
    return IMU_ori_offset

def get_coords_for_IMU_position(data):
    """ Create array with IMU positions for animation in AITViewer """
    hip_avg = np.mean( np.array([data[:,7,:], data[:,8,:]]), axis=0)
    hip_avg = hip_avg[:,np.newaxis,:]    # add middle of hip joints to coordinates
    coords_IMU = np.array(data[:,[11,12,5,6,0],:])
    coords_IMU = np.concatenate((coords_IMU, hip_avg), axis = 1)
    return coords_IMU

def normalize_vicon_jc(data):
    """ Subtract the root (midpoint between hip joints) from all joints """
    hip_avg = np.mean( np.array([data[:,7,:], data[:,8,:]]), axis=0)
    hip_avg = hip_avg[:,np.newaxis,:]
    return data - hip_avg

def lr_lambda(epoch):
    if epoch < 10:
        return 1.0  # No decay for the first 10 epochs
    else:
        return 0.95 ** (epoch - 10)  # Decay by 0.95 after each epoch starting from epoch 10

# Z-Score Standardisation function
def z_score_normalise(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    norm_data = (data - mean) / std
    return norm_data

def compute_relative_angle_for_skeleton(rotations, convention='xyz', in_degrees=False):
    """
     Compute the relative Euler angles for the skeleton given the rotations.

     Args:
         rotations (np.ndarray): Array of shape (f, n_joints, 3, 3) containing the rotation matrices for each joint.
         convention (str): The convention used for Euler angles (e.g., 'xyz', 'zyx').
         in_degrees (bool): If True, convert the angle difference from radians to degrees.

     Returns:
         np.ndarray: Array of shape (f, len(joint_pairs), 3) containing the relative Euler angles for each axis.
     """
    f, n_joints, _, _ = rotations.shape
    relative_euler_angles = []

    for parent, child in C.skeleton_pairs:
        # Extract the rotation matrices for the parent and child joints
        rot_parent = rotations[:, parent]  # Shape: (f, 3, 3)
        rot_child = rotations[:, child]  # Shape: (f, 3, 3)

        # Compute the relative rotation matrix (child relative to parent)
        relative_rotations = np.matmul(rot_child, np.transpose(rot_parent, axes=(0, 2, 1)))  # Shape: (f, 3, 3)

        # Convert the relative rotation matrices to Euler angles
        r = R.from_matrix(relative_rotations)
        euler_angles = r.as_euler(convention, degrees=in_degrees)  # Shape: (f, 3)

        # Store the Euler angles
        relative_euler_angles.append(euler_angles)

    # Stack the list into a single array with shape (f, len(joint_pairs), 3)
    relative_euler_angles = np.stack(relative_euler_angles, axis=1)

    return relative_euler_angles


from importlib.resources import path
from tracemalloc import start
import numpy as np
from scipy.spatial.transform import Rotation as R
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.lines import Lines
from aitviewer.viewer import Viewer
from aitviewer.utils.so3 import aa2rot_numpy


# Path to the vicon and imu files
path_Vicon = r'D:\ZM_Data\SonE_24\Norm_Post_24_vicon.npz'
path_IMU = r'D:\ZM_Data\SonE_24\Norm_Post_24_imu.npz' 


def load_vicon_data(path):
    data = np.load(path)
    data = data['DK03_JC_00_functions']
    
    rotation_matrix = np.array([[1,0,0], [0,0,-1], [0,1,0]])
    # rotation_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data = np.matmul(data, rotation_matrix)

    print('Vicon data:', data.shape)
    print('right knee', data[:,10,0])
    return data

def load_IMU_quat_data(path):
    data = np.load(path)
    quat_data = data['quat']
    return quat_data

def transform_quat_to_rot_matrix(path_IMU, rotation_matrix):
    quat_data = load_IMU_quat_data(path_IMU)
    quat_data = quat_data[:,:,[1,2,3,0]] # adjust order of quaternions to scipy module

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
    print('Ori shape:', ori.shape)

    return ori


def start_viewer(render):
    v = Viewer()
    v.scene.add(render)
    v.run()

def get_coords_for_lines(data_vicon):
    # Lines modeled to represent limbs and core
    neck = np.mean( np.array([data_vicon[:,1,:], data_vicon[:,4,:]]), axis=0)
    hip_avg = np.mean( np.array([data_vicon[:,7,:], data_vicon[:,8,:]]), axis=0)
    head = data_vicon[:,0,:]
    
    
    """ Lines: mode = 'lines'
        Start and End coordinate of a line are saved to the person array.
        Order of the keypoints can be found in the data__conversion_Vicon.py file.
    """
    person = np.stack((head[:,:], neck[:,:], # head
    data_vicon[:,1,:], data_vicon[:,4,:], data_vicon[:,1,:], data_vicon[:,2,:], data_vicon[:,2,:], data_vicon[:,3,:], # shoulders and left arm
    data_vicon[:,4,:], data_vicon[:,5,:], data_vicon[:,5,:], data_vicon[:,6,:], # right arm
    neck, hip_avg, data_vicon[:,7,:], data_vicon[:,8,:], # core and hips
    data_vicon[:,7,:], data_vicon[:,9,:], data_vicon[:,9,:], data_vicon[:,11,:], data_vicon[:,11,:], data_vicon[:,13,:], # left leg
    data_vicon[:,8,:], data_vicon[:,10,:], data_vicon[:,10,:], data_vicon[:,12,:], data_vicon[:,12,:], data_vicon[:,14,:] # right leg
    ), axis = 1)
          
    return person


data_vicon = load_vicon_data(path_Vicon)

# Joint Centers modeled to spheres
spheres = Spheres(data_vicon)

# Getting points to create stick figure
person01 = get_coords_for_lines(data_vicon)


# Coordinates of IMU sensor positions
hip_avg = np.mean( np.array([data_vicon[:,7,:], data_vicon[:,8,:]]), axis=0)
hip_avg = hip_avg[:,np.newaxis,:]    # add middle of hip joints to coordinates
cords_IMU = np.array(data_vicon[:,[11,12,3,6,0],:])
cords_IMU = np.concatenate((cords_IMU, hip_avg), axis = 1)

rotation_matrix = np.array([[1,0,0], [0,0,-1], [0,1,0]])
unit_matrix = np.array([[1,0,0], [0,1,0], [0,0,1]])

ori = transform_quat_to_rot_matrix(path_IMU, unit_matrix)
ori_rot = transform_quat_to_rot_matrix(path_IMU, rotation_matrix)


if ori.shape[0]-data_vicon.shape[0] >= 0:
    start_frame_imu = ori.shape[0] - data_vicon.shape[0]
    start_frame_vicon = 0
else:
    start_frame_vicon = data_vicon.shape[0] - ori.shape[0]
    start_frame_imu = 0
print('Starting Frame Vicon:', start_frame_vicon, 'Starting Frame IMU:', start_frame_imu)


p1_lines = Lines(person01[start_frame_vicon:,:,:], mode = "lines")
p2_lines = Lines(person01[start_frame_vicon:,:,:], mode = 'lines', position = np.array([1.0, 0.0, 0.0]))
rigid_body = RigidBodies(cords_IMU[start_frame_vicon:,:,:], ori[start_frame_imu:,:,:])
rigit_body_rot = RigidBodies(cords_IMU[start_frame_vicon:,:,:], ori_rot[start_frame_imu:,:,:], position=np.array([1.0, 0.0, 0.0]))

v = Viewer()
v.scene.add(spheres, p1_lines, p2_lines, rigid_body, rigit_body_rot) # left_leg, right_leg, hip, shoulders, core, rigid_body)
v.run()


"""
spheres = Spheres(data_vicon, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
spheres.mesh.rotation = spheres.rotation

# lines modelled to represent limbs and core
left_arm = Lines(data_vicon[:,1:4,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
left_arm.mesh.rotation = left_arm.rotation

right_arm = Lines(data_vicon[:,4:7,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
right_arm.mesh.rotation = right_arm.rotation

left_leg = Lines(data_vicon[:,7:15:2,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
left_leg.mesh.rotation = left_leg.rotation

right_leg = Lines(data_vicon[:,8:15:2,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
right_leg.mesh.rotation = right_leg.rotation

hip = Lines(data_vicon[:,7:9,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
hip.mesh.rotation = hip.rotation

shoulders = Lines(data_vicon[:,[1,4],:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
shoulders.mesh.rotation = shoulders.rotation


neck = np.mean( np.array([data_vicon[:,1,:], data_vicon[:,4,:]]), axis=0 )
hip_avg = np.mean( np.array([data_vicon[:,7,:], data_vicon[:,8,:]]), axis=0 )
pts_core = np.stack((data_vicon[:,0,:], neck, hip_avg), axis = 1)

core = Lines(pts_core, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
core.mesh.rotation = core.rotation

# coordinates of IMU sensor positions
cords_IMU = np.array(data_vicon[:,[11,12,3,6,0],:])
hip_3d = hip_avg[:,np.newaxis,:]    # add middle of hip joints to coordinates
cords_IMU = np.concatenate((cords_IMU, hip_3d), axis = 1)


ori = transform_quat_to_rot_matrix(path_IMU)
rigid_body = RigidBodies(cords_IMU, ori[:data_vicon.shape[0],:,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
rigid_body.rotation = rigid_body.rotation
# start_viewer(spheres, rigid_body)

print('data vicon shape', data_vicon.shape[0])
"""

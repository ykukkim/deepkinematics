from pathlib import Path

import numpy as np
import pandas as pd
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.spheres import Spheres
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

segments =['SonE_02:Pelvis', 'SonE_02:LUpLeg', 'SonE_02:RUpLeg', 'SonE_02:LLowLeg', 'SonE_02:RLowLeg', 'SonE_02:LFoot', 'SonE_02:RFoot']

data = pd.read_csv(r'D:\ZM_Data\SonE_02\Vicon\Norm_Pre.csv', skiprows=2, keep_default_na = False,on_bad_lines='skip', low_memory=False)#, sep=',' ) skiprows=4362,

participant_number = '02'
experiment = 'Norm_Pre'

path_Vicon = Path('D:/ZM_Data/SonE_{}/{}_{}_vicon.npz'.format(participant_number, experiment, participant_number))
path_IMU = Path('D:/ZM_Data/SonE_{}/{}_{}_imu.npz'.format(participant_number, experiment, participant_number))


def get_coords_for_lines(data_vicon):
    """ Lines to represent stickfigure in AIT viewer """
     
    """ Lines: mode = 'lines'
        Start and End coordinate of a line are saved to the person array.
        Order of the keypoints can be found in the data_conversion_Vicon.py file.
    """
    stick_figure = np.stack((data_vicon[:,0,:], data_vicon[:,1,:], #
    data_vicon[:,0,:], data_vicon[:,2,:], data_vicon[:,1,:], data_vicon[:,3,:], data_vicon[:,3,:], data_vicon[:,5,:],
    data_vicon[:,2,:], data_vicon[:,4,:], data_vicon[:,4,:], data_vicon[:,6,:], 
    ), axis = 1)
         
    return stick_figure

def load_vicon_data(path):
    """ load joint center and segment orientation from npz files """
    data = np.load(path)
    data_jc = data['DK03_JC_00_functions']
    data_ori = data['ori']
    data_angles = data['ori_euler']
    
    return data_jc, data_ori, data_angles


data_vicon_jc, data_vicon_ori, data_vicon_angles = load_vicon_data(path_Vicon)
# print(data_vicon_angles[0,:2,...])

cols = []
for segment in segments:
    if segment in data.columns:
        cols.append(data.columns.get_loc(segment))
        

# Extracting the angle and position vector out of the pandas dataframe
angle = []
position = []
for i in range(len(segments)):
    angle.append(np.array(data.iloc[2:,cols[i]:cols[i]+3]))
    position.append(np.array(data.iloc[2:,cols[i]+3:cols[i]+6]))

# changing order of numpy array to (Frames, Segment, Coordinates)
angle = np.moveaxis(np.array(angle, dtype=float),(0,1,2),(1,0,2))
position = np.moveaxis(np.array(position, dtype=float),(0,1,2),(1,0,2))
position = position/1000



# print left lower limb angles
# plt.figure()
# plt.plot(angle[:,5,:])

# # plt.figure()
# # plt.plot(R.from_matrix(data_vicon_angles[:,0,:]).as_euler('YXZ', degrees = True)*1000)
# plt.show()



orientation = []
for i in range(len(segments)):
    if i < 5:
        orientation.append(R.from_euler('XYZ', angle[:,i,:], degrees = True).as_matrix())
    else:
        orientation.append(R.from_euler('XYZ', angle[:,i,:], degrees = True).as_matrix())
orientation = np.moveaxis(np.array(orientation, dtype=float),(0,1,2),(1,0,2))

pelvis_inv = []
bone_offset =[]
for i in range(orientation.shape[0]):
    # pelvis_inv.append(np.linalg.inv(orientation[i,0,...]))
    bone_offset.append(np.matmul(position[i,1,:] -position[i,0,:], np.linalg.inv(orientation[i,0,...])))


# pelvis_inv = np.array(pelvis_inv, dtype=float)
# print(pelvis_inv.shape)
bone_offset = np.array(bone_offset, dtype=float)
# print(bone_offset)
print(np.mean(bone_offset, axis = 0))
print(np.std(bone_offset, axis = 0))

# plt.figure()
# plt.plot(bone_offset)
# plt.show()

LHip =  np.dot(np.mean(bone_offset, axis = 0), orientation[:,0,...]) + position[:,0,...]
LHip = LHip[:,np.newaxis,:]

# bone_offset = np.dot((position[:,1,:] -position[:,0,:]), pelvis_inv[:,...])
# print(bone_offset.shape)


# orientation[:,5,...] = np.matmul(orientation[:,5,...],orientation[:,3,...])

person = get_coords_for_lines(position)

# spheres = Spheres(data_vicon_jc, color = (2,0,0,1), radius = 0.02, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
# spheres.rotation = spheres.rotation

predicted = Spheres(LHip, color = (2,0,0,1), radius = 0.02, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
predicted.rotation = predicted.rotation

ground_truth = Lines(person[:,:,:], mode = "lines", color = (1,0,0,1), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
ground_truth.rotation = ground_truth.rotation

rigid_body = RigidBodies(position[:,:,:], orientation[:,:,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
rigid_body.rotation = rigid_body.rotation

# rigid_body = RigidBodies(data_vicon_jc[:,[7,8,9,10,11,12],:], data_vicon_angles[:,1:,:], rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
# rigid_body.rotation = rigid_body.rotation


v = Viewer()
v.scene.add(rigid_body, ground_truth, predicted)
v.run()
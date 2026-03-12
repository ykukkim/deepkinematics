import mat73
import numpy as np
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.spheres import Spheres
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer
from ezc3d import c3d
from scipy.spatial.transform import Rotation as R


def Vicon_import_data_from_c3d(path):
    c = c3d(path)
    data_labels = list(c['parameters']['POINT']['LABELS']['value'])
    # print(data_labels)
    return c, data_labels

def Vicon_get_coords_of_jc(data, index_jc):
    data = data['data']['points'][:3, index_jc, :]/1000 
    data = data.transpose()
    return data

def get_index_of_labels(labels_of_data, labels):
    """ Get index of markers of interest in lables list. """
    index = []
    for label in labels:
        if label in labels_of_data:
            index.append(labels_of_data.index(label))
       
    return index

labels = ['LHEE', 'RHEE', 'LTO3', 'RTO3', 'LTTT', 'RTTT','LASI', 'RASI', 'LWRA', 'RWRA']

path = r'D:\ZM_Data\SonE_23'
data_matlab = mat73.loadmat(path + '\IMU\GaitSummaryIMU.mat', use_attrdict=True) #, matlab_compatible = True) #, mat_dtype = True)#, use_attrdict=True) 

# def IMU_get_data_from_mat_file(pos, idx, sensor_type, trim='trim'):
    # tmp = IMU_data_matlab['GaitSummaryIMU']['ModulesData'][pos]['synced'][trim][idx][sensor_type]
#     return tmp

# pelvis_global_trans = data_matlab['Total']['SonE_02']['Pelvis']['Output_GetSegmentGlobalTranslation']

# print(data_matlab['Total']['SonE_02'].dtype) #['Pelvis'])#['Pelvis'])

ori_ankle_L = data_matlab['GaitSummaryIMU']['ModulesData']['ankle_L']['synced']['trim'][0]['quat']
ori_ankle_R = data_matlab['GaitSummaryIMU']['ModulesData']['ankle_R']['synced']['trim'][0]['quat']
ori_arm_L = data_matlab['GaitSummaryIMU']['ModulesData']['arm_L']['synced']['trim'][0]['quat']
ori_arm_R = data_matlab['GaitSummaryIMU']['ModulesData']['arm_R']['synced']['trim'][0]['quat']
ori_ankle_L = ori_ankle_L[:,[1,2,3,0]]
ori_ankle_R = ori_ankle_R[:,[1,2,3,0]]
ori_arm_L = ori_arm_L[:,[1,2,3,0]]
ori_arm_R = ori_arm_R[:,[1,2,3,0]]
ori_ankle_L = R.from_quat(ori_ankle_L).as_matrix()
ori_ankle_R = R.from_quat(ori_ankle_R).as_matrix()
ori_arm_L = R.from_quat(ori_arm_L).as_matrix()
ori_arm_R = R.from_quat(ori_arm_R).as_matrix()
ori = np.stack((ori_ankle_L, ori_ankle_R, ori_arm_L, ori_arm_R))
ori = np.moveaxis(ori, (0,1,2,3),(1,0,2,3))
print(ori.shape)

data, data_labels = Vicon_import_data_from_c3d(path + '\Vicon\TUG_Pre.c3d')
index = get_index_of_labels(data_labels, labels)
jc = Vicon_get_coords_of_jc(data, index)
print(jc.shape)

# np.zeros((ori_ankle_L.shape[0],1,3))

spheres = Spheres(jc, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
spheres.rotation = spheres.rotation

rigid_body = RigidBodies(jc[:,[0,1,8,9],...], ori[:,...], position=np.array([0.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
rigid_body.rotation = rigid_body.rotation

v = Viewer()
v.scene.add(spheres, rigid_body)
v.run()
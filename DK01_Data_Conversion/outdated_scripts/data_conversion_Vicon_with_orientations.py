import numpy as np
from ezc3d import c3d
import os
import os

import numpy as np
from ezc3d import c3d

# Path to subject
path = r'D:\ZM_Data\SonE_20'

# Order of Joint Centers
# if order is changed, data_visualization.py has to be adjusted!
labels_joint_centers = ['Head_CtoHThorax_score', 'LThorax_LUpArm_score', 'LUpArm_LLoArm_score', 
'LWRA', 'RThorax_RUpArm_score', 'RUpArm_RLoArm_score',
'RWRA', 'LHJC', 'RHJC', 'LKJC', 'RKJC', 'LAJC', 'RAJC', 'LTO3', 'RTO3']

labels_orientation = ['LTIO', 'LTIA', 'LTIL', 'LTIP', 'RTIO', 'RTIA', 'RTIL', 'RTIP', 'LLoArmO', 'LLoArmA', 'LLoArmL', 'LLoArmP', 
'RLoArmO', 'RLoArmA', 'RLoArmL', 'RLoArmP', 'HeadO', 'HeadA', 'HeadL', 'HeadP', 'PELO', 'PELA', 'PELL', 'PELP']

# Functions
def create_numpy_folder(path):
    os.makedirs(path + '\\numpy_arrays', exist_ok= True)

def import_data_from_c3d(path):
    c = c3d(path)
    data_labels = list(c['parameters']['POINT']['LABELS']['value'])
    # print(data_labels)
    return c, data_labels

def get_c3d_filenames(path):
    dir_list = os.listdir(path)
    for dir in dir_list:
        if 'c3d' not in dir:
            dir_list.remove(dir)           
    return dir_list  
      
def get_index_of_labels(data_labels, labels):
    index = []
    for label in labels:
        if label in data_labels:
            index.append(data_labels.index(label))
        else:
            continue

    """ add control that all 24 orientations are imported!!! """
    if len(labels) == 15:
        assert len(index) == len(labels), "Joint center markers are missing!"
        # print('Joint center markers are missing!')
    
    elif len(labels) == 24:
        assert len(index) == len(labels), "Orientation markers are missing!"

    return index

def get_coords_of_jc(index_jc):
    data = c['data']['points'][:3, index_jc, :]/1000 
    data = data.transpose()
    return data


def compute_orientation_of_segments(data):
     
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


def safe_data_to_numpy_array(path, dir, joint_center, orientations):
    participant_nbr = ''.join(filter(str.isdigit, path))
    path = path + '\\' + dir.split('.')[0] + '_' +  participant_nbr + '_vicon'
    np.savez_compressed(path, jc = joint_center, ori = orientations)
    print('Joint Center:', joint_center.shape)
    print('Segment Orientation:', orientations.shape)
    print('File {} has been converted.'.format(dir.split('.')[0]))
    print()

    
# check if numpy array folder exists:
# create_numpy_folder(path)

# check for all .c3d files in the folder:
# dir_list = get_c3d_filenames(path + '\\Vicon\\')
dir_list = ['Norm_Pre.c3d', 'Norm_Post.c3d']


for dir in dir_list:
    c, data_labels = import_data_from_c3d(path + '\Vicon\\' + dir)
    index_jc = get_index_of_labels(data_labels, labels_joint_centers)
    index_ori = get_index_of_labels(data_labels, labels_orientation)
        
    data_joint_centers = get_coords_of_jc(index_jc)
    data_orientation = get_coords_of_jc(index_ori)
    segment_orientations = compute_orientation_of_segments(data_orientation)
    safe_data_to_numpy_array(path, dir, data_joint_centers, segment_orientations)
    
print()
print('All files have been converted!')
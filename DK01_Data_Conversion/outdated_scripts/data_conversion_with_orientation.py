import os

import numpy as np
from ezc3d import c3d

# Path to subject
path = r'D:\ZM_Data\SonE_21'

# Order of Joint Centers
# if order is changed, data_visualization.py has to be adjusted!
labels = ['Head_CtoHThorax_score', 'LThorax_LUpArm_score', 'LUpArm_LLoArm_score', 
'LWRA', 'RThorax_RUpArm_score', 'RUpArm_RLoArm_score',
'RWRA', 'LHJC', 'RHJC', 'LKJC', 'RKJC', 'LAJC', 'RAJC', 'LTO3', 'RTO3']

# Functions
def create_numpy_folder(path):
    os.makedirs(path + '\\numpy_arrays', exist_ok= True)

def import_data_from_c3d(path):
    c = c3d(path)
    data_labels = list(c['parameters']['POINT']['LABELS']['value'])
    return c, data_labels

def get_c3d_filenames(path):
    dir_list = os.listdir(path)
    for dir in dir_list:
        if 'c3d' not in dir:
            dir_list.remove(dir)           
    return dir_list  
      
def get_index_of_labels(data_labels):
    index_jc = []
    for label in labels:
        if label in data_labels:
            index_jc.append(data_labels.index(label))
        else:
            continue
    return index_jc

def get_coords_of_jc(index_jc):
    data = c['data']['points'][:3, index_jc, :]/1000 
    data = data.transpose()
    return data

def safe_data_to_numpy_array(path, dir, data):
    participant_nbr = ''.join(filter(str.isdigit, path))
    path = path + '\\' + dir.split('.')[0] + '_' +  participant_nbr + '_vicon'
    np.savez_compressed(path, data)
    print(data.shape)
    print('File {} has been converted.'.format(dir.split('.')[0]))


# check if numpy array folder exists:
# create_numpy_folder(path)

# check for all .c3d files in the folder:
# dir_list = get_c3d_filenames(path + '\\Vicon\\')
dir_list = ['Norm_Pre.c3d', 'Norm_Post.c3d']


for dir in dir_list:
    c, data_labels = import_data_from_c3d(path + '\Vicon\\' + dir)
    index_jc = get_index_of_labels(data_labels)
    data = get_coords_of_jc(index_jc)
    safe_data_to_numpy_array(path,dir,data)
    
print()
print('All files have been converted!')
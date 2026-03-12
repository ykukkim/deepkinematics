import os

import mat73
import numpy as np
from ezc3d import c3d

# Path to subject
path = r'D:\ZM_Data\SonE_01'

# Order of Joint Centers
""" if order is changed, data_visualization.py has to be adjusted! """

Vicon_labels_joint_centers = ['Head_CtoHThorax_score', 'LThorax_LUpArm_score', 'LUpArm_LLoArm_score', 
'LWRA', 'RThorax_RUpArm_score', 'RUpArm_RLoArm_score',
'RWRA', 'LHJC', 'RHJC', 'LKJC', 'RKJC', 'LAJC', 'RAJC', 'LTO3', 'RTO3']

Vicon_labels_orientation = ['LTIO', 'LTIA', 'LTIL', 'LTIP', 'RTIO', 'RTIA', 'RTIL', 'RTIP', 'LLoArmO', 'LLoArmA', 'LLoArmL', 'LLoArmP', 
'RLoArmO', 'RLoArmA', 'RLoArmL', 'RLoArmP', 'HeadO', 'HeadA', 'HeadL', 'HeadP', 'PELO', 'PELA', 'PELL', 'PELP']

IMU_labels_sensor_pos = ['ankle_L', 'ankle_R', 'arm_L', 'arm_R', 'head', 'trunk']

#======================================================================================================================
#======================================== Functions Vicon =============================================================
#======================================================================================================================

def Vicon_import_data_from_c3d(path):
    c = c3d(path)
    data_labels = list(c['parameters']['POINT']['LABELS']['value'])
    # print(data_labels)
    return c, data_labels

def Vicon_get_c3d_filenames(path):
    dir_list = os.listdir(path)
    for dir in dir_list:
        if 'c3d' not in dir:
            dir_list.remove(dir)           
    return dir_list  
      
def Vicon_get_index_of_labels(data_labels, labels):
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


def Vicon_safe_data_to_numpy_array(path, dir, joint_center, orientations):
    participant_nbr = ''.join(filter(str.isdigit, path))
    path = path + '\\' + dir.split('.')[0] + '_' +  participant_nbr + '_vicon'
    np.savez_compressed(path, jc = joint_center, ori = orientations)
    print('Joint Center:', joint_center.shape)
    print('Segment Orientation:', orientations.shape)
    print('File {} has been converted.'.format(dir))
    print()

#======================================================================================================================
#======================================== Functions IMU ===============================================================
#======================================================================================================================

def IMU_get_data_from_mat_file(pos, idx, sensor_type):
    tmp = IMU_data_matlab['GaitSummaryIMU']['ModulesData'][pos]['synced']['trim'][idx][sensor_type]
    return tmp


def IMU_get_index_of_files():       # search for Norm_Pre and Norm_Post files (should have around 75k to 85k frames)
    tmp = IMU_data_matlab['GaitSummaryIMU']['ModulesData']['ankle_L']['synced']['trim']
    index = []
    for i in range(len(tmp)):
        if tmp[i]['acc'].shape[0] > 75000 and tmp[i]['acc'].shape[0] < 85000:
            index.append(i)
    
    return index


def IMU_get_and_save_data_to_numpy(index, sf, frm_corrupt):
    for pos in IMU_labels_sensor_pos:
        
        if pos == IMU_labels_sensor_pos[0]:
            acc_data = IMU_get_data_from_mat_file(pos, index, 'acc')
            gyro_data = IMU_get_data_from_mat_file(pos, index, 'gyro')
            mag_data = IMU_get_data_from_mat_file(pos, index, 'mag')
            quat_data = IMU_get_data_from_mat_file(pos, index, 'quat')
                        
        else:
            acc_data = np.dstack((acc_data, IMU_get_data_from_mat_file(pos, index, 'acc')))
            gyro_data = np.dstack((gyro_data, IMU_get_data_from_mat_file(pos, index, 'gyro')))
            mag_data = np.dstack((mag_data, IMU_get_data_from_mat_file(pos, index, 'mag')))
            quat_data = np.dstack((quat_data, IMU_get_data_from_mat_file(pos, index, 'quat')))           
        
    
    acc_data = np.moveaxis(acc_data,(0,1,2),(0,2,1))
    gyro_data = np.moveaxis(gyro_data,(0,1,2),(0,2,1))
    mag_data = np.moveaxis(mag_data,(0,1,2),(0,2,1))
    quat_data = np.moveaxis(quat_data,(0,1,2),(0,2,1))

    acc_data = np.delete(acc_data[sf:,...], frames_corrupt, axis = 0)
    gyro_data = np.delete(gyro_data[sf:,...], frames_corrupt, axis = 0)
    mag_data = np.delete(mag_data[sf:,...], frames_corrupt, axis = 0)
    quat_data = np.delete(quat_data[sf:,...], frames_corrupt, axis = 0)


    participant_nbr = ''.join(filter(str.isdigit, path))

    np.savez_compressed(path + '\\' + dir_list[i].split('.')[0] + '_' + participant_nbr + '_imu', acc = acc_data, gyro = gyro_data, mag = mag_data, quat = quat_data)
    
    print('File:', dir_list[i].split('.')[0])
    print('acc_data', acc_data.shape)
    print('gyro_data', gyro_data.shape)
    print('mag_data', mag_data.shape)
    print('quat_data', quat_data.shape, '\n')

#======================================================================================================================
#======================================== Functions Synchronization ===================================================
#======================================================================================================================


def synchronize_data(frames_Vicon, frames_IMU):
    if frames_IMU - frames_Vicon >= 0:
        start_frame_imu = frames_IMU - frames_Vicon
        start_frame_vicon = 0
    else:
        start_frame_vicon = frames_Vicon - frames_IMU
        start_frame_imu = 0
    # print('Starting Frame Vicon:', start_frame_vicon, 'Starting Frame IMU:', start_frame_imu)

    return start_frame_vicon, start_frame_imu


def check_for_nan_in_Vicon_data(data_jc, data_ori):
    frms_corrupt = []
    for i in range(data_jc.shape[0]):
        
        if np.isnan(data_jc[i,...]).any():
            frms_corrupt.append(i)

        if np.isnan(data_ori[i,...]).any():
            if i not in frms_corrupt:
                frms_corrupt.append(i)

    frms_corrupt.sort()
    print('Corrupted Frames:', frms_corrupt)
    return frms_corrupt



# check for all .c3d files in the folder:
# dir_list = Vicon_get_c3d_filenames(path + '\\Vicon\\')
dir_list = ['Norm_Pre.c3d', 'Norm_Post.c3d']

IMU_data_matlab = mat73.loadmat(path + '\IMU\GaitSummaryIMU.mat', use_attrdict=True) 

IMU_index_of_files = IMU_get_index_of_files()


for i in range(len(dir_list)):  
    dir = dir_list[i]
    Vicon_data, Vicon_data_labels = Vicon_import_data_from_c3d(path + '\Vicon\\' + dir)
    Vicon_index_jc = Vicon_get_index_of_labels(Vicon_data_labels, Vicon_labels_joint_centers)
    Vicon_index_ori = Vicon_get_index_of_labels(Vicon_data_labels, Vicon_labels_orientation)        
    
    Vicon_data_joint_centers = Vicon_get_coords_of_jc(Vicon_data, Vicon_index_jc)
    Vicon_data_orientation = Vicon_get_coords_of_jc(Vicon_data, Vicon_index_ori)
    
    Vicon_segment_orientations = Vicon_compute_orientation_of_segments(Vicon_data_orientation)
    
    IMU_data = IMU_get_data_from_mat_file('ankle_L', IMU_index_of_files[i], 'quat')
    
    sf_vicon, sf_imu = synchronize_data(Vicon_data_joint_centers.shape[0], IMU_data.shape[0])

    Vicon_data_joint_centers = Vicon_data_joint_centers[sf_vicon:,...]
    Vicon_segment_orientations = Vicon_segment_orientations[sf_vicon:,...]

    frames_corrupt = check_for_nan_in_Vicon_data(Vicon_data_joint_centers, Vicon_segment_orientations)

    Vicon_data_joint_centers = np.delete(Vicon_data_joint_centers, frames_corrupt, axis = 0)
    Vicon_segment_orientations = np.delete(Vicon_segment_orientations, frames_corrupt, axis = 0)


    Vicon_safe_data_to_numpy_array(path, dir, Vicon_data_joint_centers, Vicon_segment_orientations)
    IMU_get_and_save_data_to_numpy(IMU_index_of_files[i], sf_imu, frames_corrupt)
    
print()
print('All files have been converted!')

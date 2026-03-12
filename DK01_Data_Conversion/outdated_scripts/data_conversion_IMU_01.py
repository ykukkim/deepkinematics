import os

import mat73
import numpy as np

# path to the GaitsummaryIMU.mat file
path = r'D:\ZM_Data\SonE_21'

# list of sensor positions
sensor_pos = ['ankle_L', 'ankle_R', 'arm_L', 'arm_R', 'head', 'trunk']
files_to_convert = ['Norm_Pre', 'Norm_Post']

data_matlab = mat73.loadmat(path + '\IMU\GaitSummaryIMU.mat', use_attrdict=True) 


def get_data_from_mat_file(pos, idx, sensor_type):
    tmp = data_matlab['GaitSummaryIMU']['ModulesData'][pos]['synced']['trim'][idx][sensor_type]
    return tmp

def create_numpy_folder(path):
    os.makedirs(path + '\\numpy_arrays', exist_ok= True)

def get_index_of_files():       # search for Norm_Pre and Norm_Post files (should have around 75k to 85k frames)
    tmp = data_matlab['GaitSummaryIMU']['ModulesData']['ankle_L']['synced']['trim']
    index = []
    for i in range(len(tmp)):
        if tmp[i]['acc'].shape[0] > 75000 and tmp[i]['acc'].shape[0] < 85000:
            index.append(i)
    
    return index

index_of_files = get_index_of_files()

for i in range(len(files_to_convert)):       

    for pos in sensor_pos:
        
        if pos == sensor_pos[0]:
            acc_data = get_data_from_mat_file(pos, index_of_files[i], 'acc')
            gyro_data = get_data_from_mat_file(pos, index_of_files[i], 'gyro')
            mag_data = get_data_from_mat_file(pos, index_of_files[i], 'mag')
            quat_data = get_data_from_mat_file(pos, index_of_files[i], 'quat')
                        
        else:
            acc_data = np.dstack((acc_data, get_data_from_mat_file(pos, index_of_files[i], 'acc')))
            gyro_data = np.dstack((gyro_data, get_data_from_mat_file(pos, index_of_files[i], 'gyro')))
            mag_data = np.dstack((mag_data, get_data_from_mat_file(pos, index_of_files[i], 'mag')))
            quat_data = np.dstack((quat_data, get_data_from_mat_file(pos, index_of_files[i], 'quat')))           
        
    
    acc_data = np.moveaxis(acc_data,(0,1,2),(0,2,1))
    gyro_data = np.moveaxis(gyro_data,(0,1,2),(0,2,1))
    mag_data = np.moveaxis(mag_data,(0,1,2),(0,2,1))
    quat_data = np.moveaxis(quat_data,(0,1,2),(0,2,1))

    participant_nbr = ''.join(filter(str.isdigit, path))

    np.savez_compressed(path + '\\' + files_to_convert[i] + '_' + participant_nbr + '_imu', acc = acc_data, gyro = gyro_data, mag = mag_data, quat = quat_data)
    
    print('File:', files_to_convert[i])
    print('acc_data', acc_data.shape)
    print('gyro_data', gyro_data.shape)
    print('mag_data', mag_data.shape)
    print('quat_data', quat_data.shape, '\n')

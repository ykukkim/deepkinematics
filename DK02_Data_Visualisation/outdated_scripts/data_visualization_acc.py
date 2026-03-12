import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from DK00_Utils import DK00_UT00_config as config_v

Particiapnt_idx = 22
Particiapnt_idx_2 = 0
trial_idx = 0

# Create path to the vicon and IMU npz-files of subject 1 & 2
path_vicon_files = f'{config_v.dir_path}/SonE_{str(config_v.subject[Particiapnt_idx]).zfill(2)}/{config_v.trials[trial_idx]}_{str(config_v.subject[Particiapnt_idx]).zfill(2)}_vicon.npz'
path_imu_files = f'{config_v.dir_path}/SonE_{str(config_v.subject[Particiapnt_idx]).zfill(2)}/{config_v.trials[trial_idx]}_{str(config_v.subject[Particiapnt_idx]).zfill(2)}_imu.npz'

def load_IMU_acc_data(path):
    data = np.load(path)
    acc_data = data['acc']
    return acc_data

def load_vicon_data(path):
    """ load joint center and segment orientation from npz files """
    data = np.load(path)
    data_jc = data['DK03_JC_00_functions']
    data_ori = data['ori']
    
    print('Vicon joint center:', data_jc.shape)
    print('Vicon segment orientation:', data_ori.shape)

    return data_jc, data_ori

def print_acc(data, start_frame, end_frame, sensor, name):
    plt.figure(name)
    plt.title('Left Ankle')
    
    plt.subplot(311)
    plt.plot(data[start_frame:end_frame,sensor,0])
    plt.ylim(bottom=-6, top=6)
    plt.title('X-acc')

    plt.subplot(312)
    plt.plot(data[start_frame:end_frame,sensor,1])
    plt.ylim(bottom=-6, top=6)
    plt.title('Y-acc')

    plt.subplot(313)
    plt.plot(data[start_frame:end_frame,sensor,2])
    plt.ylim(bottom=-6, top=6)
    plt.title('Z-acc')

    plt.subplots_adjust(hspace=0.6, wspace=0.35)
    plt.show(block=False)

acc = load_IMU_acc_data(path_imu_files)
print(acc.shape)

data_vicon_jc, data_vicon_ori = load_vicon_data(path_vicon_files)

# HS_left = np.load(path_vicon_files)['lcp'][:,1]
# HS_contact_bool = HS_left.astype(dtype=bool)
# # data_jc = data['DK03_JC_00_functions']
# # data_ori = data['ori']

sf = 0
ef = -1
# end_frame_plot = end_frame - start_frame

# plt.plot(np.linalg.norm(acc[:,[0],:], axis=2))
# plt.show()

# left ankle
# print_acc(acc,start_frame, end_frame, 0, 'left ankle')

# right ankle
# print_acc(acc,start_frame, end_frame, 1, 'right ankle')

# left hand
# print_acc(acc,start_frame, end_frame, 2, 'left hand')

# right hand
# print_acc(acc,start_frame, end_frame, 3, 'right hand')

# head
# print_acc(acc,start_frame, end_frame, 4, 'head')

# trunk
# print_acc(acc,start_frame, end_frame, 5, 'trunk')


# input("hit[enter] to end.")
# plt.close("all")

""" Left Heel """
plt.figure('Left Heel Acc-Pos')
plt.title('Left Ankle')

plt.subplot(411)
plt.plot(acc[sf:ef,0,0])
plt.ylim(bottom=-6, top=6)
plt.title('X-acc')

plt.subplot(412)
plt.plot(acc[sf:ef,0,1])
plt.ylim(bottom=-6, top=6)
plt.title('Y-acc')

plt.subplot(413)
plt.plot(acc[sf:ef,0,2])
plt.ylim(bottom=-6, top=6)
plt.title('Z-acc')

plt.subplot(414)
plt.plot(data_vicon_jc[sf:ef,11,2])
plt.title('Vicon-Z left anklle')

""" Left Toe """
plt.figure('Left Toe Acc-Pos')
plt.title('Left Toe')

plt.subplot(411)
plt.plot(acc[sf:ef,0,0])
plt.ylim(bottom=-6, top=6)
plt.title('X-acc')

plt.subplot(412)
plt.plot(acc[sf:ef,0,1])
plt.ylim(bottom=-6, top=6)
plt.title('Y-acc')

plt.subplot(413)
plt.plot(acc[sf:ef,0,2])
plt.ylim(bottom=-6, top=6)
plt.title('Z-acc')

plt.subplot(414)
plt.plot(data_vicon_jc[sf:ef,13,2])
plt.title('Vicon-Z right Heel')

# plt.show()

plt.figure('Left Heel')
plt.title('Left Heel')
plt.subplot(211)
plt.plot(np.linalg.norm((acc[sf:ef:,0,0], acc[sf:ef:,0,1], acc[sf:ef:,0,2]), axis = 0))

plt.subplot(212)
plt.plot(data_vicon_jc[sf:ef:,11,2])
# plt.plot(np.arange(sf,ef)[HS_contact_bool[sf:ef]], data_vicon_jc[sf:ef,11,2][HS_contact_bool[sf:ef]], "x")
# plt.show()

plt.figure('Left Toe')
plt.title('Left Toe')
plt.subplot(211)
plt.plot(np.linalg.norm((acc[sf:ef,0,0], acc[sf:ef,0,1], acc[sf:ef,0,2]), axis = 0))

plt.subplot(212)
plt.plot(data_vicon_jc[sf:ef,13,2])
# plt.plot(np.arange(sf,ef)[HS_contact_bool[sf:ef]], data_vicon_jc[sf:ef,11,2][HS_contact_bool[sf:ef]], "x")
plt.show()
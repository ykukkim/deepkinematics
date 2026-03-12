import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg'
import matplotlib.pyplot as plt

from DK00_Utils import DK00_UT00_config as config_v
from DK00_Utils.DK00_UT00_utils import z_score_normalise

def load_data(file_path, data_key):
    """Load data from an npz file."""
    data = np.load(file_path)
    return data[data_key]

def plot_sensor_data(data, sensor_index, sensor_name, start_frame, end_frame, vicon_data):
    """
    Plot X, Y, Z accelerometer data for a specified sensor.
    If sensor name contains 'Left' or 'Right', plot Vicon's Z-axis data for the respective heel.
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'{sensor_name} Acc-Pos')

    for i, axis in enumerate(['X', 'Y', 'Z']):
        axs[i].plot(data[start_frame:end_frame, sensor_index, i])
        axs[i].set_ylim(bottom=-6, top=6)
        axs[i].set_title(f'{axis}-acc')
        axs[i].set_ylabel('Acceleration')

    # Plot Vicon's Z-axis data for left or right heel
    if 'Left' in sensor_name:
        axs[3].plot(vicon_data[start_frame:end_frame, 11, 2])  # Left heel index 11
        axs[3].set_title('Vicon-Z left heel')
        axs[3].set_ylabel('Position [m]')
    elif 'Right' in sensor_name:
        axs[3].plot(vicon_data[start_frame:end_frame, 13, 2])  # Right heel index 13
        axs[3].set_title('Vicon-Z right heel')
        axs[3].set_ylabel('Position [m]')

    axs[3].set_xlabel('Samples [n]')
    plt.subplots_adjust(hspace=0.6)
    plt.show()

def plot_combined_data(acc_data, vicon_data, sensor_index, vicon_index, title, start_frame, end_frame):
    """
    Plot the norm of acceleration data and Vicon data for a specified sensor.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(title)

    # Plot norm of acceleration data
    axs[0].plot(np.linalg.norm(acc_data[start_frame:end_frame, sensor_index, :], axis=1))
    axs[0].set_title('Norm of Acceleration')
    axs[0].set_ylabel('Acceleration')

    # Plot Vicon data
    axs[1].plot(vicon_data[start_frame:end_frame, vicon_index, 2])
    axs[1].set_title('Vicon-Z Data')
    axs[1].set_ylabel('Position [m]')
    axs[1].set_xlabel('Samples [n]')

    plt.subplots_adjust(hspace=0.4)
    plt.show()

if __name__ == "__main__":

    for subject in [44]:#range(0,76):
        if subject in [5,38,39,52,55,57,58,60,68]:
            # These subjects are skipped due to bad walking, missing VICON markers
            # or missing IMU synchronization
            continue

        subject_nr = str(subject).zfill(2)
        print('Subject: ', subject_nr)
        for file in config_v.trials:

            path_vicon_files = f'{config_v.dir_path}/SonE_{subject_nr}/{file}_{subject_nr}_vicon.npz'
            path_imu_files = f'{config_v.dir_path}/SonE_{subject_nr}/fk/{file}_{subject_nr}_imu.npz'
            path_fk_files = f'{config_v.dir_path}/SonE_{subject_nr}/fk/{file}_{subject_nr}_position.npz'
            print(f'SonE_{subject_nr} - {file}')

            # Sampling factors
            sf_bvh = int(100 / 50)  # Sampling factor for bvh
            sf_imu = int(200 / 50)  # Sampling factor for imu

            # Load data
            acc_data = load_data(path_imu_files, 'acc')[::sf_imu, ...]
            pose_data = load_data(path_fk_files, 'position')[::sf_bvh, ...]
            vicon_data = load_data(path_vicon_files, 'jc')[::sf_imu, ...]
            vicon_data_fk = np.load(path_vicon_files)
            vicon_contact = np.concatenate((vicon_data_fk['lcp'][::sf_imu, ...], vicon_data_fk['rcp'][::sf_imu, ...]), axis=1)

            min_length = min(acc_data.shape[0], vicon_contact.shape[0])

            # Trim IMU data to the minimum length
            acc_data = acc_data[:min_length, ...]
            pose_data = pose_data[:min_length,...]

            # Trim Vicon data to the minimum length
            vicon_contact = vicon_contact[:min_length, ...]
            vicon_data = vicon_data[:min_length,...]

            # Plot combined data for right foot
            sensor_data = acc_data[:, 0, :]
            sensor_normalised = z_score_normalise(sensor_data)

            # Set font properties
            font = {'family': 'Arial',
                    'weight': 'normal',
                    'size': 22}

            plt.rc('font', **font)

            # Create the plot
            fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

            # Right Foot - JC
            axs[0].plot(vicon_data[:, 12, 2], label='Heel Trajectory (z-axis)', color='blue')
            axs[0].plot(np.arange(vicon_data.shape[0])[vicon_contact[:, 3] == 1], vicon_data[vicon_contact[:, 3] == 1, 12, 2],
                        "x", label='Contact', color='orange')
            axs[0].set_title('VICON - Heel Trajectory (z-axis)')
            axs[0].set_ylabel('[m]')
            axs[0].legend()

            # Right Foot - VT Acc
            axs[1].plot(sensor_normalised[:, 2], label='Heel Trajectory (z-axis)', color='blue')
            axs[1].plot(np.arange(vicon_data.shape[0])[vicon_contact[:, 3] == 1], vicon_data[vicon_contact[:, 3] == 1, 12, 2],
                        "x", label='Contact', color='orange')
            axs[1].set_title('IMU - Heel Trajectory (z-axis)')
            axs[1].set_ylabel('[m]')
            axs[1].set_xlabel('Frames')
            axs[1].legend()

            # Right Foot - FK
            axs[2].plot(pose_data[:, 23, 2], label='Heel Trajectory (z-axis)', color='blue')
            axs[2].plot(np.arange(pose_data.shape[0])[vicon_contact[:, 3] == 1], pose_data[vicon_contact[:, 3] == 1, 23, 2],
                        "x", label='Contact', color='orange')
            axs[2].set_title('BVH - Heel Trajectory (z-axis)')
            axs[2].set_ylabel('[m]')
            axs[2].legend()

            # Adjust layout and show plot
            plt.subplots_adjust(hspace=0.4)
            plt.ion()  # Turns on interactive mode
            plt.show()


            #    # Plot data for different IMU sensors
            # sensor_names = ['Left Ankle', 'Right Ankle', 'Left Hand', 'Right Hand', 'Head', 'Trunk']  # 0,1,2,3,4,5
            # for sensor_index, sensor_name in enumerate(sensor_names):
            #     # Extract specific sensor data
            #     sensor_data = acc_data[:, sensor_index, :]
            #     # Apply Z-Score Standardisation
            #     sensor_normalised = z_score_normalise(sensor_data)
            #     # Plot the data for the specific sensor
            #     plot_sensor_data(sensor_normalised, sensor_index, sensor_name, start_frame, end_frame, pose_data)
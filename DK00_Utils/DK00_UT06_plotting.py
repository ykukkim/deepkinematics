import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R



# Function to convert rotation matrices to quaternions
def rotation_matrices_to_euler_angles(rotation_matrices):
    euler_angles = np.zeros((rotation_matrices.shape[0], rotation_matrices.shape[1], rotation_matrices.shape[2], 3))
    for i in range(rotation_matrices.shape[0]):
        for j in range(rotation_matrices.shape[1]):
            for k in range(rotation_matrices.shape[2]):
                r = R.from_matrix(rotation_matrices[i, j, k])
                euler_angles[i, j, k] = r.as_euler('xyz', degrees=True)  # 'xyz' is the rotation order
    return euler_angles

# Function to plot joint orientations
def plot_euler_angles(euler_angles, euler_angles_gt, joint_idx=0):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Joint {joint_idx} Euler Angles Over Time')

    time = np.arange(euler_angles.shape[1])

    # Plot Roll
    ax[0].plot(time, euler_angles[0, :, joint_idx, 0], label='Estimated Roll')
    ax[0].plot(time, euler_angles_gt[0, :, joint_idx, 0], label='Ground Truth Roll', linestyle='dashed')
    ax[0].set_ylabel('Roll (degrees)')
    ax[0].legend()

    # Plot Pitch
    ax[1].plot(time, euler_angles[0, :, joint_idx, 1], label='Estimated Pitch')
    ax[1].plot(time, euler_angles_gt[0, :, joint_idx, 1], label='Ground Truth Pitch', linestyle='dashed')
    ax[1].set_ylabel('Pitch (degrees)')
    ax[1].legend()

    # Plot Yaw
    ax[2].plot(time, euler_angles[0, :, joint_idx, 2], label='Estimated Yaw')
    ax[2].plot(time, euler_angles_gt[0, :, joint_idx, 2], label='Ground Truth Yaw', linestyle='dashed')
    ax[2].set_ylabel('Yaw (degrees)')
    ax[2].set_xlabel('Frame')
    ax[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_pose_angles(pose, pose_gt, joint_idx=0):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Joint {joint_idx} Euler Angles Over Time')

    time = np.arange(pose.shape[0])
    # time = np.arange(pose.shape[1])

    # Plot Roll
    ax[0].plot(time, pose[0, :, joint_idx, 0], label='Estimated X')
    ax[0].plot(time, pose_gt[0, :, joint_idx, 0], label='Ground Truth X', linestyle='dashed')
    ax[0].set_ylabel('X (degrees)')
    ax[0].legend()

    # Plot Pitch
    ax[1].plot(time, pose[0, :, joint_idx, 1], label='Estimated Y')
    ax[1].plot(time, pose_gt[0, :, joint_idx, 1], label='Ground Truth Y', linestyle='dashed')
    ax[1].set_ylabel('Y (degrees)')
    ax[1].legend()

    # Plot Yaw
    ax[2].plot(time, pose[0, :, joint_idx, 2], label='Estimated Z')
    ax[2].plot(time, pose_gt[0, :, joint_idx, 2], label='Ground Truth Z', linestyle='dashed')
    ax[2].set_ylabel('Yaw (degrees)')
    ax[2].set_xlabel('Z')
    ax[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
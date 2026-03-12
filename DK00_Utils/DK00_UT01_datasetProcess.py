import torch
import numpy as np
import scipy.signal as signal
from scipy.signal import hilbert

from torch.nn.utils.rnn import pad_sequence
from scipy.spatial.transform import Rotation as R

from DK00_Utils.DK00_UT00_config import CONSTANTS as C

class RealSample(object):
    """
    Wrapper to store IMU data with corresponding ground-truth data.
    Supports both Vicon Joint Center and Segment Orientations (version 1)
    and joint positions, joint rotations, and bone offsets for forward kinematics (version 2).
    """

    def __init__(self, idx_split, imu_acc, imu_gyro, imu_ori,
                 vicon_jc=None, vicon_ori=None, vicon_contact=None,
                 pose=None, velocity=None, joint_rotations=None,
                 relative_joints = None, relative_phase = None,
                 bone_offsets=None,
                 root=None, ori_offsets=None, version=None, **kwargs):
        """
        Initialize the RealSample object.

        Parameters depend on the version used.

        Common parameters:
        :param idx_split: Index of train/test split of unseen motion sequence.
        :param imu_acc: Acceleration (np.array of shape (F, N_SENSORS, 3)).
        :param imu_gyro: Angular velocity (np.array of shape (F, N_SENSORS, 3)).
        :param imu_ori: Sensor orientation (np.array of shape (F, N_IMU_Sensors, 3, 3) or (F, N_IMU_Sensors, 4)).

        Version 1 parameters:
        :param vicon_jc: Vicon joint center (np.array of shape (F, N_MARKERS, 3)).
        :param vicon_ori: Vicon segment orientation (np.array of shape (F, N_Vicon_Segments, 3, 3)).
        :param vicon_contact: Contact phase (np.array of shape (F, 4)).

        Version 2 parameters:
        :param pose: Joint positions (np.array of shape (N, N_JOINTS, 3)).
        :param joint_rotations: Joint rotations (intrinsic, np.array of shape (N, N_JOINTS, 3, 3)).
        :param bone_offsets: Bone offsets (np.array of shape (N_JOINTS, 3)).

        Shared parameters:
        :param root: Midpoint between hip joints (optional).
        :param ori_offsets: Initial offset of IMU orientation to Vicon segment orientation (optional).
        :param version: Data version ('JC' for Joint Center or 'FK' for Forward Kinematics).
        """
        # Sequence ID and Frames
        self.trial_name = kwargs.get('trial')
        self.subject_nr = kwargs.get('subject')
        self.idx_split = idx_split
        self.VERSION = version

        # IMU Data
        self.imu_acc = imu_acc
        self.imu_gyro = imu_gyro
        self.imu_ori = imu_ori
        self.ori_offsets = ori_offsets
        self.velocity = velocity

        # Vicon Data
        self.vicon_jc = vicon_jc # pose without root
        self.vicon_ori = vicon_ori
        self.vicon_contact = vicon_contact

        # root data for both JC and FK
        self.root = root

        # Determine the number of frames
        self.n_frames = vicon_jc.shape[0] if self.VERSION == 'JC' else pose.shape[0] if pose is not None else 0

        if self.VERSION == 'FK':
            self.pose = pose # pose without root
            self.joint_rotations = joint_rotations
            self.bone_offsets = bone_offsets
            self.relative_joints = relative_joints
            self.relative_phases = relative_phase

    @staticmethod
    def z_score_normalize(data):
        """
        Perform z-score normalization on the data.

        :param data: Data to be normalized (np.array).
        :return: Normalized data.
        """
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        return (data - mean) / std

    @staticmethod
    def bandpass_filter(data, sf, lowcut, highcut):
        """
        Apply a bandpass filter to the data.

        :param data: Input data to be filtered (np.array).
        :param sf: Sampling frequency in Hz.
        :param lowcut: Low cutoff frequency in Hz.
        :param highcut: High cutoff frequency in Hz.
        :return: Filtered data.
        """
        nyquist = 0.5 * sf
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        padlen = 3 * max(len(b), len(a))

        if data.shape[0] <= padlen:
            raise ValueError(f"The length of the input vector must be greater than padlen ({padlen}).")

        return signal.filtfilt(b, a, data, axis=0)

    def process_data(self, sf, lowcut=0.1, highcut=15):
        """
        Filter and normalize IMU accelerometer and gyroscope data.

        :param sf: Sampling frequency in Hz.
        :param lowcut: Low cutoff frequency for the filter in Hz.
        :param highcut: High cutoff frequency for the filter in Hz.
        """
        # # Filter and normalize IMU Accelerometer data
        # self.imu_acc = self.z_score_normalize(self.bandpass_filter(self.imu_acc, sf, lowcut, highcut))
        #
        # # Filter and normalize IMU Gyroscope data
        # self.imu_gyro = self.z_score_normalize(self.bandpass_filter(self.imu_gyro, sf, lowcut, highcut))

        # Filter and normalize IMU Accelerometer data
        self.imu_acc = self.bandpass_filter(self.imu_acc, sf, lowcut, highcut)

        # Filter and normalize IMU Gyroscope data
        self.imu_gyro = self.bandpass_filter(self.imu_gyro, sf, lowcut, highcut)
        self.imu_acc = np.copy(self.imu_acc)  # Make sure the array has positive strides
        self.imu_gyro = np.copy(self.imu_gyro)  # Make sure the array has positive strides

    @classmethod
    def calculate_velocity(cls, positions, sampling_rate=50):
        """
        Calculate velocity from positions using finite differences.

        :param positions: Positions (np.array).
        :param sampling_rate: Sampling rate in Hz.
        :return: Velocities (np.array).
        """
        delta_t = 1.0 / sampling_rate
        displacements = np.diff(positions, axis=0)
        velocities = displacements / delta_t
        return np.vstack((np.zeros((1, velocities.shape[1], velocities.shape[2])), velocities))

    @classmethod
    def compute_relative_angle_for_skeleton(self, rotations, convention='xyz', in_degrees=False):
        """
         Compute the relative Euler angles for the skeleton given the rotations.

         Args:
             rotations (np.ndarray): Array of shape (f, n_joints, 3, 3) containing the rotation matrices for each joint.
             convention (str): The convention used for Euler angles (e.g., 'xyz', 'zyx').
             in_degrees (bool): If True, convert the angle difference from radians to degrees.

         Returns:
             np.ndarray: Array of shape (f, len(joint_pairs), 3) containing the relative Euler angles for each axis.
         """
        f, n_joints, _, _ = rotations.shape
        euler_all = []

        for joint in range(n_joints):
            convention = C.JOINT_EULER_CONVENTIONS.get(C.FK_JOINTS_FUll[joint], 'xyz')
            r = R.from_matrix(rotations[:, joint])  # shape: (f, 3, 3)
            euler = r.as_euler(convention, degrees=in_degrees)  # shape: (f, 3)

            # Store the Euler angles
            euler_all.append(euler)

        return np.stack(euler_all, axis=1)  # shape: (f, n_joints, 3)

    @classmethod
    def compute_relative_phase_for_skeleton(self, rotations, convention='xyz', in_degrees=False):
        """
        Compute the instantaneous joint phase for each joint using Euler angles.

        Args:
            rotations (np.ndarray): Array of shape (f, n_joints, 3, 3) with local rotation matrices.
            convention (str): Euler angle convention for decomposition (e.g., 'xyz').
            axis (int): Axis index to extract phase from (0: x, 1: y, 2: z).
            in_degrees (bool): If True, output Euler angles and phase in degrees.

        Returns:
            np.ndarray: Array of shape (f, n_joints) containing joint phases on the selected axis.
        """
        f, n_joints, _, _ = rotations.shape
        phase_angles = []
        axis = 0

        for joint in range(n_joints):
            # Determine Euler convention for this joint (if different per joint)
            convention = C.JOINT_EULER_CONVENTIONS.get(C.FK_JOINTS_FUll[joint], 'xyz')

            # Convert rotation matrices to Euler angles
            r = R.from_matrix(rotations[:, joint])
            euler = r.as_euler(convention, degrees=False)  # always compute in radians for phase

            # Extract and normalise signal along specified axis
            signal = euler[:, axis] - np.mean(euler[:, axis])

            # Compute analytic signal and extract instantaneous phase
            analytic = hilbert(signal)
            phase = np.angle(analytic)  # values in [-π, π]

            phase_angles.append(phase)

        # Stack into shape (f, n_joints)
        joint_phases = np.stack(phase_angles, axis=1)

        # Optionally convert to degrees
        if in_degrees:
            joint_phases = np.degrees(joint_phases)

        return joint_phases

    @classmethod
    def from_npz(cls, path, npz_file, sf, version, idx_split=None):
        """
        Load IMU and BVH data from disk, downsample it, and create a RealSample object.

        :param path: Path to the data.
        :param npz_file: Tuple containing (subject number, trial name).
        :param sf: Desired sampling frequency in Hz (max 100Hz).
        :param version: Data version ('JC' for Joint Center or 'FK' for Forward Kinematics).
        :param idx_split: Index specifying the train/test split.
        :return: RealSample object.
        """
        subject, trial = npz_file
        id = {'subject': subject, 'trial': trial}

        # Sampling factor IMU/bvh
        sf_bvh = int(100 / sf)  # Sampling factor for bvh
        sf_imu = int(200 / sf)  # Sampling factor for imu

        # Load Vicon data
        vicon = np.load(f'{path}/SonE_{subject}/{trial}_{subject}_vicon.npz')
        vicon_jc = vicon['jc'][::sf_imu, ...]
        vicon_ori = vicon['ori'][::sf_imu, ...]
        vicon_contact = np.concatenate((vicon['lcp'][::sf_imu, ...], vicon['rcp'][::sf_imu, ...]), axis=1)

        if version == 'JC':
            imu = np.load(f'{path}/SonE_{subject}/{trial}_{subject}_imu.npz')

            # Load IMU data
            imu_acc = imu['acc'][::sf_imu, ...]
            imu_gyro = imu['gyro'][::sf_imu, ...]
            imu_ori = imu['quat'][::sf_imu, ...]

            velocity = cls.calculate_velocity(vicon_jc, sampling_rate=sf_imu)

            return cls(idx_split, imu_acc, imu_gyro, imu_ori, vicon_jc=vicon_jc,
                       vicon_ori=vicon_ori, vicon_contact=vicon_contact,
                       velocity=velocity, version=version, **id)

        elif version == 'FK':
            imu = np.load(f'{path}/SonE_{subject}/fk/{trial}_{subject}_imu.npz')
            pose = np.load(f'{path}/SonE_{subject}/fk/{trial}_{subject}_position.npz')
            joint_rotations = np.load(f'{path}/SonE_{subject}/fk/{trial}_{subject}_joint_rotation.npz')
            skeleton = np.load(f'{path}/SonE_{subject}/fk/SonE_{subject}_skeleton.npz')

            # Load IMU data
            imu_acc = imu['acc'][::sf_imu, ...]
            imu_gyro = imu['gyro'][::sf_imu, ...]
            imu_ori = imu['quat'][::sf_imu, ...]

            # Load BVH data
            pose = pose['position'][::sf_bvh, ...]
            joint_rotations = joint_rotations['joint_rotations'][::sf_bvh, ...]
            bone_offsets = skeleton['offset']

            min_length = min(imu_acc.shape[0], vicon_ori.shape[0])

            # Trim data to the minimum length
            imu_acc, imu_gyro, imu_ori = imu_acc[:min_length], imu_gyro[:min_length], imu_ori[:min_length]
            vicon_ori, vicon_contact = vicon_ori[:min_length], vicon_contact[:min_length]
            pose, joint_rotations = pose[:min_length], joint_rotations[:min_length]

            velocity = cls.calculate_velocity(pose, sampling_rate=sf_bvh)
            relative_joints = cls.compute_relative_angle_for_skeleton(joint_rotations)
            relative_phase = cls.compute_relative_phase_for_skeleton(joint_rotations)

            # Create the RealSample object
            obj = cls(idx_split, imu_acc, imu_gyro, imu_ori,
                      vicon_ori=vicon_ori, vicon_contact=vicon_contact,
                      pose=pose, velocity=velocity, joint_rotations=joint_rotations,
                      relative_joints = relative_joints, relative_phase = relative_phase,
                      bone_offsets=bone_offsets, version=version, sf=sf, **id)

            # IMU data processing and relative phase calculation for FK version
            obj.process_data(sf, lowcut=0.1, highcut=10)

            return obj

        else:
            raise ValueError(f"Invalid version: {version}. Expected 'JC' or 'FK'.")

    def extract_window(self, start_frame, end_frame, window_size):
        """Extract a subsequence from `start_frame` to `end_frame` (non-inclusive)."""
        sf, ef = start_frame, end_frame

        idx_split = self.idx_split

        imu_acc = self.imu_acc[sf:ef, ...]
        imu_gyro = self.imu_gyro[sf:ef, ...]
        imu_ori = self.imu_ori[sf:ef, ...]

        vicon_ori = self.vicon_ori[sf:ef, ...]
        vicon_contact = self.vicon_contact[sf:ef, ...]
        velocity = self.velocity[sf:ef, ...]
        ori_offsets = self.ori_offsets

        if self.VERSION == 'JC':
            vicon_jc = self.vicon_jc[sf:ef, ...]
            rs = RealSample(idx_split=idx_split, imu_acc=imu_acc, imu_gyro=imu_gyro, imu_ori=imu_ori,
                            vicon_jc=vicon_jc, vicon_ori=vicon_ori, vicon_contact=vicon_contact,
                            velocity=velocity, ori_offsets=ori_offsets, trial=self.trial_name, version=self.VERSION,
                            subject=self.subject_nr)

        elif self.VERSION == 'FK':
            pose = self.pose[sf:ef, ...]
            joint_rotations = self.joint_rotations[sf:ef, ...]
            relative_joints = self.relative_joints[sf:ef, ...]
            relative_phase = self.relative_phases[sf:ef, ...]
            bone_offsets = self.bone_offsets
            rs = RealSample(idx_split=idx_split, imu_acc=imu_acc, imu_gyro=imu_gyro, imu_ori=imu_ori,
                            vicon_ori=vicon_ori, vicon_contact=vicon_contact, velocity=velocity, ori_offsets=ori_offsets,
                            pose=pose, joint_rotations=joint_rotations,
                            relative_joints = relative_joints, relative_phase = relative_phase,
                            bone_offsets=bone_offsets,
                            trial=self.trial_name, version=self.VERSION, subject=self.subject_nr)
        return rs

    def align_imu_orientation(self):
        """ Align IMU Orientation to the vicon/bvh segment orientation. """
        self.imu_ori = np.matmul(self.imu_ori, self.ori_offsets)

    def correct_left_ankle_orientation(self):
        """ Rotate left ankle orientation by 180 degrees around z-axis. """
        if self.trial_name == 'Norm_Pre' and self.subject_nr in C.LEFT_FOOT_MOVEMENT_CORRECTION:
            quat_data = self.imu_ori[:, 0, [1, 2, 3, 0]]  # adjust order of quaternions to scipy module (x,y,z,w)
            rot_data = R.from_quat(quat_data).as_matrix()
            rot_data= np.matmul([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], rot_data)
            self.imu_ori[:, 0, ...] = R.from_matrix(rot_data).as_quat()

    def transform_quat_to_rot_vector(self):
        """ Transform quaternions to rotation vectors."""
        quat = self.imu_ori
        f, s = quat.shape[0], quat.shape[1]
        quat = quat.reshape(f, -1, 4)[:, :, [1, 2, 3, 0]]  # adjust order of quaternions to scipy module (x,y,z,w)

        rot_mat = np.empty((f, s, 3))
        for i in range(s):
            rot_mat[:, i, ...] = R.from_quat(quat[:, i, :]).as_rotvec()

        self.imu_ori = rot_mat

    def transform_quat_to_rot_matrix(self):
        """ Transform quaternions to rotation matrices."""
        # 0 = ankel_L, 1 = ankle_R, 2 = arm_L, 3 = arm_R, 4 = head, 5 = trunk
        quat = self.imu_ori
        f, s = quat.shape[0], quat.shape[1]
        quat = quat.reshape(f, -1, 4)[:, :, [1, 2, 3, 0]]  # adjust order of quaternions to scipy module (x,y,z,w)

        rot_mat = np.empty((f, s, 3, 3))
        for i in range(s):
            rot_mat[:, i, ...] = R.from_quat(quat[:, i, :]).as_matrix()

        self.imu_ori = rot_mat

    def to_tensor(self):
        """Convert numpy arrays to torch tensor and flatten to (windowsize, -1)."""
        ws = self.n_frames  # windowsize
        self.imu_acc = torch.from_numpy(self.imu_acc.reshape(ws, -1)).to(C.DTYPE)
        self.imu_gyro = torch.from_numpy(self.imu_gyro.reshape(ws, -1)).to(C.DTYPE)
        self.imu_ori = torch.from_numpy(self.imu_ori.reshape(ws, -1)).to(C.DTYPE)

        self.vicon_contact = torch.from_numpy(self.vicon_contact.reshape(ws, -1)).to(C.DTYPE)
        self.vicon_ori = torch.from_numpy(self.vicon_ori.reshape(ws, -1)).to(C.DTYPE)
        self.velocity = torch.from_numpy(self.velocity.reshape(ws, -1)).to(C.DTYPE)

        if hasattr(self, 'root') and self.root is not None:
            self.root = torch.from_numpy(self.root.reshape(ws, -1)).to(C.DTYPE)

        if self.VERSION == 'JC':
            self.vicon_jc = torch.from_numpy(self.vicon_jc.reshape(ws, -1)).to(C.DTYPE)

        elif self.VERSION == 'FK':
            self.pose = torch.from_numpy(self.pose.reshape(ws, -1)).to(C.DTYPE)
            self.joint_rotations = torch.from_numpy(self.joint_rotations.reshape(ws, -1)).to(C.DTYPE)
            self.relative_joints = torch.from_numpy(self.relative_joints).to(C.DTYPE)
            self.relative_phases = torch.from_numpy(self.relative_phases).to(C.DTYPE)
            self.bone_offsets = torch.from_numpy(self.bone_offsets).to(C.DTYPE)

    "Joint Centre Approach"

    def normalize_root(self):
        """ Normalize IMU orientations with IMU sensor located at the trunk """
        # R_new = inv(R_pelvis) @ R_old
        inv = np.linalg.inv(self.imu_ori[:, 5, ...])
        for i in range(self.imu_ori.shape[1]):
            self.imu_ori[:, i, ...] = np.matmul(inv, self.imu_ori[:, i, ...])

    def normalize_vicon_jc(self, lh, rh):
        """ Fix midpoint between hip joints to the origin. """
        left_hip = self.vicon_jc[:, lh, :]
        right_hip = self.vicon_jc[:, rh, :]
        hip_midpoint = np.mean(np.array([left_hip, right_hip]), axis=0)
        hip_midpoint = hip_midpoint[:, np.newaxis, :]
        self.root = hip_midpoint
        self.vicon_jc = self.vicon_jc - hip_midpoint

    def normalize_left_thigh(self, lh, lk):
        """ Normalize all subjects to a femur length of 45 cm. """
        femur_length = 0.45
        scaling_factor = femur_length / np.linalg.norm(self.vicon_jc[0, lh, :] - self.vicon_jc[0, lk, :])
        self.vicon_jc = self.vicon_jc * scaling_factor

    def rotate_vicon_orientation(self):
        """ Rotate the VICON segment orientation to represent the expected IMU orientation. """

        # Left and right foot
        self.vicon_ori[:, 0:2, ...] = np.matmul(self.vicon_ori[:, 0:2, ...],
                                                R.from_euler('x', 180, degrees=True).as_matrix())
        self.vicon_ori[:, 0:2, ...] = np.matmul(self.vicon_ori[:, 0:2, ...],
                                                R.from_euler('y', -30, degrees=True).as_matrix())
        # Left & right arm
        self.vicon_ori[:, 2:4, ...] = np.matmul(self.vicon_ori[:, 2:4, ...],
                                                R.from_euler('z', -90, degrees=True).as_matrix())
        # Left arm
        self.vicon_ori[:, 2:3, ...] = np.matmul(self.vicon_ori[:, 2:3, ...],
                                                R.from_euler('x', -180, degrees=True).as_matrix())
        # Head
        self.vicon_ori[:, 4, ...] = np.matmul(self.vicon_ori[:, 4, ...],
                                              R.from_euler('y', 90, degrees=True).as_matrix())
        # Trunk
        self.vicon_ori[:, 5, ...] = np.matmul(self.vicon_ori[:, 5, ...],
                                              R.from_euler('x', 180, degrees=True).as_matrix())
        self.vicon_ori[:, 5, ...] = np.matmul(self.vicon_ori[:, 5, ...],
                                              R.from_euler('y', 90, degrees=True).as_matrix())


class RealBatch(object):
    """
    Collate `RealSample`s into batches.
    """

    def __init__(self, seq_lengths=None, imu_acc=None, imu_gyro=None, imu_ori=None,
                 vicon_jc=None, vicon_ori=None, vicon_contact=None,
                 root=None, pose=None, velocity=None, joint_rotations=None,
                 relative_joints = None, relative_phase = None,
                 bone_offsets=None, VERSION=None):
        """
        Initializer:
        N: size of batch, F: frames, N_SENSORS: number of IMU Sensors
        :param n_frames: Number of frames for trial
        :param imu_acc: A Tensor of shape (N, F, N_SENSORS * 3)
        :param imu_gyro: A Tensor of shape (N, F, N_SENSORS * 3)
        :param imu_ori: A Tensor of shape (N, F, N_SENSORS * 4)
        :param pose: A Tensor of shape (N, F, N_JOINTS * 3)
        :param joint_rotations: A Tensor of shape (N, F, N_JOINTS * 3 * 3)
        :param bone_offsets: A Tensor of shape (N, N_JOINTS * 3)
        """
        self.seq_lengths = seq_lengths  # windowsize
        self.imu_acc = imu_acc  # (N, F, N_Sensors * 3)
        self.imu_gyro = imu_gyro  # (N, F, N_Sensors * 3)
        self.imu_ori = imu_ori  # (N, F, N_Sensors * 3 * 3)

        self.vicon_jc = vicon_jc  # (N, F, N_MARKERS* 3)
        self.vicon_ori = vicon_ori  # (N, F, S_Segments * 3 * 3)
        self.vicon_contact = vicon_contact  # (N, F, 4)
        self.velocity = velocity

        self.root = root  # (N, F, 3)

        self.VERSION = VERSION

        if self.VERSION == 'FK':
            self.pose = pose
            self.joint_rotations = joint_rotations
            self.bone_offsets = bone_offsets
            self.relative_joints = relative_joints
            self.relative_phase = relative_phase

    @property
    def n_markers(self):
        """(Return number of markers.) Redundant at the time!!"""
        return self.imu_ori.shape[-1] // 3

    @property
    def batch_size(self):
        """Mini-batch size"""
        if self.VERSION == 'JC':
            return self.vicon_jc.shape[0]
        else:
            return self.pose.shape[0]

    @property
    def seq_length(self):
        """ The sequence length of this batch (i.e. including potential padding)."""
        if self.VERSION == 'JC':
            return self.vicon_jc.shape[1]
        else:
            return self.pose.shape[1]

    @property
    def get_joint_rotations(self):
        """ Return VICON segment orientation. """
        if self.VERSION == 'JC':
            return self.vicon_ori
        else:
            return self.joint_rotations

    @property
    def get_joint_positions(self):
        """ Return VICON joint center. """
        if self.VERSION == 'JC':
            return self.vicon_jc
        else:
            return self.pose

    @property
    def poses_root(self):
        """ Return the root position. """
        if self.VERSION == 'JC':
            return self.root[:, 0, :]
        else:
            return self.root

    @staticmethod
    def from_sample_list(samples):
        """Collect a set of `RealSample`s into a batch."""

        # Retrieve the VERSION from the first sample
        VERSION = samples[0].VERSION

        # Initialize lists for data collection
        seq_lengths, imu_acc, imu_gyro, imu_ori, vicon_jc, vicon_ori, vicon_contact, velocity = [], [], [], [], [], [], [], []
        pose, joint_rotations, relative_joints, relative_phases, bone_offsets, root = ([] for _ in range(6))

        # Iterate through the samples
        for sample in samples:
            if sample is not None:
                # Sequence length
                seq_lengths.append(sample.n_frames)

                # IMU data
                imu_acc.append(sample.imu_acc)
                imu_gyro.append(sample.imu_gyro)
                imu_ori.append(sample.imu_ori)

                # Vicon data
                vicon_ori.append(sample.vicon_ori)
                vicon_contact.append(sample.vicon_contact)
                velocity.append(sample.velocity)

                # Root data
                if sample.root is not None:
                    root.append(sample.root)

                # Depending on VERSION, append the appropriate data
                if VERSION == 'JC':
                    vicon_jc.append(sample.vicon_jc)
                elif VERSION == 'FK':
                    pose.append(sample.pose)
                    joint_rotations.append(sample.joint_rotations)
                    relative_joints.append(sample.relative_joints)
                    relative_phases.append(sample.relative_phases)
                    bone_offsets.append(sample.bone_offsets)

        # Construct the batch data dictionary
        batch_data = {
            'seq_lengths': torch.from_numpy(np.array(seq_lengths)),
            'imu_acc': pad_sequence(imu_acc, batch_first=True),
            'imu_gyro': pad_sequence(imu_gyro, batch_first=True),
            'imu_ori': pad_sequence(imu_ori, batch_first=True),
            'vicon_ori': pad_sequence(vicon_ori, batch_first=True),
            'vicon_contact': pad_sequence(vicon_contact, batch_first=True),
            'velocity': pad_sequence(velocity, batch_first=True),
            'VERSION': VERSION
        }

        # Add root if available
        if root:
            batch_data['root'] = pad_sequence(root, batch_first=True)

        # Handle additional data based on the VERSION
        if VERSION == 'JC':
            additional_data = {'vicon_jc': pad_sequence(vicon_jc, batch_first=True)}
            batch_data.update(additional_data)
            return RealBatch(**batch_data)

        elif VERSION == 'FK':
            additional_data = {
                'pose': pad_sequence(pose, batch_first=True),
                'joint_rotations': pad_sequence(joint_rotations, batch_first=True),
                'relative_joints': pad_sequence(relative_joints, batch_first=True),
                'relative_phase': pad_sequence(relative_phases, batch_first=True),
                'bone_offsets': pad_sequence(bone_offsets, batch_first=True)
            }
            batch_data.update(additional_data)
            return RealBatch(**batch_data)

    def get_inputs(self, sf=None, ef=None, **kwargs):
        """
        Constructs a dictionary of inputs based on the version of the dataset.

        Parameters:
            sf (int): Optional. Start frame for slicing data.
            ef (int): Optional. End frame for slicing data.
            **kwargs: Additional keyword arguments that might be used for future extensions.

        Returns:
            dict: A dictionary containing relevant data fields.
        """

        # Common data fields
        output = {
            'imu_acc': self.imu_acc[sf:ef] if sf is not None and ef is not None else self.imu_acc,
            'imu_gyro': self.imu_gyro[sf:ef] if sf is not None and ef is not None else self.imu_gyro,
            'imu_ori': self.imu_ori[sf:ef] if sf is not None and ef is not None else self.imu_ori,
            'vicon_jc': self.vicon_jc[sf:ef] if sf is not None and ef is not None else self.vicon_jc,
            'vicon_ori': self.vicon_ori[sf:ef] if sf is not None and ef is not None else self.vicon_ori,
            'vicon_contact': self.vicon_contact[sf:ef] if sf is not None and ef is not None else self.vicon_contact,
            'velocity': self.velocity[
                        sf:ef] if sf is not None and ef is not None and self.velocity is not None else self.velocity,
            'root':self.root[sf:ef] if sf is not None and ef is not None and self.root is not None else self.root
        }

        # Version-specific data fields
        if self.VERSION == 'FK':
            output.update({
                'pose': self.pose[sf:ef] if sf is not None and ef is not None else self.pose,
                'joint_rotations': self.joint_rotations[sf:ef] if sf is not None and ef is not None else self.joint_rotations,
                'relative_joints': self.relative_joints[sf:ef] if sf is not None and ef is not None else self.relative_joints,
                'relative_phase':self.relative_phase[sf:ef] if sf is not None and ef is not None else self.relative_phase,
                'bone_offsets': self.bone_offsets[sf:ef] if sf is not None and ef is not None else self.bone_offsets
            })

        return output

    def to_gpu(self):
        """Move data to GPU (if configured)."""
        self.seq_lengths = self.seq_lengths.to(dtype=torch.int, device=C.DEVICE)
        self.imu_acc = self.imu_acc.to(dtype=C.DTYPE, device=C.DEVICE)
        self.imu_gyro = self.imu_gyro.to(dtype=C.DTYPE, device=C.DEVICE)
        self.imu_ori = self.imu_ori.to(dtype=C.DTYPE, device=C.DEVICE)
        self.vicon_contact = self.vicon_contact.to(dtype=C.DTYPE, device=C.DEVICE)
        self.vicon_ori = self.vicon_ori.to(dtype=C.DTYPE, device=C.DEVICE)
        self.velocity = self.velocity.to(dtype=C.DTYPE,
                                         device=C.DEVICE) if self.velocity is not None else self.velocity
        self.root = self.root.to(dtype=C.DTYPE,
                                 device=C.DEVICE) if self.root is not None else self.root

        if self.VERSION == 'JC':
            self.vicon_jc = self.vicon_jc.to(dtype=C.DTYPE, device=C.DEVICE)

        elif self.VERSION == 'FK':
            self.pose = self.pose.to(dtype=C.DTYPE, device=C.DEVICE)
            self.joint_rotations = self.joint_rotations.to(dtype=C.DTYPE, device=C.DEVICE)
            self.relative_joints = self.relative_joints.to(dtype=C.DTYPE, device=C.DEVICE) if self.relative_joints is not None else self.relative_joints
            self.relative_phase = self.relative_phase.to(dtype=C.DTYPE, device=C.DEVICE) if self.relative_joints is not None else self.relative_joints
            self.bone_offsets = self.bone_offsets.to(dtype=C.DTYPE, device=C.DEVICE)

        return self

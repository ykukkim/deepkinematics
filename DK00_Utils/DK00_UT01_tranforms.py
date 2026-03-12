import torch
import numpy as np

from random import choice
from scipy.spatial.transform import Rotation as R
from DK00_Utils.DK00_UT00_config import CONSTANTS as C

"Common Classes/Functions"

def get_end_to_end_preprocess_fn(config):
    """Factory function to return the preprocessing function depending on the given configuration."""
    if config.VERSION == 'JC':
        root_extraction = RootPreparation()
    else:
        normalize_joint_pos = NormalizeJointPositions()
        root_diff = RootDifferentiation(config)

    def _preprocess_fn(sample, mode='all', **kwargs):

        if config.VERSION == 'JC':
            if getattr(config, 'predict_root', False):
                root_delta = root_extraction(sample)
                sample.root_delta = root_delta
            else:
                sample.root_delta = None
        elif config.VERSION == 'FK':
            sample = normalize_joint_pos(sample)
            if getattr(config, 'predict_root', False):
                root_delta = root_diff(sample)
                sample.root_delta = root_delta
            # if mode == 'all':
            #     sample = sample_markers(DK03_FK_00_functions(normalize_root(sample)))
            #     return noise_fn(sample)
            # elif mode == 'normalize_only':
            #     return normalize_root(sample)
            # elif mode == 'after_normalize':
            #     return noise_fn(sample_markers(DK03_FK_00_functions(sample)))
            # else:
            #     raise ValueError("Mode '{}' unknown.".format(mode))
        return sample

    return _preprocess_fn

class CorrectIMUOrientation(object):
    """
    Rotate the IMU orientation around the z-axis by 180 degrees to
    match the walking direction.
    Adjust the direction of movement for the left foot to be 'natural'
    """

    def __init__(self, left_foot_corr=[]):
        self.left_foot_corr = left_foot_corr

    def __call__(self, sample):
        sample.correct_left_ankle_orientation()
        return sample

class ExtractInitialIMUOffset(object):
    """
    Calculate offset of IMU sensors to bvh segments.
    """

    def __init__(self, config):
        # Number of frames to calculate initial offset from
        self.n_offset = config.n_offset
        self.VERSION = config.VERSION
        self.predict_orientation = config.predict_orientation

        if self.VERSION == 'FK':
            self.global_rot = config.global_joint_rot

            # Bvh segments where sensors are mounted on
            self.sensor_pos_idxs = [28, 23, 17, 11, 8, 0]
            self.fk = ForwardKinematics(C.FK_PARENTS_Full)

    def __call__(self, sample):

        f = self.n_offset
        imu_rots = np.empty((f, 6, 3, 3))
        offsets = []

        if self.predict_orientation:
            sample.rotate_vicon_orientation()

        if self.VERSION == 'JC':

            for i in range(6):

                # Transform IMU Orientation from quat to rotation matrix
                imu_rots[:, i, ...] = R.from_quat(sample.imu_ori[:f, i, [1, 2, 3, 0]]).as_matrix()

                # Calculate average rotation R so that imu_rots * R ~ joint rotations
                offsets.append(self.average_rotation(imu_rots[:f, i, ...], sample.vicon_ori[:f, i, ...]))

        else:
            idx = self.sensor_pos_idxs

            bone_offsets = sample.bone_offsets
            j_rots = sample.joint_rotations[:f, ...]

            _, rots = self.fk.fk(bone_offsets, j_rots)
            rots = rots[:, idx, ...]

            for i in range(6):

                # Transform IMU Orientation from quat to rotation matrix
                imu_rots[:, i, ...] = R.from_quat(sample.imu_ori[:f, i, [1, 2, 3, 0]]).as_matrix()

                # Calculate average rotation R so that imu_rots * R ~ joint rotations
                if self.global_rot:
                    offsets.append(self.average_rotation(imu_rots[:, i, ...], rots[:, i, ...]))
                else:
                    offsets.append(self.average_rotation(imu_rots[:, i, ...], j_rots[:, i, ...]))

        sample.ori_offsets = np.array(offsets)
        return sample

    @staticmethod
    def average_rotation(rot1, rot2):
        """
        Compute the rotation needed to align `rot1` and `rot2` as an average along the first dimension.
        :param rot1: A tensor of shape (N, 3, 3).
        :param rot2: A tensor of shape (N, 3, 3).
        :return: The average rotation R such that rot1 * R ~ rot2. The shape of R is (3, 3).
        """
        rot1, rot2 = torch.from_numpy(rot1), torch.from_numpy(rot2)
        n = rot1.shape[0]
        m = torch.matmul(rot1.transpose(1, 2), rot2).sum(dim=0) / n

        u, _, v = torch.svd(m)

        # Ensure det(R) == 1 by fixing the sign of the last singular value.
        z = torch.eye(3, device=rot1.device, dtype=rot1.dtype)
        z[-1, -1] *= torch.sign(torch.det(torch.matmul(u, v.transpose(0, 1))))

        # Construct R.
        r = torch.matmul(u, torch.matmul(z, v.transpose(0, 1)))
        r = np.array(r)
        return r

class ExtractWindow(object):
    def __init__(self, config, window_size, rng=None, mode='random_split', train=False):
        assert mode in ['random', 'random_split', 'random_unseen', 'beginning', 'middle', 'unseen_motion']
        if mode == 'random' or mode == 'random_split':
            assert rng is not None, "Random number generator must be provided for mode '{}'.".format(mode)
        self.window_size = window_size
        self.rng = rng
        self.mode = mode
        self.train = train
        self.padding_value = 0.0
        self.VERSION = config.VERSION

    def __call__(self, sample):
        """ Data is split into 8 equal parts, each of the 6 middle parts can be assigned to the test set (read out of csv file),
        the first and last part contain the jumping sequence and therefore are avoided. Training is done by choosing a
        starting frame from the two intervals that represent the training data. The test interval gets skipped in training."""
        idx = sample.idx_split  # Get index of the Test split sequence, index is in range of 1 to 6
        if sample.n_frames > self.window_size:

            num, div = sample.n_frames, 8  # num: number of frames, 8 fold split
            # create list with the start frames of the 8 sections
            sf_split = [(num // div + (1 if x < num % div else 0)) * x for x in range(div)]

            if self.mode == 'beginning':
                sf, ef = 0, self.window_size

            elif self.mode == 'middle':
                mid = sample.n_frames // 2
                sf = mid - self.window_size // 2
                ef = sf + self.window_size

            elif self.mode == 'random':
                """ Remove first 10 seconds and last 10 seconds for training"""
                sf = self.rng.randint(500, sample.n_frames - self.window_size + 1 - 500)
                ef = sf + self.window_size

            elif self.mode == 'unseen_motion':
                """ Extract entire unseen motion sequence"""
                sf = sf_split[idx]
                ef = sf_split[idx + 1]

            elif self.mode in ['random_split', 'random_unseen']:

                if self.train:
                    # sf = self.rng.randint(*choice([(0, sf_split[idx] - self.window_size),
                    #                                (sf_split[idx + 1], sample.n_frames - self.window_size + 1)]))
                    intervals = [(0, sf_split[idx] - self.window_size),
                                 (sf_split[idx + 1], sample.n_frames - self.window_size + 1)]
                    # Add more random intervals
                    additional_intervals = [
                        (self.rng.randint(0, sf_split[idx] - self.window_size),
                         self.rng.randint(0, sf_split[idx] - self.window_size)),
                        (self.rng.randint(sf_split[idx + 1], sample.n_frames - self.window_size + 1),
                         self.rng.randint(sf_split[idx + 1], sample.n_frames - self.window_size + 1))
                    ]
                    intervals.extend(additional_intervals)

                    # Filter out invalid intervals
                    intervals = [(start, end) for start, end in intervals if start < end]
                    chosen_interval = intervals[self.rng.randint(0, len(intervals) - 1)]
                    sf = self.rng.randint(*chosen_interval)
                else:
                    sf = self.rng.randint(sf_split[idx], sf_split[idx + 1] - self.window_size + 1)

                ef = sf + self.window_size

            else:
                raise ValueError("Mode '{}' for window extraction unknown.".format(self.mode))

            # Check if the interval length exceeds the window size
            if ef - sf > self.window_size:
                raise ValueError("The interval length exceeds the window size.")

            return sample.extract_window(sf, ef, self.window_size)
        else:
            return None

class TransformOrientation(object):
    """
    Transform IMU Orientation representation.
    Possible options: quaternions, rotation vector, rotatation matrix.
    If remove_ori_offset is true, remove the initial offset the imu
    orientations have in respect to the segment orientations.
    """

    def __init__(self, config):
        self.use_quats = config.use_quats
        # self.use_rot_vec = config.use_rot_vec
        self.ori_offset = config.ori_offset
        self.Version = config.VERSION

    def __call__(self, sample):
        if self.use_quats:
            return sample
        # elif self.use_rot_vec:
        #     sample.transform_quat_to_rot_vector()
        #     return sample
        else:
            sample.transform_quat_to_rot_matrix()
            if self.ori_offset:
                sample.align_imu_orientation()
            return sample

class ToTensor(object):
    """Convert sample to torch tensors."""

    def __call__(self, sample):
        sample.to_tensor()
        return sample


"JC"

class RootPreparation(object):
    """ Extract root movement for later prediction. """

    def __call__(self, sample):
        # Calculate Root: Midpoint between hip joints
        root = sample.root.clone()  # (N, F, 3)
        # Medio-lateral
        root[:, :, 0] = root[:, :, 0] - torch.mean(root[:, :, 0], dim=1, keepdim=True)
        # Anterior-posterior
        root[:, :, 1] = torch.zeros_like(root[:, :, 1])
        # Vertical
        root[:, :, 2] = root[:, :, 2] - torch.mean(root[:, :, 2], dim=1, keepdim=True)

        return root

class NormalizeHeight(object):
    """ Normalize Height of subjects by scaling left femur to 0.45m. """

    def __init__(self, config):
        self.normalize_height = getattr(config, 'normalize_height', True)

        self.left_hip = 7
        self.left_knee = 9

    def __call__(self, sample):
        if self.normalize_height:
            sample.normalize_left_thigh(self.left_hip, self.left_knee)
        return sample

class NormalizeViconJointCenter(object):
    """ Calculate midpoint between hip joints and subtract result from all
    joint centers framewise. """

    def __init__(self, config):
        self.remove_root = getattr(config, 'normalize_joint_pos', False)

        self.left_hip = 7
        self.right_hip = 8

    def __call__(self, sample):
        if self.remove_root:
            sample.normalize_vicon_jc(self.left_hip, self.right_hip)
        return sample

class NormalizeRootIMU(object):
    """ Use trunk IMU sensor to normalize all other sensors. """

    def __call__(self, sample):
        sample.normalize_root()
        return sample

"FK"

class NormalizeJointPositions(object):
    """
    Normalise joint positions by translating the root joint (default index 0) to the origin.
    Stores the original root joint position separately for possible inverse transformation.
    """

    def __init__(self, root_index=0):
        self.root_index = root_index

    def __call__(self, sample):
        marker_pos = sample.pose  # Expected shape: (n, f, j*3)
        n, f = marker_pos.shape[0], marker_pos.shape[1]

        # Reshape to (n, f, j, 3)
        marker_pos = marker_pos.reshape(n, f, -1, 3)

        # Extract root joint position
        root_pos = marker_pos[:, :, [self.root_index], :]

        # Subtract root position to normalise all joints
        pose = marker_pos - root_pos

        # Flatten back and assign
        sample.pose = pose.reshape(n, f, -1)
        sample.root = root_pos

        return sample

class RootDifferentiation(object):
    """
    Differentiate the change in root position.
    """

    def __init__(self, config):
        velocity = 3.5  # [km/h]
        self.vel = velocity / 3.6  # [m/s]
        sampling_freq = getattr(config, 'm_sampling_freq', 50)
        self.delta_t = 1 / sampling_freq  # [s]

        self.add_velocity = getattr(config, 'add_gait_vel', False)

    def __call__(self, sample):
        root = sample.root
        n, f = root.shape[0], root.shape[1]

        root_delta = torch.diff(root, dim=1).to(C.DEVICE)

        if self.add_velocity:
            # Minus, since subjects are walking in negative y-direction
            root_delta[:, :, :, 1] -= self.delta_t * self.vel

        root_delta = torch.cat((torch.zeros(n, 1, 1, 3).to(C.DEVICE), root_delta), dim=1)

        return root_delta

class ForwardKinematics(object):
    """
    FK Engine.
    """

    def __init__(self, parents, left_mult=False, major_joints=None, norm_idx=None, no_root=True):
        # self.offsets = offsets
        # if norm_idx is not None:
        #     self.offsets = self.offsets / np.linalg.norm(self.offsets[norm_idx])
        self.parents = parents
        self.n_joints = len(parents)
        self.major_joints = major_joints
        self.left_mult = left_mult
        self.no_root = no_root
        # assert self.offsets.shape[0] == self.n_joints

    def fk(self, offsets, joint_angles, root=None):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            joint_angles: np array of shape (N, n_joints*3*3)

        Returns:
            The 3D joint positions as a an array of shape (N, n_joints, 3)
        """
        # assert joint_angles.shape[-1] == self.n_joints * 9
        assert offsets.shape[0] == self.n_joints
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3, 3])
        n_frames = angles.shape[0]
        positions = np.zeros([n_frames, self.n_joints, 3])
        rotations = np.zeros([n_frames, self.n_joints, 3, 3])  # intermediate storage of global rotation matrices
        if self.left_mult:
            offsets = offsets[np.newaxis, np.newaxis, ...]  # (1, 1, n_joints, 3)
        else:
            offsets = offsets[np.newaxis, ..., np.newaxis]  # (1, n_joints, 3, 1)

        for j in range(self.n_joints):
            if self.parents[j] == -1:
                # this is the root, we don't consider any root translation
                if root is not None:
                    positions[:, j] = root
                else:
                    positions[:, j] = 0.0
                rotations[:, j] = angles[:, j]
            else:
                # this is a regular joint
                if self.left_mult:
                    positions[:, j] = np.squeeze(np.matmul(offsets[:, :, j], rotations[:, self.parents[j]])) + \
                                      positions[:, self.parents[j]]
                    rotations[:, j] = np.matmul(angles[:, j], rotations[:, self.parents[j]])
                else:
                    positions[:, j] = np.squeeze(np.matmul(rotations[:, self.parents[j]], offsets[:, j])) + \
                                      positions[:, self.parents[j]]
                    rotations[:, j] = np.matmul(rotations[:, self.parents[j]], angles[:, j])

        return positions, rotations

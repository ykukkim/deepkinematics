import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

class IMUVisualization():

    # def __init__(self, ):
    #     """ """
    #     self.subject_nr = subject_nr
    #     self.trial = trial
    #     self.dir = dir

    #     self._load_imu_orientation()

    def load_imu_orientation(self, folder=''):
        """ Load IMU orientation and transfrom from quaternions to rotation matrices. """
        sf = self.sf
        nr = self.subject_nr
        path = self.dir + f'/SonE_{nr}/{folder}{self.trial}_{nr}_imu.npz'

        imu_quat = np.load(path)['quat'][::sf, ...]
        imu_quat = imu_quat[:, :, [1, 2, 3, 0]]

        imu_ori = []
        for i in range(imu_quat.shape[1]):
            imu_ori.append(R.from_quat(imu_quat[:, i, :]).as_matrix())

        imu_ori = np.swapaxes(np.array(imu_ori), 0, 1)

        return imu_ori

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

    def calculate_offset_and_compute_alignment(self, imu_ori, segment_ori, frms=10):
        """ Compute the rotation needed to align `imu_ori` to `segment_ori` and align the orientations. """
        offset = np.zeros((6, 3, 3))
        # Compute rotational offset for all six IMU sensor
        for i in range(imu_ori.shape[1]):
            offset[i, ...] = self.average_rotation(imu_ori[:frms, i, :], segment_ori[:frms, i, :])
        # Align IMU_ori to Vicon_ori for the entire sequence
        imu_ori = np.matmul(imu_ori, offset)
        return imu_ori


class ViconVisualization(IMUVisualization):
    """ Import joint center and segment orientation from.
        Remove root movement if desired.
        Create array of line endpoints for visuazlization in aitviewer.
    """

    def __init__(self, subject_nr, trial, dir, freq=200):
        """ Initializer:
        Args:
        :param subject_nr: Number of the subject.
        :param trial: Name of the trial.
        :param dir: Directory where IMU and Vicon files are stored.
        :param freq: Sampling Frequency
        """
        self.subject_nr = subject_nr
        self.trial = trial

        self.dir = dir

        self.sf = int(200 / freq)  # Donwsampling Factor

        self._load_vicon_data()

        self._lines = None
        self._imu_position = None

    @property
    def lines(self):
        if self._lines == None:
            self._lines = self._get_coords_for_lines()
        return self._lines

    @property
    def imu_pos(self):
        if self._imu_position == None:
            self._imu_position = self._get_imu_position()
        return self._imu_position

    @property
    def imu_ori_aligned(self):
        imu_ori = self.load_imu_orientation()
        imu_ori = self.calculate_offset_and_compute_alignment(imu_ori, self.segment_ori)
        return imu_ori

    @property
    def imu_ori(self):
        return self.load_imu_orientation()

    def _load_vicon_data(self):
        """ Load joint center markers and segment orientation. """
        sf = self.sf
        nr = self.subject_nr
        path = self.dir + f'/SonE_{nr}/{self.trial}_{nr}_vicon.npz'

        data = np.load(path)
        self.joint_center = data['jc'][::sf, ...]
        self.segment_ori = data['ori'][::sf, ...]

    def remove_root_movement(self):
        """ Subtract the root (midpoint between hip joints) from all joints. """
        root = np.mean(np.array([self.joint_center[:, [7], :], self.joint_center[:, [8], :]]), axis=0)
        self.root = root
        self.joint_center = self.joint_center - root
        self._imu_position = None
        self._lines = None
    def get_joint_center(self):
        return self.joint_center

    def _get_imu_position(self):
        """ Create array with IMU positions for animation in AITViewer. """
        jc = self.joint_center
        root = np.mean(np.array([jc[:, [7], :], jc[:, [8], :]]), axis=0)
        imu_pos = np.array(jc[:, [11, 12, 5, 6, 0], :])
        imu_pos = np.concatenate((imu_pos, root), axis=1)
        return imu_pos

    def rotate_to_theoretical_imu_orientation(self):
        """ Rotate segment orientation to match the theoretical IMU orientation. """
        # Feet
        self.segment_ori[:, 0:2, ...] = np.matmul(self.segment_ori[:, 0:2, ...],
                                                  R.from_euler('y', -180, degrees=True).as_matrix())
        self.segment_ori[:, 0:2, ...] = np.matmul(self.segment_ori[:, 0:2, ...],
                                                  R.from_euler('z', 180, degrees=True).as_matrix())
        # Left & Right Arm
        self.segment_ori[:, 2:4, ...] = np.matmul(self.segment_ori[:, 2:4, ...],
                                                  R.from_euler('z', -90, degrees=True).as_matrix())
        # Left Arm
        self.segment_ori[:, 2:3, ...] = np.matmul(self.segment_ori[:, 2:3, ...],
                                                  R.from_euler('x', -180, degrees=True).as_matrix())
        # Head
        self.segment_ori[:, 4, ...] = np.matmul(self.segment_ori[:, 4, ...],
                                                R.from_euler('y', 90, degrees=True).as_matrix())
        # Trunk
        self.segment_ori[:, 5, ...] = np.matmul(self.segment_ori[:, 5, ...],
                                                R.from_euler('x', 180, degrees=True).as_matrix())
        self.segment_ori[:, 5, ...] = np.matmul(self.segment_ori[:, 5, ...],
                                                R.from_euler('y', 90, degrees=True).as_matrix())

    def _get_coords_for_lines(self):
        """
        Create endpoints for Lines to visualize joint center as
        a stickfigure in aitviewer
        """
        joint_center = self.joint_center
        # Prepare neck and hip keypoints for stickfigure visualization
        neck = np.mean(np.array([joint_center[:, 1, :], joint_center[:, 2, :]]), axis=0)
        root = np.mean(np.array([joint_center[:, 7, :], joint_center[:, 8, :]]), axis=0)

        stick_figure = np.stack((
            joint_center[:, 0, :], neck[:, :],  # head
            joint_center[:, 1, :], joint_center[:, 2, :],  # shoulder
            joint_center[:, 1, :], joint_center[:, 3, :],  # left upper arm
            joint_center[:, 3, :], joint_center[:, 5, :],  # left lower arm
            joint_center[:, 2, :], joint_center[:, 4, :],  # right upper arm
            joint_center[:, 4, :], joint_center[:, 6, :],  # right lower arm
            neck, root,  # core
            joint_center[:, 7, :], joint_center[:, 8, :],  # hips
            joint_center[:, 7, :], joint_center[:, 9, :],  # left upper leg
            joint_center[:, 9, :], joint_center[:, 11, :],  # left lower leg
            joint_center[:, 11, :], joint_center[:, 13, :],  # left foot
            joint_center[:, 8, :], joint_center[:, 10, :],  # right upper leg
            joint_center[:, 10, :], joint_center[:, 12, :],  # right lower leg
            joint_center[:, 12, :], joint_center[:, 14, :]  # right foot
        ), axis=1)

        return stick_figure


class BvhVisualization(IMUVisualization):
    """
    Load bvh style movement from numpy arrays and visualize it.
    """

    def __init__(self, subject_nr, trial, dir, freq=100):
        """
        Initializer:
        :param path: Path to the subjects forward kinematics folder.
        :param subject_nr: Number of subject.
        :param trial: Name of trial.
        :param sf: Sampling Frequency:
        """
        self.subject_nr = subject_nr
        self.trial = trial

        self.dir = dir
        self.path = dir + f'/SonE_{subject_nr}/fk/'

        self.sf = int(200 / freq)  # Downsampling Factor

        self._load_skeleton_info()
        self.n_joints = len(self.parents)

        self.get_connections_for_skeletons()

        self._positions = None
        self._local_positions = None

    def _load_skeleton_info(self):
        """
        Load skeletal information of participant.

        Args:
        :param offsets: Bone offsets of participant. (n_joints, 3)
        :param parents: List with parent information for joints. (n_joints)
        """
        path = self.path + f'/SonE_{self.subject_nr}_skeleton.npz'
        data = np.load(path)

        self.offsets = data['offset']
        self.parents = data['parent']

    def get_connections_for_skeletons(self):
        """ Create list with joint conections to visualize it with Skeleton class in aitviewer. """
        connections = []
        for i in range(1, len(self.parents)):
            connections.append((i, self.parents[i]))

        self.connections = connections

    @property
    def positions(self):
        """ Global joint positions (with root movement). """
        if self._positions is None:
            self._positions = self._load_joint_positions()
        return self._positions

    @property
    def local_positions(self):
        """ Local joint positions (relative to fixed root). """
        if self._local_positions is None:
            self._local_positions = self._calculate_local_positions()
        return self._local_positions

    @property
    def imu_pos(self):
        return self._get_imu_position()

    @property
    def imu_ori(self):
        return self.load_imu_orientation('/fk/')

    def _get_imu_position(self):
        """ Create array with IMU positions for animation in AITViewer. """
        if self._positions is not None:
            pos = self._positions
        elif self._local_positions is not None:
            pos = self._local_positions
        else:
            raise ("No joint positions found.")

        imu_pos = np.concatenate((pos[:, [29], :], pos[:, [24], :], pos[:, [18], :],
                                  pos[:, [12], :], pos[:, [7], :], pos[:, [0], :]),
                                 axis=1)
        return imu_pos

    def _load_joint_positions(self):
        """ Load joint positions with original root. """
        path = self.path + f'{self.trial}_{self.subject_nr}_position.npz'
        return np.load(path)['position']

    def _calculate_local_positions(self):
        """ Calculate local joint positions relative to root. """
        joint_angles = self._load_local_rotations()

        return self.fk(joint_angles)

    def _load_local_rotations(self):
        """ Load local roations of joints for forward kinematics. """
        path = self.path + f'{self.trial}_{self.subject_nr}_joint_rotation.npz'
        return np.load(path)['joint_rotations']

    def fk(self, joint_angles, root=None):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            joint_angles: np array of shape (N, n_joints*3*3)

        Returns:
            The 3D joint positions as a an array of shape (N, n_joints, 3)
        """
        # assert joint_angles.shape[-1] == self.n_joints * 9
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3, 3])
        n_frames = angles.shape[0]
        positions = np.zeros([n_frames, self.n_joints, 3])
        rotations = np.zeros([n_frames, self.n_joints, 3, 3])  # intermediate storage of global rotation matrices

        offsets = self.offsets[np.newaxis, ..., np.newaxis]  # (1, n_joints, 3, 1)

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
                positions[:, j] = np.squeeze(np.matmul(rotations[:, self.parents[j]], offsets[:, j])) + \
                                  positions[:, self.parents[j]]
                rotations[:, j] = np.matmul(rotations[:, self.parents[j]], angles[:, j])

        return positions


class ForwardKinematics(object):
    """
    FK Engine.
    """

    def __init__(self, offsets, parents, left_mult=False, major_joints=None, norm_idx=None, no_root=True):
        self.offsets = offsets
        if norm_idx is not None:
            self.offsets = self.offsets / np.linalg.norm(self.offsets[norm_idx])
        self.parents = parents
        self.n_joints = len(parents)
        self.major_joints = major_joints
        self.left_mult = left_mult
        self.no_root = no_root
        assert self.offsets.shape[0] == self.n_joints

    def fk(self, joint_angles, root=None):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            joint_angles: np array of shape (N, n_joints*3*3)

        Returns:
            The 3D joint positions as a an array of shape (N, n_joints, 3)
        """
        # assert joint_angles.shape[-1] == self.n_joints * 9
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3, 3])
        n_frames = angles.shape[0]
        positions = np.zeros([n_frames, self.n_joints, 3])
        rotations = np.zeros([n_frames, self.n_joints, 3, 3])  # intermediate storage of global rotation matrices
        if self.left_mult:
            offsets = self.offsets[np.newaxis, np.newaxis, ...]  # (1, 1, n_joints, 3)
        else:
            offsets = self.offsets[np.newaxis, ..., np.newaxis]  # (1, n_joints, 3, 1)

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

        return positions
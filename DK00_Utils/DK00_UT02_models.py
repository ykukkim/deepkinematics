import roma
import math
import torch
import numpy as np
import torch.nn as nn

from DK00_Utils.DK00_UT00_config import CONSTANTS as C
from DK00_Utils.DK00_UT00_utils import clean_string
from DK00_Utils.DK00_UT02_layers import MLPLayer, RNNLayer, GCN, TransformerEncoder, SPL, get_key_padding_mask
from DK00_Utils.DK00_UT02_loss import weighted_mse,weighted_euler_angle_loss,geodesic_loss,circular_loss,AdaptiveLossWrapper


def create_model(config, *args):
    """Create the model according to the configuration object and return it in a tuple.

    Args:
        config: A configuration object containing model settings and preferences.
        *args: Additional arguments required for model initialization.

    Returns:
        A tuple containing the created model(s). For 'att' type, returns both VelocityRegressionNetwork and AttentionModel.
    """

    # Single model cases wrapped in a tuple
    if config.m_type == 'rnn':
        return (SimpleRNN(config, *args),)
    elif config.m_type == 'vrnrnn':
        return VelocityRegressionNetwork(config), SimpleRNN(config, *args)
    elif config.m_type == 'att':
        return (AttentionModel(config, *args),)
    elif config.m_type == 'vrnatt':
        return VelocityRegressionNetwork(config, *args), AttentionModel(config, *args)
    elif config.m_type == 'diff':
        return (DiffusionTransformerModel(config, *args),)
    elif config.m_type == 'vrndiff':
        return VelocityRegressionNetwork(config, *args), DiffusionTransformerModel(config, *args)
    elif config.m_type == 'rnndct':
        return (RNNDCT(config, *args),)
    elif config.m_type == 'attdct':
        return (DCTattention(config, *args),)
    else:
        raise ValueError(f"Model type '{config.m_type}' unknown.")

class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, embed_dim):
        super(PositionalEncoding, self).__init__()
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_length, embed_dim))

        # Initialize the parameter with small values (optional)
        nn.init.normal_(self.pos_encoding, mean=0, std=0.1)

    def forward(self, x):
        # x is expected to be of shape (batch_size, seq_len, embed_dim)
        return x + self.pos_encoding[:, :x.size(1), :]

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_period=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        freqs = torch.exp(exponent).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb

class BaseModel(nn.Module):
    """
    A base class to handle some tasks common to all models.
    """

    def __init__(self, config):
        super(BaseModel, self).__init__()

        self.config = config
        self.VERSION = config.VERSION
        self.n_frames = config.window_size
        self.bs_train = config.bs_train

        # Input
        self.use_acc_gyro = getattr(config, 'use_acc_gyro', False)
        self.use_orientation = getattr(config, 'use_orientation', False)

        # Output
        self.predict_arms = getattr(config, 'predict_arms', False)
        self.predict_head = getattr(config, 'predict_head', False)
        self.predict_contact = getattr(config, 'predict_contact', False)
        self.predict_velocity = getattr(config, 'predict_velocity', False)
        self.predict_joints = getattr(config, 'predict_joints', False)
        self.predict_phase = getattr(config, 'predict_phase', False)
        self.predict_orientation = getattr(config, 'predict_orientation', False)
        self.predict_root = getattr(config, 'predict_root', False)
        self.predict_spl = getattr(config, 'predict_spl', False)

        self.skeleton_pairs = C.skeleton_pairs

        # Model Configuration
        self.m_type = config.m_type
        if self.predict_head and self.predict_arms:
            self.n_sensors = 6
        elif self.predict_head and not self.predict_arms:
            self.n_sensors = 4
        elif not self.predict_head and not self.predict_arms:
            self.n_sensors = 3

        # Loss
        self.pose_loss_w = getattr(config, 'm_pose_loss', None)
        self.contact_loss_w = getattr(config, 'm_contact_loss', None)
        self.orientation_loss_w = getattr(config, 'm_orientation_loss', None)
        self.root_loss_w = getattr(config, 'm_root_loss', None)

        if self.VERSION == 'JC':

            if self.predict_arms and self.predict_head:
                self.indices_to_fill = list(range(len(C.JC_EVAL_JOINTS_Full)))
                self.eval_joints = C.JC_EVAL_JOINTS_Full
            elif not self.predict_arms and self.predict_head:
                self.indices_to_fill = [C.JC_LABELS_Full.index(joint) for joint in C.JC_LABELS_NoArms]
                self.eval_joints = C.JC_EVAL_JOINTS_NoArms
            elif not self.predict_arms and not self.predict_head:
                self.indices_to_fill = [C.JC_LABELS_Full.index(joint) for joint in C.JC_LABELS_NoArmsHead]
                self.eval_joints = C.JC_EVAL_JOINTS_NoArmsHead

            self.indices_missing = [i for i in range(len(C.JC_EVAL_JOINTS_Full)) if i not in self.indices_to_fill]
        else:
            self.FK_skeleton = C.FK_skeleton_Full
            self.parents = C.FK_PARENTS_Full
            self.n_joints = len(C.FK_EVAL_JOINTS_FUll)
            self.left_mult = False
            self.get_foot_joint_idx()

            if self.predict_arms and self.predict_head:
                self.n_joints_spl = len(C.FK_JOINTS_FUll)
                self.indices_to_fill = list(range(len(C.FK_EVAL_JOINTS_FUll)))
                self.eval_joints = C.FK_EVAL_JOINTS_FUll
            elif not self.predict_arms and self.predict_head:
                self.n_joints_spl = len(C.FK_JOINTS_NoArms)
                self.indices_to_fill = [C.FK_JOINTS_FUll.index(joint) for joint in C.FK_JOINTS_NoArms]
                self.eval_joints = C.FK_EVAL_JOINTS_NoArms
            elif not self.predict_arms and not self.predict_head:
                self.n_joints_spl = len(C.FK_JOINTS_NoArmsHead)
                self.indices_to_fill = [C.FK_JOINTS_FUll.index(joint) for joint in C.FK_JOINTS_NoArmsHead]
                self.eval_joints = C.FK_EVAL_JOINTS_NoArmsHead

            self.indices_missing = [i for i in range(len(C.FK_JOINTS_FUll)) if i not in self.indices_to_fill]

            self.joint_rot_loss_w = getattr(config, 'm_joint_rot_loss', None)
            self.velocity_loss_w = getattr(config, 'm_velocity_loss', None)
            self.joints_loss_w = getattr(config, 'm_joints_loss', None)
            self.phase_loss_w = getattr(config, 'm_phase_loss', None)
            self.foot_loss_w = getattr(config, 'm_foot_loss', None)

        self.set_input_output_size()
        self.create_model()

    def set_input_output_size(self):

        "Input Type"
        # If raw acceleration and gyroscope (3 axial)
        if self.use_acc_gyro:
            input_size = 3 * (3 + 3)
            if not self.predict_arms and self.predict_head:
                input_size += 1 * (3 + 3)
            elif self.predict_arms and self.predict_head:
                input_size += 3 * (3 + 3)

        elif self.use_orientation:
            # Orientation in rotation matrix
            input_size = 3 * 3 * 3  # Feet, Pelvis
            if not self.predict_arms and self.predict_head:
                input_size += 1 * 3 * 3
            elif self.predict_arms and self.predict_head:
                input_size += 3 * 3 * 3

        elif self.use_acc_gyro and self.use_orientation:
            input_size = 3 * 3 * 3 + (3 * (3 + 3))  # Feet, Pelvis
            if not self.predict_arms and self.predict_head:
                input_size += 1 * 3 * 3 + 1 * (3 + 3)
            elif self.predict_arms and self.predict_head:
                input_size += 3 * 3 * 3 + 2 * (3 + 3)

        if self.config.use_quats:
            input_size *= 4 / 9
        if self.use_acc_gyro is not True:
            input_size += 2  # Add normed acc for feet

        "Output Type -> if orientation is interested"
        if self.predict_orientation:
            # Orientation in rotation matrix
            output_size_ori = 3 * 3 * 3  # Feet, Pelvis
            output_eval_ori = [0, 1, 5]

            if not self.predict_arms and self.predict_head:
                output_size_ori += 1 * 3 * 3
                output_eval_ori = [0, 1, 4, 5]
            elif self.predict_arms and self.predict_head:
                output_size_ori += 3 * 3 * 3
                output_eval_ori = [0, 1, 2, 3, 4, 5]

            self.output_size_ori = output_size_ori
            self.output_eval_ori = output_eval_ori

        " Add input and output size to config "
        if self.config.VERSION == 'JC':
            if self.predict_arms and self.predict_head:
                output_size = len(C.JC_EVAL_JOINTS_Full) * 3
            elif not self.predict_arms and self.predict_head:
                output_size = len(C.JC_EVAL_JOINTS_NoArms) * 3
            elif not self.predict_arms and not self.predict_head:
                output_size = len(C.JC_EVAL_JOINTS_NoArmsHead) * 3
        elif self.config.VERSION == 'FK':
            if self.predict_arms and self.predict_head:
                output_size = len(C.FK_EVAL_JOINTS_FUll) * 3 * 3
            elif not self.predict_arms and self.predict_head:
                output_size = len(C.FK_EVAL_JOINTS_NoArms) * 3 * 3
            elif not self.predict_arms and not self.predict_head:
                output_size = len(C.FK_EVAL_JOINTS_NoArmsHead) * 3 * 3

        self.input_size = input_size
        self.output_size = output_size

    def get_foot_joint_idx(self):

        foot_joints = C.FK_FOOT_JOINTS

        if self.predict_arms and self.predict_head:
            joints = C.FK_JOINTS_FUll
        elif not self.predict_arms and self.predict_head:
            joints = C.FK_JOINTS_NoArms
        elif not self.predict_arms and not self.predict_head:
            joints = C.FK_JOINTS_NoArmsHead

        self.foot_index = [joints.index(index) for index in foot_joints]

        # noinspection PyAttributeOutsideInit

    def create_model(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def model_name(self):
        base_name = ''
        base_name += '-pol{}'.format(self.pose_loss_w)  # pose loss

        if self.VERSION == 'JC':
            base_name = ''
            base_name += '-rl{}'.format(self.root_loss_w)  # root loss
        elif self.VERSION == 'FK':
            """A summary string of this model."""
            base_name += '-jrotl{}'.format(self.joint_rot_loss_w)
            base_name += '-fl{}'.format(self.foot_loss_w)

        if self.predict_contact:
            base_name += '-cl{}'.format(self.contact_loss_w)  # contact loss
        if self.predict_velocity:
            base_name += '-vel{}'.format(self.velocity_loss_w) # relative phases loss
        if self.predict_joints:
            base_name += '-jphl{}'.format(self.joints_loss_w)  # relative phases loss
        if self.predict_phase:
            base_name += '-phl{}'.format(self.phase_loss_w) # relative phases loss
        if self.predict_orientation:
            base_name += '-oril{}'.format(self.orientation_loss_w)  # orientation
        if self.predict_root:
            base_name += '-rl{}'.format(self.root_loss_w)

        base_name += '-{}'.format(self.config.scheduler)
        base_name += '-{}'.format(self.config.optimizer)
        base_name += '-lr{}'.format(self.config.lr)
        base_name += '-wd{}'.format(self.config.weight_decay)
        base_name += '-ns{}'.format(self.n_sensors)
        base_name += '-bs{}'.format(self.bs_train)

        if self.predict_root:
            base_name += '-root'
        if self.predict_spl:
            base_name += '-spl'

        return base_name

    def forward(self, batch, window_size=None, is_new_sequence=True):
        """The forward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, batch, model_out, writer=None, global_step=None):
        n, f = batch.batch_size, batch.seq_length

        # Loss dictionary
        loss_dict = {}
        total_loss = 0.0

        self.mse = nn.MSELoss()
        self.bce_with_log = nn.BCEWithLogitsLoss()

        """ The backward pass. """
        if self.VERSION == 'JC':
            # predicted
            pose_hat = model_out['pose_hat']
            pose_hat = pose_hat.reshape(n, f, -1, 3)  # n, f, full size fk * 3
            pose_hat = pose_hat[:, :, self.eval_joints, :]

            # ground truth
            pose = batch.vicon_jc.reshape(n, f, -1, 3)  # n, f, full size fk * 3
            pose = pose[:, :, self.eval_joints, :]

            "Losses"
            pose_loss = self.mse(pose_hat, pose)
            total_loss = pose_loss

            loss_vals['pose'] = pose_loss.cpu().item()

            if self.predict_contact:
                contact, contact_hat = batch.vicon_contact, model_out['contact_hat']
                contact_loss = self.bce_with_log(contact_hat, contact)

                total_loss += self.contact_loss_w * contact_loss
                loss_vals['contact'] = contact_loss.cpu().item()

            if self.predict_orientation:
                orientation = batch.vicon_ori.reshape(n, f, -1, 3, 3)
                orientation = orientation[:, :, self.output_eval_ori, :, :]
                orientation_hat = model_out['orientation_hat'].reshape(n, f, -1, 3, 3)
                orientation_loss = self.mse(orientation_hat, orientation)

                total_loss += self.orientation_loss_w * orientation_loss
                loss_vals['orientation'] = orientation_loss.cpu().item()

            if self.predict_root:
                # root = batch.root_delta
                root, root_hat = batch.root, model_out['root_hat']
                root_loss = self.mse(root_hat, root)

                total_loss += self.root_loss_w * root_loss
                loss_vals['root'] = root_loss.cpu().item()

        elif self.VERSION == 'FK':

            # ground truth
            joint_rot = batch.joint_rotations.reshape(n, f, -1, 3, 3)
            pose = batch.pose.reshape(n, f, -1, 3)  # n, f, full size fk * 3
            joint_rot = joint_rot[:, :, self.eval_joints, :]
            pose = pose[:, :, self.eval_joints, :]

            # predicted
            pose_hat, joint_rot_hat = model_out['pose_hat'], model_out['joint_rot_hat']
            pose_hat = pose_hat[:, :, self.eval_joints, :]
            joint_rot_hat = joint_rot_hat[:, :, self.eval_joints, :]

            # Pose loss
            if self.m_type in ('vrnrnn', 'rnn', 'att', 'vrnatt', 'diff', 'vrndiff'):
                pose_loss = weighted_mse(pose_hat, pose, self.foot_loss_w, self.foot_index)
            else:
                pose_loss = self.mse(pose_hat, pose)

            # Joint rotation loss
            joint_rot_loss = self.mse(joint_rot_hat, joint_rot)

            total_loss += self.joint_rot_loss_w * joint_rot_loss + self.pose_loss_w * pose_loss
            loss_dict['joint_rot'] = joint_rot_loss
            loss_dict['pose'] = pose_loss

            # Diffusion noise loss
            if 'vrndiff' in self.m_type or 'diff' in self.m_type:
                noise_loss = self.mse(model_out['predicted_noise'], model_out['true_noise'])
                loss_dict['noise'] = noise_loss

            # Contact loss
            if self.predict_contact:
                contact = batch.vicon_contact
                contact_hat = model_out['contact_hat']
                contact_loss = self.bce_with_log(contact_hat, contact)
                loss_dict['contact'] = contact_loss
                total_loss += self.contact_loss_w * contact_loss

            # Velocity loss
            if self.predict_velocity:
                velocity = batch.velocity
                velocity_hat = model_out['velocity_hat']
                velocity_loss = self.mse(velocity_hat, velocity)
                loss_dict['velocity'] = velocity_loss
                total_loss += self.velocity_loss_w * velocity_loss

            # Relative joint angles (e.g., coordination)
            if self.predict_joints:
                relative_joints = batch.relative_joints
                relative_joints_hat = model_out['relative_joints_hat'].reshape(n, f, -1, 3)
                joint_angle_loss = weighted_euler_angle_loss(relative_joints_hat, relative_joints)
                loss_dict['relative_joints_angle'] = joint_angle_loss
                total_loss += self.joints_loss_w * joint_angle_loss

            # Relative phase loss
            if self.predict_phase:
                relative_phase = batch.relative_phase
                relative_phase_hat = model_out['relative_phase_hat']
                phase_loss = circular_loss(relative_phase_hat, relative_phase)
                loss_dict['relative_joints_phase'] = phase_loss
                total_loss += self.phase_loss_w * phase_loss

            if self.predict_root:
                # TO DO
                # 1. Predicting the root position directly
                # 2. Predicting the root velcoity.

                root, root_hat = batch.root, model_out['root_hat']

                root_delta = batch.root_delta.reshape(n, f, -1)
                root = self.get_root(root_delta)  # calculates root from root velocity

                root_delta_hat = root_hat.reshape(n, f, -1)
                root_pos_hat = self.get_root(root_delta_hat)

                # root_loss = mean_L1_loss(root_hat, root_delta)

                root_delta_loss = self.mse(root_delta_hat, root_delta)
                root_pos_loss = self.mse(root_pos_hat, root)

                root_loss = root_delta_loss + 0.1 * root_pos_loss

                total_loss += self.root_loss_w * root_loss
                loss_vals['root'] = root_loss.cpu().item()

            if self.predict_orientation: # only for JC
                orientation = batch.vicon_ori.reshape(n, f, -1, 3, 3)
                orientation = orientation[:, :, self.output_eval_ori, :, :]
                orientation_hat = model_out['orientation_hat'].reshape(n, f, -1, 3, 3)
                orientation_loss = self.mse(orientation_hat, orientation)

                # total_loss += self.orientation_loss_w * orientation_loss
                loss_vals['orientation'] = orientation_loss

        # Initialise adaptive loss wrapper if not already done
        if not hasattr(self, 'adaptive_loss'):
            self.adaptive_loss = AdaptiveLossWrapper(list(loss_dict.keys()))

        # Compute total loss and per-task weighted losses
        total_loss, weighted_losses = self.adaptive_loss(loss_dict)

        if self.training:
            total_loss.backward()

        # Logging
        loss_vals = {k: v.cpu().item() for k, v in loss_dict.items()}
        loss_vals.update({f'{k}_w': v.cpu().item() for k, v in weighted_losses.items()})

        return total_loss, loss_vals, loss_dict

    def prepare_inputs(self, batch_inputs):
        """Expects offsets in the input and reverts them."""
        n, f = batch_inputs['imu_ori'].shape[0], batch_inputs['imu_ori'].shape[1]
        imu_acc = batch_inputs['imu_acc'].reshape((n, f, 6, -1))
        imu_gyro = batch_inputs['imu_gyro'].reshape((n, f, 6, -1))
        imu_ori = batch_inputs['imu_ori'].reshape((n, f, 6, -1))

        if self.n_sensors == 4:
            imu_acc = imu_acc[:, :, [0, 1, 4, 5], :]
            imu_gyro = imu_gyro[:, :, [0, 1, 4, 5], :]
            imu_ori = imu_ori[:, :, [0, 1, 4, 5], :]

        elif self.n_sensors == 3:
            imu_acc = imu_acc[:, :, [0, 1, 4], :]
            imu_gyro = imu_gyro[:, :, [0, 1, 4], :]
            imu_ori = imu_ori[:, :, [0, 1, 4], :]

        # Norm the feet acceleration
        acc = torch.linalg.norm(imu_acc[:, :, [0, 1], :], dim=-1)

        model_in = []

        if self.use_acc_gyro:
            model_in.append(imu_acc.reshape((n, f, -1)))
            model_in.append(imu_gyro.reshape((n, f, -1)))
        else:
            model_in.append(imu_ori.reshape((n, f, -1)))
            model_in.append(acc.reshape((n, f, -1)))

        return torch.cat(model_in, dim=-1)

    @staticmethod
    def special_procrustes(rot_mat):
        """ Returns the rotation matrix R minimizing the Frobenius norm."""
        n, f = rot_mat.shape[0], rot_mat.shape[1]

        rot_mat = roma.special_procrustes(rot_mat.reshape(-1, 3, 3))
        rot_mat = rot_mat.reshape(n, f, -1, 3, 3)

        return rot_mat

    def do_fk(self, offsets, joint_angles, root=None):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            offsets: torch tensor of shape (n, f, n_joints, 3)
            joint_angles: torch tensor of shape (n, f, n_joints*3*3)

        Returns:
            The 3D joint positions as a an array of shape (n, f, n_joints, 3)
        """
        assert offsets.shape[1] == self.n_joints
        n, f = joint_angles.shape[0], joint_angles.shape[1]
        angles = torch.reshape(joint_angles, (n, f, self.n_joints, 3, 3))
        positions = torch.zeros(n, f, self.n_joints, 3).to(C.DEVICE)
        rotations = torch.zeros(n, f, self.n_joints, 3, 3).to(
            C.DEVICE)  # intermediate storage of global rotation matrices
        if self.left_mult:
            offsets = offsets.unsqueeze(1).unsqueeze(1)  # (1, 1, n_joints, 3)
        else:
            offsets = offsets.unsqueeze(1).unsqueeze(-1)  # (B, 1, n_joints, 3, 1)

        for j in range(self.n_joints):
            if self.parents[j] == -1:
                # this is the root, we don't consider any root translation
                if root is not None:
                    positions[:, :, j] = root
                else:
                    positions[:, :, j] = 0.0
                rotations[:, :, j] = angles[:, :, j]
            else:
                # this is a regular joint
                if self.left_mult:
                    positions[:, :, j] = torch.squeeze(
                        torch.matmul(offsets[:, :, :, j], rotations[:, :, self.parents[j]])) + \
                                         positions[:, :, self.parents[j]]
                    rotations[:, :, j] = torch.matmul(angles[:, :, j], rotations[:, :, self.parents[j]].clone())
                else:

                    positions[:, :, j] = torch.squeeze(
                        torch.matmul(rotations[:, :, self.parents[j]], offsets[:, :, j])) + \
                                         positions[:, :, self.parents[j]]
                    rotations[:, :, j] = torch.matmul(rotations[:, :, self.parents[j]].clone(), angles[:, :, j])

        return positions

    def get_root(self, root_delta):
        deltas = torch.clone(root_delta)
        """ Sum up root differences to real position.
        :param root_delta: Difference in root position. (b, f, 3) """
        for i in range(1, deltas.shape[1]):
            deltas[:, i, :] = deltas[:, i, :] + deltas[:, i - 1, :]

        return deltas

    def window_generator(self, batch, window_size):
        """Subdivide a batch into temporal windows of length `window_size`."""
        if window_size is not None:
            seq_len = batch.seq_length
            n_windows = seq_len // window_size + int(seq_len % window_size > 0)

            for i in range(n_windows):
                sf = i * window_size
                ef = min((i + 1) * window_size, seq_len)
                seq_lengths_new = torch.tensor([ef - sf]).to(dtype=batch.seq_lengths.dtype,
                                                             device=batch.seq_lengths.device)
                batch_inputs = batch.get_inputs(sf=sf, ef=ef)
                batch_inputs['seq_lengths'] = seq_lengths_new
                yield batch_inputs
        else:
            batch_inputs = batch.get_inputs()
            batch_inputs['seq_lengths'] = batch.seq_lengths
            yield batch_inputs

    def update_joint_rot_hat(self, joint_rot_hat, indices_to_fill, indices_missing, batch):
        joint_rot_hat_temp = joint_rot_hat
        del joint_rot_hat
        joint_rot_hat = torch.zeros(
            (joint_rot_hat_temp.shape[0], joint_rot_hat_temp.shape[1], len(C.FK_JOINTS_FUll), 3, 3),
            dtype=joint_rot_hat_temp.dtype, device=C.DEVICE)

        # Fill in the values from joint_rot_hat_temp
        joint_rot_hat[:, :, indices_to_fill, :, :] = joint_rot_hat_temp

        # Reshape batch_joint_rotations to match the shape of joint_rot_hat
        relevant_rotations = batch.joint_rotations.reshape(
            joint_rot_hat.shape[0], joint_rot_hat.shape[1], len(C.FK_JOINTS_FUll), 3, 3)

        # Fill in the missing values from batch.joint_rotations
        joint_rot_hat[:, :, indices_missing, :, :] = relevant_rotations[:, :, indices_missing, :, :]

        return joint_rot_hat

    def update_pose_hat(self, pose_hat, indices_to_fill, indices_missing, batch):
        pose_hat_temp = pose_hat.reshape((pose_hat.shape[0], pose_hat.shape[1], -1, 3))
        del pose_hat
        pose_hat = torch.zeros(
            (pose_hat_temp.shape[0], pose_hat_temp.shape[1], len(C.JC_LABELS_Full), 3),
            dtype=pose_hat_temp.dtype, device=C.DEVICE)

        # Fill in the values from joint_rot_hat_temp
        pose_hat[:, :, indices_to_fill, :] = pose_hat_temp

        # Reshape batch_joint_rotations to match the shape of joint_rot_hat
        relevant_rotations = batch.vicon_jc.reshape(
            pose_hat.shape[0], pose_hat.shape[1], len(C.JC_LABELS_Full), 3)

        # Fill in the missing values from batch.joint_rotations
        pose_hat[:, :, indices_missing, :] = relevant_rotations[:, :, indices_missing, :]

        return pose_hat

# VRN
class VelocityRegressionNetwork(BaseModel):
    def __init__(self, config):
        super(VelocityRegressionNetwork, self).__init__(config)

    def create_model(self):
        self.num_directions = 2 if self.config.m_bidirectional else 1
        input_size_vrn = 3 * (3)
        if self.predict_arms and self.predict_head:
            input_size_vrn += 3 * (3)
        elif not self.predict_arms and self.predict_head:
            input_size_vrn += 1 * (3)
        output_size_vrn = self.output_size // 3 if self.VERSION == 'FK' else self.output_size

        # MLP Encoder Layers
        self.mlp = MLPLayer(input_size=input_size_vrn, embedding_size=self.config.m_embedding_VRNMLP,
                            dropout=self.config.m_dropout, use_batch_norm=self.config.use_batch_norm)

        self.rnn = RNNLayer(input_size=self.config.m_embedding_VRNMLP[-1], output_size=None,
                            num_layers=self.config.m_num_layers_VRN, hidden_units=self.config.m_hidden_units_VRN,
                            bidirectional=self.config.m_bidirectional, dropout=self.config.m_dropout,
                            learn_init_state=self.config.m_learn_init_state, use_batch_norm=self.config.use_batch_norm)

        self.output_layer = nn.Linear(self.config.m_hidden_units_VRN[-1] * self.num_directions, output_size_vrn)

    def forward(self, batch, window_size=None, is_new_sequence=True):

        n, f, channel = batch.imu_acc.shape

        if self.n_sensors == 4:
            inputs_ = batch.imu_acc.reshape(n, f, 6, 3)[:, :, [0, 1, 4, 5], :].reshape(n, f, -1)
        elif self.n_sensors == 3:
            inputs_ = batch.imu_acc.reshape(n, f, 6, 3)[:, :, [0, 1, 4], :].reshape(n, f, -1)
        else:
            inputs_ = batch.imu_acc

        mlp_in = self.mlp(inputs_)  # MLP

        lstm_out = self.rnn(mlp_in, batch.seq_lengths)  # RNN / Bi RNN

        output = self.output_layer(lstm_out)

        return output

    def backward(self, batch, model_out, writer=None, global_step=None):

        velocity = batch.velocity.reshape(batch.batch_size, batch.seq_length, -1, 3)
        velocity_hat = model_out.reshape(batch.batch_size, batch.seq_length, -1, 3)

        noofjoints = C.JC_EVAL_JOINTS_Full if self.VERSION == 'JC' else C.FK_JOINTS_FUll
        velocity_hat_temp = velocity_hat

        del velocity_hat
        velocity_hat = torch.zeros(
            (velocity_hat_temp.shape[0], velocity_hat_temp.shape[1], len(noofjoints), 3),
            dtype=velocity_hat_temp.dtype, device=C.DEVICE)

        # Fill in the values from joint_rot_hat_temp
        velocity_hat[:, :, self.indices_to_fill, :] = velocity_hat_temp.reshape(velocity_hat_temp.shape[0],
                                                                                velocity_hat_temp.shape[1], -1, 3)

        # Fill in the missing values from batch.joint_rotations
        velocity_hat[:, :, self.indices_missing, :] = velocity[:, :, self.indices_missing, :]

        loss = nn.MSELoss()(velocity_hat[:, :, self.indices_to_fill, :], velocity[:, :, self.indices_to_fill, :])
        loss = loss.cpu().item()
        return loss

# RNN
class SimpleRNN(BaseModel):
    """A uni- or bidirectional RNN."""

    def __init__(self, config, *args):
        super(SimpleRNN, self).__init__(config,*args)

    def create_model(self):

        self.num_directions = 2 if self.config.m_bidirectional else 1

        output_size_vrn = self.output_size // 3 if self.VERSION == 'FK' else self.output_size
        input_size_final = self.input_size if self.config.m_type != 'vrnrnn' else self.input_size + output_size_vrn

        "Model"
        self.rnn = RNNLayer(input_size=input_size_final, output_size=None,
                            num_layers=self.config.m_num_layers_RNN, hidden_units=self.config.m_hidden_units_RNN,
                            bidirectional=self.config.m_bidirectional, dropout=self.config.m_dropout,
                            learn_init_state=self.config.m_learn_init_state, use_batch_norm=self.config.use_batch_norm)
        "FC"
        if self.predict_spl:
            self.out_layer = SPL(input_size=self.config.m_hidden_units_RNN[-1] * self.num_directions,
                                 number_joints=self.n_joints_spl,
                                 joint_size=9,
                                 num_layers=1,
                                 hidden_units=self.config.m_hidden_units_SPL[-1],
                                 predict_arms=self.predict_arms,
                                 predict_head=self.predict_head
                                 )
        else:
            self.out_layer = nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions, self.output_size)

        if self.predict_contact:
            self.to_contact_phase = nn.Linear(in_features=self.config.m_hidden_units_RNN[-1] * self.num_directions, out_features=4)

        if self.predict_velocity:
            self.to_velocity = nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions, output_size_vrn)

        if self.predict_joints:
            self.to_joints = nn.Sequential(
                nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions, self.config.m_hidden_units_RNN[-1]),
                nn.ReLU(),
                nn.Linear(self.config.m_hidden_units_RNN[-1], len(C.FK_PARENTS_Full)*3),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )
            # self.to_joints = nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions,len(C.skeleton_pairs)*3)

        if self.predict_phase:
            self.to_relative_phase = nn.Sequential(
                nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions, self.config.m_hidden_units_RNN[-1]),
                nn.ReLU(),
                nn.Linear(self.config.m_hidden_units_RNN[-1], len(C.FK_PARENTS_Full)),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )
            # self.to_relative_phase = nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions,len(C.skeleton_pairs))

        if self.predict_orientation:
            self.orientation = nn.Linear(in_features=self.config.m_hidden_units_RNN[-1] * self.num_directions,
                                         out_features=self.output_size_ori)

        if self.predict_root:
            if self.VERSION == 'JC':
                self.to_root = nn.Linear(self.m_hidden_units_RNN[-1] * self.num_directions, 3)

            elif self.VERSION == 'FK':
                self.to_root = MLPLayer(input_size=self.config.m_hidden_units_RNN[-1] * self.num_directions,
                                   output_size=3,dropout=self.config.m_dropout,
                                   use_batch_norm=self.config.use_batch_norm)

    def model_name(self):
        """A summary string of this model."""
        # Build RNN part (layers × hidden × hidden)
        rnn_hidden = clean_string(str(self.config.m_hidden_units_RNN))
        spl_hidden = clean_string(str(self.config.m_hidden_units_SPL))
        rnn_part = f"{self.config.m_num_layers_RNN}x{rnn_hidden}x{spl_hidden}"

        # If model type includes VRN, add VRNRNN and its spec
        if self.config.m_type == 'vrnrnn':
            vrn_embed = clean_string(str(self.config.m_embedding_VRNMLP))
            vrn_hidden = clean_string(str(self.config.m_hidden_units_VRN))
            vrn_part = f"{self.config.m_num_layers_VRN}x{vrn_hidden}x{vrn_embed}"
            base_name = f"VRN-{vrn_part}-{rnn_part}"
        else:
            base_name = rnn_part

        # Add Bi prefix if bidirectional
        if self.config.m_bidirectional:
            base_name = "BiRNN" + base_name

        base_name += super(SimpleRNN, self).model_name()

        return base_name

    def forward(self, batch, velocityregression, window_size = None, is_new_sequence = True):
        """ The forward pass. """
        if is_new_sequence:
            self.rnn.final_state = None
        self.rnn.init_state = self.rnn.final_state

        # IMU Input
        inputs_temp = self.prepare_inputs(batch.get_inputs())  # (n, f, number of sensors * 3 + 2)
        inputs_ = torch.cat((velocityregression, inputs_temp),
                            dim=-1) if velocityregression is not None else inputs_temp

        lstm_out = self.rnn(inputs_, batch.seq_lengths)  # Feed Data into LSTM
        output = {}  # Initialize an empty dictionary to hold the output.

        if self.VERSION == 'JC':
            # Pose
            pose_hat = self.out_layer(lstm_out)  # (N, F, self.output_size)
            pose_hat = self.update_pose_hat(pose_hat, self.indices_to_fill, self.indices_missing, batch)
            output = {'pose_hat': pose_hat}

        elif self.VERSION == 'FK':
            "Joint Rotations with interested joints "
            joint_rot_hat = self.out_layer(lstm_out)  # (N, F, self.output_size)
            joint_rot_hat = self.special_procrustes(joint_rot_hat)
            joint_rot_hat = self.update_joint_rot_hat(joint_rot_hat, self.indices_to_fill, self.indices_missing, batch)

            # Calculate pose.
            pose_hat = self.do_fk(batch.bone_offsets, joint_rot_hat)
            output = {'joint_rot_hat': joint_rot_hat, 'pose_hat': pose_hat}

        # Contact Phase
        if self.predict_contact:
            output['contact_hat'] = self.to_contact_phase(lstm_out)

        if self.predict_velocity:
            velocity_hat = self.to_velocity(lstm_out)
            output['velocity_hat'] = velocity_hat

        if self.predict_joints:
            relative_joints_hat = self.to_joints(lstm_out)
            output['relative_joints_hat'] = relative_joints_hat

        if self.predict_phase:
            relative_phase_hat = self.to_relative_phase(lstm_out)
            output['relative_phase_hat'] = relative_phase_hat

        if self.predict_orientation:
            orientation_hat = self.orientation(lstm_out)
            output['orientation_hat'] = orientation_hat.reshape(orientation_hat.shape[0], orientation_hat.shape[1], -1,
                                                                3, 3)
        # Root
        if self.predict_root:
            output['root_hat'] = self.to_root(lstm_out)
            # output['root_hat'] = self.get_root(lstm_out) # Velocity

        return output

# Transformer-based attention model
class AttentionModel(BaseModel):

    def __init__(self, config, *args):
        super(AttentionModel, self).__init__(config, *args)

    def create_model(self):

        self.use_gcn = getattr(self.config, 'use_gcn', False)

        output_size_vrn = self.output_size // 3 if self.VERSION == 'FK' else self.output_size
        input_size_final = self.input_size if self.config.m_type != 'vrnatt' else self.input_size + output_size_vrn

        self.embedding_layer = MLPLayer(
            input_size=input_size_final,
            embedding_size=self.config.m_embedding_MLP[-1],
            dropout=self.config.m_dropout,
            use_batch_norm=self.config.use_batch_norm
        )

        self.projection_layer = nn.Linear(self.config.m_embedding_MLP[-1], self.config.m_embedding_attention[0])

        # Positional encoding
        self.positional_encoding_type = getattr(self.config, "m_positional_encoding_type", "learnable")
        if self.positional_encoding_type == "learnable":
            self.positional_encoding = PositionalEncoding(self.config.window_size, self.config.m_embedding_attention[0])
        elif self.positional_encoding_type == "sinusoidal":
            self.register_buffer("positional_encoding_buffer",
                                 self.positional_encoding_sinusoid(
                                     seq_len=self.config.window_size,
                                     d_model=self.config.m_embedding_attention[0],  # also here
                                     device=C.DEVICE
                                 ))
        else:
            raise ValueError("Unsupported positional encoding")

        # self.projection_layer_tot = nn.Linear(self.config.m_embedding_positional, self.config.m_embedding_attention[0])

        # Transformer
        self.TransformerEncoder = TransformerEncoder(
            embed_dim=self.config.m_embedding_attention,
            num_layers=self.config.m_num_layers_attention,
            num_heads=self.config.m_num_heads_attention,
            ff_hidden_dim=self.config.m_num_hidden_units_attention,
            dropout=self.config.m_dropout,
            skip_connection=self.config.m_skip_connections,
            window_size=self.config.m_window_attention
        )

        if self.use_gcn:
            self.gcn = GCN(input_feature=self.config.m_embedding_attention,
                           hidden_feature=self.config.m_num_hidden_units_attention,
                           drop_out=self.config.m_dropout,
                           num_stage=self.config.m_num_stage,
                           node_n=self.config.m_window_attention)

        if self.predict_spl:
            self.out_layer = SPL(
                input_size=self.config.m_embedding_attention[-1],
                number_joints=self.n_joints_spl,
                joint_size=9,
                num_layers=1,
                hidden_units=self.config.m_hidden_units_SPL[-1],
                predict_arms=self.predict_arms,
                predict_head=self.predict_head
            )

        else:
            self.out_layer = nn.Linear(self.config.m_embedding_attention[-1], self.output_size)

        if self.predict_contact:
            self.to_contact_phase = nn.Linear(in_features=self.config.m_embedding_attention[-1], out_features=4)

        if self.predict_velocity:
            self.to_velocity = nn.Linear(self.config.m_embedding_attention[-1], output_size_vrn)

        if self.predict_joints:
            self.to_joints = nn.Sequential(
                nn.Linear(self.config.m_embedding_attention[-1], self.config.m_embedding_attention[-1] // 2),
                nn.ReLU(),
                nn.Linear(self.config.m_embedding_attention[-1] // 2, len(C.FK_PARENTS_Full)*3),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )
            # self.to_joints = nn.Linear(self.config.m_embedding_attention[-1],len(C.skeleton_pairs)*3)

        if self.predict_phase:
            self.to_relative_phase = nn.Sequential(
                nn.Linear(self.config.m_embedding_attention[-1], self.config.m_embedding_attention[-1] // 2),
                nn.ReLU(),
                nn.Linear(self.config.m_embedding_attention[-1] // 2, len(C.FK_PARENTS_Full)),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )
            # self.to_relative_phase = nn.Linear(self.config.m_embedding_attention[-1], len(C.skeleton_pairs))

        if self.predict_orientation:
            self.orientation = nn.Linear(in_features=self.config.m_embedding_attention[-1],
                                         out_features=self.output_size_ori)

        if self.predict_root:
            if self.VERSION == 'JC':
                self.to_root = nn.Linear(self.m_embedding_attention[-1], 3)

            elif self.VERSION == 'FK':
                self.to_root = MLPLayer(input_size=self.config.m_embedding_attention[-1],
                                   output_size=3,dropout=self.config.m_dropout,
                                   use_batch_norm=self.config.use_batch_norm)

    def model_name(self):
        """A summary string of this model."""
        if self.config.m_type == 'vrnatt':
            base_name = "VRN-{}x{}x{}-ATT".format(self.config.m_num_layers_VRN,
                                                    clean_string(str(self.config.m_hidden_units_VRN)),
                                                    clean_string(str(self.config.m_embedding_VRNMLP)))
        else:
            base_name = "ATT"

        base_name += "-{}-{}x{}x{}x{}x{}x{}x{}".format(self.config.m_positional_encoding_type,
                                                 self.config.m_num_layers_attention,
                                                 clean_string(str(self.config.m_embedding_MLP)),
                                                 clean_string(str(self.config.m_embedding_attention)),
                                                 clean_string(str(self.config.m_num_hidden_units_attention)),
                                                 clean_string(str(self.config.m_num_heads_attention)),
                                                 str(self.config.m_window_attention),
                                                 clean_string(str(self.config.m_hidden_units_SPL)))

        if self.use_gcn:
            base_name += "-GCN-{}x{}".format(self.config.m_num_stage, self.config.m_num_nodes)

        base_name += super(AttentionModel, self).model_name()

        return base_name

    def positional_encoding_sinusoid(self, seq_len, d_model, device='cpu'):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                             -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, batch, velocityregression):
        """ Forward pass. """

        # ------------------------------------------
        # 1. Input
        imu_inputs = self.prepare_inputs(batch.get_inputs())  # (N, F, D_in)

        if velocityregression is not None:
            inputs_combined = torch.cat((velocityregression, imu_inputs), dim=-1)
        else:
            inputs_combined = imu_inputs

        # Generate key padding mask (True = padded position)
        key_padding_mask = get_key_padding_mask(inputs_combined)

        # ------------------------------------------
        # 2. Embedding
        x = self.embedding_layer(inputs_combined)  # (N, F, D_emb)

        # ------------------------------------------
        # 3. Projection to the Positional Encoding
        x = self.projection_layer(x)  # (N, F, D_pos)

        if self.positional_encoding_type == "learnable":
            x = self.positional_encoding(x)
        elif self.positional_encoding_type == "sinusoidal":
            x = x + self.positional_encoding_buffer[:, :x.size(1), :].to(x.device)

        # ------------------------------------------
        # 4. Transformer Encoder
        att_out = self.TransformerEncoder(x, padding_mask=key_padding_mask)

        output = {}

        # GCN
        if self.use_gcn:
            att_out = self.gcn(att_out)

        if self.VERSION == 'JC':
            pose_hat = self.out_layer(att_out)  # (N, F, self.output_size)
            pose_hat = self.update_pose_hat(pose_hat, self.indices_to_fill, self.indices_missing, batch)
            output['pose_hat'] = pose_hat

        elif self.VERSION == 'FK':

            """Joint Rotations with interested joints """
            joint_rot_hat = self.out_layer(att_out)  # (N, F, self.output_size)
            joint_rot_hat = self.special_procrustes(joint_rot_hat)
            joint_rot_hat = self.update_joint_rot_hat(joint_rot_hat, self.indices_to_fill, self.indices_missing, batch)
            pose_hat = self.do_fk(batch.bone_offsets, joint_rot_hat)             # Calculate pose.
            output['joint_rot_hat'] = joint_rot_hat
            output['pose_hat'] = pose_hat

        # Contact Phase
        if self.predict_contact:
            output['contact_hat'] = self.to_contact_phase(att_out)

        if self.predict_velocity:
            velocity_hat = self.to_velocity(att_out)
            output['velocity_hat'] = velocity_hat

        if self.predict_joints:
            relative_joints_hat = self.to_joints(att_out)
            output['relative_joints_hat'] = relative_joints_hat

        # Phase
        if self.predict_phase:
            relative_phase_hat = self.to_relative_phase(att_out)
            output['relative_phase_hat'] = relative_phase_hat

        # Orientation
        if self.predict_orientation:
            orientation_hat = self.orientation(att_out)
            output['orientation_hat'] = orientation_hat.reshape(orientation_hat.shape[0], orientation_hat.shape[1], -1, 3, 3)

        # Root
        if self.predict_root:
            output['root_hat'] = self.to_root(att_out)
            # output['root_hat'] = self.get_root(lstm_out) # Velocity

        return output

# Diffusion-based attention model
class DiffusionTransformerModel(BaseModel):
    def __init__(self, config, *args):
        super(DiffusionTransformerModel, self).__init__(config, *args)

    def create_model(self):
        output_size_vrn = self.output_size // 3 if self.VERSION == 'FK' else self.output_size
        input_size_final = self.input_size if self.config.m_type != 'vrndiff' else self.input_size + output_size_vrn

        self.embedding_layer = MLPLayer(
            input_size=input_size_final,
            embedding_size=self.config.m_embedding_MLP[-1],
            dropout=self.config.m_dropout,
            use_batch_norm=self.config.use_batch_norm
        )
        self.project_to_att = nn.Linear(self.config.m_embedding_MLP[-1], self.config.m_embedding_attention[0])

        # Positional encoding
        self.positional_encoding_type = getattr(self.config, "m_positional_encoding_type", "learnable")
        if self.positional_encoding_type == "learnable":
            self.positional_encoding = PositionalEncoding(self.config.window_size, self.config.m_embedding_attention[0])
        elif self.positional_encoding_type == "sinusoidal":
            self.register_buffer("positional_encoding_buffer",
                                 self.positional_encoding_sinusoid(
                                     seq_len=self.config.window_size,
                                     d_model=self.config.m_embedding_attention[0],
                                     device=C.DEVICE
                                 ))
        else:
            raise ValueError("Unsupported positional encoding")

        # Transformer
        self.TransformerEncoder = TransformerEncoder(
            embed_dim=self.config.m_embedding_attention,
            num_layers=self.config.m_num_layers_attention,
            num_heads=self.config.m_num_heads_attention,
            ff_hidden_dim=self.config.m_num_hidden_units_attention,
            dropout=self.config.m_dropout,
            skip_connection=self.config.m_skip_connections,
            window_size=self.config.m_window_attention
        )

        self.output_layer = nn.Linear(self.config.m_embedding_attention[-1], self.output_size)
        self.output_layer_noise = nn.Linear(self.config.m_embedding_attention[-1], self.config.m_embedding_attention[-1])

        # Projection layer
        self.projection_layer = nn.Linear(self.output_size, self.config.m_embedding_attention[-1])

        # Diffusion schedule
        self.register_buffer('betas', self.get_betas(self.config.beta_schedule, self.config.num_diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(1.0 - self.betas, dim=0))

        # Timestep embedding
        self.timestep_embedding = SinusoidalTimestepEmbedding(self.config.m_embedding_attention[-1])
        self.t_embed_mlp = nn.Sequential(
            nn.Linear(self.config.m_embedding_attention[-1], self.config.m_embedding_attention[-1]),
            nn.ReLU(),
            nn.Linear(self.config.m_embedding_attention[-1], self.config.m_embedding_attention[-1])
        )

        if getattr(self.config, 'use_gcn', False):
            self.gcn = GCN(input_feature=self.config.m_embedding_attention,
                           hidden_feature=self.config.m_num_hidden_units_attention,
                           drop_out=self.config.m_dropout,
                           num_stage=self.config.m_num_stage,
                           node_n=self.config.m_window_attention)

        if self.predict_spl:
            self.out_layer = SPL(
                input_size=self.config.m_embedding_attention[-1],
                number_joints=self.n_joints_spl,
                joint_size=9,
                num_layers=1,
                hidden_units=self.config.m_hidden_units_SPL[-1],
                predict_arms=self.predict_arms,
                predict_head=self.predict_head
            )

        else:
            self.out_layer = nn.Linear(self.config.m_embedding_attention[-1], self.output_size)

        if self.predict_contact:
            self.to_contact_phase = nn.Linear(in_features=self.config.m_embedding_attention[-1], out_features=4)

        if self.predict_velocity:
            self.to_velocity = nn.Linear(self.config.m_embedding_attention[-1], output_size_vrn)

        if self.predict_joints:
            self.to_joints = nn.Sequential(
                nn.Linear(self.config.m_embedding_attention[-1], self.config.m_embedding_attention[-1] // 2),
                nn.ReLU(),
                nn.Linear(self.config.m_embedding_attention[-1] // 2, len(C.FK_PARENTS_Full)*3),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )

        if self.predict_phase:
            self.to_relative_phase = nn.Sequential(
                nn.Linear(self.config.m_embedding_attention[-1], self.config.m_embedding_attention[-1] // 2),
                nn.ReLU(),
                nn.Linear(self.config.m_embedding_attention[-1] // 2, len(C.FK_PARENTS_Full)),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )

        if self.predict_orientation:
            self.orientation = nn.Linear(in_features=self.config.m_embedding_attention[-1],
                                         out_features=self.output_size_ori)

        if self.predict_root:
            if self.VERSION == 'JC':
                self.to_root = nn.Linear(self.m_embedding_attention[-1], 3)

            elif self.VERSION == 'FK':
                self.to_root = MLPLayer(input_size=self.config.m_embedding_attention[-1],
                                   output_size=3,dropout=self.config.m_dropout,
                                   use_batch_norm=self.config.use_batch_norm)

    def model_name(self):
        """A summary string of this model."""
        if self.config.m_type == 'vrndiff':
            base_name = "VRN-{}x{}x{}-Diff".format(self.config.m_num_layers_VRN,
                                                    clean_string(str(self.config.m_hidden_units_VRN)),
                                                    clean_string(str(self.config.m_embedding_VRNMLP)))
        else:
            base_name = "Diff"

        base_name += "-{}-{}x{}x{}x{}x{}x{}x{}x{}".format(self.config.m_positional_encoding_type,
                                                 self.config.m_num_layers_attention,
                                                 clean_string(str(self.config.m_embedding_MLP)),
                                                 clean_string(str(self.config.m_embedding_attention)),
                                                 clean_string(str(self.config.m_num_hidden_units_attention)),
                                                 clean_string(str(self.config.m_num_heads_attention)),
                                                 str(self.config.num_diffusion_steps),
                                                 str(self.config.m_window_attention),
                                                 clean_string(str(self.config.m_hidden_units_SPL)))

        base_name += super(DiffusionTransformerModel, self).model_name()

        return base_name

    def get_betas(self, schedule: str, num_steps: int):
        if schedule == 'linear':
            return torch.linspace(1e-4, 0.02, num_steps)
        elif schedule == 'cosine':
            return self.betas_from_cosine_schedule(num_steps)
        else:
            raise ValueError(f"Unsupported beta schedule: {schedule}")

    @staticmethod
    def betas_from_cosine_schedule(num_diffusion_steps: int, s: float = 0.008):
        steps = num_diffusion_steps + 1
        x = torch.linspace(0, num_diffusion_steps, steps)
        alphas_bar = torch.cos(((x / num_diffusion_steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def positional_encoding_sinusoid(self, seq_len, d_model, device='cpu'):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                             -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, batch, velocityregression):
        imu_inputs = self.prepare_inputs(batch.get_inputs())
        motion_gt = batch.joint_rotations
        B, T, J = motion_gt.shape
        motion_gt = motion_gt.view(B, T, -1)

        if velocityregression is not None:
            inputs_combined = torch.cat((velocityregression, imu_inputs), dim=-1)
        else:
            inputs_combined = imu_inputs

        # 1. Sample timestep and generate noise
        t = torch.randint(0, self.betas.shape[0], (B,), device=C.DEVICE).long()
        noise = torch.randn_like(motion_gt)
        alpha_bar_t = self.alpha_bars[t].view(B, 1, 1)

        # 2. Generate noised motion
        motion_noised_raw = torch.sqrt(alpha_bar_t) * motion_gt + torch.sqrt(1 - alpha_bar_t) * noise

        # 3. Project both noised motion and noise into transformer space
        motion_noised = self.projection_layer(motion_noised_raw)  # → used in input
        noise_proj = self.projection_layer(noise)  # → used in loss

        # 4. Positional + timestep encoding:
        x_cond = self.embedding_layer(inputs_combined)
        x_cond = self.project_to_att(x_cond)

        if self.positional_encoding_type == "learnable":
            x_cond = self.positional_encoding(x_cond)
        elif self.positional_encoding_type == "sinusoidal":
            x_cond = x_cond + self.positional_encoding_buffer[:, :T, :].to(C.DEVICE)

        # Timestep embedding
        t_embed = self.timestep_embedding(t)                  # [B, D]
        t_embed = self.t_embed_mlp(t_embed)                   # [B, D]
        t_embed = t_embed.unsqueeze(1).repeat(1, T, 1)        # [B, T, D]

        # Fuse
        x = motion_noised + x_cond + t_embed
        att_out = self.TransformerEncoder(x)
        predicted_noise = self.output_layer_noise(att_out)

        # 6. Return everything needed
        output = {
            'predicted_noise': predicted_noise,
            'true_noise': noise_proj}

        if self.VERSION == 'JC':
            pose_hat = self.output_layer(x)  # (N, F, self.output_size)
            pose_hat = self.update_pose_hat(pose_hat, self.indices_to_fill, self.indices_missing, batch)
            output['pose_hat'] = pose_hat

        elif self.VERSION == 'FK':

            """Joint Rotations with interested joints """

            joint_rot_hat = self.out_layer(att_out)  # (N, F, self.output_size)

            joint_rot_hat = self.special_procrustes(joint_rot_hat)

            joint_rot_hat = self.update_joint_rot_hat(joint_rot_hat, self.indices_to_fill, self.indices_missing, batch)

            pose_hat = self.do_fk(batch.bone_offsets, joint_rot_hat)  # Calculate pose.

            output['joint_rot_hat'] = joint_rot_hat

            output['pose_hat'] = pose_hat

            # Contact Phase
            if self.predict_contact:
                output['contact_hat'] = self.to_contact_phase(att_out)

            if self.predict_velocity:
                velocity_hat = self.to_velocity(att_out)
                output['velocity_hat'] = velocity_hat

            if self.predict_joints:
                relative_joints_hat = self.to_joints(att_out)
                output['relative_joints_hat'] = relative_joints_hat

            # Phase
            if self.predict_phase:
                relative_phase_hat = self.to_relative_phase(att_out)
                output['relative_phase_hat'] = relative_phase_hat

            # Orientation

            if self.predict_orientation:
                orientation_hat = self.orientation(att_out)

                output['orientation_hat'] = orientation_hat.reshape(orientation_hat.shape[0], orientation_hat.shape[1], -1, 3, 3)

            # Root

            if self.predict_root:
                output['root_hat'] = self.to_root(att_out)

                # output['root_hat'] = self.get_root(lstm_out) # Velocity

            return output

    @torch.no_grad()
    def sample(self, imu_data, T, J, D):
        B = imu_data.shape[0]
        motion_dim = J * D
        x = torch.randn(B, T, motion_dim, device=imu_data.device)

        x_cond = self.embedding_layer(imu_data)
        x_cond = self.project_to_att(x_cond)

        if self.positional_encoding_type == "learnable":
            x_cond = self.positional_encoding(x_cond)
        elif self.positional_encoding_type == "sinusoidal":
            x_cond = x_cond + self.positional_encoding_buffer[:, :T, :].to(x_cond.device)

        for t in reversed(range(self.betas.shape[0])):
            t_tensor = torch.full((B,), t, device=imu_data.device, dtype=torch.long)
            t_embed = self.timestep_embedding(t_tensor)
            t_embed = self.t_embed_mlp(t_embed).unsqueeze(1).repeat(1, T, 1)

            x_fused = x + x_cond + t_embed
            x_encoded = self.TransformerEncoder(x_fused)
            noise_pred = self.output_layer(x_encoded)

            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            x = (1 / torch.sqrt(1 - beta)) * (x - beta / torch.sqrt(1 - alpha_bar) * noise_pred)

            # # Non - Deterministic
            # beta = self.betas[t]
            # x = (1 / torch.sqrt(1 - beta)) * (x - beta / torch.sqrt(1 - alpha_bar) * noise_pred)
            # if t > 0:
            #     x += torch.sqrt(beta) * torch.randn_like(x)

            # Deterministiic
            if t > 0:
                alpha_bar_prev = self.alpha_bars[t - 1]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=x.device)

            # Estimate x0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

            # Deterministic update (η = 0)
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * noise_pred

        return x.view(B, T, J, D)

# DCT
class RNNDCT(BaseModel):

    def __init__(self, config):
        super(RNNDCT, self).__init__(config)

        self.n_dct = config.n_dct
        self.dct_m, self.idct_m = self.get_dct_matrix(self.n_frames)

    def create_model(self):

        self.num_directions = 2 if self.config.m_bidirectional else 1
        output_size_vrn = self.output_size // 3 if self.VERSION == 'FK' else self.output_size
        input_size_final = self.input_size if self.config.m_type != 'vrnrnn' else self.input_size + output_size_vrn

        # MLP Encoder Layers
        self.embedding_layer = MLPLayer(input_size=input_size_final, embedding_size=self.config.m_embedding_MLP[-1],
                            dropout=self.config.m_dropout, use_batch_norm=self.config.use_batch_norm)

        self.rnn = RNNLayer(input_size=input_size_final, output_size=None,
                            num_layers=self.config.m_num_layers_RNN, hidden_units=self.config.m_hidden_units_RNN,
                            bidirectional=self.config.m_bidirectional, dropout=self.config.m_dropout,
                            learn_init_state=self.config.m_learn_init_state, use_batch_norm=self.config.use_batch_norm)

        if self.predict_spl:
            self.out_layer = SPL(
                input_size=self.config.m_hidden_units_RNN[-1] * self.num_directions,
                number_joints=self.n_joints_spl,
                joint_size=9,
                num_layers=1,
                hidden_units=self.config.m_hidden_units_SPL[-1],
                predict_arms=self.predict_arms,
                predict_head=self.predict_head
            )

        else:
            self.out_layer = nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions, self.output_size)

        if self.predict_contact:
            self.to_contact_phase = nn.Linear(in_features=self.config.m_hidden_units_RNN[-1] * self.num_directions, out_features=4)

        if self.predict_velocity:
            self.to_velocity = nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions, output_size_vrn)

        if self.predict_joints:
            self.to_joints = nn.Sequential(
                nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions, self.config.m_hidden_units_RNN[-1] // 2 * self.num_directions),
                nn.ReLU(),
                nn.Linear(self.config.m_hidden_units_RNN[-1] // 2 * self.num_directions, len(C.FK_PARENTS_Full)*3),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )

        if self.predict_phase:
            self.to_relative_phase = nn.Sequential(
                nn.Linear(self.config.m_hidden_units_RNN[-1] * self.num_directions, self.config.m_hidden_units_RNN[-1] // 2 * self.num_directions),
                nn.ReLU(),
                nn.Linear(self.config.m_hidden_units_RNN[-1] // 2 * self.num_directions, len(C.FK_PARENTS_Full)),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )

        if self.predict_orientation:
            self.orientation = nn.Linear(in_features=279,
                                         out_features=self.output_size_ori)

        if self.predict_root:
            if self.VERSION == 'JC':
                self.to_root = nn.Linear(self.m_hidden_units_RNN[-1] * self.num_directions, 3)

            elif self.VERSION == 'FK':
                self.to_root = MLPLayer(input_size=self.config.m_hidden_units_RNN[-1] * self.num_directions,
                                   output_size=3,dropout=self.config.m_dropout,
                                   use_batch_norm=self.config.use_batch_norm)

    def model_name(self):
        """A summary string of this model."""
        if self.predict_spl:
            base_name = "DCT-RNNMLP-{}x{}x{}x{}".format(self.config.m_num_layers_RNN,
                                                        clean_string(str(self.config.m_hidden_units_RNN)),
                                                        clean_string(str(self.config.m_embedding_MLP)),
                                                        clean_string(str(self.config.m_hidden_units_SPL)))
        else:
            base_name = "DCT-RNNMLP-{}x{}x{}".format(self.config.m_num_layers_RNN,
                                                     clean_string(str(self.config.m_hidden_units_RNN)),
                                                     clean_string(str(self.config.m_embedding_MLP)))

        # base_name = "DCT-RNN-{}x{}".format(self.config.m_num_layers_RNN, clean_string(str(self.config.m_hidden_units_RNN)))
        if self.config.m_bidirectional:
            base_name = "Bi" + base_name

        base_name += super(RNNDCT, self).model_name()
        return base_name

    @staticmethod
    def get_dct_matrix(N):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        dct_m, idct_m = torch.Tensor(dct_m).to(C.DEVICE), torch.Tensor(idct_m).to(C.DEVICE)
        return dct_m, idct_m

    def forward(self, batch, window_size=None, is_new_sequence=True):
        """ The forward pass. """
        if is_new_sequence:
            self.rnn.final_state = None
        self.rnn.init_state = self.rnn.final_state

        n = batch.batch_size

        # ------------------------------------------
        # 1. Input

        # IMU Input
        inputs_ = self.prepare_inputs(batch.get_inputs())  # (n, f, number of senosrs * 3 + 2)

        # Apply DCT to the input data
        dct_m = self.dct_m.clone()
        inputs_dct = torch.matmul(dct_m.unsqueeze(dim=0), inputs_)

        # ------------------------------------------
        # 2. Embedding
        # Embedding
        # x = self.embedding_layer(inputs_dct)  # (n, f, n_emb)

        # ------------------------------------------
        # 3. RNN
        rnn_out = self.rnn(inputs_dct, batch.seq_lengths)

        # Map embeddings to output dimension
        out = self.out_layer(rnn_out)  # (n, f, jc)

        # Apply IDCT to transform back to time domain
        idct_m = self.idct_m.clone()  # Apply IDCT to transform back to time domain
        out_time = torch.matmul(idct_m.unsqueeze(dim=0).repeat(n, 1, 1)[:, :, :self.n_dct],
                                out[:, :self.n_dct])  # IDCT applied → time domain

        output = {}  # Initialize an empty dictionary to hold the output.

        if self.VERSION == 'JC':
            # Pose
            pose_hat = torch.matmul(idct_m.unsqueeze(dim=0).repeat(n, 1, 1)[:, :, :self.n_dct],
                                    out[:, :self.n_dct])
            pose_hat = self.update_pose_hat(pose_hat, self.indices_to_fill, self.indices_missing, batch)

            output = {}  # Initialize an empty dictionary to hold the output.

        elif self.VERSION == 'FK':

            joint_rot_hat = self.special_procrustes(out_time)
            joint_rot_hat = self.update_joint_rot_hat(joint_rot_hat, self.indices_to_fill, self.indices_missing, batch)
            pose_hat = self.do_fk(batch.bone_offsets, joint_rot_hat)
            output = {'pose_hat': pose_hat, 'joint_rot_hat': joint_rot_hat, 'root_hat': None}

            # Contact Phase
            if self.predict_contact:
                output['contact_hat'] = self.to_contact_phase(rnn_out)

            if self.predict_velocity:
                velocity_hat = self.to_velocity(rnn_out)
                output['velocity_hat'] = velocity_hat

            if self.predict_joints:
                relative_joints_hat = self.to_joints(rnn_out)
                output['relative_joints_hat'] = relative_joints_hat

            if self.predict_phase:
                relative_phase_hat = self.to_relative_phase(rnn_out)
                output['relative_phase_hat'] = relative_phase_hat

            if self.predict_orientation:
                orientation_hat = self.orientation(out_time)
                output['orientation_hat'] = orientation_hat.reshape(orientation_hat.shape[0], orientation_hat.shape[1],
                                                                    -1,                                                    3, 3)
            # Root
            if self.predict_root:
                output['root_hat'] = self.to_root(out_time)
                # output['root_hat'] = self.get_root(lstm_out) # Velocity

            return output

# DCT with Attention
class DCTattention(BaseModel):
    """
    Copied and adapted from "https://github.com/wei-mao-2019/HisRepItself/tree/master".
    """

    def __init__(self, config, *args):
        super(DCTattention, self).__init__(config, *args)
        self.n_dct = config.window_size
        self.dct_m, self.idct_m = self.get_dct_matrix(self.n_frames)

    def create_model(self):

        output_size_vrn = self.output_size // 3 if self.VERSION == 'FK' else self.output_size
        input_size_final = self.input_size if self.config.m_type != 'vrnrnn' else self.input_size + output_size_vrn

        self.embedding_layer = MLPLayer(input_size=input_size_final,
                                        embedding_size=self.config.m_embedding_MLP[-1],
                                        dropout=self.config.m_dropout,
                                        use_batch_norm=self.config.use_batch_norm
                                        )

        self.project_to_att = nn.Linear(self.config.m_embedding_MLP[-1], self.config.m_embedding_attention[0])

        # Positional encoding
        self.positional_encoding_type = getattr(self.config, "m_positional_encoding_type", "learnable")
        if self.positional_encoding_type == "learnable":
            self.positional_encoding = PositionalEncoding(self.config.window_size, self.config.m_embedding_attention[0])
        elif self.positional_encoding_type == "sinusoidal":
            self.register_buffer("positional_encoding_buffer",
                                 self.positional_encoding_sinusoid(
                                     seq_len=self.config.window_size,
                                     d_model=self.config.m_embedding_attention[0],
                                     device=C.DEVICE
                                 ))
        else:
            raise ValueError("Unsupported positional encoding")

        self.TransformerEncoder = TransformerEncoder(
            embed_dim=self.config.m_embedding_attention,
            num_layers=self.config.m_num_layers_attention,
            num_heads=self.config.m_num_heads_attention,
            ff_hidden_dim=self.config.m_num_hidden_units_attention,
            dropout=self.config.m_dropout,
            skip_connection=self.config.m_skip_connections,
            window_size=self.config.m_window_attention
        )

        if getattr(self.config, 'use_gcn', False):
            self.gcn = GCN(input_feature=self.config.m_embedding, hidden_feature=self.config.m_hidden_units,
                           drop_out=self.config.m_dropout,
                           num_stage=self.config.num_stage,
                           node_n=self.config.m_num_nodes)


        if self.predict_spl:
            self.out_layer = SPL(
                input_size=self.config.m_embedding_attention[-1],
                number_joints=self.n_joints_spl,
                joint_size=9,
                num_layers=1,
                hidden_units=self.config.m_hidden_units_SPL[-1],
                predict_arms=self.predict_arms,
                predict_head=self.predict_head
            )

        else:
            self.out_layer = nn.Linear(self.config.m_embedding_attention[-1], self.output_size)

        if self.predict_contact:
            self.to_contact_phase = nn.Linear(in_features=self.config.m_embedding_attention[-1], out_features=4)

        if self.predict_velocity:
            self.to_velocity = nn.Linear(self.config.m_embedding_attention[-1], output_size_vrn)

        if self.predict_joints:
            self.to_joints = nn.Sequential(
                nn.Linear(self.config.m_embedding_attention[-1], self.config.m_embedding_attention[-1] // 2),
                nn.ReLU(),
                nn.Linear(self.config.m_embedding_attention[-1] // 2, len(C.skeleton_pairs)*3),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )

        if self.predict_phase:
            self.to_relative_phase = nn.Sequential(
                nn.Linear(self.config.m_embedding_attention[-1], self.config.m_embedding_attention[-1] // 2),
                nn.ReLU(),
                nn.Linear(self.config.m_embedding_attention[-1] // 2, len(C.skeleton_pairs)),
                nn.Tanh(),  # to constrain output to [-1, 1]
            )

    def model_name(self):
        """A summary string of this model."""
        base_name = "DCTATT"

        if self.predict_spl:

            base_name += "-{}-{}x{}x{}x{}x{}x{}x{}".format(self.config.m_positional_encoding_type,
                                                     self.config.m_num_layers_attention,
                                                     clean_string(str(self.config.m_embedding_MLP)),
                                                     clean_string(str(self.config.m_embedding_attention)),
                                                     clean_string(str(self.config.m_num_hidden_units_attention)),
                                                     clean_string(str(self.config.m_num_heads_attention)),
                                                     str(self.config.m_window_attention),
                                                     clean_string(str(self.config.m_hidden_units_SPL)))
        else:
            base_name += "-{}-{}x{}x{}x{}x{}x{}".format(self.config.m_positional_encoding_type,
                                                           self.config.m_num_layers_attention,
                                                           clean_string(str(self.config.m_embedding_MLP)),
                                                           clean_string(str(self.config.m_embedding_attention)),
                                                           clean_string(str(self.config.m_num_hidden_units_attention)),
                                                           clean_string(str(self.config.m_num_heads_attention)),
                                                           str(self.config.m_window_attention))

        if self.config.use_gcn:
            base_name += "-GCN-{}x{}".format(self.num_stage, self.d_model)
        base_name += super(DCTattention, self).model_name()
        return base_name

    @staticmethod
    def get_dct_matrix(N):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        dct_m, idct_m = torch.Tensor(dct_m).to(C.DEVICE), torch.Tensor(idct_m).to(C.DEVICE)
        return dct_m, idct_m

    def positional_encoding_sinusoid(self, seq_len, d_model, device='cpu'):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                             -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, batch):
        """ Forward pass. """

        n = batch.batch_size

        # ------------------------------------------
        # 1. Input

        # IMU Input
        inputs_ = self.prepare_inputs(batch.get_inputs())  # (n, f, number of senosrs * 3 + 2)

        # Apply DCT to the input data
        dct_m = self.dct_m.clone()
        inputs_dct = torch.matmul(dct_m.unsqueeze(dim=0), inputs_)

        # Generate key padding mask (True = padded position)
        key_padding_mask = get_key_padding_mask(inputs_dct)

        # ------------------------------------------
        # 2. Embedding
        # Embedding
        x = self.embedding_layer(inputs_dct)  # (n, f, n_emb)
        x = self.project_to_att(x)

        # ------------------------------------------
        # 3. Projection to the Positional Encoding

        if self.positional_encoding_type == "learnable":
            x = self.positional_encoding(x)
        elif self.positional_encoding_type == "sinusoidal":
            x = x + self.positional_encoding_buffer[:, :x.size(1), :].to(x.device)

        # ------------------------------------------
        # 4. Transformer Encoder
        # Self-Attention
        att_out = self.TransformerEncoder(x, padding_mask=key_padding_mask)

        # Map embeddings to output dimension
        out = self.out_layer(att_out)  # (n, f, jc)

        # GCN
        if getattr(self.config, 'use_gcn', False):
            out = out.transpose(1, 2)  # (n, jc, f)
            out = self.gcn(out)
            out = out.transpose(1, 2)  # (n, f, jc)

        # Apply IDCT to transform back to time domain
        idct_m = self.idct_m.clone()
        out_time = torch.matmul(idct_m.unsqueeze(dim=0).repeat(n, 1, 1)[:, :, :self.n_dct],
                        out[:, :self.n_dct])  # IDCT applied → time domain

        if self.VERSION == 'JC':
            # Pose
            pose_hat = torch.matmul(idct_m.unsqueeze(dim=0).repeat(n, 1, 1)[:, :, :self.n_dct],
                                    out[:, :self.n_dct])
            pose_hat = self.update_pose_hat(pose_hat, self.indices_to_fill, self.indices_missing, batch)

            output = {'root_hat': None, 'contact_hat': None, 'pose_hat': pose_hat}

        elif self.VERSION == 'FK':
            joint_rot_hat = self.special_procrustes(out_time)
            joint_rot_hat = self.update_joint_rot_hat(joint_rot_hat, self.indices_to_fill, self.indices_missing, batch)
            pose_hat = self.do_fk(batch.bone_offsets, joint_rot_hat)
            output = {'pose_hat': pose_hat, 'joint_rot_hat': joint_rot_hat, 'root_hat': None}

            # Contact Phase
            if self.predict_contact:
                output['contact_hat'] = self.to_contact_phase(att_out)

            if self.predict_velocity:
                velocity_hat = self.to_velocity(att_out)
                output['velocity_hat'] = velocity_hat

            if self.predict_joints:
                relative_joints_hat = self.to_joints(att_out)
                output['relative_joints_hat'] = relative_joints_hat

            if self.predict_phase:
                relative_phase_hat = self.to_relative_phase(att_out)
                output['relative_phase_hat'] = relative_phase_hat

            if self.predict_orientation:
                orientation_hat = self.orientation(att_out)
                output['orientation_hat'] = orientation_hat.reshape(orientation_hat.shape[0], orientation_hat.shape[1],
                                                                    -1,                                                    3, 3)
            # Root
            if self.predict_root:
                output['root_hat'] = self.to_root(att_out)
                # output['root_hat'] = self.get_root(lstm_out) # Velocity

            return output


import os
import cv2
import torch
import numpy as np
import pandas as pd

from tabulate import tabulate
from openpyxl import load_workbook
from collections import defaultdict

from torchmetrics.classification import BinaryAccuracy

from DK00_Utils.DK00_UT00_config import CONSTANTS as C
from DK00_Utils.DK00_UT04_logginghelpers import mask_from_seq_lengths


class MetricsEngine(object):
    """Helper class to compute metrics over a dataset."""

    def __init__(self, config):
        self.VERSION = config.VERSION
        self.m_type = config.m_type
        self.predict_arms = getattr(config, 'predict_arms', False)
        self.predict_head = getattr(config, 'predict_head', False)
        self.DEVICE = C.DEVICE

        # Joint and sensor index selection
        if self.predict_arms and self.predict_head:
            self.eval_joints = C.JC_EVAL_JOINTS_Full if self.VERSION == 'JC' else C.FK_EVAL_JOINTS_FUll
            self.eucl_eval_joints = C.FK_JOINTS_FUll
            self.eval_ori = [0, 1, 2, 3, 4, 5]
            self.eval_names = ['LF', 'RF', 'LArm', 'RArm', 'HD', 'PV']

        elif not self.predict_arms and self.predict_head:
            self.eval_joints = C.JC_EVAL_JOINTS_NoArms if self.VERSION == 'JC' else C.FK_EVAL_JOINTS_NoArms
            self.eucl_eval_joints = C.FK_JOINTS_NoArms
            self.eval_ori = [0, 1, 4, 5]
            self.eval_names = ['LF', 'RF', 'HD','PV']

        elif not self.predict_arms and not self.predict_head:
            self.eval_joints = C.JC_EVAL_JOINTS_NoArmsHead if self.VERSION == 'JC' else C.FK_EVAL_JOINTS_NoArmsHead
            self.eucl_eval_joints = C.FK_JOINTS_NoArmsHead
            self.eval_ori = [0, 1, 5]
            self.eval_names = ['LF', 'RF', 'PV']

        if self.VERSION == 'FK': ## using this one currently as these are the main joints are interested
            self.eucl_eval_joints =C.FK_JOINTS_FUll

            self.phase_joints = ['RhipUleg','LHipUleg',
                                 'RUlegleg','Rlegft','Rftrtoe','Rtoertoeend',
                                 'LUlegleg','Llegft','Lftrtoe','Ltoertoeend',
                                 'S3_RShoulder','S3_Lshoulder','Rshoulderarm','Lshoulderarm',
                                 'RarmRfarm','LarmLfarm','RfarmRhand','LfarmLhand']


        self.storage = defaultdict(list)

    def reset(self):
        self.storage.clear()

    def _get_mask(self, seq_lengths, n, f, device):
        if seq_lengths is not None:
            return torch.arange(f, device=device).expand(n, f) < seq_lengths.unsqueeze(1)
        return torch.ones(n, f, dtype=torch.bool, device=device)

    def _apply_mask(self, data, mask):
        return data[mask] if mask is not None else data

    def _euclidean_distance(self, x, y, mask=None):
        diff = x - y
        dist = torch.norm(diff, dim=-1)  # shape: (N, F, J)
        return self._apply_mask(dist, mask).cpu().numpy()

    def _geodesic_error(self, R1, R2, mask=None):
        rel = torch.matmul(R1.transpose(-1, -2), R2)
        trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
        theta = torch.acos(torch.clamp((trace - 1) / 2, -0.999, 0.999))
        deg = theta * (180.0 / np.pi)
        return self._apply_mask(deg, mask).cpu().numpy()

    def _angle_difference(self, pred, target, mask=None):
        diff = torch.remainder(pred - target + np.pi, 2 * np.pi) - np.pi
        error = torch.mean(torch.abs(diff), dim=-1)  # mean over axis x/y/z
        return self._apply_mask(error, mask).cpu().numpy()

    def compute(self, pose=None,  pose_hat=None,  joint_rot=None,  joint_rot_hat=None, contact=None,  contact_hat=None,
                velocity=None, velocity_hat=None,
                relative_joints=None, relative_joints_hat=None,
                relative_phase=None, relative_phase_hat=None,
                ori_rot=None, ori_rot_hat = None, pose_root=None, pose_root_hat=None,
                seq_lengths=None) :
        """
        Compute the metrics.
        :param pose: The ground-truth pose without the root as a tensor of shape (N, F, N_JOINTS*3).
        :param joint_rot: The ground-truth joint rotation as a tensor of shape  (N, F, N_JOINTS*3*3)
        :param pose_hat: The predicted pose (N, F, N_JOINTS, 3).
        :param joint_rot_hat: The predicted shape. If None the ground-truth shape is assumed (N, F, N_JOINTS, 3, 3).
        :param seq_lengths: An optional tensor of shape (N, ) indicating the true sequence length.
        :param pose_root: An optional tensor of shape (N, F, 3) indicating the ground-truth root pose.
        :param pose_root_hat: An optional tensor of shape (N, F, 3) indicating the estimated root pose.
        """

        n, f = pose.shape[0], pose.shape[1]
        mask = self._get_mask(seq_lengths, n, f, pose.device)

        if pose is not None and pose_hat is not None:
            pose = pose.reshape(n, f, -1, 3)[:, :, self.eval_joints, :]
            pose_hat = pose_hat[:, :, self.eval_joints, :]
            self.storage['mpjpe'].append(self._euclidean_distance(pose, pose_hat, mask))

        if joint_rot is not None and joint_rot_hat is not None:
            joint_rot = joint_rot.reshape(n, f, -1, 3, 3)[:, :, self.eval_joints, :, :]
            joint_rot_hat = joint_rot_hat[:, :, self.eval_joints, :, :]
            self.storage['mpjae'].append(self._geodesic_error(joint_rot, joint_rot_hat, mask))

        if contact is not None and contact_hat is not None:
            ba = BinaryAccuracy(multidim_average='global').to(self.DEVICE)
            self.storage['contact'].append(ba(contact_hat, contact).item())

        if velocity is not None and velocity_hat is not None:
            velocity = velocity.reshape(n, f, -1, 3)[:, :, self.eval_joints, :]
            velocity_hat = velocity_hat.reshape(n, f, -1, 3)[:, :, self.eval_joints, :]
            self.storage['velocity'].append(self._euclidean_distance(velocity, velocity_hat, mask))

        if relative_joints is not None and relative_joints_hat is not None:
            self.storage['rel_joints'].append(self._angle_difference(relative_joints_hat.reshape(n, f, -1, 3), relative_joints, mask))

        if relative_phase is not None and relative_phase_hat is not None:
            self.storage['rel_phase'].append(self._angle_difference(relative_phase_hat, relative_phase, mask))

        if ori_rot is not None and ori_rot_hat is not None:
            ori_rot = ori_rot[:, :, self.eval_ori, :, :]
            ori_rot_hat = ori_rot_hat[:, :, self.eval_ori, :, :]
            self.storage['ori_rot'].append(self._geodesic_error(ori_rot, ori_rot_hat, mask))


    def _aggregate(self, key):
        if key not in self.storage or len(self.storage[key]) == 0:
            return None, None
        data = np.concatenate(self.storage[key], axis=0)
        # the average of the per-joint mean errors
        # the average of the per-joint standard deviations
        return np.mean(data, axis = 0), np.std(data, axis = 0)

    def get_metrics(self):
        metrics = {}

        for key, label, unit in [
            ('mpjpe', 'MPJPE', 'mm'),
            ('mpjae', 'MPJAE', 'deg'),
            ('velocity', 'Velocity Error', 'mm'),
            ('rel_joints', 'RelJoint', 'rad'),
            ('rel_phase', 'RelPhase', 'rad'),
            #('ori_rot', 'MPORE', 'deg'),
        ]:
            mean, std = self._aggregate(key)
            if mean is not None:
                scale = 1000.0 if unit == 'mm' else 1.0
                metrics[f'{label} [{unit}]'] = np.mean(mean) * scale
                metrics[f'{label} STD'] = np.mean(std) * scale

        return metrics

    def save_metrics_excel(self, model_dir, sub_index, config):
        """
        Export per-joint and per-sensor metric distributions to Excel sheets.
        """

        def compute_stats(data_list):
            if not data_list:
                return None, None
            data = np.concatenate(data_list, axis=0)
            return np.mean(data, axis=0), np.std(data, axis=0)

        data_dicts = {}

        # Map storage key to readable name and optional label source
        metric_map = {
            'mpjpe': ('Euclidean_Distance',  self.eucl_eval_joints),
            'mpjae': ('Joint_Rotation_Distance', self.eucl_eval_joints),
            'rel_joints': ('Relative_joint_angle', self.phase_joints),
            'rel_phase': ('Relative_Phase_angle', self.phase_joints),
        }

        for key, (sheet_name, labels) in metric_map.items():
            if key not in self.storage or not self.storage[key]:
                continue
            mean, std = compute_stats(self.storage[key])
            data_dicts[sheet_name] = {
                'Label': labels,
                'Mean': mean,
                'Std': std
            }

        # Prepare path
        subject = str(sub_index).zfill(2)
        output_path = os.path.join(model_dir, f'SonE_{subject}')
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, 'Metrics_Report.xlsx')

        # Write each dictionary to a separate sheet
        with pd.ExcelWriter(file_path, mode='w') as writer:
            for sheet_name, data in data_dicts.items():
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def to_pretty_string(metrics: dict, model_name: str) -> str:
        """
        Convert dictionary of metrics to formatted table string.
        """
        headers = ['Model'] + list(metrics.keys())
        row = [model_name] + [f'{v:.4f}' for v in metrics.values()]
        return tabulate([row], headers=headers, tablefmt='grid')
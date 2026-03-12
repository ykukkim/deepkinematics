import torch
import torch.nn as nn
import numpy as np

from scipy.spatial.transform import Rotation as R
from DK00_Utils.DK00_UT00_config import CONSTANTS as C

class AdaptiveLossWrapper(nn.Module):
    def __init__(self, loss_names):
        """
        Adaptive weighting of multiple loss terms using uncertainty-based weighting.

        Args:
            loss_names (list[str]): List of loss component names, e.g. ['pose', 'rot', 'vel']
        """
        super().__init__()
        self.log_sigmas = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0)) for name in loss_names
        })

    def forward(self, loss_dict):
        """
        Compute the total weighted loss.

        Args:
            loss_dict (dict[str, torch.Tensor]): Dictionary of raw loss values.

        Returns:
            total_loss (torch.Tensor): Scalar loss with adaptive weighting
            weighted_losses (dict[str, torch.Tensor]): Individual weighted components (no gradients detached)
        """
        total_loss = 0.0
        weighted_losses = {}

        for name, loss in loss_dict.items():
            log_sigma = self.log_sigmas[name]
            weighted = torch.exp(-2 * log_sigma) * loss + log_sigma
            total_loss += weighted
            weighted_losses[name] = weighted

        return total_loss, weighted_losses

def mean_L1_loss(root_delta_hat, root_delta):
    """
    Calculate L1-Norm framewise and give mean back as loss.
    :param root_diff_hat: Predicted root delta. (b, f, 3)
    :param root_delta: Framewise change in root position. (b, f, 3)

    Returns:
    Mean of L1-Norm as loss.
    """
    normed_deltas = torch.norm(root_delta_hat - root_delta, p=1, dim=2)

    loss = torch.mean(normed_deltas)

    return loss

def weighted_mse(pose_hat, pose, fl_weight, foot_index):
    """
    Use different weight for foot position prediction.
    :param fl_weight: Foot loss weight.
    :param foot_index: Indexes of foot joints.
    """
    diff = pose_hat - pose

    mse = (diff * diff)
    mse = mse.mean(dim=0).mean(dim=0).mean(dim=-1)

    mse[foot_index] = mse[foot_index] * fl_weight

    return mse.mean()

def weighted_euler_angle_loss(predicted_euler_angles, target_euler_angles, weights=torch.tensor([10.0, 1.0, 1.0])):
    """
    Compute a weighted MSE loss for Euler angles with angle wrapping.

    Args:
        predicted_euler_angles (torch.Tensor): shape (n, f, J, 3)
        target_euler_angles (torch.Tensor): shape (n, f, J, 3)
        weights (torch.Tensor): shape (3,) - weights for (pitch, yaw, roll)

    Returns:
        torch.Tensor: The weighted MSE loss
    """
    # Wrap angle difference to [-π, π]
    angle_diff = torch.remainder(predicted_euler_angles - target_euler_angles + np.pi, 2 * np.pi) - np.pi
    # angle_diff = predicted_euler_angles - target_euler_angles

    # Safe broadcasting of weights
    weights = weights.view(1, 1, 1, 3).to(predicted_euler_angles.device)

    # Weighted squared error
    weighted_diff = weights.to(C.DEVICE) * angle_diff ** 2

    return torch.mean(weighted_diff)

def geodesic_loss(R_pred, R_true):
    # R_pred, R_true: shape [N, F, J, 3, 3]
    R_rel = torch.matmul(R_pred.transpose(-2, -1), R_true)
    trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)  # shape [N, F, J]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))
    return theta.mean()

def circular_loss(pred, target):
    diff = torch.remainder(pred - target + np.pi, 2 * np.pi) - np.pi
    return (diff ** 2).mean()
import os
import glob
import torch
import time
import wandb
import zipfile
import numpy as np
from DK00_Utils.DK00_UT00_config import CONSTANTS as C


class WandB():

    def __init__(self, project=None, experiment=None, mode='online', dir=str, VERSION=str, init_run=True):
        # Choose directory based on version
        self.version = VERSION
        dir_suffix = 'JC' if VERSION == 'JC' else 'FK'
        dir_with_extra = os.path.join(dir, dir_suffix)

        # Rest of the initialization
        if not os.path.exists(dir_with_extra):
            os.makedirs(dir_with_extra)

        self.run = wandb.init(project=project, name=experiment, dir=dir_with_extra, mode=mode, allow_val_change=True)


    def update_epoch(self, n_epoch):
        wandb.config.epoch = n_epoch

    def get_value(self, key):
        return wandb.config[key]

    def update_config(self, config, n_paras):
        """
        Updates and logs model configuration to Weights & Biases (W&B),
        including base training parameters and model-specific architecture settings.

        Args:
            config: A configuration object (Namespace or similar).
            n_paras: Total number of trainable model parameters.
        """

        # Base config (always logged)
        base_config = {
            "sampling_frequency": getattr(C, 'SAMPLING_FREQUENCY', None),
            "Parameters": n_paras,
            "lr": getattr(config, 'lr', None),
            "batch_size": getattr(config, 'bs_train', None),
            "window_size": getattr(config, 'window_size', None),
            "use_acc_gyro": getattr(config, 'use_acc_gyro', None),
            "use_quats": getattr(config, 'use_quats', None),
            "use_orientation": getattr(config, 'use_orientation', None),

            "predict_arms": getattr(config, 'predict_arms', None),
            "predict_head": getattr(config, 'predict_head', None),
            "predict_contact": getattr(config, 'predict_contact', None),
            "predict_velocity": getattr(config, 'predict_velocity', None),
            "predict_joints": getattr(config, 'predict_joints', None),
            "predict_phase": getattr(config, 'predict_phase', None),
            "predict_root": getattr(config, 'predict_root', None),
            "predict_orientation": getattr(config, 'predict_orientation', None),

            # Loss weights
            "pose_loss": getattr(config, 'm_pose_loss', None),
            "contact_loss": getattr(config, 'm_contact_loss', None),
            "velocity_loss": getattr(config, 'm_velocity_loss', None),
            "root_loss": getattr(config, 'm_root_loss', None),
            "foot_loss": getattr(config, 'm_foot_loss', None),
        }

        # FK-specific losses
        if config.VERSION == "FK":
            base_config.update({
                "joint_rot_loss": getattr(config, 'm_joint_rot_loss', None),
                "joints_loss": getattr(config, 'm_joint_loss', None),
                "phase_loss": getattr(config, 'm_phase_loss', None),
            })

        # Model-specific parameters (optional, depending on model type)
        model_config = {
            "m_dropout": getattr(config, 'm_dropout', None),
            "m_num_layers": getattr(config, 'm_num_layers', None),
        }

        # Merge model-specific config from predefined sets
        model_type = config.m_type.lower()
        model_specific_keys = {
            'rnn': ['m_num_layers_RNN', 'm_hidden_units_RNN', 'm_hidden_units_SPL'],
            'rnndct': ['m_embedding_MLP', 'm_num_layers_RNN', 'm_hidden_units_RNN'],
            'vrnrnn': ['m_num_layers_RNN', 'm_hidden_units_RNN',
                       'm_num_layers_VRN', 'm_hidden_units_VRN', 'm_embedding_VRNMLP',
                       'm_hidden_units_SPL'],
            'att': ['m_embedding_MLP', 'm_num_layers_attention', 'm_num_hidden_units_attention',
                    'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                    'm_hidden_units_SPL'],
            'attdct': ['m_embedding_MLP', 'm_num_layers_attention', 'm_num_hidden_units_attention',
                       'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                       'm_hidden_units_SPL'],
            'diff': ['m_embedding_MLP', 'num_diffusion_steps',
                     'm_num_layers_attention', 'm_num_hidden_units_attention',
                     'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                     'm_hidden_units_SPL'],
            'vrnatt': ['m_num_layers_VRN', 'm_hidden_units_VRN', 'm_embedding_VRNMLP',
                       'm_embedding_MLP', 'm_num_layers_attention', 'm_num_hidden_units_attention',
                       'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                       'm_hidden_units_SPL'],
            'vrndiff': ['m_num_layers_VRN', 'm_hidden_units_VRN', 'm_embedding_VRNMLP',
                        'num_diffusion_steps',
                        'm_embedding_MLP', 'm_num_layers_attention', 'm_num_hidden_units_attention',
                        'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                        'm_hidden_units_SPL'],
        }

        selected_keys = model_specific_keys.get(model_type, [])
        for key in selected_keys:
            model_config[key] = getattr(config, key, None)

        # Merge all configs
        combined_config = {**base_config, **model_config}

        # Clean up None values before logging
        clean_config = {k: v for k, v in combined_config.items() if v is not None}

        # Push to wandb config (or similar)
        self._update_global_config(clean_config)

    def _update_global_config(self, config) -> None:
        """
        Helper method to update global configuration in Weights & Biases.

        Args:
            config: Configuration dictionary to be logged.
        """
        wandb.config.update(config)

    def save_as_graph(self, config) -> None:
        """
        Logs specific parameters as a graph, based on the version of the model.

        Args:
            config: An object encapsulating the model's configuration parameters.
        """
        log_data = {
            "pose_loss": getattr(config, 'm_pose_loss', None),
            "contact_loss": getattr(config, 'm_contact_loss', None),
            "phase_loss": getattr(config, 'orientation_loss', None),
            "root_loss": getattr(config, 'm_root_loss', None),
            "orientation_loss": getattr(config, 'orientation_loss', None),
        }

        # Remove None values before logging
        clean_log_data = {k: v for k, v in log_data.items() if v is not None}

        # Log the data using Weights & Biases
        wandb.log(clean_log_data)

    def log_to_wandb(self, adict, name, prefix=''):
        new_dict = {}
        for k in adict:
            new_dict[k + f' {prefix} {name}'] = adict[k]
        wandb.log({**new_dict})

def count_parameters(params):
    """Count number of trainable parameters in `model`."""
    return sum(p.numel() for p in params if p.requires_grad)

def model_memory_usage(model):
    total_memory = 0
    for param in model.parameters():
        # Memory size is the number of elements multiplied by the element size (in bytes)
        param_memory = param.numel() * param.element_size()
        total_memory += param_memory
        total_memory = total_memory / (1024 ** 2) # MB
    return total_memory

def get_model_dir(experiment_dir, experiment_id, model_id):
    """Return the directory in `experiment_dir` that contains the given `model_id` string."""
    model_dir = glob.glob(os.path.join(experiment_dir, experiment_id, model_id), recursive=False)
    return None if len(model_dir) == 0 else model_dir[0]

def create_model_dir(experiment_dir, experiment_id, model_summary, other_summary=None):
    """Create a new model directory."""
    model_name = "{}".format(model_summary )
    if other_summary:
        model_name = '{}-{}'.format(model_name, other_summary)
    model_dir = os.path.join(experiment_dir,experiment_id, model_name)
    if os.path.exists(model_dir):
        model_dir = get_model_dir(experiment_dir, experiment_id)
    else:
        os.makedirs(model_dir)
    return model_dir

def zip_files(file_list, output_file):
    """Stores files in a zip."""
    if not output_file.endswith('.zip'):
        output_file += '.zip'
    ofile = output_file
    counter = 0
    while os.path.exists(ofile):
        counter += 1
        ofile = output_file.replace('.zip', '_{}.zip'.format(counter))
    zipf = zipfile.ZipFile(ofile, mode="w", compression=zipfile.ZIP_DEFLATED)
    for f in file_list:
        zipf.write(f)
    zipf.close()

def mask_from_seq_lengths(seq_lengths, max_seq_len=None):
    """
    Creates a mask of shape (len(seq_lenghts), S) where S = max(seq_lengths) such that mask[i, j] == 1
    if mask[i, j] < seq_lenghts[i] and 0 otherwise. E.g. if seq_lengths is [4, 3, 2, 4]
    the mask will look like:
       [1, 1, 1, 1]
       [1, 1, 1, 0]
       [1, 1, 0, 0]
       [1, 1, 1, 1]
    :param seq_lengths: A tensor of integers.
    :param max_seq_len: If given will be used instead of max(seq_lengths).
    :return: The described mask as a tensor of shape (len(seq_lengths), max(seq_lengths)).
    """
    if max_seq_len is None:
        max_seq_len = torch.max(seq_lengths)
    n = seq_lengths.shape[0]
    t = torch.arange(max_seq_len).repeat(n, 1).to(dtype=seq_lengths.dtype, device=seq_lengths.device)
    mask = t < seq_lengths.unsqueeze(1)
    return mask

def evaluate_and_log(dataset_name, loader, model, vrn, preprocess_fn, metrics_evaluator, wandb_run, experiment_id, epoch,  VERSION, modeltype):
    start = time.time()
    losses = evaluate(loader, model,vrn, preprocess_fn, metrics_evaluator,VERSION = VERSION, modeltype=modeltype)
    metrics = metrics_evaluator.get_metrics()
    elapsed = time.time() - start

    loss_string = ' '.join(['{}: {:.6f}'.format(k, losses[k]) for k in losses])
    print('[VAL {} Epoch {:0>3d}] {} elapsed: {:.3f} secs'.format(dataset_name.upper(), epoch, loss_string, elapsed))
    print(metrics_evaluator.to_pretty_string(metrics, "{} {}".format(experiment_id, dataset_name.upper())))

    # # Log to Weights & Biases
    if wandb_run:
        wandb_run.log_to_wandb(metrics, dataset_name)
        wandb_run.log_to_wandb(losses, dataset_name, 'loss')

    return losses, metrics

def evaluate(data_loader, net, vrn_net=None, preprocess_fn=None,  metrics_engine=None, window_size=None, VERSION=str,modeltype=str):
    """
    Evaluate the model on a hold-out dataset.
    :param data_loader: The DataLoader object to loop over the validation set.
    :param net: The primary model for evaluation.
    :param vrn_net: The optional Velocity Reconstruction Network model.
    :param preprocess_fn: A function that preprocesses a batch.
    :param metrics_engine: Metrics engine to compute additional metrics besides the loss.
    :param window_size: Sliding window size for large batches, if necessary.
    :param VERSION: The version of the evaluation process or model being used.
    :return: All losses aggregated over the entire validation set.
    """
    # Put the primary model in evaluation mode.
    net.eval()
    if vrn_net is not None:
        vrn_net.eval()  # Only if vrn_net exists.

    loss_vals_agg = []
    n_samples = 0
    metrics_engine.reset()

    with torch.no_grad():
        for b, abatch in enumerate(data_loader):
            # Move data to GPU.
            batch_gpu = abatch.to_gpu()

            # Preprocess.
            batch_gpu = preprocess_fn(batch_gpu)

            # Step 1: Pseudo velocity estimation, only if vrn_net is provided.
            vrn_output = vrn_net(batch_gpu) if vrn_net is not None else None

            model_out = net(batch_gpu, vrn_output) if modeltype in ('rnn','vrnrnn','att', 'vrnatt','diff','vrndiff') else net(batch_gpu)

            # Compute the loss.
            _, loss_vals, _ = net.backward(batch_gpu, model_out)

            if vrn_output is not None:
                loss_vals['velocity_hat'] = vrn_net.backward(batch_gpu, vrn_output)
                value = loss_vals.pop('total_loss', None)
                if value is not None: # Add the key back to the dictionary at the end
                    loss_vals['total_loss'] = value

            # Additional metrics computation, with adjustments for different versions.
            if VERSION == 'JC':
                metrics_engine.compute(pose=batch_gpu.vicon_jc, pose_hat=model_out['pose_hat'],
                                       contact=batch_gpu.vicon_contact, contact_hat=model_out.get('contact_hat', None),
                                       rot = batch_gpu.vicon_ori, rot_hat = model_out.get('orientation_hat', None),
                                       seq_lengths=batch_gpu.seq_lengths,
                                       pose_root=batch_gpu.poses_root)
            else:
                metrics_engine.compute(pose=batch_gpu.pose, pose_hat=model_out['pose_hat'],
                                       joint_rot=batch_gpu.joint_rotations, joint_rot_hat=model_out['joint_rot_hat'],
                                       contact=batch_gpu.vicon_contact, contact_hat=model_out.get('contact_hat', None),
                                       velocity=batch_gpu.velocity, velocity_hat=model_out.get('velocity_hat', None),
                                       relative_joints=batch_gpu.relative_joints, relative_joints_hat=model_out.get('relative_joints_hat', None),
                                       relative_phase=batch_gpu.relative_phase, relative_phase_hat=model_out.get('relative_phase_hat', None),
                                       seq_lengths=batch_gpu.seq_lengths,
                                       pose_root=batch_gpu.poses_root)

            n_samples += batch_gpu.batch_size

    return loss_vals


def save_checkpoint_if_best(current_eval_metric, best_metric, model, vrn, optimizer, epoch, i, checkpoint_file,
                            val_counter):
    if best_metric - current_eval_metric > 3:
        print(' *** New best model ***')
        best_metric = current_eval_metric
        val_counter = 0
        checkpoint_data = {
            'iteration': i,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if vrn is not None:
            checkpoint_data['vrn_state_dict'] = vrn.state_dict()  # Only save if vrn is provided

        torch.save(checkpoint_data, checkpoint_file)
    else:
        val_counter += 1

    return best_metric, val_counter

def final_evaluation_and_log(dataset_name, loader, model, vrn, preprocess_fn, metrics_evaluator, wandb_run, experiment_id, VERSION=str,modeltype=str):
    # Evaluate the dataset
    final_losses = evaluate(loader, model, vrn, preprocess_fn, metrics_evaluator, VERSION=VERSION,modeltype=modeltype)
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in final_losses.items()])
    print('[{} FINAL] {}'.format(dataset_name.upper(), loss_string))

    # Print final metrics
    final_metrics = metrics_evaluator.get_metrics()
    print('FINAL {} METRICS'.format(dataset_name.upper()))
    print(metrics_evaluator.to_pretty_string(final_metrics, "{} {}".format(experiment_id, dataset_name.upper())))

    # # Log to Weights & Biases
    if wandb_run:
        wandb_run.log_to_wandb(final_metrics, dataset_name)
        wandb_run.log_to_wandb(final_losses, dataset_name, 'loss')

"Only for JC"
def get_lines(joints):
    """ Lines to represent stickfigure in AIT viewer, Lines: mode='lines' """

    nr_of_joints = joints.shape[1]

    neck = np.mean(np.array([joints[:, 1, :], joints[:, 2, :]]), axis=0)

    if nr_of_joints == 11:
        hip_avg = np.mean(np.array([joints[:, 3, :], joints[:, 4, :]]), axis=0)

        stick_figure = np.stack((joints[:, 0, :], neck,  # head
                                 joints[:, 1, :], joints[:, 2, :],  # shoulders
                                 neck, hip_avg,  # core
                                 joints[:, 3, :], joints[:, 4, :],  # hips
                                 joints[:, 3, :], joints[:, 5, :], joints[:, 5, :], joints[:, 7, :], joints[:, 7, :],
                                 joints[:, 9, :],  # left leg
                                 joints[:, 4, :], joints[:, 6, :], joints[:, 6, :], joints[:, 8, :], joints[:, 8, :],
                                 joints[:, 10, :]  # right leg
                                 ), axis=1)

    if nr_of_joints == 15:
        hip_avg = np.mean(np.array([joints[:, 7, :], joints[:, 8, :]]), axis=0)

        stick_figure = np.stack((joints[:, 0, :], neck[:, :],  # head
                                 joints[:, 1, :], joints[:, 2, :],  # shoulders
                                 joints[:, 1, :], joints[:, 3, :], joints[:, 3, :], joints[:, 5, :],  # left arm
                                 joints[:, 2, :], joints[:, 4, :], joints[:, 4, :], joints[:, 6, :],  # right arm
                                 neck, hip_avg,  # core
                                 joints[:, 7, :], joints[:, 8, :],  # hips
                                 joints[:, 7, :], joints[:, 9, :], joints[:, 9, :], joints[:, 11, :], joints[:, 11, :],
                                 joints[:, 13, :],  # left leg
                                 joints[:, 8, :], joints[:, 10, :], joints[:, 10, :], joints[:, 12, :],
                                 joints[:, 12, :], joints[:, 14, :]  # right leg
                                 ), axis=1)

    return stick_figure


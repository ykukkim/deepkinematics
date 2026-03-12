import os
import glob
import torch
import numpy as np
import pandas as pd
import scipy.io as sio

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from DK00_Utils.DK00_UT00_config import CONSTANTS as C
from DK00_Utils.DK00_UT00_config import Configuration
from DK00_Utils.DK00_UT01_dataset import ValDataset, RealDataset
from DK00_Utils.DK00_UT01_datasetProcess import RealBatch
from DK00_Utils.DK00_UT01_tranforms import CorrectIMUOrientation, ExtractInitialIMUOffset,\
    ToTensor, TransformOrientation, \
    get_end_to_end_preprocess_fn,\
    NormalizeViconJointCenter, NormalizeHeight
from DK00_Utils.DK00_UT02_models import create_model


""" Evalutation after training i.e. loading the trained model"""
def get_model_configs_and_checkpoints(config):
    """
    Fetch and load configurations and checkpoint paths for the specified model ID.

    Parameters:
        model_id (str): The unique identifier for the model of interest.

    Returns:
        list of dicts: A list where each element is a dictionary containing the model configuration
                       and its corresponding checkpoint path. If no configurations or checkpoints are found,
                       it returns an empty list.
    """
    model_dirs = get_model_dir_eval(C.save_model_DIR, config)
    model_data = []

    for model_dir in model_dirs:
        config_path = os.path.join(model_dir, 'config.json')
        checkpoint_path = os.path.join(model_dir, 'test_model.pth')  # Assuming 'model.pth' is the checkpoint filename

        # Ensure both config and checkpoint exist
        if os.path.exists(config_path) and os.path.exists(checkpoint_path):
            model_config = Configuration.from_json(config_path)
            model_data.append({
                "config": model_config,
                "checkpoint": checkpoint_path,
                "model_dir": model_dir  # Including model_dir for completeness and utility
            })

    if not model_data:
        print(f"No valid configuration and checkpoint files found for model ID {config.model_id}")

    return model_data

def load_model(model_data):
    """Load a model along with its preprocessing functions based on a model ID."""

    # For simplicity, assuming we're dealing with the first model configuration and checkpoint
    model_config = model_data["config"]
    checkpoint_path = model_data["checkpoint"]
    model_dir = model_data["model_dir"]
    # print(f'{checkpoint_path}\n')
    net, vrn = create_model_with_config(model_config)
    net.to(C.DEVICE)

    # Assuming 'VRN' in model_config might mean checking a specific attribute or key in the configuration
    vrn.to(C.DEVICE) if vrn is not None else None

    # Load model weights from checkpoint
    if os.path.isfile(checkpoint_path):
        load_model_weights(checkpoint_path, net)
        print(f'Loaded weights from {checkpoint_path}')
    else:
        print(f"Checkpoint file not found in {model_dir}")

    preprocess_fn = get_end_to_end_preprocess_fn(model_config)
    return net, vrn, model_config, model_dir, preprocess_fn

def load_eval_data(config,sub_indx, model_config,shuffle=False):
    """Load model and the dataset."""
    partition = config.partition if hasattr(config, 'partition') else 'test_specific'
    assert partition in ['easy', 'hard', 'unseen', 'test_specific', 'unseen_specific']

    bs = config.n_samples if hasattr(config, 'n_samples') else 8

    if model_config.VERSION == 'JC':
        transform = [CorrectIMUOrientation(),
                     ExtractInitialIMUOffset(model_config),
                     NormalizeHeight(model_config),
                     NormalizeViconJointCenter(model_config),
                     ToTensor()]

    elif model_config.VERSION == 'FK':
        transform = [ExtractInitialIMUOffset(model_config),
                     ToTensor()]

    if partition == 'easy':

        transform = transforms.Compose(transform)
        valid_data = RealDataset(test_set='easy', transform=transform, train=False,version=model_config.VERSION)
        eval_loader = DataLoader(valid_data,
                                 batch_size=bs,
                                 shuffle=False,
                                 collate_fn=RealBatch.from_sample_list)
    elif partition == 'hard':

        transform = transforms.Compose(transform)
        valid_data = RealDataset(test_set='hard', transform=transform, train=False,version=model_config.VERSION)
        eval_loader = DataLoader(valid_data,
                                 batch_size=bs,
                                 shuffle=False,
                                 collate_fn=RealBatch.from_sample_list)
    elif partition == 'unseen_motion':

        transform = transforms.Compose(transform)
        valid_data = RealDataset(test_set='unseen', transform=transform, train=False,version=model_config.VERSION)
        eval_loader = DataLoader(valid_data,
                                 batch_size=bs,
                                 shuffle=False,
                                 collate_fn=RealBatch.from_sample_list)
    elif partition == 'test_specific':
        #subject_nr = config.subject if hasattr(config, 'subject') else 22
        subject_nr = sub_indx
        subject_nr = str(subject_nr).zfill(2)
        trial = config.trial if hasattr(config, 'trial') else 'Norm_Pre'
        transform = transforms.Compose(transform)
        valid_data = ValDataset(subject_nr, trial, transform=transform,version=model_config.VERSION)
        eval_loader = DataLoader(valid_data,
                                 batch_size=bs,
                                 shuffle=False,
                                 collate_fn=RealBatch.from_sample_list)
    elif partition == 'unseen_specific':
        subject_nr = sub_indx#config.subject if hasattr(config, 'subject') else 22
        subject_nr = str(subject_nr).zfill(2)
        trial = config.trial if hasattr(config, 'trial') else 'Norm_Pre'
        transform = transforms.Compose(transform)
        valid_data = ValDataset(subject_nr, trial, transform=transform,version=model_config.VERSION)
        eval_loader = DataLoader(valid_data,
                                 batch_size=bs,
                                 shuffle=False,
                                 collate_fn=RealBatch.from_sample_list)

    return eval_loader

def get_model_dir_eval(experiment_dir, config):
    """Return a list of directories within `experiment_dir` related to `model_id`."""
    dir_list = glob.glob(os.path.join(experiment_dir, config.experimentid, "*-*"), recursive=False)
    # matching_directories = [dir_path for dir_path in dir_list if
    #                         any(search_string in os.path.basename(dir_path) for search_string in config.m_type)]
    matching_directories = [dir_path for dir_path in dir_list]


    return matching_directories

def create_model_with_config(model_config):
    """Create and return model instances based on configuration, streamlined for VRN condition."""
    models = create_model(model_config)
    models_on_device = tuple(model for model in models)
    if len(models_on_device) == 1:
        model = models_on_device[0]
        vrn_net = None
    elif len(models_on_device) == 2:
        vrn_net, model = models_on_device

    return model, vrn_net

def load_model_weights(checkpoint_file, net, state_key='model_state_dict'):
    """Load and apply checkpoint weights to a network model."""
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=C.DEVICE, weights_only= True)
        net.load_state_dict(checkpoint.get(state_key, {}), strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

def window_generator(batch: RealBatch, window_size):
    """Subdivide a batch into temporal windows of size `window_size`. This is helpful when GPU memory is limited."""
    if window_size is not None:
        assert isinstance(batch, RealBatch)
        seq_len = batch.seq_length
        n_windows = seq_len // window_size  # + int(seq_len % window_size > 0)

        for i in range(3, n_windows):
            sf = i * window_size
            ef = min((i + 1) * window_size, seq_len)
            seq_lengths_new = torch.from_numpy(np.array([ef - sf])).to(dtype=batch.seq_lengths.dtype,
                                                                       device=batch.seq_lengths.device)
            if batch.VERSION == 'JC':
                achunk = RealBatch(seq_lengths=seq_lengths_new, imu_acc=batch.imu_acc[:, sf:ef],
                                   imu_gyro=batch.imu_gyro[:, sf:ef], imu_ori=batch.imu_ori[:, sf:ef],
                                   vicon_jc=batch.vicon_jc[:, sf:ef], vicon_ori=batch.vicon_ori[:, sf:ef],
                                   vicon_contact=batch.vicon_contact,VERSION=batch.VERSION)
            else:
                achunk = RealBatch(seq_lengths=seq_lengths_new, imu_acc=batch.imu_acc[:, sf:ef],
                                   imu_gyro=batch.imu_gyro[:, sf:ef], imu_ori=batch.imu_ori[:, sf:ef],
                                   vicon_ori=batch.vicon_ori[:, sf:ef],relative_phase=batch.relative_phase[:, sf:ef],
                                   relative_joints = batch.relative_joints[:, sf:ef],
                                   pose=batch.pose[:, sf:ef], joint_rotations=batch.joint_rotations[:, sf:ef],
                                   vicon_contact=batch.vicon_contact[:, sf:ef], root=batch.root[:, sf:ef],
                                   bone_offsets=batch.bone_offsets,VERSION=batch.VERSION)
            yield achunk
    else:
        yield batch

def save_to_matlab_struct(data, sub_indx,config, sf, model_dir, version=str, **kwargs):
    """ Save kinematic data to Matlab struct. """

    # Handling pose_data based on version
    if version == 'modified':
        data = data.squeeze().cpu().numpy()
    data_type = kwargs.get('data_type')
    ef = data.shape[0] * int(200 / sf)
    subject = str(sub_indx).zfill(2)
    trial = config.trial

    # IMU data source based on version
    imu_data_path = f'/SonE_{subject}/' + ('fk/' if version == 'modified' else '')
    imu_data = np.load(C.DATA_DIR + imu_data_path + f'{trial}_{subject}_imu.npz')
    data = data[:, ...] * 1000 if data_type != 'Joint_rot' else  data[:, ...]# Change unit from [m] to [mm]
    data = data.cpu().numpy()

    acc_temp = imu_data['acc'][:ef, ...]
    gyro_temp = imu_data['gyro'][:ef, ...]
    acc = acc_temp[::int(200 / sf),...]
    gyro = gyro_temp[::int(200 / sf),...]

    if config.partition == 'unseen_specific':
        """ Reconstruct train/test split for unseen motion"""
        df = int(200 / sf)
        num, div = acc.shape[0], 8  # num: number of frames, 8 fold split
        num = num // df + (1 if num % 2 > 0 else 0)

        # create list with the start frames of the 8 sections
        list_split = [(num // div + (1 if x < num % div else 0)) * x for x in range(div)]
        test_set_unseen = pd.read_csv(C.UNSEEN_MOTION_CSV)
        idx = test_set_unseen.loc[int(subject), config.trial]

        startf, endf = int(list_split[idx] * df), (list_split[idx + 1] * df)

        acc = acc[startf:endf, ...]
        gyro = gyro[startf:endf, ...]

        kwargs['annex'] = kwargs.get('annex', '') + '_um'

    struct = {}

    struct['SF'] = sf  # State sampling frequency

    # Marker Data Mapping based on version
    if version == 'JC':
        # Original mapping
        struct['Marker'] = {}
        struct['Marker']['Head'] = data[:, 0, :]
        struct['Marker']['Left_Shoulder'] = data[:, 1, :]
        struct['Marker']['Right_Shoulder'] = data[:, 2, :]
        struct['Marker']['LHJC'] = data[:, 3, :]
        struct['Marker']['RHJC'] = data[:, 4, :]
        struct['Marker']['LKJC'] = data[:, 5, :]
        struct['Marker']['RKJC'] = data[:, 6, :]
        struct['Marker']['LAJC'] = data[:, 7, :]
        struct['Marker']['RAJC'] = data[:, 8, :]
        struct['Marker']['LTO3'] = data[:, 9, :]
        struct['Marker']['RTO3'] = data[:, 10, :]
    elif version == 'FK':
        if data_type == 'joint_rot':
            # Joint Rotation Data Mapping (only for version 'FK')
            struct['JointsRot'] = {}
            struct['JointsRot']['Hip'] = data[:, 0, :]
            struct['JointsRot']['Head'] = data[:, 7, :]
            struct['JointsRot']['Neck'] = data[:, 4, :]
            struct['JointsRot']['LSHO'] = data[:, 15, :]
            struct['JointsRot']['RSHO'] = data[:, 9, :]
            struct['JointsRot']['LARM'] = data[:, 16, :]
            struct['JointsRot']['LFAM'] = data[:, 17, :]
            struct['JointsRot']['LHND'] = data[:, 18, :]
            struct['JointsRot']['RARM'] = data[:, 10, :]
            struct['JointsRot']['RFAM'] = data[:, 11, :]
            struct['JointsRot']['LHJC'] = data[:, 26, :]
            struct['JointsRot']['RHJC'] = data[:, 21, :]
            struct['JointsRot']['LKJC'] = data[:, 27, :]
            struct['JointsRot']['RKJC'] = data[:, 22, :]
            struct['JointsRot']['LAJC'] = data[:, 28, :]
            struct['JointsRot']['RAJC'] = data[:, 23, :]
            struct['JointsRot']['LTO3'] = data[:, 29, :]
            struct['JointsRot']['RTO3'] = data[:, 24, :]
        else:
            # Modified mapping
            struct['Marker'] = {}
            struct['Marker']['Hip'] = data[:, 0, :]
            struct['Marker']['Head'] = data[:, 7, :]
            struct['Marker']['Neck'] = data[:, 4, :]
            struct['Marker']['LSHO'] = data[:, 15, :]
            struct['Marker']['RSHO'] = data[:, 9, :]
            struct['Marker']['LARM'] = data[:, 16, :]
            struct['Marker']['LFAM'] = data[:, 17, :]
            struct['Marker']['LHND'] = data[:, 18, :]
            struct['Marker']['RARM'] = data[:, 10, :]
            struct['Marker']['RFAM'] = data[:, 11, :]
            struct['Marker']['RHND'] = data[:, 12, :]
            struct['Marker']['LHJC'] = data[:, 26, :]
            struct['Marker']['RHJC'] = data[:, 21, :]
            struct['Marker']['LKJC'] = data[:, 27, :]
            struct['Marker']['RKJC'] = data[:, 22, :]
            struct['Marker']['LAJC'] = data[:, 28, :]
            struct['Marker']['RAJC'] = data[:, 23, :]
            struct['Marker']['LTO3'] = data[:, 29, :]
            struct['Marker']['RTO3'] = data[:, 24, :]

    struct['IMU'] = {}
    struct['IMU']['ankle_L'] = {}
    struct['IMU']['ankle_R'] = {}
    struct['IMU']['arm_L'] = {}
    struct['IMU']['arm_R'] = {}
    struct['IMU']['head'] = {}
    struct['IMU']['trunk'] = {}
    struct['IMU']['ankle_L']['acc'] = acc[:, 0, :]
    struct['IMU']['ankle_L']['gyro'] = gyro[:, 0, :]
    struct['IMU']['ankle_R']['acc'] = acc[:, 1, :]
    struct['IMU']['ankle_R']['gyro'] = gyro[:, 1, :]
    struct['IMU']['arm_L']['acc'] = acc[:, 2, :]
    struct['IMU']['arm_L']['gyro'] = gyro[:, 2, :]
    struct['IMU']['arm_R']['acc'] = acc[:, 3, :]
    struct['IMU']['arm_R']['gyro'] = gyro[:, 3, :]
    struct['IMU']['head']['acc'] = acc[:, 4, :]
    struct['IMU']['head']['gyro'] = gyro[:, 4, :]
    struct['IMU']['trunk']['acc'] = acc[:, 5, :]
    struct['IMU']['trunk']['gyro'] = gyro[:, 5, :]

    path = model_dir + f'/SonE_{subject}/'
    os.makedirs(path, exist_ok=True)

    annex = kwargs.get('annex', '')

    sio.savemat(model_dir + f'/SonE_{subject}/{trial}{annex}.mat', {'VD': struct})

def save_fk_results(batch, model_out_all, sub_indx, config, model_config, model_indx, sf):
    hat_pose = torch.cat(model_out_all['pose_hat'], dim=1).squeeze()
    gt_pose = torch.cat(batch['gt_pose'],dim=0).reshape(-1,len(C.FK_EVAL_JOINTS_FUll), 3)
    gt_root =  torch.cat(batch['gt_root'],dim=0).reshape(-1,1,3)
    hat_pose_rec = hat_pose + gt_root.to(C.DEVICE)
    gt_pose_rec = gt_pose + gt_root.to(C.DEVICE)

    hat_joint_rot = torch.cat(model_out_all['joint_rot_hat'], dim=1).squeeze()
    gt_joint_rot = torch.cat(batch['gt_joint_rot'], dim=0).reshape(1, -1, len(C.FK_EVAL_JOINTS_FUll), 3, 3).squeeze()

    if model_config.predict_root:
        root_pred = get_predicted_root(
            torch.cat([item for item in model_out_all['root_hat'] if item is not None], dim=1), model_config)
        root_pred = root_pred.squeeze().reshape(-1, 1, 3)
        pose_hat_root = pose_hat + root_pred
        save_to_matlab_struct(pose_hat_root, config, sf, model_indx['model_dir'],  data_type = 'pose',annex='_pred_root')

    save_to_matlab_struct(hat_pose_rec, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION, data_type ='pose', annex='_root_rec_hat')
    save_to_matlab_struct(gt_pose_rec, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION, data_type ='pose', annex='_root_rec_gt')
    save_to_matlab_struct(hat_pose, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION,data_type = 'pose', annex='_pose_hat')
    save_to_matlab_struct(gt_pose, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION, data_type = 'pose',annex='_pose_gt')
    save_to_matlab_struct(hat_joint_rot, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION, data_type ='joint_rot', annex='_joint_rot_hat')
    save_to_matlab_struct(gt_joint_rot, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION, data_type ='joint_rot', annex='_joint_rot_gt')

def save_jc_results(batch, model_out_all, sub_indx, config, model_config, model_indx, sf):

    pose_hat = torch.cat(model_out_all['pose_hat'], dim=1).squeeze()
    pose_gt = batch.vicon_jc.squeeze().reshape(-1, len(C.JC_EVAL_JOINTS_Full), 3)
    pose_rec_hat = pose_hat + batch.root.squeeze().reshape(-1, 1, 3).to(C.DEVICE)[:pose_hat.shape[0]]
    pose_rec_gt = batch.pose.reshape(-1,len(C.JC_EVAL_JOINTS_Full), 3)

    # pose_hat = pose_hat.squeeze().reshape(-1, len(C.JC_EVAL_JOINTS_Full), 3)
    # pose_hat = pose_hat[:, [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14]]

    # pose_gt = pose_gt[:, [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14]]
    # root = batch.root.reshape(-1, 1, 3).cpu().numpy()
    # pose_gt += root

    if model_config.predict_root:
        root_pred = get_predicted_root(
            torch.cat([item for item in model_out_all['root_hat'] if item is not None], dim=1), model_config)
        root_pred = root_pred.squeeze().reshape(-1, 1, 3)
        pose_hat_root = pose_hat + root_pred
        save_to_matlab_struct(pose_hat_root, config, sf, model_indx['model_dir'],  data_type = 'pose',annex='_pred_root')

    save_to_matlab_struct(pose_rec_hat, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION,data_type = 'pose', annex='_root_hat')
    save_to_matlab_struct(pose_rec_gt, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION,data_type = 'pose', annex='_root_gt')
    save_to_matlab_struct(pose_hat, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION,data_type = 'pose', annex='_pose_hat')
    save_to_matlab_struct(pose_gt, sub_indx, config, sf, model_indx['model_dir'], version=model_config.VERSION, data_type = 'pose',annex='_pose_gt')

def get_skeleton_connections(config):
    """ Create point connections for skeleton visualization. """
    connections = []
    for i in range(1, len(C.FK_PARENTS_Full)):
            connections.append((i, C.FK_PARENTS_Full[i]))

    return connections

def get_predicted_root(root_diff, config):
    """ Integrate root difference to get root position. """

    deltas = torch.clone(root_diff)
    add_velocity = getattr(config, 'add_gait_vel', False)

    vel = torch.tensor([3.5 / 3.6 / 50]).to(C.DEVICE)

    for i in range(1, deltas.shape[1]):
        if add_velocity:
            deltas[:, i, 1] += vel
        deltas[:, i, :] = deltas[:, i, :] + deltas[:, i - 1, :]

    return deltas

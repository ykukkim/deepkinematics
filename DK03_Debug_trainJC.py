import os
import glob
import time

import torch
import numpy as np
import torch.optim as optim

from itertools import product
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau

# Custom Library
from DK00_Utils.DK00_UT00_config import CONSTANTS as C
from DK00_Utils.DK00_UT00_config import Configuration
from DK00_Utils.DK00_UT00_utils import set_random_seed, lr_lambda
from DK00_Utils.DK00_UT01_dataset import RealDataset
from DK00_Utils.DK00_UT01_datasetProcess import RealBatch
from DK00_Utils.DK00_UT01_tranforms import CorrectIMUOrientation, ExtractInitialIMUOffset, ExtractWindow, IMUNormaliser, TransformOrientation, ToTensor, \
    NormalizeViconJointCenter, NormalizeHeight, get_end_to_end_preprocess_fn
from DK00_Utils.DK00_UT02_models import create_model
from DK00_Utils.DK00_UT03_metrics import MetricsEngine
from DK00_Utils import DK00_UT04_logginghelpers as H
from DK00_Utils.DK00_UT05_helpersEval import load_model_weights

def main(config=None):

    set_random_seed(0, deterministic=True)  # Example: setting seed to 42

    rng_extractor = np.random.RandomState(0)
    """ Data and Dataset Process """
    train_transform = transforms.Compose([CorrectIMUOrientation(),
                                          ExtractInitialIMUOffset(config),
                                          ExtractWindow(config, config.window_size, rng_extractor, mode='random_unseen',
                                                        train=True),
                                          TransformOrientation(config),
                                          IMUNormaliser(config),
                                          NormalizeHeight(config),
                                          NormalizeViconJointCenter(config),
                                          ToTensor()])

    unseen_subjects_transform = transforms.Compose([CorrectIMUOrientation(),
                                                    ExtractInitialIMUOffset(config),
                                                    ExtractWindow(config, config.window_size, rng_extractor,
                                                                  mode='random',
                                                                  train=False),
                                                    TransformOrientation(config),
                                                    IMUNormaliser(config),
                                                    NormalizeHeight(config),
                                                    NormalizeViconJointCenter(config),
                                                    ToTensor()])

    unseen_motion_transform = transforms.Compose([CorrectIMUOrientation(),
                                                  ExtractInitialIMUOffset(config),
                                                  ExtractWindow(config, config.window_size, rng_extractor,
                                                                mode='random_unseen',
                                                                train=False),
                                                  TransformOrientation(config),
                                                  IMUNormaliser(config),
                                                  NormalizeHeight(config),
                                                  NormalizeViconJointCenter(config),
                                                  ToTensor()])

    train_data = RealDataset(transform=train_transform,
                             sampling_freq=C.SAMPLING_FREQUENCY,
                             version=config.VERSION,
                             train=True)

    val_unseen_subjects_easy = RealDataset(test_set='easy', transform=unseen_subjects_transform,
                                           sampling_freq=C.SAMPLING_FREQUENCY,
                                           version=config.VERSION,
                                           train=False)

    val_unseen_subjects_hard = RealDataset(test_set='hard', transform=unseen_subjects_transform,
                                           sampling_freq=C.SAMPLING_FREQUENCY,
                                           version=config.VERSION,
                                           train=False)

    val_unseen_motion = RealDataset(test_set='unseen_motion', transform=unseen_motion_transform,
                                    sampling_freq=C.SAMPLING_FREQUENCY,
                                    version=config.VERSION,
                                    train=False)

    train_loader = DataLoader(train_data, batch_size=config.bs_train, shuffle=True,
                              collate_fn=RealBatch.from_sample_list)

    unseen_subjects_easy_loader = DataLoader(val_unseen_subjects_easy, batch_size=config.bs_eval, shuffle=False,
                                             collate_fn=RealBatch.from_sample_list)

    unseen_subjects_hard_loader = DataLoader(val_unseen_subjects_hard, batch_size=config.bs_eval, shuffle=False,
                                             collate_fn=RealBatch.from_sample_list)

    unseen_motion_loader = DataLoader(val_unseen_motion, batch_size=config.bs_eval, shuffle=False,
                                      collate_fn=RealBatch.from_sample_list)


    """ Model, metrics, preprocessing, and WandB setup """
    components = []
    experiment_id = config.experimentid
    experiment_name = f"{config.VERSION}"

    # Add components based on the configuration flags
    if config.use_quats:
        components.append("-quat")
    if config.use_rot_vec:
        components.append("-rot_vec")
    if config.use_acc_gyro:
        components.append("-acc_gyro")
    if config.use_orientation:
        components.append("-rot_mat")

    # Check if no components have been added and raise an error if true
    if not components:
        raise ValueError(
            "At least one configuration flag must be set (use_quats, use_rot_vec, use_acc_gyro, use_orientation).")

    experiment_name += ''.join(components)

    # Model, metrics, and processing initial
    models = create_model(config)
    models_on_device = tuple(model.to(C.DEVICE) for model in models)

    if len(models_on_device) == 1:
        model = models_on_device[0]
        vrn_net = None
    elif len(models_on_device) == 2:
        vrn_net, model = models_on_device

    experiment_name = f"{model.model_name()}-{experiment_name}"

    # Retrieve the existing directory for the provided experiment_id
    directory_path = os.path.join(C.save_model_DIR, experiment_id, experiment_name)
    if os.path.exists(directory_path):
        model_dir = H.get_model_dir(C.save_model_DIR, experiment_id, experiment_name)
    else:
        model_dir = H.create_model_dir(C.save_model_DIR, experiment_id, experiment_name)

    # Save code as zip into the model directory.
    code_files = glob.glob('./**/*.py', recursive=True)
    H.zip_files(code_files, os.path.join(model_dir, 'code.zip'))
    config.to_json(os.path.join(model_dir, 'config.json'))
    checkpoint_file = os.path.join(model_dir, 'test_model.pth')
    print('Saving checkpoints to {}'.format(checkpoint_file))

    model_params = [param for model in models_on_device for param in model.parameters()]
    wandb_run = []

    """ Optimizer """
    # Initialize the optimizer and scheduler
    if config.optimizer == "adam":
        optimizer = optim.Adam(model_params, lr=config.lr, betas=(0.85, 0.95), weight_decay=1e-4)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model_params, lr=config.lr)
    else:
        raise Exception("Optimization not found.")

    if config.scheduler == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.7)
    elif config.scheduler == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    else:
        raise Exception("Scheduler not found.")

    """ Training Loop """
    me = MetricsEngine(config)
    preprocess_fn = get_end_to_end_preprocess_fn(config)

    best_valid_loss_pose = float('inf')
    stop_training = False
    val_counter = 0

    for epoch in range(config.n_epochs):
        for i, abatch in enumerate(train_loader):
            start_time = time.time()

            optimizer.zero_grad()

            batch_gpu = abatch.to_gpu()

            batch_gpu = preprocess_fn(batch_gpu)

            # Step 1: Pseudo velocity estimation
            vrn_output = vrn_net(batch_gpu) if vrn_net else None

            # Step 2: Pose estimation with global context
            model_out = model(batch_gpu, vrn_output) if config.m_type in ('att', 'vrnatt') else model(batch_gpu)

            total_loss, loss_values = model.backward(batch_gpu, model_out, config.m_type)

            if vrn_output is not None:
                loss_values['velocity_hat'] = vrn_net.backward(batch_gpu, vrn_output, config.m_type)
                value = loss_values.pop('total_loss') + loss_values['velocity_hat']
                loss_values['total_loss'] = value

            optimizer.step()
            # Only call scheduler.step() if it's not ReduceLROnPlateau
            if config.scheduler in ['lambda', 'step']:
                scheduler.step()

            """Checking during plotting"""
            # pose_hat = model_out['pose_hat'].cpu().detach().numpy()

            # pose_gt = batch_gpu.pose.cpu().detach().numpy()
            # pose_gt = pose_gt.reshape(4, 200, 31, 3)
            # pose_gt = pose_gt[:,:, C.FK_EVAL_JOINTS_FUll,:] if config.predict_arms else pose_gt[:,:,C.FK_EVAL_JOINTS_NoArms,:]

            # plot_pose_angles(pose_hat, pose_gt, joint_idx=18)
            """-----------------------------------"""

            if i == 0:
                # Save every epoch
                elapsed_time = time.time() - start_time
                loss_string = ' '.join(['{}: {:.6f}'.format(k, loss_values[k]) for k in loss_values])
                print('[TRAIN Epoch {:0>3d}] | iteration {:0>5d}  {} elapsed: {:.3f} secs'.format(
                    epoch+1, i, loss_string, elapsed_time))

            if (epoch + 1) % config.eval_every == 0 and i == 0:
                model.eval()
                if vrn_net is not None:
                    vrn_net.eval()

                H.evaluate_and_log(dataset_name='easy', loader=unseen_subjects_easy_loader, model=model, vrn=vrn_net,
                                   preprocess_fn=preprocess_fn,metrics_evaluator=me,
                                   wandb_run=wandb_run, experiment_id = experiment_id,
                                   epoch=epoch+1, VERSION=config.VERSION, modeltype=config.m_type)
                H.evaluate_and_log(dataset_name='hard', loader=unseen_subjects_hard_loader, model=model, vrn=vrn_net,
                                   preprocess_fn=preprocess_fn, metrics_evaluator=me,
                                   wandb_run=wandb_run, experiment_id=experiment_id,
                                   epoch=epoch+1,  VERSION=config.VERSION, modeltype=config.m_type)
                unseen_motion_losses, unseen_motion_metrics = H.evaluate_and_log(dataset_name='unseen', loader=unseen_motion_loader, model=model, vrn=vrn_net,
                                   preprocess_fn=preprocess_fn,metrics_evaluator=me,
                                   wandb_run=wandb_run, experiment_id = experiment_id,
                                   epoch=epoch+1, VERSION=config.VERSION, modeltype=config.m_type)

                current_eval_metric_pose = unseen_motion_metrics['MPJPE [mm]']

                if (best_valid_loss_pose - current_eval_metric_pose > 3):
                    best_valid_loss_pose = min(best_valid_loss_pose, current_eval_metric_pose)

                    print(' *** New best model ***')
                    val_counter = 0
                    checkpoint_data = {
                        'iteration': i,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(checkpoint_data, checkpoint_file)

                else:
                    val_counter += 1
                if val_counter == 10:
                    stop_training = True
                    wandb_run.update_epoch(epoch)
                    break

                if vrn_net:
                    vrn_net.train()
                model.train()

        if stop_training:
            print(f'Training has stopped after {i} iterations.')
            break

    """ Main code for final evaluation """
    load_model_weights(checkpoint_file, model)

    # Final evaluation on Easy, Hard, and Unseen test sets
    H.final_evaluation_and_log(dataset_name='EASY', loader=unseen_subjects_easy_loader, model=model, vrn=vrn_net,
                               preprocess_fn=preprocess_fn, metrics_evaluator=me, wandb_run=wandb_run,
                               experiment_id=experiment_id, VERSION=config.VERSION,modeltype=config.m_type)
    H.final_evaluation_and_log(dataset_name='HARD', loader=unseen_subjects_hard_loader, model=model, vrn=vrn_net,
                               preprocess_fn=preprocess_fn, metrics_evaluator=me, wandb_run=wandb_run,
                               experiment_id=experiment_id, VERSION=config.VERSION,modeltype=config.m_type)
    H.final_evaluation_and_log(dataset_name='UNSEEN', loader=unseen_motion_loader, model=model, vrn=vrn_net,
                               preprocess_fn=preprocess_fn, metrics_evaluator=me, wandb_run=wandb_run,
                               experiment_id=experiment_id, VERSION=config.VERSION,modeltype=config.m_type)

if __name__ == '__main__':

    # Parse the initial configuration outside the wandb agent call
    initial_config = Configuration.parse_cmd(version='JC', optimizer='adam', scheduler='step', m_type='rnndct',
                                             experimentid='JC')

    sweep_configuration = initial_config.get_sweep_configuration()
    param_names = list(sweep_configuration['parameters'].keys())
    param_values = [sweep_configuration['parameters'][name]['values'] for name in param_names]

    combinations = list(product(*param_values))

    # Iterate over all combinations and update the configuration
    for combo in combinations:
        combo_dict = dict(zip(param_names, combo))
        initial_config.update(combo_dict)
        print(initial_config.__dict__)
        main(initial_config)

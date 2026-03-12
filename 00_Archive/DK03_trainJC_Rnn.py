import os
import glob
import time

import wandb
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup

# Custom Library
from DK00_Utils.DK00_UT00_config import CONSTANTS as C
from DK00_Utils.DK00_UT00_config import Configuration
from DK00_Utils.DK00_UT00_utils import set_random_seed, lr_lambda
from DK00_Utils.DK00_UT01_dataset import RealDataset
from DK00_Utils.DK00_UT01_datasetProcess import RealBatch
from DK00_Utils.DK00_UT01_tranforms import CorrectIMUOrientation, ExtractInitialIMUOffset, ExtractWindow, \
    TransformOrientation, ToTensor, \
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
                                          NormalizeHeight(config),
                                          NormalizeViconJointCenter(config),
                                          ToTensor()])

    unseen_subjects_transform = transforms.Compose([CorrectIMUOrientation(),
                                                    ExtractInitialIMUOffset(config),
                                                    ExtractWindow(config, config.window_size, rng_extractor,
                                                                  mode='random',
                                                                  train=False),
                                                    TransformOrientation(config),
                                                    NormalizeHeight(config),
                                                    NormalizeViconJointCenter(config),
                                                    ToTensor()])

    unseen_motion_transform = transforms.Compose([CorrectIMUOrientation(),
                                                  ExtractInitialIMUOffset(config),
                                                  ExtractWindow(config, config.window_size, rng_extractor,
                                                                mode='random_unseen',
                                                                train=False),
                                                  TransformOrientation(config),
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

    val_unseen_motion = RealDataset(test_set='unseen', transform=unseen_motion_transform,
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
    if config.use_acc_gyro:
        components.append("-acc_gyro")
    if config.use_orientation:
        components.append("-rot_mat")

    # Check if no components have been added and raise an error if true
    if not components:
        raise ValueError(
            "At least one configuration flag must be set (use_quats, use_acc_gyro, use_orientation).")

    experiment_name += ''.join(components)

    # Initialize WandB with the initial experiment_name, and update config settings
    wandb_run = H.WandB(project=f'{config.experimentid}-{experiment_name}', experiment= f'{config.experimentid}-{experiment_name}', mode='online',
                        dir=C.save_wandb_DIR, VERSION=config.VERSION)

    config.update({k: v for k, v in wandb.config.items()})

    # Model, metrics, and processing initial
    models = create_model(config)  # Can be one or more models in a tuple
    models_on_device = tuple(model.to(C.DEVICE) for model in models)

    if len(models_on_device) == 1:
        model = models_on_device[0]
        vrn_net = None

    elif len(models_on_device) == 2:
        vrn_net, model = models_on_device

    # Retrieve the existing directory for the provided experiment_id
    experiment_name = f"{model.model_name()}-{experiment_name}"
    cleaned_experiment_name = experiment_name.replace("[", "").replace("]", "")
    directory_path = os.path.join(C.save_model_DIR, experiment_id, cleaned_experiment_name)
    if os.path.exists(directory_path):
        model_dir = H.get_model_dir(C.save_model_DIR, experiment_id, cleaned_experiment_name)
    else:
        model_dir = H.create_model_dir(C.save_model_DIR, experiment_id, cleaned_experiment_name  )

    # Save code as zip into the model directory.
    code_files = glob.glob('./**/*.py', recursive=True)
    H.zip_files(code_files, os.path.join(model_dir, 'code.zip'))
    config.to_json(os.path.join(model_dir, 'config.json'))
    checkpoint_file = os.path.join(model_dir, 'test_model.pth')
    print('Saving checkpoints to {}'.format(checkpoint_file))

    model_params = [param for model in models_on_device for param in model.parameters()]
    n_model_param = H.count_parameters(model_params)

    wandb_run.run.name = experiment_name
    wandb_run.update_config(config, n_model_param)
    wandb_run.save_as_graph(config)
    wandb_run.run.save()

    """ Optimizer & Scheduler """
    # Initialize the optimizer and scheduler
    if config.optimizer == "adam":
        optimizer = optim.Adam(model_params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(model_params, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model_params, lr=config.lr)
    else:
        raise Exception("Optimization not found.")

    if config.scheduler == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif config.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.7)
    elif config.scheduler == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.01,
                                      threshold_mode='rel', cooldown=2, min_lr=1e-6)
    elif config.scheduler == 'cosine':
        total_steps = config.n_epochs * len(train_data)
        warmup_steps = int(0.15*total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        raise Exception("Scheduler not found.")

    """ Training Loop """
    me = MetricsEngine(config)
    preprocess_fn = get_end_to_end_preprocess_fn(config)

    best_valid_loss_pose = float('inf')

    val_counter = 0
    stop_training = False

    for epoch in range(config.n_epochs):
        for i, abatch in enumerate(train_loader):
            start_time = time.time()

            optimizer.zero_grad()

            batch_gpu = abatch.to_gpu()

            batch_gpu = preprocess_fn(batch_gpu)

            # Step 1: Pseudo velocity estimation
            vrn_output = vrn_net(batch_gpu) if vrn_net is not None else None

            # Step 2: Pose estimation with global context
            model_out = model(batch_gpu, vrn_output) if config.m_type in ('rnn', 'vrnrnn', 'att', 'vrnatt') else model(batch_gpu)

            total_loss, loss_values = model.backward(batch_gpu, model_out, config.m_type)

            if vrn_output is not None:
                loss_values['velocity_hat'] = vrn_net.backward(batch_gpu, vrn_output, config.m_type)

            optimizer.step()
            scheduler.step(total_loss) if config.scheduler == 'reduce' else scheduler.step()

            if i == 0:
                elapsed_time = time.time() - start_time
                loss_string = ' '.join([f'{k}: {loss_values[k]:.6f}' for k in loss_values])
                print('[TRAIN Epoch {:0>3d}] | iteration {:0>5d}  {} elapsed: {:.3f} secs'.format(
                    epoch + 1, i, loss_string, elapsed_time))
                wandb_run.log_to_wandb(loss_values, 'TRAIN', 'loss')

            if (epoch + 1) % config.eval_every == 0 and i == 0:
                model.eval()
                if vrn_net is not None:
                    vrn_net.eval()

                easy_motion_losses, easy_motion_metrics = H.evaluate_and_log(dataset_name='easy',
                                                                             loader=unseen_subjects_easy_loader,
                                                                             model=model, vrn=vrn_net,
                                                                             preprocess_fn=preprocess_fn,
                                                                             metrics_evaluator=me,
                                                                             wandb_run=wandb_run,
                                                                             experiment_id=experiment_id,
                                                                             epoch=epoch + 1,
                                                                             VERSION=config.VERSION,
                                                                             modeltype=config.m_type)

                hard_motion_losses, hard_motion_metrics = H.evaluate_and_log(dataset_name='hard',
                                                                             loader=unseen_subjects_hard_loader,
                                                                             model=model, vrn=vrn_net,
                                                                             preprocess_fn=preprocess_fn,
                                                                             metrics_evaluator=me,
                                                                             wandb_run=wandb_run,
                                                                             experiment_id=experiment_id,
                                                                             epoch=epoch + 1,
                                                                             VERSION=config.VERSION,
                                                                             modeltype=config.m_type)

                unseen_motion_losses, unseen_motion_metrics = H.evaluate_and_log(dataset_name='unseen',
                                                                                 loader=unseen_motion_loader,
                                                                                 model=model, vrn=vrn_net,
                                                                                 preprocess_fn=preprocess_fn,
                                                                                 metrics_evaluator=me,
                                                                                 wandb_run=wandb_run,
                                                                                 experiment_id=experiment_id,
                                                                                 epoch=epoch + 1,
                                                                                 VERSION=config.VERSION,
                                                                                 modeltype=config.m_type)

                current_eval_metric_pose = hard_motion_metrics['MPJPE [mm]']

                if (best_valid_loss_pose - current_eval_metric_pose > 3) :

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

                if val_counter == 5:
                    stop_training = True
                    wandb_run.update_epoch(epoch + 1)
                    break

                if vrn_net:
                    vrn_net.train()
                model.train()

        if stop_training:
            print(f'Training has stopped after {epoch + 1} Epochs.')
            break

    load_model_weights(checkpoint_file, model)

    # Final evaluation on Easy, Hard, and Unseen test sets
    H.final_evaluation_and_log(dataset_name='EASY', loader=unseen_subjects_easy_loader, model=model, vrn=vrn_net,
                               preprocess_fn=preprocess_fn, metrics_evaluator=me, wandb_run=wandb_run,
                               experiment_id=experiment_id, VERSION=config.VERSION, modeltype=config.m_type)
    H.final_evaluation_and_log(dataset_name='HARD', loader=unseen_subjects_hard_loader, model=model, vrn=vrn_net,
                               preprocess_fn=preprocess_fn, metrics_evaluator=me, wandb_run=wandb_run,
                               experiment_id=experiment_id, VERSION=config.VERSION, modeltype=config.m_type)
    H.final_evaluation_and_log(dataset_name='UNSEEN', loader=unseen_motion_loader, model=model, vrn=vrn_net,
                               preprocess_fn=preprocess_fn, metrics_evaluator=me, wandb_run=wandb_run,
                               experiment_id=experiment_id, VERSION=config.VERSION, modeltype=config.m_type)


if __name__ == '__main__':
    # Parse the initial configuration outside the wandb agent call
    initial_config = Configuration.parse_cmd(version='JC', optimizer='adam', scheduler='step', m_type='rnn',
                                             experimentid='JC')

    # Define or get the sweep configuration from the initial configuration
    sweep_configuration = initial_config.get_sweep_configuration()
    os.environ['WANDB_DIR'] = os.path.join(C.save_wandb_DIR, initial_config.experimentid)

    # Initialize the sweep - name the prject with version,m_type and exeperiemnt_id
    sweep_id = wandb.sweep(sweep=sweep_configuration,
                           project=f'{initial_config.experimentid}')

    wandb.agent(sweep_id, lambda: main(initial_config))

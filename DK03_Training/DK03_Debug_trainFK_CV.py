"""Debug entry point for cross-validation training and aggregation."""

# --- path bootstrap: allow running from this subfolder (adds repo root for DK00_Utils) ---
import os as _dk_os, sys as _dk_sys; _dk_sys.path.insert(0, _dk_os.path.dirname(_dk_os.path.dirname(_dk_os.path.abspath(__file__))))

import os
import glob
import time
import pandas as pd


import torch
import numpy as np
import torch.optim as optim

from itertools import product
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup

# Custom Library
from DK00_Utils.DK00_UT00_config import CONSTANTS as C
from DK00_Utils.DK00_UT00_config import Configuration, create_stratified_folds
from DK00_Utils.DK00_UT00_utils import set_random_seed, lr_lambda
from DK00_Utils.DK00_UT01_dataset import RealDataset
from DK00_Utils.DK00_UT01_datasetProcess import RealBatch
from DK00_Utils.DK00_UT01_tranforms import ExtractWindow,ToTensor, get_end_to_end_preprocess_fn
from DK00_Utils.DK00_UT02_models import create_model
from DK00_Utils.DK00_UT03_metrics import MetricsEngine
from DK00_Utils import DK00_UT04_logginghelpers as H
from DK00_Utils.DK00_UT05_helpersEval import load_model_weights

def clean_metric_dict(m):
    return {k: float(v) for k, v in m.items()}

def print_summary(name, summary):
    print(f"\n{name} CV RESULTS")
    for k, v in summary.items():
        print(f"{k}: {v['mean']:.2f} ± {v['std']:.2f}")

def aggregate_metrics(metric_list):
    keys = metric_list[0].keys()
    summary = {}

    for k in keys:
        values = [m[k] for m in metric_list]
        summary[k] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }

    return summary

def flatten_cv_results(cv_results_per_fold):
    rows = []

    for fold_data in cv_results_per_fold:
        fold = fold_data['fold']

        for split in ['easy', 'hard', 'unseen']:
            metrics = fold_data[split]

            row = {
                'Fold': fold,
                'Split': split
            }

            for k, v in metrics.items():
                row[k] = v

            rows.append(row)

    return pd.DataFrame(rows)

def combine_summaries(easy, hard, unseen):
    metrics = easy.keys()
    rows = []

    for m in metrics:
        rows.append({
            'Metric': m,
            'Easy Mean': easy[m]['mean'],
            'Easy Std': easy[m]['std'],
            'Hard Mean': hard[m]['mean'],
            'Hard Std': hard[m]['std'],
            'Unseen Mean': unseen[m]['mean'],
            'Unseen Std': unseen[m]['std'],
        })

    return pd.DataFrame(rows)

def add_pretty_columns(df):
    df['Easy'] = df.apply(lambda x: f"{x['Easy Mean']:.2f} ± {x['Easy Std']:.2f}", axis=1)
    df['Hard'] = df.apply(lambda x: f"{x['Hard Mean']:.2f} ± {x['Hard Std']:.2f}", axis=1)
    df['Unseen'] = df.apply(lambda x: f"{x['Unseen Mean']:.2f} ± {x['Unseen Std']:.2f}", axis=1)
    return df

def main(config=None):
    """Run each debug CV fold and write an aggregate results workbook."""

    set_random_seed(0, deterministic=True)
    rng_extractor = np.random.RandomState(0)

    """ Data and Dataset Process """
    train_transform = transforms.Compose([ExtractWindow(config, config.window_size, rng_extractor, 'random_unseen',
                                                        train=True),
                                          ToTensor()])

    unseen_subjects_transform = transforms.Compose([ExtractWindow(config, config.window_size, rng_extractor, 'random',
                                                                  train=False),
                                                    ToTensor()])

    unseen_motion_transform = transforms.Compose([ExtractWindow(config, config.window_size, rng_extractor,
                                                                'random_unseen',
                                                                train=False),
                                                  ToTensor()])

    if config.use_cross_validation:
        cv_folds = create_stratified_folds(config.cv_folds)
        cv_results = {
            'easy': [],
            'hard': [],
            'unseen': []
        }

        cv_results_per_fold = []

    for fold_idx, (train_subj, val_subj) in enumerate(cv_folds):
        print(f"Fold {fold_idx + 1}")

        train_data = RealDataset(subjects = train_subj, transform=train_transform, version=config.VERSION, train=True)

        val_unseen_subjects_easy = RealDataset(subjects = val_subj, test_set='easy', transform=unseen_subjects_transform, version=config.VERSION,
                                               train=False)

        val_unseen_subjects_hard = RealDataset(subjects = val_subj, test_set='hard', transform=unseen_subjects_transform, version=config.VERSION,
                                               train=False)

        val_unseen_motion = RealDataset(subjects = train_subj, test_set='unseen', transform=unseen_motion_transform, version=config.VERSION,
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

        # Model, metrics, and processing initial
        models = create_model(config)
        models_on_device = tuple(model.to(C.DEVICE) for model in models)

        if len(models_on_device) == 1:
            model = models_on_device[0]
            vrn_net = None
        elif len(models_on_device) == 2:
            vrn_net, model = models_on_device

        experiment_name = f"{model.model_name()}-{experiment_name}-fold{fold_idx}"

        # Retrieve the existing directory for the provided experiment_id
        cleaned_experiment_name = experiment_name.replace("[", "").replace("]", "")
        directory_path = os.path.join(C.save_model_DIR, experiment_id, cleaned_experiment_name)
        if os.path.exists(directory_path):
            model_dir = H.get_model_dir(C.save_model_DIR, experiment_id, cleaned_experiment_name)
        else:
            model_dir = H.create_model_dir(C.save_model_DIR, experiment_id, cleaned_experiment_name)

        # Save code as zip into the model directory.
        code_files = glob.glob('./**/*.py', recursive=True)
        H.zip_files(code_files, os.path.join(model_dir, 'code.zip'))
        config.to_json(os.path.join(model_dir, 'config.json'))
        checkpoint_file = os.path.join(model_dir, f"test_model-{fold_idx}.pth")
        print('Saving checkpoints to {}'.format(checkpoint_file))

        model_params = [param for model in models_on_device for param in model.parameters()]
        wandb_run = []

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
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.01,
                                          threshold_mode='rel', cooldown=5, min_lr=1e-6)
        elif config.scheduler == 'cosine':
            # The scheduler is stepped once per batch, not once per sample.
            total_steps = config.n_epochs * len(train_loader)
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
        best_valid_loss_joint_rot = float('inf')

        val_counter = 0
        stop_training = False

        for epoch in range(config.n_epochs):
            for i, abatch in enumerate(train_loader):
                start_time = time.time()

                optimizer.zero_grad()

                batch_gpu = abatch.to_gpu()

                batch_gpu = preprocess_fn(batch_gpu)

                # Step 1: Pseudo velocity estimation
                vrn_output = vrn_net(batch_gpu) if vrn_net else None

                # Step 2: Pose estimation with global context
                model_out = model(batch_gpu, vrn_output) if config.m_type in ('rnn','vrnrnn','att', 'vrnatt','diff','vrndiff') else model(batch_gpu)

                total_loss, loss_values, loss_dict = model.backward(batch_gpu, model_out, config.m_type)

                if vrn_output is not None:
                    loss_values['velocity_hat'] = vrn_net.backward(batch_gpu, vrn_output, config.m_type)
                    value = loss_values.pop('total_loss', None)
                    if value is not None: # Add the key back to the dictionary at the end
                        loss_values['total_loss'] = value

                optimizer.step()
                if config.scheduler == 'reduce':
                    scheduler.step(total_loss.detach().item())
                else:
                    scheduler.step()
                loss_values['total_loss'] = total_loss.detach().cpu().item()

                """Checking during plotting"""
                # joint_rot_plot = model_out['joint_rot_hat'].cpu().detach().numpy()
                # pose_hat = model_out['pose_hat'].cpu().detach().numpy()

                # joint_rot_gt = batch_gpu.joint_rotations.cpu().detach().numpy()
                # joint_rot_gt = joint_rot_gt.reshape(4, 200, 31, 3, 3)
                # joint_rot_gt = joint_rot_gt[:,:, C.FK_EVAL_JOINTS_FUll,:] if config.predict_arms else joint_rot_gt[:,:,C.FK_EVAL_JOINTS_NoArms,:]

                # pose_gt = batch_gpu.pose.cpu().detach().numpy()
                # pose_gt = pose_gt.reshape(4, 200, 31, 3)
                # pose_gt = pose_gt[:,:, C.FK_EVAL_JOINTS_FUll,:] if config.predict_arms else pose_gt[:,:,C.FK_EVAL_JOINTS_NoArms,:]

                # euler_angles_hat = rotation_matrices_to_euler_angles(joint_rot_plot)
                # euler_angles_gt = rotation_matrices_to_euler_angles(joint_rot_gt)

                # plot_euler_angles(euler_angles_hat, euler_angles_gt, joint_idx=18)
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
                    current_eval_metric_joint_rot = unseen_motion_metrics['MPJAE [deg]']

                    if (best_valid_loss_pose - current_eval_metric_pose > 3 or
                            best_valid_loss_joint_rot - current_eval_metric_joint_rot > 1):

                        best_valid_loss_pose = min(best_valid_loss_pose, current_eval_metric_pose)
                        best_valid_loss_joint_rot = min(best_valid_loss_joint_rot, current_eval_metric_joint_rot)

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
                        break

                    if vrn_net:
                        vrn_net.train()
                    model.train()

            if stop_training:
                print(f'Training has stopped after {epoch+1} Epochs.')
                break

        """ Main code for final evaluation """
        load_model_weights(checkpoint_file, model)

        # Final evaluation on Easy, Hard, and Unseen test sets
        metric_easy = H.final_evaluation_and_log(dataset_name='EASY', loader=unseen_subjects_easy_loader, model=model, vrn=vrn_net,
                                   preprocess_fn=preprocess_fn, metrics_evaluator=me, wandb_run=wandb_run,
                                   experiment_id=experiment_id, VERSION=config.VERSION,modeltype=config.m_type)

        metric_hard = H.final_evaluation_and_log(dataset_name='HARD', loader=unseen_subjects_hard_loader, model=model, vrn=vrn_net,
                                   preprocess_fn=preprocess_fn, metrics_evaluator=me, wandb_run=wandb_run,
                                   experiment_id=experiment_id, VERSION=config.VERSION,modeltype=config.m_type)

        metric_unseen = H.final_evaluation_and_log(dataset_name='UNSEEN', loader=unseen_motion_loader, model=model, vrn=vrn_net,
                                   preprocess_fn=preprocess_fn, metrics_evaluator=me, wandb_run=wandb_run,
                                   experiment_id=experiment_id, VERSION=config.VERSION,modeltype=config.m_type)

        metric_easy_clean = clean_metric_dict(metric_easy)
        metric_hard_clean = clean_metric_dict(metric_hard)
        metric_unseen_clean = clean_metric_dict(metric_unseen)

        # For aggregation
        cv_results['easy'].append(metric_easy_clean)
        cv_results['hard'].append(metric_hard_clean)
        cv_results['unseen'].append(metric_unseen_clean)

        # Full traceability (per fold)
        cv_results_per_fold.append({
            'fold': fold_idx,
            'train_subjects': train_subj,
            'val_subjects': val_subj,
            'easy': metric_easy_clean,
            'hard': metric_hard_clean,
            'unseen': metric_unseen_clean
        })

    easy_summary = aggregate_metrics(cv_results['easy'])
    hard_summary = aggregate_metrics(cv_results['hard'])
    unseen_summary = aggregate_metrics(cv_results['unseen'])

    summary_df = combine_summaries(easy_summary, hard_summary, unseen_summary)
    summary_df = add_pretty_columns(summary_df)

    folds_df = flatten_cv_results(cv_results_per_fold)

    save_path = os.path.join(directory_path, 'cv_results.xlsx')

    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        folds_df.to_excel(writer, sheet_name='PerFold', index=False)

if __name__ == '__main__':

    # Parse the initial configuration outside the wandb agent call
    initial_config = Configuration.parse_cmd(version='FK', optimizer='adam', m_positional_encoding_type = 'sinusodial', scheduler='reduce', m_type='vrnrnn',
                                             experimentid='FK_test_rnn_inputconfigure')

    sweep_configuration = initial_config.get_sweep_configuration()
    param_names = list(sweep_configuration['parameters'].keys())
    param_values = [sweep_configuration['parameters'][name]['values'] for name in param_names]

    combinations = list(product(*param_values))

    # Iterate over all combinations and update the configuration
    for combo in combinations:
        combo_dict = dict(zip(param_names, combo))

        # Update initial_config with the current combination
        initial_config.update(combo_dict)

        # Print the updated configuration for debugging
        print(initial_config.__dict__)

        # Call the main function with the updated configuration
        main(initial_config)

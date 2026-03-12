import os
import torch
import numpy as np
import argparse
import matplotlib
matplotlib.use('TkAgg')

from tabulate import tabulate
from collections import defaultdict

from aitviewer.renderables.spheres import Spheres # DCT
from aitviewer.renderables.skeletons import Skeletons # FK
from aitviewer.renderables.lines import Lines # JC
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer

from DK00_Utils.DK00_UT00_config import CONSTANTS as C
from DK00_Utils import DK00_UT04_logginghelpers as H
from DK00_Utils.DK00_UT03_metrics import MetricsEngine
from DK00_Utils.DK00_UT05_helpersEval import get_model_configs_and_checkpoints, load_model, load_eval_data, \
    save_fk_results, save_jc_results, window_generator,\
    get_skeleton_connections, get_predicted_root

def main(config):

    model_data_list = get_model_configs_and_checkpoints(config)

    for model_indx, model in enumerate(model_data_list, start=1):
        print('Counting Number of Models: {}/{} '.format(model_indx, len(model_data_list)))
        torch.cuda.empty_cache()
        net, vrn_net, model_config, _, preprocess_fn = load_model(model)
        windowsize = model_config.window_size

        for sub_indx in config.subject:

            test_loader = load_eval_data(config, sub_indx, model_config)

            if vrn_net is not None:
                vrn_net.eval()

            net.eval()

            me = MetricsEngine(model_config)

            with torch.no_grad():
                metric_vals_ind = []
                for i, batch in enumerate(test_loader):

                    batch = preprocess_fn(batch)

                    out_all_model = defaultdict(list)
                    out_all_gt  = defaultdict(list)

                    for c, achunk in enumerate(window_generator(batch, window_size=windowsize)):

                        batch_gpu = achunk.to_gpu()

                        # Step 1: Pseudo velocity estimation
                        vrn_output = vrn_net(batch_gpu) if vrn_net else None

                        # Step 2: Pose estimation with global context
                        model_out = net(batch_gpu, vrn_output) if model_config.m_type in ('rnn', 'vrnrnn', 'att', 'vrnatt', 'diff','vrndiff') else net(
                            batch_gpu)

                        if model_config.VERSION == "JC":
                            me.compute(pose=batch_gpu.vicon_jc, pose_hat=model_out['pose_hat'],
                                                   contact=batch_gpu.vicon_contact,
                                                   contact_hat=model_out.get('contact_hat', None),
                                                   rot=batch_gpu.vicon_ori,
                                                   rot_hat=model_out.get('orientation_hat', None),
                                                    seq_lengths=batch_gpu.seq_lengths,
                                                   pose_root=batch_gpu.root)
                            gt_poses.append(batch_gpu.vicon_jc)

                        elif model_config.VERSION == "FK":
                            me.compute(pose=batch_gpu.pose, pose_hat=model_out['pose_hat'],
                                       joint_rot=batch_gpu.joint_rotations,
                                       joint_rot_hat=model_out['joint_rot_hat'],
                                       relative_joints=batch_gpu.relative_joints,
                                       relative_joints_hat=model_out.get('relative_joints_hat', None),
                                       relative_phase=batch_gpu.relative_phase,
                                       relative_phase_hat=model_out.get('relative_phase_hat', None),
                                       contact=batch_gpu.vicon_contact,
                                       contact_hat=model_out.get('contact_hat', None),
                                       seq_lengths=batch_gpu.seq_lengths,
                                       pose_root=batch_gpu.root)

                            out_all_gt['gt_joint_rot'].append(batch_gpu.joint_rotations)
                            out_all_gt['gt_pose'].append(batch_gpu.pose)
                            out_all_gt['gt_root'].append(batch_gpu.root)
                            out_all_gt['gt_contact'].append(batch_gpu.vicon_contact)

                        for k in model_out:
                            out_all_model[k].append(model_out[k])

                metrics = me.get_metrics()
                metric_vals_ind.append([sub_indx, 'Overall average'] + [metrics[k] for k in metrics])
                headers = [k for k in metrics]
                summary_string = tabulate(metric_vals_ind, headers=['Nr', 'E2E {}'.format(config.experimentid)] + headers)
                print(summary_string)
                print("=" * 80 + "\n")

            if config.visualize:

                if model_config.VERSION == "JC":
                    pose = torch.cat(pose_true, dim=1).squeeze().reshape(-1, len(C.JC_LABELS_Full), 3).cpu().numpy()
                    pose_hat = torch.cat(out_all_model['pose_hat'], dim=1).squeeze().reshape(-1, len(C.JC_LABELS_Full),
                                                                                             3).cpu().numpy()

                    # Add root to gt and prediction
                    if model_config.predict_root:
                        root_pred = get_predicted_root(model_out['root_hat'], model_config)
                        root_pred = root_pred.squeeze().reshape(-1, 1, 3).cpu().numpy()
                        pose_hat += root_pred

                        root = pose_true.root.squeeze().reshape(-1, 1, 3).cpu().numpy()
                        root_rec = get_predicted_root(pose_true.root_delta.squeeze().reshape(1, -1, 3), model_config)
                        root_rec = root_rec.squeeze().reshape(-1, 1, 3).cpu().numpy()
                        pose += root

                elif model_config.VERSION == "FK":

                    pose = torch.cat(out_all_gt['gt_pose'], dim=1).squeeze().reshape(-1, len(C.FK_JOINTS_FUll), 3).cpu().numpy()
                    pose_hat = torch.cat(out_all_model['pose_hat'], dim=1).squeeze().reshape(-1, len(C.FK_JOINTS_FUll), 3).cpu().numpy()

                    # Add root to gt and prediction
                    if model_config.predict_root:
                        root_pred = get_predicted_root(model_out['root_hat'], model_config)
                        root_pred = root_pred.squeeze().reshape(-1, 1, 3).cpu().numpy()
                        pose_hat += root_pred

                        # root = out_all_gt['root'].squeeze().reshape(-1, 1, 3).cpu().numpy()
                        # root_rec = get_predicted_root(pose_true.root_delta.squeeze().reshape(1, -1, 3), model_config)
                        # root_rec = root_rec.squeeze().reshape(-1, 1, 3).cpu().numpy()
                        pose += root

                if model_config.m_type not in('rnndct', 'attdct'):
                    if model_config.VERSION == 'JC':

                        lines_pred = H.get_lines(pose_hat)

                        lines_gt = H.get_lines(pose)

                        predicted = Lines(lines_pred, mode="lines", color=(1, 0, 0, 1),
                                          position=np.array([0.0, 0.0, 0.0]),
                                          rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))

                        original = Lines(lines_gt, mode="lines", color=(0, 1, 0, 1),
                                         position=np.array([1.0, 0.0, 0.0]),
                                         rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
                    if model_config.VERSION == 'FK':
                        connections = get_skeleton_connections(model_config)
                        predicted = Skeletons(pose_hat, connections,
                                             position=np.array([1.0, 0.0, 0.0]), color=(1, 0, 0, 1),
                                             rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
                        predicted.rotation = predicted.rotation

                        original = Skeletons(pose, connections,
                                                position=np.array([0.0, 0.0, 0.0]), color=(0, 1, 0, 1),
                                                rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
                        original.rotation = original.rotation

                elif model_config.m_type in ('rnndct', 'attdct'):
                    predicted = Spheres(pose_hat, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
                    original = Spheres(pose, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi),
                                             position=[1.0, 0.0, 0.0],
                                             color=np.array([0, 1, 0, 1]))

                v = Viewer()
                # Add content here (e.g. animations, markers)y
                v.scene.add(predicted, original)
                v.run()

            if config.save_metrics_excel:
                me.save_metrics_excel(model['model_dir'], sub_indx, config)

            if config.save_matlab:
                sf = model_config.m_sampling_freq if hasattr(model_config, 'm_sampling_freq') else 50.0

                if model_config.VERSION == "FK":
                    save_fk_results(out_all_gt, out_all_model, sub_indx, config, model_config, model, sf)

                elif model_config.VERSION == "JC":
                    save_jc_results(batch, out_all_model, sub_indx, config, model_config, model, sf)

            if config.plot:

                pose_hat = model_out['pose_hat'].squeeze().cpu().numpy()
                if model_out.get('root_hat') is not None:
                    root_pred = get_predicted_root(model_out['root_hat'], model_config)
                    root_pred = root_pred.squeeze().reshape(-1, 1, 3).cpu().numpy()
                    # pose_hat += root_pred
                    # root_pred = model_out['root_hat'].cpu().numpy()
                    # root_pred = np.expand_dims(root_pred, axis=1)
                    root_pred_mean = np.mean(root_pred, axis=0)

                pose = batch.pose.squeeze().reshape(-1, len(C.FK_JOINTS_FUll), 3).cpu().numpy()
                if getattr(model_config, 'predict_root', False):
                    # root = batch.root.squeeze().reshape(-1, 1, 3).cpu().numpy()
                    root_rec = get_predicted_root(batch.root_delta.squeeze().reshape(1, -1, 3), model_config)
                    root_rec = root_rec.squeeze().reshape(-1, 1, 3).cpu().numpy()
                    # root_rec = batch.root_delta[0].cpu().numpy()
                    root_rec_mean = np.mean(root_rec, axis=0)
                    # pose += root

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--experimentid', type=str, default="Search_BIRNN")
    p.add_argument('--partition', default='test_specific',
                   choices=['easy', 'hard', 'unseen', 'test_specific', 'unseen_specific'])
    p.add_argument('--trial', default='Norm_Post', type=str, choices=['Norm_Pre', 'White', 'Pink', 'Norm_Post'],
                   help='Trials to chose from for visualization.')
    p.add_argument('--subject', default=['75'], type=int, help='Chose subject to visualize.')
    #['04','09','14','24','51','54','67','81']
    p.add_argument('--save_matlab', action='store_true', help='Save kinematic data to matlab struct')
    p.add_argument('--save_metrics_excel', action='store_true', help='Save mean & std of euclidean distance per joint')
    p.add_argument('--visualize', action='store_false', help='Visualize subject trial.')
    p.add_argument('--plot', action='store_true', help='Placeholder for plots.')
    args = p.parse_args()

    main(args)

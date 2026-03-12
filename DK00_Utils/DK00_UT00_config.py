"""Basic configuration Settings for visualization and training parameters"""
import os
import ast
import json
import torch
import pprint
import argparse
import socket
from sys import platform

try:
    ip_address = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    ip_address = None  # Handle cases where the IP address cannot be resolved

if platform == "linux" or platform == "linux2":

    dir_path = f'/cluster/project/hilliges/yonkim/TrainingData/'
    save_dir_path = f'/cluster/project/hilliges/yonkim/Models_trained'
    save_dir_path_Final = f'/cluster/project/hilliges/yonkim/Models_trained'
    save_wandb_path = f'/cluster/project/hilliges/yonkim/wandb'
    save_sweep_path = f'/cluster/project/hilliges/yonkim/'
    UNSEEN_MOTION_CSV = f'/cluster/home/yonkim/dl_humanmotion/DK00_Utils/test_set_unseen.csv'

    if ip_address == "192.168.10.141" and not None:

        dir_path = f'/1TB/DataForWork/DeepKinematics/Training Data'
        save_dir_path = f'/1TB/DataForWork/DeepKinematics/Models_trained'
        save_dir_path_Final = f'/1TB/DataForWork/DeepKinematics/Models_trained/Final'
        save_wandb_path = f'/1TB/DataForWork/DeepKinematics/wandb'
        save_sweep_path = f'/1TB/DataForWork/DeepKinematics'
        UNSEEN_MOTION_CSV = f'/home/yong/Desktop/dl_humanmotion/DK00_Utils/test_set_unseen.csv'

elif platform == "win32":
    dir_path = f'D:/04_DeepKinematics/Training Data/'
    save_dir_path = f'D:/04_DeepKinematics/Models_trained'
    save_dir_path_Final = f'D:/04_DeepKinematics/Models_trained/Final'
    save_wandb_path = f'D:/04_DeepKinematics/wandb'
    save_sweep_path = f'D:/04_DeepKinematics/'
    UNSEEN_MOTION_CSV = f'C:/Users/ykuk0/Desktop/dl_humanmotion/DK00_Utils/test_set_unseen.csv'

elif platform == "darwin":
    dir_path = f'/Users/ykk/DataForWork/DeepKinematics/Training Data'
    save_dir_path = f'/Users/ykk/DataForWork/DeepKinematics/Manuscript'
    save_dir_path_Final = f'/Users/ykk/DataForWork/DeepKinematics/Manuscript/Final'
    save_wandb_path = f'/Users/ykk/DataForWork/DeepKinematics/wandb'
    save_sweep_path = f'/Users/ykk/DataForWork/DeepKinematics'
    UNSEEN_MOTION_CSV = f'/Users/ykk/Library/Mobile Documents/com~apple~CloudDocs/00_Others/01_ETH/01_PhD/04_Publications/00_UnderReview/09_IMU_DeepKinematics/dl_humanmotion-main/test_set_unseen.csv'

SAMPLING_FACTOR = 4
SAMPLING_FREQUENCY = 50

FILE_LIST = ['Norm_Pre', 'Norm_Post', 'White', 'Pink', 'FTSS_Pre', 'FTSS_Post', 'TUG_Pre', 'TUG_Post']

"Data allocation for training"

""" Training set """
subjects_train = ['01', '02', '03', '06', '07', '10',
'11', '12', '13', '15', '16', '17', '18', '19', '20',
'32', '33', '34', '35', '36','40',
'41', '42', '43', '44', '45', '46','47','48','49','50',
'53','59',
'63','64','65', '66','67','69','70',
'73','74','76','78','79',
'83','85','86','87']

""" Validation set: easy """
# Young
subjects_easy = ['21','22','23','25','26','27','28','29','30']

""" Validation set: hard """
# Older
subjects_hard = ['75','77','62','72','82']

subjects_test = ['04','09','14','24','31','51','54','56','67','80']

# For quick test
subjects_train = ['75']
subjects_easy = ['24']
subjects_hard = ['82']
subjects_test = ['04']

LEFT_FOOT_MOVEMENT_CORRECTION = ['03', '04', '06', '07', '09', '11', '12', '13', '14', '15', '16',
                                 '17', '19', '21', '22', '23', '24', '25', '26', '28', '30', '31',
                                 '32', '33', '34', '35', '36', '40', '41', '42', '47', '48', '51',
                                 '56', '61', '63', '64', '65', '67', '69', '70', '71', '72', '73',
                                 '75', '80', '81']

"GPU Settings"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")  # Use CPU if neither CUDA nor M1 GPU is available

DTYPE = torch.float32

"Subject Loop"
# Creating the range
subject = [num for num in range(1, 88) if num not in [5, 38, 39, 52, 55, 57, 58, 60, 68, 84]]
trials = [FILE_LIST[i] for i in [0, 1, 2, 3]]

"VICON Labels"
marker_labels = ['RTO1', 'RTO3', 'RTO5', 'RHEE', 'RMMA', 'RLMA', 'RTMT', 'RTLF',
                 'RTTT', 'RTIB', 'RMCO', 'RLCO', 'RTFR', 'RTLL', 'RTLH', 'LTO1',
                 'LTO3', 'LTO5', 'LHEE', 'LMMA', 'LLMA', 'LTMT', 'LTLF', 'LTTT',
                 'LTIB', 'LMCO', 'LLCO', 'LTFR', 'LTLL', 'LTLH', 'RASI', 'RTMS',
                 'RPSI', 'SACR', 'LPSI', 'LTMS', 'LASI', 'RSHO', 'CVC7', 'LSHO',
                 'MSTC', 'RWRA', 'RWUL', 'RFRA', 'RFUL', 'RMEC', 'RLEC', 'RHVT',
                 'RHLT', 'LWRA', 'LWUL', 'LFRA', 'LFUL', 'LMEC', 'LLEC', 'LHVT',
                 'LHLT', 'RFHD', 'RBHD', 'LBHD', 'LFHD']

VICON_LABELS_JOINT_CENTER = ['Head_CtoHThorax_score', 'LThorax_LUpArm_score', 'RThorax_RUpArm_score',
                             'LUpArm_LLoArm_score', 'RUpArm_RLoArm_score', 'LWRA', 'RWRA',
                             'LHJC', 'RHJC', 'LKJC', 'RKJC', 'LAJC', 'RAJC', 'LTO3', 'RTO3', 'LHEE', 'RHEE']

VICON_LABELS_SEGMENT_ORIENTATION = ['LTIO', 'LTIA', 'LTIL', 'LTIP', 'RTIO', 'RTIA', 'RTIL',
                                    'RTIP', 'LLoArmO', 'LLoArmA', 'LLoArmL', 'LLoArmP',
                                    'RLoArmO', 'RLoArmA', 'RLoArmL', 'RLoArmP', 'HeadO',
                                    'HeadA', 'HeadL', 'HeadP', 'PELO', 'PELA', 'PELL', 'PELP']

"""JOINT CENTRE -- Parameters """

JC_LABELS_Full = ['Head', 'Left Shoulder', 'Right Shoulder',
                         'Left Elbow', 'Right Elbow', 'Left Wrist',
                         'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle',
                         'Right Ankle', 'Left Toe', 'Right Toe']

JC_LABELS_NoArms = ['Head', 'Left Shoulder', 'Right Shoulder',
                    'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
                    'Left Toe',
                    'Right Toe']

JC_LABELS_NoArmsHead = ['Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle', 'Left Toe','Right Toe']


JC_EVAL_JOINTS_Full = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # Full body

JC_EVAL_JOINTS_NoArms = [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14]  # Evaluation Parameters without ARMS

JC_EVAL_JOINTS_NoArmsHead = [7, 8, 9, 10, 11, 12, 13, 14]  # Evaluation Parameters without ARMS

"""Forward Kinematics-- Parameters """
FK_JOINTS_FUll = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'Head_end0',
                  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHand_end0', 'RightHand_end1',
                  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHand_end0', 'LeftHand_end1',
                  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToeBase_end0',
                  'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToeBase_end0']

FK_JOINTS_NoArms = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'Head_end0',
                    'RightShoulder', 'LeftShoulder',
                    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToeBase_end0',
                    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToeBase_end0']

FK_JOINTS_NoArmsHead = ['Hips',
                    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToeBase_end0',
                    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToeBase_end0']

FK_FOOT_JOINTS = ['RightFoot', 'RightToeBase', 'RightToeBase_end0','LeftFoot', 'LeftToeBase','LeftToeBase_end0']
FK_Main_JOINTS = FK_FOOT_JOINTS

FK_PARENTS_Full = [-1, 0, 1, 2, 3, 4, 5, 6, 7,  # Upper trunk to head
                   4, 9, 10, 11, 12, 12,  # Right Shoulder
                   4, 15, 16, 17, 18, 18,  # Left Shoulder
                   0, 21, 22, 23, 24,  # Right leg
                   0, 26, 27, 28, 29]  # Left Leg

FK_PARENTS_NoArms = [-1, 0, 1, 2, 3, 4, 5, 6, 7,  # Upper trunk to head
                     4, 4,  # Parents for shoulders
                     0, 21, 22, 23, 24,  # Right leg
                     0, 26, 27, 28, 29]  # Left Leg

FK_PARENTS_NoArmsHead = [-1, 0, # Upper trunk to head
                     0, 21, 22, 23, 24,  # Right leg
                     0, 26, 27, 28, 29]  # Left Leg

FK_EVAL_JOINTS_FUll = [0, 1, 2, 3, 4, 5, 6, 7, 8,  # Upper trunk to head
                       9, 10, 11, 12, 13, 14,  # Right Shoulder
                       15, 16, 17, 18, 19, 20,  # Left Shoulder
                       21, 22, 23, 24, 25,  # Right leg
                       26, 27, 28, 29, 30]  # Left leg

FK_EVAL_JOINTS_NoArms = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                         9, 15,
                         21, 22, 23, 24, 25,  # Right leg
                         26, 27, 28, 29, 30]  # Left leg

FK_EVAL_JOINTS_NoArmsHead = [0,
                         21, 22, 23, 24, 25,  # Right leg
                         26, 27, 28, 29, 30]  # Left leg

FK_SKELETON_Full = [
    [(-1, 0, 'Hips')],
    [(0, 1, 'Spine'), (0, 21, 'RightUpLeg'), (0, 26, 'LeftUpLeg')],
    [(1, 2, 'Spine1'), (21, 22, 'RightLeg'), (26, 27, 'LeftLeg')],
    [(2, 3, 'Spine2'), (22, 23, 'RightFoot'), (27, 28, 'LeftFoot')],
    [(3, 4, 'Spine3'), (23, 24, 'RightToeBase'), (28, 29, 'LeftToeBase')],
    [(4, 5, 'Neck'), (24, 25, 'RightToeBase_end0'), (29, 30, 'LeftToeBase_end0')],
    [(5, 6, 'Neck1')],
    [(6, 7, 'Head')],
    [(7, 8, 'Head_end0')],
    [(4, 9, 'RightShoulder'), (4, 15, 'LeftShoulder')],
    [(9, 10, 'RightArm'), (15, 16, 'LeftArm')],
    [(10, 11, 'RightForeArm'), (16, 17, 'LeftForeArm')],
    [(11, 12, 'RightHand'), (17, 18, 'LeftHand')],
    [(12, 13, 'RightHand_end0'), (18, 19, 'LeftHand_end0')],
    [(12, 14, 'RightHand_end1'), (18, 20, 'LeftHand_end1')]
]

FK_SKELETON_NoArms = [[(-1, 0, 'Hips')],
                      [(0, 1, 'Spine'), (0, 21, 'RightUpLeg'), (0, 26, 'LeftUpLeg')],
                      [(1, 2, 'Spine1'), (21, 22, 'RightLeg'), (26, 27, 'LeftLeg')],
                      [(2, 3, 'Spine2'), (22, 23, 'RightFoot'), (27, 28, 'LeftFoot')],
                      [(3, 4, 'Spine3'), (23, 24, 'RightToeBase'), (28, 29, 'LeftToeBase')],
                      [(4, 5, 'Neck'), (24, 25, 'RightToeBase_end0'), (29, 30, 'LeftToeBase_end0')],
                      [(5, 6, 'Neck1')],
                      [(6, 7, 'Head')],
                      [(7, 8, 'Head_end0')],
                      [(4, 9, 'RightShoulder'), (4, 15, 'LeftShoulder')]]

FK_SKELETON_NoArmsHead = [[(-1, 0, 'Hips')],
                      [(0, 21, 'RightUpLeg'), (0, 26, 'LeftUpLeg')],
                      [(21, 22, 'RightLeg'), (26, 27, 'LeftLeg')],
                      [(22, 23, 'RightFoot'), (27, 28, 'LeftFoot')],
                      [(23, 24, 'RightToeBase'), (28, 29, 'LeftToeBase')],
                      [(24, 25, 'RightToeBase_end0'), (29, 30, 'LeftToeBase_end0')]]

skeleton_pairs = [
    (0, 21),   # Hips -> RightUpLeg
    (0, 26),   # Hips -> LeftUpLeg
    (21, 22),  # RightUpLeg -> RightLeg
    (22, 23),  # RightLeg -> RightFoot
    (23, 24),  # RightFoot -> RightToeBase
    (24, 25),  # RightToeBase -> RightToeBase_end0
    (26, 27),  # LeftUpLeg -> LeftLeg
    (27, 28),  # LeftLeg -> LeftFoot
    (28, 29),  # LeftFoot -> LeftToeBase
    (29, 30),  # LeftToeBase -> LeftToeBase_end0
    (4, 9),    # Neck -> RightShoulder
    (4, 15),   # Neck -> LeftShoulder
    (9, 10),   # RightShoulder -> RightArm
    (15, 16),  # LeftShoulder -> LeftArm
    (10, 11),  # RightArm -> RightForeArm
    (16, 17),  # LeftArm -> LeftForeArm
    (11, 12),  # RightForeArm -> RightHand
    (17, 18),  # LeftForeArm -> LeftHand
]

JOINT_EULER_CONVENTIONS = {
    'Hips': 'xzy',
    'Spine': 'xzy',
    'Spine1': 'xzy',
    'Spine2': 'xzy',
    'Spine3': 'xzy',
    'Neck': 'xzy',
    'Neck1': 'xzy',
    'Head': 'xzy',
    'RightShoulder': 'yzx',
    'RightArm': 'xzy',
    'RightForeArm': 'zyx',
    'RightHand': 'xzy',
    'LeftShoulder': 'yzx',
    'LeftArm': 'xzy',
    'LeftForeArm': 'zyx',
    'LeftHand': 'xzy',
    'RightUpLeg': 'xzy',
    'RightLeg': 'xyz',
    'RightFoot': 'xzy',
    'RightToeBase': 'xyz',
    'LeftUpLeg': 'xzy',
    'LeftLeg': 'xyz',
    'LeftFoot': 'xzy',
    'LeftToeBase': 'xyz',
}

class Constants(object):
    """
    A singleton for some common constants.
    """

    class __Constants:
        def __init__(self):
            # Environment setup
            self.DEVICE = DEVICE
            self.DTYPE = DTYPE
            self.DATA_DIR = dir_path
            self.save_model_DIR = save_dir_path
            self.save_model_DIR_Final = save_dir_path_Final
            self.save_wandb_DIR = save_wandb_path
            self.save_sweep_DIR = save_sweep_path
            self.subjects_train = subjects_train
            self.subjects_easy = subjects_easy
            self.subjects_hard = subjects_hard
            self.UNSEEN_MOTION_CSV = UNSEEN_MOTION_CSV

            # Factor to divide sampling frequency by and downsample sequences
            self.SAMPLING_FACTOR = SAMPLING_FACTOR
            self.SAMPLING_FREQUENCY = SAMPLING_FREQUENCY

            # Up-to-Date for 01 to 75 except for 63 to 65 and 49,50
            self.LEFT_FOOT_MOVEMENT_CORRECTION = LEFT_FOOT_MOVEMENT_CORRECTION
            self.skeleton_pairs = skeleton_pairs

            # Joints of Interest for JC
            self.JC_LABELS_Full = JC_LABELS_Full
            self.JC_LABELS_NoArms = JC_LABELS_NoArms
            self.JC_LABELS_NoArmsHead = JC_LABELS_NoArmsHead  # no arms

            # Joints for Evaluation for JC
            self.JC_EVAL_JOINTS_Full = JC_EVAL_JOINTS_Full  # FUll
            self.JC_EVAL_JOINTS_NoArms = JC_EVAL_JOINTS_NoArms  # no arms
            self.JC_EVAL_JOINTS_NoArmsHead = JC_EVAL_JOINTS_NoArmsHead  # no arms

            "FK"
            # Joints of Interest for FK
            self.FK_PARENTS_Full = FK_PARENTS_Full
            self.FK_PARENTS_NoArms = FK_PARENTS_NoArms
            self.FK_PARENTS_NoArmsHead = FK_PARENTS_NoArmsHead

            self.FK_JOINTS_FUll = FK_JOINTS_FUll
            self.FK_JOINTS_NoArms = FK_JOINTS_NoArms
            self.FK_JOINTS_NoArmsHead = FK_JOINTS_NoArmsHead
            self.FK_FOOT_JOINTS = FK_FOOT_JOINTS

            # Joints for Evaluation
            self.FK_EVAL_JOINTS_FUll = FK_EVAL_JOINTS_FUll
            self.FK_EVAL_JOINTS_NoArms = FK_EVAL_JOINTS_NoArms
            self.FK_EVAL_JOINTS_NoArmsHead = FK_EVAL_JOINTS_NoArmsHead

            # SPL Layer Config
            self.FK_skeleton_Full = FK_SKELETON_Full
            self.FK_skeleton_NoArms = FK_SKELETON_NoArms
            self.FK_skeleton_NoArmsHead = FK_SKELETON_NoArmsHead

            self.JOINT_EULER_CONVENTIONS = JOINT_EULER_CONVENTIONS
    instance = None

    def __new__(cls, *args, **kwargs):
        if not Constants.instance:
            Constants.instance = Constants.__Constants()
        return Constants.instance

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)


class Configuration(object):
    """ Configuration options for training/eval/test runs. """

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        # String representation for pretty printing
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_units(input_str):
        """Helper function to parse hidden units from string to list."""
        try:
            # print(f"Parsing input: {input_str}")  # Debug: print input being parsed
            units = ast.literal_eval(input_str)
            # print(f"Evaluated hidden units: {hidden_units}")
            if isinstance(units, list) and all(isinstance(i, list) for i in units):
                return units

            else:
                raise argparse.ArgumentTypeError("Each entry must be a list of integers")
        except (SyntaxError, ValueError):
            raise argparse.ArgumentTypeError("Inputs must be a valid list of lists of integers")

    @staticmethod
    def parse_cmd(version=str, m_type=str, experimentid=str, m_positional_encoding_type = str, optimizer=str, scheduler=str):

        parser = argparse.ArgumentParser()

        # Dataset & Learning Configuration
        parser.add_argument('--VERSION', type=str, default=version, help='JC or FK?')
        parser.add_argument('--experimentid', type=str, default=experimentid, help='Use this experiment ID or create new one.')
        parser.add_argument('--eval_every', type=int, default=5, help='Evaluate validation set every so many epochs.')
        parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs.')
        parser.add_argument('--optimizer', type=str, default=optimizer, help= 'Type of Optimizer.')
        parser.add_argument('--scheduler', type=str, default=scheduler, help= 'Type of scheduler.')
        parser.add_argument('--m_type', type=str, default=m_type, help='The type of model.')
        parser.add_argument('--lr', nargs='+', type=float, default=[1e-3], help='Learning rate.')
        parser.add_argument('--weight_decay', nargs='+', type=float, default=[1e-1], help='Weight Decay.')
        parser.add_argument('--n_offset', type=int, default=10, help='Number of frames for offset calculation.')
        parser.add_argument('--window_size', type=int, default=500, help='Number of frames to extract per sequence.')
        parser.add_argument('--eval_window_size', type=int, default=None, help='Window size for evaluation on test set.')
        parser.add_argument('--bs_train', type=int, default=128, help='Batch size for the training set.')
        parser.add_argument('--bs_eval', type=int, default=64, help='Batch size for valid/test set.')
        parser.add_argument('--load', action='store_true', help='Whether to load the model with the given ID.')

        # Intput Types
        parser.add_argument('--m_sampling_freq', type=float, default=50, help='Sampling frequency of data.')
        parser.add_argument('--ori_offset', action='store_false', help='Align IMU orientation to VICON/bvh segments.')
        parser.add_argument('--use_acc_gyro', action='store_true', help='Use raw acc and gyro  as input.')
        parser.add_argument('--use_orientation', action='store_true', help='Use orientation as input.')
        parser.add_argument('--use_quats', action='store_true', help='Use quaternions as input.')
        parser.add_argument('--predict_arms', action='store_true', help='Also predict arm joint center.')
        parser.add_argument('--predict_head', action='store_true', help='Also predict head joint center.')
        parser.add_argument('--predict_contact', action='store_true', help='Do contact prediction.')
        parser.add_argument('--predict_velocity', action='store_true', help='Do phase prediction.')
        parser.add_argument('--predict_joints', action='store_true', help='Do phase prediction.')
        parser.add_argument('--predict_phase', action='store_true', help='Do phase prediction.')
        parser.add_argument('--predict_orientation', action='store_true', help='Do orientation prediction.')
        parser.add_argument('--predict_root', action='store_true', help='Predict root trajectory.')# TO DO FOR JC AND FK
        parser.add_argument('--predict_spl', action='store_true', help='Do contact prediction.')

        # Loss
        parser.add_argument('--m_pose_loss', nargs='+', type=float, default=[1], help='pose loss.')
        parser.add_argument('--m_contact_loss', nargs='+', type=float, default=[1], help='Weighted foot prediction loss.')
        parser.add_argument('--m_velocity_loss', nargs='+', type=float, default=[1], help='Weighted foot prediction loss.')
        parser.add_argument('--m_orientation_loss', nargs='+', type=float, default=[1], help='Weighted Orientation prediction loss.')
        parser.add_argument('--m_foot_loss', nargs='+', type=float, default=[1], help='Weighted foot prediction loss.')
        parser.add_argument('--m_root_loss', nargs='+',type=float, default=[1], help="Root prediction loss.")

        args, unknown = parser.parse_known_args()

        if args.VERSION == 'JC':
            # JC INPUT SPECIFIC
            parser.add_argument('--normalize_height', action='store_false', help='Normalize the height of subjects.')
            parser.add_argument('--normalize_joint_pos', action='store_false', help='Fix midpoint between hip joints to origin.')

        if args.VERSION == 'FK':
            parser.add_argument('--m_joint_rot_loss', nargs='+', type=float, default=[1], help='Joint Rotation loss.')
            parser.add_argument('--m_joints_loss', nargs='+', type=float, default=[1], help='Joint loss.')
            parser.add_argument('--m_phase_loss', nargs='+', type=float, default=[1], help='Phase loss.')

            # FK INPUT SPECIFIC
            parser.add_argument('--global_joint_rot', action='store_false', help='Use global joint rotation for offset calc.')
            parser.add_argument('--add_gait_vel', action='store_true', help='Add gait velocity to position deltas.')

        # Model Configuration
        parser.add_argument('--m_dropout', nargs='+', type=float, default=[0.3], help='Dropout applied on inputs.')
        parser.add_argument('--m_bidirectional', action='store_true', help='Bidirectional RNN.')
        parser.add_argument('--m_learn_init_state', action='store_true', help='Learn initial hidden state.')
        parser.add_argument('--m_skip_connections', action='store_true', help='Skip connections in the MLP.')
        parser.add_argument('--use_batch_norm', action='store_true', help='Batch Normalisation.')
        parser.add_argument('--m_hidden_units_SPL', type=Configuration.parse_units, default=[[64]])

        if 'rnn' in args.m_type :
            parser.add_argument('--m_num_layers_RNN', nargs='+', type=int, default=[2])
            parser.add_argument('--m_hidden_units_RNN', type=Configuration.parse_units, default=[[64], [256, 512]])

            if 'vrn' in args.m_type:
                parser.add_argument('--m_num_layers_VRN', nargs='+', type=int, default=[2])
                parser.add_argument('--m_hidden_units_VRN', type=Configuration.parse_units, default=[[64], [256, 512]])
                parser.add_argument('--m_embedding_VRNMLP', type=Configuration.parse_units, default=[[16], [16, 32]])

            if 'dct' in args.m_type:
                parser.add_argument('--m_embedding_MLP', type=Configuration.parse_units, default=[[256]])
                parser.add_argument('--m_num_stage', nargs='+', type=int, default=[2])
                parser.add_argument('--m_num_nodes', nargs='+', type=int, default=[48])
                parser.add_argument('--n_dct', type=int, default=70)

        elif 'att' in args.m_type or 'diff' in args.m_type:

            parser.add_argument('--num_diffusion_steps', nargs='+',type=int, default=[1000])
            parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'])

            # MLP
            parser.add_argument('--m_embedding_MLP', type=Configuration.parse_units, default=[[256]])

            # Positional Encoder
            parser.add_argument('--m_positional_encoding_type', type=str, default=m_positional_encoding_type)

            # Attention
            parser.add_argument('--m_num_layers_attention', nargs='+', type=int, default=[4])
            parser.add_argument('--m_num_hidden_units_attention', type=Configuration.parse_units, default=[[32]])
            parser.add_argument('--m_num_heads_attention', type=Configuration.parse_units, default=[[32]])
            parser.add_argument('--m_embedding_attention', type=Configuration.parse_units, default=[[32]])
            parser.add_argument('--m_window_attention', nargs='+', type=int, default=[200])

            # gcn
            parser.add_argument('--use_gcn', action='store_true')

            if 'vrn' in args.m_type:
                parser.add_argument('--m_num_layers_VRN', nargs='+', type=int, default=[2])
                parser.add_argument('--m_hidden_units_VRN', type=Configuration.parse_units, default=[[512]])
                parser.add_argument('--m_embedding_VRNMLP', type=Configuration.parse_units, default=[[256]])

        argparse.BooleanOptionalAction
        config = parser.parse_args()

        # print("Parsed m_hidden_units:", config.m_hidden_units)
        return Configuration(vars(config))

    def get_sweep_configuration(self):

        base_keys = {
            'm_dropout', 'weight_decay', 'lr',
            'm_pose_loss', 'm_joint_rot_loss',
            'm_velocity_loss', 'm_contact_loss',
            'm_joints_loss','m_phase_loss',
            'm_root_loss', 'm_foot_loss',
        }

        config_map = {
            'rnn': base_keys | {
                'm_num_layers_RNN', 'm_hidden_units_RNN', 'm_hidden_units_SPL'
            },
            'rnndct': base_keys | {
                'm_embedding_MLP', 'm_num_layers_RNN', 'm_hidden_units_RNN',
                'm_hidden_units_SPL'
            },
            'vrnrnn': base_keys | {
                'm_num_layers_RNN', 'm_hidden_units_RNN',
                'm_num_layers_VRN', 'm_hidden_units_VRN', 'm_embedding_VRNMLP',
                'm_hidden_units_SPL'
            },
            'att': base_keys | {
                'm_embedding_MLP',
                'm_num_layers_attention', 'm_num_hidden_units_attention',
                'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                'm_hidden_units_SPL'
            },
            'attdct': base_keys | {
                'm_embedding_MLP',
                'm_num_layers_attention', 'm_num_hidden_units_attention',
                'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                'm_hidden_units_SPL'
            },
            'diff': base_keys | {
                'm_embedding_MLP','num_diffusion_steps',
                'm_num_layers_attention', 'm_num_hidden_units_attention',
                'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                'm_hidden_units_SPL'
            },
            'vrnatt': base_keys | {
                'm_num_layers_VRN', 'm_hidden_units_VRN', 'm_embedding_VRNMLP',
                'm_embedding_MLP', 'm_num_layers_attention', 'm_num_hidden_units_attention',
                'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                'm_hidden_units_SPL'
            },
            'vrndiff': base_keys | {
                'm_num_layers_VRN', 'm_hidden_units_VRN', 'm_embedding_VRNMLP',
                'num_diffusion_steps',
                'm_embedding_MLP', 'm_num_layers_attention', 'm_num_hidden_units_attention',
                'm_num_heads_attention', 'm_embedding_attention', 'm_window_attention',
                'm_hidden_units_SPL'
            }
        }

        allowed_keys = config_map.get(self.m_type, set())

        if self.VERSION == 'FK':
            allowed_keys |= {'m_joint_rot_loss'}

        # Flatten sweep parameters directly (avoid nesting under 'parameters')
        sweep_parameters = {
            key: {'values': val if isinstance(val, list) else [val]}
            for key, val in self.__dict__.items()
            if key in allowed_keys and val is not None
        }

        sweep_configuration = {
            'method': 'grid',
            'metric': {
                'goal': 'minimize',
                'name': 'MPJAE [deg]  unseen',
                # 'name': 'MPJAE STD  unseen',
            },

            'parameters': sweep_parameters
        }

        return sweep_configuration

    def update(self, adict):
        """Update the configuration attributes with a given dictionary."""
        self.__dict__.update(adict)

    @staticmethod
    def from_json(json_path):
        with open(json_path, 'r') as f:
            config = json.load(f)
        return Configuration(config)

    def to_json(self, json_path):
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)

CONSTANTS = Constants()
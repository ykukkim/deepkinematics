""" 
Functions used to import IMU (Inertial Measurement Unit) data (acc, gyro, mag, quat) 
stored in Matlab-structs and body markers from Vicon stored in c3d files into 
numpy arrays, synchronize Vicon to IMU and save both to the disk as npz-files.
"""

import mat73
import numpy as np
from DK00_Utils import DK00_UT00_config as config_v

from ezc3d import c3d
from DK01_Data_Conversion.outdated_scripts.bvh import Bvh

INCH = 2.54

def synchronize_vicon_to_imu(frames_vicon, frames_imu):
    """ 
    Synchronize Vicon and IMU trials by comparing their number of frames.
    If number of frames is not equal, delete excess frames at start of sequence. 
    
    Args:
    :param frames_vicon: Number of frames of Vicon trial. (Number)
    :param frames_imu: Number of frames of IMU trial. (Number)

    Returns:
    :start_frame_vicon: Start frame for Vicon trial.
    :start_frame_imu: Start frame for IMU trial.
    """
    if frames_imu - frames_vicon >= 0:
        start_frame_imu = frames_imu - frames_vicon
        start_frame_vicon = 0
    else:
        start_frame_vicon = frames_vicon - frames_imu
        start_frame_imu = 0
    return start_frame_vicon, start_frame_imu


class C3dFile:
    """ 
    Import joint center/orientation markers from c3d file. 
    Calculate segment orientation from orientation markers.
    Check joint center markers and segment orientation for NaNs. 
    Delete frames which have NaNs.  
    Adjust start frame for sync. Save data to disk.
    """
    def __init__(self, path, file, subject_nr):
        """ 
        Initializer:
        :param path: Path to the subject (string)
        :param file: Filename (string)
        :param subject_nr: Number of Subject (string)
        
        Vars:
        :joint_centers: (N, n_joints, 3)
            Order: Head, left shoulder, right shoulder, left elbow, right elbow, 
                left wrist, right wrist, left hip, right hip, left knee, right knee,
                left ankle, right ankle, left toe, right toe, left heel, right heel
        :orientation_markers: (N, n_sensors, 3)   
            Order: left ankle, right ankle, left arm, right arm, head, trunk    
            Description: Four markers define the orientation of a segment.
                The last letter of the marker denotes its axis: O=Origin, A=X-axis, L=Y-axis, P=Z-axis
        """  
        self.path = path
        self.file = file
        self.subject_nr = subject_nr

        # c = c3d(self.path + '/Vicon/' + self.file + '.c3d')
        c = c3d(self.path  + '/'+ self.file + '.c3d')

        self.markers = c['data']['points'].transpose()
        self.label_list = list(c['parameters']['POINT']['LABELS']['value'])

        self.n_frames = self.markers.shape[0]

        self.joint_centers = self.extract_coordinates(config_v.VICON_LABELS_JOINT_CENTER)
        self.orientation_markers = self.extract_coordinates(config_v.VICON_LABELS_SEGMENT_ORIENTATION)
        
        del(self.markers)
        del(self.label_list)                
   
    def extract_coordinates(self, labels):
        """ 
        Import joint center/orientation markers by their index. 
                
        Args: 
        :param labels: Label of markers. (List)
        
        Return:
        :data: Marker positions in meters.  (F, J, 3)
        """
        index = self.get_indeces_of_labels(labels)
        data = self.markers[:, index, :3]/1000 # convert from [mm] to [m]
        return data

    def get_indeces_of_labels(self, labels):
        """ 
        Extract the index of markers in label_list. 
        
        Args:
        :param labels: List of labels of joint center/orientation markers. (List)

        Returns:
        :index: Indexes of joint/center markers. (List)
        """
        index = []
        for label in labels:
            index.append(self.label_list.index(label))
        
        """ Add control to check if all necessary markers are imported!!! """
        if len(labels) == 17:
            assert len(index) == len(labels), "Joint center markers are missing!"   
        elif len(labels) == 24:
            assert len(index) == len(labels), "Orientation markers are missing!"

        return index
    
    def adjust_start_frame(self, sf):
        """ 
        Adjust start frame to be in sync with IMU data. 
        
        Args:
        :param sf: Start frame of Vicon trial. (Number)
        """
        self.joint_centers = self.joint_centers[sf:,...]
        self.segment_orientation = self.segment_orientation[sf:,...]
    

    def check_data_for_NaN(self):
        """ 
        Check joint center markers and segment orientation for NaNs and
        save index of frames including them in a list and delete those frames. 

        Returns:
        :corrupted_frames: List with indexes of frames having NaNs in data. 
        """
        corrupted_frames = []
        for i in range(self.joint_centers.shape[0]):
            
            if np.isnan(self.joint_centers[i,...]).any():
                corrupted_frames.append(i)

            if np.isnan(self.segment_orientation[i,...]).any():
                if i not in corrupted_frames:
                    corrupted_frames.append(i)

        corrupted_frames.sort()
        print('Corrupted Frames:', corrupted_frames)
        self.corrupted_frames = corrupted_frames
        self.delete_corrupted_frames()
        return corrupted_frames   
    
    
    def delete_corrupted_frames(self):
        """ Delete frames containing NaNs in joint center markers/segment orientation from sequence. """
        self.joint_centers = np.delete(self.joint_centers, self.corrupted_frames, axis = 0)
        self.segment_orientation = np.delete(self.segment_orientation, self.corrupted_frames, axis = 0)


    @np.errstate(all='ignore')
    def compute_segment_orientation(self):
        """ 
        Compute rotation matrix representing the segment orientation
        from four orientation markers. 

        Args:
        :orientation_markers: (N, n_orientations * 4, 3)

        Returns:
        :segment_orientation: (N, n_orientations, 3, 3)    
            (Order of n_orientations: left ankle, right ankle, left arm, 
            right arm, head, trunk)       
        """   
        data = self.orientation_markers  
        segments = np.empty([data.shape[0], 6, 3, 3])
        for i in range(6):
            # X-axis
            ox = np.subtract(data[:,4*i+1,:], data[:,4*i,:])
            x = np.divide(ox, np.linalg.norm(ox, axis=-1, keepdims = True))
            # Y-axis
            oy = np.subtract(data[:,4*i+2,:], data[:,4*i,:])
            y = np.divide(oy, np.linalg.norm(oy, axis=-1, keepdims = True))
            # Z-axis
            oz = np.subtract(data[:,4*i+3,:], data[:,4*i,:])
            z = np.divide(oz, np.linalg.norm(oz, axis=-1, keepdims = True))

            segments[:, i,...] = np.transpose(np.stack((x,y,z), axis = 1), axes = (0,2,1))

        self.segment_orientation = segments


    def save_to_disk(self, **kwargs):
        """ Save joint center markers, segment orientations and if selected contact phase to a npz file. """
        path = self.path + f'/{self.file}_{self.subject_nr}_vicon'
        np.savez_compressed(path, jc = self.joint_centers[:,:15,:] , ori = self.segment_orientation, **kwargs)
        print(f'Vicon: SonE_{self.subject_nr} {self.file} has been converted.')



class IMUFile:
    """ 
    Import acceleration (acc), angular velocity (gyro), magnetic field (mag)
    and sensor orientation(quat) from Matlab struct.
    Trim data to match bvh/c3d and adjust the start frame accordingly.
    Save trimmed/synced data to disk in a npz-file.
    """
    def __init__(self, path, subject_nr):
        """ 
        Initializer:
        :param path: Path to the subject. (string)
        :param subject_nr: Number of Subject. (string)

        Returns:
        :label_convention: List with used labels in the Matlab struct.
        :indeces_of_trials: Index of the row where trial is saved in the Matlab struct.
        :n_frames: Number of frames a trial has.
        """
        self.path = path
        self.subject_nr = subject_nr

        data = mat73.loadmat(path + f'/GaitSummaryIMU_ModulesData_SonE_{subject_nr}.mat', use_attrdict=True)
        self.data = data['GaitSummaryIMU_ModulesData']

        self.label_convention = self.get_label_convention()

        self.indeces_of_trials = self.get_indeces_of_trials()
        self.n_frames = self.get_nr_of_frames_for_trials()


    def get_label_convention(self):
        """ 
        Identify which sensor label convention is used in the Matlab struct. 
        """
        IMU_SENSOR_LABELS_OLD = ['ankle_L', 'ankle_R', 'arm_L', 'arm_R', 'head', 'trunk']
        IMU_SENSOR_LABELS_NEW = ['ankle_L', 'ankle_R', 'arm_left', 'arm_right', 'head', 'trunk']

        if 'arm_L' in self.data.keys():
            return IMU_SENSOR_LABELS_OLD
        
        return IMU_SENSOR_LABELS_NEW 

    
    def get_indeces_of_trials(self):       
        """ 
        Save row index of a trial in a dictionary.
            Example: dict['walk'] = 3 

        :return: Dictionary of indeces of trials.
        """
        data = self.data['ankle_L']['synced']['trim']

        indices = {}
        for i in range(len(data)):          
            indices[data[i]['Trials']] = i

        return indices
    
    def get_nr_of_frames_for_trials(self):       
        """  Save dictionary with the number of frames for every trial. """
        data = self.data['ankle_L']['synced']['trim']

        n_frames = {}
        for i in range(len(data)):          
            n_frames[data[i]['Trials']] = data[i]['acc'].shape[0]

        return n_frames
    
    
    def concatenate_data(self, trial, sf, corrupted_frames):
        """ 
        Extract acc/gyro/mag/quat data from Matlab struct and concatenate it together. 

        Args:
        :param trial: Name of trial to be converted.
        :param sf: Start frame of the IMU trial.
        :param corrupted_frames: List of indices of frames containing NaNs.

        Returns:
        acc: (N, n_sensors, 3), 
        gyro: (N, n_sensors, 3)
        mag: (N, n_sensors, 3)
        quat: (N, n_sensors, 4)     
            (Order of n_sensors: left ankle, right ankle, left arm, right arm, head, trunk)       
        """    
        index = self.indeces_of_trials[trial]

        acc = np.empty([self.n_frames[trial], 6, 3])    # [Frames, Sensors, Acc]
        gyro = np.empty([self.n_frames[trial], 6, 3])   # [Frames, Sensors, Gyro]
        mag = np.empty([self.n_frames[trial], 6, 3])    # [Frames, Sensors, Mag]
        quat = np.empty([self.n_frames[trial], 6, 4])   # [Frames, Sensors, Quat]
        
        for i in range(len(self.label_convention)):
            sensor = self.label_convention[i]

            acc[:,i,:] = self.data[sensor]['synced']['trim'][index]['acc']
            gyro[:,i,:] = self.data[sensor]['synced']['trim'][index]['gyro']
            mag[:,i,:] = self.data[sensor]['synced']['trim'][index]['mag']
            quat[:,i,:] = self.data[sensor]['synced']['trim'][index]['quat']

        # Delete corrupted frames and adjust start and end frame
        acc = np.delete(acc[sf:,...], corrupted_frames, axis = 0)
        gyro = np.delete(gyro[sf:,...], corrupted_frames, axis = 0)
        mag = np.delete(mag[sf:,...], corrupted_frames, axis = 0)
        quat = np.delete(quat[sf:,...], corrupted_frames, axis = 0) 

        return {'acc': acc, 'gyro': gyro, 'mag': mag, 'quat': quat}       

    
    def trim_data(self, data, ef):
        """ 
        Trim acc/gyro/mag/quat to according length.

        Args:
        :param data: Input data to trim. (F, T, :)
        :param ef: End frame of sequence. 
        """
        for i in ['acc', 'gyro', 'mag', 'quat']:
            data[i] = data[i][:ef,:,:]
        return data
    
    def save_to_disk(self, file, folder = '', **kwargs):
        """ Save acc/gyro/mag/quat to disk in a npz-file. """
        path = self.path + f'/{folder}{file}_{self.subject_nr}_imu'

        np.savez_compressed(path, **kwargs) 
        print(f'IMU:   SonE_{self.subject_nr} {file} has been converted.')



class ForwardKinematics(object):
    """
    FK Engine.
    """
    def __init__(self, offsets, parents, left_mult=False, major_joints=None, norm_idx=None, no_root=True):
        self.offsets = offsets
        if norm_idx is not None:
            self.offsets = self.offsets / np.linalg.norm(self.offsets[norm_idx])
        self.parents = parents
        self.n_joints = len(parents)
        self.major_joints = major_joints
        self.left_mult = left_mult
        self.no_root = no_root
        assert self.offsets.shape[0] == self.n_joints
        
    def fk(self, joint_angles, root = None):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            joint_angles: np array of shape (N, n_joints*3*3)

        Returns:
            The 3D joint positions as a an array of shape (N, n_joints, 3)
        """
        # assert joint_angles.shape[-1] == self.n_joints * 9
        angles = np.reshape(joint_angles, [-1, self.n_joints, 3, 3])
        n_frames = angles.shape[0]
        positions = np.zeros([n_frames, self.n_joints, 3])
        rotations = np.zeros([n_frames, self.n_joints, 3, 3])  # intermediate storage of global rotation matrices
        if self.left_mult:
            offsets = self.offsets[np.newaxis, np.newaxis, ...]  # (1, 1, n_joints, 3)
        else:
            offsets = self.offsets[np.newaxis, ..., np.newaxis]  # (1, n_joints, 3, 1)

                   

        for j in range(self.n_joints):
            if self.parents[j] == -1:
                # this is the root, we don't consider any root translation
                if root is not None:
                    positions[:,j] = root
                else:
                    positions[:, j] = 0.0                    
                rotations[:, j] = angles[:, j]
            else:
                # this is a regular joint
                if self.left_mult:
                    positions[:, j] = np.squeeze(np.matmul(offsets[:, :, j], rotations[:, self.parents[j]])) + \
                                        positions[:, self.parents[j]]
                    rotations[:, j] = np.matmul(angles[:, j], rotations[:, self.parents[j]])
                else:
                    positions[:, j] = np.squeeze(np.matmul(rotations[:, self.parents[j]], offsets[:, j])) + \
                                        positions[:, self.parents[j]]
                    rotations[:, j] = np.matmul(rotations[:, self.parents[j]], angles[:, j])

        return positions
    


class BvhFile():
    """
    Import positions, rotations and local rotations from bvh.
    Extract skeletal information from bvh.
    Trim data to match IMU and adjust the start frame accordingly.
    Save synced data to disk in npz-file.
    """

    def __init__(self, path, subject_nr, file):
        """ 
        Initializer:
        :param path: Path to the subject
        :param subject_nr: Number of Subject (SonE)
        :param file: Filename
        """
        self.path = path
        self.subject_nr = subject_nr
        self.file = file

        self.bvh = self.load_bvh_file(self.file)


    def load_bvh_file(self, file):
        """ 
        Read bvh file. 
        :param file: Filename.
        """
        bvh = Bvh()
        path = self.path + f'bvh/{file}.bvh'
        bvh.parse_string(open(path).read())
        return bvh
    

    def extract_motion(self):
        """
        Returns:
            positions: (N, n_joints, 3)
            local_rotations: (N, n_joints, 3, 3)
        """
        positions, rotations, local_rotations = self.bvh.all_frame_poses()

        # White trial is split into two parts
        if self.file == 'White':
            bvh_white2 = self.load_bvh_file('White2')
            positions_w2, rototations_w2, local_rotations_w2 = bvh_white2.all_frame_poses()

            positions = np.concatenate((positions, positions_w2), axis=0)
            local_rotations = np.concatenate((local_rotations, local_rotations_w2), axis=0)

        positions = positions * INCH / 100

        self.positions = positions
        self.local_rotations = local_rotations

    
    def synchronize_motion(self, sf):
        """ 
        Adjust start frame of bvh data. 
        :param sf: Starting frame for trial.
        """
        self.positions = self.positions[sf:,:,:]
        self.local_rotations = self.local_rotations[sf:,:,:,:]

        self.nframes = self.positions.shape[0]


    def save_to_disk(self):
        """ Save positions and local_rotations to npz-file. """
        path = self.path + f'/fk/{self.file}_{self.subject_nr}'
        np.savez_compressed(path + '_position', 
                            position = self.positions, frequency = 100)

        np.savez_compressed(path + '_joint_rotation',
                             joint_rotations = self.local_rotations, frequency = 100)

        print(f'BVH:   SonE_{self.subject_nr} {self.file} has been converted.')

    
    def get_skeletal_information(self):
        """ Extract offset and parent information from bvh file. """
        joints = list(self.bvh.joints.values())
        parents = []
        offsets = []

        for joint in joints:
            offsets.append(joint.offset)
            
            if joint.parent is None:
                parents.append(-1)
                continue

            parents.append(joints.index(joint.parent))
        
        self.parents = parents
        # Change units from [inch] to [m]
        self.offsets = np.array(offsets) * INCH / 100
    

    def save_skeleton(self):
        """ Save Skeleton to disk. """
        self.get_skeletal_information()

        np.savez_compressed(self.path + f'/fk/SonE_{self.subject_nr}_skeleton.npz',
                            offset = self.offsets, parent = self.parents)
        print(f'BVH:   SonE_{self.subject_nr} Skeleton has been converted.')

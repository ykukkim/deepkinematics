""" Visualisation of IMU orientaton, VICON Orientation, and VICON Markers on AITVIEWER """
# 1  Orienation of Vicon segments"
# 2  Theoratical IMU Orienation using Vicon segments of subject_1
# 3  Orientation of IMUs raw orientation
# 4  Orientation of IMUS with corrected left ankle movement
# 5  Orientation of IMUS with corrected left ankle movement alignned to the Vicon segment orientation by calculating the offset
import numpy as np
from DK00_Utils import DK00_UT00_config as config_v

from scipy.spatial.transform import Rotation as R
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.lines import Lines
from aitviewer.viewer import Viewer
from aitviewer.utils.so3 import aa2rot_numpy
from DK00_Utils.DK00_UT00_utils import load_vicon_data, load_IMU_quat_and_transform_to_rot_matrix, get_coords_for_lines, \
    calculate_offset_and_compute_alignment, get_coords_for_IMU_position


if __name__ == "__main__":

    for subject in [44]:#range(0,76):
        if subject in [5,38,39,52,55,57,58,60,68]:
            # These subjects are skipped due to bad walking, missing VICON markers
            # or missing IMU synchronization
            continue

        subject_nr = str(subject).zfill(2)
        subject_nr_2 = str(subject+1).zfill(2)

        print('Subject: ', subject_nr)
        print('Subject: ', subject_nr_2)

        # Define start frame for visualization
        start_frame_imu, start_frame_vicon = 0, 0

        for file in config_v.trials:

            # Create path to the vicon and IMU npz-files of subject x
            path_Vicon_1 = f'{config_v.dir_path}/SonE_{subject_nr}/{file}_{subject_nr}_vicon.npz'
            path_IMU_1 = f'{config_v.dir_path}/SonE_{subject_nr}/{file}_{subject_nr}_imu.npz'

            path_Vicon_2 = f'{config_v.dir_path}/SonE_{subject_nr}/{file}_{subject_nr}_vicon.npz'
            path_IMU_2 = f'{config_v.dir_path}/SonE_{subject_nr}/{file}_{subject_nr}_imu.npz'

            """ Load VICON and IMU data of subjects"""
            data_vicon_jc_s1, data_vicon_ori_s1 = load_vicon_data(path_Vicon_1)
            data_vicon_jc_s2, data_vicon_ori_s2 = load_vicon_data(path_Vicon_2)

            ori_imu_s1 = load_IMU_quat_and_transform_to_rot_matrix(path_IMU_1)
            ori_imu_s2 = load_IMU_quat_and_transform_to_rot_matrix(path_IMU_2)

            """ Adjust the Vicon segment orientation to the theoretical orientation of the IMUs """
            data_vicon_ori_s1_adjusted = np.copy(data_vicon_ori_s1)

            # Left & Right Foot
            data_vicon_ori_s1_adjusted[:,0:2,...] = np.matmul(data_vicon_ori_s1_adjusted[:,0:2,...], R.from_euler('x', 180, degrees = True).as_matrix())
            data_vicon_ori_s1_adjusted[:,0:2,...] = np.matmul(data_vicon_ori_s1_adjusted[:,0:2,...], R.from_euler('y', -30, degrees = True).as_matrix())
            # Left & Right Arm
            data_vicon_ori_s1_adjusted[:,2:4,...] = np.matmul(data_vicon_ori_s1_adjusted[:,2:4,...], R.from_euler('z', -90, degrees = True).as_matrix())
            data_vicon_ori_s1_adjusted[:,2:3,...] = np.matmul(data_vicon_ori_s1_adjusted[:,2:3,...], R.from_euler('x', -180, degrees = True).as_matrix())
            # Trunk
            data_vicon_ori_s1_adjusted[:,5,...] = np.matmul(data_vicon_ori_s1_adjusted[:,5,...], R.from_euler('x', 180, degrees = True).as_matrix())
            data_vicon_ori_s1_adjusted[:,5,...] = np.matmul(data_vicon_ori_s1_adjusted[:,5,...], R.from_euler('y', 90, degrees = True).as_matrix())
            # Head
            data_vicon_ori_s1_adjusted[:,4,...] = np.matmul(data_vicon_ori_s1_adjusted[:,4,...], R.from_euler('y', 90, degrees = True).as_matrix())
            # Calculate drift (offset) for first 10 frames and remove offset
            # ori_imu_s1_offset = calculate_offset_and_compute_alignment(ori_imu_s1, data_vicon_ori_s1)
            # ori_imu_s2_offset = calculate_offset_and_compute_alignment(ori_imu_s2, data_vicon_ori_s2)

            """ Correct rotation of the IMU Ori for subject_1 """
            "Some participant's left foot IMU sensor has the complete opossition direction"
            ori_imu_s1_corrected_ankle_l = np.copy(ori_imu_s1)
            ori_imu_s1_offset = np.copy(ori_imu_s1)
            ori_imu_s1_corrected_ankle_l[:,0,...] = np.matmul([[-1,0,0],[0,-1,0],[0,0,1]], ori_imu_s1_corrected_ankle_l[:,0,...])      # left foot, rotate 180 aroud z-axis

            """ Correct rotation of the IMU Ori and aligned by to the vicon segmenet computing offset for subject_1 """
            # Compute offset of IMU orientation in relation to Vicon segment orientation for first 10 frames and remove it for the entire sequence
            ori_imu_s1_offset = calculate_offset_and_compute_alignment(ori_imu_s1_corrected_ankle_l, data_vicon_ori_s1_adjusted)

            "Creating Figures"
            # Get endpoints of lines to assemble stick figure in aitviewer
            subject_1_pts = get_coords_for_lines(data_vicon_jc_s1)
            subject_2_pts = get_coords_for_lines(data_vicon_jc_s2)

            # Get coordinates of the joints where the IMU sensors are positioned
            coords_IMU_s1 = get_coords_for_IMU_position(data_vicon_jc_s1)
            coords_IMU_s2 = get_coords_for_IMU_position(data_vicon_jc_s2)

            # # Joint Centers modeled to spheres
            # spheres = Spheres(data_vicon_jc_s1, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi), color = (2,0,0,1), radius = 0.02)
            # spheres.rotation = spheres.rotation

            s1_lines = Lines(subject_1_pts, mode = "lines", position=np.array([0.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s1_lines.rotation = s1_lines.rotation

            s1_lines_Viconseg = Lines(subject_1_pts, mode = "lines", position=np.array([1.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s1_lines_Viconseg.rotation = s1_lines_Viconseg.rotation

            s1_lines_rawImU = Lines(subject_1_pts, mode = "lines", position=np.array([2.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s1_lines_rawImU.rotation = s1_lines_rawImU.rotation

            s1_lines_ankleL_corrected = Lines(subject_1_pts, mode = "lines", position=np.array([3.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s1_lines_ankleL_corrected.rotation = s1_lines_ankleL_corrected.rotation

            s1_lines_AdjustedImU = Lines(subject_1_pts, mode = "lines", position=np.array([4.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s1_lines_AdjustedImU.rotation = s1_lines_AdjustedImU.rotation

            # Orienation of Vicon segments of subject_1
            rigid_body_segment_ori_s1 = RigidBodies(coords_IMU_s1, data_vicon_ori_s1, position=np.array([0.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            rigid_body_segment_ori_s1.rotation = rigid_body_segment_ori_s1.rotation

            # Theoratical IMU Orienation using Vicon segments of subject_1
            rigid_body_adjusted_segment_ori_s1 = RigidBodies(coords_IMU_s1, data_vicon_ori_s1_adjusted, position=np.array([1.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            rigid_body_adjusted_segment_ori_s1.rotation = rigid_body_adjusted_segment_ori_s1.rotation

            # Orientation of IMUs raw orientation
            rigid_body_raw_IMU_rot_s1 = RigidBodies(coords_IMU_s1, ori_imu_s1, position=np.array([2.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            rigid_body_raw_IMU_rot_s1.rotation = rigid_body_raw_IMU_rot_s1.rotation

            # Orientation of IMUS with corrected left ankle movement
            rigid_body_s1_IMU_ankle_corrected = RigidBodies(coords_IMU_s1,ori_imu_s1_corrected_ankle_l, position=np.array([3.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            rigid_body_s1_IMU_ankle_corrected.rotation = rigid_body_s1_IMU_ankle_corrected.rotation

            # Orientation of IMUS with corrected left ankle movement alignned to the Vicon segment orientation by calculating the offset
            rigid_body_s1_aligned = RigidBodies(coords_IMU_s1,ori_imu_s1_offset, position=np.array([4.0, 0.0, 0.0]), rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            rigid_body_s1_aligned.rotation = rigid_body_s1_aligned.rotation

            v = Viewer()
            v.scene.add(s1_lines, s1_lines_Viconseg, rigid_body_segment_ori_s1, rigid_body_adjusted_segment_ori_s1)
            v.scene.add(s1_lines_rawImU, s1_lines_ankleL_corrected, rigid_body_raw_IMU_rot_s1,rigid_body_s1_IMU_ankle_corrected)
            v.scene.add(s1_lines_AdjustedImU, rigid_body_s1_aligned)
            v.run()


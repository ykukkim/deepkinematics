""" Calculate the VICON segment Orientation and adjust it to the theoretical IMU Orientation """

import conversion_functions as cf
import numpy as np
from scipy.spatial.transform import Rotation as R

# labels of segments where IMUs are mounted.
# last letter of label describes what marker is presented
# O == origin, A == x-axis, L == y-axis, P == z-axis
labels_segments = ['LFOO', 'LFOA', 'LFOL', 'LFOP', 'RFOO', 'RFOA', 'RFOL', 'RFOP', 'LLoArmO', 'LLoArmA', 'LLoArmL', 'LLoArmP', 
'RLoArmO', 'RLoArmA', 'RLoArmL', 'RLoArmP', 'HeadO', 'HeadA', 'HeadL', 'HeadP', 'PELO', 'PELA', 'PELL', 'PELP']

def calculate_segment_orientation(path):
    """ Calculate VICON orientation of IMU segments and rotate into correct orientation. """
    data_c3d, labels_c3d = cf.import_data_from_c3d(path)
    index_ori = cf.get_index_of_labels(labels_c3d, labels_segments)
    data_ori = cf.get_coords_of_markers(data_c3d, index_ori)
    segment_oris = cf.compute_orientation_of_segments(data_ori)

    """ Rotate segment orientations into original orientation of IMUs """
    """ Feet """
    segment_oris[:,0:2,...] = np.matmul(segment_oris[:,0:2,...], R.from_euler('y', -90, degrees = True).as_matrix())
    segment_oris[:,0:2,...] = np.matmul(segment_oris[:,0:2,...], R.from_euler('z', 180, degrees = True).as_matrix())

    """ Left & Right Arm """
    segment_oris[:,2:4,...] = np.matmul(segment_oris[:,2:4,...], R.from_euler('z', -90, degrees = True).as_matrix())
    """ Left Arm """
    segment_oris[:,2:3,...] = np.matmul(segment_oris[:,2:3,...], R.from_euler('x', -180, degrees = True).as_matrix())

    """ Head """
    segment_oris[:,4,...] = np.matmul(segment_oris[:,4,...], R.from_euler('y', 90, degrees = True).as_matrix())

    """ Trunk """
    segment_oris[:,5,...] = np.matmul(segment_oris[:,5,...], R.from_euler('x', 180, degrees = True).as_matrix())
    segment_oris[:,5,...] = np.matmul(segment_oris[:,5,...], R.from_euler('y', 90, degrees = True).as_matrix())

    return segment_oris
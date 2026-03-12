import numpy as np
from scipy.spatial.transform import Rotation as R


def euler_angles_to_rot(data):
    """ Convert Euler Angles from VICON back to rotation matrices. """
    data = data * 1000      # due to marker unit conversion from [mm] to [m] angles are affected as well
                            # multiplying by 1000 reverts this mistake

    # change euler angle direction from clockwise to anti-clockwise
    data[:,0,2] = data[:,0,2] * -1      # Pelvis (root)
    data[:,1,:] = data[:,1,:] * -1      # Left Hip
    data[:,2,0] = data[:,2,0] * -1      # Right Hip
    data[:,3,1] = data[:,3,1] * -1      # Left Knee
    data[:,3,2] = data[:,3,2] * -1      # Left Knee
    data[:,5,:] = data[:,5,:] * -1      # Left Ankle
    data[:,6,0] = data[:,6,0] * -1      # Right Ankle
    
    pelvis = R.from_euler('YXZ', data[:,0,:], degrees = True).as_matrix()
    LHip = R.from_euler('YXZ', data[:,1,:], degrees = True).as_matrix()
    RHip = R.from_euler('YXZ', data[:,2,:], degrees = True).as_matrix()
    LKnee = R.from_euler('YXZ', data[:,3,:], degrees = True).as_matrix()
    RKnee = R.from_euler('YXZ', data[:,4,:], degrees = True).as_matrix()
    LAnkle = R.from_euler('YZX', data[:,5,:], degrees = True).as_matrix()
    RAnkle = R.from_euler('YZX', data[:,6,:], degrees = True).as_matrix()

    ori = np.stack((pelvis, LHip, RHip, LKnee, RKnee, LAnkle, RAnkle), axis = 1)
    return ori
""" Visualisation of IMU orientation on VICON data """
"""  Remove root movement if desired  to see effect of the root"""
"Root removal will shift the figure completely."

import numpy as np

from DK00_Utils import DK00_UT00_config as config_v
from DK00_Utils.DK00_UT00_utilsVisualisation import ViconVisualization

from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.lines import Lines
from aitviewer.viewer import Viewer
from aitviewer.utils.so3 import aa2rot_numpy

#  R G B A, A = intesnity
# Red 1 0 0 0
# Green 0 1 0 1
# Blue  0 0 1 1
# ML AP VT

if __name__ == "__main__":

    for subject in [75]:#range(0,76):
        if subject in [5,38,39,52,55,57,58,60,68]:
            # These subjects are skipped due to bad walking, missing VICON markers
            # or missing IMU synchronization
            continue
        subject_nr = str(subject).zfill(2)
        print('Subject: ', subject_nr)
        for file in config_v.trials:
            subject_one = ViconVisualization(subject_nr, file, config_v.dir_path)

            spheres = Spheres(subject_one.joint_center, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi),
                              color=(0, 0, 1, 1), radius=0.02)
            spheres.rotation = spheres.rotation

            s1_lines = Lines(subject_one.lines, mode="lines", color=(1, 0, 0, 1),
                             rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s1_lines.rotation = s1_lines.rotation

            # replace imu_ori -> imu_ori_aligned: Align IMU_ori to Vicon_ori for the entire sequence
            s1_segment_ori = RigidBodies(subject_one.imu_pos, subject_one.imu_ori,
                                         position=np.array([0, 0, 0]),
                                         rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s1_segment_ori.rotation = s1_segment_ori.rotation

            "Removing Root Movement -- only if intrested in seeing the effect of root removal" \
            "Root removal will shift the figure completely."
            subject_one.remove_root_movement()

            spheres2 = Spheres(subject_one.joint_center, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi),
                               color=(0, 0, 1, 1), radius=0.02)
            spheres2.rotation = spheres2.rotation

            s2_lines = Lines(subject_one.lines, mode="lines", color=(0, 1, 0, 1),
                             rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s2_lines.rotation = s2_lines.rotation

            s2_segment_ori = RigidBodies(subject_one.imu_pos, subject_one.imu_ori_aligned,
                                         rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s2_segment_ori.rotation = s2_segment_ori.rotation


            args = (spheres, s1_lines, s1_segment_ori, spheres2,s2_lines,s2_segment_ori)
            v = Viewer()
            v.scene.add(*args)
            v.run()

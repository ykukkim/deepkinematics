import numpy as np

from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.lines import Lines
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer

from DK00_Utils.DK00_UT00_utilsVisualisation import BvhVisualization
from DK00_Utils.DK00_UT00_utilsVisualisation import ViconVisualization
from DK00_Utils import DK00_UT00_config as config_v



if __name__ == "__main__":

    for subject in [75]:#range(0,76):
        if subject in [5,38,39,52,55,57,58,60,68]:
            # These subjects are skipped due to bad walking, missing VICON markers
            # or missing IMU synchronization
            continue
        subject_nr = str(subject).zfill(2)
        print('Subject: ', subject_nr)

        for file in config_v.trials:

            bvh = BvhVisualization(subject_nr, file, config_v.dir_path)
            vicon = ViconVisualization(subject_nr, file, config_v.dir_path, freq=100)

            # BVH
            skeleton = Skeletons(bvh.positions, bvh.connections,
                                rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi),
                                position=np.array([0.0, 0.0, 0.0]))
            skeleton.rotation = skeleton.rotation

            # BVH
            skeleton_local = Skeletons(bvh.local_positions, bvh.connections,
                                    rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi),
                                    position=np.array([1.0, 0.0, 0.0]))
            skeleton_local.rotation = skeleton_local.rotation

            rigid_body_imu = RigidBodies(bvh.imu_pos, bvh.imu_ori,
                                        position=np.array([0.0, 0.0, 0.0]),
                                        rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            rigid_body_imu.rotation = rigid_body_imu.rotation

            # Vicon
            spheres = Spheres(vicon.joint_center, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi),
                              position=np.array([-1.0, 0.0, 0.0]),
                              color = (2,0,0,1), radius=0.02)
            spheres.rotation = spheres.rotation

            s1_lines = Lines(vicon.lines, mode = "lines",
                            position=np.array([-1.0, 0.0, 0.0]),
                            rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            s1_lines.rotation = s1_lines.rotation

            rigid_body_vicon = RigidBodies(vicon.imu_pos, vicon.imu_ori,
                                        position=np.array([-1.0, 0.0, 1.0]),
                                        rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))
            rigid_body_vicon.rotation = rigid_body_vicon.rotation

            v = Viewer()
            v.scene.add(skeleton, spheres, spheres, s1_lines)
            v.run()

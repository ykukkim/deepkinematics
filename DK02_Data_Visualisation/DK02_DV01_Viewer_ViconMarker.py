""" Visualisation of VICON Markers from c3d on AITVIEWER """
"All the markers from VICON are shown as spheres"

import numpy as np
from DK00_Utils import DK00_UT00_config as config_v

from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer
from aitviewer.utils.so3 import aa2rot_numpy
from ezc3d import c3d

if __name__ == "__main__":

    for subject in [4]:#range(0,76):
        if subject in [5,38,39,52,55,57,58,60,68]:
            # These subjects are skipped due to bad walking, missing VICON markers
            # or missing IMU synchronization
            continue
        subject_nr = str(subject).zfill(2)
        print('Subject: ', subject_nr)

        for file in config_v.trials:
            # c = c3d(f'{config_v.dir_path}\\SonE_{subject_nr}\\{file}.c3d')
            c = c3d('/Users/yonkim/DataForWork/DeepKinematics/Training Data/SonE_04/Norm_Pre.c3d')
            markers = c['data']['points'].transpose()
            label_list = list(c['parameters']['POINT']['LABELS']['value'])

            index = []
            for label in config_v.marker_labels:
                index.append(label_list.index(label))

            vicon_markers = markers[:, index, :3] / 1000

            spheres = Spheres(vicon_markers, rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi),  color = (2,0,0,1), radius = 0.02)

            v = Viewer()
            v.scene.add(spheres)
            v.run()
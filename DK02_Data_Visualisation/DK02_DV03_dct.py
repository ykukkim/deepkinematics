import numpy as np

from aitviewer.renderables.spheres import Spheres
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer

from DK00_Utils.DK00_UT00_utilsVisualisation import BvhVisualization
from DK00_Utils import DK00_UT00_config as config_v

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


if __name__ == "__main__":

    config_v.dir_path = 'D:\\04_DeepKinematics\\Training Data'

    for subject in [44]:  # range(0,76):
        if subject in [5, 38, 39, 52, 55, 57, 58, 60, 68]:
            # These subjects are skipped due to bad walking, missing VICON markers
            # or missing IMU synchronization
            continue
        subject_nr = str(subject).zfill(2)
        print('Subject: ', subject_nr)

        for file in config_v.trials:

            bvh = BvhVisualization(subject_nr, file, config_v.dir_path)

            # coeff, f = dct(bvh.local_positions[:50,...])

            # joints = idct(coeff, f)

            n_dct = 20

            joints = bvh.local_positions

            dct_m, idct_m = get_dct_matrix(500)

            joints = joints[1000:1500,:].reshape(500, -1)

            src_val = np.matmul(dct_m[:n_dct], joints)  # (n_dct, 93)

            out = np.matmul(idct_m[:, :n_dct], src_val).reshape(-1, 31, 3) # (500, 93)

            # eucl_dist = []
            # for i in range(1,21):
            #     n_dct = int(250 * (i / 20))
            #     src_val = np.matmul(dct_m[:n_dct], joints)

            #     out = np.matmul(idct_m[:, :n_dct], src_val).reshape(-1, 31, 3)

            #     eucl_mean = np.mean(np.mean(np.sqrt(np.sum((out - bvh.local_positions[1000:1250])**2, axis=-1)), axis=0))
            #     eucl_dist.append(eucl_mean)

            # f1 = plt.figure()
            # plt.title('Euclidean distance - Percentage of DCT parameter')
            # plt.xlabel('[%]')
            # plt.ylabel('Euclidean Distance [m]')
            # plt.plot(np.arange(5,101,5), eucl_dist)
            # plt.grid(True)
            # plt.show()

            # eucl_mean_per_joint = np.mean(np.sqrt(np.sum((out - bvh.local_positions[1000:1500])**2, axis=-1)), axis=0)
            # print(np.mean(eucl_mean_per_joint, axis=0))

            # print(eucl_mean_per_joint[[23, 24, 28, 29]])

            spheres = Spheres(out,
                              rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi))

            spheres_gt = Spheres(bvh.local_positions[1000:1500],
                                 rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi),
                                 position = [1.0, 0.0, 0.0],
                                 color=np.array([0,1,0,1]))

            v = Viewer()
            v.scene.add(spheres, spheres_gt)
            v.run()

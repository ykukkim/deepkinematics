"""Display the configured Vicon markers from one or more C3D files."""

import argparse
from pathlib import Path

import numpy as np
from aitviewer.renderables.spheres import Spheres
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer
from ezc3d import c3d

from DK00_Utils import DK00_UT00_config as config_v


def load_markers(path):
    """Load the configured marker subset in metres for AITViewer."""
    recording = c3d(str(path))
    markers = recording["data"]["points"].transpose()
    labels = list(recording["parameters"]["POINT"]["LABELS"]["value"])
    missing = [label for label in config_v.marker_labels if label not in labels]
    if missing:
        raise ValueError(f"{path} is missing configured markers: {', '.join(missing)}")
    indices = [labels.index(label) for label in config_v.marker_labels]
    return markers[:, indices, :3] / 1000.0


def main():
    """Open an interactive viewer for each requested C3D recording."""
    default_file = Path(config_v.dir_path) / "SonE_04" / "Norm_Pre.c3d"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c3d", nargs="*", type=Path, default=[default_file])
    args = parser.parse_args()

    for path in args.c3d:
        print(f"Viewing {path}")
        vicon_markers = load_markers(path)
        spheres = Spheres(
            vicon_markers,
            rotation=aa2rot_numpy(np.array([-0.5, 0, 0]) * np.pi),
            color=(1, 0, 0, 1),
            radius=0.02,
        )
        viewer = Viewer()
        viewer.scene.add(spheres)
        viewer.run()


if __name__ == "__main__":
    main()

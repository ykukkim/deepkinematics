"""Compatibility exports for the canonical joint-kinematics implementation.

Historically this file was a complete copy of
``DK04_Gait_Parameter/DK04_Gait08_joint_kinematics.py``. Re-exporting the
canonical module keeps existing figure scripts working while preventing the two
statistical implementations from drifting apart.
"""

import importlib
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

_implementation = importlib.import_module(
    "DK04_Gait_Parameter.DK04_Gait08_joint_kinematics"
)

# Include private helpers as well as public functions because older local figure
# scripts may have imported either. Dunder attributes remain local to this shim.
for _name, _value in vars(_implementation).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

__all__ = [name for name in vars(_implementation) if not name.startswith("_")]


if __name__ == "__main__":
    _implementation.main()

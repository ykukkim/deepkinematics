"""Expose the analysis modules' synthetic checks through pytest."""

import numpy as np

from DK04_Gait_Parameter.DK04_Gait03_agreement import selftest as agreement_selftest
from DK04_Gait_Parameter.DK04_Gait08_joint_kinematics import (
    selftest as joint_kinematics_selftest,
)
from DK04_Gait_Parameter.DK04_Gait09_gyro_event_detection import gyro_ic, match_events
from DK04_Gait_Parameter.DK04_Gait11_calibration_decomposition import (
    selftest as calibration_selftest,
)
from DK04_Gait_Parameter.DK04_Gait12_joint_rom_decomposition import (
    selftest as joint_rom_selftest,
)


def test_agreement_statistics():
    """Check ICC, bias, and concordance on known synthetic values."""
    agreement_selftest()


def test_joint_kinematics_statistics():
    """Check rotations, waveform errors, and SPM behaviour."""
    joint_kinematics_selftest()


def test_calibration_decomposition():
    """Check between- and within-participant slope recovery."""
    calibration_selftest()


def test_joint_rom_decomposition():
    """Check the joint-ROM agreement statistics."""
    joint_rom_selftest()


def test_event_matching_is_one_to_one():
    """A single IMU event cannot satisfy two neighbouring OMC events."""
    matches, errors = match_events([100, 102], [101], tolerance=3)
    assert matches == 1
    assert len(errors) == 1


def test_event_detector_handles_flat_signal():
    """A flat gyroscope channel should return no contacts, not divide by zero."""
    assert gyro_ic(np.zeros(1000)).size == 0

#!/usr/bin/env python3
"""Validate foot-gyroscope initial contacts against OMC heel strikes.

The input NPZ must contain ``<subject>|gL/gR`` gyroscope signals and
``<subject>|hsL/hsR`` OMC heel-strike indices. Matching is one-to-one within
each requested tolerance, so sensitivity and positive predictive value are not
inflated by reusing a detected event.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew


DEFAULT_SUBJECTS = ["04", "09", "14", "24", "51", "54", "67", "81"]


def gyro_ic(signal, sampling_frequency=200.0, orientation="skew"):
    """Detect initial contacts after each positive mid-swing gyroscope peak."""
    b, a = butter(4, 12 / (sampling_frequency / 2), "low")
    filtered = filtfilt(b, a, np.asarray(signal, dtype=float))
    if orientation == "skew":
        if skew(filtered) < 0:
            filtered = -filtered  # orient the broad mid-swing peak positively
    elif orientation == "magnitude":
        if abs(filtered.min()) > abs(filtered.max()):
            filtered = -filtered
    else:
        raise ValueError("orientation must be 'skew' or 'magnitude'")

    scale = np.max(np.abs(filtered))
    if not np.isfinite(scale) or scale == 0:
        return np.array([], dtype=int)
    normalised = filtered / scale
    swing_peaks, _ = find_peaks(
        normalised, height=0.4, distance=int(0.7 * sampling_frequency)
    )

    contacts = []
    for index, peak in enumerate(swing_peaks):
        end = (
            swing_peaks[index + 1]
            if index + 1 < len(swing_peaks)
            else min(len(normalised), peak + int(1.3 * sampling_frequency))
        )
        segment = normalised[peak:end]
        if len(segment) < int(0.3 * sampling_frequency):
            continue
        negative_peak = peak + int(np.argmin(segment))
        search = normalised[
            negative_peak:min(end, negative_peak + int(0.45 * sampling_frequency))
        ]
        if len(search) >= 3:
            contacts.append(negative_peak + int(np.argmax(search)))
    return np.asarray(sorted(set(contacts)), dtype=int)


def match_events(omc_events, imu_events, tolerance, sampling_frequency=200.0):
    """One-to-one nearest-neighbour matching and timing errors in milliseconds."""
    omc_events = np.asarray(omc_events, dtype=int)
    imu_events = np.asarray(imu_events, dtype=int)
    used = np.zeros(len(imu_events), dtype=bool)
    timing_errors = []
    matches = 0
    for event in omc_events:
        if len(imu_events) == 0:
            continue
        nearest = int(np.argmin(np.abs(imu_events - event)))
        if abs(imu_events[nearest] - event) <= tolerance and not used[nearest]:
            used[nearest] = True
            matches += 1
            timing_errors.append(
                (imu_events[nearest] - event) / sampling_frequency * 1000
            )
    return matches, timing_errors


def evaluate(data, subjects, tolerances_ms, sampling_frequency, discard_before, orientation):
    """Aggregate event-detection accuracy and print participant-level sensitivity."""
    aggregate = {tolerance: [0, 0, 0, []] for tolerance in tolerances_ms}
    participant_sensitivity = {}
    report_tolerance = tolerances_ms[len(tolerances_ms) // 2]

    for subject in subjects:
        row = {}
        for side in ("L", "R"):
            omc = data[f"{subject}|hs{side}"]
            detected = gyro_ic(
                data[f"{subject}|g{side}"], sampling_frequency, orientation
            )
            detected = detected[detected > int(discard_before * sampling_frequency)]
            for tolerance_ms in tolerances_ms:
                tolerance_frames = int(round(tolerance_ms * sampling_frequency / 1000))
                matches, errors = match_events(
                    omc, detected, tolerance_frames, sampling_frequency
                )
                totals = aggregate[tolerance_ms]
                totals[0] += len(omc)
                totals[1] += len(detected)
                totals[2] += matches
                totals[3].extend(errors)
                if tolerance_ms == report_tolerance:
                    row[side] = matches / len(omc) if len(omc) else np.nan
        participant_sensitivity[subject] = row

    print(f"=== orientation={orientation} ===")
    for tolerance_ms, (n_omc, n_gyro, matches, errors) in aggregate.items():
        errors = np.asarray(errors, dtype=float)
        sensitivity = matches / n_omc if n_omc else np.nan
        precision = matches / n_gyro if n_gyro else np.nan
        bias = np.mean(errors) if len(errors) else np.nan
        mae = np.mean(np.abs(errors)) if len(errors) else np.nan
        print(
            f"  tolerance +/-{tolerance_ms:3d} ms: sensitivity {sensitivity:.3f} "
            f"PPV {precision:.3f} bias {bias:+.1f} ms MAE {mae:.1f} ms"
        )
    print(
        f"  per-participant sensitivity at {report_tolerance} ms:",
        {
            subject: {side: round(value, 3) for side, value in values.items()}
            for subject, values in participant_sensitivity.items()
        },
    )


def main():
    """Parse the validation inputs and report detector performance."""
    default_data = Path(__file__).resolve().parent / "data" / "_omc_gyro_extract.npz"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--tolerances-ms", nargs="+", type=int, default=[40, 60, 100])
    parser.add_argument("--sampling-frequency", type=float, default=200.0)
    parser.add_argument(
        "--discard-before", type=float, default=5.0,
        help="discard detections in the first N seconds (default: 5)",
    )
    parser.add_argument("--orientation", choices=["skew", "magnitude"], default="skew")
    args = parser.parse_args()

    with np.load(args.data) as data:
        evaluate(
            data,
            args.subjects,
            args.tolerances_ms,
            args.sampling_frequency,
            args.discard_before,
            args.orientation,
        )


if __name__ == "__main__":
    main()

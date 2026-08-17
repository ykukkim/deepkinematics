# DK04_Gait_Parameter — Analysis Pipeline (Python)

A from-scratch, `.mat`-free analysis pipeline that regenerates **every reading in the
paper directly from the trained models into CSV**. It replaces the old MATLAB
`GaitSummary` pipeline after a data-integrity issue (a corrupted `GaitSummary_DL_ATT.mat`
that silently held BiLSTM's reconstruction). Every stage is transparent and auditable:
raw IMU → model → metrics → CSV, with nothing hidden in a `.mat`.

## Scripts (pipeline order, not numeric order)

Run in this order. The `GaitXX` numbers are kept from the original layout, so they are
**not** monotonic with run order — follow the arrows, not the numbers.

| Run | Script | Output |
|-----|--------|--------|
| 1. Model inference | `DK04_Gait07_run_inference.py` | `pred_<subj>.npz` per model |
| 2. Reconstruction error (MPJAE, MPJPE) | `DK04_Gait02_reconstruction.py` | `reconstruction_metrics.csv` |
| 3. Joint kinematics (hip/knee/ankle waveforms) | `DK04_Gait08_joint_kinematics.py` | `joint_kinematics_*.csv` |
| 4. Spatio-temporal (step/stride length & width) | `DK04_Gait01_spatiotemporal.py` | `spatiotemporal_perstride.csv` |
| 5. Agreement (Bland–Altman, ICC, CCC) | `DK04_Gait03_agreement.py` | `agreement_summary.csv` |

Reporting helpers kept from the original folder (unchanged, run after the pipeline):
`DK04_Gait05_Mean_into_Excel_R_L.py`, `DK04_Gait06_FindingBestModel.py` /
`DK04_Gait06_FindingBestModel_Euclidean.py`, and `Functions_py/bland_altman_plots.py`.

The MATLAB originals these Python scripts replace live in `_Archive/`.

## The one MATLAB step (kept deliberately)

**Gait-event detection** (`_Archive/Functions/StepDetection/StepDetection_DL.m` →
`EventDetection_legs_DL.m`) finds heel-strike / toe-off from the **foot IMU gyroscope**.
These events do **not** depend on any model — the same events apply to OMC and every
model — so they are a fixed upstream *input*, not a result. It stays in MATLAB because it
is a wavelet-based detector (Wavelet Toolbox `centfrq`/`cwt`/`thd`) that does not port to
Python reliably. Run it once per trial and export the events (see below).

## Data flow

```
                    ┌─ (MATLAB, once) StepDetection_DL ─→ events.npz/csv ─┐
raw IMU + OMC ─→ Gait07_run_inference ─→ pred_<subj>.npz ─┬─→ Gait02 reconstruction ─→ CSV
                                                          ├─→ Gait08 kinematics      ─→ CSV
                                                          ├─→ Gait01 spatio-temporal ←─┘ (needs events) ─→ CSV
                                                          └─→ Gait03 agreement (on Gait01's per-stride CSV) ─→ CSV
```

## Events input format

Stage 4 (spatio-temporal) consumes gait events as an `.npz` with, per subject, 1-based
frame indices (MATLAB convention): `"<subj>|HSleft"`, `"<subj>|HSright"`,
`"<subj>|TOleft"`, `"<subj>|TOright"`. Export these once from your MATLAB `GaitEvents`
struct; the Python side then never touches `.mat` again.

Note: stage 3 (joint kinematics) does **not** use these events — it detects the gait
cycle directly from the knee kinematics (`--segmentation kinematic`, the default),
because the stored foot-gyro events are not synchronised to the reconstructed joint
arrays. Stage 4 (spatio-temporal), which works on foot **positions**, does use them.

## Validation status

- **Spatio-temporal (Gait01):** ported from `GaitWork_Treadmill` / `strLength_Treadmill`
  / `stepwidth` / `f_approxVelocity_treadmill`; reproduces the MATLAB gait parameters to
  **≤ 0.01 cm** on the GT trial (SonE_04/09/14).
- **Reconstruction (Gait02):** MPJAE cross-subject reproduces the manuscript (ATT ≈ 12.3°
  vs 12.26° on the 7-subject check; GT correlation 1.000 throughout).
- **Agreement (Gait03):** ICC validated against the Shrout & Fleiss benchmark (`--selftest`).
- **Kinematics (Gait08):** knee waveform recovers the clinical ~60° swing peak after the
  kinematic-segmentation fix.

## Requirements

`numpy scipy pandas matplotlib torch` (+ the repo's `DK00_Utils` on `PYTHONPATH` for
Gait07 inference). `spm1d` is optional (Gait08 SPM; falls back to a per-frame test if absent).

## Convention notes

- Joint index map (model `FK_JOINTS_FUll` → clinical): `LHJC=LeftUpLeg`, `LKJC=LeftLeg`,
  `LAJC=LeftFoot`, and the right-side equivalents.
- Sagittal flexion = `-rad2deg(rotm2eul(R,'XYZ')[0])`; the model's rotation matrices are
  already in the BVH/GaitSummary convention (no extra transpose from the model output).
- Marker axes: col 0 = X (medio-lateral), col 1 = Y (anterior-posterior / treadmill
  travel), col 2 = Z (vertical); positions in mm, gait parameters in cm.
- Age groups: young = subject ID 1–50, older = 51–87.

## Manuscript update — Python gait events + FVA validation (Gait09)

The current manuscript detects gait events **once from the raw foot gyroscopes in
Python**, shared across OMC and all models, and validates them against the foot-velocity
algorithm (FVA, O'Connor 2007). This single script replaces the "one MATLAB step" noted
above and is the sole event detector behind the reported results:

| Script | Output |
|--------|--------|
| `DK04_Gait09_gyro_event_detection.py` | Detects initial contacts from the raw foot-gyro sagittal axis (4th-order zero-phase 12 Hz Butterworth, skewness sign, mid-swing peaks >40 % of the max-normalised signal @ 0.7 s spacing, IC = local maximum after the terminal-swing minimum, the reversal into stance) **and** validates them against the stored FVA reference — sensitivity 0.978, PPV 0.979, MAE 19.5 ms at ±60 ms (Table VII). Verified to reproduce these numbers exactly. This is the exact method described in Appendix A. |

> Note: an earlier alternative detector (`EventDetection_legs_DL` integration/derivative
> port) was found to be non-functional (0 % sensitivity) and was removed to
> `_to_delete/broken_detector_variant/` so the repository contains only the one detector
> that matches the manuscript.

`data/` holds the npz inputs used by these scripts and by the figure scripts in
`../DK05_Figures/`: `_omc_gyro_extract.npz` (OMC/FVA events + raw foot gyro),
`posfeet.npz` (pelvis-centred reconstructed & OMC positions), `footpitch.npz`,
and `flex_{BILSTM,ATT,DIFF}.npz` (per-model sagittal joint flexion).

Result tables and figures are rendered by `Plot_ResultTables.py`,
`Plot_ResultFigures.py`, and `Plot_BlandAltman_Spatial.py` in `../DK05_Figures/`.

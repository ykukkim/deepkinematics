# DeepKinematics

DeepKinematics reconstructs full-body kinematics from six wearable IMUs and
evaluates the reconstruction against optical motion capture (OMC). The active
code covers motion-capture conversion, PyTorch training and inference,
spatiotemporal gait analysis, joint-kinematics agreement, and publication
figures.

The repository also contains archived scripts and generated manuscript results.
Those are retained as provenance and are not part of the maintained pipeline.

## Supported environment

The maintained environment uses Python 3.11, PyTorch 2.x, and AITViewer 1.14.
Create it from the direct dependency list:

```bash
conda env create -f environment.yml
conda activate deepkinematics
```

The default PyTorch wheel supports CPU on all platforms and Metal (MPS) on
compatible Macs. For a CUDA cluster, install the wheel matching the cluster's
CUDA runtime using the [official PyTorch selector](https://pytorch.org/get-started/locally/)
after creating the environment.

The environment intentionally records only direct dependencies. This avoids the
machine-specific build packages and absolute Conda prefix produced by a full
environment export.

## Project paths

`DK00_Utils/DK00_UT00_config.py` derives paths from this checkout. No user or
cluster path is hard-coded. The following optional environment variables override
the defaults:

| Variable | Default |
| --- | --- |
| `DEEPKINEMATICS_ROOT` | project folder containing `00_Codes`, `00_Data`, and `00_Results` |
| `DEEPKINEMATICS_DATA_DIR` | `<root>/00_Data/Data` |
| `DEEPKINEMATICS_MODEL_DIR` | `<root>/00_Results` |
| `DEEPKINEMATICS_FINAL_MODEL_DIR` | `<root>/00_Data/Models` |
| `DEEPKINEMATICS_WANDB_DIR` | `<root>/00_Results/wandb` |
| `DEEPKINEMATICS_SWEEP_DIR` | `<root>/00_Results/sweeps` |
| `DEEPKINEMATICS_UNSEEN_MOTION_CSV` | `DK00_Utils/test_set_unseen.csv` |

For example:

```bash
export DEEPKINEMATICS_DATA_DIR=/cluster/project/example/TrainingData
export DEEPKINEMATICS_MODEL_DIR=/cluster/project/example/Models
```

## Maintained code map

| Folder | Purpose |
| --- | --- |
| `DK00_Utils` | Configuration, datasets, transforms, layers, models, losses, metrics, logging, and evaluation helpers |
| `DK01_Data_Conversion` | Convert C3D, BVH, and IMU MATLAB data to model-ready NumPy files |
| `DK02_Data_Visualisation` | Inspect OMC markers, orientations, skeletons, and joint angles |
| `DK03_Training` | Single-split and participant-stratified cross-validation training |
| `DK04_Gait_Parameter` | Inference, reconstruction metrics, gait parameters, agreement, calibration, and ROM decomposition |
| `DK05_Figures` | Publication-figure entry points; statistical helpers live in `DK04_Gait_Parameter` |

`DK03_Training/Archive` and the project-level `00_Codes/_to_delete` directory are
frozen historical material. Do not import from them in new analyses.

## Core workflow

Run commands from this `dl_humanmotion-main` folder so imports and relative output
locations are predictable.

### 1. Convert raw data

The conversion scripts read the directory configured by
`DEEPKINEMATICS_DATA_DIR`:

```bash
python DK01_Data_Conversion/DK01_DC01_JC_conversion.py
python DK01_Data_Conversion/DK01_DC01_FKbvh_conversion.py
```

Conversion is data- and licence-dependent. Review the subject exclusions and
input locations before starting a bulk conversion.

### 2. Train a model

Inspect the full configuration interface with:

```bash
python DK03_Training/DK03_trainFK.py --help
```

A minimal forward-kinematics run starts with the data modality and model type,
for example:

```bash
python DK03_Training/DK03_trainFK.py \
  --VERSION FK --m_type rnn --use_acc_gyro \
  --optimizer adamw --scheduler cosine
```

Weights & Biases logging is enabled by the training entry points. Use the normal
W&B environment settings for offline or online runs. The Slurm examples in
`DK03_Training/Slurm` resolve the project path from their own location and accept
`PYTHON_BIN` and `DEEPKINEMATICS_LOG_DIR` overrides.

### 3. Run held-out inference

Each model directory must contain `config.json` and `test_model.pth`:

```bash
python DK04_Gait_Parameter/DK04_Gait07_run_inference.py \
  --code . \
  --model-dir /path/to/model \
  --data-dir "$DEEPKINEMATICS_DATA_DIR" \
  --subjects 04 09 14 24 51 54 67 81 \
  --trial Norm_Post \
  --out ../../00_Results/predictions/MODEL
```

Checkpoint loading uses `weights_only=True`; do not weaken this for untrusted
checkpoint files.

### 4. Run gait and agreement analyses

The numbered scripts document their inputs and outputs in their module docstrings
and `--help` text. The main maintained stages are:

- `DK04_Gait01_spatiotemporal.py`: treadmill-corrected spatial and temporal gait parameters.
- `DK04_Gait02_reconstruction.py`: per-subject joint-rotation and position errors.
- `DK04_Gait03_agreement.py`: Bland-Altman, repeated-measures LoA, ICC, CCC, MAE, and RMSE.
- `DK04_Gait08_joint_kinematics.py`: gait-cycle waveforms, discrete joint features, and SPM1D.
- `DK04_Gait09_gyro_event_detection.py`: validate gyroscope initial-contact events against OMC.
- `DK04_Gait10_within_between_agreement.py`: separate participant-level and cycle-level spatial calibration.
- `DK04_Gait11_calibration_decomposition.py`: clustered-bootstrap spatial calibration decomposition.
- `DK04_Gait12_joint_rom_decomposition.py`: between-participant and within-foot ROM calibration.

Use `--help` before running a data-producing command. Output folders may already
contain manuscript artefacts.

## Validation checks

Four analysis modules contain deterministic synthetic-data self-tests:

```bash
python DK04_Gait_Parameter/DK04_Gait03_agreement.py --selftest
python DK04_Gait_Parameter/DK04_Gait08_joint_kinematics.py --selftest
python DK04_Gait_Parameter/DK04_Gait11_calibration_decomposition.py --selftest
python DK04_Gait_Parameter/DK04_Gait12_joint_rom_decomposition.py --selftest
```

For a fast source audit without loading study data:

```bash
python -m compileall -q DK00_Utils DK01_Data_Conversion \
  DK02_Data_Visualisation DK03_Training DK04_Gait_Parameter DK05_Figures
ruff check .
```

## Data conventions

Processed training data are stored per participant (`SonE_XX`). IMU arrays use
sensor order: left ankle, right ankle, left arm, right arm, head, trunk.

Joint-centre files:

- `*_imu.npz`: `acc`, `gyro`, `mag`, and `quat`; quaternions are stored as
  `(w, x, y, z)` and converted to SciPy's `(x, y, z, w)` convention in code.
- `*_vicon.npz`: `jc`, `ori`, `lcp`, and `rcp` at 200 Hz.

Forward-kinematics files:

- `*_skeleton.npz`: skeletal offsets and parent indices.
- `*_imu.npz`: the six-IMU signals.
- `*_joint_rotation.npz`: local joint rotation matrices and sampling frequency.
- `*_position.npz`: global joint positions and sampling frequency.

The held-out manuscript analyses use participants `04`, `09`, `14`, `24`, `51`,
`54`, `67`, and `81`. Do not silently change this set: it defines the evaluated
test cohort and therefore changes the reported results.

## Reproducibility notes

- Gait events are shared across OMC and model reconstructions; temporal agreement
  therefore evaluates the event detector rather than the learned reconstruction.
- Participant-cluster bootstraps resample participants, never individual strides.
- Mean-centring makes within-participant mean bias zero by construction; interpret
  the within-participant slope and centred error instead.
- Keep generated model `config.json`, checkpoint, and archived source ZIP together.
  Existing manuscript checkpoints were produced with PyTorch 2.x and should be
  validated with the self-tests before any full result regeneration.

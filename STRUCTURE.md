# Deep Kinematics — code structure

All code lives under `dl_humanmotion-main/`, organised by pipeline stage.

```
dl_humanmotion-main/
├── DK00_Utils/               shared config, dataset, models, layers, losses, metrics, eval helpers
├── DK01_Data_Conversion/     BVH / joint-centre / IMU conversion
├── DK02_Data_Visualisation/  raw-data, Vicon and orientation viewers
├── DK03_Training/            training entry points (FK/JC), per-model scripts (Att, Diff, Rnn=BiLSTM, Vrn*),
│                             debug/test variants, and dkscript_*.sh SLURM launchers.
│                             Each .py has a 2-line path bootstrap so it imports DK00_Utils from any cwd.
├── DK04_Gait_Parameter/      gait-parameter analysis (spatiotemporal, reconstruction, agreement, kinematics,
│   │                         inference) + the current manuscript's Python gait-event pipeline:
│   ├── DK04_Gait09_gyro_event_detection.py   raw foot-gyro initial-contact detector
│   ├── DK04_Gait10_event_validation_FVA.py   event validation vs FVA reference (Table VII)
│   ├── data/                 npz inputs (OMC/gyro extract, reconstructed positions, per-model flexion)
│   └── Functions_py/          shared MATLAB-port helpers
├── DK05_Figures/             figure + table generation
│   ├── Plot_ResultTables.py         result tables
│   ├── Plot_ResultFigures.py        kinematic + spatial result figures
│   ├── Plot_BlandAltman_Spatial.py  Bland–Altman panels (Fig. 5)
│   ├── helpers.py                    ICC / CCC / CMC / cycle-segmentation / SPM helpers
│   └── plots/                        rendered output
├── README.md                 original project README
└── environment.yml           conda environment
```

## Cleanup / consolidation notes

- Removed clutter to `../_to_delete/` (nothing deleted — empty it when satisfied):
  caches, IDE files, OS files, MATLAB autosaves, `*.bak`, `outdated_scripts/`, the
  vendored SPM1D MATLAB toolbox, and old Zeni-era analysis scripts.
- The manuscript analysis (former standalone `DK06_Manuscript_Analysis/`) was folded
  into the existing structure: the unique **gyro event detector** and **FVA validation**
  went to `DK04_Gait_Parameter/` (Gait09, Gait10); the **table/figure** scripts went to
  `DK05_Figures/`; the shared `data/` moved to `DK04_Gait_Parameter/data/`.
- The overlapping spatial-agreement and joint-kinematics scripts duplicated
  `DK04_Gait01`/`Gait03`/`Gait08`, so they were retired to
  `../_to_delete/dk06_compute_dropped/` (DK04's versions are kept).
- All DK03 model training scripts (Att/Diff/Rnn/Vrn* + launchers) were preserved.

`_to_delete/` still holds everything moved out, in case anything needs rescuing.

# DL_HumanMotion
The work is to investigate the feasbility of 3-D full body reconstruction using sparse 6 IMU sensors.

## Installation

This work was tested on Windows 10 with Python 3.9 and CUDA 11.7.
To manage your environment miniconda is recommended.
To install all necessary packages execute the commands below.

```
git clone https://gitlab.ethz.ch/lmb_ncm/dl_humanmotion.git
cd dl_humanmotion
conda create -n dl_humanmotion python=3.9.17
conda activate dl_humanmotion
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
conda install -c conda-forge ezc3d
```

## Training code

Detailed code descriptions are further down the page, this section is for whom may be familiar with the project.

#### Model options

`rnn`, `mlprnn` ,`rnndct`, `attdct`, `att`, `vrnatt`

#### Remark

- **Velocity addition to root velocity is hardcoded for SonE_22, has to be rewritten if other subjects/more subjects are used for training.**

i.e. `python DK03_trainFK_Debug.py --VERSION 'FK' or 'JC' --bs_train 128 --m_type rnn --eval_every 10 --n_epochs 100 --optimizer 'adam' --scheduler 'step' --lr 1e-3 --m_num_layers 1 -
-m_hidden_units 256 --m_learn_init_state --m_bidirectional --use_acc_gyro --predict_contact --predict_orientation  --use_batch_norm
`
### Evaluation

For the evaluation of trained models, use the command:
`python DK03_testJCKF.py --model_id` and append the model ID and if wanted other options.

i.e. ` python DK03_testJCKF.py --trial 'Norm_Post' --save_matlab --save_eucl_dist`


# Data

Already processed data can be found here: 

`\\hest.nas.ethz.ch\green_groups_lmb_public\Projects\NCM\NCM_SonEMS\project_only\04_Students\01_Finished\07_Ahaeflig\Training Data`
 
Matlab (IMU) files and c3d files can be found here:

`\\hest.nas.ethz.ch\green_groups_lmb_public\Projects\NCM\NCM_SonEMS\project_only\02_Data\Healthy_Young OR Healthy_Older`
## Overview of data

For an exact listing of all data check [processing_overview]('\\hest.nas.ethz.ch\green_groups_lmb_public\Projects\NCM\NCM_SonEMS\project_only\04_Students\01_Finished\07_Ahaeflig\ProcessingOverview') sheet.  

Data Excluded due to IMU faulty reading and VICON
* SonE_05 (IMU did not sync)  
* SonE_37 (IMU did not sync correctly)  
* SonE_38 (IMU did not sync)  
* SonE_39 (IMU did not sync)  
* SonE_52 (excluded from study)  
* SonE_55 (marker is missing)  
* SonE_57 (excluded from study)  
* SonE_60 (Marker is missing)  
* SonE_68 (IMU did not sync)  
* SonE_84 (IMU data is missing)  

**Data from SonE_88 to SonE_102 are not available for FK, requires VICON shogun licence for bvh conversion**  

## Structure of Data Directory
The data conversion and training pipelines are based on the folder structure below.    

Training_Data  
├── SonE_01  
├── SonE_02 <br>
│&emsp; ├── bvh -> Folder containing bvh files of trials <br>
│&emsp; ├── fk -> Folder containing npz files of forward kinematics approach  <br>
│&emsp; ├── npz-files -> npz files of the joint center approach <br>
│&emsp; ├── IMU -> GaitSummary_DL Matlab struct; re-transfer from the server required due to data space <br>
│&emsp; └── Vicon -> Folder containing c3d files of trials; re-transfer from the server required due to data space

For the creation of npz files for training the following files are needed:  
- GaitSummaryIMU_DL_SonE_XX for the extraction of IMU data.  
- C3d for the extraction of Vicon markers and for synchronization in case of bvh conversion.  
- bvh files for bvh conversion.

## Joint Center npz-files
All data of the joint center prediction is sampled at 200Hz.

### **_imu.npz
Contains the following arrays:  
acc: &emsp; Acceleration (Frames, Sensors, 3)  
gyro: &emsp; Gyroscope/Angular velocity (Frames, Sensors, 3)  
mag: &emsp; Magnetometer (Frames, Sensors, 3)  
quat: &emsp; Sensor Orientation in quaternions (Frames, Sensors, 4), Order of quats: (w,x,y,z)  
The sensors' indices are documented in the table below.  

Sensor | Index
--- | --- 
left ankle | 0 
right ankle | 1 
left arm | 2 
right arm | 3 
head | 4 
trunk | 5 

### **_vicon.npz
Contains the following arrays:  
jc: &emsp; Joint Center (Frames, Joint Center, 3)  
ori: &emsp; Segment Orientation (Frames, Sensor Positions, 3, 3)  
lcp: &emsp; Left Contact Phase (Frames, 2), First toe contact, then heel contact      
rcp: &emsp; Right Contact Phase (Frames, 2)  

Joint | Index
--- | --- 
head | 0 
left shoulder | 1 
right shoulder | 2 
left elbow | 3 
right elbow | 4 
left wrist | 5 
right wrist | 6
left hip | 7 
right hip | 8 
left knee | 9 
right knee | 10 
left ankle | 11
right ankle | 12
left toe | 13
right toe | 14

## Forward Kinematics npz-files
IMU data is sampled at 200Hz whereas bvh data has a sampling frequency of 100 Hz.

### **_skeleton.npz
Holds the skeletal information of the subject.  
offset: &emsp; Array, that saves length of segments as vectors (31, 3).  
parent: &emsp; List indicating parent joint of current joint (31).  

### **_imu.npz
Holds the same information as the **_imu.npz for the joint center prediction.  

### **_joint_rotation.npz
joint_rotations: &emsp; Movement as local rotations of joints (Frames, 31, 3, 3).  
frequency: &emsp; Sampling frequency.  

### **_position.npz  
position: &emsp; Global position of joints (Frames, 31, 3).  
frequency: &emsp; Sampling frequency  

Joint | Index
--- | --- 
Hips | 0
Spine | 1
Spine1 | 2
Spine2 | 3
Spine3 | 4
Neck | 5
Neck1 | 6
Head | 7
Head_end0 | 8
RightShoulder | 9
RightArm | 10
RightForeArm | 11
RightHand | 12
RightHand_end0 | 13
RightHand_end1 | 14
LeftShoulder | 15
LeftArm | 16
LeftForeArm | 17
LeftHand | 18
LeftHand_end0 | 19
LeftHand_end1 | 20
RightUpLeg | 21
RightLeg | 22
RightFoot | 23
RightToeBase | 24
RightToeBase_end0 | 25
LeftUpLeg | 26
LeftLeg | 27
LeftFoot | 28
LeftToeBase | 29
LeftToeBase_end0 | 30

# Codes Overview
The folders and scripts contained in it are described in this section.  

* [DK00_Utils](../main/DK00_Utils/)
* [Data Conversion](../main/DK01_Data_Conversion)
* [Data Visualisation](../main/DK02_Data_Visualisation)
* [DK04_Giat Parameter](../main/DK04_Giat Parameter)

### Utils for DL

The [DK00_Utils](../main/DK00_Utils/) folder consists all the relevant functions and classes that are used for training and evaluating the model.

### Data Conversion
The [DK01_Data_Conversion](../main/DK01_Data_Conversion) folder consists two data conversion scripts

* [DK01_DC01_JC_conversion](../main/DK01_Data_Conversion/DK01_DC01_JC_conversion.py) synchronizes and converts data from c3d files and Matlab structs to npz-files for training.
  - The [step detection algorithm](../main/DK01_Data_Conversion/DK00_DC00_JC_step_dectection.py) has some issues with jumping/tripping.If there is an error it has to be debugged by hand. Most of the times a TOloc has to be deleted.

* [DK01_DC01_FKbvh_conversion](../main/DK01_Data_Conversion/DK01_DC01_FKbvh_conversion.py) synchronizes and converts data from bvh files and Matlab structs to npz-files for training.

### [Data Visualisation](../main/DK02_Data_Visualisation)

#### VICON, FK and IMU Data Plot

Script [DK02_DV00_PlottingRawData](../main/DK02_Data_Visualisation/DK02_DV00_PlottingRawData.py) plots the selected sensor and corresponding vicon/bvh joint.
* plots the contact phase prediction amongst sensors, vicon and bvh.
* plots the acceleration of the feet sensor together with the heel position.

### Vicon_visualization 

Script [DK02_DV01_Viewer_ViconMarker](../main/DK02_Data_Visualisation/DK02_DV01_Viewer.ViconMarker.py) simply displays markers directly from c3d

Script [DK02_DV01_Viewer_Vicon.py](../main/Data_Visualization/DK02_DV01_Viewer.Vicon.py) displays vicon joint centere markers alone or together, with an option to remove root.

Script [DK02_DV02_Viwer_ViocnAndOrientations](../main/Data_Visualization/DK02_DV01_Viewer.Vicon.py) displays vicon joint center markers alone or together with the vicon segment orienation or imu orientation.

- The following options are provided by the class:  
  - `segment_ori` &emsp; (Variable) Segment orientation of Vicon.
  - `lines` &emsp; (Variable) Lines for the visulization with the AITviewer.
  - `imu_pos` &emsp; (Variable) Positions of IMU sensors.
  - `imu_ori` &emsp; (Variable) Orientations of IMU sensors.'
  - `imu_ori_aligned` &emsp; (Variable) Orientations of IMU sensor aligned to Vicon segment orientation.
  - `imu_ori_corrected_left_ankle`(Function) If called, rotates left foot, rotate 180 aroud z-axis
  - `remove_root_movement` &emsp; (Function) If called, fixes root (midpoint between hip joints) to the origin for Vicon joint center markers.
  - `rotate_to_theoretical_imu_orientation` &emsp; (Function) If called, rotates segment orientation to expected IMU orientation. (Relevant for IMU orientation correction)
### bvh_visualization

The [DK02_DV03_bvh.py](../main/Data_Visualization/DK02_DV03_bvh.py) script enables the visualization of bvh skeletons together with imu orientations.

The following options are provided by the class:  
- `connections` &emsp; (Variable) Connections of key points for skeleton creation.
- `positions` &emsp; (Variable) Global key point positions.
- `local_positions` &emsp; (Variable) Local key point positions (root fixed to origin).
- `imu_pos` &emsp; (Variable) Positions of IMU sensors.
- `imu_ori` &emsp; (Variable) Orientations of IMU sensors.

### dct_visualization

The [DK02_DV03_dct.py](../main/Data_Visualization/DK02_DV03_dct.py) script experiments with dct transform and the number of coefficients.


### angular_error_visualization

The [angular_error_visualization.py](../main/Code_Alex/Data_Visualization/angular_error_visualization.py) script plots the drift of the IMU orientation in comparison to the Vicon segment orientation over the duration of a trial.

## [Gait Parameter](../main/DK04_Gait_Parameter)
The gait parameter folder contains the Matlab pipeline to calculate gait parameter from the ground truth and predicted joint center kinematics.  
The kinematic data has to be saved in Matlab structs which can be created in the evaluate.py function by using the tag `--save_matlab`.

### DK04_Gait01_spatiotemporal
Parameter to change:  
- `TestPath1` Path to data folder with the following structure:  
└── Model ID             
&emsp; &emsp; ├── SonE_XX &emsp;   
&emsp; &emsp; └── SonE_YY  
&emsp; &emsp; &emsp; &emsp;├── Trial &emsp; &emsp; Matlab struct with predicted kinematic data  
&emsp; &emsp; &emsp; &emsp;└── Trial_gt &emsp; Matlab struct with ground truth kinematic data

- `addpath` Path to functions: [Functions](../main/DK04_Gait_ParameterFunctions)

### DK04_Gait02_Bland_Altman_plots
Parameter to change:  
- `TestPath1` Path to data.  
- `subjectloop` Subject(s) to plot.

### DK04_Gait03_Mean_into_Excel_R_L  
Parameter to change:  
- `TestPath1` Path to the data.

# Suggestions
- Change evaluation strategy in train.py to not rely on single window of 500 frames (current approach) but on entire unseen sequence or for the easy and hard test set on the entire trial.

# Contact
If questions arise regarding the code you can contact me under alexander.haefliger@gmail.com# deepkinematics

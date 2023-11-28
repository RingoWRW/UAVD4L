# A dataset for UAV localization
This is UAVLoc, a pipeline for 6-DoF uav visual localization.
## Install dependencies
- If `conda` environment is available \
```
conda env create -f setup/environment.yml
conda activate uavloc
```
- Otherwise, if `conda` environment is not readily available:
```
python3 -m venv venvuavloc
source venvuavloc/bin/activate
pip3 install pip -U && pip3 install -r setup/requirements.txt
```
***
## pipeline
The pipeline include 2 steps
1) off-line dataset generation
2) on-line uav localization (GPS-denied)
***
### off-line dataset generation
#### Preparation
 - textured 3D mesh (.obj)
 - angle txt file (bamu_euler.txt --> pitch roll yaw)
 - digital surface model (DSM)
 - camera sensor parameter (sensor width, sensor height, focal lenght)
#### Run 
`python -m targetloc.generate_dataset.pipeline_dataset_generation_limit`
#### output
- pose.txt (DB_pose)
- intrinsic.txt (DB_intrinsic)
- test (points corresponding to .obj) 
- RGB images and depth images
#### dataset structure
- For generation synthetic dataset
> |---  Render_all     &emsp;     # synthetic dataset \
> &emsp;    |--- images  &emsp;   # synthetic RGB images \
> &emsp;    |--- depths  &emsp;   # synthetic depth images \
> &emsp;    |--- db_pose.txt  &emsp;   # database pose w2c  (Quaternion, Position:wxyz,xyz) \
> &emsp;    |--- db_intrinsics.txt  &emsp;  # database intrinsic \
- For query image, unditorted image, generate IMU and intrinsic .txt
> |--- Query &emsp; # query dataset \
> &emsp; |--- images &emsp;  # RGB images \
> &emsp; &emsp; |--- W (wide-angle)  \
> &emsp; &emsp; |--- Z (zoom-in) \
> &emsp; |--- intrinsics \
> &emsp; &emsp; |-- w_intrinsic.txt \
> &emsp; &emsp; |-- z_intrinsic.txt \
> &emsp; |--- poses \
> &emsp; &emsp; |-- w_pose.txt \
> &emsp; &emsp; |-- z_pose.txt \
> &emsp; &emsp; |-- gt_pose.txt 
***
### on-line uav localization
#### Preparation
 - offline generation datasets
    - include RGB images, depths images, pose txt, intrinsic txt
 - query images
    - include RGB images, IMU sensor, intrinsic (sensor width, sensor height, focal length, calibrate coefficient)
    - if you want to localize the target position, you also should provide _Z.JPG and intrinsic, and .json file (2D coordinates.)
#### Run 
`python -m targetloc.localization_detection.pipeline_other_city` \
if you use with-gravity to solve PnP, you should compile poselib, and you can refer to [poselib](https://github.com/vlarsson/PoseLib). \
For more environmental configurations, please refer to [hloc](https://github.com/cvg/Hierarchical-Localization).
#### output
- global descriptor .h5
- local descriptor .h5
- uav localization result .txt
- target localition result .txt (optimal)
#### Attention
Our image coordinate system selected as 114E, if you want to use our code, please modified the image coordinate system.
```
read_EXIF.py  --> (108)
read_SRT.py  -->  (37)
eval.py  --> (272)
```
#### version
- for uav localization \
   We provide detector-based (SPP+SPG, SPP+NN, ...) and detector-free (LoFTR).
- for target localization \
   We provide mouse-based and .json file -based.

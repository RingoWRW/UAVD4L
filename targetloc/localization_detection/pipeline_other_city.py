from pathlib import Path
from pprint import pformat
import argparse
import json
import os
import sys
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from hloc import extract_features, match_features,extract_features_db, extract_features_query,extract_global_features_db,extract_global_features_query
from hloc import pairs_from_prior_guided_retrieval, eval
from hloc import check, match
from hloc import ray_casting, undistort,localize_sensloc_topk, localize_sensloc_withgravity, ray_casting_new
from hloc.visualization import visualize_from_h5,  visualiza_retrieval_from_txt
from sensloc.utils.preprocess import video_to_frame, read_SRT, read_EXIF, read_gt, read_RTK_YU
from ultralytics.yolo.v8.detect import predict
from sensloc.utils.preprocess.read_SRT import load_json


'''
detector-based uav-localization method, also provide target localition
'''


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, default='datasets/CityofStars',
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default='output/targetloc/CityofStars_all/',#!
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--num_covis', type=int, default=20,
                    help='Number of image pairs for SfM, default: %(default)s')
parser.add_argument('--num_loc', type=int, default=3,
                    help='Number of image pairs for loc, default: %(default)s')

parser.add_argument('--config_file',default='sensloc/config/online_uavloc/uavloc_detect_based.json', type=str, help='configuration file')#!

args = parser.parse_args()

with open(args.config_file) as fp:
        config = json.load(fp)
# Setup the paths

dataset = args.dataset  # where everything is loaded
outputs = args.outputs  # where everything will be saved

input_EXIF_photo = config["read_EXIF"]["input_EXIF_photo"]
distortion = config["read_EXIF"]["distortion"]
query_gt_xml = config["read_EXIF"]["gt_xml"]
RTK_path = config["read_RTK"]["input_RTK_path"]

images_db = Path(config["dataset_image"]["images"])
depth = Path(config["dataset_image"]["depths"])

query_path = Path(config["queries"]) 
images_query_w = query_path / 'images/W'
images_query_z = query_path / 'images/Z'
sensor_path = query_path / 'poses/w_pose.txt'
query_intrinsics = query_path / 'intrinsics/w_intrinsic.txt'
gt_pose = query_path / 'poses/gt_pose.txt'
gt_position = query_path / 'poses/gt_position.txt'
loc_pairs = outputs / f'pairs-query-openibl-{args.num_loc}.txt'  # top-k retrieved by Prior  dir
results = outputs / f'WideAngle_hloc_feats-svcnn_{args.num_loc}.txt'#!
results_position = outputs / f'predictXYZ.txt'
eval_results = outputs / f'eval_results.txt'
eval_results_target = outputs / f'eval_results_target.txt'
visualize_output = outputs / 'retrieval_images/'
match_output = outputs / 'match_images/'
pipeline_results = outputs / f'pipeline_results.txt'

if not os.path.exists(visualize_output):
    os.makedirs(visualize_output)
if not os.path.exists(match_output):
    os.makedirs(match_output)


# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

# pick one of the configurations for extraction and matching
retrieval_conf = extract_features.confs['openibl']
feature_conf = extract_features.confs['superpoint_max'] #superpoint_max sift
matcher_conf = match_features.confs['superglue'] #NN-ratio superglue

# read query IMU, calibrate them, and 
# TODO check DB image == len(pose)
# check.main(images_db, config["dataset_image"]["read_path"], config["dataset_image"]["write_path"])
db_poses = config["dataset_image"]["pose_txt"]
db_intrinsics = config["dataset_image"]["intrinsic_txt"]
read_EXIF.main(input_EXIF_photo, query_gt_xml, query_path)
undistort.main(input_EXIF_photo, images_query_w, images_query_z, query_intrinsics, distortion)
# read_RTK_YU.main(input_EXIF_photo, RTK_path, gt_position)


# extract db global features at first time, save into .h5 format
global_descriptors_db = extract_global_features_db.main(retrieval_conf, images_db, outputs)
global_descriptors_query = extract_global_features_query.main(retrieval_conf, images_query_w, outputs)

# extract db local features at first time, save into .h5 format
features_db = extract_features_db.main(feature_conf, images_db, depth, db_poses, db_intrinsics, outputs)
features_query = extract_features_query.main(feature_conf, images_query_w, outputs)

# guided retrieval with euler prior, without gps
pairs_from_prior_guided_retrieval.main(
    global_descriptors_db, global_descriptors_query, loc_pairs, args.num_loc,
     db_pose=db_poses, sensor_path=sensor_path, with_gps = False, rot_thres=30) 

# ======= retrieval visualize ==========
# images_query_w = Path("/media/guan/data/20230512feicuiwan/H20t/seq_all_new/")
# visualiza_retrieval_from_txt(images_db, images_query_w, loc_pairs, visualize_output)

# matching according descriptors
loc_matches = match_features.main(
    matcher_conf, loc_pairs, feature_conf['output_db'],feature_conf['output_query'], outputs)

# ======== localize visualize ==========
# visualize_from_h5(images_db, images_query_w, loc_pairs, features_db, features_query, loc_matches, match_output)
# # visualize_from_h5(images_db, images_query_w, loc_pairs, "/media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/outputs/targetloc/CityofStars_all/db_feats-svcnn.h5", "/media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/outputs/targetloc/CityofStars_all/query_feats-svcnn.h5", loc_matches, match_output)



## ======== with gravity PnP ==========
localize_sensloc_withgravity.main(
    query_intrinsics,
    loc_pairs,
    features_db,
    features_query,
    sensor_path,
    loc_matches,
    eval_results,
)

## ===== without gravity PnP ========
# localize_sensloc_topk.main(
#     query_intrinsics,
#     loc_pairs,
#     features_db,
#     features_query,
#     loc_matches,
#     eval_results,
#     covisibility_clustering=False,
# ) 


# UAV-loc eval
eval.evaluate(eval_results, gt_pose, pipeline_results)

# ============================= optimal target localization ===============================
# dectection returen center_points
# center_points_list center_point_dict:{{img_name:[{'car1':[center_point_x]},{}....]}}
# center_points_list = predict.main(images_query_z)

# target position calculation
# TODO match.txt generation W-Z
match.main(config["z_file_path"]["all_image"], config["z_file_path"]["write_path"])
z_json_path = config["z_file_path"]["json"]
z_w_match = config["z_file_path"]["match_file_path"]
object_name = config["ray_casting"]["object_name"]
detect_intrinsics = query_path / 'intrinsics/z_intrinsic.txt'

center_points_list =  load_json(z_json_path)

# set True: .json file // False: by mouse, should set z_path(_Z.JPG path)
ray_casting_new.main(config["ray_casting"], eval_results, detect_intrinsics, results_position, center_points_list, z_w_match, True)
eval.main(results, gt_pose, gt_position, results_position, object_name, eval_results_target)

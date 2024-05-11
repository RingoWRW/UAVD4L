from pathlib import Path
from pprint import pformat
import argparse
import json
import os
import sys
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from sensloc.utils.preprocess import generate_render_obj_compare
from sensloc.utils.blender import blender_engine_limit

"""
内存受限制, 根据块数和obj生成pose来渲染
"""

parser = argparse.ArgumentParser()
parser.add_argument('--config_file',default='sensloc/config/offline_dataset/offline_dataset_generation_limit.json', type=str, help='configuration file')#!
args = parser.parse_args()

with open(args.config_file) as fp:
    config = json.load(fp)


# obj path
obj_path = config["generate_render_pose"]["obj_path"]
obj_dict_path = config["generate_render_pose"]["obj_dict_path"]
orgin_coord = config["generate_render_pose"]["orgin_coord"]
oritation_path = config["generate_render_pose"]["oritation_path"]
intrinsic = config["generate_render_pose"]["intrinsic"]
base_height = config["generate_render_pose"]["base_height"]
write_path = config["generate_render_pose"]["write_path"]
block_num = config["generate_render_pose"]["block_num"]
DSM_config = config["generate_render_pose"]["config"]
# blender setting
blender_config = config["blender_engine"]["config"]

# step1: generate poses and render .txt file
generate_render_obj_compare.main(obj_path, obj_dict_path, orgin_coord, oritation_path, intrinsic, base_height, write_path, block_num, DSM_config)

# step2: generate rendering db images and depth images (optional)
blender_engine_limit.main(blender_config)
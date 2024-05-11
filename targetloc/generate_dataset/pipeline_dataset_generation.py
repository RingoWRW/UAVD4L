from pathlib import Path
from pprint import pformat
import argparse
import json
import os
import sys
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from sensloc.utils.preprocess import generate_render_pose
from sensloc.utils.blender import blender_engine

"""
内存不受限制
"""

parser = argparse.ArgumentParser()
parser.add_argument('--config_file',default='sensloc/config/offline_dataset/offline_dataset_generation.json', type=str, help='configuration file')#!
args = parser.parse_args()

with open(args.config_file) as fp:
    config = json.load(fp)

blender_config = config["blender_engine"]
db_render_path = Path(config["generate_render_pose"]["image"]["write_path"])
db_intrinsics = db_render_path / "db_intrinsics.txt"
db_poses = db_render_path / "db_pose.txt"
images_db = Path(config["blender_engine"]["db_images"])

# step1: generate poses by setting the altitude and focal lens (optional)
generate_render_pose.main(config["generate_render_pose"]["image"])
# step2: generate rendering db images and depth images (optional)
blender_engine.main(blender_config, db_intrinsics, db_poses, images_db)
import bpy 
import os 
import numpy as np
import mathutils

input_recons = '/home/ubuntu/Documents/code/SensLoc/datasets/AirLoc_wild/jinxia_2304/Production_1/Data'
origin_coord = [399961, 3138435, 0]

###############load obj    
for tile in os.listdir(input_recons):
    obj_path = os.path.join(input_recons, tile, tile+'.obj')
    bpy.ops.import_scene.obj(filepath=obj_path)
    

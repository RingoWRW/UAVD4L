from copy import deepcopy
from math import floor
from tkinter import Scale
import h5py
import logging
import pickle
import cv2
import pycolmap
from collections import defaultdict
import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData,PlyElement
import random
def interpolate_depth(pos, depth):

    ids = torch.arange(0, pos.shape[0])
    depth = depth[:,:,0]
    h, w = depth.size()
    
    
    
    i = pos[:, 0]
    j = pos[:, 1]

    # Valid corners 验证坐标是否越界
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    # ids = ids[valid_corners]

    # Valid depth验证深度
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    # 深度有效的点的index
    ids = ids[valid_depth]
    
    # Interpolation 插值深度
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    #插值出来的深度
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]


# 读取深度
def read_valid_depth(depth_exr,mkpts1r):
    depth = cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)
    depth = torch.tensor(depth)  #?
    mkpts1r_a = torch.unsqueeze(mkpts1r.cpu()[:,0],0)
    mkpts1r_b =  torch.unsqueeze(mkpts1r.cpu()[:,1],0)
    mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0)

    depth, _, valid = interpolate_depth(mkpts1r_inter , depth)

    return depth,valid


def resample(points, k):
    """Resamples the points such that there is exactly k points.

    If the input point cloud has <= k points, it is guaranteed the
    resampled point cloud contains every point in the input.
    If the input point cloud has > k points, it is guaranteed the
    resampled point cloud does not contain repeated point.
    """
        # rand_idxs = np.random.choice(, k, replace=False)
    rand_idxs = torch.LongTensor(random.sample(range(points.shape[0]), k))
    sample_points = torch.index_select(points, 0, rand_idxs)
    return sample_points


def write_ply(save_path,points,text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    pc_tensor = torch.tensor(pc_array)
    return pc_tensor
     


def Get_Points3D(depth_exr, R, t, K, w,h):   #c2w  points[n,2]
    scale = 32
    depth = cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)
    # depth = cv2.resize(depth, (floor(depth.shape[1]/scale), floor(depth.shape[0]/scale)))  
    w = floor(depth.shape[1] / scale)
    h = floor(depth.shape[0] / scale)
    points = torch.zeros(w*h, 2)

    depth = torch.tensor(depth)  #?
    index = 0    
    for i in range(h):  
        for j in range(w):
            points[index][0] = i * scale
            points[index][1] = j * scale
            index += 1
    depth, _, valid = interpolate_depth(points, depth)
    points_2D = torch.cat([points, torch.ones_like(points[ :, [0]])], dim=-1)
    points_2D = points_2D.T  #[3, n]
    t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
    Points_3D = R @ K @ (depth * points_2D) + t   
    return Points_3D    #[3,n]



# def read_valid_depth(depth_exr,mkpts1r):
#     depth = cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)
#     depth = torch.tensor(depth)  #?
#     mkpts1r_a = torch.unsqueeze(mkpts1r.cpu()[:,0],0)
#     mkpts1r_b =  torch.unsqueeze(mkpts1r.cpu()[:,1],0)
#     mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0)

#     depth, _, valid = interpolate_depth(mkpts1r_inter , depth)

#     return depth,valid

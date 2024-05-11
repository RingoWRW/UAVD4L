import os
import cv2
import os
import torch
import numpy as np
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def interpolate_depth(pos, depth):
    ids = torch.arange(0, pos.shape[0])
    
    depth = depth[:,:,0]
    h, w = depth.size()
    
    if depth.is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    
    
    
    i = pos[:, 0] #h
    j = pos[:, 1] #w

    # Valid corners, check whether it is out of range
    i_top_left = torch.floor(i).long().to(device)
    j_top_left = torch.floor(j).long().to(device)
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long().to(device)
    j_top_right = torch.ceil(j).long().to(device)
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long().to(device)
    j_bottom_left = torch.floor(j).long().to(device)
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long().to(device)
    j_bottom_right = torch.ceil(j).long().to(device)
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


    # Valid depth
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

    # vaild index
    ids = ids[valid_depth]
    
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    #depth is got from interpolation
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)
    

    return [interpolated_depth, pos, ids]

# read depth
def read_valid_depth(depth_exr,mkpts1r):
    
    if os.path.exists(depth_exr):
        # import pdb;pdb.set_trace()
        depth = cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(depth_exr)
        if depth is None:
            print(depth_exr)
            return depth, None
        depth = torch.tensor(depth).to(device)
        mkpts1r = torch.tensor(mkpts1r).to(device)
        mkpts1r_a = torch.unsqueeze(mkpts1r[:,0],0)
        mkpts1r_b =  torch.unsqueeze(mkpts1r[:,1],0)
        mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0).to(device)

        depth, _, valid = interpolate_depth(mkpts1r_inter , depth)
    else:
        print(depth_exr)
        return None, None
  

    return depth.cpu(), valid
def Get_Points3D(depth, R, t, K, points):   # points[n,2]

    if points.shape == (2,):
        points = torch.unsqueeze(points,0)
    elif len(points) == 0:
        return None

    if points.shape[-1] != 3:
            # points = torch.unsqueeze(points,0)
        points_2D = torch.cat([points, torch.ones_like(points[:, [0]])], dim=-1)
        points_2D = points_2D.T 
    t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
    Points_3D = (R @ K @ (depth * points_2D) + t)   #[3,n]
    
    return Points_3D.T    #[n, 3]


    
        
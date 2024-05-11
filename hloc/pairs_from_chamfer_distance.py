import argparse
import os
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
import cv2
import scipy.spatial
from scipy.spatial.transform import Rotation
from utils.pc_utils import Get_Points3D
from tqdm import tqdm
import random
from plyfile import PlyData,PlyElement
DEFAULT_DIST_THRESH = 1000  # in meters
DEFAULT_ROT_THRESH = 10  # in degrees (option: 90 or 60 or 45)

# from . import logger
# from .utils.parsers import parse_image_lists
# from .utils.read_write_model import read_images_binary, read_images_text
# from .utils.io import list_h5_names
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
def parse_query_pose_list(path):
    all_pose_c2w = []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            if len(data) > 1:
                q, t = np.split(np.array(data[1:], float), [4])

                R = np.asmatrix(qvec2rotmat(q))   #!c2w
                t = -R.T @ t
                R = R.T
                Pose_c2w = np.identity(4)
                Pose_c2w[0:3,0:3] = R
                Pose_c2w[0:3, 3] = t
                all_pose_c2w.append([name, Pose_c2w])
        
    return all_pose_c2w
def parse_pose_list(path):
    Rs = []
    ts = []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            if len(data) > 1:
                q, t = np.split(np.array(data[1:], float), [4])

                R = qvec2rotmat(q)   #!c2w
                Rs.append(R)
                ts.append(t)
    Rs = np.stack(Rs, 0) 
    ts = np.stack(ts, 0)

    # Invert the poses from world-to-camera to camera-to-world.
    Rs = Rs.transpose(0, 2, 1)
    ts = -(Rs @ ts[:, :, None])[:, :, 0]

    # Only use 2d position (x,y)
    ts_2d = ts[:, :-1]
        
    return ts_2d, Rs
# def read_render_instrincs():
#     w, h, fx, fy, cx, cy = []
def read_reference_instrincs(intrinsc_path, ):
    all_K = {}
    with open(intrinsc_path,'r') as file:
        for line in file:
            data_line=line.strip("\n").split(' ')
            img_name = data_line[0]
            # print(data_line)
            w,h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]   #! :8
            focal_length = fx
            K_w2c = np.array([
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            # all_query_name.append(img_name)
            all_K[img_name] = [K_w2c,focal_length, w, h]
    return all_K
            


def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(f'Unknown type of image list: {names}.'
                             'Provide either a list or a path to a list file.')
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
    if name2idx is None:
        with h5py.File(str(path), 'r', libver='latest') as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), 'r', libver='latest') as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(scores: torch.Tensor,
                            invalid: np.array,
                            num_select: int,
                            min_score: Optional[float] = None):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float('-inf'))

    topk = torch.topk(scores, num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs


def main(descriptors, output, num_matched,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None):
    logger.info('Extracting image pairs from a retrieval database.')

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_descriptors = descriptors
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors)
               for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    query_names_h5 = list_h5_names(descriptors)

    if db_model:
        images = read_images_binary(db_model / 'images.bin')
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')
    query_names = parse_names(query_prefix, query_list, query_names_h5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    query_desc = get_descriptors(query_names, descriptors)
    sim = torch.einsum('id,jd->ij', query_desc.to(device), db_desc.to(device))

    # Avoid self-matching
    self = np.array(query_names)[:, None] == np.array(db_names)[None]
    pairs = pairs_from_score_matrix(sim, self, num_matched, min_score=0)
    pairs = [(query_names[i], db_names[j]) for i, j in pairs]

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
# 读取深度
def read_q_instrincs(intrinsc_path):
    all_K = []
    # all_query_name =[]
    with open(intrinsc_path,'r') as file:
        for line in file:
            data_line=line.strip("\n").split(' ')
            
        #     print(data_line)
            img_name = data_line[0]
            w,h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]   #! :8
            focal_length = fx
            K_w2c = np.array([
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            # all_query_name.append(img_name)
            all_K.append([img_name,K_w2c,focal_length])
    
    return all_K

def get_render_candidate(all_render_name, query_name):
    render_candidate = []
    query = (query_name.split('/')[-1]).split('.')[0]
    for render_name in all_render_name:
        if query in render_name:
            if '_0.' in render_name:
                render_candidate.append(render_name)
    return render_candidate

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
def square_distance(pcd1, pcd2):
    """
    Squared distance between any two points in the two point clouds.
    """
    return torch.sum((pcd1[ :, None, :].contiguous() - pcd2[ None, :, :].contiguous()) ** 2, dim=-1)


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
     


if __name__ == "__main__":
    intrinsics_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/sup/intrinsics.txt"
    query_render_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/images/images_upright/render/"
    reference_render_path = "/mnt/sda/"
    query_render_path = "/mnt/sda/"
    reference_pose_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/sup/db_pose.txt"
    query_pose_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/sup/seed_pose.txt"
    save_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/sup/"
    save_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/sup/retrival.txt"
    all_sequence_intrinsics = read_reference_instrincs(intrinsics_path)
    refer_pose = parse_query_pose_list(reference_pose_path)
    query_pose = parse_query_pose_list(query_pose_path)
    
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
    key_intrinscs_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/sup/phone_day_intrinsics.txt"
    images_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/images/images_upright/"
    ply_base_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/sup/ply/"
    all_K= read_q_instrincs(key_intrinscs_path)
    all_key_name = list(np.array(all_K)[:,0])
    all_query_name = list(np.array(query_pose)[:,0])
    all_refer_name = list(np.array(refer_pose)[:,0])
    #=====X,Y
    query_ts_2d, query_Rs = parse_pose_list(query_pose_path)
    db_ts_2d, db_Rs = parse_pose_list(reference_pose_path)
    dist = scipy.spatial.distance.cdist(query_ts_2d, db_ts_2d)
    index = np.unravel_index(dist.argmin(), dist.shape)
    # print(all_query_name[index[0]], all_refer_name[index[1]])

    all_query_pointclouds = []
    num = 0
    
    pbar = tqdm(total=len(all_key_name), unit='pts')
    
    for imgq_name in all_key_name:
        imgq_pth = images_path + imgq_name
        imgq = imgq_name.split('/')[-1]  #! get short name
        render_candidate = get_render_candidate(all_query_name, imgq)
        points_candidate = []
        for query in render_candidate:          
            img = query.split('/')
            pc_path = ply_base_path + imgq.split('.')[0] + '.ply'
            # if os.path.exists(pc_path):
            #         # load pc
            #     query_pc = read_ply(pc_path)
            #     # sample_P3D = resample(query_pc, 2048)
            #     all_query_pointclouds.append(query_pc)
            #     continue
            query_pth = query_render_path + query
            depth_exr = query_render_path + img[0] +'/' + img[1]+'/' + img[2] +'/'+ img[3] +'/' +'depth/'+img[5]+'/'+ img[-1].split('.')[0]+'0001.exr'  #!
            if not os.path.exists(depth_exr):
                print(depth_exr)
            #设置内参
            k_name = img[1]+'/'+img[2]+'/'+img[3] +'/'
            K_w2c = all_sequence_intrinsics[k_name][0]   
            w = all_sequence_intrinsics[k_name][2] 
            h = all_sequence_intrinsics[k_name][3] 
            K_w2c = torch.tensor(K_w2c).cpu().float()
            K_c2w = K_w2c.inverse()
            
            render_prior_idx = all_query_name.index(query)
            pose_c2w = torch.tensor(query_pose[render_prior_idx][1]).float()
            
            Points_3D = Get_Points3D(depth_exr, pose_c2w[:3, :3], pose_c2w[:3, 3], K_c2w, w, h)  #!k_c2w
            Points_3D = Points_3D.T
            points_candidate.append(Points_3D)
        if len(points_candidate) > 0:
            point_clouds = torch.cat(points_candidate, dim = 0)
            # points_3d_np = torch.stack(points_candidate)
            sample_P3D = resample(point_clouds, 2048)
            # write_ply(save_path + img[-1].split('.')[0] + '.ply', sample_P3D )
            all_query_pointclouds.append(sample_P3D)
        pbar.update(1)
    pbar.close()  
            
        
        # depth,valid=read_valid_depth(depth_exr_final,mkpts1r_final)  #!depth render name bingo
    all_refer_pointclouds = []
    num = 0
    pbar = tqdm(total=len(all_refer_name), unit='pts')
    for imgr in all_refer_name:
        img = imgr.split('/') 
        pc_path = ply_base_path + img[-1].split('.')[0] + '.ply'
        # if os.path.exists(pc_path):
        #         # load pc
        #     refer_pc = read_ply(pc_path)
        #     sample_P3D = resample(refer_pc, 1024)
        #     all_refer_pointclouds.append(sample_P3D)
        #     pbar.update(1)
        #     continue
        refer_pth = query_render_path + imgr
        depth_exr = query_render_path + imgr.split('.')[0] + '0001.exr'
        if not os.path.exists(depth_exr):
            print(depth_exr)
        #intrinsics
        k_name = img[1][2] +'/'
        K_w2c = all_sequence_intrinsics[k_name][0]   
        w = all_sequence_intrinsics[k_name][2] 
        h = all_sequence_intrinsics[k_name][3] 
        K_w2c = torch.tensor(K_w2c).cpu().float()
        K_c2w = K_w2c.inverse()
        #extrinsics
        render_prior_idx = all_refer_name.index(imgr)
        pose_c2w = torch.tensor(refer_pose[render_prior_idx][1]).float()
        #pose
        depth = cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)
        depth = torch.tensor(depth)  #?

        Points_3D = Get_Points3D(depth_exr, pose_c2w[:3, :3], pose_c2w[:3, 3], K_c2w, w, h)  #!k_c2w
        Points_3D = Points_3D.T
        sample_P3D = resample(Points_3D, 2048)

        np_sample_3D = sample_P3D.cpu().numpy()
        # pcd.points = o3d.utility.Vector3dVector(np_sample_3D)
        # o3d.io.write_point_cloud(save_path, pcd)
        
        #==========query=================     
        all_refer_pointclouds.append(sample_P3D)
        pbar.update(1)
    pbar.close() 
    chamfer_distance = torch.zeros([len(all_query_pointclouds), len(all_refer_pointclouds)])
    pbar = tqdm(total=len(all_query_pointclouds), unit='pts')
    with open(save_path, 'w') as file_w: 
        for i in range(len(all_query_pointclouds)):   
            q_name = all_key_name[i]
            for j in range(len(all_refer_pointclouds)):
                r_name = all_refer_name[j]
                query_pointcloud = all_query_pointclouds[i]
                refer_pointcloud = all_refer_pointclouds[j]
                dist = torch.min(square_distance(query_pointcloud, refer_pointcloud), dim=-1)[0]
                kmin_dist = torch.topk(dist, 20, largest = False)[0]  #kmin_dist = torch.sort(dist, descending=False)
                chamfer = torch.mean(kmin_dist, dim = 0)
                chamfer_distance[i][j] = chamfer
            rq_dist = torch.squeeze(chamfer_distance[i,:])
            k = 50
            kmin_rq = torch.topk(rq_dist, k, largest = False)
            index_list = kmin_rq[1]
            score = kmin_rq[0]            
            for ind in range(len(index_list)):
                r = all_refer_name[index_list[ind]] 
                out_str = q_name+' '+ r+ ' ' + str(score[ind])+' \n'
                file_w.write(out_str)
                # print(chamfer_distance[i][j])
            pbar.update(1)
    pbar.close() 
                
    
         
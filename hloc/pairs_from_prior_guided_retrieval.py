import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
import scipy.spatial
from scipy.spatial.transform import Rotation
import time
from . import logger
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary, qvec2rotmat, read_images_text
from .utils.io import list_h5_names
from .transform import parse_intrinsic_list, parse_pose_list



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
            raise ValueError(
                f'Unknown type of image list: {names}.'
                'Provide either a list or a path to a list file.'
            )
    else:
        names = names_all
    return names


def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
    if name2idx is None:
        with h5py.File(str(path), 'r') as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            # mo
            try:
                with h5py.File(str(path[name2idx[n]]), 'r') as fd:
                    desc.append(fd[n][key].__array__())
            except:
                continue
    return torch.from_numpy(np.stack(desc, 0)).float()


def pairs_from_score_matrix(
    scores: torch.Tensor,
    invalid: np.array,
    num_select: int,
    min_score: Optional[float] = None,
):
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


def get_db_retrieval_pose(images):
    Rs = []
    ts = []
    for image in images.values():
        R = image.qvec2rotmat()
        t = image.tvec
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
def get_retrieval_pose(path):
    poses = {}
    names = []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0].split('/')[-1]
            q, t = np.split(np.array(data[1:], float), [4])

            # rotation
            R = qvec2rotmat(q)

            poses[name] = R, t
            names.append(name)

    Rs = []
    ts = []

    for k, _ in poses.items():
        R, t = poses[k]
        Rs.append(R)
        ts.append(t)
    Rs = np.stack(Rs, 0)
    ts = np.stack(ts, 0)

    # Invert the poses from world-to-camera to camera-to-world.
    Rs = Rs.transpose(0, 2, 1)
    ts = -(Rs @ ts[:, :, None])[:, :, 0]

    # Only use 2d position (x,y)
    ts_2d = ts[:, :-1]

    return ts_2d, Rs, names

def get_query_sensor_pose(names, path):
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0].split('/')[-1]
            q, t = np.split(np.array(data[1:], float), [4])

            # rotation
            R = qvec2rotmat(q)

            poses[name] = R, t

    Rs = []
    ts = []

    for n in names:
        R, t = poses[n]
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


def main(
    db_descriptors,
    query_descriptors,
    output,
    num_matched,
    query_prefix=None,
    query_list=None,
    sensor_path=None,
    db_prefix=None,
    db_list=None,
    db_pose=None,
    with_gps=False,
    rot_thres=360,
    dist_thres=150,
):
    logger.info(
        'Extracting image pairs from a retrieval database with SENSOR guide.')
    t_start = time.time()
    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(
        db_descriptors) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())

    query_names_h5 = list_h5_names(query_descriptors)

    if db_pose:
        db_ts_2d, db_Rs, db_names = get_retrieval_pose(db_pose)  
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)# ! long time
    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')

   
    query_names = parse_names(query_prefix, query_list, query_names_h5)
    # gps-priored retrieval
    # import pdb;pdb.set_trace()
    query_ts_2d, query_Rs = get_query_sensor_pose(query_names, sensor_path)
    dist = scipy.spatial.distance.cdist(query_ts_2d, db_ts_2d)
    
    # Instead of computing the angle between two camera orientations,
    # we compute the angle between the principal axes, as two images rotated
    # around their principal axis still observe the same scene.
    # feature-based retrieval
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # import pdb; pdb.set_trace();
    db_desc = get_descriptors(db_names, db_descriptors, name2db)# ! long time
    query_desc = get_descriptors(query_names, query_descriptors)
    # original
    # db_axes = db_Rs[:, :, -1]
    # query_axes = query_Rs[:, :, -1]   
    # dots = np.einsum('mi,ni->mn', query_axes, db_axes, optimize=True)
    # dR = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
    # invalid_axes = dR >= rot_thres 

    # db_axes_pitch = db_Rs[:, :, -1] # pitch
    # query_axes_pitch = query_Rs[:, :, -1]
    # db_axes_yaw = db_Rs[:, :, 0] # yaw
    # query_axes_yaw = query_Rs[:, :, 0]
    # dots_pitch = np.einsum('mi,ni->mn', query_axes_pitch, db_axes_pitch, optimize=True)
    # dR_pitch = np.rad2deg(np.arccos(np.clip(dots_pitch, -1., 1.)))

    # dots_yaw = np.einsum('mi,ni->mn', query_axes_yaw, db_axes_yaw, optimize=True)
    # dR_yaw = np.rad2deg(np.arccos(np.clip(dots_yaw, -1., 1.)))
    # invalid_axes = np.logical_or(((dR_yaw) >= rot_thres),((dR_pitch) >= rot_thres))
    #  # !

    # yaw + pitch + roll
    
    yaw1 = np.arctan2(query_Rs[:, 1, 0], query_Rs[:, 0, 0])
    pitch1 = np.arctan2(query_Rs[:, 2, 1] , query_Rs[:, 2, 2])
    roll1 = -np.arcsin(query_Rs[:, 2, 0])
    yaw2 = np.arctan2(db_Rs[:, 1, 0], db_Rs[:, 0, 0])
    pitch2 = np.arctan2(db_Rs[:, 2, 1], db_Rs[:, 2, 2])
    roll2 = -np.arcsin(db_Rs[:, 2, 0])
    dYaw_pitch = np.zeros((len(query_Rs), len(db_Rs), 3)) # 创建一个空数组来保存yaw和pitch差值
    # import pdb;pdb.set_trace()
    dYaw_pitch[:, :, 0] = np.rad2deg(yaw1[:, None] - yaw2)
    dYaw_pitch[:, :, 1] = np.rad2deg(pitch1[:, None] - pitch2)
    dYaw_pitch[:, :, 2] = np.rad2deg(roll1[:, None] - roll2)

    dYaw_pitch[:, :, 0] = np.mod(dYaw_pitch[:, :, 0] + 180.0, 360.0) - 180.0
    dYaw_pitch[:, :, 1] = np.mod(dYaw_pitch[:, :, 1] + 180.0, 360.0) - 180.0
    dYaw_pitch[:, :, 2] = np.mod(dYaw_pitch[:, :, 2] + 180.0, 360.0) - 180.0
    # import pdb;pdb.set_trace()
    abs_dYaw_pitch = np.abs(dYaw_pitch)
    invalid_axes = np.any(abs_dYaw_pitch >= rot_thres, axis=-1)



    if with_gps:
        dist = scipy.spatial.distance.cdist(query_ts_2d, db_ts_2d)
        invalid_dist = dist >= dist_thres  # !
        invalid = np.logical_or(invalid_dist, invalid_axes)
    else:
        invalid = invalid_axes

    sim = torch.einsum('id,jd->ij', query_desc.to(device),
                       db_desc.to(device))  
    
    # Sensor-guided matching with spatial invalid
    # import pdb; pdb.set_trace()
    pairs = pairs_from_score_matrix(sim, invalid, num_matched, min_score=0)
   
    pairs = [(query_names[i], db_names[j]) for i, j in pairs]

    t_end = time.time()
    # print("retrieval:", t_end - time1)
    logger.info(f'Get pairs uses {t_end-t_start} seconds.')
    logger.info(f'Found {len(pairs)} pairs.')
    # import pdb;pdb.set_trace()
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--query_list', type=Path)
    parser.add_argument('--sensor_path', type=Path)
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--db_model', type=Path)
    parser.add_argument('--db_descriptors', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)

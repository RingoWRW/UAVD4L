import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
import scipy.spatial
from scipy.spatial.transform import Rotation

from . import logger
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary, qvec2rotmat
from .utils.io import list_h5_names

DEFAULT_DIST_THRESH = 20  # in meters
DEFAULT_ROT_THRESH = 60  # in degrees (option: 90 or 60)

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
        with h5py.File(str(path), 'r') as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), 'r') as fd:
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


def get_query_retrieval_pose(names, path):
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            t, q = np.split(np.array(data[1:], float), [3])

            # rotation
            R = qvec2rotmat(q)

            poses[name] = t, R

    Rs = []
    ts = []
    for name in names:
        t, R = poses[name]
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
    

def main(descriptors, output, num_matched,
         query_prefix=None, query_list=None, 
         sensor_path=None, db_prefix=None, 
         db_list=None, db_model=None, db_descriptors=None):
    logger.info('Extracting image pairs from a retrieval database with SENSOR guide.')

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
        db_ts_2d, db_Rs = get_db_retrieval_pose(images)
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')
    query_names = parse_names(query_prefix, query_list, query_names_h5)

    # gps-priored retrieval
    query_ts_2d, query_Rs = get_query_retrieval_pose(query_names, sensor_path)
    dist = scipy.spatial.distance.cdist(query_ts_2d, db_ts_2d)

    # Instead of computing the angle between two camera orientations,
    # we compute the angle between the principal axes, as two images rotated
    # around their principal axis still observe the same scene.
    db_axes = db_Rs[:, :, -1]
    query_axes = query_Rs[:, :, -1]
    dots = np.einsum('mi,ni->mn', query_axes, db_axes, optimize=True)
    dR = np.rad2deg(np.arccos(np.clip(dots, -1., 1.)))

    invalid_dist = (dist >= DEFAULT_DIST_THRESH)
    invalid_axes = (dR >= DEFAULT_ROT_THRESH)
    invalid = np.logical_or(invalid_dist, invalid_axes)

    # feature-based retrieval
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    query_desc = get_descriptors(query_names, descriptors)
    sim = torch.einsum('id,jd->ij', query_desc.to(device), db_desc.to(device))

    # Sensor-guided matching with spatial invalid
    pairs = pairs_from_score_matrix(sim, invalid, num_matched, min_score=0)
    pairs = [(query_names[i], db_names[j]) for i, j in pairs]

    logger.info(f'Found {len(pairs)} pairs.')
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
